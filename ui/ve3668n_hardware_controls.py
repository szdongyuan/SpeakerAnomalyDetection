"""Small VE hardware panel and an instance-owned, queued discovery boundary.

The panel stages rates; only the hardware dialog's explicit OK persists them.
Factories accept on_result=callback, like DiscoveryService. No SDK calls or
wait_idle/wait_closed barriers run here, including during close/cancellation.
"""
from copy import deepcopy

from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QGridLayout

from base.hardware_selection import is_ve_input, resolve_ve_input
from base.ve3668n_discovery import DiscoveryService
from base.ve3668n_input import validate_sample_rate
from consts.ve3668n_consts import VE_SAMPLE_RATES


class VEDiscoveryBridge(QObject):
    result_ready = pyqtSignal(object)
    _event = pyqtSignal(object)

    def __init__(self, parent=None, *, discovery_factory=None):
        super().__init__(parent)
        self.service = None
        self._factory = discovery_factory or DiscoveryService
        self._generation = None
        self._closed = False
        self._event.connect(self._deliver, Qt.QueuedConnection)

    def start(self):
        if self._closed:
            return
        if self.service is None:
            self.service = self._factory(on_result=self._event.emit)
            self._generation = self.service.start()
        else:
            self._generation = self.service.refresh()
        return self._generation

    def refresh(self):
        return self.start()

    def cancel(self):
        self._generation = None
        if self.service is not None:
            self.service.cancel()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._generation = None
        if self.service is not None:
            self.service.close()

    @pyqtSlot(object)
    def _deliver(self, event):
        if not self._closed and event.generation == self._generation:
            self.result_ready.emit(event)


class VE3668NHardwareControls(QWidget):
    backend_changed = pyqtSignal(str)
    input_changed = pyqtSignal(object)
    busy_edit_rejected = pyqtSignal()

    def __init__(self, parent=None, *, profile_store, calibration_store,
                 initial_device=None, initial_channels=(), discovery_factory=None, busy_check=None):
        super().__init__(parent)
        self.profile_store = profile_store
        self.calibration_store = calibration_store
        self.discovery = VEDiscoveryBridge(self, discovery_factory=discovery_factory)
        self.discovery.result_ready.connect(self._on_discovered)
        self._current = deepcopy(initial_device) if is_ve_input(initial_device) else None
        self._devices = {}
        self.selected_channels = list(initial_channels)
        self._draft_rates = {}
        self._busy = False
        self._busy_check = busy_check or (lambda: self._busy)
        self._closed = False
        self._pending_event = None
        self._rendering = False
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("普通声卡", "sounddevice")
        self.backend_combo.addItem("VE3668N", "vkinging")
        self.backend_combo.setCurrentIndex(1 if self._current else 0)
        self._backend = self.backend_combo.currentData()
        self.device_combo = QComboBox()
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setEditable(False)
        for rate in VE_SAMPLE_RATES:
            self.sample_rate_combo.addItem(f"{rate} Hz", rate)
        self.sample_rate_combo.setCurrentIndex(-1)
        self.fixed_label = QLabel("固定 IEPE / V / ±10 V；按信号频带选择采样率")
        self.diagnostic_label = QLabel()
        self.diagnostic_label.setWordWrap(True)
        layout = QGridLayout(self)
        layout.addWidget(QLabel("输入后端"), 0, 0)
        layout.addWidget(self.backend_combo, 0, 1)
        layout.addWidget(self.device_combo, 0, 2)
        layout.addWidget(self.sample_rate_combo, 0, 3)
        layout.addWidget(self.fixed_label, 1, 1, 1, 3)
        layout.addWidget(self.diagnostic_label, 2, 1, 1, 3)
        self.backend_combo.currentIndexChanged.connect(self._on_backend)
        self.device_combo.currentIndexChanged.connect(self._on_device)
        self.device_combo.activated[int].connect(self._on_device)
        self.sample_rate_combo.currentIndexChanged.connect(self._on_rate)
        self._show_backend()

    @property
    def selected_device(self):
        return deepcopy(self._current)

    @property
    def inventory(self):
        return list(self._devices.get(self._machine_key(), {}).get("physical_channels", []))

    def _machine_key(self):
        value = (self._current or {}).get("machine_id")
        return value if isinstance(value, str) else None

    def set_busy(self, busy):
        self._busy = busy
        for widget in (self.backend_combo, self.device_combo, self.sample_rate_combo):
            widget.setEnabled(not busy)
        if not busy and self._pending_event is not None:
            event, self._pending_event = self._pending_event, None
            self._on_discovered(event)

    def cancel_discovery(self):
        self._pending_event = None
        self.discovery.cancel()

    def close_discovery(self):
        self._closed = True
        self._pending_event = None
        self.discovery.close()

    def _show_backend(self):
        ve = self.backend_combo.currentData() == "vkinging"
        for widget in (self.device_combo, self.sample_rate_combo, self.fixed_label, self.diagnostic_label):
            widget.setVisible(ve)

    def _on_backend(self):
        if self._rendering or self._reject_busy_edit():
            return
        self._backend = self.backend_combo.currentData()
        self._show_backend()
        self.backend_changed.emit(self._backend)

    def _reject_busy_edit(self):
        if not self._busy_check():
            return False
        # Qt has already changed the widget before currentIndexChanged fires.
        # Restore the staged values, not the original on-disk selection.
        self._render_selection()
        self.set_busy(True)
        self.busy_edit_rejected.emit()
        return True

    def _render_selection(self):
        config = (self._current or {}).get("input_config")
        self._rendering = True
        try:
            self.backend_combo.setCurrentIndex(self.backend_combo.findData(self._backend))
            self.device_combo.setCurrentIndex(
                self.device_combo.findData(self._machine_key()) if self._current else -1)
            self.sample_rate_combo.setCurrentIndex(
                self.sample_rate_combo.findData(config["sample_rate"]) if config else -1)
        finally:
            self._rendering = False
        self._show_backend()

    def refresh(self):
        if self._closed or self._busy_check() or self.backend_combo.currentData() != "vkinging":
            return
        self._pending_event = None
        self._devices = {}
        if self._current:
            self._current = {**self._current, "available": False,
                             "diagnostic": "正在检查 VE3668N；确认前不可用"}
            self.input_changed.emit(self.selected_device)
        self.diagnostic_label.setText("正在检查 VE3668N…")
        self.discovery.refresh()

    def _on_discovered(self, event):
        if self._closed:
            return
        if self._busy_check():
            self._pending_event = event
            return
        self._devices = {item["machine_id"]: deepcopy(item) for item in event.result.devices} if event.handles_released else {}
        machine_id = self._machine_key()
        self._rendering = True
        self.device_combo.clear()
        for key, device in self._devices.items():
            self.device_combo.addItem(f"VE3668N · {key} · {device['name']}", key)
        if self._current and machine_id not in self._devices:
            self.device_combo.addItem(f"VE3668N · {machine_id} · 不可用", machine_id)
        self.device_combo.setCurrentIndex(self.device_combo.findData(machine_id) if self._current else -1)
        self._rendering = False
        if self._current:
            self._current = resolve_ve_input(self._current,
                self.selected_channels or self._current.get("physical_channels"),
                tuple(self._devices.values()), profile_store=self.profile_store,
                calibration_store=self.calibration_store, diagnostic="; ".join(event.result.diagnostics))
            self._apply_draft()
            self._render_current()
        else:
            self.diagnostic_label.setText("; ".join(event.result.diagnostics) or "请选择 VE3668N 设备")

    def _on_device(self):
        if self._rendering or self._reject_busy_edit():
            return
        device = self._devices.get(self.device_combo.currentData())
        if device is None:
            return
        if device["machine_id"] != self._machine_key() or (self._current or {}).get("selection_error"):
            self.selected_channels = []
        else:
            # Only explicit device selection may discard missing routes. Keep
            # the relative order of surviving choices, including same-ID clicks.
            self.selected_channels = [channel for channel in self.selected_channels
                                      if channel in device["physical_channels"]]
        self._current = resolve_ve_input(device, device["physical_channels"], [device],
            profile_store=self.profile_store, calibration_store=self.calibration_store)
        self._apply_draft()
        self._render_current()

    def _apply_draft(self):
        machine_id = self._machine_key()
        if (machine_id in self._draft_rates and machine_id in self._devices
                and not self._current.get("selection_error")
                and all(channel in self._devices[machine_id]["physical_channels"]
                        for channel in self.selected_channels)):
            verified = deepcopy(self._devices[machine_id])
            verified["input_config"]["sample_rate"] = self._draft_rates[machine_id]
            self._current = verified

    def _on_rate(self):
        if self._rendering or self._reject_busy_edit() or self._current is None:
            return
        rate = self.sample_rate_combo.currentData()
        if rate is None:
            return
        if self._machine_key() is None:
            return
        self._draft_rates[self._machine_key()] = validate_sample_rate(rate)
        self._apply_draft()
        self.diagnostic_label.setText("采样率已修改，尚未保存；确定后应用")
        self.input_changed.emit(self.selected_device)

    def _render_current(self):
        self._render_selection()
        self.diagnostic_label.setText(self._current.get("diagnostic", "可用；采样率在确定后保存"))
        self.input_changed.emit(self.selected_device)
