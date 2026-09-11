"""Instance-owned asynchronous VK discovery and device selection."""
from copy import deepcopy

from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from base.hardware_selection import is_ve_input, resolve_ve_input
from base.ve3668n_discovery import DiscoveryService


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


class VE3668NHardwareControls(QObject):
    """Own discovery and the selected input; recording parameters are read-only."""

    inventory_changed = pyqtSignal()
    input_changed = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, profile_store, calibration_store,
                 initial_device=None, initial_channels=(), discovery_factory=None, busy_check=None):
        super().__init__(parent)
        self.profile_store = profile_store
        self.calibration_store = calibration_store
        self.discovery = VEDiscoveryBridge(self, discovery_factory=discovery_factory)
        self.discovery.result_ready.connect(self._on_discovered)
        self._saved_device = deepcopy(initial_device) if is_ve_input(initial_device) else None
        self._saved_channels = list(initial_channels) if self._saved_device else []
        self._current = None
        self._devices = {}
        self.selected_channels = []
        self.backend = "vkinging" if self._saved_device else "sounddevice"
        self._busy = False
        self._busy_check = busy_check or (lambda: self._busy)
        self._closed = False
        self._pending_event = None

    @property
    def selected_device(self):
        return deepcopy(self._current)

    @property
    def devices(self):
        devices = list(deepcopy(self._devices).values())
        machine_id = (self._current or {}).get("machine_id")
        if self._current and (not isinstance(machine_id, str) or machine_id not in self._devices):
            devices.append(deepcopy(self._current))
        return devices

    @property
    def inventory(self):
        if not self._current or not self._current.get("available"):
            return []
        return list(self._current["physical_channels"])

    def set_busy(self, busy):
        self._busy = busy
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

    def set_backend(self, backend):
        if self._closed or self._busy_check():
            return
        self.backend = backend
        self.cancel_discovery()
        self._devices = {}
        self._current = None
        self.selected_channels = []

    def refresh(self):
        if self._closed or self._busy_check() or self.backend != "vkinging":
            return
        self._pending_event = None
        self._devices = {}
        self.selected_channels = []
        self._current = deepcopy(self._saved_device)
        if self._current:
            self._current.update(available=False, input_config=None,
                                 diagnostic="正在检查 VK 设备")
        self.inventory_changed.emit()
        self.input_changed.emit(self.selected_device)
        self.status_changed.emit("正在检查 VK 设备…")
        self.discovery.refresh()

    def _on_discovered(self, event):
        if self._closed or self.backend != "vkinging":
            return
        if self._busy_check():
            self._pending_event = event
            return
        self._devices = {item["machine_id"]: deepcopy(item) for item in event.result.devices} if event.handles_released else {}
        diagnostic = "; ".join(event.result.diagnostics)
        self.selected_channels = []
        if self._saved_device:
            channels = self._saved_channels or self._saved_device.get("physical_channels")
            self._current = resolve_ve_input(self._saved_device, channels,
                tuple(self._devices.values()), profile_store=self.profile_store,
                calibration_store=self.calibration_store, diagnostic=diagnostic,
                load_profile=False)
            if self._current.get("available"):
                self.selected_channels = list(self._saved_channels)
            diagnostic = self._current.get("diagnostic", diagnostic)
        else:
            self._current = None
        self._publish(diagnostic or ("请选择 VK 设备" if self._devices else "未发现 VK 设备"))

    def select_device(self, machine_id):
        if self._closed or self.backend != "vkinging" or self._busy_check():
            return
        self.selected_channels = []
        device = self._devices.get(machine_id)
        self._current = None if device is None else resolve_ve_input(
            device, device["physical_channels"], [device],
            profile_store=self.profile_store, calibration_store=self.calibration_store,
            load_profile=False)
        self._publish((self._current or {}).get("diagnostic", ""))

    def _publish(self, diagnostic):
        self.inventory_changed.emit()
        self.input_changed.emit(self.selected_device)
        self.status_changed.emit(diagnostic)
