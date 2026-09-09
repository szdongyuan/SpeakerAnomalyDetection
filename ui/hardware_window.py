from cgi import print_arguments
import sys
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHeaderView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
)

from base.sound_device_manager import SoundDeviceManager
from base.hardware_selection import is_ve_input, save_ve_selection, restore_or_default, save_if_changed
from base.ve3668n_stores import VEInputProfileStore, VECalibrationStore
from ui.ve3668n_hardware_controls import VE3668NHardwareControls
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.config_dialog_base import ConfigDialogBase


class SingleCheckTableView(QTableView):
    """
    单列勾选表格：每个表格同一时间最多只能勾选一项（允许全不勾选）。
    """

    checked_payload_changed = pyqtSignal(object)  # payload 或 None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = QStandardItemModel(0, 1, self)
        self.setModel(self._model)

        self._guard = False
        self._model.itemChanged.connect(self._on_item_changed)

        self._init_view_style()

    def _init_view_style(self) -> None:
        # 取消选择高亮：禁用行选择，仅保留勾选交互
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.NoSelection)
        self.setEditTriggers(QTableView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setTextElideMode(Qt.ElideNone)

        # 单列：让列宽自动填满可用空间，避免触发文本省略
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # 覆盖项目全局 QTableView::item 的 border-top（横线）
        # 并将 selected 背景置为透明，避免出现点击高亮
        # 仅对本控件生效，避免影响其它页面表格样式
        self.setStyleSheet(
            """
            QTableView::item { border: none; }
            QTableView::item:selected { background: transparent; color: inherit; }
            QTableView::item:focus { outline: none; }
            """
        )

    def clear_options(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    def set_options(self, options: Iterable[Tuple[str, Any]]) -> None:
        self.clear_options()
        for text, payload in options:
            item = QStandardItem(str(text))
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            item.setEditable(False)
            item.setData(payload, Qt.UserRole)
            self._model.appendRow(item)

    def checked_payload(self) -> Optional[Any]:
        for row in range(self._model.rowCount()):
            item = self._model.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                return item.data(Qt.UserRole)
        return None

    def set_checked_by_predicate(self, predicate) -> None:
        """
        按条件勾选第一条匹配项（未匹配则全部取消勾选）。
        predicate: (payload) -> bool
        """
        self._guard = True
        try:
            hit_row = None
            for row in range(self._model.rowCount()):
                payload = self._model.item(row, 0).data(Qt.UserRole)
                if predicate(payload):
                    hit_row = row
                    break
            for row in range(self._model.rowCount()):
                item = self._model.item(row, 0)
                if item is None or not item.isCheckable():
                    continue
                item.setCheckState(Qt.Checked if row == hit_row else Qt.Unchecked)
        finally:
            self._guard = False
        self.checked_payload_changed.emit(self.checked_payload())

    def _on_item_changed(self, item: QStandardItem) -> None:
        if self._guard:
            return
        if item.column() != 0 or not item.isCheckable():
            return

        if item.checkState() == Qt.Checked:
            # 保证单选：勾选当前行时，取消其它行勾选
            self._guard = True
            try:
                for row in range(self._model.rowCount()):
                    if row == item.row():
                        continue
                    other = self._model.item(row, 0)
                    if other is not None and other.isCheckable() and other.checkState() == Qt.Checked:
                        other.setCheckState(Qt.Unchecked)
            finally:
                self._guard = False

        # 勾选/取消都发信号（取消时 payload 可能为 None）
        self.checked_payload_changed.emit(self.checked_payload())


class MultiCheckTableView(QTableView):
    """
    单列勾选表格：允许多选（可同时勾选多项）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = QStandardItemModel(0, 1, self)
        self.setModel(self._model)
        self._init_view_style()

    def _init_view_style(self) -> None:
        # 取消选择高亮：禁用行选择，仅保留勾选交互
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.NoSelection)
        self.setEditTriggers(QTableView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setTextElideMode(Qt.ElideNone)

        # 单列：让列宽自动填满可用空间，避免触发文本省略
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # 覆盖项目全局 QTableView::item 的 border-top（横线）
        # 并将 selected 背景置为透明，避免出现点击高亮
        self.setStyleSheet(
            """
            QTableView::item { border: none; }
            QTableView::item:selected { background: transparent; color: inherit; }
            QTableView::item:focus { outline: none; }
            """
        )

    def clear_options(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    def set_options(self, options: Iterable[Tuple[str, Any]]) -> None:
        self.clear_options()
        for text, payload in options:
            item = QStandardItem(str(text))
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            item.setEditable(False)
            item.setData(payload, Qt.UserRole)
            self._model.appendRow(item)

    def checked_payloads(self) -> List[Any]:
        checked: List[Any] = []
        for row in range(self._model.rowCount()):
            item = self._model.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                checked.append(item.data(Qt.UserRole))
        return checked


@dataclass
class HardwareSelectionState:
    speaker_device: Optional[Dict[str, Any]] = None
    speaker_channels: List[int] = None
    mic_device: Optional[Dict[str, Any]] = None
    mic_channels: List[int] = None
    api_name: Optional[str] = None

    def __post_init__(self):
        if self.speaker_channels is None:
            self.speaker_channels = []
        if self.mic_channels is None:
            self.mic_channels = []


class HardwareSelectionModel:
    """
    Model：负责设备枚举、驱动过滤、通道生成、选择状态保存。
    """

    def __init__(self, initial_state: Optional[HardwareSelectionState] = None):
        self.sdm = SoundDeviceManager()
        self.devices_by_api: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.state = deepcopy(initial_state) if initial_state else HardwareSelectionState()

    def refresh(self) -> None:
        self.sdm.refresh_available_device()
        self.devices_by_api = self.sdm.get_device_info() or {}
        if self.state.api_name not in self.devices_by_api:
            self.state.api_name = next(iter(self.devices_by_api.keys()), None)

    def api_names(self) -> List[str]:
        return list(self.devices_by_api.keys())

    def set_api(self, api_name: str) -> None:
        self.state.api_name = api_name
        # 切换驱动后，清空选择（由 Controller 负责尝试恢复）
        self.state.speaker_device = None
        self.state.speaker_channels = []
        if not (self.state.mic_device or {}).get("backend") == "vkinging":
            self.state.mic_device = None
            self.state.mic_channels = []

    def speaker_devices(self) -> List[Dict[str, Any]]:
        if not self.state.api_name or self.state.api_name not in self.devices_by_api:
            return []
        return self.devices_by_api[self.state.api_name].get("output", []) or []

    def mic_devices(self) -> List[Dict[str, Any]]:
        if not self.state.api_name or self.state.api_name not in self.devices_by_api:
            return []
        return self.devices_by_api[self.state.api_name].get("input", []) or []

    @staticmethod
    def channels_for_device(device: Optional[Dict[str, Any]], kind: str) -> List[int]:
        if not device:
            return []
        if kind == "mic" and device.get("backend") == "vkinging":
            return list(device.get("physical_channels") or [])
        if kind == "speaker":
            n = int(device.get("max_output_channels") or 0)
        else:
            n = int(device.get("max_input_channels") or 0)
        # 通道索引采用 0-based（返回值从 0 开始）
        return list(range(0, n))


class HardwareSelectionView(ConfigDialogBase):
    """
    View：只负责界面与基本控件，不负责设备枚举/选择逻辑。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("硬件选择")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # 顶部：选择驱动
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(QLabel("选择驱动"))
        self.driver_combo = QComboBox()
        self.driver_combo.setMinimumWidth(220)
        top_layout.addWidget(self.driver_combo)

        # 3 张表：扬声器设备（单选）、麦克风设备（单选）、麦克风通道（多选）
        self.speaker_device_table = SingleCheckTableView()
        self.mic_device_table = SingleCheckTableView()
        self.mic_channel_table = MultiCheckTableView()

        speaker_device_box = self._wrap_table_group("选择设备", self.speaker_device_table)
        mic_device_box = self._wrap_table_group("选择设备", self.mic_device_table)
        mic_channel_box = self._wrap_table_group("选择通道", self.mic_channel_table)

        # 左右两个大框
        left_big = QGroupBox("选择扬声器")
        left_layout = QVBoxLayout()
        left_layout.addWidget(speaker_device_box)
        left_big.setLayout(left_layout)

        right_big = QGroupBox("选择麦克风")
        right_layout = QVBoxLayout()
        right_layout.addWidget(mic_device_box)
        right_layout.addWidget(mic_channel_box)
        right_big.setLayout(right_layout)

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(left_big)
        middle_layout.addWidget(right_big)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        self.refresh_btn = QPushButton(" 刷  新 ")
        self.ok_btn = QPushButton(" 确  定 ")
        bottom_layout.addWidget(self.refresh_btn)
        bottom_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        bottom_layout.addWidget(self.ok_btn)
        bottom_layout.setContentsMargins(0, 0, 11, 0)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addSpacing(6)
        layout.addLayout(middle_layout)
        layout.addSpacing(6)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        self.resize(980, 560)
        self.apply_config_dialog_theme()

    @staticmethod
    def _wrap_table_group(title: str, table: QTableView) -> QGroupBox:
        box = QGroupBox(title)
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 16, 8, 8)
        lay.addWidget(table)
        box.setLayout(lay)
        return box

    def on_exec(self) -> int:
        return self.exec()


class HardwareSelectionController:
    """
    Controller：绑定信号，驱动 Model 与 View 的同步。
    """

    def __init__(self, model: HardwareSelectionModel, view: HardwareSelectionView, *,
                 profile_store=None, calibration_store=None, discovery_factory=None,
                 busy_check=None, selection_path=None):
        self.model = model
        self.view = view
        self.selection_path = selection_path
        self.busy_check = busy_check or (lambda: False)
        self._rendering_channels = False
        self.profile_store = profile_store if profile_store is not None else VEInputProfileStore()
        self.calibration_store = calibration_store if calibration_store is not None else VECalibrationStore()
        self.view.ve_controls = VE3668NHardwareControls(view,
            profile_store=self.profile_store, calibration_store=self.calibration_store,
            initial_device=model.state.mic_device, initial_channels=model.state.mic_channels,
            discovery_factory=discovery_factory, busy_check=self.busy_check)
        view.layout().insertWidget(1, self.view.ve_controls)
        view.finished.connect(self.view.ve_controls.close_discovery)
        view.destroyed.connect(self.view.ve_controls.discovery.close)
        self._soundcard_state = deepcopy(model.state) if not is_ve_input(model.state.mic_device) else None
        self._initial_state = HardwareSelectionState(
            speaker_device=model.state.speaker_device,
            speaker_channels=list(model.state.speaker_channels),
            mic_device=model.state.mic_device,
            mic_channels=list(model.state.mic_channels),
            api_name=model.state.api_name,
        )

        self._bind_signals()
        self.refresh_and_render(try_restore=True)
        self._busy_timer = QTimer(view)
        self._busy_timer.setInterval(100)
        self._busy_timer.timeout.connect(self._update_busy_state)
        view.finished.connect(self._busy_timer.stop)
        self._busy_timer.start()
        self._update_busy_state()

    @staticmethod
    def _device_key(device: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Any]]:
        if not device:
            return None
        if is_ve_input(device):
            return "vkinging", device.get("machine_id")
        return device.get("name"), int(device.get("hostapi") or -1)

    def _bind_signals(self) -> None:
        self.view.refresh_btn.clicked.connect(lambda: self.refresh_and_render(try_restore=True))
        self.view.driver_combo.currentTextChanged.connect(self._on_api_changed)

        self.view.speaker_device_table.checked_payload_changed.connect(self._on_speaker_device_checked)
        self.view.mic_device_table.checked_payload_changed.connect(self._on_mic_device_checked)

        self.view.ok_btn.clicked.connect(self._on_ok_clicked)
        self.view.ve_controls.backend_changed.connect(self._on_backend_changed)
        self.view.ve_controls.input_changed.connect(self._on_ve_input_changed)
        self.view.ve_controls.busy_edit_rejected.connect(self._update_busy_state)
        self.view.mic_channel_table.model().itemChanged.connect(self._on_channel_changed)

    def refresh_and_render(self, try_restore: bool) -> None:
        if self.busy_check():
            return
        last_state = HardwareSelectionState(
            speaker_device=self.model.state.speaker_device,
            speaker_channels=list(self.model.state.speaker_channels),
            mic_device=self.model.state.mic_device,
            mic_channels=list(self.model.state.mic_channels),
            api_name=self.model.state.api_name,
        )

        self.model.refresh()

        # 1) render combo
        api_names = self.model.api_names()
        self.view.driver_combo.blockSignals(True)
        try:
            self.view.driver_combo.clear()
            self.view.driver_combo.addItems(api_names)
            if self.model.state.api_name:
                self.view.driver_combo.setCurrentText(self.model.state.api_name)
        finally:
            self.view.driver_combo.blockSignals(False)

        # 2) render device tables for current api
        self._render_devices()

        if try_restore:
            self._try_restore_selection(last_state)
        self.view.ve_controls.refresh()

    def _render_devices(self) -> None:
        speaker_opts = [(d.get("name", ""), d) for d in self.model.speaker_devices()]
        mic_opts = [(d.get("name", ""), d) for d in self.model.mic_devices()]
        self.view.speaker_device_table.set_options(speaker_opts)
        self.view.mic_device_table.set_options(mic_opts)

        # 清空麦克风通道（等设备勾选后再填充）
        if is_ve_input(self.model.state.mic_device):
            self._render_ve_channels()
        else:
            self.view.mic_channel_table.set_options([])

    def _try_restore_selection(self, last_state: HardwareSelectionState) -> None:
        # 恢复驱动
        if last_state.api_name and last_state.api_name in self.model.devices_by_api:
            self.model.state.api_name = last_state.api_name
            self.view.driver_combo.setCurrentText(last_state.api_name)

        # 恢复设备（用 name+hostapi 匹配）
        last_s_key = self._device_key(last_state.speaker_device)
        if last_s_key:
            self.view.speaker_device_table.set_checked_by_predicate(
                lambda p: self._device_key(p) == last_s_key
            )

        last_m_key = self._device_key(last_state.mic_device)
        if last_m_key and not is_ve_input(last_state.mic_device):
            self.view.mic_device_table.set_checked_by_predicate(lambda p: self._device_key(p) == last_m_key)

        # 恢复麦克风通道（在设备恢复后，通道表已刷新，这里再勾选）
        if last_state.mic_channels and not is_ve_input(last_state.mic_device):
            self._restore_channels(self.view.mic_channel_table, last_state.mic_channels)

    @staticmethod
    def _restore_channels(table: MultiCheckTableView, channels: List[int]) -> None:
        want = set(int(x) for x in channels)
        # 直接操作 model item 勾选
        model = table.model()
        for row in range(model.rowCount()):
            item = model.item(row, 0)
            if item is None:
                continue
            payload = item.data(Qt.UserRole)
            if int(payload) in want:
                item.setCheckState(Qt.Checked)

    def _on_api_changed(self, api_name: str) -> None:
        if self._reject_busy_edit():
            return
        self.model.set_api(api_name)
        self._render_devices()

    def _on_speaker_device_checked(self, payload: object) -> None:
        if self._reject_busy_edit():
            return
        self.model.state.speaker_device = payload if isinstance(payload, dict) else None
        # 不再提供扬声器通道选择，保持为空兼容返回结构
        self.model.state.speaker_channels = []

    def _on_mic_device_checked(self, payload: object) -> None:
        if self._reject_busy_edit():
            return
        if self.view.ve_controls.backend_combo.currentData() == "vkinging":
            return
        self.model.state.mic_device = payload if isinstance(payload, dict) else None
        channels = self.model.channels_for_device(self.model.state.mic_device, "mic")
        self.model.state.mic_channels = []
        # 显示仍用 In1..InN，但 payload/返回值为 0..N-1
        self.view.mic_channel_table.set_options([(f"In{i + 1}", i) for i in channels])

    def _on_ok_clicked(self) -> None:
        if self.busy_check():
            self._update_busy_state()
            QMessageBox.warning(self.view, "提示", "录音或校准进行中，请等待完成后再修改硬件设置")
            return
        speaker = self.model.state.speaker_device
        mic = self.model.state.mic_device

        if self.view.ve_controls.backend_combo.currentData() == "vkinging":
            channels = list(self.model.state.mic_channels)
            legacy = self._soundcard_state
            legacy_selection = None if legacy is None else {
                "api_name": legacy.api_name,
                "mic_name": (legacy.mic_device or {}).get("name"),
                "mic_channels": list(legacy.mic_channels),
                "speaker_name": (legacy.speaker_device or {}).get("name"),
                "speaker_channels": list(legacy.speaker_channels),
            }
            try:
                committed = save_ve_selection(mic, speaker, channels, [],
                    profile_store=self.profile_store, calibration_store=self.calibration_store,
                    path=self.selection_path, api_name=self.model.state.api_name,
                    legacy_soundcard_selection=legacy_selection)
            except (ValueError, OSError) as exc:
                self.view.ve_controls.diagnostic_label.setText(f"未保存：{exc}")
                QMessageBox.warning(self.view, "VE 硬件未保存", str(exc))
                return
            if speaker is not None:
                SoundDeviceManager.change_default_output_device(int(speaker["index"]))
            self.model.state.mic_device = committed
            self.model.state.mic_channels = list(channels)
            self.view.accept()
            return

        mic_channels = sorted(int(x) for x in self.view.mic_channel_table.checked_payloads())

        if speaker and mic and mic_channels:
            self.model.state.speaker_channels = []
            self.model.state.mic_channels = mic_channels
        else:
            QMessageBox.warning(self.view, "提示", "请选择扬声器和麦克风设备，以及麦克风通道！")
            return None

        mic_idx = int(mic.get("index")) if mic else -1
        speaker_idx = int(speaker.get("index")) if speaker else -1
        if is_ve_input(self._initial_state.mic_device):
            try:
                save_if_changed(mic, speaker, mic_channels, [], path=self.selection_path, strict=True)
            except OSError as exc:
                QMessageBox.warning(self.view, "硬件未保存", str(exc))
                return
        SoundDeviceManager.change_default_device(mic_idx, speaker_idx)

        self.view.accept()

    def _render_ve_channels(self):
        channels = self.view.ve_controls.inventory
        self._rendering_channels = True
        try:
            self.view.mic_channel_table.set_options([(f"AIN{i + 1}", i) for i in channels])
            selected = {value for value in self.model.state.mic_channels if type(value) is int}
            for row in range(self.view.mic_channel_table.model().rowCount()):
                item = self.view.mic_channel_table.model().item(row)
                if item.data(Qt.UserRole) in selected:
                    item.setCheckState(Qt.Checked)
        finally:
            self._rendering_channels = False
        self.view.mic_device_table.setEnabled(False)

    def _on_channel_changed(self, item):
        if self._rendering_channels or self._reject_busy_edit():
            return
        if not is_ve_input(self.model.state.mic_device):
            self.model.state.mic_channels = sorted(self.view.mic_channel_table.checked_payloads())
            return
        channel = item.data(Qt.UserRole)
        selected = self.model.state.mic_channels
        if item.checkState() == Qt.Checked and channel not in selected:
            selected.append(channel)
        elif item.checkState() != Qt.Checked and channel in selected:
            selected.remove(channel)
        self.view.ve_controls.selected_channels = list(selected)

    def _reject_busy_edit(self):
        if not self.busy_check():
            return False
        state = self.model.state
        combo = self.view.driver_combo
        previous = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(combo.findText(state.api_name or ""))
        finally:
            combo.blockSignals(previous)
        for table, device in ((self.view.speaker_device_table, state.speaker_device),
                              (self.view.mic_device_table, state.mic_device)):
            previous = table.blockSignals(True)
            try:
                table.set_checked_by_predicate(lambda value: self._device_key(value) == self._device_key(device))
            finally:
                table.blockSignals(previous)
        self._rendering_channels = True
        try:
            model = self.view.mic_channel_table.model()
            for row in range(model.rowCount()):
                item = model.item(row)
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in state.mic_channels else Qt.Unchecked)
        finally:
            self._rendering_channels = False
        self._update_busy_state()
        return True

    def _update_busy_state(self):
        busy = bool(self.busy_check())
        self.view.ve_controls.set_busy(busy)
        for widget in (self.view.driver_combo, self.view.speaker_device_table,
                       self.view.mic_channel_table, self.view.refresh_btn, self.view.ok_btn):
            widget.setEnabled(not busy)
        self.view.mic_device_table.setEnabled(
            not busy and self.view.ve_controls.backend_combo.currentData() != "vkinging")

    def _on_ve_input_changed(self, device):
        if self.view.ve_controls.backend_combo.currentData() != "vkinging":
            return
        self.model.state.mic_channels = list(self.view.ve_controls.selected_channels)
        self.model.state.mic_device = device
        self._render_ve_channels()

    def _on_backend_changed(self, backend):
        if self.busy_check():
            return
        if backend == "vkinging":
            self._soundcard_state = deepcopy(self.model.state)
            self._soundcard_state.mic_channels = sorted(self.view.mic_channel_table.checked_payloads())
            self._on_ve_input_changed(self.view.ve_controls.selected_device)
            self.view.ve_controls.refresh()
        else:
            self.view.ve_controls.cancel_discovery()
            if self._soundcard_state is None:
                mic, speaker, channels, speaker_channels = restore_or_default(
                    path=self.selection_path, soundcard_only=True, apply_defaults=False)
                device, bucket = (mic, "input") if mic else (speaker, "output")
                api_name = next((name for name, devices in self.model.devices_by_api.items()
                    if any(self._device_key(item) == self._device_key(device)
                           for item in devices.get(bucket, []))), None)
                self.model.state = HardwareSelectionState(mic_device=mic, mic_channels=channels,
                    speaker_device=speaker, speaker_channels=speaker_channels, api_name=api_name)
            else:
                self.model.state = deepcopy(self._soundcard_state)
            self.view.mic_device_table.setEnabled(True)
            self.refresh_and_render(try_restore=True)

    def on_exec(self):
        result = self.view.on_exec()
        if result == QDialog.Accepted:
            return (
                True,
                self.model.state.speaker_device,
                list(self.model.state.speaker_channels),
                self.model.state.mic_device,
                list(self.model.state.mic_channels),
            )
        # 取消：返回初始状态（与 HardwareWindow 一致的“回退”语义）
        return (
            False,
            self._initial_state.speaker_device,
            list(self._initial_state.speaker_channels),
            self._initial_state.mic_device,
            list(self._initial_state.mic_channels),
        )


def _normalize_channel_indices(channels: Optional[Iterable[Any]]) -> List[int]:
    """
    将通道输入规范化为 0-based 的 int 列表。

    兼容：
    - [0, 1, 2]
    - ["Out1", "Out2"] / ["In1", "In2"]（按 1-based 解析后转 0-based）
    """
    if not channels:
        return []
    out: List[int] = []
    for x in channels:
        if x is None:
            continue
        if isinstance(x, str):
            s = x.strip()
            # 支持 Out1 / In1 / 1
            if s.lower().startswith(("out", "in")):
                s = s[3:] if s.lower().startswith("out") else s[2:]
            try:
                n = int(s)
                # 约定字符串通道为 1-based（Out1 表示索引 0）
                out.append(max(0, n - 1))
            except Exception:
                continue
        else:
            try:
                out.append(int(x))
            except Exception:
                continue
    # 去重 + 排序
    return sorted(set(out))


def open_hardware_selection_window(
    driver: Optional[str] = None,
    speaker_device: Optional[Dict[str, Any]] = None,
    speaker_channels: Optional[Iterable[Any]] = None,
    mic_device: Optional[Dict[str, Any]] = None,
    mic_channels: Optional[Iterable[Any]] = None,
    *, profile_store=None, calibration_store=None, discovery_factory=None,
    busy_check=None, selection_path=None,
):
    """
    打开硬件选择窗口，并支持根据传入参数进行初始化回填：
    - driver：Host API 名称（例如 WASAPI/ASIO/MME 等）
    - speaker_device / mic_device：sounddevice 设备 dict
    - speaker_channels / mic_channels：通道索引（0-based int）或显示字符串（Out1/In1）
    - 返回值：``(accepted, speaker_device, speaker_channels, mic_device, mic_channels)``
    """

    initial_state = HardwareSelectionState(
        speaker_device=speaker_device,
        speaker_channels=_normalize_channel_indices(speaker_channels),
        mic_device=mic_device,
        mic_channels=list(mic_channels or []) if is_ve_input(mic_device) else _normalize_channel_indices(mic_channels),
        api_name=driver,
    )

    model = HardwareSelectionModel(initial_state=initial_state)
    view = HardwareSelectionView()
    controller = HardwareSelectionController(model, view, profile_store=profile_store,
        calibration_store=calibration_store, discovery_factory=discovery_factory,
        busy_check=busy_check, selection_path=selection_path)
    try:
        return controller.on_exec()
    finally:
        view.ve_controls.close_discovery()
        controller._busy_timer.stop()
        view.deleteLater()
