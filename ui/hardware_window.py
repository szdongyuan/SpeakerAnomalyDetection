import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QHeaderView, QHBoxLayout, QVBoxLayout

from base.hardware_management import (
    HardwareManagementError,
    HardwareManagementRepository,
    MissingHardwareTablesError,
)
from base import sound_device_manager as sound_device_manager_module
from base.sound_device_manager import SoundDeviceManager
from consts import model_consts
from consts.audio_consts import normalize_float_bit_depth
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import MessageBox, PushButton, Label, ComboBox, GroupBox, TableView


class SingleCheckTableView(TableView):
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
        self.setSelectionBehavior(TableView.SelectRows)
        self.setSelectionMode(TableView.NoSelection)
        self.setEditTriggers(TableView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setTextElideMode(Qt.ElideNone)

        # 单列：让列宽自动填满可用空间，避免触发文本省略
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # 覆盖项目全局 TableView::item 的 border-top（横线）
        # 并将 selected 背景置为透明，避免出现点击高亮
        # 仅对本控件生效，避免影响其它页面表格样式
        self.setStyleSheet(
            """
            TableView::item { border: none; }
            TableView::item:selected { background: transparent; color: inherit; }
            TableView::item:focus { outline: none; }
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


class MultiCheckTableView(TableView):
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
        self.setSelectionBehavior(TableView.SelectRows)
        self.setSelectionMode(TableView.NoSelection)
        self.setEditTriggers(TableView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setTextElideMode(Qt.ElideNone)

        # 单列：让列宽自动填满可用空间，避免触发文本省略
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # 覆盖项目全局 TableView::item 的 border-top（横线）
        # 并将 selected 背景置为透明，避免出现点击高亮
        self.setStyleSheet(
            """
            TableView::item { border: none; }
            TableView::item:selected { background: transparent; color: inherit; }
            TableView::item:focus { outline: none; }
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
    mic_device: Optional[Dict[str, Any]] = None
    mic_channels: List[int] = None
    api_name: Optional[str] = None

    def __post_init__(self):
        if self.mic_channels is None:
            self.mic_channels = []


class HardwareSelectionModel:
    """
    Model：负责注册硬件读取、驱动过滤、注册通道读取、选择状态保存。
    """

    def __init__(self, initial_state: Optional[HardwareSelectionState] = None, repository=None):
        self.repository = repository or HardwareManagementRepository()
        self.devices_by_api: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.state = initial_state or HardwareSelectionState()

    def refresh(self) -> None:
        if not self.repository.tables_exist():
            self.devices_by_api = {}
            self.state.api_name = None
            raise MissingHardwareTablesError("硬件管理表不存在，请使用最新版数据库")

        self.devices_by_api = self.repository.list_assets_for_selection() or {}
        if self.state.api_name not in self.devices_by_api:
            self.state.api_name = next(iter(self.devices_by_api.keys()), None)

    def api_names(self) -> List[str]:
        return list(self.devices_by_api.keys())

    def set_api(self, api_name: str) -> None:
        self.state.api_name = api_name
        # 切换驱动后，清空选择（由 Controller 负责尝试恢复）
        self.state.speaker_device = None
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

    def channels_for_device(self, device: Optional[Dict[str, Any]], kind: str) -> List[int]:
        return [int(channel.get("channel_index")) for channel in self.channel_rows_for_device(device, kind)]

    def channel_rows_for_device(self, device: Optional[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        if not device:
            return []
        direction = "output" if kind == "speaker" else "input"
        rows = self.repository.list_channels(device.get("hardware_id"), direction=direction)
        return sorted(rows, key=self._channel_sort_key)

    @staticmethod
    def _channel_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
        try:
            return (0, int(row.get("channel_index")), str(row.get("channel_label", "")))
        except (TypeError, ValueError):
            return (1, 0, str(row.get("channel_label", "")))


class HardwareSelectionView(QDialog):
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
        top_layout.addWidget(Label("选择驱动"))
        self.driver_combo = ComboBox()
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
        left_big = GroupBox("选择扬声器")
        left_layout = QVBoxLayout()
        left_layout.addWidget(speaker_device_box)
        left_big.setLayout(left_layout)

        right_big = GroupBox("选择麦克风")
        right_layout = QVBoxLayout()
        right_layout.addWidget(mic_device_box)
        right_layout.addWidget(mic_channel_box)
        right_big.setLayout(right_layout)

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(left_big)
        middle_layout.addWidget(right_big)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        self.ok_btn = PushButton(" 确  定 ")
        bottom_layout.addStretch()
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

    @staticmethod
    def _wrap_table_group(title: str, table: TableView) -> GroupBox:
        box = GroupBox(title)
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 16, 8, 8)
        lay.addWidget(table)
        box.setLayout(lay)
        return box


class HardwareSelectionController:
    """
    Controller：绑定信号，驱动 Model 与 View 的同步。
    """

    def __init__(self, model: HardwareSelectionModel, view: HardwareSelectionView):
        self.model = model
        self.view = view
        self._initial_state = HardwareSelectionState(
            speaker_device=model.state.speaker_device,
            mic_device=model.state.mic_device,
            mic_channels=list(model.state.mic_channels),
            api_name=model.state.api_name,
        )

        self._bind_signals()
        self.refresh_and_render(try_restore=True)

    @staticmethod
    def _matches_restore_device(candidate: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> bool:
        if not candidate or not previous:
            return False
        if previous.get("hardware_id"):
            return candidate.get("hardware_id") == previous.get("hardware_id")
        previous_name = previous.get("device_name") or previous.get("name")
        previous_api = previous.get("hostapi_name")
        return (
            bool(previous_name)
            and bool(previous_api)
            and candidate.get("device_name") == previous_name
            and candidate.get("hostapi_name") == previous_api
        )

    def _bind_signals(self) -> None:
        self.view.driver_combo.currentTextChanged.connect(self._on_api_changed)

        self.view.speaker_device_table.checked_payload_changed.connect(self._on_speaker_device_checked)
        self.view.mic_device_table.checked_payload_changed.connect(self._on_mic_device_checked)

        self.view.ok_btn.clicked.connect(self._on_ok_clicked)

    def refresh_and_render(self, try_restore: bool) -> None:
        last_state = HardwareSelectionState(
            speaker_device=self.model.state.speaker_device,
            mic_device=self.model.state.mic_device,
            mic_channels=list(self.model.state.mic_channels),
            api_name=self.model.state.api_name,
        )

        try:
            self.model.refresh()
        except MissingHardwareTablesError:
            MessageBox.warning(self.view, "提示", "硬件管理表不存在，请使用最新版数据库")
            self._render_empty()
            return None
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as e:
            MessageBox.warning(self.view, "提示", str(e))
            self._render_empty()
            return None

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

    def _render_devices(self) -> None:
        speaker_opts = [(self._asset_label(d), d) for d in self.model.speaker_devices()]
        mic_opts = [(self._asset_label(d), d) for d in self.model.mic_devices()]
        self.view.speaker_device_table.set_options(speaker_opts)
        self.view.mic_device_table.set_options(mic_opts)

        # 清空麦克风通道（等设备勾选后再填充）
        self.view.mic_channel_table.set_options([])

    @staticmethod
    def _asset_label(device: Dict[str, Any]) -> str:
        return str(device.get("display_name") or device.get("device_name") or device.get("name") or "")

    def _render_empty(self) -> None:
        self.view.driver_combo.blockSignals(True)
        try:
            self.view.driver_combo.clear()
        finally:
            self.view.driver_combo.blockSignals(False)
        self.view.speaker_device_table.set_options([])
        self.view.mic_device_table.set_options([])
        self.view.mic_channel_table.set_options([])

    def _try_restore_selection(self, last_state: HardwareSelectionState) -> None:
        # 恢复驱动
        if last_state.api_name and last_state.api_name in self.model.devices_by_api:
            self.model.state.api_name = last_state.api_name
            self.view.driver_combo.setCurrentText(last_state.api_name)

        # 恢复设备：注册设备按 hardware_id；旧 runtime dict 按 hostapi_name + device_name 精确匹配。
        if last_state.speaker_device:
            self.view.speaker_device_table.set_checked_by_predicate(
                lambda p: self._matches_restore_device(p, last_state.speaker_device)
            )

        if last_state.mic_device:
            self.view.mic_device_table.set_checked_by_predicate(
                lambda p: self._matches_restore_device(p, last_state.mic_device)
            )

        # 恢复麦克风通道（在设备恢复后，通道表已刷新，这里再勾选）
        if last_state.mic_channels:
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
        self.model.set_api(api_name)
        self._render_devices()

    def _on_speaker_device_checked(self, payload: object) -> None:
        self.model.state.speaker_device = payload if isinstance(payload, dict) else None

    def _on_mic_device_checked(self, payload: object) -> None:
        self.model.state.mic_device = payload if isinstance(payload, dict) else None
        self.model.state.mic_channels = []
        try:
            channels = self.model.channel_rows_for_device(self.model.state.mic_device, "mic")
        except MissingHardwareTablesError:
            MessageBox.warning(self.view, "提示", "硬件管理表不存在，请使用最新版数据库")
            channels = []
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as e:
            MessageBox.warning(self.view, "提示", str(e))
            channels = []
        channel_options = self._mic_channel_options(channels)
        if channel_options is None:
            MessageBox.warning(self.view, "提示", "已注册麦克风通道数据无效，请检查硬件管理数据库。")
            channel_options = []
        self.view.mic_channel_table.set_options(channel_options)

    @classmethod
    def _mic_channel_options(cls, channels: List[Dict[str, Any]]) -> Optional[List[Tuple[Any, int]]]:
        options: List[Tuple[Any, int]] = []
        for channel in channels:
            channel_index = cls._validated_channel_index(channel.get("channel_index"))
            if channel_index is None:
                return None
            options.append((channel.get("channel_label", channel.get("channel_index")), channel_index))
        return options

    @staticmethod
    def _validated_channel_index(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            channel_index = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped.isdecimal():
                return None
            channel_index = int(stripped)
        else:
            return None
        if channel_index < 0:
            return None
        return channel_index

    def _on_ok_clicked(self) -> None:
        speaker = self.model.state.speaker_device
        mic = self.model.state.mic_device

        mic_channels = sorted(int(x) for x in self.view.mic_channel_table.checked_payloads())

        if not (speaker and mic and mic_channels):
            MessageBox.warning(self.view, "提示", "请选择扬声器和麦克风设备，以及麦克风通道！")
            return None

        SoundDeviceManager.refresh_available_device()

        try:
            runtime_devices = SoundDeviceManager.get_device_info()
        except Exception as e:
            MessageBox.warning(self.view, "提示", f"当前音频设备枚举失败，请检查设备连接或重试。\n{e}")
            return None

        runtime_index = self._runtime_devices_by_api(runtime_devices)
        mic_match = self._match_registered_asset(mic, runtime_index)
        speaker_match = self._match_registered_asset(speaker, runtime_index)
        if mic_match[0] == "missing" or speaker_match[0] == "missing":
            MessageBox.warning(self.view, "提示", "已注册硬件当前不可用，请检查设备连接后重试。")
            return None
        if mic_match[0] == "ambiguous" or speaker_match[0] == "ambiguous":
            MessageBox.warning(self.view, "提示", "已注册硬件匹配到多个当前设备，无法安全应用。")
            return None

        runtime_mic = mic_match[1]
        runtime_speaker = speaker_match[1]
        mic_max_input_channels = runtime_mic.get("max_input_channels")
        if not self._is_positive_int(mic_max_input_channels):
            MessageBox.warning(self.view, "提示", "当前麦克风设备不支持输入通道，请检查设备连接或重新选择。")
            return None
        if any(channel < 0 or channel >= mic_max_input_channels for channel in mic_channels):
            MessageBox.warning(self.view, "提示", "当前麦克风设备不支持所选输入通道，请检查设备连接或重新选择。")
            return None
        if not self._is_positive_int(runtime_speaker.get("max_output_channels")):
            MessageBox.warning(self.view, "提示", "当前扬声器设备不支持输出通道，请检查设备连接或重新选择。")
            return None

        mic_idx = int(runtime_mic.get("index"))
        speaker_idx = int(runtime_speaker.get("index"))
        mic_payload = self._augment_runtime_device(runtime_mic, mic)
        speaker_payload = self._augment_runtime_device(runtime_speaker, speaker)

        try:
            selected_devices_snapshot = self._snapshot_selected_devices_config()
        except OSError as e:
            MessageBox.warning(self.view, "提示", f"硬件配置保存失败，请检查权限或重试。\n{e}")
            return None

        try:
            SoundDeviceManager.save_selected_devices(mic_payload, speaker_payload, mic_channels)
        except Exception as e:
            try:
                self._restore_selected_devices_config(selected_devices_snapshot)
            except OSError as rollback_error:
                MessageBox.warning(
                    self.view,
                    "提示",
                    f"硬件配置保存失败，且已保存配置回滚失败，请检查权限或重试。\n{e}\n{rollback_error}",
                )
                return None
            MessageBox.warning(self.view, "提示", f"硬件配置保存失败，请检查权限或重试。\n{e}")
            return None

        try:
            SoundDeviceManager.change_default_device(mic_idx, speaker_idx)
        except Exception as e:
            try:
                self._restore_selected_devices_config(selected_devices_snapshot)
            except OSError as rollback_error:
                MessageBox.warning(
                    self.view,
                    "提示",
                    f"硬件配置应用失败，且已保存配置回滚失败，请检查权限或重试。\n{e}\n{rollback_error}",
                )
                return None
            MessageBox.warning(self.view, "提示", f"硬件配置应用失败，请检查设备连接或重试。\n{e}")
            return None

        self.model.state.speaker_device = speaker_payload
        self.model.state.mic_device = mic_payload
        self.model.state.mic_channels = mic_channels
        self.view.accept()

    @staticmethod
    def _is_positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _runtime_devices_by_api(runtime_devices: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
        devices_by_api: Dict[str, List[Dict[str, Any]]] = {}
        seen = set()
        for api_name, groups in (runtime_devices or {}).items():
            api_devices = devices_by_api.setdefault(api_name, [])
            for direction in ("input", "output"):
                for device in (groups or {}).get(direction, []) or []:
                    key = (
                        api_name,
                        device.get("index"),
                        device.get("name"),
                        id(device) if device.get("index") is None else None,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    api_devices.append(device)
        return devices_by_api

    @staticmethod
    def _match_registered_asset(asset: Dict[str, Any], runtime_index: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        matches = [
            device
            for device in runtime_index.get(asset.get("hostapi_name"), [])
            if device.get("name") == asset.get("device_name")
        ]
        if not matches:
            return "missing", None
        if len(matches) > 1:
            return "ambiguous", None
        return "ok", matches[0]

    @staticmethod
    def _augment_runtime_device(runtime_device: Dict[str, Any], asset: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(runtime_device)
        registered_metadata = {
            "hardware_id": asset.get("hardware_id"),
            "display_name": asset.get("display_name"),
            "device_name": asset.get("device_name"),
            "hardware_type": asset.get("hardware_type"),
            "hostapi_name": asset.get("hostapi_name"),
            "samplerate": asset.get("samplerate"),
            "bit_depth": normalize_float_bit_depth(asset.get("bit_depth")),
            "latency_ms": asset.get("latency_ms"),
        }
        for key, value in registered_metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    @staticmethod
    def _snapshot_selected_devices_config() -> Tuple[str, bool, bytes]:
        path = sound_device_manager_module.AUDIO_DEVICE_CONFIG_PATH
        try:
            with open(path, "rb") as file:
                return path, True, file.read()
        except FileNotFoundError:
            return path, False, b""

    @staticmethod
    def _restore_selected_devices_config(snapshot: Tuple[str, bool, bytes]) -> None:
        path, existed, data = snapshot
        if existed:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as file:
                file.write(data)
            return

        try:
            os.remove(path)
        except FileNotFoundError:
            return

    def get_selected_devices(self):
        if self.view.result() == QDialog.Accepted:
            return (
                self.model.state.speaker_device,
                self.model.state.mic_device,
                list(self.model.state.mic_channels),
            )
        return (
            self._initial_state.speaker_device,
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
            label = s.lower()
            # 支持 Out1 / In1 / 1；字符串通道按 1-based 解析，0 和非法标签忽略。
            if label.startswith("out"):
                s = s[3:]
            elif label.startswith("in"):
                s = s[2:]
            if not s.isdecimal():
                continue
            n = int(s)
            if n > 0:
                out.append(n - 1)
        elif isinstance(x, int) and not isinstance(x, bool):
            if x >= 0:
                out.append(x)
        else:
            continue
    # 去重 + 排序
    return sorted(set(out))


def open_hardware_selection_window(
    driver: Optional[str] = None,
    speaker_device: Optional[Dict[str, Any]] = None,
    mic_device: Optional[Dict[str, Any]] = None,
    mic_channels: Optional[Iterable[Any]] = None,
    repository=None,
):
    """
    打开硬件选择窗口，并支持根据传入参数进行初始化回填：
    - driver：Host API 名称（例如 WASAPI/ASIO/MME 等）
    - speaker_device / mic_device：sounddevice 设备 dict
    - mic_channels：通道索引（0-based int）或显示字符串（Out1/In1）
    """

    initial_state = HardwareSelectionState(
        speaker_device=speaker_device,
        mic_device=mic_device,
        mic_channels=_normalize_channel_indices(mic_channels),
        api_name=driver,
    )

    model = HardwareSelectionModel(initial_state=initial_state, repository=repository)
    view = HardwareSelectionView()
    controller = HardwareSelectionController(model, view)
    view.exec()
    result = controller.get_selected_devices()
    return result
