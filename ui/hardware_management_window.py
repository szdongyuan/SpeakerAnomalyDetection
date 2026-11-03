from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QVBoxLayout,
)

from base.hardware_management import (
    HardwareManagementError,
    HardwareManagementRepository,
)
from base.sound_device_manager import SoundDeviceManager, sd
from base.vkinging_input import is_vkinging_device, validate_sample_rate
from consts import model_consts
from consts.audio_consts import VALID_SAMPLE_RATES, VALID_BIT_DEPTHS, DEVICE_ENUMERATION_BASE_EXCEPTIONS
from consts.vkinging_consts import VKINGING_SAMPLE_RATE_MIN, VKINGING_SAMPLE_RATE_MAX
from ui.custom_ui_widget.widgets import ComboBox, Label, LineEdit, MessageBox, PushButton, SpinBox, TableView


class VkingingSampleRateControl(ComboBox):
    """Convenience presets plus strict free-form integer entry at commit time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(ComboBox.NoInsert)
        for rate in (44100, 48000, 51200):
            self.addItem(str(rate), rate)
        self.setValue(51200)
        self.setToolTip("可选择常用采样率，或输入 4000–102400 Hz 范围内的整数。")

    def minimum(self):
        return VKINGING_SAMPLE_RATE_MIN

    def maximum(self):
        return VKINGING_SAMPLE_RATE_MAX

    def setValue(self, value):
        self.setCurrentText(str(validate_sample_rate(value)))

    def value(self):
        # Leave invalid text visible; never clamp it or recover a previous value.
        text = self.currentText().strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError("采样率必须为 4000–102400 Hz 范围内的整数。")
        return validate_sample_rate(int(text))


class HardwareRegistrationDialog(QDialog):
    def __init__(self, parent=None, repository=None, vkinging_bridge=None):
        super().__init__(parent)
        self.repository = repository or HardwareManagementRepository()
        self.vkinging_bridge = vkinging_bridge
        self._devices_by_api = {}
        self._discovery_error = None
        self._init_ui()
        self._load_devices()
        if self.vkinging_bridge is not None:
            self.vkinging_bridge.state_changed.connect(self._on_backend_state)
            self._request_discovery()

    def _init_ui(self):
        self.setWindowTitle("注册硬件")

        self.api_combo = ComboBox()
        self.device_combo = ComboBox()
        self.display_name_edit = LineEdit()
        self.display_name_edit.setPlaceholderText("未填写时使用设备名称")
        self.samplerate_combo = ComboBox()
        self.samplerate_spin = VkingingSampleRateControl()
        self.samplerate_spin.hide()
        self.bit_depth_combo = ComboBox()
        self.latency_spin = SpinBox()

        self._set_samplerate_options(device_selected=False)

        for bit_depth in VALID_BIT_DEPTHS:
            self.bit_depth_combo.addItem(str(bit_depth), bit_depth)
        self.bit_depth_combo.setCurrentText("32")

        self.latency_spin.setRange(0, 1000)
        self.latency_spin.setValue(100)

        form = QFormLayout()
        form.addRow(Label("驱动"), self.api_combo)
        form.addRow(Label("显示名称"), self.display_name_edit)
        form.addRow(Label("设备"), self.device_combo)
        samplerate_layout = QHBoxLayout()
        samplerate_layout.addWidget(self.samplerate_combo)
        samplerate_layout.addWidget(self.samplerate_spin)
        form.addRow(Label("采样率"), samplerate_layout)
        form.addRow(Label("位深"), self.bit_depth_combo)

        self.refresh_btn = PushButton("刷新")
        self.register_btn = PushButton("注册")
        self.cancel_btn = PushButton("取消")
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.register_btn)
        buttons.addWidget(self.cancel_btn)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.api_combo.currentTextChanged.connect(self._on_api_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.register_btn.clicked.connect(self.accept_registration)
        self.cancel_btn.clicked.connect(self.reject)

    def _load_devices(self, preferred_api=None, preferred_device_key=None):
        self._discovery_error = None
        self._devices_by_api = self._collect_devices_by_api()
        if self.vkinging_bridge is not None:
            self._devices_by_api["vkinging"] = [dict(d) for d in self.vkinging_bridge.devices]
        if self._discovery_error is not None:
            MessageBox.warning(self, "提示", str(self._discovery_error))
        self.api_combo.blockSignals(True)
        self.api_combo.clear()
        self.api_combo.addItems(list(self._devices_by_api.keys()))
        preferred_api_available = (
            preferred_api is not None
            and self.api_combo.findText(preferred_api) >= 0
        )
        if preferred_api_available:
            self.api_combo.setCurrentText(preferred_api)
        self.api_combo.blockSignals(False)

        if self.api_combo.count() > 0:
            self._on_api_changed(self.api_combo.currentText())
            if (
                preferred_api_available
                and self.api_combo.currentText() == preferred_api
                and preferred_device_key is not None
            ):
                self._restore_device_selection(preferred_device_key)
        else:
            self._populate_devices([])

    def _on_backend_state(self, state):
        self.refresh_btn.setEnabled(state == "idle")
        self.register_btn.setEnabled(state == "idle")

    def _request_discovery(self):
        request_id = self.vkinging_bridge.request_discovery(self, self._on_discovery_event)
        self._on_backend_state(self.vkinging_bridge.state)
        return request_id

    def _on_discovery_event(self, event):
        if event.kind not in ("succeeded", "failed", "cancelled"):
            return
        api = self.api_combo.currentText()
        key = self._device_restore_key(self.device_combo.currentData())
        self._load_devices(preferred_api=api, preferred_device_key=key)
        if self.vkinging_bridge.diagnostics:
            MessageBox.warning(self, "设备发现", "\n".join(self.vkinging_bridge.diagnostics))

    def refresh_devices(self):
        if self.vkinging_bridge is not None:
            if self.vkinging_bridge.busy:
                return
            self._request_discovery()
        previous_api = self.api_combo.currentText()
        previous_device_key = self._device_restore_key(self.device_combo.currentData())
        try:
            SoundDeviceManager.refresh_available_device()
        except DEVICE_ENUMERATION_BASE_EXCEPTIONS + (getattr(sd, "PortAudioError", RuntimeError),) as exc:
            MessageBox.warning(self, "提示", str(exc))
            return
        self._load_devices(preferred_api=previous_api, preferred_device_key=previous_device_key)

    def _collect_devices_by_api(self):
        api_names = []
        api_device_indexes = {}
        try:
            api_info = SoundDeviceManager.get_api_info()
        except DEVICE_ENUMERATION_BASE_EXCEPTIONS + (getattr(sd, "PortAudioError", RuntimeError),) as exc:
            self._discovery_error = exc
            return {}
        if isinstance(api_info, dict):
            api_info = [api_info]
        for api in api_info or []:
            name = api.get("name") if isinstance(api, dict) else None
            if name and name not in api_names:
                api_names.append(name)
            if name and isinstance(api, dict):
                api_device_indexes[name] = list(api.get("devices") or [])

        try:
            devices_by_api = SoundDeviceManager.get_device_info() or {}
        except DEVICE_ENUMERATION_BASE_EXCEPTIONS + (getattr(sd, "PortAudioError", RuntimeError),) as exc:
            self._discovery_error = exc
            return {}
        try:
            runtime_devices = sd.query_devices() or []
        except DEVICE_ENUMERATION_BASE_EXCEPTIONS + (getattr(sd, "PortAudioError", RuntimeError),) as exc:
            self._discovery_error = exc
            return {}

        if not api_names:
            api_names = list(devices_by_api.keys())

        collected = {}
        for api_name in api_names:
            api_devices = devices_by_api.get(api_name, {}) or {}
            unique_devices = []
            seen = set()

            def add_device(device):
                if not isinstance(device, dict):
                    return
                key = (device.get("index"), device.get("name"))
                if key in seen:
                    return
                seen.add(key)
                unique_devices.append(device)

            for device_index in api_device_indexes.get(api_name, []):
                try:
                    add_device(runtime_devices[device_index])
                except (IndexError, KeyError, TypeError):
                    continue
            for device in list(api_devices.get("input", []) or []) + list(api_devices.get("output", []) or []):
                add_device(device)
            collected[api_name] = unique_devices
        return collected

    def _populate_devices(self, devices):
        self.device_combo.blockSignals(True)
        try:
            self.device_combo.clear()
            self.device_combo.addItem("", None)
            for device in devices:
                self.device_combo.addItem(device.get("name", ""), device)
        finally:
            self.device_combo.blockSignals(False)
        self._on_device_changed(0)

    @staticmethod
    def _device_restore_key(device):
        if not isinstance(device, dict):
            return None
        if is_vkinging_device(device):
            return ("vkinging", device.get("hardware_id"))
        index = device.get("index")
        name = device.get("name")
        if index is not None:
            return ("index_name", index, name)
        return ("name", name)

    @classmethod
    def _device_matches_restore_key(cls, device, restore_key):
        if not isinstance(device, dict) or restore_key is None:
            return False
        if restore_key[0] == "vkinging":
            return is_vkinging_device(device) and device.get("hardware_id") == restore_key[1]
        if restore_key[0] == "index_name":
            return device.get("index") == restore_key[1] and device.get("name") == restore_key[2]
        return False

    def _restore_device_selection(self, restore_key):
        if restore_key is None:
            return
        for index in range(1, self.device_combo.count()):
            if self._device_matches_restore_key(self.device_combo.itemData(index), restore_key):
                self.device_combo.setCurrentIndex(index)
                return

        if restore_key[0] == "index_name":
            name = restore_key[2]
        elif restore_key[0] == "name":
            name = restore_key[1]
        else:
            return
        if not name:
            return

        same_name_indexes = [
            index
            for index in range(1, self.device_combo.count())
            if (
                isinstance(self.device_combo.itemData(index), dict)
                and self.device_combo.itemData(index).get("name") == name
            )
        ]
        if len(same_name_indexes) == 1:
            self.device_combo.setCurrentIndex(same_name_indexes[0])

    def _on_api_changed(self, api_name):
        self._populate_devices(self._devices_by_api.get(api_name, []))

    def _on_device_changed(self, _index):
        device = self.device_combo.currentData()
        native = is_vkinging_device(device)
        self.samplerate_spin.setVisible(native)
        self.samplerate_combo.setVisible(not native)
        if native:
            self.samplerate_spin.setValue(device.get("samplerate", 51200))
            return
        if not device:
            self._set_samplerate_options(device_selected=False)
            return

        self._set_samplerate_options(device_selected=True)
        default_samplerate = int(float(device.get("default_samplerate") or 0))
        samplerate = default_samplerate if default_samplerate in VALID_SAMPLE_RATES else 48000
        self.samplerate_combo.setCurrentText(str(samplerate))

    def _set_samplerate_options(self, device_selected):
        self.samplerate_combo.blockSignals(True)
        try:
            self.samplerate_combo.clear()
            if not device_selected:
                self.samplerate_combo.addItem("")
                return
            for samplerate in VALID_SAMPLE_RATES:
                self.samplerate_combo.addItem(str(samplerate), samplerate)
        finally:
            self.samplerate_combo.blockSignals(False)

    def accept_registration(self):
        if self.vkinging_bridge is not None and self.vkinging_bridge.busy:
            MessageBox.warning(self, "提示", "设备正在使用或尚未释放，请稍后重试。")
            return
        device = self.device_combo.currentData()
        if self._discovery_error is not None and not is_vkinging_device(device):
            MessageBox.warning(self, "提示", str(self._discovery_error))
            return
        if not device:
            MessageBox.warning(self, "提示", "请选择设备")
            return
        try:
            samplerate = self.samplerate_spin.value() if is_vkinging_device(device) else int(self.samplerate_combo.currentText())
            bit_depth = int(self.bit_depth_combo.currentData())
            latency_ms = int(self.latency_spin.value())
        except (TypeError, ValueError):
            MessageBox.warning(self, "提示", "硬件参数无效")
            return

        display_name = self.display_name_edit.text().strip()
        if not display_name:
            display_name = str(device.get("name") or "").strip()

        try:
            self.repository.register_asset(
                device,
                self.api_combo.currentText(),
                display_name,
                samplerate,
                bit_depth,
                latency_ms,
            )
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            MessageBox.warning(self, "提示", str(exc))
            return

        self.accept()


class HardwareManagementTableModel(QAbstractTableModel):
    READ_ONLY_COLUMNS = (
        "hardware_id",
        "hardware_type",
        "device_name",
        "hostapi_name",
        "max_input_channels",
        "max_output_channels",
        "updated_at",
    )
    EDITABLE_COLUMNS = ("display_name", "samplerate", "bit_depth")
    COLUMNS = (
        "display_name",
        "hostapi_name",
        "device_name",
        "hardware_type",
        "samplerate",
        "bit_depth",
        "latency_ms",
        "max_input_channels",
        "max_output_channels",
        "updated_at",
    )
    HEADER_LABELS = {
        "hardware_id": "硬件ID",
        "hardware_type": "类型",
        "display_name": "显示名称",
        "device_name": "设备名称",
        "hostapi_name": "驱动",
        "samplerate": "采样率",
        "bit_depth": "位深",
        "latency_ms": "延迟(ms)",
        "max_input_channels": "输入通道",
        "max_output_channels": "输出通道",
        "updated_at": "更新时间",
    }

    selection_restore_requested = pyqtSignal(str)
    asset_updated = pyqtSignal(str, str, dict)

    columns = COLUMNS

    def __init__(self, repository, parent=None, load_initial=True, mutation_allowed=None):
        super().__init__(parent)
        self.repository = repository
        self.mutation_allowed = mutation_allowed
        self.assets = []
        self._checked_hardware_id = None
        if load_initial:
            self.reload()

    def reload(self):
        assets = self.repository.list_assets()
        self.beginResetModel()
        try:
            self.assets = assets
            if self.row_for_hardware_id(self._checked_hardware_id) is None:
                self._checked_hardware_id = None
        finally:
            self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self.assets)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self.columns)

    def column_index(self, field):
        return self.columns.index(field)

    def row_for_hardware_id(self, hardware_id):
        for row, asset in enumerate(self.assets):
            if asset.get("hardware_id") == hardware_id:
                return row
        return None

    def checked_hardware_id(self):
        return self._checked_hardware_id

    def checked_row(self):
        return self.row_for_hardware_id(self._checked_hardware_id)

    def set_checked_hardware_id(self, hardware_id):
        hardware_id = str(hardware_id) if hardware_id is not None else None
        if hardware_id is not None and self.row_for_hardware_id(hardware_id) is None:
            hardware_id = None
        if hardware_id == self._checked_hardware_id:
            return
        old_row = self.checked_row()
        self._checked_hardware_id = hardware_id
        new_row = self.checked_row()
        self._emit_check_state_changed(old_row)
        self._emit_check_state_changed(new_row)

    def _emit_check_state_changed(self, row):
        if row is None:
            return
        index = self.index(row, self.column_index("display_name"))
        self.dataChanged.emit(index, index, [Qt.CheckStateRole])

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.assets):
            return None
        field = self.columns[index.column()]
        value = self.assets[index.row()].get(field)
        if role in (Qt.DisplayRole, Qt.EditRole):
            return value
        if role == Qt.CheckStateRole and field == "display_name":
            return Qt.Checked if self.assets[index.row()].get("hardware_id") == self._checked_hardware_id else Qt.Unchecked
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.HEADER_LABELS.get(self.columns[section], self.columns[section])
        return section + 1

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        field = self.columns[index.column()]
        if field in self.EDITABLE_COLUMNS and (self.mutation_allowed is None or self.mutation_allowed()):
            flags |= Qt.ItemIsEditable
        if field == "display_name":
            flags |= Qt.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        field = self.columns[index.column()]
        row = index.row()
        if row >= len(self.assets):
            return False
        if role == Qt.CheckStateRole and field == "display_name":
            hardware_id = str(self.assets[row]["hardware_id"])
            if value == Qt.Checked:
                self.set_checked_hardware_id(hardware_id)
            elif value == Qt.Unchecked and self._checked_hardware_id == hardware_id:
                self.set_checked_hardware_id(None)
            return True
        if role != Qt.EditRole:
            return False
        if self.mutation_allowed is not None and not self.mutation_allowed():
            return False
        if field not in self.EDITABLE_COLUMNS:
            return False

        old_asset = dict(self.assets[row])
        try:
            if field == "samplerate" and is_vkinging_device(old_asset):
                normalized_value = validate_sample_rate(value)
            else:
                normalized_value = self._normalize_value(field, value)
        except ValueError as exc:
            MessageBox.warning(None, "提示", str(exc))
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return False

        hardware_id = old_asset["hardware_id"]
        try:
            updated = self.repository.update_asset_fields(hardware_id, {field: normalized_value})
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            MessageBox.warning(None, "提示", str(exc))
            try:
                self.reload()
            except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as reload_exc:
                if row < len(self.assets):
                    self.assets[row] = old_asset
                MessageBox.warning(None, "提示", str(reload_exc))
                self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return False
        if not updated:
            MessageBox.warning(None, "提示", "硬件记录不存在")
            try:
                self.reload()
            except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as reload_exc:
                if row < len(self.assets):
                    self.assets[row] = old_asset
                MessageBox.warning(None, "提示", str(reload_exc))
                self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return False

        local_asset = dict(old_asset)
        local_asset[field] = normalized_value
        self.assets[row] = local_asset
        try:
            refreshed_asset = self.repository.get_asset(hardware_id)
            readback_error = None
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            refreshed_asset = None
            readback_error = exc
        if not refreshed_asset:
            try:
                self.reload()
            except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as reload_exc:
                if row < len(self.assets):
                    self.assets[row] = local_asset
                if readback_error is not None:
                    MessageBox.warning(None, "提示", str(readback_error))
                MessageBox.warning(None, "提示", str(reload_exc))
                first = self.index(row, 0)
                last = self.index(row, len(self.columns) - 1)
                self.dataChanged.emit(first, last, [Qt.DisplayRole, Qt.EditRole])
                self.asset_updated.emit(str(hardware_id), field, dict(local_asset))
                return True
            reloaded_row = self.row_for_hardware_id(hardware_id)
            if reloaded_row is not None:
                self.asset_updated.emit(str(hardware_id), field, dict(self.assets[reloaded_row]))
            return True
        self.assets[row] = refreshed_asset
        first = self.index(row, 0)
        last = self.index(row, len(self.columns) - 1)
        self.dataChanged.emit(first, last, [Qt.DisplayRole, Qt.EditRole])
        self.asset_updated.emit(str(hardware_id), field, dict(refreshed_asset))
        return True

    def _keep_committed_field_visible(self, hardware_id, field, value):
        row = self.row_for_hardware_id(hardware_id)
        if row is None or self.assets[row].get(field) == value:
            return
        self.assets[row] = dict(self.assets[row])
        self.assets[row][field] = value
        index = self.index(row, self.column_index(field))
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])

    @staticmethod
    def _normalize_value(field, value):
        if field == "display_name":
            text = str(value).strip()
            if not text:
                raise ValueError("显示名称不能为空")
            return text
        if field == "samplerate":
            try:
                samplerate = int(value)
            except (TypeError, ValueError):
                raise ValueError("采样率必须为 44100 或 48000")
            if samplerate not in VALID_SAMPLE_RATES:
                raise ValueError("采样率必须为 44100 或 48000")
            return samplerate
        if field == "bit_depth":
            try:
                bit_depth = int(value)
            except (TypeError, ValueError):
                raise ValueError("位深必须为 32 或 64")
            if bit_depth not in VALID_BIT_DEPTHS:
                raise ValueError("位深必须为 32 或 64")
            return bit_depth
        if field == "latency_ms":
            try:
                latency = int(value)
            except (TypeError, ValueError):
                raise ValueError("延迟必须为 0 到 1000 的整数")
            if latency < 0 or latency > 1000:
                raise ValueError("延迟必须为 0 到 1000 的整数")
            return latency
        return value

    def remove_row(self, row):
        if row < 0 or row >= len(self.assets):
            return False
        removed_hardware_id = self.assets[row].get("hardware_id")
        self.beginRemoveRows(QModelIndex(), row, row)
        try:
            del self.assets[row]
        finally:
            self.endRemoveRows()
        if removed_hardware_id == self._checked_hardware_id:
            self._checked_hardware_id = None
        return True


class FocuslessItemDelegate(QStyledItemDelegate):
    @staticmethod
    def focusless_option(option):
        focusless = QStyleOptionViewItem(option)
        focusless.state &= ~QStyle.State_HasFocus
        focusless.state &= ~QStyle.State_Selected
        return focusless

    def paint(self, painter, option, index):
        super().paint(painter, self.focusless_option(option), index)


class ComboValueDelegate(FocuslessItemDelegate):
    def __init__(self, choices, parent=None):
        super().__init__(parent)
        self.choices = list(choices)

    def createEditor(self, parent, _option, _index):
        editor = ComboBox(parent)
        for text, value in self.choices:
            editor.addItem(text, value)
        editor._combo_value_delegate_ready = False
        editor.currentIndexChanged.connect(lambda _idx: self._commit_if_ready(editor))
        return editor

    def _commit_if_ready(self, editor):
        if getattr(editor, "_combo_value_delegate_ready", False):
            self.commitData.emit(editor)

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        found = editor.findData(value)
        if found < 0:
            found = editor.findText(str(value))
        editor.blockSignals(True)
        try:
            editor.setCurrentIndex(max(0, found))
        finally:
            editor.blockSignals(False)
            editor._combo_value_delegate_ready = True

    def setModelData(self, editor, model, index):
        value = editor.currentData()
        if value != model.data(index, Qt.EditRole):
            model.setData(index, value, Qt.EditRole)


class HardwareSampleRateDelegate(ComboValueDelegate):
    def createEditor(self, parent, option, index):
        asset = index.model().assets[index.row()]
        if is_vkinging_device(asset):
            return VkingingSampleRateControl(parent)
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, VkingingSampleRateControl):
            editor.setValue(index.data(Qt.EditRole))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, VkingingSampleRateControl):
            try:
                value = editor.value()
            except ValueError as exc:
                MessageBox.warning(editor, "采样率配置", str(exc))
                return
            model.setData(index, value, Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class LatencyDelegate(QStyledItemDelegate):
    def createEditor(self, parent, _option, _index):
        editor = SpinBox(parent)
        editor.setRange(0, 1000)
        return editor

    def setEditorData(self, editor, index):
        editor.setValue(int(index.model().data(index, Qt.EditRole) or 0))

    def setModelData(self, editor, model, index):
        model.setData(index, int(editor.value()), Qt.EditRole)


class HardwareManagementWindow(QDialog):
    def __init__(
        self,
        parent=None,
        repository=None,
        on_selected_devices_invalidated=None,
        on_selected_devices_updated=None,
        selected_devices_provider=None,
        audio_workflow_active_provider=None,
        vkinging_bridge=None,
    ):
        super().__init__(parent)
        self.repository = repository or HardwareManagementRepository()
        self.on_selected_devices_invalidated = on_selected_devices_invalidated
        self.on_selected_devices_updated = on_selected_devices_updated
        self.selected_devices_provider = selected_devices_provider
        self.audio_workflow_active_provider = audio_workflow_active_provider
        self.vkinging_bridge = vkinging_bridge
        self.initial_load_failed = False
        self._init_ui()
        try:
            self.model = HardwareManagementTableModel(self.repository, self, mutation_allowed=lambda: not self._audio_workflow_active())
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            self.initial_load_failed = True
            MessageBox.warning(self, "提示", str(exc))
            self.model = HardwareManagementTableModel(self.repository, self, load_initial=False, mutation_allowed=lambda: not self._audio_workflow_active())
        self.table.setModel(self.model)
        self.model.selection_restore_requested.connect(self._restore_selected_hardware_id)
        self.model.asset_updated.connect(self._notify_selected_devices_updated)
        self._configure_table()

    def _init_ui(self):
        self.setWindowTitle("硬件管理")
        self.table = TableView()
        self.register_btn = PushButton("注册")
        self.delete_btn = PushButton("删除")

        actions = QHBoxLayout()
        actions.addWidget(self.register_btn)
        actions.addWidget(self.delete_btn)
        actions.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(actions)
        self.setLayout(layout)
        self.resize(1180, 620)

        self.register_btn.clicked.connect(self.open_registration_dialog)
        self.delete_btn.clicked.connect(self.delete_selected_asset)

    def _configure_table(self):
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(TableView.SelectItems)
        self.table.setSelectionMode(TableView.SingleSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet(f"{self.table.styleSheet()}\nTableView::item:focus {{ outline: none; }}")
        self.table.setItemDelegate(FocuslessItemDelegate(self.table))
        self.table.setItemDelegateForColumn(
            self.model.column_index("samplerate"),
            HardwareSampleRateDelegate([(str(rate), rate) for rate in VALID_SAMPLE_RATES], self.table),
        )
        self.table.setItemDelegateForColumn(
            self.model.column_index("bit_depth"),
            ComboValueDelegate([(f"{depth}bit", depth) for depth in VALID_BIT_DEPTHS], self.table),
        )
        self.table.setColumnHidden(self.model.column_index("latency_ms"), True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.model.column_index("display_name"), QHeaderView.Stretch)
        header.setSectionResizeMode(self.model.column_index("device_name"), QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)

    def refresh(self):
        try:
            self.model.reload()
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            MessageBox.warning(self, "提示", str(exc))

    def _restore_selected_hardware_id(self, hardware_id):
        self.model.set_checked_hardware_id(hardware_id)

    def open_registration_dialog(self):
        if self._audio_workflow_active():
            self._warn_hardware_change_during_audio_workflow()
            return
        dialog = HardwareRegistrationDialog(self, repository=self.repository, vkinging_bridge=self.vkinging_bridge)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh()

    @staticmethod
    def _device_matches_hardware_id(device, hardware_id):
        return isinstance(device, dict) and str(device.get("hardware_id")) == str(hardware_id)

    def _current_selection_matches_hardware_id(self, hardware_id):
        if not callable(self.selected_devices_provider):
            return False
        selected_devices = self.selected_devices_provider()
        if isinstance(selected_devices, dict):
            candidates = (selected_devices.get("mic"), selected_devices.get("speaker"))
        else:
            try:
                candidates = tuple(selected_devices)
            except TypeError:
                candidates = ()
        return any(self._device_matches_hardware_id(device, hardware_id) for device in candidates[:2])

    def _notify_selected_devices_updated(self, hardware_id, _field, asset):
        if not self._current_selection_matches_hardware_id(hardware_id):
            return
        if callable(self.on_selected_devices_updated):
            self.on_selected_devices_updated(hardware_id, dict(asset))

    @staticmethod
    def _clear_result_matched(clear_result):
        if hasattr(clear_result, "matched"):
            return bool(clear_result.matched)
        return bool(clear_result)

    @staticmethod
    def _clear_result_failed(clear_result):
        return bool(getattr(clear_result, "clear_failed", False))

    def _warn_selected_device_cache_clear_failed(self, clear_result):
        message = "已删除硬件，但未能清除已保存的音频设备选择，请重新选择设备。"
        error = getattr(clear_result, "error", "")
        if error:
            message = f"{message}\n{error}"
        MessageBox.warning(self, "提示", message)

    def _audio_workflow_active(self):
        if self.vkinging_bridge is not None and self.vkinging_bridge.busy:
            return True
        if not callable(self.audio_workflow_active_provider):
            return False
        return bool(self.audio_workflow_active_provider())

    def _warn_hardware_change_during_audio_workflow(self):
        MessageBox.warning(self, "提示", "播放或录音进行中，请等待完成后再修改硬件设置")

    def delete_selected_asset(self):
        row = self.model.checked_row()
        if row is None:
            MessageBox.warning(self, "提示", "请选择要删除的硬件")
            return
        if self._audio_workflow_active():
            self._warn_hardware_change_during_audio_workflow()
            return
        hardware_id = self.model.assets[row]["hardware_id"]
        result = MessageBox.question(self, "确认删除", "确认删除选中的硬件？")
        if result != MessageBox.Yes:
            return
        if self._audio_workflow_active():
            self._warn_hardware_change_during_audio_workflow()
            return
        try:
            deleted = self.repository.delete_asset(hardware_id)
        except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
            MessageBox.warning(self, "提示", str(exc))
            self.refresh()
            return
        if not deleted:
            MessageBox.warning(self, "提示", "硬件记录不存在，删除失败")
            self.refresh()
            return
        current_selection_matched = self._current_selection_matches_hardware_id(hardware_id)
        selected_devices_clear_result = SoundDeviceManager.clear_selected_devices_for_deleted_hardware(hardware_id)
        if self._clear_result_failed(selected_devices_clear_result):
            self._warn_selected_device_cache_clear_failed(selected_devices_clear_result)
        if (
            (current_selection_matched or self._clear_result_matched(selected_devices_clear_result))
            and callable(self.on_selected_devices_invalidated)
        ):
            self.on_selected_devices_invalidated(hardware_id)
        self.model.remove_row(row)
        self.refresh()


def open_hardware_management_window(parent=None, repository=None, audio_workflow_active_provider=None, vkinging_bridge=None):
    repository = repository or HardwareManagementRepository()
    try:
        tables_exist = repository.tables_exist()
    except (HardwareManagementError,) + model_consts.SQLITE_REPOSITORY_EXCEPTIONS as exc:
        MessageBox.warning(parent, "提示", str(exc))
        return None
    if not tables_exist:
        MessageBox.warning(parent, "提示", "硬件管理表不存在，请使用最新版数据库")
        return None

    on_selected_devices_invalidated = getattr(parent, "on_selected_audio_hardware_deleted", None)
    if not callable(on_selected_devices_invalidated):
        on_selected_devices_invalidated = None
    on_selected_devices_updated = getattr(parent, "on_registered_audio_hardware_updated", None)
    if not callable(on_selected_devices_updated):
        on_selected_devices_updated = None
    selected_devices_provider = None
    if parent is not None:
        selected_devices_provider = lambda: (getattr(parent, "mic", None), getattr(parent, "speaker", None))

    window = HardwareManagementWindow(
        parent=parent,
        repository=repository,
        on_selected_devices_invalidated=on_selected_devices_invalidated,
        on_selected_devices_updated=on_selected_devices_updated,
        selected_devices_provider=selected_devices_provider,
        audio_workflow_active_provider=audio_workflow_active_provider,
        vkinging_bridge=vkinging_bridge,
    )
    if window.initial_load_failed:
        return None
    window.exec_()
    return window
