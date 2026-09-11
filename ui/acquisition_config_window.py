import os
import sys
from collections.abc import Mapping
from copy import deepcopy

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QStyle, QVBoxLayout
from PyQt5.QtWidgets import QToolButton, QWidget

from base.sound_device_manager import SoundDeviceManager
from base.ve3668n_input import validate_range_index
from base.ve3668n_recording_config import resolve_ve_recording_config
from base.recording_preview_config import (
    resolve_recording_preview_time_mode,
    validate_recording_preview_time_mode,
)
from consts import model_consts
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from consts.running_consts import DEFAULT_DIR
from consts.ve3668n_consts import (
    VE_BACKEND, VE_RANGE_INDEX_CONFIG_KEY, VE_RANGE_LABELS,
    VE_SAMPLE_RATE_MIN, VE_SAMPLE_RATE_MAX, VE_SAMPLE_RATES,
)
from ui.config_dialog_base import ConfigDialogBase


class BaseConfigWindow(ConfigDialogBase):
    def __init__(self, mic=None):
        super().__init__()
        self.final_data = None
        if mic is not None:
            self.mic = mic
        else:
            _, self.mic = SoundDeviceManager().get_default_device("mic", refresh=False)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        self.main_layout = QVBoxLayout(self)

        self.apply_config_dialog_theme()

    def create_cancel_ok_buttons(self):
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton(" 取 消")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        ok_btn = QPushButton(" 确 认")
        ok_btn.clicked.connect(self.on_click_ok_btn)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def on_click_ok_btn(self):
        pass

    def on_click_cancel_btn(self):
        self.close()

    def exec(self):
        super().exec()
        return self.final_data


class RecordConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None, speaker=None, speaker_channels=None, *, ve_profile_provider=None):
        super().__init__(mic=mic)
        self.setWindowTitle("录制音频")
        self.setMinimumWidth(560)
        self.resize(560, 420)
        self.input_data = input_data or {}
        self._is_vk = (self.mic or {}).get("backend") == VE_BACKEND
        self.ve_profile_provider = ve_profile_provider
        if speaker is not None or (self.mic or {}).get("backend") == "vkinging":
            self.speaker = speaker
        else:
            _, self.speaker = SoundDeviceManager().get_default_device("speaker", refresh=False)
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = QGroupBox("输入")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_time = QLabel("音频时长:")
        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.1, 600)
        self.time_input.setDecimals(1)
        self.time_input.setValue(float(self.input_data.get("total_time", 4.0)))
        self.time_input.setSingleStep(0.1)
        self.time_input.setSuffix(" 秒")

        label_samplerate = QLabel("采样率:")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        self.samplerate_combo.setCurrentText(str(self.input_data.get("sample_rate", 44100)))
        self._sample_rate_load_error = None
        if self._is_vk:
            self.samplerate_combo.clear()
            self.samplerate_combo.setEditable(True)
            self.samplerate_combo.setInsertPolicy(QComboBox.NoInsert)
            self.samplerate_combo.addItems([str(rate) for rate in VE_SAMPLE_RATES])
            self.samplerate_combo.setValidator(QIntValidator(
                VE_SAMPLE_RATE_MIN, VE_SAMPLE_RATE_MAX, self.samplerate_combo))
            rate_detail = ({"sample_rate": self.input_data["sample_rate"]}
                           if "sample_rate" in self.input_data else {})
            profile = self.mic.get("input_config")
            try:
                # Discovery supplies identity, not the persisted queue fallback.
                # Explicit values stay in this dialog's independent repair flow.
                if not rate_detail and self.ve_profile_provider is not None:
                    profile = None
                    profile = self.ve_profile_provider(self.mic)
                rate = resolve_ve_recording_config(
                    rate_detail, fallback_profile=profile)["sample_rate"]
            except (ValueError, OSError) as exc:
                self._sample_rate_load_error = str(exc)
                rate = (self.input_data["sample_rate"] if "sample_rate" in self.input_data
                        else profile.get("sample_rate", "配置不可用")
                        if isinstance(profile, Mapping) else "配置不可用")
                self.samplerate_combo.setToolTip(str(exc))
            self.samplerate_combo.setEditText(str(rate))
            self.samplerate_combo.editTextChanged.connect(self._on_sample_rate_edited)
            self.samplerate_combo.lineEdit().textEdited.connect(self._on_sample_rate_edited)
            self.samplerate_combo.activated.connect(self._on_sample_rate_edited)
        else:
            loaded_rate = str(self.input_data.get("sample_rate", 44100))
            if self.samplerate_combo.findText(loaded_rate) == -1:
                self.samplerate_combo.addItem(loaded_rate)
                index = self.samplerate_combo.count() - 1
                self.samplerate_combo.setCurrentIndex(index)
                self.samplerate_combo.model().item(index).setEnabled(False)

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        else:
            self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")

        self.recording_advanced_toggle = QToolButton()
        self.recording_advanced_toggle.setObjectName("recording_advanced_toggle")
        self.recording_advanced_toggle.setText("高级设置")
        self.recording_advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.recording_advanced_toggle.setArrowType(Qt.RightArrow)
        self.recording_advanced_toggle.setCheckable(True)
        self.recording_advanced_panel = QWidget()
        self.recording_advanced_panel.setObjectName("recording_advanced_panel")
        advanced_layout = QGridLayout(self.recording_advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setHorizontalSpacing(20)
        advanced_layout.setVerticalSpacing(15)
        self.recording_advanced_panel.hide()
        self.recording_advanced_toggle.toggled.connect(self._set_advanced_visible)

        label_streaming_recording = QLabel("实时波形:")
        self.streaming_recording_checkbox = QCheckBox("启用")
        self.streaming_recording_checkbox.setChecked(
            bool(self.input_data.get("use_streaming_recording", False))
        )
        self.preview_time_mode_label = QLabel("预览显示方式:")
        self.preview_time_mode_combo = QComboBox()
        self.preview_time_mode_combo.addItem(
            "最新 10 秒", PREVIEW_TIME_MODE_RELATIVE_LATEST
        )
        self.preview_time_mode_combo.addItem(
            "累计显示", PREVIEW_TIME_MODE_CUMULATIVE
        )
        self.preview_time_mode_error_label = QLabel()
        self.preview_time_mode_error_label.setWordWrap(True)
        self._invalid_preview_time_mode = None
        self._preview_time_mode_needs_repair = False
        try:
            preview_time_mode = resolve_recording_preview_time_mode(self.input_data)
        except ValueError as exc:
            self._preview_time_mode_needs_repair = True
            self._invalid_preview_time_mode = self.input_data.get(
                RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
            )
            self.preview_time_mode_combo.setCurrentIndex(-1)
            self.preview_time_mode_error_label.setText(str(exc))
        else:
            self.preview_time_mode_combo.setCurrentIndex(
                self.preview_time_mode_combo.findData(preview_time_mode)
            )
        self.preview_time_mode_combo.currentIndexChanged.connect(
            self._on_preview_time_mode_changed
        )
        label_recording_root = QLabel("音频保存根目录:")
        self.recording_root_input = QLineEdit()
        self.recording_root_input.setText(
            str(self.input_data.get(model_consts.RECORDING_ROOT_CONFIG_KEY, "") or "")
        )
        self.recording_root_input.setPlaceholderText("audio_data/stored_data")
        self.select_recording_root_action = self.recording_root_input.addAction(
            self.style().standardIcon(QStyle.SP_DirIcon),
            QLineEdit.TrailingPosition,
        )
        self.select_recording_root_action.setToolTip("选择音频保存根目录")
        self.select_recording_root_action.triggered.connect(self._select_recording_root)
        self.recording_root_input.textChanged.connect(
            self._update_recording_root_tooltip
        )
        self._update_recording_root_tooltip(self.recording_root_input.text())
        self.default_recording_root_btn = QPushButton("默认路径")
        self.default_recording_root_btn.clicked.connect(self.recording_root_input.clear)
        recording_root_layout = QHBoxLayout()
        recording_root_layout.setContentsMargins(0, 0, 0, 0)
        recording_root_layout.setSpacing(8)
        recording_root_layout.addWidget(self.recording_root_input)
        recording_root_layout.addWidget(self.default_recording_root_btn)
        self.streaming_recording_checkbox.toggled.connect(self._on_streaming_recording_toggled)

        self._on_streaming_recording_toggled(self.streaming_recording_checkbox.isChecked())

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)
        grid_layout.addWidget(label_samplerate, 1, 0)
        grid_layout.addWidget(self.samplerate_combo, 1, 1)
        grid_layout.addWidget(label_input_device, 2, 0)
        grid_layout.addWidget(self.input_device_display, 2, 1)
        advanced_layout.addWidget(label_streaming_recording, 0, 0)
        advanced_layout.addWidget(self.streaming_recording_checkbox, 0, 1)
        advanced_layout.addWidget(self.preview_time_mode_label, 1, 0)
        advanced_layout.addWidget(self.preview_time_mode_combo, 1, 1)
        advanced_layout.addWidget(self.preview_time_mode_error_label, 2, 0, 1, 2)
        advanced_layout.addWidget(label_recording_root, 3, 0)
        advanced_layout.addLayout(recording_root_layout, 3, 1)
        self.ve_range_combo = QComboBox()
        self.ve_range_combo.setObjectName("ve_range_combo")
        for index, label in enumerate(VE_RANGE_LABELS):
            self.ve_range_combo.addItem(label, index)
        range_label = QLabel("输入量程:")
        if self._is_vk:
            try:
                range_index = validate_range_index(self.input_data.get(VE_RANGE_INDEX_CONFIG_KEY, 0))
            except ValueError as exc:
                self.ve_range_combo.setCurrentIndex(-1)
                self.ve_range_combo.setToolTip(str(exc))
            else:
                self.ve_range_combo.setCurrentIndex(self.ve_range_combo.findData(range_index))
        advanced_layout.addWidget(range_label, 4, 0)
        advanced_layout.addWidget(self.ve_range_combo, 4, 1)
        range_label.setVisible(self._is_vk)
        self.ve_range_combo.setVisible(self._is_vk)
        grid_layout.addWidget(self.recording_advanced_toggle, 3, 0, 1, 2)
        grid_layout.addWidget(self.recording_advanced_panel, 4, 0, 1, 2)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        try:
            if self._sample_rate_load_error:
                raise ValueError(self._sample_rate_load_error)
            text = self.samplerate_combo.currentText()
            if not text.isascii() or not text.isdecimal():
                raise ValueError("sample_rate 必须为整数，范围 8000–102400 Hz。")
            sample_rate = int(text)
            if self._is_vk:
                sample_rate = resolve_ve_recording_config({"sample_rate": sample_rate})["sample_rate"]
            elif text not in ("44100", "48000"):
                raise ValueError("当前声卡不支持此采样率，请重新选择。")
        except ValueError as exc:
            self.samplerate_combo.setFocus()
            QMessageBox.warning(self, "设置警告", str(exc))
            return
        if self._is_vk:
            try:
                range_index = validate_range_index(self.ve_range_combo.currentData())
                resolve_ve_recording_config({
                    "sample_rate": sample_rate, VE_RANGE_INDEX_CONFIG_KEY: range_index})
            except ValueError as exc:
                self._focus_advanced_field(self.ve_range_combo)
                QMessageBox.warning(self, "设置警告", str(exc))
                return
        try:
            preview_time_mode = validate_recording_preview_time_mode(
                self.preview_time_mode_combo.currentData()
            )
        except ValueError as exc:
            self.final_data = None
            self.preview_time_mode_error_label.setText(str(exc))
            self._preview_time_mode_needs_repair = True
            self._invalid_preview_time_mode = self.input_data.get(
                RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
            )
            self._refresh_preview_time_mode_visibility()
            self._focus_advanced_field(self.preview_time_mode_combo)
            QMessageBox.warning(self, "设置警告", str(exc))
            return
        recording_root = str(self.recording_root_input.text() or "").strip()
        if recording_root and not os.path.isdir(recording_root):
            self._focus_advanced_field(self.recording_root_input)
            QMessageBox.warning(self, "设置警告", "音频保存根目录不存在，请重新选择。")
            return
        self.final_data = deepcopy(self.input_data)
        for key in tuple(self.final_data):
            if key.startswith("monitor_"):
                del self.final_data[key]
        self.final_data.update({
            "total_time": self.time_input.value(),
            "sample_rate": sample_rate,
            "use_streaming_recording": bool(self.streaming_recording_checkbox.isChecked()),
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: preview_time_mode,
            model_consts.RECORDING_ROOT_CONFIG_KEY: (
                os.path.abspath(recording_root) if recording_root else ""
            ),
        })
        if self._is_vk:
            self.final_data[VE_RANGE_INDEX_CONFIG_KEY] = range_index
        self.accept()

    def _on_sample_rate_edited(self, _value):
        self._sample_rate_load_error = None
        self.samplerate_combo.setToolTip("")

    def _set_advanced_visible(self, visible):
        self.recording_advanced_panel.setVisible(visible)
        self.recording_advanced_toggle.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)
        self._resize_for_advanced_settings()

    def _resize_for_advanced_settings(self):
        self.recording_advanced_panel.parentWidget().layout().activate()
        self.main_layout.activate()
        self.resize(self.width(), self.sizeHint().expandedTo(self.minimumSize()).height())

    def _focus_advanced_field(self, field):
        self.recording_advanced_toggle.setChecked(True)
        field.setFocus()

    def _select_recording_root(self):
        current_root = str(self.recording_root_input.text() or "").strip()
        initial_root = (
            current_root
            if os.path.isdir(current_root)
            else model_consts.STORED_RECORDED_PATH
        )
        selected_root = QFileDialog.getExistingDirectory(
            self,
            "选择音频保存根目录",
            initial_root,
        )
        if selected_root:
            self.recording_root_input.setText(os.path.normpath(selected_root))

    def _update_recording_root_tooltip(self, recording_root):
        selected_root = str(recording_root or "").strip()
        effective_root = selected_root or model_consts.STORED_RECORDED_PATH
        self.recording_root_input.setToolTip(
            os.path.abspath(os.path.normpath(effective_root))
        )

    def _on_streaming_recording_toggled(self, checked: bool):
        self._refresh_preview_time_mode_visibility()

    def _on_preview_time_mode_changed(self, _index):
        try:
            validate_recording_preview_time_mode(
                self.preview_time_mode_combo.currentData()
            )
        except ValueError:
            return
        self._preview_time_mode_needs_repair = False
        self._invalid_preview_time_mode = None
        self.preview_time_mode_error_label.clear()
        self._refresh_preview_time_mode_visibility()

    def _refresh_preview_time_mode_visibility(self):
        if not hasattr(self, "preview_time_mode_combo"):
            return
        needs_repair = self._preview_time_mode_needs_repair
        has_effective_preview = self.streaming_recording_checkbox.isChecked()
        visible = needs_repair or has_effective_preview
        self.preview_time_mode_label.setVisible(visible)
        self.preview_time_mode_combo.setVisible(visible)
        self.preview_time_mode_error_label.setVisible(needs_repair)
        if self.recording_advanced_panel.isVisible():
            self._resize_for_advanced_settings()


class ImportAudioConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None):
        super().__init__(mic=mic)
        self.setWindowTitle("导入音频")
        self.input_data = input_data or {}
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = QGroupBox("导入音频设置")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_samplerate = QLabel("采样率")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        default_sr = self.input_data.get("sample_rate", 44100)
        self.samplerate_combo.setCurrentText(str(default_sr))

        grid_layout.addWidget(label_samplerate, 0, 0)
        grid_layout.addWidget(self.samplerate_combo, 0, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        self.final_data = {
            "sample_rate": int(self.samplerate_combo.currentText()),
        }
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow({"total_time": 4.0, "sample_rate": 44100})
    window.show()
    app.exec_()
