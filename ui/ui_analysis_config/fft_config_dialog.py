import re
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget

from base.core_algorithm.response.fft_consts import (
    FFT_SIZE_PRESETS,
    MAX_FFT_SIZE,
    MAX_OVERLAP_RATIO,
    MIN_FFT_SIZE,
)
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    GroupBox,
    Label,
    LineEdit,
    MessageBox,
    SpinBox,
)
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.ui_src import ui_resources


class FftConfigWindow(SemanticAnalysisConfigDialogBase):
    """FFT 分析配置窗口，不包含主频识别功能。"""

    FFT_PRESETS = FFT_SIZE_PRESETS
    WINDOWS = ["hann", "hamming", "blackman", "boxcar"]
    X_AXIS_SCALES = ["linear", "log"]
    BASELINE_DISPLAY_MODES = {
        "overlay": "叠加显示",
        "delta": "差值显示",
    }

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type_str = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "FFT"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FFT 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(
                self.load_config,
                self.available_channels,
                self,
            )
            self.add_semantic_section("input", widget=self.channel_selector)

        compute_widget = QWidget(self)
        compute_layout = QVBoxLayout(compute_widget)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(12)

        fft_group = GroupBox("FFT 参数")
        fft_layout = QVBoxLayout(fft_group)

        fft_size_layout = QHBoxLayout()
        fft_size_layout.addWidget(Label("FFT 点数:"))
        self.fft_size_box = ComboBox()
        self.fft_size_box.setEditable(True)
        self.fft_size_box.lineEdit().setFont(self.fft_size_box.font())
        self.fft_size_box.addItems([str(value) for value in self.FFT_PRESETS])
        self.fft_size_box.setCurrentText(str(int(self.load_config.get("n_fft", 4096))))
        fft_size_layout.addWidget(self.fft_size_box, 1)
        fft_layout.addLayout(fft_size_layout)

        window_layout = QHBoxLayout()
        window_layout.addWidget(Label("窗函数:"))
        self.window_combo = ComboBox()
        self.window_combo.addItems(self.WINDOWS)
        self.window_combo.setCurrentText(str(self.load_config.get("window", "hann")))
        window_layout.addWidget(self.window_combo, 1)
        fft_layout.addLayout(window_layout)

        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(Label("重叠率:"))
        self.overlap_spin = DoubleSpinBox()
        self.overlap_spin.setRange(0, MAX_OVERLAP_RATIO * 100.0)
        self.overlap_spin.setDecimals(0)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(float(self.load_config.get("overlap_ratio", 0.5)) * 100.0)
        overlap_layout.addWidget(self.overlap_spin, 1)
        fft_layout.addLayout(overlap_layout)
        compute_layout.addWidget(fft_group)

        self.weighting_selector = WeightingSelectorWidget(
            self.load_config,
            allowed_options=("Z", "A", "C"),
            default="Z",
            parent=self,
        )
        compute_layout.addWidget(self.weighting_selector)
        self.add_semantic_section("compute", widget=compute_widget)

        display_widget = QWidget(self)
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(12)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(Label("横轴:"))
        self.x_axis_combo = ComboBox()
        self.x_axis_combo.addItems(self.X_AXIS_SCALES)
        self.x_axis_combo.setCurrentText(str(self.load_config.get("x_axis_scale", "log")))
        axis_layout.addWidget(self.x_axis_combo, 1)
        display_layout.addLayout(axis_layout)

        self.focus_checkbox = CheckBox("启用频率聚焦范围")
        self.focus_checkbox.setChecked(bool(self.load_config.get("focus_range_enabled", True)))
        self.focus_checkbox.stateChanged.connect(self._on_focus_changed)
        display_layout.addWidget(self.focus_checkbox)

        self.focus_widget = QWidget(self)
        focus_layout = QHBoxLayout(self.focus_widget)
        focus_layout.setContentsMargins(0, 0, 0, 0)
        focus_layout.addWidget(Label("最低:"))
        self.focus_min_spin = SpinBox()
        self.focus_min_spin.setRange(1, 48000)
        self.focus_min_spin.setSuffix(" Hz")
        self.focus_min_spin.setValue(int(self.load_config.get("focus_min_hz", 100)))
        focus_layout.addWidget(self.focus_min_spin)
        focus_layout.addWidget(Label("最高:"))
        self.focus_max_spin = SpinBox()
        self.focus_max_spin.setRange(1, 96000)
        self.focus_max_spin.setSuffix(" Hz")
        self.focus_max_spin.setValue(int(self.load_config.get("focus_max_hz", 20000)))
        focus_layout.addWidget(self.focus_max_spin)
        display_layout.addWidget(self.focus_widget)
        self._on_focus_changed(self.focus_checkbox.checkState())
        self.add_semantic_section("display", widget=display_widget)

        baseline_widget = QWidget(self)
        baseline_layout = QVBoxLayout(baseline_widget)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(12)

        file_layout = QHBoxLayout()
        file_layout.addWidget(Label("背景音频:"))
        self.baseline_path_edit = LineEdit()
        self.baseline_path_edit.setReadOnly(True)
        self.baseline_path_edit.setText(str(self.load_config.get("baseline_file_path", "") or ""))
        action = self.baseline_path_edit.addAction(
            QIcon(":/ui/icon/folder-s.png"),
            LineEdit.TrailingPosition,
        )
        action.setToolTip("选择背景噪声音频")
        action.triggered.connect(self._on_baseline_file_clicked)
        file_layout.addWidget(self.baseline_path_edit, 1)
        baseline_layout.addLayout(file_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(Label("显示方式:"))
        self.baseline_mode_combo = ComboBox()
        for value, label in self.BASELINE_DISPLAY_MODES.items():
            self.baseline_mode_combo.addItem(label, value)
        saved_mode = str(self.load_config.get("baseline_display_mode", "overlay"))
        mode_index = self.baseline_mode_combo.findData(saved_mode)
        self.baseline_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        mode_layout.addWidget(self.baseline_mode_combo, 1)
        baseline_layout.addLayout(mode_layout)

        self.baseline_smooth_checkbox = CheckBox("使用 1/3 倍频程平滑背景基线")
        self.baseline_smooth_checkbox.setChecked(
            bool(self.load_config.get("baseline_smooth_third_octave", False))
        )
        baseline_layout.addWidget(self.baseline_smooth_checkbox)
        self.add_semantic_section("reference", widget=baseline_widget)

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type="FFT",
        )
        self.add_semantic_section("judgment", widget=self.threshold_widget)

    def _on_focus_changed(self, state):
        if hasattr(self, "focus_widget"):
            self.focus_widget.setEnabled(state == Qt.Checked)

    def _on_baseline_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景噪声音频",
            DEFAULT_DIR,
            filter="音频文件 (*.wav *.flac *.mp3);;所有文件 (*.*)",
        )
        if file_path:
            self.baseline_path_edit.setText(file_path)

    def get_default_config(self):
        try:
            n_fft = int(self.fft_size_box.currentText())
        except (TypeError, ValueError):
            n_fft = 0
        config = {
            "n_fft": n_fft,
            "window": self.window_combo.currentText(),
            "overlap_ratio": float(self.overlap_spin.value()) / 100.0,
            "x_axis_scale": self.x_axis_combo.currentText(),
            "focus_range_enabled": self.focus_checkbox.isChecked(),
            "focus_min_hz": int(self.focus_min_spin.value()),
            "focus_max_hz": int(self.focus_max_spin.value()),
            "baseline_file_path": self.baseline_path_edit.text().strip(),
            "baseline_display_mode": self.baseline_mode_combo.currentData(),
            "baseline_smooth_third_octave": self.baseline_smooth_checkbox.isChecked(),
            "analysis_channel": self.channel_selector.current_channel()
            if self.show_channel_selector and hasattr(self, "channel_selector")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }
        config.update(self.weighting_selector.get_config())
        config.update(self.threshold_widget.get_config())
        return config

    def _validate_config(self):
        if not self.threshold_widget.validate():
            return False
        config = self.get_default_config()
        if not (MIN_FFT_SIZE <= int(config["n_fft"]) <= MAX_FFT_SIZE):
            MessageBox.warning(
                self,
                "设置警告",
                f"FFT 点数必须在 {MIN_FFT_SIZE} ~ {MAX_FFT_SIZE} 范围内。",
            )
            return False
        if config["focus_range_enabled"] and config["focus_max_hz"] <= config["focus_min_hz"]:
            MessageBox.warning(self, "设置警告", "频率聚焦上限必须大于下限。")
            return False
        return True

    def on_default_btn_clicked(self):
        if not self._validate_config():
            return
        save_flag = self.config_manager.save_default_config(
            self.model_type_str,
            self.get_default_config(),
        )
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_config():
            return
        self.accept()
        return self.get_default_config()
