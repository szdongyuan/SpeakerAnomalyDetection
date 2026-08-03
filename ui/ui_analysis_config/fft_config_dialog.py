import re
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.acoustic_analysis.specific_consts.fft_consts import (
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


class FftConfigWindow(SemanticAnalysisConfigDialogBase):
    """FFT 频谱分析配置窗口。"""

    DEFAULT_CONFIG = {
        "analysis_channel": 0,
        "n_fft": 4096,
        "window": "hann",
        "overlap_ratio": 0.5,
        "weighting": "Z",
        "x_axis_scale": "log",
        "focus_range_enabled": True,
        "focus_min_hz": 100,
        "focus_max_hz": 20000,
        "baseline_file_path": "",
        "baseline_display_mode": "overlay",
        "baseline_smooth_third_octave": False,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [],
        "manual_lower_segments": [],
    }
    FFT_PRESETS = FFT_SIZE_PRESETS
    WINDOWS = ["hann", "hamming", "blackman", "boxcar"]
    X_AXIS_SCALES = ["linear", "log"]
    BASELINE_DISPLAY_MODES = {
        "overlay": "叠加显示",
        "delta": "差值显示",
    }

    def __init__(
        self,
        config_manager,
        model_type,
        available_channels: Optional[List[int]] = None,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type_str = "".join(
            re.findall(r"[A-Za-z]", str(model_type))
        ) or "FFT"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels

        full_config = self.config_manager.load_config()
        self.load_config = dict(self.DEFAULT_CONFIG)
        self.load_config.update(full_config.get(self.config_key, {}))
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
        fft_layout = QFormLayout(fft_group)
        fft_layout.setContentsMargins(8, 12, 8, 8)
        fft_layout.setHorizontalSpacing(12)
        fft_layout.setVerticalSpacing(8)
        fft_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fft_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        fft_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.fft_size_box = ComboBox()
        self.fft_size_box.setEditable(True)
        self.fft_size_box.lineEdit().setFont(self.fft_size_box.font())
        self.fft_size_box.addItems(
            [str(value) for value in self.FFT_PRESETS]
        )
        self.fft_size_box.setCurrentText(
            str(int(self.load_config.get("n_fft", 4096)))
        )
        fft_layout.addRow(Label("FFT 点数:"), self.fft_size_box)

        self.window_combo = ComboBox()
        self.window_combo.addItems(self.WINDOWS)
        self.window_combo.setCurrentText(
            str(self.load_config.get("window", "hann"))
        )
        fft_layout.addRow(Label("窗函数:"), self.window_combo)

        self.overlap_spin = DoubleSpinBox()
        self.overlap_spin.setRange(0, MAX_OVERLAP_RATIO * 100.0)
        self.overlap_spin.setDecimals(0)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(
            float(self.load_config.get("overlap_ratio", 0.5)) * 100.0
        )
        fft_layout.addRow(Label("重叠率:"), self.overlap_spin)
        compute_layout.addWidget(fft_group)

        self.weighting_selector = WeightingSelectorWidget(
            self.load_config,
            allowed_options=("Z", "A", "C"),
            default="Z",
            parent=self,
        )
        compute_layout.addWidget(self.weighting_selector)
        self.add_semantic_section("compute", widget=compute_widget)

        self.spectrum_display_widget = QWidget(self)
        display_layout = QVBoxLayout(self.spectrum_display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(12)

        self.spectrum_expand_button = QToolButton(self)
        self.spectrum_expand_button.setText("频谱显示设置")
        self.spectrum_expand_button.setCheckable(True)
        self.spectrum_expand_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.spectrum_expand_button.setCursor(Qt.PointingHandCursor)
        self.spectrum_expand_button.setStyleSheet(
            f"color: {ui_style_const.COLOR_TEXT}; "
            "font-size: 16px; font-weight: 600; border: none;"
        )
        display_layout.addWidget(self.spectrum_expand_button)

        self.spectrum_content_widget = QWidget(self)
        spectrum_layout = QVBoxLayout(self.spectrum_content_widget)
        spectrum_layout.setContentsMargins(16, 2, 0, 0)
        spectrum_layout.setSpacing(10)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(Label("横轴:"))
        self.x_axis_combo = ComboBox()
        self.x_axis_combo.addItems(self.X_AXIS_SCALES)
        self.x_axis_combo.setCurrentText(
            str(self.load_config.get("x_axis_scale", "log"))
        )
        axis_layout.addWidget(self.x_axis_combo, 1)
        spectrum_layout.addLayout(axis_layout)

        self.focus_checkbox = CheckBox("启用频率聚焦范围")
        self.focus_checkbox.setChecked(
            bool(self.load_config.get("focus_range_enabled", True))
        )
        self.focus_checkbox.stateChanged.connect(self._on_focus_changed)
        spectrum_layout.addWidget(self.focus_checkbox)

        self.focus_widget = QWidget(self)
        focus_layout = QHBoxLayout(self.focus_widget)
        focus_layout.setContentsMargins(0, 0, 0, 0)
        focus_layout.addWidget(Label("最低:"))
        self.focus_min_spin = SpinBox()
        self.focus_min_spin.setRange(1, 48000)
        self.focus_min_spin.setSuffix(" Hz")
        self.focus_min_spin.setValue(
            int(self.load_config.get("focus_min_hz", 100))
        )
        focus_layout.addWidget(self.focus_min_spin)
        focus_layout.addWidget(Label("最高:"))
        self.focus_max_spin = SpinBox()
        self.focus_max_spin.setRange(1, 96000)
        self.focus_max_spin.setSuffix(" Hz")
        self.focus_max_spin.setValue(
            int(self.load_config.get("focus_max_hz", 20000))
        )
        focus_layout.addWidget(self.focus_max_spin)
        spectrum_layout.addWidget(self.focus_widget)
        display_layout.addWidget(self.spectrum_content_widget)
        self._on_focus_changed(self.focus_checkbox.checkState())

        self.add_semantic_section(
            "display",
            widget=self.spectrum_display_widget,
        )
        self.spectrum_expand_button.toggled.connect(
            self._set_spectrum_display_expanded
        )
        self._set_spectrum_display_expanded(False)

        baseline_widget = QWidget(self)
        baseline_layout = QVBoxLayout(baseline_widget)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(12)

        file_layout = QHBoxLayout()
        file_layout.addWidget(Label("背景音频:"))
        self.baseline_path_edit = LineEdit()
        self.baseline_path_edit.setReadOnly(True)
        self.baseline_path_edit.setText(
            str(self.load_config.get("baseline_file_path", "") or "")
        )
        baseline_action = self.baseline_path_edit.addAction(
            QIcon(DEFAULT_DIR + "ui/ui_pic/folder/folder-s.png"),
            LineEdit.TrailingPosition,
        )
        baseline_action.setToolTip("选择背景噪声音频")
        baseline_action.triggered.connect(
            self._on_baseline_file_clicked
        )
        file_layout.addWidget(self.baseline_path_edit, 1)
        baseline_layout.addLayout(file_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(Label("显示方式:"))
        self.baseline_mode_combo = ComboBox()
        for value, label in self.BASELINE_DISPLAY_MODES.items():
            self.baseline_mode_combo.addItem(label, value)
        saved_mode = str(
            self.load_config.get("baseline_display_mode", "overlay")
        )
        mode_index = self.baseline_mode_combo.findData(saved_mode)
        self.baseline_mode_combo.setCurrentIndex(
            mode_index if mode_index >= 0 else 0
        )
        mode_layout.addWidget(self.baseline_mode_combo, 1)
        baseline_layout.addLayout(mode_layout)

        self.baseline_smooth_checkbox = CheckBox(
            "使用 1/3 倍频程平滑背景基线"
        )
        self.baseline_smooth_checkbox.setChecked(
            bool(
                self.load_config.get(
                    "baseline_smooth_third_octave",
                    False,
                )
            )
        )
        baseline_layout.addWidget(self.baseline_smooth_checkbox)
        self.add_semantic_section("reference", widget=baseline_widget)

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type="FFT",
            allow_manual_limits=True,
        )
        self.add_threshold_curve_sections(
            self.threshold_widget,
            self.load_config,
        )
        self.enable_plot_view_config(
            self.load_config,
            "Hz",
            self._plot_view_y_unit(),
            True,
            True,
            self.x_axis_combo.currentText() == "log",
        )

        self.x_axis_combo.currentIndexChanged.connect(
            self._sync_plot_view_x_constraint
        )
        self.weighting_selector.combo_box.currentIndexChanged.connect(
            self._sync_plot_view_y_unit
        )
        self.baseline_mode_combo.currentIndexChanged.connect(
            self._sync_plot_view_y_unit
        )
        self._sync_plot_view_y_unit()

    def _set_spectrum_display_expanded(self, expanded):
        expanded = bool(expanded)
        self.spectrum_expand_button.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.spectrum_content_widget.setVisible(expanded)
        self.spectrum_content_widget.updateGeometry()
        self.spectrum_display_widget.updateGeometry()
        self._on_display_config_expanded(expanded)

    def _sync_plot_view_x_constraint(self, _index=None):
        self.plot_view_config_widget.set_positive_x(
            self.x_axis_combo.currentText() == "log"
        )

    def _plot_view_y_unit(self):
        if self.baseline_mode_combo.currentData() == "delta":
            return "dB"
        weighting = self.weighting_selector.current_weighting()
        return f"dB({weighting}) SPL"

    def _sync_plot_view_y_unit(self, _index=None):
        y_unit = self._plot_view_y_unit()
        self.plot_view_config_widget.set_axis_units(None, y_unit)
        self.threshold_widget.limit_graph.setLabel("left", y_unit)

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
            "baseline_smooth_third_octave": (
                self.baseline_smooth_checkbox.isChecked()
            ),
            "analysis_channel": (
                self.channel_selector.current_channel()
                if self.show_channel_selector
                and hasattr(self, "channel_selector")
                else int(self.load_config.get("analysis_channel", 0) or 0)
            ),
        }
        config.update(self.weighting_selector.get_config())
        config.update(self.threshold_widget.get_config())
        return self.merge_plot_view_config(config)

    def _validate_config(self):
        config = self.get_default_config()
        if not (
            MIN_FFT_SIZE
            <= int(config["n_fft"])
            <= MAX_FFT_SIZE
        ):
            MessageBox.warning(
                self,
                "设置警告",
                f"FFT 点数必须在 {MIN_FFT_SIZE} ~ {MAX_FFT_SIZE} 范围内。",
            )
            return False
        if (
            config["focus_range_enabled"]
            and config["focus_max_hz"] <= config["focus_min_hz"]
        ):
            MessageBox.warning(
                self,
                "设置警告",
                "频率聚焦上限必须大于下限。",
            )
            return False
        if (
            config["baseline_display_mode"] == "delta"
            and not config["baseline_file_path"]
        ):
            MessageBox.warning(
                self,
                "设置警告",
                "差值显示需要先选择背景音频。",
            )
            return False
        if not self.validate_plot_view_config():
            return False
        return self.threshold_widget.validate()

    def on_default_btn_clicked(self):
        if not self._validate_config():
            return
        save_flag = self.config_manager.save_default_config(
            self.model_type_str,
            self.get_default_config(),
        )
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = dict(self.DEFAULT_CONFIG)
        self.load_config.update(
            self.config_manager.load_config().get(self.config_key, {})
        )
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_config():
            return
        self.accept()
        return self.get_default_config()
