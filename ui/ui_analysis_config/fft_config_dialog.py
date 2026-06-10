from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

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
    PlainTextEdit,
    PushButton,
    SpinBox,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.ui_src import ui_resources


class FftConfigWindow(QDialog):
    """FFT 分析配置窗口。"""

    FFT_PRESETS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65535]
    WINDOWS = ["hann", "hamming", "blackman", "boxcar"]
    WEIGHTINGS = ["Z", "A", "C"]
    X_AXIS_SCALES = ["linear", "log"]
    BASELINE_DISPLAY_MODES = {
        "overlay": "叠加显示",
        "delta": "差值显示",
    }

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.show_channel_selector = available_channels is not None
        self.available_channels = self._normalize_available_channels(available_channels)
        self.init_ui()

    @staticmethod
    def _normalize_available_channels(available_channels):
        channels = []
        try:
            channels = sorted({int(ch) for ch in (available_channels or [])})
        except Exception:
            channels = []
        if not channels:
            channels = [0]
        return channels

    def _create_channel_layout(self):
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(Label("通道:"))
        self.channel_combo_box = ComboBox()
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.load_config.get("analysis_channel", None)
        if saved_channel is None or int(saved_channel) not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(int(saved_channel))
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        channel_layout.addWidget(self.channel_combo_box, 1)
        return channel_layout

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("FFT 分析配置")
        self.setMinimumSize(460, 720)
        self.resize(460, 760)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(12)
        content_layout.setContentsMargins(0, 0, 0, 0)
        if self.show_channel_selector:
            content_layout.addLayout(self._create_channel_layout())

        param_group = GroupBox("FFT 参数")
        param_group.setLayout(self._create_fft_param_layout())
        content_layout.addWidget(param_group)

        baseline_group = GroupBox("背景噪声基线")
        baseline_group.setLayout(self._create_baseline_layout())
        content_layout.addWidget(baseline_group)

        dominant_group = GroupBox("主音识别")
        dominant_group.setLayout(self._create_dominant_tone_layout())
        content_layout.addWidget(dominant_group)

        self.threshold_widget = ThresholdConfigWidget(parent=self, load_config=self.load_config, model_type="FFT")
        self.threshold_widget.setMaximumHeight(320)
        content_layout.addWidget(self.threshold_widget)
        content_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setObjectName("fft_config_scroll_area")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addLayout(self.create_btn())
        self.setLayout(main_layout)

    def _create_fft_param_layout(self):
        layout = QVBoxLayout()

        fft_layout = QHBoxLayout()
        fft_layout.addWidget(Label("FFT 点数:"))
        self.fft_size_box = ComboBox()
        self.fft_size_box.setEditable(True)
        self.fft_size_box.addItems([str(v) for v in self.FFT_PRESETS])
        self.fft_size_box.setCurrentText(str(int(self.load_config.get("n_fft", 4096))))
        fft_layout.addWidget(self.fft_size_box, 1)
        layout.addLayout(fft_layout)

        window_layout = QHBoxLayout()
        window_layout.addWidget(Label("窗函数:"))
        self.window_combo = ComboBox()
        self.window_combo.addItems(self.WINDOWS)
        self.window_combo.setCurrentText(str(self.load_config.get("window", "hann")))
        window_layout.addWidget(self.window_combo, 1)
        layout.addLayout(window_layout)

        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(Label("重叠率:"))
        self.overlap_spin = DoubleSpinBox()
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setDecimals(0)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(float(self.load_config.get("overlap_ratio", 0.5)) * 100.0)
        overlap_layout.addWidget(self.overlap_spin, 1)
        layout.addLayout(overlap_layout)

        weighting_layout = QHBoxLayout()
        weighting_layout.addWidget(Label("计权方式:"))
        self.weighting_combo = ComboBox()
        self.weighting_combo.addItems(self.WEIGHTINGS)
        self.weighting_combo.setCurrentText(str(self.load_config.get("weighting", "Z")))
        weighting_layout.addWidget(self.weighting_combo, 1)
        layout.addLayout(weighting_layout)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(Label("横轴:"))
        self.x_axis_combo = ComboBox()
        self.x_axis_combo.addItems(self.X_AXIS_SCALES)
        self.x_axis_combo.setCurrentText(str(self.load_config.get("x_axis_scale", "log")))
        axis_layout.addWidget(self.x_axis_combo, 1)
        layout.addLayout(axis_layout)

        self.focus_checkbox = CheckBox("启用频率聚焦范围")
        self.focus_checkbox.setChecked(bool(self.load_config.get("focus_range_enabled", True)))
        self.focus_checkbox.stateChanged.connect(self._on_focus_changed)
        layout.addWidget(self.focus_checkbox)

        self.focus_widget = QWidget()
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
        layout.addWidget(self.focus_widget)
        self._on_focus_changed(self.focus_checkbox.checkState())

        return layout

    def _create_baseline_layout(self):
        layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        file_layout.addWidget(Label("背景音频:"))
        self.baseline_path_edit = LineEdit()
        self.baseline_path_edit.setReadOnly(True)
        self.baseline_path_edit.setText(str(self.load_config.get("baseline_file_path", "") or ""))
        icon = QIcon(":/ui/icon/folder-s.png")
        action = self.baseline_path_edit.addAction(icon, LineEdit.TrailingPosition)
        action.setToolTip("选择背景噪声音频")
        action.triggered.connect(self._on_baseline_file_clicked)
        file_layout.addWidget(self.baseline_path_edit, 1)
        layout.addLayout(file_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(Label("显示方式:"))
        self.baseline_mode_combo = ComboBox()
        for value, label in self.BASELINE_DISPLAY_MODES.items():
            self.baseline_mode_combo.addItem(label, value)
        saved_mode = str(self.load_config.get("baseline_display_mode", "overlay"))
        idx = self.baseline_mode_combo.findData(saved_mode)
        self.baseline_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        mode_layout.addWidget(self.baseline_mode_combo, 1)
        layout.addLayout(mode_layout)

        self.baseline_smooth_checkbox = CheckBox("使用 1/3 倍频程做平滑")
        self.baseline_smooth_checkbox.setChecked(bool(self.load_config.get("baseline_smooth_third_octave", False)))
        layout.addWidget(self.baseline_smooth_checkbox)
        return layout

    def _create_dominant_tone_layout(self):
        layout = QVBoxLayout()

        self.dominant_tone_checkbox = CheckBox("启用主音识别")
        self.dominant_tone_checkbox.setChecked(bool(self.load_config.get("dominant_tone_enabled", False)))
        layout.addWidget(self.dominant_tone_checkbox)

        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(Label("最小 prominence:"))
        self.dominant_prominence_spin = DoubleSpinBox()
        self.dominant_prominence_spin.setRange(0, 100)
        self.dominant_prominence_spin.setDecimals(1)
        self.dominant_prominence_spin.setSuffix(" dB")
        self.dominant_prominence_spin.setValue(float(self.load_config.get("dominant_tone_min_prominence_db", 3.0)))
        prominence_layout.addWidget(self.dominant_prominence_spin, 1)
        layout.addLayout(prominence_layout)

        self.dominant_use_display_curve_checkbox = CheckBox("使用当前显示曲线识别")
        self.dominant_use_display_curve_checkbox.setChecked(
            bool(self.load_config.get("dominant_tone_use_display_curve", True))
        )
        layout.addWidget(self.dominant_use_display_curve_checkbox)

        layout.addWidget(Label("频率区间:"))
        self.dominant_intervals_edit = PlainTextEdit()
        self.dominant_intervals_edit.setPlaceholderText("示例:\n100, 500, Low\n500, 2000, Mid")
        self.dominant_intervals_edit.setPlainText(str(self.load_config.get("dominant_tone_intervals_text", "") or ""))
        self.dominant_intervals_edit.setMaximumHeight(80)
        layout.addWidget(self.dominant_intervals_edit)
        return layout

    def _on_focus_changed(self, state):
        enabled = state == Qt.Checked
        if hasattr(self, "focus_widget"):
            self.focus_widget.setEnabled(enabled)

    def _on_baseline_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景噪声音频",
            DEFAULT_DIR,
            filter="音频文件 (*.wav *.flac *.mp3);;所有文件 (*.*)",
        )
        if file_path:
            self.baseline_path_edit.setText(file_path)

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        try:
            n_fft = int(self.fft_size_box.currentText())
        except Exception:
            n_fft = 0
        config = {
            "n_fft": n_fft,
            "window": self.window_combo.currentText(),
            "overlap_ratio": float(self.overlap_spin.value()) / 100.0,
            "weighting": self.weighting_combo.currentText(),
            "x_axis_scale": self.x_axis_combo.currentText(),
            "focus_range_enabled": self.focus_checkbox.isChecked(),
            "focus_min_hz": int(self.focus_min_spin.value()),
            "focus_max_hz": int(self.focus_max_spin.value()),
            "baseline_file_path": self.baseline_path_edit.text().strip(),
            "baseline_display_mode": self.baseline_mode_combo.currentData(),
            "baseline_smooth_third_octave": self.baseline_smooth_checkbox.isChecked(),
            "dominant_tone_enabled": self.dominant_tone_checkbox.isChecked(),
            "dominant_tone_intervals_text": self.dominant_intervals_edit.toPlainText(),
            "dominant_tone_min_prominence_db": float(self.dominant_prominence_spin.value()),
            "dominant_tone_use_display_curve": self.dominant_use_display_curve_checkbox.isChecked(),
            "analysis_channel": int(self.channel_combo_box.currentData())
            if self.show_channel_selector and hasattr(self, "channel_combo_box")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }
        config.update(self.threshold_widget.get_config())
        return config

    def _validate_config(self):
        if not self.threshold_widget.validate():
            return False
        config = self.get_default_config()
        if not (512 <= int(config["n_fft"]) <= 65535):
            MessageBox.warning(self, "设置警告", "FFT 点数必须在 512 ~ 65535 范围内。")
            return False
        if config["focus_range_enabled"] and config["focus_max_hz"] <= config["focus_min_hz"]:
            MessageBox.warning(self, "设置警告", "频率聚焦上限必须大于下限。")
            return False
        return True

    def on_default_btn_clicked(self):
        if not self._validate_config():
            return
        save_flag = self.config_manager.save_default_config("FFT", self.get_default_config())
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self._validate_config():
            return
        self.accept()
        return self.get_default_config()
