from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QVBoxLayout

from base.core_algorithm.modulation_map import default_modulation_config
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    GroupBox,
    Label,
    MessageBox,
    PlainTextEdit,
    PushButton,
    SpinBox,
)
from ui.ui_src import ui_resources


class ModulationConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.show_channel_selector = available_channels is not None
        self.available_channels = self._normalize_available_channels(available_channels)

        defaults = default_modulation_config()
        saved = self.config_manager.load_config().get(model_type, {})
        self.load_config = defaults
        if isinstance(saved, dict):
            self.load_config.update(saved)
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

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("Modulation 分析配置")
        self.setMinimumSize(560, 720)
        self.resize(620, 760)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        if self.show_channel_selector:
            layout.addLayout(self._create_channel_layout())

        layout.addWidget(self._create_tone_group())
        layout.addWidget(self._create_range_group())
        layout.addWidget(self._create_algorithm_group())
        layout.addWidget(self._create_display_group())
        layout.addStretch()
        layout.addLayout(self.create_btn())
        self.setLayout(layout)

    def _create_channel_layout(self):
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(Label("通道:"))
        self.channel_combo_box = ComboBox()
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.load_config.get("analysis_channel", None)
        try:
            saved_channel = int(saved_channel)
        except (TypeError, ValueError):
            saved_channel = int(self.available_channels[0])
        if saved_channel not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(saved_channel)
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        channel_layout.addWidget(self.channel_combo_box, 1)
        return channel_layout

    def _create_tone_group(self):
        group = GroupBox("主音与机械参数")
        layout = QGridLayout()

        self.main_tones_edit = PlainTextEdit()
        self.main_tones_edit.setMaximumHeight(70)
        self.main_tones_edit.setPlaceholderText("每行一个主音频率，例如:\n1200\n3500")
        self.main_tones_edit.setPlainText(self._format_float_lines(self.load_config.get("main_tones_hz", [])))
        layout.addWidget(Label("主音频率(Hz):"), 0, 0)
        layout.addWidget(self.main_tones_edit, 0, 1, 1, 3)

        self.fan_rpm_spin = self._double_spin(0, 60000, 1, self.load_config.get("fan_rpm", 4500), " RPM")
        self.blade_count_spin = SpinBox()
        self.blade_count_spin.setRange(0, 100)
        self.blade_count_spin.setValue(int(self.load_config.get("blade_count", 2) or 0))
        self.threshold_spin = self._double_spin(
            0,
            200,
            2,
            self.load_config.get("threshold_percent", 10.0),
            " %",
        )

        layout.addWidget(Label("风扇转速:"), 1, 0)
        layout.addWidget(self.fan_rpm_spin, 1, 1)
        layout.addWidget(Label("叶片数:"), 1, 2)
        layout.addWidget(self.blade_count_spin, 1, 3)
        layout.addWidget(Label("阈值:"), 2, 0)
        layout.addWidget(self.threshold_spin, 2, 1)

        self.extra_mechanical_edit = PlainTextEdit()
        self.extra_mechanical_edit.setMaximumHeight(55)
        self.extra_mechanical_edit.setPlaceholderText("可选，每行一个额外机械调制频率")
        self.extra_mechanical_edit.setPlainText(
            self._format_float_lines(self.load_config.get("mechanical_freqs_hz", []))
        )
        layout.addWidget(Label("额外机械频率(Hz):"), 3, 0)
        layout.addWidget(self.extra_mechanical_edit, 3, 1, 1, 3)

        group.setLayout(layout)
        return group

    def _create_range_group(self):
        group = GroupBox("频率范围")
        layout = QGridLayout()
        signal_range = self._range_values("signal_freq_range_hz", [0.0, 10000.0])
        mod_range = self._range_values("mod_freq_range_hz", [0.0, 200.0])

        self.signal_min_spin = self._double_spin(0, 96000, 1, signal_range[0], " Hz")
        self.signal_max_spin = self._double_spin(1, 96000, 1, signal_range[1], " Hz")
        self.mod_min_spin = self._double_spin(0, 5000, 1, mod_range[0], " Hz")
        self.mod_max_spin = self._double_spin(1, 5000, 1, mod_range[1], " Hz")

        layout.addWidget(Label("信号频率下限:"), 0, 0)
        layout.addWidget(self.signal_min_spin, 0, 1)
        layout.addWidget(Label("信号频率上限:"), 0, 2)
        layout.addWidget(self.signal_max_spin, 0, 3)
        layout.addWidget(Label("调制频率下限:"), 1, 0)
        layout.addWidget(self.mod_min_spin, 1, 1)
        layout.addWidget(Label("调制频率上限:"), 1, 2)
        layout.addWidget(self.mod_max_spin, 1, 3)

        group.setLayout(layout)
        return group

    def _create_algorithm_group(self):
        group = GroupBox("算法参数")
        layout = QGridLayout()

        self.window_combo = ComboBox()
        self.window_combo.addItems(["hamming", "hann", "blackman"])
        window_type = str(self.load_config.get("window_type", "hamming") or "hamming")
        idx = self.window_combo.findText(window_type)
        self.window_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.stft_nfft_combo = ComboBox()
        self.stft_nfft_combo.addItems([str(2**i) for i in range(8, 16)])
        nfft_text = str(int(self.load_config.get("stft_nfft", 2048) or 2048))
        idx = self.stft_nfft_combo.findText(nfft_text)
        self.stft_nfft_combo.setCurrentIndex(idx if idx >= 0 else self.stft_nfft_combo.findText("2048"))

        self.frame_length_spin = self._double_spin(4, 1000, 1, self.load_config.get("frame_length_ms", 30.0), " ms")
        self.frame_shift_spin = self._double_spin(1, 1000, 1, self.load_config.get("frame_shift_ms", 10.0), " ms")
        self.envelope_window_spin = self._double_spin(
            1,
            200,
            1,
            self.load_config.get("envelope_window_ms", 10.0),
            " ms",
        )
        self.envelope_shift_spin = self._double_spin(
            0.1,
            200,
            2,
            self.load_config.get("envelope_shift_ms", 1.0),
            " ms",
        )

        layout.addWidget(Label("窗函数:"), 0, 0)
        layout.addWidget(self.window_combo, 0, 1)
        layout.addWidget(Label("STFT NFFT:"), 0, 2)
        layout.addWidget(self.stft_nfft_combo, 0, 3)
        layout.addWidget(Label("调制帧长:"), 1, 0)
        layout.addWidget(self.frame_length_spin, 1, 1)
        layout.addWidget(Label("调制步长:"), 1, 2)
        layout.addWidget(self.frame_shift_spin, 1, 3)
        layout.addWidget(Label("包络窗长:"), 2, 0)
        layout.addWidget(self.envelope_window_spin, 2, 1)
        layout.addWidget(Label("包络步长:"), 2, 2)
        layout.addWidget(self.envelope_shift_spin, 2, 3)

        group.setLayout(layout)
        return group

    def _create_display_group(self):
        group = GroupBox("显示与匹配")
        layout = QGridLayout()

        self.signal_step_spin = self._double_spin(
            0,
            1000,
            1,
            self.load_config.get("signal_freq_display_step_hz", 1.0),
            " Hz",
        )
        self.mod_bin_spin = self._double_spin(
            0,
            1000,
            1,
            self.load_config.get("mod_freq_bin_hz", 1.0),
            " Hz",
        )
        self.smoothing_spin = SpinBox()
        self.smoothing_spin.setRange(1, 99)
        self.smoothing_spin.setValue(int(self.load_config.get("smoothing_points", 3) or 1))
        self.main_tone_width_spin = self._double_spin(
            1,
            5000,
            1,
            self.load_config.get("main_tone_search_width_hz", 160.0),
            " Hz",
        )
        self.mechanical_tolerance_spin = self._double_spin(
            0,
            1000,
            1,
            self.load_config.get("mechanical_match_tolerance_hz", 20.0),
            " Hz",
        )
        self.rotation_harmonics_spin = SpinBox()
        self.rotation_harmonics_spin.setRange(0, 20)
        self.rotation_harmonics_spin.setValue(int(self.load_config.get("rotation_harmonics", 2) or 0))
        self.bpf_harmonics_spin = SpinBox()
        self.bpf_harmonics_spin.setRange(0, 20)
        self.bpf_harmonics_spin.setValue(int(self.load_config.get("bpf_harmonics", 1) or 0))
        self.show_hotspots_checkbox = CheckBox("显示全局热点")
        self.show_hotspots_checkbox.setChecked(bool(self.load_config.get("show_global_hotspots", True)))

        layout.addWidget(Label("信号轴步长:"), 0, 0)
        layout.addWidget(self.signal_step_spin, 0, 1)
        layout.addWidget(Label("调制轴分箱:"), 0, 2)
        layout.addWidget(self.mod_bin_spin, 0, 3)
        layout.addWidget(Label("平滑点数:"), 1, 0)
        layout.addWidget(self.smoothing_spin, 1, 1)
        layout.addWidget(Label("主音搜索宽度:"), 1, 2)
        layout.addWidget(self.main_tone_width_spin, 1, 3)
        layout.addWidget(Label("机械匹配容差:"), 2, 0)
        layout.addWidget(self.mechanical_tolerance_spin, 2, 1)
        layout.addWidget(Label("转频谐波数:"), 2, 2)
        layout.addWidget(self.rotation_harmonics_spin, 2, 3)
        layout.addWidget(Label("BPF谐波数:"), 3, 0)
        layout.addWidget(self.bpf_harmonics_spin, 3, 1)
        layout.addWidget(self.show_hotspots_checkbox, 3, 2, 1, 2)

        group.setLayout(layout)
        return group

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton(" 确 认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(default_btn)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    @staticmethod
    def _double_spin(min_value, max_value, decimals, value, suffix=""):
        spin = DoubleSpinBox()
        spin.setRange(float(min_value), float(max_value))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value or 0.0))
        spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _format_float_lines(values):
        if isinstance(values, str):
            return values
        if not isinstance(values, (list, tuple)):
            return ""
        return "\n".join(f"{float(v):g}" for v in values)

    def _range_values(self, key, default_pair):
        value = self.load_config.get(key, default_pair)
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        else:
            parts = list(value) if isinstance(value, (list, tuple)) else []
        if len(parts) < 2:
            return [float(default_pair[0]), float(default_pair[1])]
        return [float(parts[0]), float(parts[1])]

    @staticmethod
    def _parse_float_lines(text):
        values = []
        for raw in (text or "").replace(";", "\n").replace(",", "\n").splitlines():
            item = raw.strip()
            if not item:
                continue
            values.append(float(item))
        return values

    def get_default_config(self):
        try:
            main_tones = self._parse_float_lines(self.main_tones_edit.toPlainText())
            mechanical_freqs = self._parse_float_lines(self.extra_mechanical_edit.toPlainText())
        except ValueError:
            MessageBox.warning(self, "设置警告", "频率列表只能包含数字。")
            return None
        if not main_tones:
            MessageBox.warning(self, "设置警告", "请至少输入一个主音频率。")
            return None
        if self.signal_max_spin.value() <= self.signal_min_spin.value():
            MessageBox.warning(self, "设置警告", "信号频率上限必须大于下限。")
            return None
        if self.mod_max_spin.value() <= self.mod_min_spin.value():
            MessageBox.warning(self, "设置警告", "调制频率上限必须大于下限。")
            return None
        if self.frame_shift_spin.value() > self.frame_length_spin.value():
            MessageBox.warning(self, "设置警告", "调制步长不应大于调制帧长。")
            return None

        return {
            "main_tones_hz": main_tones,
            "fan_rpm": float(self.fan_rpm_spin.value()),
            "blade_count": int(self.blade_count_spin.value()),
            "threshold_percent": float(self.threshold_spin.value()),
            "signal_freq_range_hz": [float(self.signal_min_spin.value()), float(self.signal_max_spin.value())],
            "mod_freq_range_hz": [float(self.mod_min_spin.value()), float(self.mod_max_spin.value())],
            "window_type": self.window_combo.currentText(),
            "stft_nfft": int(self.stft_nfft_combo.currentText()),
            "frame_length_ms": float(self.frame_length_spin.value()),
            "frame_shift_ms": float(self.frame_shift_spin.value()),
            "envelope_window_ms": float(self.envelope_window_spin.value()),
            "envelope_shift_ms": float(self.envelope_shift_spin.value()),
            "signal_freq_display_step_hz": float(self.signal_step_spin.value()),
            "mod_freq_bin_hz": float(self.mod_bin_spin.value()),
            "smoothing_points": int(self.smoothing_spin.value()),
            "rotation_harmonics": int(self.rotation_harmonics_spin.value()),
            "bpf_harmonics": int(self.bpf_harmonics_spin.value()),
            "mechanical_freqs_hz": mechanical_freqs,
            "main_tone_search_width_hz": float(self.main_tone_width_spin.value()),
            "mechanical_match_tolerance_hz": float(self.mechanical_tolerance_spin.value()),
            "min_modulation_depth_percent": float(self.load_config.get("min_modulation_depth_percent", 1.0)),
            "show_global_hotspots": bool(self.show_hotspots_checkbox.isChecked()),
            "analysis_channel": int(self.channel_combo_box.currentData())
            if self.show_channel_selector and hasattr(self, "channel_combo_box")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        save_flag = self.config_manager.save_default_config("Modulation", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        self.accept()
        return config_data
