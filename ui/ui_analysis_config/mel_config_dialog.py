from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QVBoxLayout

from base.core_algorithm.mel_spectrogram import default_mel_config
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
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


class MelConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.show_channel_selector = available_channels is not None
        self.available_channels = self._normalize_available_channels(available_channels)

        defaults = default_mel_config()
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
        self.setMinimumSize(520, 620)
        self.resize(580, 700)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        if self.show_channel_selector:
            layout.addLayout(self._create_channel_layout())
        layout.addWidget(self._create_tone_group())
        layout.addWidget(self._create_frequency_group())
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
        group = GroupBox("主音")
        layout = QGridLayout()

        self.main_tones_edit = PlainTextEdit()
        self.main_tones_edit.setMaximumHeight(70)
        self.main_tones_edit.setPlaceholderText("每行一个主音频率，例如:\n1200\n3500")
        self.main_tones_edit.setPlainText(self._format_float_lines(self.load_config.get("main_tones_hz", [])))

        self.main_tone_width_spin = self._double_spin(
            0,
            20000,
            1,
            self.load_config.get("main_tone_search_width_hz", 160.0),
            " Hz",
        )

        layout.addWidget(Label("主音频率(Hz):"), 0, 0)
        layout.addWidget(self.main_tones_edit, 0, 1, 1, 3)
        layout.addWidget(Label("主音搜索宽度:"), 1, 0)
        layout.addWidget(self.main_tone_width_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_frequency_group(self):
        group = GroupBox("频率")
        layout = QGridLayout()

        self.fmin_spin = self._double_spin(0, 96000, 1, self.load_config.get("fmin_hz", 100.0), " Hz")
        self.fmax_spin = self._double_spin(1, 96000, 1, self.load_config.get("fmax_hz", 20000.0), " Hz")

        layout.addWidget(Label("频率下限:"), 0, 0)
        layout.addWidget(self.fmin_spin, 0, 1)
        layout.addWidget(Label("频率上限:"), 0, 2)
        layout.addWidget(self.fmax_spin, 0, 3)

        group.setLayout(layout)
        return group

    def _create_algorithm_group(self):
        group = GroupBox("Mel 参数")
        layout = QGridLayout()

        self.window_combo = ComboBox()
        self.window_combo.addItems(["hamming", "hann", "blackman"])
        window = str(self.load_config.get("window", "hamming") or "hamming")
        idx = self.window_combo.findText(window)
        self.window_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.stft_nfft_combo = ComboBox()
        self.stft_nfft_combo.addItems([str(2**i) for i in range(8, 17)])
        nfft_text = str(int(self.load_config.get("stft_nfft", 4096) or 4096))
        idx = self.stft_nfft_combo.findText(nfft_text)
        self.stft_nfft_combo.setCurrentIndex(idx if idx >= 0 else self.stft_nfft_combo.findText("4096"))

        self.frame_length_spin = self._double_spin(1, 1000, 1, self.load_config.get("frame_length_ms", 30.0), " ms")
        self.frame_shift_spin = self._double_spin(1, 1000, 1, self.load_config.get("frame_shift_ms", 10.0), " ms")
        self.n_mels_spin = SpinBox()
        self.n_mels_spin.setRange(8, 512)
        self.n_mels_spin.setValue(int(self.load_config.get("n_mels", 128) or 128))

        layout.addWidget(Label("窗函数:"), 0, 0)
        layout.addWidget(self.window_combo, 0, 1)
        layout.addWidget(Label("STFT NFFT:"), 0, 2)
        layout.addWidget(self.stft_nfft_combo, 0, 3)
        layout.addWidget(Label("帧长:"), 1, 0)
        layout.addWidget(self.frame_length_spin, 1, 1)
        layout.addWidget(Label("步长:"), 1, 2)
        layout.addWidget(self.frame_shift_spin, 1, 3)
        layout.addWidget(Label("Mel 通道数:"), 2, 0)
        layout.addWidget(self.n_mels_spin, 2, 1)

        group.setLayout(layout)
        return group

    def _create_display_group(self):
        group = GroupBox("显示")
        layout = QGridLayout()

        self.colormap_box = ComboBox()
        self.colormap_box.addItems(["magma", "inferno", "plasma", "viridis"])
        color_map = str(self.load_config.get("color_map", "magma") or "magma")
        idx = self.colormap_box.findText(color_map)
        self.colormap_box.setCurrentIndex(idx if idx >= 0 else 0)

        core_range = self._range_values("core_range_hz", [2000.0, 5000.0])
        self.core_low_spin = self._double_spin(1, 96000, 1, core_range[0], " Hz")
        self.core_high_spin = self._double_spin(1, 96000, 1, core_range[1], " Hz")
        self.dynamic_range_spin = self._double_spin(
            10,
            120,
            1,
            self.load_config.get("dynamic_range_db", 65.0),
            " dB",
        )

        layout.addWidget(Label("配色:"), 0, 0)
        layout.addWidget(self.colormap_box, 0, 1)
        layout.addWidget(Label("动态范围:"), 0, 2)
        layout.addWidget(self.dynamic_range_spin, 0, 3)
        layout.addWidget(Label("核心频段下限:"), 1, 0)
        layout.addWidget(self.core_low_spin, 1, 1)
        layout.addWidget(Label("核心频段上限:"), 1, 2)
        layout.addWidget(self.core_high_spin, 1, 3)

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

    @staticmethod
    def _parse_float_lines(text):
        values = []
        for raw in (text or "").replace(";", "\n").replace(",", "\n").splitlines():
            item = raw.strip()
            if not item:
                continue
            values.append(float(item))
        return values

    def _range_values(self, key, default_pair):
        value = self.load_config.get(key, default_pair)
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        else:
            parts = list(value) if isinstance(value, (list, tuple)) else []
        if len(parts) < 2:
            return [float(default_pair[0]), float(default_pair[1])]
        return [float(parts[0]), float(parts[1])]

    def get_default_config(self):
        try:
            main_tones = self._parse_float_lines(self.main_tones_edit.toPlainText())
        except ValueError:
            MessageBox.warning(self, "设置警告", "主音频率列表只能包含数字。")
            return None
        if not main_tones:
            MessageBox.warning(self, "设置警告", "请至少输入一个主音频率。")
            return None
        if self.fmax_spin.value() <= self.fmin_spin.value():
            MessageBox.warning(self, "设置警告", "频率上限必须大于下限。")
            return None
        if self.frame_shift_spin.value() > self.frame_length_spin.value():
            MessageBox.warning(self, "设置警告", "步长不应大于帧长。")
            return None
        if self.core_high_spin.value() <= self.core_low_spin.value():
            MessageBox.warning(self, "设置警告", "核心频段上限必须大于下限。")
            return None

        return {
            "main_tones_hz": main_tones,
            "main_tone_search_width_hz": float(self.main_tone_width_spin.value()),
            "fmin_hz": float(self.fmin_spin.value()),
            "fmax_hz": float(self.fmax_spin.value()),
            "frame_length_ms": float(self.frame_length_spin.value()),
            "frame_shift_ms": float(self.frame_shift_spin.value()),
            "window": self.window_combo.currentText(),
            "stft_nfft": int(self.stft_nfft_combo.currentText()),
            "n_mels": int(self.n_mels_spin.value()),
            "color_map": self.colormap_box.currentText(),
            "dynamic_range_db": float(self.dynamic_range_spin.value()),
            "core_range_hz": [float(self.core_low_spin.value()), float(self.core_high_spin.value())],
            "mel_scale_range": self._range_values("mel_scale_range", [0.0, 8000.0]),
            "analysis_channel": int(self.channel_combo_box.currentData())
            if self.show_channel_selector and hasattr(self, "channel_combo_box")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        save_flag = self.config_manager.save_default_config("Mel", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        self.accept()
        return config_data
