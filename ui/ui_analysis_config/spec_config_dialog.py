from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, Label, PushButton, ComboBox, SpinBox, MessageBox
from ui.ui_src import ui_resources


class SpecConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
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
        channel_layout.addWidget(self.channel_combo_box)
        channel_layout.addStretch()
        return channel_layout

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 420)
        layout = QVBoxLayout()
        if self.show_channel_selector:
            layout.addLayout(self._create_channel_layout())
        spec_param_group_box = GroupBox("频谱图参数配置")
        param_layout = self.create_spec_param()
        spec_param_group_box.setLayout(param_layout)
        btn_layout = self.create_btn()

        layout.addWidget(spec_param_group_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def create_spec_param(self):
        freq_scale_label = Label("频率轴类型")
        self.freq_scale_box = ComboBox()
        self.freq_scale_box.addItems(["linear", "log"])
        freq_scale_type = self.load_config.get("freq_scale_type", "linear")
        self.freq_scale_box.setCurrentText(freq_scale_type)
        freq_scale_layout = QHBoxLayout()
        freq_scale_layout.addWidget(freq_scale_label)
        freq_scale_layout.addWidget(self.freq_scale_box)

        fft_size_label = Label("FFT 窗长")
        self.fft_size_box = ComboBox()
        fft_sizes = [str(2**i) for i in range(7, 14)]
        self.fft_size_box.addItems(fft_sizes)
        fft_size = str(self.load_config.get("n_fft", 2048))
        self.fft_size_box.setCurrentText(fft_size)
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(fft_size_label)
        fft_layout.addWidget(self.fft_size_box)

        hop_length_label = Label("时间步长")
        self.hop_length_box = ComboBox()
        hop_lengths = [str(2**i) for i in range(4, 13)]
        self.hop_length_box.addItems(hop_lengths)
        hop_length = str(self.load_config.get("hop_length", 256))
        self.hop_length_box.setCurrentText(hop_length)
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(hop_length_label)
        hop_layout.addWidget(self.hop_length_box)

        window_func_label = Label("窗函数")
        self.window_func_box = ComboBox()
        self.window_func_box.addItems(["hann", "hamming", "blackman"])
        window_func = self.load_config.get("window_func", "hann")
        self.window_func_box.setCurrentText(window_func)
        window_layout = QHBoxLayout()
        window_layout.addWidget(window_func_label)
        window_layout.addWidget(self.window_func_box)

        colormap_label = Label("配色")
        self.colormap_box = ComboBox()
        self.colormap_box.addItems(["viridis", "plasma", "magma", "inferno"])
        color_map = self.load_config.get("color_map", "viridis")
        self.colormap_box.setCurrentText(color_map)
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_box)

        top_limit_label = Label("上限")
        top_limit_label.setObjectName("toplimitlabel")
        self.top_limit_spinbox = SpinBox()
        top_limit = self.load_config.get("top_limit", 70)
        self.top_limit_spinbox.setValue(top_limit)
        top_limit_layout = QHBoxLayout()
        top_limit_layout.addWidget(top_limit_label)
        top_limit_layout.addWidget(self.top_limit_spinbox)

        bottom_limit_label = Label("下限")
        bottom_limit_label.setObjectName("bottomlimitlabel")
        self.bottom_limit_spinbox = SpinBox()
        bottom_limit = self.load_config.get("bottom_limit", 50)
        self.bottom_limit_spinbox.setValue(bottom_limit)
        bottom_limit_layout = QHBoxLayout()
        bottom_limit_layout.addWidget(bottom_limit_label)
        bottom_limit_layout.addWidget(self.bottom_limit_spinbox)

        layout = QVBoxLayout()
        layout.addLayout(top_limit_layout)
        layout.addStretch()
        layout.addLayout(bottom_limit_layout)
        layout.addStretch()

        self.limit_group_box = GroupBox()
        self.limit_group_box.setLayout(layout)

        self.custom_limit_checkbox = CheckBox("自定义")
        self.on_custom_limit_checkbox_changed(self.custom_limit_checkbox.isChecked())
        self.custom_limit_checkbox.stateChanged.connect(self.on_custom_limit_checkbox_changed)
        self.custom_limit_checkbox.setChecked(self.load_config.get("custom_limit", False))

        param_layout = QVBoxLayout()
        param_layout.addStretch()
        param_layout.addLayout(freq_scale_layout)
        param_layout.addStretch()
        param_layout.addLayout(fft_layout)
        param_layout.addStretch()
        param_layout.addLayout(hop_layout)
        param_layout.addStretch()
        param_layout.addLayout(window_layout)
        param_layout.addStretch()
        param_layout.addLayout(colormap_layout)
        param_layout.addStretch()
        param_layout.addWidget(self.custom_limit_checkbox)
        param_layout.addStretch()
        param_layout.addWidget(self.limit_group_box)
        param_layout.addSpacing(10)
        param_layout.setSpacing(10)
        return param_layout

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

    def on_custom_limit_checkbox_changed(self, state):
        if state == Qt.Checked:
            self.limit_group_box.setEnabled(True)
        else:
            self.limit_group_box.setEnabled(False)

    def get_default_config(self):
        default_config = {
            "n_fft": int(self.fft_size_box.currentText()),
            "hop_length": int(self.hop_length_box.currentText()),
            "window_func": self.window_func_box.currentText(),
            "color_map": self.colormap_box.currentText(),
            "freq_scale_type": self.freq_scale_box.currentText(),
            "top_limit": self.top_limit_spinbox.value(),
            "bottom_limit": self.bottom_limit_spinbox.value(),
            "custom_limit": self.custom_limit_checkbox.isChecked(),
            "analysis_channel": int(self.channel_combo_box.currentData())
            if self.show_channel_selector and hasattr(self, "channel_combo_box")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }
        if default_config["custom_limit"] and default_config["top_limit"] <= default_config["bottom_limit"]:
            MessageBox.warning(self, "设置警告", "上下限配置数据错误，请检查配置!")
            return
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        save_flag = self.config_manager.save_default_config("Spec", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        self.accept()
        return config_data
