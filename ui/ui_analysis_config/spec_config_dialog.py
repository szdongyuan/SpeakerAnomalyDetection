"""Spectrogram analysis configuration dialog."""

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QVBoxLayout

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    ComboBox,
    GroupBox,
    Label,
    MessageBox,
    SpinBox,
)
from ui.ui_analysis_config.common_widgets import (
    AnalysisChannelSpinBoxWidget,
    SemanticAnalysisConfigDialogBase,
)


class SpecConfigWindow(SemanticAnalysisConfigDialogBase):
    """Spectrogram configuration window using the shared semantic layout."""

    FFT_SIZES = tuple(2**power for power in range(7, 14))
    HOP_LENGTHS = tuple(2**power for power in range(4, 13))
    WINDOW_FUNCTIONS = ("hann", "hamming", "blackman")
    COLOR_MAPS = ("viridis", "plasma", "magma", "inferno")
    FREQUENCY_SCALES = ("linear", "log")

    def __init__(
        self,
        config_manager,
        model_type,
        available_channels: Optional[List[int]] = None,
        restrict_analysis_channel: bool = False,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.load_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.restrict_analysis_channel = restrict_analysis_channel
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Spec 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = AnalysisChannelSpinBoxWidget(
                self.load_config,
                self.available_channels,
                self,
                restrict_to_available_channels=(
                    self.restrict_analysis_channel
                ),
            )
            self.add_semantic_section(
                "input",
                widget=self.channel_selector,
            )

        compute_group = GroupBox("频谱计算参数", self)
        compute_layout = QFormLayout(compute_group)
        compute_layout.setContentsMargins(8, 12, 8, 8)
        compute_layout.setHorizontalSpacing(12)
        compute_layout.setVerticalSpacing(8)
        compute_layout.setLabelAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        compute_layout.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )

        self.freq_scale_box = ComboBox(self)
        self.freq_scale_box.addItems(self.FREQUENCY_SCALES)
        self.freq_scale_box.setCurrentText(
            str(self.load_config.get("freq_scale_type", "linear"))
        )
        compute_layout.addRow(Label("频率轴类型:"), self.freq_scale_box)

        self.fft_size_box = ComboBox(self)
        self.fft_size_box.addItems(
            [str(value) for value in self.FFT_SIZES]
        )
        self.fft_size_box.setCurrentText(
            str(self.load_config.get("n_fft", 2048))
        )
        compute_layout.addRow(Label("FFT 窗长:"), self.fft_size_box)

        self.hop_length_box = ComboBox(self)
        self.hop_length_box.addItems(
            [str(value) for value in self.HOP_LENGTHS]
        )
        self.hop_length_box.setCurrentText(
            str(self.load_config.get("hop_length", 256))
        )
        compute_layout.addRow(Label("时间步长:"), self.hop_length_box)

        self.window_func_box = ComboBox(self)
        self.window_func_box.addItems(self.WINDOW_FUNCTIONS)
        self.window_func_box.setCurrentText(
            str(self.load_config.get("window_func", "hann"))
        )
        compute_layout.addRow(Label("窗函数:"), self.window_func_box)
        self.add_semantic_section("compute", widget=compute_group)

        display_group = GroupBox("频谱显示参数", self)
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(8, 12, 8, 8)
        display_layout.setSpacing(10)

        color_layout = QFormLayout()
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setLabelAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self.colormap_box = ComboBox(self)
        self.colormap_box.addItems(self.COLOR_MAPS)
        self.colormap_box.setCurrentText(
            str(self.load_config.get("color_map", "viridis"))
        )
        color_layout.addRow(Label("配色:"), self.colormap_box)
        display_layout.addLayout(color_layout)

        self.custom_limit_checkbox = CheckBox("自定义色阶范围", self)
        self.custom_limit_checkbox.setChecked(
            bool(self.load_config.get("custom_limit", False))
        )
        self.custom_limit_checkbox.stateChanged.connect(
            self.on_custom_limit_checkbox_changed
        )
        display_layout.addWidget(self.custom_limit_checkbox)

        self.limit_group_box = GroupBox("色阶范围", self)
        limit_layout = QFormLayout(self.limit_group_box)
        limit_layout.setContentsMargins(8, 12, 8, 8)
        limit_layout.setLabelAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self.top_limit_spinbox = SpinBox(self)
        self.top_limit_spinbox.setValue(
            int(self.load_config.get("top_limit", 70))
        )
        limit_layout.addRow(Label("上限:"), self.top_limit_spinbox)
        self.bottom_limit_spinbox = SpinBox(self)
        self.bottom_limit_spinbox.setValue(
            int(self.load_config.get("bottom_limit", 50))
        )
        limit_layout.addRow(Label("下限:"), self.bottom_limit_spinbox)
        display_layout.addWidget(self.limit_group_box)
        self.on_custom_limit_checkbox_changed(
            self.custom_limit_checkbox.checkState()
        )
        self.add_semantic_section("display", widget=display_group)

    def on_custom_limit_checkbox_changed(self, state):
        self.limit_group_box.setEnabled(state == Qt.Checked)

    def get_default_config(self):
        config = {
            "n_fft": int(self.fft_size_box.currentText()),
            "hop_length": int(self.hop_length_box.currentText()),
            "window_func": self.window_func_box.currentText(),
            "color_map": self.colormap_box.currentText(),
            "freq_scale_type": self.freq_scale_box.currentText(),
            "top_limit": self.top_limit_spinbox.value(),
            "bottom_limit": self.bottom_limit_spinbox.value(),
            "custom_limit": self.custom_limit_checkbox.isChecked(),
            "analysis_channel": (
                self.channel_selector.current_channel()
                if self.show_channel_selector
                else int(
                    self.load_config.get("analysis_channel", 0)
                    or 0
                )
            ),
        }
        if (
            config["custom_limit"]
            and config["top_limit"] <= config["bottom_limit"]
        ):
            MessageBox.warning(
                self,
                "设置警告",
                "上下限配置数据错误，请检查配置!",
            )
            return None
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        save_flag = self.config_manager.save_default_config(
            "Spec",
            config_data,
        )
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        self.accept()
        return config_data
