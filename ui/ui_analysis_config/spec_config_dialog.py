from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from consts.acoustic_analysis.specific_consts import spec_consts
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, Label, ComboBox, SpinBox, MessageBox
from ui.ui_analysis_config.common_widgets import ChannelSelectorWidget, SemanticAnalysisConfigDialogBase


class SpecConfigWindow(SemanticAnalysisConfigDialogBase):
    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.init_ui()

    def init_ui(self):
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(self.load_config, self.available_channels, self)
            self.add_semantic_section("input", widget=self.channel_selector)
        compute_group_box = GroupBox("频谱计算参数")
        compute_group_box.setLayout(self.create_compute_param())
        display_group_box = GroupBox("频谱显示参数")
        display_group_box.setLayout(self.create_display_param())
        self.add_semantic_section("compute", widget=compute_group_box)
        self.add_semantic_section("display", widget=display_group_box)

    def create_compute_param(self):
        param_layout = QVBoxLayout()
        spectrum_mode_label = Label("频谱模式")
        self.freq_scale_box = ComboBox()
        for mode, label, tooltip in spec_consts.SPEC_SPECTRUM_MODES:
            self.freq_scale_box.addItem(label, mode)
            self.freq_scale_box.setItemData(self.freq_scale_box.count() - 1, tooltip, Qt.ToolTipRole)
        freq_scale_type = self.load_config.get("freq_scale_type", spec_consts.DEFAULT_SPEC_MODE)
        mode_index = self.freq_scale_box.findData(freq_scale_type)
        self.freq_scale_box.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        spectrum_mode_layout = QHBoxLayout()
        spectrum_mode_layout.addWidget(spectrum_mode_label)
        spectrum_mode_layout.addWidget(self.freq_scale_box)
        param_layout.addLayout(spectrum_mode_layout)
        param_layout.addStretch()

        fft_size_label = Label("FFT 窗长")
        self.fft_size_box = ComboBox()
        self.fft_size_box.addItems([str(value) for value in spec_consts.SPEC_FFT_SIZE_PRESETS])
        fft_size = str(self.load_config.get("n_fft", spec_consts.DEFAULT_SPEC_N_FFT))
        self.fft_size_box.setCurrentText(fft_size)
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(fft_size_label)
        fft_layout.addWidget(self.fft_size_box)

        hop_length_label = Label("时间步长")
        self.hop_length_box = ComboBox()
        self.hop_length_box.addItems([str(value) for value in spec_consts.SPEC_HOP_LENGTH_PRESETS])
        hop_length = str(self.load_config.get("hop_length", spec_consts.DEFAULT_SPEC_HOP_LENGTH))
        self.hop_length_box.setCurrentText(hop_length)
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(hop_length_label)
        hop_layout.addWidget(self.hop_length_box)

        window_func_label = Label("窗函数")
        self.window_func_box = ComboBox()
        self.window_func_box.addItems(spec_consts.SPEC_WINDOW_OPTIONS)
        window_func = self.load_config.get("window_func", spec_consts.DEFAULT_SPEC_WINDOW)
        self.window_func_box.setCurrentText(window_func)
        window_layout = QHBoxLayout()
        window_layout.addWidget(window_func_label)
        window_layout.addWidget(self.window_func_box)

        param_layout.addLayout(fft_layout)
        param_layout.addStretch()
        param_layout.addLayout(hop_layout)
        param_layout.addStretch()
        param_layout.addLayout(window_layout)
        self.mel_param_group = self._create_mel_param_group()
        param_layout.addWidget(self.mel_param_group)
        self.freq_scale_box.currentIndexChanged.connect(self._on_spectrum_mode_changed)
        self._on_spectrum_mode_changed()
        param_layout.setSpacing(10)
        return param_layout

    def _selected_spectrum_mode(self):
        return self.freq_scale_box.currentData() or spec_consts.DEFAULT_SPEC_MODE

    def _on_spectrum_mode_changed(self, _index=None):
        self.mel_param_group.setVisible(self._selected_spectrum_mode() == spec_consts.SPEC_MODE_MEL)
        self.mel_param_group.updateGeometry()
        QTimer.singleShot(0, self._resize_after_spectrum_mode_changed)

    def _resize_after_spectrum_mode_changed(self):
        self.section_container.adjustSize()
        self._refresh_section_container_minimum_height()

    def _create_mel_param_group(self):
        group = GroupBox("Mel 参数")
        layout = QVBoxLayout()

        n_mels_layout = QHBoxLayout()
        n_mels_layout.addWidget(Label("Mel 频带数量"))
        self.mel_n_mels_spinbox = SpinBox()
        self.mel_n_mels_spinbox.setRange(
            spec_consts.MIN_MEL_BAND_COUNT,
            spec_consts.MAX_MEL_BAND_COUNT,
        )
        self.mel_n_mels_spinbox.setValue(
            int(self.load_config.get("mel_n_mels", spec_consts.DEFAULT_MEL_BAND_COUNT))
        )
        n_mels_layout.addWidget(self.mel_n_mels_spinbox)
        layout.addLayout(n_mels_layout)

        fmin_layout = QHBoxLayout()
        fmin_layout.addWidget(Label("频率下限"))
        self.mel_fmin_spinbox = SpinBox()
        self.mel_fmin_spinbox.setRange(
            spec_consts.MIN_MEL_FREQUENCY_HZ,
            spec_consts.MAX_MEL_FREQUENCY_HZ,
        )
        self.mel_fmin_spinbox.setSuffix(" Hz")
        self.mel_fmin_spinbox.setValue(
            int(self.load_config.get("mel_fmin_hz", spec_consts.DEFAULT_MEL_FMIN_HZ))
        )
        fmin_layout.addWidget(self.mel_fmin_spinbox)
        layout.addLayout(fmin_layout)

        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(Label("频率上限"))
        self.mel_fmax_spinbox = SpinBox()
        self.mel_fmax_spinbox.setRange(
            spec_consts.MEL_FMAX_NYQUIST_SPINBOX_VALUE,
            spec_consts.MAX_MEL_FREQUENCY_HZ,
        )
        self.mel_fmax_spinbox.setSpecialValueText(
            spec_consts.MEL_FMAX_NYQUIST_DISPLAY_TEXT
        )
        self.mel_fmax_spinbox.setSuffix(" Hz")
        mel_fmax_mode = self.load_config.get(
            "mel_fmax_mode", spec_consts.DEFAULT_MEL_FMAX_MODE
        )
        if mel_fmax_mode == spec_consts.MEL_FMAX_MODE_MANUAL:
            self.mel_fmax_spinbox.setValue(int(self.load_config["mel_fmax_hz"]))
        else:
            self.mel_fmax_spinbox.setValue(
                spec_consts.MEL_FMAX_NYQUIST_SPINBOX_VALUE
            )
        fmax_layout.addWidget(self.mel_fmax_spinbox)
        layout.addLayout(fmax_layout)

        group.setLayout(layout)
        return group

    def create_display_param(self):
        colormap_label = Label("配色")
        self.colormap_box = ComboBox()
        self.colormap_box.addItems(spec_consts.SPEC_COLOR_MAP_OPTIONS)
        color_map = self.load_config.get("color_map", spec_consts.DEFAULT_SPEC_COLOR_MAP)
        self.colormap_box.setCurrentText(color_map)
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_box)

        top_limit_label = Label("上限")
        top_limit_label.setObjectName("toplimitlabel")
        self.top_limit_spinbox = SpinBox()
        self.top_limit_spinbox.setRange(
            spec_consts.MIN_SPEC_COLOR_LIMIT_DB,
            spec_consts.MAX_SPEC_COLOR_LIMIT_DB,
        )
        top_limit = self.load_config.get("top_limit", spec_consts.DEFAULT_SPEC_TOP_LIMIT)
        self.top_limit_spinbox.setValue(top_limit)
        top_limit_layout = QHBoxLayout()
        top_limit_layout.addWidget(top_limit_label)
        top_limit_layout.addWidget(self.top_limit_spinbox)

        bottom_limit_label = Label("下限")
        bottom_limit_label.setObjectName("bottomlimitlabel")
        self.bottom_limit_spinbox = SpinBox()
        self.bottom_limit_spinbox.setRange(
            spec_consts.MIN_SPEC_COLOR_LIMIT_DB,
            spec_consts.MAX_SPEC_COLOR_LIMIT_DB,
        )
        bottom_limit = self.load_config.get("bottom_limit", spec_consts.DEFAULT_SPEC_BOTTOM_LIMIT)
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
        self.custom_limit_checkbox.setChecked(
            self.load_config.get("custom_limit", spec_consts.DEFAULT_SPEC_CUSTOM_LIMIT)
        )

        param_layout = QVBoxLayout()
        param_layout.addLayout(colormap_layout)
        param_layout.addStretch()
        param_layout.addWidget(self.custom_limit_checkbox)
        param_layout.addStretch()
        param_layout.addWidget(self.limit_group_box)
        param_layout.addSpacing(10)
        param_layout.setSpacing(10)
        return param_layout

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def on_custom_limit_checkbox_changed(self, state):
        if state == Qt.Checked:
            self.limit_group_box.setEnabled(True)
        else:
            self.limit_group_box.setEnabled(False)

    def get_default_config(self):
        mel_fmax_hz = self.mel_fmax_spinbox.value()
        mel_fmax_mode = (
            spec_consts.MEL_FMAX_MODE_NYQUIST
            if mel_fmax_hz == spec_consts.MEL_FMAX_NYQUIST_SPINBOX_VALUE
            else spec_consts.MEL_FMAX_MODE_MANUAL
        )
        default_config = {
            "n_fft": int(self.fft_size_box.currentText()),
            "hop_length": int(self.hop_length_box.currentText()),
            "window_func": self.window_func_box.currentText(),
            "freq_scale_type": self._selected_spectrum_mode(),
            "mel_n_mels": self.mel_n_mels_spinbox.value(),
            "mel_fmin_hz": self.mel_fmin_spinbox.value(),
            "mel_fmax_mode": mel_fmax_mode,
            "color_map": self.colormap_box.currentText(),
            "top_limit": self.top_limit_spinbox.value(),
            "bottom_limit": self.bottom_limit_spinbox.value(),
            "custom_limit": self.custom_limit_checkbox.isChecked(),
            "analysis_channel": self.channel_selector.current_channel()
            if self.show_channel_selector and hasattr(self, "channel_selector")
            else int(
                self.load_config.get(
                    "analysis_channel",
                    spec_consts.DEFAULT_SPEC_ANALYSIS_CHANNEL,
                )
                or spec_consts.DEFAULT_SPEC_ANALYSIS_CHANNEL
            ),
        }
        if mel_fmax_mode == spec_consts.MEL_FMAX_MODE_MANUAL:
            default_config["mel_fmax_hz"] = mel_fmax_hz
        if default_config["freq_scale_type"] == spec_consts.SPEC_MODE_MEL:
            if (
                mel_fmax_mode == spec_consts.MEL_FMAX_MODE_MANUAL
                and mel_fmax_hz <= default_config["mel_fmin_hz"]
            ):
                MessageBox.warning(self, "设置警告", "Mel 频率上限必须大于频率下限!")
                return
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

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not config_data:
            return
        self.accept()
        return config_data
