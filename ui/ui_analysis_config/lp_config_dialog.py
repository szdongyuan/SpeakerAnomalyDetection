from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from typing import List, Optional

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import GroupBox, Label, SpinBox
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    MultiChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)


class LPConfigWindow(SemanticAnalysisConfigDialogBase):
    def __init__(
        self, config_manager, model_type,
        available_channels: Optional[List[int]] = None,
        allow_multiple_channels: bool = False,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.show_channel_selector = available_channels is not None
        self.allow_multiple_channels = allow_multiple_channels
        self.available_channels = available_channels

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LP 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_click_default_btn,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            selector_type = (
                MultiChannelSelectorWidget
                if self.allow_multiple_channels else ChannelSelectorWidget
            )
            self.channel_selector = selector_type(
                self.load_config,
                self.available_channels,
                self,
            )
            self.add_semantic_section("input", widget=self.channel_selector)
        self.add_semantic_section(
            "detection",
            widget=self.create_lp_config_box(),
        )

    def create_lp_config_box(self):
        lp_config_box = GroupBox("松散颗粒参数配置")
        lp_config_box_layout = QVBoxLayout()
        trigger_threshold_layout = self.create_trigger_threshold_layout()
        comfirm_threshold_layout = self.create_confirm_threshold_layout()
        max_check_duration_layout = self.create_max_check_duration_layout()
        min_check_duration_layout = self.create_min_check_duration_layout()
        max_stimulus_frequency_layout = self.create_stimulus_max_frequency_layout()
        loose_particle_layout = self.create_loose_particle_num_layout()
        lp_config_box_layout.addLayout(trigger_threshold_layout)
        lp_config_box_layout.addStretch()
        lp_config_box_layout.addLayout(comfirm_threshold_layout)
        lp_config_box_layout.addStretch()
        lp_config_box_layout.addLayout(max_check_duration_layout)
        lp_config_box_layout.addStretch()
        lp_config_box_layout.addLayout(min_check_duration_layout)
        lp_config_box_layout.addStretch()
        lp_config_box_layout.addLayout(max_stimulus_frequency_layout)
        lp_config_box_layout.addStretch()
        lp_config_box_layout.addLayout(loose_particle_layout)
        lp_config_box_layout.setSpacing(10)
        lp_config_box_layout.setContentsMargins(10, 20, 10, 20)
        lp_config_box.setLayout(lp_config_box_layout)

        return lp_config_box

    def create_trigger_threshold_layout(self):
        trigger_threshold_label = Label("触发阈值:")
        self.trigger_threshold_spinbox = SpinBox()
        self.trigger_threshold_spinbox.setSuffix(" dB")
        self.trigger_threshold_spinbox.setValue(self.load_config.get("trigger_threshold", 0))
        self.trigger_threshold_spinbox.setAlignment(Qt.AlignRight)
        trigger_threshold_layout = QHBoxLayout()
        trigger_threshold_layout.addWidget(trigger_threshold_label)
        trigger_threshold_layout.addWidget(self.trigger_threshold_spinbox)

        return trigger_threshold_layout

    def create_confirm_threshold_layout(self):
        confirm_threshold_label = Label("确认区间:")
        self.hysterests_threshold_spinbox = SpinBox()
        self.hysterests_threshold_spinbox.setSuffix(" dB")
        self.hysterests_threshold_spinbox.setValue(self.load_config.get("hysterests_threshold", 0))
        self.hysterests_threshold_spinbox.setAlignment(Qt.AlignRight)
        confirm_threshold_layout = QHBoxLayout()
        confirm_threshold_layout.addWidget(confirm_threshold_label)
        confirm_threshold_layout.addWidget(self.hysterests_threshold_spinbox)

        return confirm_threshold_layout

    def create_min_check_duration_layout(self):
        min_check_duration_label = Label("最小检测时长:")
        self.min_check_duration_spinbox = SpinBox()
        self.min_check_duration_spinbox.setSuffix(" ms")
        self.min_check_duration_spinbox.setValue(self.load_config.get("min_check_duration", 0))
        self.min_check_duration_spinbox.setRange(0, 1000)
        self.min_check_duration_spinbox.setAlignment(Qt.AlignRight)
        min_check_duration_layout = QHBoxLayout()
        min_check_duration_layout.addWidget(min_check_duration_label)
        min_check_duration_layout.addWidget(self.min_check_duration_spinbox)

        return min_check_duration_layout

    def create_max_check_duration_layout(self):
        max_check_duration_label = Label("最大检测时长:")
        self.max_check_duration_spinbox = SpinBox()
        self.max_check_duration_spinbox.setSuffix(" ms")
        self.max_check_duration_spinbox.setValue(self.load_config.get("max_check_duration", 0))
        self.max_check_duration_spinbox.setRange(0, 1000)
        self.max_check_duration_spinbox.setAlignment(Qt.AlignRight)
        max_check_duration_layout = QHBoxLayout()
        max_check_duration_layout.addWidget(max_check_duration_label)
        max_check_duration_layout.addWidget(self.max_check_duration_spinbox)

        return max_check_duration_layout

    def create_loose_particle_num_layout(self):
        loose_particle_num_label = Label("允许松散颗粒数量:")
        self.loose_particle_num_spinbox = SpinBox()
        self.loose_particle_num_spinbox.setValue(self.load_config.get("loose_particle_num", 0))
        self.loose_particle_num_spinbox.setAlignment(Qt.AlignRight)
        loose_particle_num_layout = QHBoxLayout()
        loose_particle_num_layout.addWidget(loose_particle_num_label)
        loose_particle_num_layout.addWidget(self.loose_particle_num_spinbox)

        return loose_particle_num_layout

    def create_stimulus_max_frequency_layout(self):
        stimulus_max_frequency_label = Label("信号最大频率:")
        self.stimulus_max_frequency_spinbox = SpinBox()
        self.stimulus_max_frequency_spinbox.setSuffix(" Hz")
        self.stimulus_max_frequency_spinbox.setRange(10, 24000)
        self.stimulus_max_frequency_spinbox.setValue(self.load_config.get("cutoff_freq", 0))
        self.stimulus_max_frequency_spinbox.setAlignment(Qt.AlignRight)

        stimulus_max_frequency_layout = QHBoxLayout()
        stimulus_max_frequency_layout.addWidget(stimulus_max_frequency_label)
        stimulus_max_frequency_layout.addWidget(self.stimulus_max_frequency_spinbox)

        return stimulus_max_frequency_layout

    def create_btn_layout(self):
        return self.create_standard_button_layout(
            self.on_click_default_btn,
            self.on_click_ok_btn,
        )

    def get_default_config(self):
        default_config = {
            "trigger_threshold": self.trigger_threshold_spinbox.value(),
            "hysterests_threshold": self.hysterests_threshold_spinbox.value(),
            "min_check_duration": self.min_check_duration_spinbox.value(),
            "max_check_duration": self.max_check_duration_spinbox.value(),
            "loose_particle_num": self.loose_particle_num_spinbox.value(),
            "cutoff_freq": self.stimulus_max_frequency_spinbox.value(),
        }
        if self.show_channel_selector:
            default_config.update(self.channel_selector.get_config())
        else:
            default_config.update(
                ChannelSelectorWidget.normalized_config(self.load_config)
            )
        return default_config

    def on_click_default_btn(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("LP", config_data)
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
        self.accept()
        return config_data
