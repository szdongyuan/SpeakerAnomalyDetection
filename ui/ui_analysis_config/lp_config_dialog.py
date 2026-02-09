from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox
from typing import List, Optional

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class LPConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
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
        channel_layout.addWidget(QLabel("通道:"))
        self.channel_combo_box = QComboBox()
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.load_config.get("analysis_channel", None)
        if saved_channel is None or int(saved_channel) not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(int(saved_channel))
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        channel_layout.addWidget(self.channel_combo_box)
        return channel_layout

    def init_ui(self):
        self.setMinimumSize(350, 350)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        layout = QVBoxLayout()
        layout.addLayout(self._create_channel_layout())
        lp_config_box = self.create_lp_config_box()
        btn_layout = self.create_btn_layout()
        layout.addWidget(lp_config_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qlabel_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qgroupbox_style
        )

    def create_lp_config_box(self):
        lp_config_box = QGroupBox("松散颗粒参数配置")
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
        trigger_threshold_label = QLabel("触发阈值:")
        self.trigger_threshold_spinbox = QSpinBox()
        self.trigger_threshold_spinbox.setSuffix(" dB")
        self.trigger_threshold_spinbox.setValue(self.load_config.get("trigger_threshold", 0))
        self.trigger_threshold_spinbox.setAlignment(Qt.AlignRight)
        trigger_threshold_layout = QHBoxLayout()
        trigger_threshold_layout.addWidget(trigger_threshold_label)
        trigger_threshold_layout.addWidget(self.trigger_threshold_spinbox)

        return trigger_threshold_layout

    def create_confirm_threshold_layout(self):
        confirm_threshold_label = QLabel("确认区间:")
        self.hysterests_threshold_spinbox = QSpinBox()
        self.hysterests_threshold_spinbox.setSuffix(" dB")
        self.hysterests_threshold_spinbox.setValue(self.load_config.get("hysterests_threshold", 0))
        self.hysterests_threshold_spinbox.setAlignment(Qt.AlignRight)
        confirm_threshold_layout = QHBoxLayout()
        confirm_threshold_layout.addWidget(confirm_threshold_label)
        confirm_threshold_layout.addWidget(self.hysterests_threshold_spinbox)

        return confirm_threshold_layout

    def create_min_check_duration_layout(self):
        min_check_duration_label = QLabel("最小检测时长:")
        self.min_check_duration_spinbox = QSpinBox()
        self.min_check_duration_spinbox.setSuffix(" ms")
        self.min_check_duration_spinbox.setValue(self.load_config.get("min_check_duration", 0))
        self.min_check_duration_spinbox.setRange(0, 1000)
        self.min_check_duration_spinbox.setAlignment(Qt.AlignRight)
        min_check_duration_layout = QHBoxLayout()
        min_check_duration_layout.addWidget(min_check_duration_label)
        min_check_duration_layout.addWidget(self.min_check_duration_spinbox)

        return min_check_duration_layout

    def create_max_check_duration_layout(self):
        max_check_duration_label = QLabel("最大检测时长:")
        self.max_check_duration_spinbox = QSpinBox()
        self.max_check_duration_spinbox.setSuffix(" ms")
        self.max_check_duration_spinbox.setValue(self.load_config.get("max_check_duration", 0))
        self.max_check_duration_spinbox.setRange(0, 1000)
        self.max_check_duration_spinbox.setAlignment(Qt.AlignRight)
        max_check_duration_layout = QHBoxLayout()
        max_check_duration_layout.addWidget(max_check_duration_label)
        max_check_duration_layout.addWidget(self.max_check_duration_spinbox)

        return max_check_duration_layout

    def create_loose_particle_num_layout(self):
        loose_particle_num_label = QLabel("允许松散颗粒数量:")
        self.loose_particle_num_spinbox = QSpinBox()
        self.loose_particle_num_spinbox.setValue(self.load_config.get("loose_particle_num", 0))
        self.loose_particle_num_spinbox.setAlignment(Qt.AlignRight)
        loose_particle_num_layout = QHBoxLayout()
        loose_particle_num_layout.addWidget(loose_particle_num_label)
        loose_particle_num_layout.addWidget(self.loose_particle_num_spinbox)

        return loose_particle_num_layout

    def create_stimulus_max_frequency_layout(self):
        stimulus_max_frequency_label = QLabel("信号最大频率:")
        self.stimulus_max_frequency_spinbox = QSpinBox()
        self.stimulus_max_frequency_spinbox.setSuffix(" Hz")
        self.stimulus_max_frequency_spinbox.setRange(10, 24000)
        self.stimulus_max_frequency_spinbox.setValue(self.load_config.get("cutoff_freq", 0))
        self.stimulus_max_frequency_spinbox.setAlignment(Qt.AlignRight)

        stimulus_max_frequency_layout = QHBoxLayout()
        stimulus_max_frequency_layout.addWidget(stimulus_max_frequency_label)
        stimulus_max_frequency_layout.addWidget(self.stimulus_max_frequency_spinbox)

        return stimulus_max_frequency_layout

    def create_btn_layout(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton("设为默认")
        ok_btn = QPushButton(" 确  定 ")
        default_btn.clicked.connect(self.on_click_default_btn)
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)

        return btn_layout

    def get_default_config(self):
        default_config = {
            "trigger_threshold": self.trigger_threshold_spinbox.value(),
            "hysterests_threshold": self.hysterests_threshold_spinbox.value(),
            "min_check_duration": self.min_check_duration_spinbox.value(),
            "max_check_duration": self.max_check_duration_spinbox.value(),
            "loose_particle_num": self.loose_particle_num_spinbox.value(),
            "cutoff_freq": self.stimulus_max_frequency_spinbox.value(),
            "analysis_channel": int(self.channel_combo_box.currentData()),
        }
        return default_config

    def on_click_default_btn(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("LP", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data
