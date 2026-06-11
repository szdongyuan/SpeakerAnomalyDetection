"""
SPL (Sound Pressure Level) 分析配置对话框
"""

import re
from typing import List, Optional

from PyQt5.QtWidgets import QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, RadioButton
from ui.ui_analysis_config.common_widgets import (
    AnalysisConfigDialogBase,
    ChannelSelectorWidget,
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.ui_src import ui_resources


class SplConfigWindow(AnalysisConfigDialogBase):
    """SPL 分析配置对话框"""

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.model_type = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "SPL"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.init_ui()

    def init_ui(self):
        # 默认高度偏小会把阈值绘图区域压缩得很矮，导致“显示不完整”的观感
        height = 570 if self.model_type == "SPLF" else 430
        self.setMinimumSize(380, height)
        self.resize(380, height)

        layout = QVBoxLayout()

        # SPL: time-domain smoothing checkbox
        self.smooth_chk_box = None
        if self.model_type != "SPLF":
            self.smooth_chk_box = CheckBox("平滑")
            self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
            self.smooth_chk_box.stateChanged.connect(self.get_default_config)

        # SPLF: calculation mode (fundamental-only vs total RMS SPL)
        self.splf_mode_group_box = None
        if self.model_type == "SPLF":
            self.splf_mode_group_box = GroupBox("SPLF 计算方式")
            self.radio_fundamental = RadioButton("仅基频")
            self.radio_total = RadioButton("总SPL")

            mode = str(self.load_config.get("splf_calc_mode", "fundamental") or "fundamental").lower()
            if mode == "total":
                self.radio_total.setChecked(True)
            else:
                self.radio_fundamental.setChecked(True)

            mode_layout = QVBoxLayout()
            mode_layout.addWidget(self.radio_fundamental)
            mode_layout.addWidget(self.radio_total)
            self.splf_mode_group_box.setLayout(mode_layout)

            self.smoothing_selector = OctaveSmoothingSelectorWidget(self.load_config, parent=self)

        # SPLF only: golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = None
        if self.model_type == "SPLF":
            self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        self.weighting_selector = WeightingSelectorWidget(self.load_config, parent=self)

        btn_layout = self.create_btn()

        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(self.load_config, self.available_channels, self)
            layout.addWidget(self.channel_selector)
        layout.addWidget(self.weighting_selector)
        if self.splf_mode_group_box is not None:
            layout.addWidget(self.splf_mode_group_box)
            layout.addWidget(self.smoothing_selector)
        elif self.smooth_chk_box is not None:
            layout.addWidget(self.smooth_chk_box)
        if self.golden_chk_box is not None:
            layout.addWidget(self.golden_chk_box)
        layout.addWidget(self.threshold_widget)
        layout.addStretch()
        layout.addLayout(btn_layout)
        layout.setSpacing(10)
        self.setLayout(layout)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def get_default_config(self):
        """获取配置数据"""
        config = {}
        if self.model_type != "SPLF" and self.smooth_chk_box is not None:
            config["smooth_checked"] = self.smooth_chk_box.isChecked()
        if self.model_type == "SPLF":
            calc_mode = "fundamental"
            if hasattr(self, "radio_total") and self.radio_total.isChecked():
                calc_mode = "total"
            config["splf_calc_mode"] = calc_mode
            config.update(self.smoothing_selector.get_config())
            if self.golden_chk_box is not None:
                config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        config.update(self.weighting_selector.get_config())
        if self.show_channel_selector and hasattr(self, "channel_selector"):
            config.update(self.channel_selector.get_config())
        else:
            config["analysis_channel"] = int(self.load_config.get("analysis_channel", 0) or 0)
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config(self.model_type, config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        self.accept()
        return config_data
