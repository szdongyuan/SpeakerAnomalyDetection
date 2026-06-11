"""
SPL (Sound Pressure Level) 分析配置对话框
"""

import re
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, Label, PushButton, ComboBox, RadioButton
from ui.ui_analysis_config.config_normalization import normalize_octave_smoothing, weighting_to_display_label
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.ui_src import ui_resources


class SplConfigWindow(QDialog):
    """SPL 分析配置对话框"""

    OCTAVE_SMOOTHING_LABELS = {
        "不平滑": 0,
        "1/1 Oct": 1,
        "1/3 Oct": 3,
        "1/6 Oct": 6,
        "1/12 Oct": 12,
        "1/24 Oct": 24,
        "1/48 Oct": 48,
    }

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.model_type = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "SPL"
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

            # SPLF: octave smoothing dropdown (frequency-domain)
            self.smooth_combo_box = ComboBox()
            self.smooth_combo_box.addItems(list(self.OCTAVE_SMOOTHING_LABELS.keys()))
            selected_oct = normalize_octave_smoothing(self.load_config, default=0)
            selected_label = next(
                (k for k, v in self.OCTAVE_SMOOTHING_LABELS.items() if v == selected_oct),
                "不平滑",
            )
            self.smooth_combo_box.setCurrentText(selected_label)

        # SPLF only: golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = None
        if self.model_type == "SPLF":
            self.golden_chk_box = CheckBox("使用黄金样本")
            self.golden_chk_box.setChecked(self.load_config.get("golden_sample_checked", False))

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        weighting_label = Label("计权方式:")
        self.weighting_combo = ComboBox()
        self.weighting_combo.addItems(["Z（None）", "A", "B", "C", "D"])
        weighting_value = weighting_to_display_label(self.load_config.get("weighting", "Z"))
        index = self.weighting_combo.findText(weighting_value)
        if index >= 0:
            self.weighting_combo.setCurrentIndex(index)
        else:
            self.weighting_combo.setCurrentIndex(0)

        threshold_weighting_layout = QHBoxLayout()
        threshold_weighting_layout.addWidget(weighting_label)
        threshold_weighting_layout.addWidget(self.weighting_combo)
        threshold_weighting_layout.addStretch()

        btn_layout = self.create_btn()

        if self.show_channel_selector:
            layout.addLayout(self._create_channel_layout())
        layout.addLayout(threshold_weighting_layout)
        if self.splf_mode_group_box is not None:
            layout.addWidget(self.splf_mode_group_box)

            smooth_layout = QHBoxLayout()
            smooth_layout.addWidget(Label("平滑"))
            smooth_layout.addWidget(self.smooth_combo_box)
            layout.addLayout(smooth_layout)
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
        """获取配置数据"""
        config = {}
        if self.model_type != "SPLF" and self.smooth_chk_box is not None:
            config["smooth_checked"] = self.smooth_chk_box.isChecked()
        if self.model_type == "SPLF":
            calc_mode = "fundamental"
            if hasattr(self, "radio_total") and self.radio_total.isChecked():
                calc_mode = "total"
            config["splf_calc_mode"] = calc_mode
            smooth_label = self.smooth_combo_box.currentText()
            config["octave_smoothing"] = int(self.OCTAVE_SMOOTHING_LABELS.get(smooth_label, 0))
            if self.golden_chk_box is not None:
                config["golden_sample_checked"] = self.golden_chk_box.isChecked()
        config.update(self.threshold_widget.get_config())
        weighting_value = self.weighting_combo.currentText()
        if weighting_value == "Z（None）":
            config["weighting"] = "Z"
        else:
            config["weighting"] = weighting_value
        if self.show_channel_selector and hasattr(self, "channel_combo_box"):
            config["analysis_channel"] = int(self.channel_combo_box.currentData())
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
