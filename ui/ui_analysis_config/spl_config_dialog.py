"""
SPL (Sound Pressure Level) 分析配置对话框
"""

import re

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QPushButton,
)

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


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

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.model_type = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "SPL"
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        # 默认高度偏小会把阈值绘图区域压缩得很矮，导致“显示不完整”的观感
        height = 570 if self.model_type == "SPLF" else 430
        self.setMinimumSize(380, height)
        self.resize(380, height)

        layout = QVBoxLayout()

        # SPL: time-domain smoothing checkbox
        self.smooth_chk_box = None
        if self.model_type != "SPLF":
            self.smooth_chk_box = QCheckBox("平滑")
            self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
            self.smooth_chk_box.stateChanged.connect(self.get_default_config)

        # SPLF: calculation mode (fundamental-only vs total RMS SPL)
        self.splf_mode_group_box = None
        if self.model_type == "SPLF":
            self.splf_mode_group_box = QGroupBox("SPLF 计算方式")
            self.radio_fundamental = QRadioButton("仅基频")
            self.radio_total = QRadioButton("总SPL")

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
            self.smooth_combo_box = QComboBox()
            self.smooth_combo_box.addItems(list(self.OCTAVE_SMOOTHING_LABELS.keys()))
            selected_oct = self.load_config.get("octave_smoothing", None)
            if selected_oct is None:
                selected_oct = 6 if self.load_config.get("smooth_checked", False) else 0
            selected_oct = int(selected_oct)
            selected_label = next(
                (k for k, v in self.OCTAVE_SMOOTHING_LABELS.items() if v == selected_oct),
                "不平滑",
            )
            self.smooth_combo_box.setCurrentText(selected_label)

        # SPLF only: golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = None
        if self.model_type == "SPLF":
            self.golden_chk_box = QCheckBox("使用黄金样本")
            self.golden_chk_box.setChecked(self.load_config.get("golden_sample_checked", False))

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        weighting_label = QLabel("计权方式:")
        self.weighting_combo = QComboBox()
        self.weighting_combo.addItems(["Z（None）", "A", "B", "C", "D"])
        weighting_value = self.load_config.get("weighting", "Z（None）")
        if weighting_value in ["None", "Z"]:
            weighting_value = "Z（None）"
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

        layout.addLayout(threshold_weighting_layout)
        if self.splf_mode_group_box is not None:
            layout.addWidget(self.splf_mode_group_box)

            smooth_layout = QHBoxLayout()
            smooth_layout.addWidget(QLabel("平滑"))
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
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
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
