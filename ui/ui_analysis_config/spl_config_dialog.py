"""
SPL (Sound Pressure Level) 分析配置对话框
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QVBoxLayout, QPushButton

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils, check_upper_lower_limit
from ui.signal_analysis_window import Frequency
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class SplConfigWindow(QDialog):
    """SPL 分析配置对话框"""

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(380, 350)
        self.resize(380, 350)

        layout = QVBoxLayout()

        # 平滑选项
        self.smooth_chk_box = QCheckBox("是否平滑")
        self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
        self.smooth_chk_box.stateChanged.connect(self.get_default_config)

        # 使用通用阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            upper_range=(0, 500),
            lower_range=(0, 500),
            default_upper=self.load_config.get("upper_limit", 100.0),
            default_lower=self.load_config.get("lower_limit", 0.0),
            load_config=self.load_config,
            csv_validator=Frequency.load_excel_limit
        )

        btn_layout = self.create_btn()

        layout.addWidget(self.smooth_chk_box)
        layout.addStretch()
        layout.addWidget(self.threshold_widget)
        layout.addStretch()
        layout.addLayout(btn_layout)
        layout.setSpacing(10)
        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qdoublespinbox_style
        )

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
        config = {"smooth_checked": self.smooth_chk_box.isChecked()}
        config.update(self.threshold_widget.get_config())
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        if check_upper_lower_limit(config_data, self):
            return
        save_flag = self.config_manager.save_default_config("SPL", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        if check_upper_lower_limit(config_data, self):
            return
        self.accept()
        return config_data
