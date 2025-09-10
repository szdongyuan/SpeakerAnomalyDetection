import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QRadioButton, QLineEdit, QDoubleSpinBox

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils, check_upper_lower_limit


class FrConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.file_path = self.load_config.get("config_dir", None)
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(380, 350)
        self.resize(380, 350)
        layout = QVBoxLayout()
        self.smooth_chk_box = QCheckBox("是否平滑")
        self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
        self.smooth_chk_box.stateChanged.connect(self.get_default_config)
        limit_layout = self.create_limit()
        btn_layout = self.create_btn()
        layout.addWidget(self.smooth_chk_box)
        layout.addStretch()
        layout.addLayout(limit_layout)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qdoublespinbox_style
        )

    def create_limit(self):
        self.limit_checkbox = QCheckBox("阈值", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        self.limit_checkbox.stateChanged.connect(self.on_limit_checkbox_changed)
        self.limit_group_box = QGroupBox("选择阈值", self)
        self.limit_group_box.setMinimumSize(180, 180)
        if not self.limit_checkbox.isChecked():
            self.limit_group_box.setDisabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")
        self.radio_button_1 = QRadioButton("自定义")
        self.radio_button_1.setChecked(self.load_config.get("self_defined", True))
        self.radio_button_1.toggled.connect(self.on_radio_button_toggled)
        upper_layout = self.create_upper_lower_layout()
        self.radio_button_2 = QRadioButton("导入配置")
        self.radio_button_2.setChecked(self.load_config.get("import_config", False))
        self.radio_button_2.toggled.connect(self.on_radio_button_toggled)
        load_layout = self.create_config_dir_layout()

        limit_group_layout = QVBoxLayout()
        limit_group_layout.addWidget(self.radio_button_1)
        limit_group_layout.addLayout(upper_layout)
        limit_group_layout.addWidget(self.radio_button_2)
        limit_group_layout.addLayout(load_layout)

        self.limit_group_box.setLayout(limit_group_layout)

        limit_layout = QVBoxLayout()
        limit_layout.addWidget(self.limit_checkbox)
        limit_layout.addStretch()
        limit_layout.addWidget(self.limit_group_box)
        return limit_layout

    def create_upper_lower_layout(self):
        self.label_upper = QLabel("上限：", self)
        self.label_lower = QLabel("下限：", self)
        self.sinbox_upper = QDoubleSpinBox(self)
        self.sinbox_upper.setRange(0, 500)
        self.sinbox_upper.setValue(float(self.load_config.get("upper_limit")))
        self.sinbox_upper.textChanged.connect(self.get_default_config)
        self.spinbox_lower = QDoubleSpinBox(self)
        self.spinbox_lower.setRange(0, 500)
        self.spinbox_lower.setValue(float(self.load_config.get("lower_limit")))
        self.spinbox_lower.textChanged.connect(self.get_default_config)
        if not self.radio_button_1.isChecked():
            self.sinbox_upper.setDisabled(True)
            self.spinbox_lower.setDisabled(True)
            self.label_upper.setStyleSheet("color:rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")

        upper_layout = QHBoxLayout()
        upper_layout.addSpacing(19)
        upper_layout.addWidget(self.label_upper)
        upper_layout.addWidget(self.sinbox_upper)
        upper_layout.addWidget(self.label_lower)
        upper_layout.addWidget(self.spinbox_lower)
        return upper_layout

    def create_config_dir_layout(self):
        self.config_dir_label = QLabel("配置文件路径：")
        self.config_dir_box = QLineEdit()
        if not self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(True)
            self.config_dir_label.setStyleSheet("color: rgb(162, 162, 162);")
        self.config_dir_box.textChanged.connect(self.get_default_config)
        icon_path = DEFAULT_DIR + "ui/ui_pic/ai_window_pic/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, QLineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self.config_dir_btn_clicked)
        if self.load_config.get("config_dir"):
            config_dir_name = os.path.basename(self.load_config.get("config_dir"))
            self.config_dir_box.setText(config_dir_name)
        load_layout = QHBoxLayout()
        load_layout.addSpacing(19)
        load_layout.addWidget(self.config_dir_label)
        load_layout.addWidget(self.config_dir_box)
        return load_layout

    def on_radio_button_toggled(self):
        self.get_default_config()
        if self.radio_button_1.isChecked():
            self.config_dir_box.setDisabled(True)
            self.sinbox_upper.setDisabled(False)
            self.spinbox_lower.setDisabled(False)
            self.label_upper.setStyleSheet("color: rgb(0, 0, 0);")
            self.label_lower.setStyleSheet("color: rgb(0, 0, 0);")
            self.config_dir_label.setStyleSheet("color: rgb(162, 162, 162);")
        elif self.radio_button_2.isChecked():
            self.sinbox_upper.setDisabled(True)
            self.spinbox_lower.setDisabled(True)
            self.config_dir_box.setDisabled(False)
            self.label_upper.setStyleSheet("color:rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")
            self.config_dir_label.setStyleSheet("color: rgb(0, 0, 0);")

    def config_dir_btn_clicked(self):
        self.file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件路径", DEFAULT_DIR + "ui/ui_config", filter="All Files (*);;"
        )
        if self.file_path:
            self.config_dir = self.file_path
            config_dir_name = os.path.basename(self.file_path)
        self.config_dir_box.setText(config_dir_name)

    def on_limit_checkbox_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
            self.limit_group_box.setStyleSheet("color: rgb(0, 0, 0);")
        else:
            self.limit_group_box.setDisabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")

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
        default_config = {
            "smooth_checked": self.smooth_chk_box.isChecked(),
            "limit_checked": self.limit_checkbox.isChecked(),
            "self_defined": self.radio_button_1.isChecked(),
            "import_config": self.radio_button_2.isChecked(),
            "upper_limit": self.sinbox_upper.value(),
            "lower_limit": self.spinbox_lower.value(),
            "config_dir": self.file_path,
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self):
            return
        save_flag = self.config_manager.save_default_config("FR", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self):
            return
        self.accept()
        return config_data
