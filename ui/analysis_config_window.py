import json
import os
import sys
from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy
from PyQt5.QtWidgets import QSpacerItem, QVBoxLayout, QWidget

from base.log_manager import LogManager
from base.training_model_management import TrainingModelManagement
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR


class SplConfigWindow(QDialog):
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
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        self.smooth_chk_box = QCheckBox("是否平滑")
        self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
        self.smooth_chk_box.stateChanged.connect(self.get_default_config)
        limit_layout = self.create_limit()
        btn_layout = self.create_btn()

        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout.addWidget(self.smooth_chk_box)
        layout.addItem(v_spacer_1)
        layout.addLayout(limit_layout)
        layout.addItem(v_spacer_2)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcheckbox_stytle + 
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qlineedit_stytle + 
                           ui_style_const.qradiobutton_stytle +
                           ui_style_const.qpushbutton_stytle)

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
        self.radio_button_2 = QRadioButton("导入配置文件")
        self.radio_button_2.setChecked(self.load_config.get("import_config", False))
        self.radio_button_2.toggled.connect(self.on_radio_button_toggled)
        load_layout = self.create_config_dir_layout()

        limit_group_layout = QVBoxLayout()
        limit_group_layout.addWidget(self.radio_button_1)
        limit_group_layout.addLayout(upper_layout)
        limit_group_layout.addWidget(self.radio_button_2)
        limit_group_layout.addLayout(load_layout)

        self.limit_group_box.setLayout(limit_group_layout)

        v_spacer_3 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        limit_layout = QVBoxLayout()
        limit_layout.addWidget(self.limit_checkbox)
        limit_layout.addItem(v_spacer_3)
        limit_layout.addWidget(self.limit_group_box)
        return limit_layout

    def on_radio_button_toggled(self):
        self.get_default_config()
        if self.radio_button_1.isChecked():
            self.config_dir_box.setDisabled(True)
            self.line_edit_upper.setDisabled(False)
            self.line_edit_lower.setDisabled(False)
            self.label_upper.setStyleSheet("color: rgb(0, 0, 0);")
            self.label_lower.setStyleSheet("color: rgb(0, 0, 0);")
        elif self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(False)
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)
            self.label_upper.setStyleSheet("color: rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")

    def create_upper_lower_layout(self):
        self.label_upper = QLabel("上限：", self)
        self.label_lower = QLabel("下限：", self)
        self.line_edit_upper = QLineEdit(self)
        self.line_edit_upper.setText(self.load_config.get("upper_limit"))
        self.line_edit_upper.textChanged.connect(self.get_default_config)
        self.line_edit_lower = QLineEdit(self)
        self.line_edit_lower.setText(self.load_config.get("lower_limit"))
        self.line_edit_lower.textChanged.connect(self.get_default_config)
        if not self.radio_button_1.isChecked():
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)

        h_spacer_1 = QSpacerItem(19, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        upper_layout = QHBoxLayout()
        upper_layout.addItem(h_spacer_1)
        upper_layout.addWidget(self.label_upper)
        upper_layout.addWidget(self.line_edit_upper)
        upper_layout.addWidget(self.label_lower)
        upper_layout.addWidget(self.line_edit_lower)
        return upper_layout

    def create_config_dir_layout(self):
        self.config_dir_box = QLineEdit()
        if not self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(True)
        self.config_dir_box.textChanged.connect(self.get_default_config)
        icon_path = DEFAULT_DIR + "ui/ui_pic/ai_window_pic/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, QLineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self.config_dir_btn_clicked)
        if self.load_config.get("config_dir"):
            config_dir_name = os.path.basename(self.load_config.get("config_dir"))
            self.config_dir_box.setText(config_dir_name)
        h_spacer_1 = QSpacerItem(19, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        load_layout = QHBoxLayout()
        load_layout.addItem(h_spacer_1)
        load_layout.addWidget(self.config_dir_box)
        return load_layout

    def on_limit_checkbox_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
            self.limit_group_box.setStyleSheet("color: rgb(0, 0, 0);")
            self.on_radio_button_toggled()
        else:
            self.limit_group_box.setDisabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")

    def config_dir_btn_clicked(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "选择配置文件路径",
                                                   DEFAULT_DIR + "ui/ui_config",
                                                   filter="All Files (*);;")
        if self.file_path:
            self.config_dir = self.file_path
            config_dir_name = os.path.basename(self.file_path)
            self.config_dir_box.setText(config_dir_name)

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(default_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            "smooth_checked": self.smooth_chk_box.isChecked(),
            "limit_checked": self.limit_checkbox.isChecked(),
            "self_defined": self.radio_button_1.isChecked(),
            "import_config": self.radio_button_2.isChecked(),
            "upper_limit": self.line_edit_upper.text(),
            "lower_limit": self.line_edit_lower.text(),
            "config_dir": self.file_path
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self): return
        save_flag = self.config_manager.save_default_config("SPL", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self): return
        self.accept()
        return config_data


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
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        self.smooth_chk_box = QCheckBox("是否平滑")
        self.smooth_chk_box.setChecked(self.load_config.get("smooth_checked", False))
        self.smooth_chk_box.stateChanged.connect(self.get_default_config)
        limit_layout = self.create_limit()
        btn_layout = self.create_btn()
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addWidget(self.smooth_chk_box)
        layout.addItem(v_spacer_1)
        layout.addLayout(limit_layout)
        layout.addItem(v_spacer_2)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcheckbox_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle + 
                           ui_style_const.qradiobutton_stytle)

    def create_limit(self):
        self.limit_checkbox = QCheckBox("阈值", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
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
        limit_layout.addItem(v_spacer_1)
        limit_layout.addWidget(self.limit_group_box)
        return limit_layout

    def create_upper_lower_layout(self):
        self.label_upper = QLabel("上限：", self)
        self.label_lower = QLabel("下限：", self)
        self.line_edit_upper = QLineEdit(self)
        self.line_edit_upper.setText(self.load_config.get("upper_limit"))
        self.line_edit_upper.textChanged.connect(self.get_default_config)
        self.line_edit_lower = QLineEdit(self)
        self.line_edit_lower.setText(self.load_config.get("lower_limit"))
        self.line_edit_lower.textChanged.connect(self.get_default_config)
        if not self.radio_button_1.isChecked():
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)
            self.label_upper.setStyleSheet("color:rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")

        h_spacer_1 = QSpacerItem(19, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        upper_layout = QHBoxLayout()
        upper_layout.addItem(h_spacer_1)
        upper_layout.addWidget(self.label_upper)
        upper_layout.addWidget(self.line_edit_upper)
        upper_layout.addWidget(self.label_lower)
        upper_layout.addWidget(self.line_edit_lower)
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
        h_spacer_1 = QSpacerItem(19, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        load_layout = QHBoxLayout()
        load_layout.addItem(h_spacer_1)
        load_layout.addWidget(self.config_dir_label)
        load_layout.addWidget(self.config_dir_box)
        return load_layout

    def on_radio_button_toggled(self):
        self.get_default_config()
        if self.radio_button_1.isChecked():
            self.config_dir_box.setDisabled(True)
            self.line_edit_upper.setDisabled(False)
            self.line_edit_lower.setDisabled(False)
            self.label_upper.setStyleSheet("color: rgb(0, 0, 0);")
            self.label_lower.setStyleSheet("color: rgb(0, 0, 0);")
            self.config_dir_label.setStyleSheet("color: rgb(162, 162, 162);")
        elif self.radio_button_2.isChecked():
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)
            self.config_dir_box.setDisabled(False)
            self.label_upper.setStyleSheet("color:rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")
            self.config_dir_label.setStyleSheet("color: rgb(0, 0, 0);")

    def config_dir_btn_clicked(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "选择配置文件路径",
                                                   DEFAULT_DIR + "ui/ui_config",
                                                   filter="All Files (*);;")
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
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(default_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            "smooth_checked": self.smooth_chk_box.isChecked(),
            "limit_checked": self.limit_checkbox.isChecked(),
            "self_defined": self.radio_button_1.isChecked(),
            "import_config": self.radio_button_2.isChecked(),
            "upper_limit": self.line_edit_upper.text(),
            "lower_limit": self.line_edit_lower.text(),
            "config_dir": self.file_path
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self): return
        save_flag = self.config_manager.save_default_config("FR", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self): return
        self.accept()
        return config_data


class HdConfigWindow(QDialog):

    selected_labels_changed = pyqtSignal()
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.selected_labels = self.load_config.get("selected_labels", [])
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(300, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        harmonic_group_box = QGroupBox("谐波失真")
        harmonic_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        harmonic_slider_layout = self.create_harmonic_slider_layout()
        harmonic_slider_layout.setSpacing(12)
        self.select_all_check = QCheckBox("全选")
        self.select_all_check.setChecked(self.load_config.get("all_checked", False))
        self.select_all_check.stateChanged.connect(self.on_select_all_changed)
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        harmonic_slider_layout.addItem(v_spacer_1)
        harmonic_slider_layout.addWidget(self.select_all_check)
        harmonic_group_box.setLayout(harmonic_slider_layout)
        btn_layout = self.create_btn()
        v_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addWidget(harmonic_group_box)
        layout.addItem(v_spacer_2)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qgroupbox_stytle +
                           ui_style_const.qcheckbox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlabel_stytle)

    def create_harmonic_slider_layout(self):
        harmonic_slider_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(120, 150)
        box_container = QWidget()
        self.box_layout = QVBoxLayout()
        for i in range(2, 31):
            label = QLabel("  " + str(i))
            label.setFixedSize(90, 25)
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("QLabel:focus { outline: none; }")
            label.setAutoFillBackground(True)
            label.mousePressEvent = partial(self.on_label_click, label)
            label.enterEvent = partial(self.on_label_enter, label)
            label.leaveEvent = partial(self.on_label_leave, label)
            if i in self.selected_labels:
                label.setText("\u2713" + label.text().strip())
            self.box_layout.addWidget(label)
        if self.load_config.get("all_checked"):
            self.scroll_area.setDisabled(True)
        box_container.setLayout(self.box_layout)
        self.scroll_area.setWidget(box_container)
        harmonic_slider_layout.addWidget(self.scroll_area)
        harmonic_slider_layout.addStretch()
        return harmonic_slider_layout

    def on_select_all_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.scroll_area.setDisabled(True)
            self.scroll_area.setStyleSheet("color: rgb(162, 162, 162);")
            self.selected_labels = list(range(2, 31))
            for i in range(self.box_layout.count()):
                label = self.box_layout.itemAt(i).widget()
                text = label.text().strip()
                if not text.startswith("\u2713"):
                    label.setText("\u2713" + text)
        else:
            self.scroll_area.setDisabled(False)
            self.scroll_area.setStyleSheet("color: rgb(0, 0, 0);")
            self.selected_labels = []
            for i in range(self.box_layout.count()):
                label = self.box_layout.itemAt(i).widget()
                text = label.text().strip()
                if text.startswith("\u2713"):
                    label.setText("  " + text[1:])
        self.selected_labels_changed.emit()

    def on_label_click(self, label, event):
        checked_box = "\u2713"
        cleaned_label = ''.join(filter(str.isdigit, label.text()))
        label_value = int(cleaned_label)
        if label_value in self.selected_labels:
            self.selected_labels.remove(label_value)
            self.selected_labels.sort()
            label.setText(label.text().replace(checked_box, "").lstrip())
            label.setText("  " + label.text().replace(checked_box, ""))
        else:
            self.selected_labels.append(label_value)
            label.setText(checked_box + label.text().strip())
        self.selected_labels_changed.emit()

    def on_label_enter(self, label, event):
        label.setStyleSheet("background-color: #5099ccff;")

    def on_label_leave(self, label, event):
        label.setStyleSheet("background-color: transparent;")


    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(default_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            "selected_labels": self.selected_labels,
            "all_checked": self.select_all_check.isChecked(),
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("HD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self.selected_labels:
            QMessageBox.warning(self, "设置警告", "请选择谐波失真阶数")
        else:
            config_data = self.get_default_config()
            self.accept()
            return config_data


class AIConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_list = self.load_model_name_from_db()
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        model_box = self.create_model_layout()
        btn_layout = self.create_btn()
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addWidget(model_box)
        layout.addItem(v_spacer_1)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qgroupbox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qcombobox_stytle)
        
    def cheack_model_list(self):
        if self.analyse_model_combo_box.count() == 0:
            QMessageBox.warning(self, "设置警告", "没有可用的AI模型选型,请检查配置!")

    def create_model_layout(self):
        model_box = QGroupBox("模型")
        model_box.setMinimumSize(150, 150)
        analyse_model_label = QLabel("分析模型:")
        self.analyse_model_combo_box = QComboBox(self)
        self.analyse_model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.analyse_model_combo_box.setFixedHeight(30)
        for model_name in self.model_list:
            self.analyse_model_combo_box.addItem(model_name)
        self.analyse_model_combo_box.setCurrentText(self.load_config.get("analyse_model_name"))
        self.analyse_model_combo_box.currentTextChanged.connect(self.get_default_config)
        QTimer.singleShot(5, self.cheack_model_list)
        analyse_model_combo_layout = QHBoxLayout()
        analyse_model_combo_layout.addWidget(analyse_model_label)
        analyse_model_combo_layout.addWidget(self.analyse_model_combo_box)
        analyse_model_combo_layout.setSpacing(10)
        model_box.setLayout(analyse_model_combo_layout)
        return model_box

    def load_model_name_from_db(self):
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
        if query_code == error_code.OK:
            for idx, name in enumerate(query_result):
                query_result_idx = query_result[idx]
                input_dim = int(query_result_idx[1].split(' ')[0])
                code, stimulus_len = self.get_stimulus_len_from_json()
                if code != error_code.OK:
                    model_list.append(query_result_idx[0])
                else:
                    if input_dim == stimulus_len:
                        model_list.append(query_result_idx[0])
        return model_list

    @staticmethod
    def get_stimulus_len_from_json():
        json_file_path = DEFAULT_DIR + "ui/ui_config/stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            stimulus_len = data["stimulus_info"]["total_time"] * data["stimulus_info"]["sample_rate"]
            return error_code.OK, stimulus_len

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(default_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            "analyse_model_name": self.analyse_model_combo_box.currentText()
        }
        return default_config  

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("AI", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class PopupUtils(object):
    # """
    #     noicon : 0
    #     warning : 2
    #     question : 4
    #     information : 1
    #     critical : 3
    # """
    # @staticmethod
    # def popup_massagebox(parent, title, message, icon_type:int):
    #     msg = QMessageBox(parent)
    #     msg.setIcon(icon_type)
    #     msg.setText(message)
    #     msg.setWindowTitle(title)
    #     msg.setStandardButtons(QMessageBox.Ok)
    #     msg.exec_()

    @staticmethod
    def save_popup(parent, success_flag=True):
        save_msg = QMessageBox(parent)
        if success_flag:
            save_msg.setIcon(QMessageBox.Information)
            save_msg.setText("设置成功")
            save_msg.setWindowTitle("设置成功")
        else:
            save_msg.setIcon(QMessageBox.Critical)
            save_msg.setText("设置失败，请重试")
            save_msg.setWindowTitle("设置失败")
        save_msg.exec_()


class ConfigManager(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.default_logger = LogManager.set_log_handler("core")
        self.config = {}

    def save_config(self, type, config_data):
        if type in self.config:
            self.config[type].update(config_data)
        else:
            self.config[type] = config_data
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
                self.default_logger.info(f"The config info for {type} analysis has been saved to {self.config_file}.")
                return True
        except Exception as e:
            self.default_logger.error(f"The config info for {type} analysis save failed. {e}")
            return False
        
    def save_default_config(self, type, config_data):
        default_config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        default_config = {}
        try:
            with open(default_config_file, 'r') as f:
                default_config = json.load(f)
                if type in default_config:
                    default_config[type].update(config_data)
                else:
                    default_config[type] = config_data
            with open(default_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
                self.default_logger.info(f"The config info for {type} analysis has been saved to {default_config_file}.")
                return True
        except Exception as e:
            self.default_logger.error(f"Failed to load the default config file. {e}")
            return False

    def load_config(self):
        try:
            if self.config:
                return self.config
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            return self.config
        except Exception as e:
            self.default_logger.error(f"Failed to load the default or temp config file. {e}")
            return {}
        

def check_upper_lower_limit(config_data: dict, parent):
    if config_data["limit_checked"] is False: return False
    is_upper_limit_effect = len(config_data["upper_limit"]) > 0
    is_lower_limit_effect = len(config_data["lower_limit"]) > 0
    is_limit_effect = is_lower_limit_effect and is_upper_limit_effect
    if is_limit_effect:
        if int(config_data["upper_limit"]) <= int(config_data["lower_limit"]):
            QMessageBox.warning(parent, "设置警告", "上下限配置数据错误，请检查配置!")
            return True
        else:
            return False
    else:
        QMessageBox.warning(parent, "设置警告", "上下限配置数据错误，请检查配置!")
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
    config_manager = ConfigManager(config_file)
    window = SplConfigWindow(config_manager)
    window.show()
    # window = FrConfigWindow(config_manager)
    # window.show()
    # window = HdConfigWindow(config_manager)
    # window.show()
    # window = AIConfigWindow(config_manager)
    # window.show()
    app.exec_()
