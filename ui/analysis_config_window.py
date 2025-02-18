import json
import os
import sys
from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy
from PyQt5.QtWidgets import QSpacerItem, QVBoxLayout, QWidget

from base.log_manager import LogManager
from base.training_model_management import TrainingModelManagement
from consts import error_code
from consts.running_consts import DEFAULT_DIR


class SplConfigWindow(QDialog):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get("SPL", {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setFixedSize(300, 350)
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

    def create_limit(self):
        self.limit_checkbox = QCheckBox("限制", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        self.limit_checkbox.stateChanged.connect(self.on_limit_checkbox_changed)
        self.limit_group_box = QGroupBox("选择限制", self)
        self.limit_group_box.setMinimumSize(180, 180)
        if self.limit_checkbox.isChecked():
            self.limit_group_box.setDisabled(False)
        else:
            self.limit_group_box.setDisabled(True)
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
        elif self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(False)
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)

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

        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.label_upper)
        upper_layout.addWidget(self.line_edit_upper)
        upper_layout.addWidget(self.label_lower)
        upper_layout.addWidget(self.line_edit_lower)
        return upper_layout

    def create_config_dir_layout(self):
        config_dir_label = QLabel("配置文件路径：")
        self.config_dir_box = QLineEdit()
        if not self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(True)
        self.config_dir_box.textChanged.connect(self.get_default_config)
        icon_path = DEFAULT_DIR + "ui/ui_pic/ai_window_pic/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, QLineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self.config_dir_btn_clicked)
        self.config_dir_box.setText(self.load_config.get("config_dir"))
        load_layout = QHBoxLayout()
        load_layout.addWidget(config_dir_label)
        load_layout.addWidget(self.config_dir_box)
        return load_layout

    def on_limit_checkbox_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
        else:
            self.limit_group_box.setDisabled(True)

    def config_dir_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "选择配置文件路径",
                                                   DEFAULT_DIR + "ui/ui_config",
                                                   filter="All Files (*);;")
        if file_path:
            self.config_dir = file_path
            self.config_dir_box.setText(file_path)

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
            "config_dir": self.config_dir_box.text()
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_config("SPL", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class FrConfigWindow(QDialog):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get("FR", {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setFixedSize(300, 350)
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

    def create_limit(self):
        self.limit_checkbox = QCheckBox("限制", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.limit_checkbox.stateChanged.connect(self.on_limit_checkbox_changed)
        self.limit_group_box = QGroupBox("选择限制", self)
        self.limit_group_box.setMinimumSize(180, 180)
        if self.limit_checkbox.isChecked():
            self.limit_group_box.setDisabled(False)
        else:
            self.limit_group_box.setDisabled(True)
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

        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.label_upper)
        upper_layout.addWidget(self.line_edit_upper)
        upper_layout.addWidget(self.label_lower)
        upper_layout.addWidget(self.line_edit_lower)
        return upper_layout

    def create_config_dir_layout(self):
        config_dir_label = QLabel("配置文件路径：")
        self.config_dir_box = QLineEdit()
        if not self.radio_button_2.isChecked():
            self.config_dir_box.setDisabled(True)
        self.config_dir_box.textChanged.connect(self.get_default_config)
        icon_path = DEFAULT_DIR + "ui/ui_pic/ai_window_pic/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, QLineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self.config_dir_btn_clicked)
        self.config_dir_box.setText(self.load_config.get("config_dir"))
        load_layout = QHBoxLayout()
        load_layout.addWidget(config_dir_label)
        load_layout.addWidget(self.config_dir_box)
        return load_layout

    def on_radio_button_toggled(self):
        self.get_default_config()
        if self.radio_button_1.isChecked():
            self.config_dir_box.setDisabled(True)
            self.line_edit_upper.setDisabled(False)
            self.line_edit_lower.setDisabled(False)
        elif self.radio_button_2.isChecked():
            self.line_edit_upper.setDisabled(True)
            self.line_edit_lower.setDisabled(True)
            self.config_dir_box.setDisabled(False)

    def config_dir_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "选择配置文件路径",
                                                   DEFAULT_DIR + "ui/ui_config",
                                                   filter="All Files (*);;")
        if file_path:
            self.config_dir = file_path
            self.config_dir_box.setText(file_path)

    def on_limit_checkbox_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
        else:
            self.limit_group_box.setDisabled(True)

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
            "config_dir": self.config_dir_box.text()
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_config("FR", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class HdConfigWindow(QDialog):

    selected_labels_changed = pyqtSignal()
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get("HD", {})
        self.selected_labels = self.load_config.get("selected_labels", [])
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setFixedSize(300, 350)
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

    def create_harmonic_slider_layout(self):
        harmonic_slider_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(120, 150)
        box_container = QWidget()
        self.box_layout = QVBoxLayout()
        for i in range(2, 31):
            label = QLabel("  " + str(i))
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("QLabel:focus { outline: none; }")
            label.setAutoFillBackground(True)
            label.mousePressEvent = partial(self.on_label_click, label)
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
            self.selected_labels = list(range(2, 31))
            for i in range(self.box_layout.count()):
                label = self.box_layout.itemAt(i).widget()
                text = label.text().strip()
                if not text.startswith("\u2713"):
                    label.setText("\u2713" + text)
        else:
            self.scroll_area.setDisabled(False)
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
        save_flag = self.config_manager.save_config("HD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self.selected_labels:
            PopupUtils().close_popup(self)
        else:
            config_data = self.get_default_config()
            self.accept()
            return config_data


class AIConfigWindow(QDialog):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.model_list = self.load_model_name_from_db()
        self.load_config = self.config_manager.load_config().get("AI", {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setFixedSize(300, 350)
        layout = QVBoxLayout()
        model_box = self.create_model_layout()
        btn_layout = self.create_btn()
        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addWidget(model_box)
        layout.addItem(v_spacer_1)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def create_model_layout(self):
        model_box = QGroupBox("模型")
        model_box.setMinimumSize(150, 150)
        analyse_model_label = QLabel("分析模型:")
        self.analyse_model_combo_box = QComboBox(self)
        for model_name in self.model_list:
            self.analyse_model_combo_box.addItem(model_name)
        self.analyse_model_combo_box.setCurrentText(self.load_config.get("analyse_model_name"))
        self.analyse_model_combo_box.currentTextChanged.connect(self.get_default_config)
        analyse_model_combo_layout = QHBoxLayout()
        analyse_model_combo_layout.addWidget(analyse_model_label)
        analyse_model_combo_layout.addWidget(self.analyse_model_combo_box)
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
        save_flag = self.config_manager.save_config("AI", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class PopupUtils(object):
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
        save_msg.setStandardButtons(QMessageBox.Ok)
        save_msg.exec_()

    @staticmethod
    def close_popup(parent):
        close_msg = QMessageBox(parent)
        close_msg.setIcon(QMessageBox.Warning)
        close_msg.setText("请选择谐波失真阶数")
        close_msg.setWindowTitle("设置警告")
        close_msg.setStandardButtons(QMessageBox.Ok)
        close_msg.exec_()


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

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            return self.config
        except Exception as e:
            self.default_logger.error(f"Failed to load the default config file. {e}")
            return {}


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
