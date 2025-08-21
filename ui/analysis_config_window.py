import json
import os
import sys
from functools import partial
import librosa
import numpy as np
import soundfile as sf

import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator, QCursor
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFormLayout, QFrame, QSplitter, QToolTip
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtWidgets import QButtonGroup

from base.file_ops import FileOps
from base.load_config import ConfigManager, LoadUiConfig
from base.training_model_management import TrainingModelManagement
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.generic_feature_params_dialog import GenericFeatureParamsDialog
from ui.graph_widget import DraggablePlotWidget


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

        layout.addWidget(self.smooth_chk_box)
        layout.addStretch()
        layout.addLayout(limit_layout)
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

        limit_layout = QVBoxLayout()
        limit_layout.addWidget(self.limit_checkbox)
        limit_layout.addStretch()
        limit_layout.addWidget(self.limit_group_box)
        limit_layout.setSpacing(10)
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

        upper_layout = QHBoxLayout()
        upper_layout.addSpacing(19)
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
        load_layout = QHBoxLayout()
        load_layout.addSpacing(10)
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
        self.file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件路径", DEFAULT_DIR + "ui/ui_config", filter="All Files (*);;"
        )
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
            "upper_limit": self.line_edit_upper.text(),
            "lower_limit": self.line_edit_lower.text(),
            "config_dir": self.file_path,
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self):
            return
        save_flag = self.config_manager.save_default_config("SPL", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if check_upper_lower_limit(config_data, self):
            return
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

        upper_layout = QHBoxLayout()
        upper_layout.addSpacing(19)
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
        load_layout = QHBoxLayout()
        load_layout.addSpacing(19)
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
            "upper_limit": self.line_edit_upper.text(),
            "lower_limit": self.line_edit_lower.text(),
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
        harmonic_slider_layout.addStretch()
        harmonic_slider_layout.addWidget(self.select_all_check)
        harmonic_group_box.setLayout(harmonic_slider_layout)
        btn_layout = self.create_btn()
        layout.addWidget(harmonic_group_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
        )

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
        cleaned_label = "".join(filter(str.isdigit, label.text()))
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
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
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
    def __init__(self, config_manager, model_type, signal_len=None):
        super().__init__()
        self.signal_len = signal_len
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
        layout.addWidget(model_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
        )

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
                input_dim = int(query_result_idx[1].split(" ")[0])
                if not self.signal_len:
                    model_list.append(query_result_idx[0])
                else:
                    if input_dim == self.signal_len:
                        model_list.append(query_result_idx[0])
        return model_list

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
        default_config = {"analyse_model_name": self.analyse_model_combo_box.currentText()}
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("AI", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class SpecConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        spec_param_group_box = QGroupBox("频谱图参数配置")
        param_layout = self.create_spec_param()
        spec_param_group_box.setLayout(param_layout)
        btn_layout = self.create_btn()

        layout.addWidget(spec_param_group_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qcombobox_style
        )

    def create_spec_param(self):
        freq_scale_label = QLabel("频率轴类型")
        self.freq_scale_box = QComboBox()
        self.freq_scale_box.addItems(["linear", "log"])
        freq_scale_type = self.load_config.get("freq_scale_type", "linear")
        self.freq_scale_box.setCurrentText(freq_scale_type)
        freq_scale_layout = QHBoxLayout()
        freq_scale_layout.addWidget(freq_scale_label)
        freq_scale_layout.addWidget(self.freq_scale_box)

        fft_size_label = QLabel("FFT 窗长")
        self.fft_size_box = QComboBox()
        fft_sizes = [str(2**i) for i in range(7, 14)]
        self.fft_size_box.addItems(fft_sizes)
        fft_size = str(self.load_config.get("n_fft", 2048))
        self.fft_size_box.setCurrentText(fft_size)
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(fft_size_label)
        fft_layout.addWidget(self.fft_size_box)

        hop_length_label = QLabel("时间步长")
        self.hop_length_box = QComboBox()
        hop_lengths = [str(2**i) for i in range(4, 13)]
        self.hop_length_box.addItems(hop_lengths)
        hop_length = str(self.load_config.get("hop_length", 256))
        self.hop_length_box.setCurrentText(hop_length)
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(hop_length_label)
        hop_layout.addWidget(self.hop_length_box)

        window_func_label = QLabel("窗函数")
        self.window_func_box = QComboBox()
        self.window_func_box.addItems(["hann", "hamming", "blackman"])
        window_func = self.load_config.get("window_func", "hann")
        self.window_func_box.setCurrentText(window_func)
        window_layout = QHBoxLayout()
        window_layout.addWidget(window_func_label)
        window_layout.addWidget(self.window_func_box)

        colormap_label = QLabel("配色")
        self.colormap_box = QComboBox()
        self.colormap_box.addItems(["viridis", "plasma", "magma", "inferno"])
        color_map = self.load_config.get("color_map", "viridis")
        self.colormap_box.setCurrentText(color_map)
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_box)

        param_layout = QVBoxLayout()
        param_layout.addStretch()
        param_layout.addLayout(freq_scale_layout)
        param_layout.addStretch()
        param_layout.addLayout(fft_layout)
        param_layout.addStretch()
        param_layout.addLayout(hop_layout)
        param_layout.addStretch()
        param_layout.addLayout(window_layout)
        param_layout.addStretch()
        param_layout.addLayout(colormap_layout)
        param_layout.addStretch()
        param_layout.addSpacing(10)
        param_layout.setSpacing(10)
        return param_layout

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
            "n_fft": int(self.fft_size_box.currentText()),
            "hop_length": int(self.hop_length_box.currentText()),
            "window_func": self.window_func_box.currentText(),
            "color_map": self.colormap_box.currentText(),
            "freq_scale_type": self.freq_scale_box.currentText(),
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("Spec", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class LPConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(350, 350)
        layout = QVBoxLayout()
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


class PDConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(940, 520)
        self.resize(1000, 560)

        root_layout = QHBoxLayout()

        # left layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_detect_group())
        left_layout.addWidget(self.create_test_group())
        left_layout.addStretch()

        # advanced mode: always hidden when entering PD config
        self.advanced_visible = False
        self.btn_toggle_advanced = QPushButton("高级模式 >>>")
        self.btn_toggle_advanced.clicked.connect(self.on_toggle_advanced_mode)
        left_layout.addWidget(self.btn_toggle_advanced)

        left_layout.addLayout(self.create_btn_layout())
        left_layout.setSpacing(10)

        # right layout
        self.advanced_panel = self.create_advanced_group()
        # set the minimum width of the advanced panel to be larger, to avoid being compressed after opening
        try:
            self.advanced_panel.setMinimumWidth(360)
        except Exception:
            pass
        self.advanced_panel.setVisible(self.advanced_visible)

        root_layout.addLayout(left_layout)
        root_layout.addWidget(self.advanced_panel)
        root_layout.setSpacing(12)
        self.setLayout(root_layout)

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcombobox_style
        )
        # adapt the size according to the visibility of the panel
        self.adjustSize()

    def create_preprocess_group(self):
        group_box = QGroupBox("预处理选项")
        vbox = QVBoxLayout()

        # filter (two rows: main parameters + order)
        row_filter_main = QHBoxLayout()
        self.chk_filter = QCheckBox("滤波")
        self.chk_filter.setChecked(self.load_config.get("filter_enabled", False))
        self.chk_filter.stateChanged.connect(self.get_default_config)
        row_filter_main.addWidget(self.chk_filter)
        row_filter_main.addStretch()
        row_filter_main.addWidget(QLabel("范围(Hz):"))
        self.edit_filter_ranges = QLineEdit()
        self.edit_filter_ranges.setPlaceholderText("0,300; 700,1000;")
        self.edit_filter_ranges.setText(self.load_config.get("filter_ranges", ""))
        self.edit_filter_ranges.textChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.edit_filter_ranges)
        row_filter_main.addWidget(QLabel("类型:"))
        self.combo_filter_type = QComboBox()
        self.combo_filter_type.addItems(["带通", "带阻"])
        self.combo_filter_type.setCurrentIndex(0 if self.load_config.get("filter_type", "bandpass") == "bandpass" else 1)
        self.combo_filter_type.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.combo_filter_type)

        row_filter_order = QHBoxLayout()
        row_filter_order.addStretch()
        row_filter_order.addWidget(QLabel("阶数"))
        self.spin_filter_order = QSpinBox()
        self.spin_filter_order.setRange(1, 20)
        self.spin_filter_order.setValue(int(self.load_config.get("filter_order", 4)))
        self.spin_filter_order.valueChanged.connect(lambda _: self.get_default_config())
        row_filter_order.addWidget(self.spin_filter_order)

        # smooth (two rows: main parameters + algorithm)
        row_smooth_main = QHBoxLayout()
        self.chk_smooth = QCheckBox("平滑")
        self.chk_smooth.setChecked(self.load_config.get("smooth_enabled", False))
        self.chk_smooth.stateChanged.connect(self.get_default_config)
        row_smooth_main.addWidget(self.chk_smooth)
        row_smooth_main.addStretch()
        row_smooth_main.addWidget(QLabel("单位:"))
        self.combo_smooth_unit = QComboBox()
        self.combo_smooth_unit.addItems(["时间(秒)", "格点数"])
        self.combo_smooth_unit.setCurrentIndex(0 if self.load_config.get("smooth_unit", "time") == "time" else 1)
        self.combo_smooth_unit.currentIndexChanged.connect(lambda _: (self._update_smooth_unit_enabled(), self.get_default_config()))
        row_smooth_main.addWidget(self.combo_smooth_unit)
        self.spin_smooth_time = QDoubleSpinBox()
        self.spin_smooth_time.setRange(0.00, 999.00)
        self.spin_smooth_time.setDecimals(4)
        self.spin_smooth_time.setSingleStep(0.01)
        self.spin_smooth_time.setValue(float(self.load_config.get("smooth_time_sec", 0.02)))
        self.spin_smooth_time.valueChanged.connect(lambda _: self.get_default_config())
        row_smooth_main.addWidget(self.spin_smooth_time)
        self.spin_smooth_points = QSpinBox()
        self.spin_smooth_points.setRange(1, 99999)
        self.spin_smooth_points.setValue(int(self.load_config.get("smooth_points", 0)))
        self.spin_smooth_points.valueChanged.connect(lambda _: self.get_default_config())
        row_smooth_main.addWidget(self.spin_smooth_points)

        row_smooth_algo = QHBoxLayout()
        row_smooth_algo.addStretch()
        row_smooth_algo.addWidget(QLabel("平滑算法:"))
        self.group_smooth_algo = QButtonGroup(self)
        self.rb_algo1 = QRadioButton("平均")
        self.rb_algo2 = QRadioButton("Golay")
        self.rb_algo3 = QRadioButton("Gaussian")
        row_smooth_algo.addWidget(self.rb_algo1)
        row_smooth_algo.addWidget(self.rb_algo2)
        row_smooth_algo.addWidget(self.rb_algo3)
        self.group_smooth_algo.addButton(self.rb_algo1, 1)
        self.group_smooth_algo.addButton(self.rb_algo2, 2)
        self.group_smooth_algo.addButton(self.rb_algo3, 3)
        algo_saved = int(self.load_config.get("smooth_algo", 1))
        if algo_saved == 2:
            self.rb_algo2.setChecked(True)
        elif algo_saved == 3:
            self.rb_algo3.setChecked(True)
        else:
            self.rb_algo1.setChecked(True)
        self.group_smooth_algo.buttonClicked.connect(lambda _: self.get_default_config())

        # SPL calculation window length (no check box, default enabled; support time/grid point number)
        row_splwin = QHBoxLayout()
        row_splwin.addWidget(QLabel("SPL计算窗长"))
        row_splwin.addStretch()
        row_splwin.addWidget(QLabel("单位:"))
        self.combo_spl_window_unit = QComboBox()
        self.combo_spl_window_unit.addItems(["时间(秒)", "格点数"])
        self.combo_spl_window_unit.setCurrentIndex(0 if self.load_config.get("spl_window_unit", "time") == "time" else 1)
        self.combo_spl_window_unit.currentIndexChanged.connect(lambda _: (self._update_spl_window_unit_enabled(), self.get_default_config()))
        row_splwin.addWidget(self.combo_spl_window_unit)
        self.spin_spl_window_time = QDoubleSpinBox()
        self.spin_spl_window_time.setRange(0.000, 999.000)
        self.spin_spl_window_time.setDecimals(4)
        self.spin_spl_window_time.setSingleStep(0.001)
        self.spin_spl_window_time.setValue(float(self.load_config.get("spl_window_time_sec", 0.050)))
        self.spin_spl_window_time.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_time)
        self.spin_spl_window_points = QSpinBox()
        self.spin_spl_window_points.setRange(1, 99999)
        self.spin_spl_window_points.setValue(int(self.load_config.get("spl_window_points", 0)))
        self.spin_spl_window_points.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_points)

        vbox.addLayout(row_filter_main)
        vbox.addLayout(row_filter_order)
        # place the SPL calculation window length between the filter and smooth
        vbox.addLayout(row_splwin)
        vbox.addLayout(row_smooth_main)
        vbox.addLayout(row_smooth_algo)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        # initialize the display state
        self._update_smooth_unit_enabled()
        self._update_spl_window_unit_enabled()
        return group_box

    def create_detect_group(self):
        group_box = QGroupBox("峰值提取参数")
        vbox = QVBoxLayout()

        # peak count
        row_count = QHBoxLayout()
        self.chk_peak_count = QCheckBox("峰值个数")
        self.chk_peak_count.setChecked(self.load_config.get("peak_count_enabled", True))
        self.chk_peak_count.stateChanged.connect(self.get_default_config)
        row_count.addWidget(self.chk_peak_count)
        row_count.addStretch()
        row_count.addWidget(QLabel("最大峰数目:"))
        self.spin_peak_count = QSpinBox()
        self.spin_peak_count.setRange(1, 9999)
        self.spin_peak_count.setValue(int(self.load_config.get("peak_count", 3)))
        self.spin_peak_count.valueChanged.connect(lambda _: self.get_default_config())
        row_count.addWidget(self.spin_peak_count)
        # row_count.addWidget(QLabel("个"))

        # peak size
        row_size = QHBoxLayout()
        self.chk_peak_size = QCheckBox("峰值大小")
        self.chk_peak_size.setChecked(self.load_config.get("peak_size_enabled", True))
        self.chk_peak_size.stateChanged.connect(self.get_default_config)
        row_size.addWidget(self.chk_peak_size)
        row_size.addStretch()
        row_size.addWidget(QLabel("单位:"))
        self.combo_peak_size_unit = QComboBox()
        self.combo_peak_size_unit.addItems(["rmsV", "dBL"])
        peak_size_unit_saved = self.load_config.get("peak_size_unit", "db")
        self.combo_peak_size_unit.setCurrentIndex(0 if peak_size_unit_saved == "rms" else 1)
        self.combo_peak_size_unit.currentIndexChanged.connect(lambda _: (self._update_peak_units(), self.get_default_config()))
        row_size.addWidget(self.combo_peak_size_unit)
        self.spin_peak_size = QDoubleSpinBox()
        self.spin_peak_size.setRange(-100.0, 200.0)
        self.spin_peak_size.setDecimals(2)
        self.spin_peak_size.setSingleStep(1.0)
        self.spin_peak_size.setValue(float(self.load_config.get("peak_min_value", 100.0)))
        self.spin_peak_size.valueChanged.connect(lambda _: self.get_default_config())
        row_size.addWidget(self.spin_peak_size)

        # peak slope
        row_slope = QHBoxLayout()
        self.chk_peak_slope = QCheckBox("峰凸起度")
        self.chk_peak_slope.setChecked(self.load_config.get("peak_slope_enabled", False))
        self.chk_peak_slope.stateChanged.connect(self.get_default_config)
        row_slope.addWidget(self.chk_peak_slope)
        row_slope.addStretch()
        row_slope.addWidget(QLabel("单位:"))
        self.combo_peak_slope_unit = QComboBox()
        self.combo_peak_slope_unit.addItems(["rmsV", "dBL"])
        peak_slope_unit_saved = self.load_config.get("peak_slope_unit", "db")
        self.combo_peak_slope_unit.setCurrentIndex(0 if peak_slope_unit_saved == "rms" else 1)
        self.combo_peak_slope_unit.currentIndexChanged.connect(lambda _: (self._update_peak_units(), self.get_default_config()))
        row_slope.addWidget(self.combo_peak_slope_unit)
        self.spin_peak_slope = QDoubleSpinBox()
        self.spin_peak_slope.setRange(0.0, 200.0)
        self.spin_peak_slope.setDecimals(3)
        self.spin_peak_slope.setSingleStep(1.0)
        self.spin_peak_slope.setValue(float(self.load_config.get("peak_min_slope", 100.0)))
        self.spin_peak_slope.valueChanged.connect(lambda _: self.get_default_config())
        row_slope.addWidget(self.spin_peak_slope)

        # minimum peak distance (support time/grid point number)
        row_nms = QHBoxLayout()
        self.chk_nms = QCheckBox("最小峰间距")
        self.chk_nms.setChecked(self.load_config.get("nms_enabled", False))
        self.chk_nms.stateChanged.connect(self.get_default_config)
        row_nms.addWidget(self.chk_nms)
        row_nms.addStretch()
        row_nms.addWidget(QLabel("单位:"))
        self.combo_nms_unit = QComboBox()
        self.combo_nms_unit.addItems(["时间(秒)", "格点数"])
        self.combo_nms_unit.setCurrentIndex(0 if self.load_config.get("nms_unit", "time") == "time" else 1)
        self.combo_nms_unit.currentIndexChanged.connect(lambda _: (self._update_nms_unit_enabled(), self.get_default_config()))
        row_nms.addWidget(self.combo_nms_unit)
        self.spin_nms_time = QDoubleSpinBox()
        self.spin_nms_time.setRange(0.00, 100.00)
        self.spin_nms_time.setDecimals(3)
        self.spin_nms_time.setSingleStep(0.01)
        self.spin_nms_time.setValue(float(self.load_config.get("nms_time_sec", 0.50)))
        self.spin_nms_time.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_time)
        self.spin_nms_points = QSpinBox()
        self.spin_nms_points.setRange(1, 99999)
        self.spin_nms_points.setValue(int(self.load_config.get("nms_points", 0)))
        self.spin_nms_points.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_points)

        # duration
        row_duration = QHBoxLayout()
        self.chk_duration = QCheckBox("峰持续时间")
        self.chk_duration.setChecked(self.load_config.get("duration_enabled", False))
        self.chk_duration.stateChanged.connect(self.get_default_config)
        row_duration.addWidget(self.chk_duration)
        row_duration.addStretch()
        row_duration.addWidget(QLabel("最短"))
        self.spin_duration_min = QDoubleSpinBox()
        self.spin_duration_min.setRange(0.0, 1000.0)
        self.spin_duration_min.setDecimals(3)
        self.spin_duration_min.setSingleStep(0.001)
        self.spin_duration_min.setValue(float(self.load_config.get("duration_min", 0.0)))
        self.spin_duration_min.valueChanged.connect(lambda _: self.get_default_config())
        row_duration.addWidget(self.spin_duration_min)
        row_duration.addWidget(QLabel("最长"))
        self.spin_duration_max = QDoubleSpinBox()
        self.spin_duration_max.setRange(0.0, 1000.0)
        self.spin_duration_max.setDecimals(3)
        self.spin_duration_max.setSingleStep(0.001)
        self.spin_duration_max.setValue(float(self.load_config.get("duration_max", 0.0)))
        self.spin_duration_max.valueChanged.connect(lambda _: self.get_default_config())
        row_duration.addWidget(self.spin_duration_max)

        vbox.addLayout(row_count)
        vbox.addLayout(row_size)
        vbox.addLayout(row_slope)
        vbox.addLayout(row_nms)
        vbox.addLayout(row_duration)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        # initialize the display state
        self._update_nms_unit_enabled()
        self._update_peak_units()
        return group_box

    def create_test_group(self):
        group_box = QGroupBox("测试选项")
        vbox = QVBoxLayout()

        row_peak_condition = QHBoxLayout()
        row_peak_condition.addWidget(QLabel("峰值点数目"))
        row_peak_condition.addStretch()
        self.combo_test_peak_op = QComboBox()
        self.combo_test_peak_op.addItems([">", "<", "=", "≥", "≤"])
        saved_op = self.load_config.get("test_peak_op", "≥")
        try:
            idx = [">", "<", "=", "≥", "≤"].index(saved_op)
        except ValueError:
            idx = 3
        self.combo_test_peak_op.setCurrentIndex(idx)
        self.combo_test_peak_op.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_peak_condition.addWidget(self.combo_test_peak_op)
        self.spin_test_peak_value = QSpinBox()
        self.spin_test_peak_value.setRange(0, 1000000)
        self.spin_test_peak_value.setValue(int(self.load_config.get("test_peak_value", 3)))
        self.spin_test_peak_value.valueChanged.connect(lambda _: self.get_default_config())
        row_peak_condition.addWidget(self.spin_test_peak_value)

        vbox.addLayout(row_peak_condition)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        return group_box

    def create_advanced_group(self):
        adv_group = QGroupBox("高级选项")
        adv_layout = QVBoxLayout()
        adv_layout.addWidget(self.create_preprocess_group())

        row_convex_len = QHBoxLayout()
        row_convex_len.addWidget(QLabel("峰凸起度计算窗口"))
        row_convex_len.addStretch(1)
        row_convex_len.addWidget(QLabel("单位:"))
        self.combo_convex_unit = QComboBox()
        self.combo_convex_unit.addItems(["音频长度", "格点数", "时长(秒)"])
        self.combo_convex_unit.setCurrentIndex({"audio":0, "points":1, "time":2}.get(self.load_config.get("convex_unit", "audio"), 0))
        self.combo_convex_unit.currentIndexChanged.connect(lambda _: (self._update_convex_unit_enabled(), self.get_default_config()))
        row_convex_len.addWidget(self.combo_convex_unit)
        self.spin_convex_audio_ratio = QDoubleSpinBox()
        self.spin_convex_audio_ratio.setRange(0.001, 1.000)
        self.spin_convex_audio_ratio.setDecimals(4)
        self.spin_convex_audio_ratio.setSingleStep(0.01)
        self.spin_convex_audio_ratio.setValue(float(self.load_config.get("convex_audio_ratio", 1.0)))
        self.spin_convex_audio_ratio.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_audio_ratio)
        self.spin_convex_points = QSpinBox()
        self.spin_convex_points.setRange(1, 100000000)
        self.spin_convex_points.setValue(int(self.load_config.get("convex_points", 1024)))
        self.spin_convex_points.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_points)
        self.spin_convex_time = QDoubleSpinBox()
        self.spin_convex_time.setRange(0.000, 999.000)
        self.spin_convex_time.setDecimals(3)
        self.spin_convex_time.setSingleStep(0.1)
        self.spin_convex_time.setValue(float(self.load_config.get("convex_time_sec", 0.0)))
        self.spin_convex_time.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_time)

        # 峰持续时间参考点（始终启用）
        row_dmode = QHBoxLayout()
        row_dmode.addWidget(QLabel("峰持续时间参考点"))
        row_dmode.addStretch(1)
        row_dmode.addWidget(QLabel("单位:"))

        ref_unit_saved = self.load_config.get("duration_ref_unit", "peak")
        ref_value_saved = float(self.load_config.get("duration_ref_value", 0.50 if ref_unit_saved == "peak" else 100.0))

        self.combo_duration_ref_unit = QComboBox()
        self.combo_duration_ref_unit.addItems(["Vpeak", "dBL"])
        self.combo_duration_ref_unit.setCurrentIndex(0 if ref_unit_saved == "peak" else 1)
        self.combo_duration_ref_unit.currentIndexChanged.connect(lambda _: (self._update_duration_ref_unit(), self.get_default_config()))
        row_dmode.addWidget(self.combo_duration_ref_unit)

        self.spin_duration_ref = QDoubleSpinBox()
        self.spin_duration_ref.setDecimals(2)
        self.spin_duration_ref.setSingleStep(0.01)
        row_dmode.addWidget(self.spin_duration_ref)
        # set the range/step according to the unit, and fill the value
        self._update_duration_ref_unit()
        self.spin_duration_ref.setValue(float(ref_value_saved))
        self.spin_duration_ref.valueChanged.connect(lambda _: self.get_default_config())

        adv_layout.addLayout(row_convex_len)
        adv_layout.addLayout(row_dmode)
        adv_layout.addStretch()
        adv_layout.setSpacing(10)
        adv_layout.setContentsMargins(10, 20, 10, 20)
        adv_group.setLayout(adv_layout)
        adv_group.setMinimumWidth(260)

        self._update_convex_unit_enabled()
        return adv_group

    def on_toggle_advanced_mode(self):
        # switch only affects the display, not the configuration selection semantics
        self.advanced_visible = not getattr(self, "advanced_visible", False)
        self.advanced_panel.setVisible(self.advanced_visible)
        self.btn_toggle_advanced.setText("高级模式 <<<" if self.advanced_visible else "高级模式 >>>")
        if self.advanced_visible:
            self.setMinimumWidth(940)
        else:
            self.setMinimumWidth(820)
        self.adjustSize()

    def create_btn_layout(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        ok_btn = QPushButton(" 确  认 ")
        default_btn.clicked.connect(self.on_click_default_btn)
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            # preprocess(advanced)
            "filter_enabled": self.chk_filter.isChecked(),
            "filter_ranges": self.edit_filter_ranges.text().strip(),
            "filter_type": "bandpass" if self.combo_filter_type.currentIndex() == 0 else "bandstop",
            "smooth_enabled": self.chk_smooth.isChecked(),
            "smooth_unit": "time" if self.combo_smooth_unit.currentIndex() == 0 else "points",
            "smooth_time_sec": float(self.spin_smooth_time.value()),
            "smooth_points": int(self.spin_smooth_points.value()),

            # based parameter
            "peak_count_enabled": self.chk_peak_count.isChecked(),
            "peak_count": int(self.spin_peak_count.value()),
            "peak_size_enabled": self.chk_peak_size.isChecked(),
            "peak_size_unit": ("rms" if self.combo_peak_size_unit.currentIndex() == 0 else "db"),
            "peak_min_value": float(self.spin_peak_size.value()),
            "peak_slope_enabled": self.chk_peak_slope.isChecked(),
            "peak_slope_unit": ("rms" if self.combo_peak_slope_unit.currentIndex() == 0 else "db"),
            "peak_min_slope": float(self.spin_peak_slope.value()),
            "nms_enabled": self.chk_nms.isChecked(),
            "nms_unit": "time" if self.combo_nms_unit.currentIndex() == 0 else "points",
            "nms_time_sec": float(self.spin_nms_time.value()),
            "nms_points": int(self.spin_nms_points.value()),
            "spl_window_unit": ("time" if self.combo_spl_window_unit.currentIndex() == 0 else "points"),
            "spl_window_time_sec": float(self.spin_spl_window_time.value()),
            "spl_window_points": int(self.spin_spl_window_points.value()),
            "duration_enabled": self.chk_duration.isChecked(),
            "duration_min": float(self.spin_duration_min.value()),
            "duration_max": float(self.spin_duration_max.value()),

            # advanced mode
            "advanced_mode": bool(getattr(self, "advanced_visible", False)),
            "filter_order": int(self.spin_filter_order.value()),
            "smooth_algo": int(self.group_smooth_algo.checkedId() or 1),
            "convex_unit": ("audio" if self.combo_convex_unit.currentIndex() == 0 else ("points" if self.combo_convex_unit.currentIndex() == 1 else "time")),
            "convex_audio_ratio": float(self.spin_convex_audio_ratio.value()),
            "convex_points": int(self.spin_convex_points.value()),
            "convex_time_sec": float(self.spin_convex_time.value()),
            "duration_ref_unit": ("peak" if self.combo_duration_ref_unit.currentIndex() == 0 else "db"),
            "duration_ref_value": float(self.spin_duration_ref.value()),

            # test option
            "test_peak_op": self.combo_test_peak_op.currentText(),
            "test_peak_value": int(self.spin_test_peak_value.value()),
        }
        return default_config

    def _update_duration_ref_unit(self):
        # adjust the range and precision according to the selected unit
        is_peak_unit = self.combo_duration_ref_unit.currentIndex() == 0
        if is_peak_unit:
            self.spin_duration_ref.setRange(0.00, 1.00)
            self.spin_duration_ref.setDecimals(2)
            self.spin_duration_ref.setSingleStep(0.01)
        else:
            self.spin_duration_ref.setRange(-200.0, 500.0)
            self.spin_duration_ref.setDecimals(1)
            self.spin_duration_ref.setSingleStep(1.0)

    def _update_smooth_unit_enabled(self):
        is_time = self.combo_smooth_unit.currentIndex() == 0
        self.spin_smooth_time.setVisible(is_time)
        self.spin_smooth_points.setVisible(not is_time)

    def _update_nms_unit_enabled(self):
        is_time = self.combo_nms_unit.currentIndex() == 0
        self.spin_nms_time.setVisible(is_time)
        self.spin_nms_points.setVisible(not is_time)

    def _update_spl_window_unit_enabled(self):
        is_time = self.combo_spl_window_unit.currentIndex() == 0
        self.spin_spl_window_time.setVisible(is_time)
        self.spin_spl_window_points.setVisible(not is_time)

    def _update_convex_unit_enabled(self):
        idx = self.combo_convex_unit.currentIndex()
        is_audio = idx == 0
        is_points = idx == 1
        is_time = idx == 2
        self.spin_convex_audio_ratio.setVisible(is_audio)
        self.spin_convex_points.setVisible(is_points)
        self.spin_convex_time.setVisible(is_time)

    def _update_peak_units(self):
        is_db_for_size = self.combo_peak_size_unit.currentIndex() == 1
        if is_db_for_size:
            self.spin_peak_size.setRange(-200.0, 500.0)
            self.spin_peak_size.setDecimals(1)
            self.spin_peak_size.setSingleStep(1.0)
        else:
            # rmsV/Vmax -> 0~1 decimal
            self.spin_peak_size.setRange(0.0, 1.0)
            self.spin_peak_size.setDecimals(3)
            self.spin_peak_size.setSingleStep(0.001)
            v = float(self.spin_peak_size.value())
            if v < 0.0:
                self.spin_peak_size.setValue(0.0)
            elif v > 1.0:
                self.spin_peak_size.setValue(1.0)

        # peak slope
        is_db_for_slope = self.combo_peak_slope_unit.currentIndex() == 1
        if is_db_for_slope:
            self.spin_peak_slope.setRange(0.0, 10000.0)
            self.spin_peak_slope.setDecimals(1)
            self.spin_peak_slope.setSingleStep(1.0)
        else:
            self.spin_peak_slope.setRange(0.0, 1.0)
            self.spin_peak_slope.setDecimals(3)
            self.spin_peak_slope.setSingleStep(0.001)
            v2 = float(self.spin_peak_slope.value())
            if v2 < 0.0:
                self.spin_peak_slope.setValue(0.0)
            elif v2 > 1.0:
                self.spin_peak_slope.setValue(1.0)

    def on_click_default_btn(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("PD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class PatternMatchConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        _, self.feature_registry = self.load_features_param_config()
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.audio_file_path = None
        self.pattern_save_path = None
        self.audio_data = None
        self.sample_rate = None
        self.selected_region_time = (None, None)
        self.config_data = None

        self.feature_params = {
            key: {p_name: p_def['default'] for p_name, p_def in info['params'].items()}
            for key, info in self.feature_registry.items() if info.get('params')
        }

        self.init_ui()
        self.on_strategy_radio_changed()
        self.on_filter_toggled()
        self.on_feature_type_changed()
        self.populate_ui_from_config()

    def init_ui(self):
        self.setWindowTitle("模式匹配参数配置")
        self.setMinimumSize(800, 750)
        self.resize(800, 750)
        self.main_layout = QVBoxLayout(self)

        self.main_layout.addLayout(self.create_upload_layout())

        splitter = QSplitter(Qt.Vertical)
        self.plot_widget = self.create_plot_widget()
        splitter.addWidget(self.plot_widget)

        options_container = QWidget()
        options_layout = self.create_options_layout()
        options_container.setLayout(options_layout)
        splitter.addWidget(options_container)

        splitter.setSizes([450, 300])
        splitter.setCollapsible(0, False)
        self.main_layout.addWidget(splitter)

        btn_layout = self.create_btn_layout()
        self.main_layout.addLayout(btn_layout)

        self.setLayout(self.main_layout)
        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qdialog_style
            + ui_style_const.qframe_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qtextedit_style
        )

    def create_upload_layout(self):
        layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("请上传源音频文件...")

        upload_btn = QPushButton("上传文件")
        upload_btn.clicked.connect(self.upload_audio_file)

        self.select_path_btn = QPushButton("保存模板")
        self.select_path_btn.clicked.connect(self.select_pattern_save_path)
        self.select_path_btn.setToolTip("尚未选择模板保存路径")

        file_label = QLabel("文件操作:")
        file_label.setStyleSheet("color: black;")
        layout.addWidget(file_label)
        layout.addWidget(self.file_path_edit)
        layout.addWidget(upload_btn)
        layout.addWidget(self.select_path_btn)

        return layout

    def create_plot_widget(self):
        self.plot_curve = pg.PlotDataItem(pen='k')
        self.region = pg.LinearRegionItem(values=[0, 0], brush=(50, 150, 250, 50),
                                          pen={'color': (0, 0, 255), 'width': 2})
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.on_region_changed)

        plot_widget = DraggablePlotWidget(region_item=self.region)
        plot_widget.setBackground("white")
        plot_widget.setLabel("left", "Amplitude(V)", **{"font-size": "20px"})
        plot_widget.setLabel("bottom", "Time(s)", **{"font-size": "20px"})
        plot_widget.showGrid(x=True, y=True, alpha=0.5)
        plot_widget.addItem(self.plot_curve)
        plot_widget.addItem(self.region)
        self.region.hide()

        plot_widget.sigSelectionCancelled.connect(self.on_selection_cancelled)
        plot_widget.viewport().setMouseTracking(True)
        plot_widget.viewport().installEventFilter(self)
        plot_widget.setActive(False)
        return plot_widget

    def create_options_layout(self):
        layout = QHBoxLayout()
        processing_feature_group = self.create_processing_and_feature_group()
        strategy_group = self.create_strategy_group()
        layout.addWidget(processing_feature_group)
        layout.addWidget(strategy_group)
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
        return layout

    def create_processing_and_feature_group(self):
        group = QGroupBox("特征与预处理")
        layout = QVBoxLayout()

        feature_label = QLabel("<b>特征类型</b>")
        feature_label.setStyleSheet("color: black;")
        layout.addWidget(feature_label)

        combo_layout = QHBoxLayout()
        self.feature_combo = QComboBox()
        for key, info in self.feature_registry.items():
            self.feature_combo.addItem(info['display_name'], userData=key)
        self.feature_combo.currentIndexChanged.connect(self.on_feature_type_changed)
        combo_layout.addWidget(self.feature_combo)

        self.feature_params_btn = QPushButton("特征参数")
        self.feature_params_btn.clicked.connect(self.on_click_feature_params)
        combo_layout.addWidget(self.feature_params_btn)
        layout.addLayout(combo_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        filter_label = QLabel("<b>带阻滤波</b>")
        filter_label.setStyleSheet("color: black;")
        layout.addWidget(filter_label)
        self.filter_checkbox = QCheckBox("启用")
        self.filter_checkbox.toggled.connect(self.on_filter_toggled)
        layout.addWidget(self.filter_checkbox)

        filter_range_layout = QFormLayout()
        self.low_freq_edit = QLineEdit("0")
        self.low_freq_edit.setValidator(QIntValidator(0, 20000, self))
        self.high_freq_edit = QLineEdit("5000")
        self.high_freq_edit.setValidator(QIntValidator(0, 20000, self))
        low_freq_label = QLabel("最低频率 (Hz):")
        high_freq_label = QLabel("最高频率 (Hz):")
        low_freq_label.setStyleSheet("color: black;")
        high_freq_label.setStyleSheet("color: black;")
        filter_range_layout.addRow(low_freq_label, self.low_freq_edit)
        filter_range_layout.addRow(high_freq_label, self.high_freq_edit)

        layout.addLayout(filter_range_layout)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_strategy_group(self):
        group = QGroupBox("匹配策略")
        main_layout = QVBoxLayout()

        metric_layout = QHBoxLayout()
        metric_label = QLabel("<b>相似度度量:</b>")
        metric_label.setStyleSheet("color: black;")
        self.similarity_metric_combo = QComboBox()
        self.similarity_metric_combo.addItem("欧氏距离 (Euclidean)", "euclidean")
        self.similarity_metric_combo.addItem("余弦相似度 (Cosine)", "cosine")
        metric_layout.addWidget(metric_label)
        metric_layout.addWidget(self.similarity_metric_combo)
        main_layout.addLayout(metric_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        return_label = QLabel("<b>匹配点返回策略:</b>")
        return_label.setStyleSheet("color: black;")
        main_layout.addWidget(return_label)
        fixed_threshold_layout = QHBoxLayout()
        self.fixed_threshold_radio = QRadioButton("固定阈值:")
        self.fixed_threshold_radio.setChecked(True)
        self.fixed_threshold_radio.toggled.connect(self.on_strategy_radio_changed)
        self.threshold_edit = QLineEdit("0.9")
        self.threshold_edit.setValidator(QDoubleValidator(0.0, 100, 2, self))
        fixed_threshold_layout.addWidget(self.fixed_threshold_radio)
        fixed_threshold_layout.addWidget(self.threshold_edit)
        self.adaptive_threshold_radio = QRadioButton("自适应阈值")
        self.adaptive_threshold_radio.toggled.connect(self.on_strategy_radio_changed)
        main_layout.addLayout(fixed_threshold_layout)
        main_layout.addWidget(self.adaptive_threshold_radio)
        main_layout.addStretch()
        group.setLayout(main_layout)
        return group

    def create_btn_layout(self):
        layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        default_btn = QPushButton("设为默认")
        default_btn.clicked.connect(self.on_click_default_btn)
        layout.addWidget(default_btn)
        layout.addStretch()
        layout.addWidget(ok_btn)
        return layout

    @staticmethod
    def load_features_param_config():
        default_config_file = os.path.join(DEFAULT_DIR, "ui", "ui_config", "features_param.json")
        code, data = LoadUiConfig.load_data_from_json(default_config_file)
        if code == 0:
            return True, data
        else:
            return False, {}

    def select_pattern_save_path(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存模板", "", "WAV 文件 (*.wav)")
        if save_path:
            self.pattern_save_path = save_path
            self.select_path_btn.setToolTip(f"模板将保存到:\n{self.pattern_save_path}")

    def upload_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.wav)")
        if file_path:
            self.audio_file_path = file_path
            self.file_path_edit.setText(file_path)
            self.load_and_display_waveform(file_path)
            self.region.hide()
            self.selected_region_time = (None, None)
            self.pattern_save_path = None

    def on_region_changed(self):
        self.pattern_save_path = None
        start_time, end_time = self.region.getRegion()
        duration = abs(end_time - start_time)
        if self.region.isVisible() and duration > 1e-9:
            self.selected_region_time = tuple(sorted((start_time, end_time)))
            tooltip_text = f"时长: {duration:.3f}s"
            QToolTip.showText(QCursor.pos(), tooltip_text, self.plot_widget.viewport())
        else:
            self.selected_region_time = (None, None)
            QToolTip.hideText()

    def on_selection_cancelled(self):
        self.selected_region_time = (None, None)
        QToolTip.hideText()

    def on_click_feature_params(self):
        feature_key = self.feature_combo.currentData()
        if not feature_key or not self.feature_registry[feature_key].get('params'):
            QMessageBox.information(self, "提示", "当前特征类型没有可配置的参数。")
            return

        param_definitions = self.feature_registry[feature_key]['params']
        current_values = self.feature_params.get(feature_key, {})
        dialog = GenericFeatureParamsDialog(param_definitions, current_values)
        if dialog.exec_() == QDialog.Accepted:
            self.feature_params[feature_key] = dialog.get_params()

    def on_feature_type_changed(self):
        feature_key = self.feature_combo.currentData()
        has_params = feature_key and self.feature_registry[feature_key].get('params')
        self.feature_params_btn.setEnabled(bool(has_params))

    def on_strategy_radio_changed(self):
        is_fixed_checked = self.fixed_threshold_radio.isChecked()
        self.threshold_edit.setEnabled(is_fixed_checked)

    def on_filter_toggled(self):
        is_checked = self.filter_checkbox.isChecked()
        self.low_freq_edit.setEnabled(is_checked)
        self.high_freq_edit.setEnabled(is_checked)

    def populate_ui_from_config(self):
        if not self.load_config:
            return
        rel_audio_path = self.load_config.get("audio_file_path")
        if rel_audio_path:
            abs_audio_path = os.path.join(DEFAULT_DIR, rel_audio_path)
            if os.path.exists(abs_audio_path):
                self.audio_file_path = abs_audio_path
                self.file_path_edit.setText(self.audio_file_path)
                self.load_and_display_waveform(self.audio_file_path)
            else:
                self.file_path_edit.setText(f"未找到: {abs_audio_path}")

        self.pattern_save_path = self.load_config.get("pattern_save_path")
        if self.pattern_save_path:
            self.select_path_btn.setToolTip(f"模板将保存到:\n{self.pattern_save_path}")

        feature_type = self.load_config.get("feature_type")
        if feature_type:
            index = self.feature_combo.findData(feature_type)
            if index >= 0:
                self.feature_combo.setCurrentIndex(index)

        feature_params = self.load_config.get("feature_params")
        if feature_params and feature_type in self.feature_params:
            self.feature_params[feature_type] = feature_params

        if self.load_config.get("apply_filter"):
            self.filter_checkbox.setChecked(True)
            filter_range = self.load_config.get("filter_range_hz", [0, 5000])
            self.low_freq_edit.setText(str(filter_range[0]))
            self.high_freq_edit.setText(str(filter_range[1]))

        metric = self.load_config.get("similarity_metric")
        if metric:
            index = self.similarity_metric_combo.findData(metric)
            if index >= 0:
                self.similarity_metric_combo.setCurrentIndex(index)

        strategy = self.load_config.get("threshold_strategy")
        if strategy == "adaptive_threshold":
            self.adaptive_threshold_radio.setChecked(True)
        else:
            self.fixed_threshold_radio.setChecked(True)
            threshold_value = self.load_config.get("threshold_value", 0.9)
            self.threshold_edit.setText(str(threshold_value))

        region_time = self.load_config.get("pattern_region_time")
        if self.audio_data is not None and region_time and all(t is not None for t in region_time):
            try:
                start_frame, end_frame = region_time
                start_time = start_frame / self.sample_rate
                end_time = end_frame / self.sample_rate
                self.region.show()
                self.region.setRegion((start_time, end_time))
            except (ValueError, TypeError, ZeroDivisionError) as e:
                QMessageBox.critical(self, "错误", f"无法从配置中恢复模板区域: \n{e}")

    def load_and_display_waveform(self, file_path):
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            self.audio_data, self.sample_rate = audio_data, sample_rate
            time_array = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
            self.plot_curve.setData(time_array, audio_data)
            self.region.setBounds([0, time_array[-1]])
            self.plot_widget.setActive(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频文件失败:\n{e}")
            self.audio_data, self.sample_rate = None, None
            self.plot_curve.clear()
            self.plot_widget.setActive(False)

    def get_config(self):
        feature_key = self.feature_combo.currentData()
        start_time, end_time = self.selected_region_time
        start_frame = int(start_time * self.sample_rate) if start_time is not None else 0
        end_frame = int(end_time * self.sample_rate) if end_time is not None else 0
        config = {
            "audio_file_path": self.audio_file_path,
            "pattern_save_path": self.pattern_save_path,
            "sample_rate": self.sample_rate,
            "pattern_region_time": (start_frame, end_frame),
            "pattern_duration_sec": end_frame - start_frame,
            "feature_type": feature_key,
            "feature_params": self.feature_params.get(feature_key, {}),
            "apply_filter": self.filter_checkbox.isChecked(),
            "filter_range_hz": (None, None),
            "algorithm": "dtw",
            "similarity_metric": self.similarity_metric_combo.currentData(),
            "threshold_strategy": "fixed_threshold" if self.fixed_threshold_radio.isChecked() else "adaptive_threshold",
            "threshold_value": None,
        }

        if self.filter_checkbox.isChecked():
            config["filter_range_hz"] = (int(self.low_freq_edit.text()), int(self.high_freq_edit.text()))

        if self.fixed_threshold_radio.isChecked():
            config["threshold_value"] = float(self.threshold_edit.text())
        return config

    def on_click_default_btn(self):
        config_data = self.get_config()
        if not self.validate_config(config_data):
            return
        config_data["audio_file_path"] = FileOps.get_relative_path(self.audio_file_path, DEFAULT_DIR)
        config_data["pattern_save_path"] = FileOps.get_relative_path(self.pattern_save_path, DEFAULT_DIR)
        save_flag = self.config_manager.save_default_config("PM", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config = self.get_config()
        if not self.validate_config(config):
            return
        try:
            start_time, end_time = self.selected_region_time
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            pattern_data = self.audio_data[start_sample:end_sample]
            sf.write(config["pattern_save_path"], pattern_data, self.sample_rate)
            config["audio_file_path"] = FileOps.get_relative_path(self.audio_file_path, DEFAULT_DIR)
            config["pattern_save_path"] = FileOps.get_relative_path(self.pattern_save_path, DEFAULT_DIR)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存模板文件失败:\n{e}")
            return

        self.config_data = config
        self.accept()
        return self.config_data

    def validate_config(self, config):
        if not config["audio_file_path"]:
            QMessageBox.warning(self, "提示", "请先上传一个源音频文件。")
            return False

        if not os.path.exists(config["audio_file_path"]):
            QMessageBox.warning(self, "提示", "源音频文件不存在或路径已失效，请重新上传。")
            return False

        if config["pattern_duration_sec"] <= 0:
            QMessageBox.warning(self, "提示", "选择的模式片段无效，请在波形图上拖动选择一个区域。")
            return False
        if not config["pattern_save_path"]:
            QMessageBox.warning(self, "提示", "请先选择模板的保存路径。")
            return False
        if config["apply_filter"]:
            low, high = config["filter_range_hz"]
            if low is None or high is None or low >= high:
                QMessageBox.warning(self, "提示", "输入的频率范围无效，最低频率必须小于最高频率。")
                return False
        return True

    def eventFilter(self, watched, event):
        if watched is self.plot_widget.viewport():
            if event.type() in [QEvent.MouseMove, QEvent.HoverMove]:
                scene_pos = self.plot_widget.mapToScene(event.pos())
                viewbox_rect = self.plot_widget.getPlotItem().getViewBox().sceneBoundingRect()
                if viewbox_rect.contains(scene_pos):
                    self.plot_widget.viewport().setCursor(Qt.CrossCursor)
                else:
                    self.plot_widget.viewport().setCursor(Qt.ArrowCursor)
            elif event.type() == QEvent.Leave:
                self.plot_widget.viewport().setCursor(Qt.ArrowCursor)
        return super().eventFilter(watched, event)



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


def check_upper_lower_limit(config_data: dict, parent):
    if config_data["limit_checked"] is False:
        return False
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
    # window = SplConfigWindow(config_manager)
    # window.show()
    # window = FrConfigWindow(config_manager)
    # window.show()
    # window = HdConfigWindow(config_manager)
    # window.show()
    # window = AIConfigWindow(config_manager)
    # window.show()
    window = LPConfigWindow(config_manager, 111)
    window.show()
    app.exec_()
