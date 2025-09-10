import os
import sys
from functools import partial
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QIcon, QIntValidator, QStandardItem
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFormLayout, QFrame, QSplitter
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QTabWidget

from base.file_ops import FileOps
from base.load_audio import load_audio_simple
from base.load_config import ConfigManager
from base.training_model_management import TrainingModelManagement
from consts import error_code, ui_style_const
from consts.feature_params_consts import FEATURE_CONFIG, ALGORITHM_CONFIG
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_clip_extraction_dialog import AudioClipExtractionDialog
from ui.custom_ui_widget.custom_table_widget import DataView
from ui.generic_feature_params_dialog import GenericFeatureParamsDialog
from ui.signal_analysis_window import PatternMatch
from consts.running_consts import DEFAULT_DIR
from base.load_config import LoadUiConfig


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
        self.resize(350, 420)
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
            + ui_style_const.qcheckbox_style
            + ui_style_const.qspinbox_style
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

        top_limit_label = QLabel("上限")
        self.top_limit_spinbox = QSpinBox()
        top_limit = self.load_config.get("top_limit", 70)
        self.top_limit_spinbox.setValue(top_limit)
        top_limit_layout = QHBoxLayout()
        top_limit_layout.addWidget(top_limit_label)
        top_limit_layout.addWidget(self.top_limit_spinbox)

        bottom_limit_label = QLabel("下限")
        self.bottom_limit_spinbox = QSpinBox()
        bottom_limit = self.load_config.get("bottom_limit", 50)
        self.bottom_limit_spinbox.setValue(bottom_limit)
        bottom_limit_layout = QHBoxLayout()
        bottom_limit_layout.addWidget(bottom_limit_label)
        bottom_limit_layout.addWidget(self.bottom_limit_spinbox)
        
        layout = QVBoxLayout()
        layout.addLayout(top_limit_layout)
        layout.addStretch()
        layout.addLayout(bottom_limit_layout)
        layout.addStretch()

        self.limit_group_box = QGroupBox()
        self.limit_group_box.setLayout(layout)

        self.custom_limit_checkbox = QCheckBox("自定义")
        self.on_custom_limit_checkbox_changed(self.custom_limit_checkbox.isChecked())
        self.custom_limit_checkbox.stateChanged.connect(self.on_custom_limit_checkbox_changed)
        self.custom_limit_checkbox.setChecked(self.load_config.get("custom_limit", False))

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
        param_layout.addWidget(self.custom_limit_checkbox)
        param_layout.addStretch()
        param_layout.addWidget(self.limit_group_box)
        param_layout.addStretch()
        
        # 频谱通量配置
        specflux_layout = self.create_spectral_flux_config()
        param_layout.addLayout(specflux_layout)
        param_layout.addStretch()
        
        param_layout.addSpacing(10)
        param_layout.setSpacing(10)
        return param_layout

    def create_spectral_flux_config(self):
        """创建频谱通量配置布局"""
        # 只需要一个复选框来启用/禁用频谱通量绘图
        self.chk_specflux = QCheckBox("同时绘制 Spectral Flux")
        self.chk_specflux.setChecked(self.load_config.get("specflux_enabled", False))
        
        # 创建简单布局，只包含复选框
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chk_specflux)
        
        return main_layout

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
    
    def on_custom_limit_checkbox_changed(self, state):
        if state == Qt.Checked:
            self.limit_group_box.setEnabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(0, 0, 0);")
        else:
            self.limit_group_box.setEnabled(False)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")

    def get_default_config(self):
        default_config = {
            "n_fft": int(self.fft_size_box.currentText()),
            "hop_length": int(self.hop_length_box.currentText()),
            "window_func": self.window_func_box.currentText(),
            "color_map": self.colormap_box.currentText(),
            "freq_scale_type": self.freq_scale_box.currentText(),
            "top_limit": self.top_limit_spinbox.value(),
            "bottom_limit": self.bottom_limit_spinbox.value(),
            "custom_limit": self.custom_limit_checkbox.isChecked(),
            "specflux_enabled": self.chk_specflux.isChecked(),
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




class PDForm(QWidget):
    def __init__(self, load_config=None, channel_id=1, config_id=None):
        super().__init__()
        self.load_config = load_config or {}
        self.channel_id = int(channel_id) if channel_id else 1
        self.config_id = int(config_id) if config_id is not None else int(self.load_config.get("id", 0))
        self.init_ui()

    def init_ui(self):
        self.detect_group = self.create_detect_group()
        self.pre_group = self.create_preprocess_group()
        self.adv_group = self.create_advanced_group()
        
        # 创建"高级选项>>>"按钮
        self.advanced_options_btn = QPushButton("高级选项>>>")
        self.advanced_options_btn.clicked.connect(self.toggle_advanced_panel)
        
        # 左侧面板：峰值检测参数 + 高级选项按钮
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.detect_group)
        
        # 按钮容器，右对齐
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        btn_layout.addWidget(self.advanced_options_btn)
        btn_layout.setContentsMargins(10, 5, 10, 5)
        
        left_layout.addWidget(btn_container)
        left_layout.addStretch()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 右侧面板：预处理选项 + 高级选项
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.addWidget(self.pre_group)
        right_layout.addWidget(self.adv_group)
        right_layout.addStretch()
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)  # 左侧面板可拉伸
        splitter.setStretchFactor(1, 0)  # 右侧面板固定宽度
        
        # 默认隐藏右侧面板
        self.right_panel.setVisible(False)
        self.advanced_panel_visible = False
        
        # 主布局
        root = QHBoxLayout()
        root.addWidget(splitter)
        root.setContentsMargins(0, 0, 0, 0)
        self.setLayout(root)

    def _on_channel_changed(self):
        pass
    
    def toggle_advanced_panel(self):
        """切换高级选项面板的显示/隐藏状态"""
        self.advanced_panel_visible = not self.advanced_panel_visible
        self.right_panel.setVisible(self.advanced_panel_visible)
        
        # 更新按钮文本
        if self.advanced_panel_visible:
            self.advanced_options_btn.setText("<<<高级选项")
        else:
            self.advanced_options_btn.setText("高级选项>>>")

    def create_preprocess_group(self):
        group_box = QGroupBox("预处理选项")
        vbox = QVBoxLayout()

        row_filter_main = QHBoxLayout()
        self.chk_filter = QCheckBox("滤波")
        self.chk_filter.setChecked(self.load_config.get("filter_enabled", False))
        self.chk_filter.stateChanged.connect(self.get_config)
        row_filter_main.addWidget(self.chk_filter)
        row_filter_main.addStretch()
        row_filter_main.addWidget(QLabel("范围(Hz):"))
        self.edit_filter_ranges = QLineEdit()
        self.edit_filter_ranges.setPlaceholderText("0,300; 700,1000;")
        self.edit_filter_ranges.setText(self.load_config.get("filter_ranges", ""))
        self.edit_filter_ranges.textChanged.connect(lambda _: self.get_config())
        row_filter_main.addWidget(self.edit_filter_ranges)
        row_filter_main.addWidget(QLabel("类型:"))
        self.combo_filter_type = QComboBox()
        self.combo_filter_type.addItems(["带通", "带阻"])
        self.combo_filter_type.setCurrentIndex(0 if self.load_config.get("filter_type", "bandpass") == "bandpass" else 1)
        self.combo_filter_type.currentIndexChanged.connect(lambda _: self.get_config())
        row_filter_main.addWidget(self.combo_filter_type)

        row_filter_order = QHBoxLayout()
        row_filter_order.addStretch()
        row_filter_order.addWidget(QLabel("阶数"))
        self.spin_filter_order = QSpinBox()
        self.spin_filter_order.setRange(1, 20)
        self.spin_filter_order.setValue(int(self.load_config.get("filter_order", 4)))
        self.spin_filter_order.valueChanged.connect(lambda _: self.get_config())
        row_filter_order.addWidget(self.spin_filter_order)

        row_smooth_main = QHBoxLayout()
        self.chk_smooth = QCheckBox("平滑")
        self.chk_smooth.setChecked(self.load_config.get("smooth_enabled", False))
        self.chk_smooth.stateChanged.connect(self.get_config)
        row_smooth_main.addWidget(self.chk_smooth)
        row_smooth_main.addStretch()
        row_smooth_main.addWidget(QLabel("单位:"))
        self.combo_smooth_unit = QComboBox()
        self.combo_smooth_unit.addItems(["时间(秒)", "格点数"])
        self.combo_smooth_unit.setCurrentIndex(0 if self.load_config.get("smooth_unit", "time") == "time" else 1)
        self.combo_smooth_unit.currentIndexChanged.connect(lambda _: (self._update_smooth_unit_enabled(), self.get_config()))
        row_smooth_main.addWidget(self.combo_smooth_unit)
        self.spin_smooth_time = QDoubleSpinBox()
        self.spin_smooth_time.setRange(0.00, 999.00)
        self.spin_smooth_time.setDecimals(4)
        self.spin_smooth_time.setSingleStep(0.01)
        self.spin_smooth_time.setValue(float(self.load_config.get("smooth_time_sec", 0.02)))
        self.spin_smooth_time.valueChanged.connect(lambda _: self.get_config())
        row_smooth_main.addWidget(self.spin_smooth_time)
        self.spin_smooth_points = QSpinBox()
        self.spin_smooth_points.setRange(1, 99999)
        self.spin_smooth_points.setValue(int(self.load_config.get("smooth_points", 0)))
        self.spin_smooth_points.valueChanged.connect(lambda _: self.get_config())
        row_smooth_main.addWidget(self.spin_smooth_points)

        row_splwin = QHBoxLayout()
        row_splwin.addWidget(QLabel("SPL计算窗长"))
        row_splwin.addStretch()
        row_splwin.addWidget(QLabel("单位:"))
        self.combo_spl_window_unit = QComboBox()
        self.combo_spl_window_unit.addItems(["时间(秒)", "格点数"])
        self.combo_spl_window_unit.setCurrentIndex(0 if self.load_config.get("spl_window_unit", "time") == "time" else 1)
        self.combo_spl_window_unit.currentIndexChanged.connect(lambda _: (self._update_spl_window_unit_enabled(), self.get_config()))
        row_splwin.addWidget(self.combo_spl_window_unit)
        self.spin_spl_window_time = QDoubleSpinBox()
        self.spin_spl_window_time.setRange(0.000, 999.000)
        self.spin_spl_window_time.setDecimals(4)
        self.spin_spl_window_time.setSingleStep(0.001)
        self.spin_spl_window_time.setValue(float(self.load_config.get("spl_window_time_sec", 0.050)))
        self.spin_spl_window_time.valueChanged.connect(lambda _: self.get_config())
        row_splwin.addWidget(self.spin_spl_window_time)
        self.spin_spl_window_points = QSpinBox()
        self.spin_spl_window_points.setRange(1, 99999)
        self.spin_spl_window_points.setValue(int(self.load_config.get("spl_window_points", 0)))
        self.spin_spl_window_points.valueChanged.connect(lambda _: self.get_config())
        row_splwin.addWidget(self.spin_spl_window_points)

        vbox.addLayout(row_filter_main)
        vbox.addLayout(row_filter_order)
        vbox.addLayout(row_splwin)
        vbox.addLayout(row_smooth_main)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        self._update_smooth_unit_enabled()
        self._update_spl_window_unit_enabled()
        return group_box

    def create_detect_group(self):
        group_box = QGroupBox("峰值提取参数")
        vbox = QVBoxLayout()

        row_count = QHBoxLayout()
        self.chk_peak_count = QCheckBox("峰值个数")
        self.chk_peak_count.setChecked(self.load_config.get("peak_count_enabled", True))
        self.chk_peak_count.stateChanged.connect(self.get_config)
        row_count.addWidget(self.chk_peak_count)
        row_count.addStretch()
        row_count.addWidget(QLabel("最大峰数目:"))
        self.spin_peak_count = QSpinBox()
        self.spin_peak_count.setRange(1, 9999)
        self.spin_peak_count.setValue(int(self.load_config.get("peak_count", 3)))
        self.spin_peak_count.valueChanged.connect(lambda _: self.get_config())
        row_count.addWidget(self.spin_peak_count)

        row_size = QHBoxLayout()
        self.chk_peak_size = QCheckBox("峰值大小")
        self.chk_peak_size.setChecked(self.load_config.get("peak_size_enabled", True))
        self.chk_peak_size.stateChanged.connect(self.get_config)
        row_size.addWidget(self.chk_peak_size)
        row_size.addStretch()
        row_size.addWidget(QLabel("单位:"))
        self.combo_peak_size_unit = QComboBox()
        self.combo_peak_size_unit.addItems(["rmsV", "dBL"])
        peak_size_unit_saved = self.load_config.get("peak_size_unit", "db")
        self.combo_peak_size_unit.setCurrentIndex(0 if peak_size_unit_saved == "rms" else 1)
        self.combo_peak_size_unit.currentIndexChanged.connect(lambda _: (self._update_peak_units(), self.get_config()))
        row_size.addWidget(self.combo_peak_size_unit)
        self.spin_peak_size = QDoubleSpinBox()
        self.spin_peak_size.setRange(-100.0, 200.0)
        self.spin_peak_size.setDecimals(2)
        self.spin_peak_size.setSingleStep(1.0)
        self.spin_peak_size.setValue(float(self.load_config.get("peak_min_value", 100.0)))
        self.spin_peak_size.valueChanged.connect(lambda _: self.get_config())
        row_size.addWidget(self.spin_peak_size)

        row_slope = QHBoxLayout()
        self.chk_peak_slope = QCheckBox("峰凸起度")
        self.chk_peak_slope.setChecked(self.load_config.get("peak_slope_enabled", False))
        self.chk_peak_slope.stateChanged.connect(self.get_config)
        row_slope.addWidget(self.chk_peak_slope)
        row_slope.addStretch()
        row_slope.addWidget(QLabel("单位:"))
        self.combo_peak_slope_unit = QComboBox()
        self.combo_peak_slope_unit.addItems(["rmsV", "dBL"])
        peak_slope_unit_saved = self.load_config.get("peak_slope_unit", "db")
        self.combo_peak_slope_unit.setCurrentIndex(0 if peak_slope_unit_saved == "rms" else 1)
        self.combo_peak_slope_unit.currentIndexChanged.connect(lambda _: (self._update_peak_units(), self.get_config()))
        row_slope.addWidget(self.combo_peak_slope_unit)
        self.spin_peak_slope = QDoubleSpinBox()
        self.spin_peak_slope.setRange(0.0, 200.0)
        self.spin_peak_slope.setDecimals(3)
        self.spin_peak_slope.setSingleStep(1.0)
        self.spin_peak_slope.setValue(float(self.load_config.get("peak_min_slope", 100.0)))
        self.spin_peak_slope.valueChanged.connect(lambda _: self.get_config())
        row_slope.addWidget(self.spin_peak_slope)

        row_specflux = QHBoxLayout()
        self.chk_specflux = QCheckBox("频谱通量")
        self.chk_specflux.setChecked(self.load_config.get("specflux_enabled", False))
        self.chk_specflux.stateChanged.connect(self.get_config)
        row_specflux.addWidget(self.chk_specflux)
        row_specflux.addStretch()
        row_specflux.addWidget(QLabel("阈值:"))
        self.spin_specflux = QDoubleSpinBox()
        self.spin_specflux.setRange(0.0, 100000.0)
        self.spin_specflux.setDecimals(3)
        self.spin_specflux.setSingleStep(0.001)
        self.spin_specflux.setValue(float(self.load_config.get("specflux_min_value", 0.0)))
        self.spin_specflux.valueChanged.connect(lambda _: self.get_config())
        row_specflux.addWidget(self.spin_specflux)

        row_nms = QHBoxLayout()
        self.chk_nms = QCheckBox("最小峰间距")
        self.chk_nms.setChecked(self.load_config.get("nms_enabled", False))
        self.chk_nms.stateChanged.connect(self.get_config)
        row_nms.addWidget(self.chk_nms)
        row_nms.addStretch()
        row_nms.addWidget(QLabel("单位:"))
        self.combo_nms_unit = QComboBox()
        self.combo_nms_unit.addItems(["时间(秒)", "格点数"])
        self.combo_nms_unit.setCurrentIndex(0 if self.load_config.get("nms_unit", "time") == "time" else 1)
        self.combo_nms_unit.currentIndexChanged.connect(lambda _: (self._update_nms_unit_enabled(), self.get_config()))
        row_nms.addWidget(self.combo_nms_unit)
        self.spin_nms_time = QDoubleSpinBox()
        self.spin_nms_time.setRange(0.00, 100.00)
        self.spin_nms_time.setDecimals(3)
        self.spin_nms_time.setSingleStep(0.01)
        self.spin_nms_time.setValue(float(self.load_config.get("nms_time_sec", 0.50)))
        self.spin_nms_time.valueChanged.connect(lambda _: self.get_config())
        row_nms.addWidget(self.spin_nms_time)
        self.spin_nms_points = QSpinBox()
        self.spin_nms_points.setRange(1, 99999)
        self.spin_nms_points.setValue(int(self.load_config.get("nms_points", 0)))
        self.spin_nms_points.valueChanged.connect(lambda _: self.get_config())
        row_nms.addWidget(self.spin_nms_points)

        row_duration = QHBoxLayout()
        self.chk_duration = QCheckBox("峰持续时间")
        self.chk_duration.setChecked(self.load_config.get("duration_enabled", False))
        self.chk_duration.stateChanged.connect(self.get_config)
        row_duration.addWidget(self.chk_duration)
        row_duration.addStretch()
        row_duration.addWidget(QLabel("最短"))
        self.spin_duration_min = QDoubleSpinBox()
        self.spin_duration_min.setRange(0.0, 1000.0)
        self.spin_duration_min.setDecimals(3)
        self.spin_duration_min.setSingleStep(0.001)
        self.spin_duration_min.setValue(float(self.load_config.get("duration_min", 0.0)))
        self.spin_duration_min.valueChanged.connect(lambda _: self.get_config())
        row_duration.addWidget(self.spin_duration_min)
        row_duration.addWidget(QLabel("最长"))
        self.spin_duration_max = QDoubleSpinBox()
        self.spin_duration_max.setRange(0.0, 1000.0)
        self.spin_duration_max.setDecimals(3)
        self.spin_duration_max.setSingleStep(0.001)
        self.spin_duration_max.setValue(float(self.load_config.get("duration_max", 0.0)))
        self.spin_duration_max.valueChanged.connect(lambda _: self.get_config())
        row_duration.addWidget(self.spin_duration_max)

        vbox.addLayout(row_count)
        vbox.addLayout(row_size)
        vbox.addLayout(row_slope)
        vbox.addLayout(row_specflux)
        vbox.addLayout(row_nms)
        vbox.addLayout(row_duration)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        self._update_nms_unit_enabled()
        self._update_peak_units()
        return group_box

    def create_advanced_group(self):
        adv_group = QGroupBox("高级选项")
        adv_layout = QVBoxLayout()

        row_convex_len = QHBoxLayout()
        row_convex_len.addWidget(QLabel("峰凸起度计算窗口"))
        row_convex_len.addStretch(1)
        row_convex_len.addWidget(QLabel("单位:"))
        self.combo_convex_unit = QComboBox()
        self.combo_convex_unit.addItems(["音频长度", "格点数", "时长(秒)"])
        self.combo_convex_unit.setCurrentIndex({"audio":0, "points":1, "time":2}.get(self.load_config.get("convex_unit", "audio"), 0))
        self.combo_convex_unit.currentIndexChanged.connect(lambda _: (self._update_convex_unit_enabled(), self.get_config()))
        row_convex_len.addWidget(self.combo_convex_unit)
        self.spin_convex_audio_ratio = QDoubleSpinBox()
        self.spin_convex_audio_ratio.setRange(0.001, 1.000)
        self.spin_convex_audio_ratio.setDecimals(4)
        self.spin_convex_audio_ratio.setSingleStep(0.01)
        self.spin_convex_audio_ratio.setValue(float(self.load_config.get("convex_audio_ratio", 1.0)))
        self.spin_convex_audio_ratio.valueChanged.connect(lambda _: self.get_config())
        row_convex_len.addWidget(self.spin_convex_audio_ratio)
        self.spin_convex_points = QSpinBox()
        self.spin_convex_points.setRange(1, 100000000)
        self.spin_convex_points.setValue(int(self.load_config.get("convex_points", 1024)))
        self.spin_convex_points.valueChanged.connect(lambda _: self.get_config())
        row_convex_len.addWidget(self.spin_convex_points)
        self.spin_convex_time = QDoubleSpinBox()
        self.spin_convex_time.setRange(0.000, 999.000)
        self.spin_convex_time.setDecimals(3)
        self.spin_convex_time.setSingleStep(0.1)
        self.spin_convex_time.setValue(float(self.load_config.get("convex_time_sec", 0.0)))
        self.spin_convex_time.valueChanged.connect(lambda _: self.get_config())
        row_convex_len.addWidget(self.spin_convex_time)

        row_dmode = QHBoxLayout()
        row_dmode.addWidget(QLabel("峰持续时间参考点"))
        row_dmode.addStretch(1)
        row_dmode.addWidget(QLabel("单位:"))
        ref_unit_saved = self.load_config.get("duration_ref_unit", "peak")
        ref_value_saved = float(self.load_config.get("duration_ref_value", 0.50 if ref_unit_saved == "peak" else 100.0))
        self.combo_duration_ref_unit = QComboBox()
        self.combo_duration_ref_unit.addItems(["Vpeak", "dBL"])
        self.combo_duration_ref_unit.setCurrentIndex(0 if ref_unit_saved == "peak" else 1)
        self.combo_duration_ref_unit.currentIndexChanged.connect(lambda _: (self._update_duration_ref_unit(), self.get_config()))
        row_dmode.addWidget(self.combo_duration_ref_unit)
        self.spin_duration_ref = QDoubleSpinBox()
        self.spin_duration_ref.setDecimals(2)
        self.spin_duration_ref.setSingleStep(0.01)
        row_dmode.addWidget(self.spin_duration_ref)
        self._update_duration_ref_unit()
        self.spin_duration_ref.setValue(float(ref_value_saved))
        self.spin_duration_ref.valueChanged.connect(lambda _: self.get_config())
        
        # 频谱通量计算间隔选项
        row_hop_length = QHBoxLayout()
        row_hop_length.addWidget(QLabel("频谱通量计算间隔"))
        row_hop_length.addStretch(1)
        row_hop_length.addWidget(QLabel("格点数:"))
        self.combo_hop_length = QComboBox()
        hop_values = [str(16 * (2 ** i)) for i in range(8)]  # 16, 32, 64, 128, 256, 512, 1024
        self.combo_hop_length.addItems(hop_values)
        current_hop = str(self.load_config.get("spectral_flux_hop_length", 512))
        if current_hop in hop_values:
            self.combo_hop_length.setCurrentText(current_hop)
        else:
            self.combo_hop_length.setCurrentText("512")
        self.combo_hop_length.currentTextChanged.connect(lambda _: self.get_config())
        row_hop_length.addWidget(self.combo_hop_length)

        # 频谱通量窗口长度选项
        row_window_length = QHBoxLayout()
        row_window_length.addWidget(QLabel("频谱通量窗口长度"))
        row_window_length.addStretch(1)
        row_window_length.addWidget(QLabel("格点数:"))
        self.combo_window_length = QComboBox()

        window_values = [str(2 ** i) for i in range(6, 14)]  # 64 to 8192
        self.combo_window_length.addItems(window_values)
        current_window = str(self.load_config.get("spectral_flux_window_length", 1024))
        if current_window in window_values:
            self.combo_window_length.setCurrentText(current_window)
        else:
            self.combo_window_length.setCurrentText("1024")
        self.combo_window_length.currentTextChanged.connect(lambda _: self.get_config())
        row_window_length.addWidget(self.combo_window_length)

        adv_layout.addLayout(row_convex_len)
        adv_layout.addLayout(row_dmode)
        adv_layout.addLayout(row_hop_length)
        adv_layout.addLayout(row_window_length)
        adv_layout.addStretch()
        adv_layout.setSpacing(10)
        adv_layout.setContentsMargins(10, 20, 10, 20)
        adv_group.setLayout(adv_layout)
        adv_group.setMinimumWidth(260)
        self._update_convex_unit_enabled()
        return adv_group

    def _update_duration_ref_unit(self):
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
            self.spin_peak_size.setRange(0.0, 1.0)
            self.spin_peak_size.setDecimals(3)
            self.spin_peak_size.setSingleStep(0.001)
            v = float(self.spin_peak_size.value())
            if v < 0.0:
                self.spin_peak_size.setValue(0.0)
            elif v > 1.0:
                self.spin_peak_size.setValue(1.0)
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

    def get_config(self):
        return {
            "filter_enabled": self.chk_filter.isChecked(),
            "filter_ranges": self.edit_filter_ranges.text().strip(),
            "filter_type": "bandpass" if self.combo_filter_type.currentIndex() == 0 else "bandstop",
            "smooth_enabled": self.chk_smooth.isChecked(),
            "smooth_unit": "time" if self.combo_smooth_unit.currentIndex() == 0 else "points",
            "smooth_time_sec": float(self.spin_smooth_time.value()),
            "smooth_points": int(self.spin_smooth_points.value()),
            "peak_count_enabled": self.chk_peak_count.isChecked(),
            "peak_count": int(self.spin_peak_count.value()),
            "peak_size_enabled": self.chk_peak_size.isChecked(),
            "peak_size_unit": ("rms" if self.combo_peak_size_unit.currentIndex() == 0 else "db"),
            "peak_min_value": float(self.spin_peak_size.value()),
            "peak_slope_enabled": self.chk_peak_slope.isChecked(),
            "peak_slope_unit": ("rms" if self.combo_peak_slope_unit.currentIndex() == 0 else "db"),
            "peak_min_slope": float(self.spin_peak_slope.value()),
            "specflux_enabled": self.chk_specflux.isChecked(),
            "specflux_min_value": float(self.spin_specflux.value()),
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
            "advanced_mode": True,
            "filter_order": int(self.spin_filter_order.value()),
            "smooth_algo": 1,
            "convex_unit": ("audio" if self.combo_convex_unit.currentIndex() == 0 else ("points" if self.combo_convex_unit.currentIndex() == 1 else "time")),
            "convex_audio_ratio": float(self.spin_convex_audio_ratio.value()),
            "convex_points": int(self.spin_convex_points.value()),
            "convex_time_sec": float(self.spin_convex_time.value()),
            "duration_ref_unit": ("peak" if self.combo_duration_ref_unit.currentIndex() == 0 else "db"),
            "duration_ref_value": float(self.spin_duration_ref.value()),
            "spectral_flux_hop_length": int(self.combo_hop_length.currentText()),
            "spectral_flux_window_length": int(self.combo_window_length.currentText()),
        }

class PDTabbedPDConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        # Clear config cache to ensure we load fresh data from file
        self.config_manager.config = {}
        all_config = self.config_manager.load_config()
        self.load_config = all_config.get(model_type, {})
        self.init_ui()
        self.load_channels()
        # self.init_auto_equal_checkbox()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(980, 600)
        self.resize(1000, 620)
        v = QVBoxLayout()
        self.tabs = QTabWidget()
        try:
            self.tabs.setStyleSheet("QTabBar::tab { font-size: 20px; min-width: 100px; height: 40px; }")
        except Exception:
            pass
        
        dual_channel_group = self.create_dual_channel_analysis_group()
        merging_group = self.create_peak_merging_group()

        bottom = QHBoxLayout()
        self.btn_default = QPushButton(" 设为默认 ")
        self.btn_ok = QPushButton(" 确  认 ")
        self.btn_default.clicked.connect(self.on_click_default_btn)
        self.btn_ok.clicked.connect(self.on_click_ok_btn)
        bottom.addWidget(self.btn_default)
        bottom.addStretch()
        bottom.addWidget(self.btn_ok)
        v.addWidget(self.tabs)
        v.addWidget(dual_channel_group)
        v.addWidget(merging_group)
        v.addLayout(bottom)
        self.setLayout(v)
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

    def create_dual_channel_analysis_group(self):
        group_box = QGroupBox("双通道分析设置")
        layout = QVBoxLayout()
        
        # 双通道匹配窗口配置
        dual_channel_layout = QHBoxLayout()
        dual_channel_layout.addWidget(QLabel("双通道匹配窗口"))
        dual_channel_layout.addStretch()
        dual_channel_layout.addWidget(QLabel("时间(ms):"))
        self.spin_dual_channel_window = QSpinBox()
        self.spin_dual_channel_window.setRange(1, 1000)
        self.spin_dual_channel_window.setValue(self.load_config.get("dual_channel_window_ms", 20))
        self.spin_dual_channel_window.valueChanged.connect(lambda _: self.get_config())
        dual_channel_layout.addWidget(self.spin_dual_channel_window)
        
        layout.addLayout(dual_channel_layout)
        group_box.setLayout(layout)
        return group_box
    
    def create_peak_merging_group(self):
        group_box = QGroupBox("最终的测试通过参数")
        layout = QVBoxLayout()
        
        # 峰值数配置
        final_test_layout = QHBoxLayout()
        final_test_layout.addWidget(QLabel("峰值数"))
        final_test_layout.addStretch()
        self.final_spin_test_peak_min = QSpinBox()
        self.final_spin_test_peak_min.setRange(0, 1000000)
        self.final_spin_test_peak_max = QSpinBox()
        self.final_spin_test_peak_max.setRange(0, 1000000)
        min_v = self.load_config.get("final_test_peak_min", 0)
        max_v = self.load_config.get("final_test_peak_max", 1000000)
        self.final_spin_test_peak_min.setValue(min_v)
        self.final_spin_test_peak_max.setValue(max_v)
        self.final_spin_test_peak_min.valueChanged.connect(lambda: self.final_spin_test_peak_max.setValue(max(self.final_spin_test_peak_min.value(), self.final_spin_test_peak_max.value())))
        self.final_spin_test_peak_max.valueChanged.connect(lambda: self.final_spin_test_peak_min.setValue(min(self.final_spin_test_peak_min.value(), self.final_spin_test_peak_max.value())))
        self.final_spin_test_peak_min.valueChanged.connect(lambda _: self.get_config())
        self.final_spin_test_peak_max.valueChanged.connect(lambda _: self.get_config())
        final_test_layout.addWidget(self.final_spin_test_peak_min)
        final_test_layout.addWidget(QLabel(" ≤ 峰值数 ≤ "))
        final_test_layout.addWidget(self.final_spin_test_peak_max)
        
        layout.addLayout(final_test_layout)
        group_box.setLayout(layout)
        return group_box

    def load_channels(self):
        ch_map = {}
        # Support new nested dict format: {"channels": {"channel_1": {...}, "channel_2": {...}}, ...}
        if isinstance(self.load_config, dict):
            channels_val = self.load_config.get("channels")
            if isinstance(channels_val, dict):
                # map by numeric channel index if possible, else by order
                for key in ("channel_1", "channel_2"):
                    entry = channels_val.get(key, {})
                    if isinstance(entry, dict):
                        # directly map channel_1 -> 1, channel_2 -> 2
                        ch_num = 1 if key == "channel_1" else 2
                        ch_map[ch_num] = entry
            elif isinstance(channels_val, list) and len(channels_val) > 0:
                # legacy list format: convert to map by 'channel' or order
                for c in channels_val:
                    try:
                        ch_idx = int(c.get("channel", 0))
                    except Exception:
                        ch_idx = 0
                    if ch_idx <= 0:
                        # assign by next available (1 or 2)
                        for candidate in (1, 2):
                            if candidate not in ch_map:
                                ch_idx = candidate
                                break
                    ch_map[ch_idx] = c
            else:
                # single-channel legacy top-level config
                try:
                    ch = int(self.load_config.get("channel", 1))
                except Exception:
                    ch = 1
                ch_map[ch] = {k: v for k, v in self.load_config.items() if k != "channel"}

        # initialize used ids set for backward compatibility
        self._used_ids = {int(c.get("id")) for c in ch_map.values() if isinstance(c, dict) and c.get("id") is not None}
        self._next_id_val = max(self._used_ids) + 1 if getattr(self, "_used_ids", None) else 1
        for ch in (1, 2):
            cfg = ch_map.get(ch, {})
            cfg_id = int(cfg.get("id")) if cfg.get("id") is not None else self._next_id()
            self.add_tab(int(ch), cfg, cfg_id)

    def add_tab(self, channel_id: int, cfg: dict, config_id: int):
        form = PDForm(cfg, channel_id, config_id)
        tab_title = f"通道{int(channel_id)}"
        idx = self.tabs.addTab(form, tab_title)
        self.tabs.setCurrentIndex(idx)

    def _next_id(self):
        while True:
            nid = getattr(self, "_next_id_val", 1)
            if nid not in getattr(self, "_used_ids", set()):
                self._used_ids.add(nid)
                self._next_id_val = nid + 1
                return nid
            self._next_id_val += 1

    def _refresh_tab_title(self, form: PDForm):
        pass

    def on_add_channel(self):
        pass

    def on_remove_channel(self):
        pass

    def get_channels_config(self):
        channels = {}
        for i in range(self.tabs.count()):
            form = self.tabs.widget(i)
            cfg = form.get_config()
            try:
                ch = int(getattr(form, "channel_id", i + 1))
            except Exception:
                ch = i + 1
            if ch <= 0:
                QMessageBox.warning(self, "设置警告", "通道号必须为正整数")
                return None
            key = f"channel_{ch}"
            channels[key] = cfg
        return channels

    def get_config(self):
        """获取当前配置，用于UI控件变化时的实时更新"""
        # 这是一个空方法，主要用于响应UI控件的信号连接
        # 实际的配置保存在按钮点击事件中处理
        pass

    def on_click_default_btn(self):
        chs = self.get_channels_config()
        if chs is None:
            return
        
        config_data = {
            "channels": chs,
            "align_audio": True,
            "dual_channel_window_ms": self.spin_dual_channel_window.value(),
            "final_test_peak_min": self.final_spin_test_peak_min.value(),
            "final_test_peak_max": self.final_spin_test_peak_max.value(),
            "auto_equal_length": bool(self._auto_equal_chk.isChecked()) if hasattr(self, "_auto_equal_chk") else False,
        }
        save_flag = self.config_manager.save_default_config("PD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        chs = self.get_channels_config()
        if chs is None:
            return None
        
        config_data = {
            "type": "PD",
            "channels": chs,
            "align_audio": True,
            "dual_channel_window_ms": self.spin_dual_channel_window.value(),
            "final_test_peak_min": self.final_spin_test_peak_min.value(),
            "final_test_peak_max": self.final_spin_test_peak_max.value(),
            "auto_equal_length": bool(self._auto_equal_chk.isChecked()) if hasattr(self, "_auto_equal_chk") else False,
        }
        
        try:
            # Load current sequence config
            json_file_path = DEFAULT_DIR + "ui/ui_config/sequence_config.json"
            code, current_config = LoadUiConfig.load_data_from_json(json_file_path)
            
            if code == 0 and isinstance(current_config, list) and len(current_config) > 0:
                # Update the PD config in the nested structure
                sequence_data = current_config[0]
                for seq_key, seq_val in sequence_data.items():
                    if isinstance(seq_val, dict) and "analysis_list" in seq_val:
                        analysis_list = seq_val["analysis_list"]
                        if self.model_type in analysis_list:
                            analysis_list[self.model_type] = config_data
                            break
                
                # Save the updated config back to file
                save_flag = LoadUiConfig.save_sequence_config_to_json(current_config, json_file_path)
                if not save_flag:
                    pass  # Silent failure for production
            else:
                # Fallback: update flat structure directly if it's not the nested format
                if isinstance(current_config, dict):
                    current_config[self.model_type] = config_data
                    save_flag = LoadUiConfig.save_sequence_config_to_json(current_config, json_file_path)
                    pass  # Success
                    
        except Exception as e:
            pass  # Silent error handling for production
        
        self.accept()
        return config_data


class PatternMatchConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.feature_registry = FEATURE_CONFIG
        self.algorithm_registry = ALGORITHM_CONFIG
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self.sample_rate = self.load_config.get("sample_rate", 44100)
        self.config_data = None
        self.config_ch1 = self.load_config.get("channel_1_config", {})
        self.config_ch2 = self.load_config.get("channel_2_config", {})
        self.pattern_list = self.config_ch1.get("pattern_list", []).copy()

        self.feature_params = {
            key: {p_name: p_def['default'] for p_name, p_def in info['params'].items()}
            for key, info in self.feature_registry.items() if info.get('params')
        }
        self.algorithm_params = {
            key: {p_name: p_def['default'] for p_name, p_def in info['params'].items()}
            for key, info in self.algorithm_registry.items() if info.get('params')
        }

        self.init_ui()
        self.initial_populate_ui()
        # self.init_auto_equal_checkbox()

    def init_ui(self):
        self.setWindowTitle("双通道模式匹配参数配置")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(800, 350)
        self.resize(800, 750)
        self.main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)

        pattern_group_box = self.create_pattern_group_box()
        splitter.addWidget(pattern_group_box)

        options_container = QWidget()
        options_layout = self.create_options_layout()
        options_container.setLayout(options_layout)

        channel_management_group = self.create_channel_management_group()
        main_options_layout = QVBoxLayout()
        main_options_layout.addWidget(channel_management_group)
        main_options_layout.addWidget(options_container)
        main_options_container = QWidget()
        main_options_container.setLayout(main_options_layout)
        splitter.addWidget(main_options_container)

        splitter.setSizes([450, 300])
        splitter.setCollapsible(0, False)
        self.main_layout.addWidget(splitter)

        # Add auto equal length checkbox at lower right corner
        checkbox_layout = QHBoxLayout()
        self._auto_equal_chk = QCheckBox("自动匹配模板长度（对齐模板峰值）")
        # self._auto_equal_chk.click.connect(self.init_auto_equal_checkbox)
        self.init_auto_equal_checkbox()
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self._auto_equal_chk)
        self.main_layout.addLayout(checkbox_layout)

        btn_layout = self.create_btn_layout()
        self.main_layout.addLayout(btn_layout)

        self.setLayout(self.main_layout)
        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qdialog_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qtextedit_style
            + ui_style_const.qtableview_style
        )

    def init_auto_equal_checkbox(self):
        """Initialize auto equal checkbox with load config"""
        try:
            if isinstance(self.load_config, dict):
                auto_flag = bool(self.load_config.get("auto_equal_length", False))
                self._auto_equal_chk.setChecked(auto_flag)
        except Exception:
            pass

    def create_channel_management_group(self):
        group = QGroupBox("当前编辑通道")
        layout = QHBoxLayout()
        self.radio_ch1 = QRadioButton("通道 1")
        self.radio_ch2 = QRadioButton("通道 2")
        self.channel_button_group = QButtonGroup(self)
        self.channel_button_group.addButton(self.radio_ch1, 1)
        self.channel_button_group.addButton(self.radio_ch2, 2)
        self.radio_ch1.setChecked(True)
        self.channel_button_group.idClicked.connect(self.on_channel_selection_changed)
        layout.addWidget(self.radio_ch1)
        layout.addWidget(self.radio_ch2)
        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_pattern_group_box(self):
        self.data_view = DataView(len(self.pattern_list), 2, [])
        self.data_view.set_h_header(["模板文件", "模板时长 (s)"])
        extract_btn = QPushButton("提取模板")
        extract_btn.clicked.connect(self.on_click_extract_btn)
        add_btn = QPushButton("添加模板")
        add_btn.clicked.connect(self.on_click_add_btn)
        remove_btn = QPushButton("删除模板")
        remove_btn.clicked.connect(self.on_click_remove_btn)
        self.n_chosen_pattern_label = QLabel("已加载模板： 0")

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(extract_btn)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(self.n_chosen_pattern_label, alignment=Qt.AlignBottom)

        layout = QHBoxLayout()
        layout.addWidget(self.data_view)
        layout.addLayout(btn_layout)
        group = QGroupBox("模板选择")
        group.setLayout(layout)
        return group

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

        normalization_label = QLabel("<b>归一化方法</b>")
        layout.addWidget(normalization_label)

        self.normalization_checkbox = QCheckBox("启用归一化")
        self.normalization_checkbox.setChecked(True)
        self.normalization_checkbox.toggled.connect(self.on_normalization_toggled)
        layout.addWidget(self.normalization_checkbox)

        self.normalization_combo = QComboBox()
        self.normalization_combo.addItem("峰值归一化", "peak")
        self.normalization_combo.addItem("Z-Score 标准化", "zscore")
        self.normalization_combo.addItem("最小-最大值缩放", "minmax")
        self.normalization_combo.addItem("L2 范数归一化", "l2_norm")
        layout.addWidget(self.normalization_combo)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        filter_label = QLabel("<b>带阻滤波</b>")
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
        filter_range_layout.addRow(low_freq_label, self.low_freq_edit)
        filter_range_layout.addRow(high_freq_label, self.high_freq_edit)

        layout.addLayout(filter_range_layout)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_strategy_group(self):
        group = QGroupBox("匹配策略")
        main_layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        self.algorithm_combo = QComboBox()
        for key, info in self.algorithm_registry.items():
            self.algorithm_combo.addItem(info['display_name'], userData=key)
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.algorithm_params_btn = QPushButton("算法参数")
        self.algorithm_params_btn.clicked.connect(self.on_click_algorithm_params)
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(self.algorithm_combo, 1)
        algo_layout.addWidget(self.algorithm_params_btn)
        form_layout.addRow("<b>匹配算法:</b>", algo_layout)
        main_layout.addLayout(form_layout)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator2)
        return_strategy_layout = QVBoxLayout()
        return_label = QLabel("<b>匹配点返回策略:</b>")
        return_strategy_layout.addWidget(return_label)

        fixed_threshold_layout = QHBoxLayout()
        self.fixed_threshold_radio = QRadioButton("固定阈值:")
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setSuffix(" %")
        self.threshold_spinbox.setRange(0.00, 100.00)
        self.threshold_spinbox.setDecimals(2)
        self.threshold_spinbox.setValue(90.00)
        self.threshold_spinbox.setSingleStep(1.0)

        self.auto_calc_threshold_btn = QPushButton("自动计算")
        self.auto_calc_threshold_btn.clicked.connect(self.on_auto_calculate_threshold)

        fixed_threshold_layout.addWidget(self.fixed_threshold_radio)
        fixed_threshold_layout.addWidget(self.threshold_spinbox)
        fixed_threshold_layout.addWidget(self.auto_calc_threshold_btn)
        return_strategy_layout.addLayout(fixed_threshold_layout)

        self.adaptive_threshold_radio = QRadioButton("自适应阈值")
        self.adaptive_threshold_radio.toggled.connect(self.on_strategy_radio_changed)

        self.fixed_threshold_radio.setChecked(True)
        self.fixed_threshold_radio.toggled.connect(self.on_strategy_radio_changed)
        return_strategy_layout.addWidget(self.adaptive_threshold_radio)
        main_layout.addLayout(return_strategy_layout)
        main_layout.addStretch()
        group.setLayout(main_layout)
        return group

    def create_btn_layout(self):
        layout = QHBoxLayout()
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        default_btn = QPushButton("设为默认")
        default_btn.clicked.connect(self.on_click_default_btn)
        layout.addWidget(default_btn)
        layout.addStretch()
        layout.addWidget(ok_btn)
        return layout

    def on_metric_toggled(self):
        is_checked = self.enable_metric_checkbox.isChecked()
        self.similarity_metric_combo.setEnabled(is_checked)

    def on_normalization_toggled(self):
        is_checked = self.normalization_checkbox.isChecked()
        self.normalization_combo.setEnabled(is_checked)

    def on_click_extract_btn(self):
        dlg = AudioClipExtractionDialog(save_clip=True, dialog_title="选择模板片段")
        _, clip_path, clip_len = dlg.on_exec()
        if clip_path is not None:
            self.pattern_list.append({"clip_path": clip_path, "clip_len": clip_len})
            self.refresh_data_view()

    def on_click_add_btn(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "选择音频文件", DEFAULT_DIR + "audio_data/pattern/", "音频文件 (*.wav)")
        for file_name in file_names:
            relative_path = FileOps.get_relative_path(file_name, DEFAULT_DIR)
            pattern_data, _ = load_audio_simple(file_name, self.sample_rate)
            self.pattern_list.append({"clip_path": relative_path, "clip_len": len(pattern_data)})
        self.refresh_data_view()

    def on_click_remove_btn(self):
        row_idx = self.data_view.currentIndex().row()
        self.pattern_list.pop(row_idx)
        self.refresh_data_view()

    def on_algorithm_changed(self):
        algo_key = self.algorithm_combo.currentData()
        has_params = algo_key and self.algorithm_registry[algo_key].get('params')
        self.algorithm_params_btn.setEnabled(bool(has_params))

    def on_click_algorithm_params(self):
        algo_key = self.algorithm_combo.currentData()
        if not algo_key or not self.algorithm_registry[algo_key].get('params'):
            QMessageBox.information(self, "提示", "当前算法没有可配置的参数。")
            return

        param_definitions = self.algorithm_registry[algo_key]['params']
        current_values = self.algorithm_params.get(algo_key, {})
        dialog = GenericFeatureParamsDialog(param_definitions, current_values)
        if dialog.exec_() == QDialog.Accepted:
            self.algorithm_params[algo_key] = dialog.get_params()

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

        self.update_algorithm_combo()

    def on_strategy_radio_changed(self):
        is_fixed_checked = self.fixed_threshold_radio.isChecked()
        self.threshold_spinbox.setEnabled(is_fixed_checked)
        self.auto_calc_threshold_btn.setEnabled(is_fixed_checked)

    def on_filter_toggled(self):
        is_checked = self.filter_checkbox.isChecked()
        self.low_freq_edit.setEnabled(is_checked)
        self.high_freq_edit.setEnabled(is_checked)


    def update_algorithm_combo(self):
        """Update dropdown box based on feature dimensions algorithm"""
        feature_key = self.feature_combo.currentData()
        if not feature_key:
            return

        feature_info = self.feature_registry.get(feature_key, {})
        feature_dim = feature_info.get("dimensionality")
        view = self.algorithm_combo.view()

        for i in range(self.algorithm_combo.count()):
            algo_key = self.algorithm_combo.itemData(i)
            algo_info = self.algorithm_registry.get(algo_key, {})
            is_compatible = feature_dim in algo_info.get("compatibility", ())
            view.setRowHidden(i, not is_compatible)

        if view.isRowHidden(self.algorithm_combo.currentIndex()):
            for i in range(self.algorithm_combo.count()):
                if not view.isRowHidden(i):
                    self.algorithm_combo.setCurrentIndex(i)
                    break

    def populate_ui_from_config(self, config):
        if not config:
            return

        feature_type = config.get("feature_type")
        if feature_type:
            index = self.feature_combo.findData(feature_type)
            if index >= 0:
                self.feature_combo.setCurrentIndex(index)

        feature_params = config.get("feature_params")
        if feature_params and feature_type in self.feature_params:
            self.feature_params[feature_type] = feature_params

        apply_filter = config.get("apply_filter", False)
        self.filter_checkbox.setChecked(apply_filter)
        if apply_filter:
            filter_range = config.get("filter_range_hz", [0, 5000])
            self.low_freq_edit.setText(str(filter_range[0]))
            self.high_freq_edit.setText(str(filter_range[1]))

        apply_norm = config.get("apply_normalization", True)
        self.normalization_checkbox.setChecked(apply_norm)
        if apply_norm:
            norm_type = config.get("normalization_type", "peak")
            norm_index = self.normalization_combo.findData(norm_type)
            if norm_index >= 0:
                self.normalization_combo.setCurrentIndex(norm_index)

        algorithm = config.get("algorithm")
        if algorithm:
            index = self.algorithm_combo.findData(algorithm)
            if index >= 0:
                self.algorithm_combo.setCurrentIndex(index)

        algo_params = config.get("algorithm_params")
        if algorithm and algo_params:
            self.algorithm_params[algorithm] = algo_params

        strategy = config.get("threshold_strategy")
        if strategy == "adaptive_threshold":
            self.adaptive_threshold_radio.setChecked(True)
        else:
            self.fixed_threshold_radio.setChecked(True)
            threshold_value = config.get("threshold_value", 90.0)
            self.threshold_spinbox.setValue(threshold_value)
        
        # Handle auto equal length checkbox
        auto_equal_length = config.get("auto_equal_length", False)
        print(auto_equal_length)
        # if hasattr(self, "_auto_equal_chk"):
        self._auto_equal_chk.setChecked(auto_equal_length)


    def refresh_data_view(self):
        self.data_view.model().setRowCount(0)
        for idx, pattern in enumerate(self.pattern_list):
            self.data_view.model().setItem(idx, 0, QStandardItem(pattern["clip_path"]))
            pattern_len = np.round(pattern["clip_len"] / self.sample_rate, 3)
            self.data_view.model().setItem(idx, 1, QStandardItem(str(pattern_len)))
        self.n_chosen_pattern_label.setText("已加载模板： %s" % len(self.pattern_list))
        self.data_view.horizontalHeader().setSectionResizeMode(3)
        width = self.data_view.columnWidth(0)
        self.data_view.horizontalHeader().setSectionResizeMode(0)
        self.data_view.setColumnWidth(0, width + 60)

    def get_config(self):
        feature_key = self.feature_combo.currentData()
        algo_key = self.algorithm_combo.currentData()
        strategy = "fixed_threshold"
        if self.adaptive_threshold_radio.isChecked():
            strategy = "adaptive_threshold"
        config = {
            "sample_rate": self.sample_rate,
            "feature_type": feature_key,
            "feature_params": self.feature_params.get(feature_key, {}),
            "apply_normalization": self.normalization_checkbox.isChecked(),
            "normalization_type": self.normalization_combo.currentData() if self.normalization_checkbox.isChecked() else None,
            "apply_filter": self.filter_checkbox.isChecked(),
            "filter_range_hz": (None, None),
            "algorithm": algo_key,
            "algorithm_params": self.algorithm_params.get(algo_key, {}),
            "threshold_strategy": strategy,
            "threshold_value": None,
            "auto_equal_length": bool(self._auto_equal_chk.isChecked()) if hasattr(self, "_auto_equal_chk") else False,
        }
        if config["apply_filter"]:
            config["filter_range_hz"] = (int(self.low_freq_edit.text()), int(self.high_freq_edit.text()))
        if config["threshold_strategy"] == "fixed_threshold":
            config["threshold_value"] = self.threshold_spinbox.value()
        return config

    def on_auto_calculate_threshold(self):
        if len(self.pattern_list) < 2:
            QMessageBox.warning(self, "提示", "自动计算阈值至少需要2个OK模板。")
            return

        reply = QMessageBox.information(self, "提示",
                                        "即将根据当前加载的模板和选择的特征计算阈值，过程可能需要一些时间，是否继续？",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return

        current_ui_config = self.get_config()
        current_ui_config["pattern_list"] = self.pattern_list
        distance_threshold = PatternMatch.calculate_loocv_threshold(current_ui_config, self.sample_rate)

        if distance_threshold is None:
            QMessageBox.critical(self, "错误", "阈值计算失败，请检查模板文件或控制台输出。")
            return

        score = 1 / (1 + distance_threshold)
        score_percent = score * 100.0

        self.threshold_spinbox.setValue(score_percent)
        QMessageBox.information(self, "完成",
                                f"计算完成！\n\n建议的阈值为: {score_percent:.2f} %\n\n该值已自动填入输入框。")

    def get_patterns_from_view(self):
        patterns = []
        model = self.data_view.model
        for row in range(model.rowCount()):
            path_item, len_item = model.item(row, 0), model.item(row, 1)
            if path_item and len_item:
                try:
                    clip_len = int(float(len_item.text()) * self.sample_rate)
                    patterns.append({"clip_path": path_item.text(), "clip_len": clip_len})
                except (ValueError, TypeError):
                    pass
        return patterns

    def on_click_default_btn(self):
        current_ui_params = self.get_config()
        if self.radio_ch1.isChecked():
            self.config_ch1 = current_ui_params
            self.config_ch1['pattern_list'] = self.pattern_list
        else:
            self.config_ch2 = current_ui_params
            self.config_ch2['pattern_list'] = self.pattern_list

        if not self.validate_config(self.config_ch1) or not self.validate_config(self.config_ch2):
            return

        full_config_data = {"channel_1_config": self.config_ch1,
                            "channel_2_config": self.config_ch2}

        save_flag = self.config_manager.save_default_config("PM", full_config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        current_ui_params = self.get_config()
        if self.radio_ch1.isChecked():
            self.config_ch1 = current_ui_params
            self.config_ch1['pattern_list'] = self.pattern_list
        else:
            self.config_ch2 = current_ui_params
            self.config_ch2['pattern_list'] = self.pattern_list

        if not self.validate_config(self.config_ch1) or not self.validate_config(self.config_ch2):
            return None

        self.config_data = {"channel_1_config": self.config_ch1,
                            "channel_2_config": self.config_ch2}
        self.accept()
        return self.config_data

    def on_channel_selection_changed(self, toggled_id):
        previous_id = 3 - toggled_id

        prev_params = self.get_config()

        if previous_id == 1:
            self.config_ch1 = prev_params
            self.config_ch1['pattern_list'] = self.pattern_list
        else:
            self.config_ch2 = prev_params
            self.config_ch2['pattern_list'] = self.pattern_list

        config_to_load = self.config_ch1 if toggled_id == 1 else self.config_ch2

        self.populate_ui_from_config(config_to_load)
        self.pattern_list = config_to_load.get("pattern_list", []).copy()
        self.refresh_data_view()

    def initial_populate_ui(self):
        self.populate_ui_from_config(self.config_ch1)
        self.refresh_data_view()
        self.on_filter_toggled()
        self.on_strategy_radio_changed()
        self.on_normalization_toggled()
        self.on_feature_type_changed()
        self.on_algorithm_changed()

    def validate_config(self, config):
        if config.get("apply_filter", False):
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


class PipelineConfigWindow(QDialog):
    """
    pipeline configuration window (for inheritance)

    - select and jump to configure "前项分析" and "后项分析"
    - when saving, merge the configurations of the two analyses into the pipeline itself
    This class should be used for inheritance
    """

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        # full configuration dictionary (analysis_list)
        self.all_config = self.config_manager.load_config()
        # the saved configuration of this item
        self.load_config = self.all_config.get(model_type, {}) if isinstance(self.all_config, dict) else {}

        self.init_ui()
        self._hydrate_from_saved()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(720, 360)
        self.resize(760, 380)

        root_layout = QVBoxLayout()

        # two buttons (set by subclass)
        select_group = QGroupBox("管道节点配置")
        col_btns = QVBoxLayout()
        self.btn_head_cfg = QPushButton("配置前项…")
        self.btn_tail_cfg = QPushButton("配置后项…")
        self.btn_head_cfg.setEnabled(False)
        self.btn_tail_cfg.setEnabled(False)
        arrow_label = QLabel("↓")
        arrow_label.setAlignment(Qt.AlignCenter)
        try:
            arrow_label.setStyleSheet("font-size: 22px; color: rgb(120,120,120);")
        except Exception:
            pass
        col_btns.addStretch()
        col_btns.addWidget(self.btn_head_cfg, 0, Qt.AlignCenter)
        col_btns.addWidget(arrow_label, 0, Qt.AlignCenter)
        col_btns.addWidget(self.btn_tail_cfg, 0, Qt.AlignCenter)
        col_btns.addStretch()
        col_btns.setSpacing(8)
        select_group.setLayout(col_btns)
        # record the group reference,便于子类改标题
        self._group_box = select_group

        # bottom buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)

        root_layout.addWidget(select_group)
        root_layout.addStretch()
        root_layout.addLayout(btn_layout)

        self.setLayout(root_layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qcheckbox_style
        )

        # local storage (only used inside the pipeline)
        self.head_local_type = None
        self.tail_local_type = None
        self.head_local_config = {}
        self.tail_local_config = {}

        # subclass window title (can be set by subclass or overridden)
        self._head_window_title = None
        self._tail_window_title = None

        # enabled after subclass calls set_types / set_button_texts
        self.btn_head_cfg.clicked.connect(self.on_click_head_cfg)
        self.btn_tail_cfg.clicked.connect(self.on_click_tail_cfg)

    def _hydrate_from_saved(self):
        """if the item has saved head/tail configuration, initialize it to the local cache, avoid overlapping/clearing."""
        if isinstance(self.load_config, dict):
            head = self.load_config.get("head", {})
            tail = self.load_config.get("tail", {})
            if isinstance(head, dict):
                if head.get("type"):
                    self.head_local_type = head.get("type")
                    # 深拷贝以避免外部引用覆盖
                    try:
                        self.head_local_config = dict(head.get("config", {}))
                    except Exception:
                        self.head_local_config = head.get("config", {})
                    self.btn_head_cfg.setEnabled(True)
                    # Restore head configuration to config_manager for persistence
                    self._restore_temp_config_to_manager("head")
            if isinstance(tail, dict):
                if tail.get("type"):
                    self.tail_local_type = tail.get("type")
                    try:
                        self.tail_local_config = dict(tail.get("config", {}))
                    except Exception:
                        self.tail_local_config = tail.get("config", {})
                    self.btn_tail_cfg.setEnabled(True)
                    # Restore tail configuration to config_manager for persistence
                    self._restore_temp_config_to_manager("tail")

    def set_types(self, head_type: str, tail_type: str):
        """由子类调用，设置首/尾分析类型（如 "SPL"、"PD" 等）。"""
        self.head_local_type = head_type
        self.tail_local_type = tail_type
        self.btn_head_cfg.setEnabled(bool(head_type))
        self.btn_tail_cfg.setEnabled(bool(tail_type))
        # Restore configurations to config_manager if they exist
        if self.head_local_config and head_type:
            self._restore_temp_config_to_manager("head")
        if self.tail_local_config and tail_type:
            self._restore_temp_config_to_manager("tail")

    def set_button_texts(self, head_text: str, tail_text: str):
        """由子类调用，设置按钮文案。"""
        if head_text:
            self.btn_head_cfg.setText(str(head_text))
        if tail_text:
            self.btn_tail_cfg.setText(str(tail_text))

    def set_group_title(self, title: str):
        """由子类或外部调用，设置分组标题（默认：管道节点配置）。"""
        if hasattr(self, "_group_box") and self._group_box and title:
            self._group_box.setTitle(str(title))

    def set_child_window_titles(self, head_title: str = None, tail_title: str = None):
        """由子类调用，设置打开的首/尾配置窗体标题。"""
        self._head_window_title = head_title
        self._tail_window_title = tail_title

    def _get_slot_model_name(self, slot: str) -> str:
        """生成子窗体使用的名称/标题，子类可重写以自定义。"""
        if slot == "head" and self._head_window_title:
            return str(self._head_window_title)
        if slot == "tail" and self._tail_window_title:
            return str(self._tail_window_title)
        return f"PIPE_TMP_{slot.upper()}"

    def _create_child_dialog_by_type(self, a_type: str, model_name: str) -> QDialog:
        # 这里复用各分析项配置窗口（与 OptionList.create_config_dialog 一致）
        if a_type == "SPL":
            return SplConfigWindow(self.config_manager, model_name)
        elif a_type == "FR":
            return FrConfigWindow(self.config_manager, model_name)
        elif a_type == "HD":
            return HdConfigWindow(self.config_manager, model_name)
        elif a_type == "AI":
            return AIConfigWindow(self.config_manager, model_name, 0)
        elif a_type == "Spec":
            return SpecConfigWindow(self.config_manager, model_name)
        elif a_type == "LP":
            return LPConfigWindow(self.config_manager, model_name)
        elif a_type == "PD":
            return PDTabbedPDConfigWindow(self.config_manager, model_name)
        elif a_type == "PM":
            return PatternMatchConfigWindow(self.config_manager, model_name)
        else:
            # 未知类型，返回空对话框
            return QDialog(self)

    def _open_and_capture_local(self, a_type: str, slot: str):
        # 使用临时名称承载配置，不污染 analysis_list
        temp_name = self._get_slot_model_name(slot)
        # 在打开子窗体前，用本地缓存预填充到 config_manager.config
        self._prefill_temp_config_to_manager(slot, temp_name)
        dialog = self._create_child_dialog_by_type(a_type, temp_name)
        dialog.setWindowTitle(temp_name)
        if dialog.exec_() == QDialog.Accepted:
            try:
                updated = dialog.on_click_ok_btn()
            except Exception:
                updated = None
            if isinstance(updated, dict):
                if slot == "head":
                    self.head_local_type = a_type
                    self.head_local_config = updated
                else:
                    self.tail_local_type = a_type
                    self.tail_local_config = updated
                # 同步更新到 config_manager.config，便于下次再次打开时保留填写
                self._write_back_temp_config(slot, temp_name, updated)

    def _open_and_update_child(self, slot: str):
        # 按类型打开，名称使用临时占位
        a_type = self.head_local_type if slot == "head" else self.tail_local_type
        if not a_type:
            QMessageBox.information(self, "提示", "未设置该节点的分析类型。请在子类中调用 set_types 设置。")
            return
        self._open_and_capture_local(a_type, slot)

    def _prefill_temp_config_to_manager(self, slot: str, temp_name: str):
        """把本地缓存的 head/tail 配置写入到临时存储以便子窗体读取。
        使用独立的临时存储，不污染主配置。"""
        # Create separate temporary storage for pipeline configs
        if not hasattr(self.config_manager, "_pipeline_temp_storage"):
            self.config_manager._pipeline_temp_storage = {}
        
        local_cfg = self.head_local_config if slot == "head" else self.tail_local_config
        if isinstance(local_cfg, dict) and local_cfg:
            # 使用副本，避免子窗体原地修改带来意外引用问题
            try:
                self.config_manager._pipeline_temp_storage[temp_name] = dict(local_cfg)
            except Exception:
                self.config_manager._pipeline_temp_storage[temp_name] = local_cfg
        else:
            # 确保有键，哪怕是空 dict
            self.config_manager._pipeline_temp_storage.setdefault(temp_name, {})

    def _restore_temp_config_to_manager(self, slot: str):
        """Restore saved configuration to temporary storage for temporary name access"""
        temp_name = self._get_slot_model_name(slot)
        local_cfg = self.head_local_config if slot == "head" else self.tail_local_config
        if not hasattr(self.config_manager, "_pipeline_temp_storage"):
            self.config_manager._pipeline_temp_storage = {}
        if isinstance(local_cfg, dict) and local_cfg:
            try:
                self.config_manager._pipeline_temp_storage[temp_name] = dict(local_cfg)
            except Exception:
                self.config_manager._pipeline_temp_storage[temp_name] = local_cfg
        else:
            self.config_manager._pipeline_temp_storage.setdefault(temp_name, {})

    def _write_back_temp_config(self, slot: str, temp_name: str, updated: dict):
        """子窗体关闭后，把最新配置回写到临时存储，用于后续再次打开预填。"""
        if not hasattr(self.config_manager, "_pipeline_temp_storage"):
            self.config_manager._pipeline_temp_storage = {}
        self.config_manager._pipeline_temp_storage[temp_name] = dict(updated) if isinstance(updated, dict) else {}

    def on_click_head_cfg(self):
        self._open_and_update_child("head")

    def on_click_tail_cfg(self):
        self._open_and_update_child("tail")

    def get_default_config(self):
        # 如果未配置过，返回空配置但带类型（类型需由子类 set_types 提供）
        return {
            "type": "ED",
            "head": {"type": self.head_local_type, "config": self.head_local_config},
            "tail": {"type": self.tail_local_type, "config": self.tail_local_config},
        }

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class PipelinePdPmConfigWindow(PipelineConfigWindow):
    """PeakDetection -> PatternMatch pipeline configuration window"""

    def __init__(self, config_manager, model_type):
        super().__init__(config_manager, model_type)
        self.set_types("PD", "PM")
        self.set_button_texts("配置峰值检测参数", "配置模式匹配参数")
        self.set_child_window_titles("峰值检测参数", "模式匹配参数")
        self.set_group_title("峰值检测 -> 模式匹配")
        self._init_length_group()
        self._init_pass_condition_group()

    def _init_pass_condition_group(self):
        # pass condition: n1 ≤ matched points ≤ n2
        root_layout = self.layout()
        if not root_layout:
            return
        pass_group = QGroupBox("通过条件")
        row = QHBoxLayout()
        label_prefix = QLabel("通过条件：")
        label_mid = QLabel("≤ 匹配点数 ≤")
        self._n1_spin = QSpinBox()
        self._n2_spin = QSpinBox()
        for sp in (self._n1_spin, self._n2_spin):
            sp.setRange(0, 1000000)
            sp.setSingleStep(1)
        # 默认值
        self._n1_spin.setValue(1)
        self._n2_spin.setValue(1)
        # 约束：n2 >= n1
        def on_n1_changed(val):
            if self._n2_spin.value() < val:
                self._n2_spin.setValue(val)
        def on_n2_changed(val):
            if val < self._n1_spin.value():
                self._n2_spin.setValue(self._n1_spin.value())
        self._n1_spin.valueChanged.connect(on_n1_changed)
        self._n2_spin.valueChanged.connect(on_n2_changed)

        row.addWidget(label_prefix)
        row.addWidget(self._n1_spin)
        row.addSpacing(6)
        row.addWidget(label_mid)
        row.addSpacing(6)
        row.addWidget(self._n2_spin)
        row.addStretch()
        pass_group.setLayout(row)

        try:
            # insert after length configuration
            root_layout.insertWidget(2, pass_group)
        except Exception:
            root_layout.addWidget(pass_group)

        try:
            if isinstance(self.load_config, dict):
                cond = self.load_config.get("pass_condition", {})
                if isinstance(cond, dict):
                    n1 = int(cond.get("n1", self._n1_spin.value()))
                    n2 = int(cond.get("n2", self._n2_spin.value()))
                    self._n1_spin.setValue(max(0, n1))
                    self._n2_spin.setValue(max(self._n1_spin.value(), n2))
        except Exception:
            pass

    def _init_length_group(self):
        root_layout = self.layout()
        if not root_layout:
            return
        length_group = QGroupBox("长度控制")
        vbox = QVBoxLayout()

        # first row: left/right grid points (include peak point)
        row1 = QHBoxLayout()
        lbl_l = QLabel("左侧格点数")
        lbl_r = QLabel("右侧格点数")
        self._left_grid_spin = QSpinBox()
        self._right_grid_spin = QSpinBox()
        for sp in (self._left_grid_spin, self._right_grid_spin):
            sp.setRange(0, 9999999)
            sp.setSingleStep(1)
        row1.addWidget(lbl_l)
        row1.addWidget(self._left_grid_spin)
        row1.addSpacing(12)
        row1.addWidget(lbl_r)
        row1.addWidget(self._right_grid_spin)
        row1.addStretch()

        # load existing configuration
        try:
            if isinstance(self.load_config, dict):
                lg = int(self.load_config.get("left_grid", 0) or 0)
                rg = int(self.load_config.get("right_grid", 0) or 0)
                self._left_grid_spin.setValue(max(0, lg))
                self._right_grid_spin.setValue(max(0, rg))
        except Exception:
            pass

        vbox.addLayout(row1)
        length_group.setLayout(vbox)

        try:
            # insert after button group, before pass condition
            root_layout.insertWidget(1, length_group)
        except Exception:
            root_layout.addWidget(length_group)

    def get_default_config(self):
        cfg = super().get_default_config()
        # pipeline itself configuration
        cfg["left_grid"] = int(self._left_grid_spin.value()) if hasattr(self, "_left_grid_spin") else 0
        cfg["right_grid"] = int(self._right_grid_spin.value()) if hasattr(self, "_right_grid_spin") else 0
        # pass condition
        cfg["pass_condition"] = {
            "n1": int(self._n1_spin.value()) if hasattr(self, "_n1_spin") else 1,
            "n2": int(self._n2_spin.value()) if hasattr(self, "_n2_spin") else 1,
        }
        return cfg

    def on_click_ok_btn(self):
        n1 = int(self._n1_spin.value()) if hasattr(self, "_n1_spin") else 1
        n2 = int(self._n2_spin.value()) if hasattr(self, "_n2_spin") else 1
        if n2 < n1:
            QMessageBox.warning(self, "设置警告", "n2 应该大于等于 n1")
            return None
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
    window = PatternMatchConfigWindow(config_manager, "PM")
    window.show()
    app.exec_()
