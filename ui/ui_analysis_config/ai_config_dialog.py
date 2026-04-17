from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QMessageBox, QComboBox, QSizePolicy

from base.training_model_management import TrainingModelManagement
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


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
        self.setMinimumSize(200, 300)
        layout = QVBoxLayout()
        model_box = self.create_model_layout()
        btn_layout = self.create_btn()
        layout.addWidget(model_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

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
