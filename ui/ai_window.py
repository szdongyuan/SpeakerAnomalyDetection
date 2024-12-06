import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QGroupBox, QLabel, QApplication, QComboBox, QVBoxLayout, QMessageBox
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QTextEdit, QWidget, QPushButton

from base.training_model_management import TrainingModelManagement
from consts import error_code
from main import init_model_from_config


class AiWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.model_path = self.load_model_path_from_config()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI分析窗口")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        base_model_wdiget = BaseModel()
        layout = QHBoxLayout()
        layout.addWidget(base_model_wdiget)
        self.setLayout(layout)

    @staticmethod
    def load_model_path_from_config():
        file_path = "ui_config/model_path.txt"
        with open(file_path, 'r') as f:
            model_path = f.read()
        return model_path


class BaseModel(QWidget):
    def __init__(self):
        super().__init__()
        self.load_model = self.load_model_name_from_db()
        self.model_path = ""
        self.init_ui()

    def init_ui(self):
        base_model_layout = QVBoxLayout()
        base_model_box = self.create_model_layout()
        base_model_layout.addWidget(base_model_box)
        self.setLayout(base_model_layout)

    def create_model_layout(self):
        base_model_box = QGroupBox("模型")
        base_model_box.setFixedSize(500, 500)
        base_model_label = QLabel("基础模型:")
        self.base_model_combo_box = QComboBox(self)
        for model_name in self.load_model:
            self.base_model_combo_box.addItem(model_name)
        self.base_model_combo_box.currentIndexChanged.connect(self.combobox_clicked)
        new_model_btn = QPushButton("新建模型")
        self.text_edit = QTextEdit(self)
        self.text_edit.setFixedSize(450, 400)
        self.text_edit.setVisible(False)
        self.text_edit.setReadOnly(True)
        self.text_edit.setAlignment(Qt.AlignCenter)
        v_space_1 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_space_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        base_model_combo_layout = QHBoxLayout()

        base_model_combo_layout.addWidget(base_model_label)
        base_model_combo_layout.addWidget(self.base_model_combo_box)
        base_model_combo_layout.addWidget(new_model_btn)

        base_model_layout = QVBoxLayout()
        base_model_layout.addLayout(base_model_combo_layout)
        base_model_layout.addItem(v_space_1)
        base_model_layout.addWidget(self.text_edit)
        base_model_layout.addItem(v_space_2)
        base_model_box.setLayout(base_model_layout)
        return base_model_box

    def combobox_clicked(self):
        selected_model = self.base_model_combo_box.currentText()
        if selected_model:
            self.text_edit.setVisible(True)
            model = self.load_model_structure(selected_model)
            self.save_model_path_to_config()
            if model is not None:
                model_summary = []
                model.model.summary(print_fn=lambda x: model_summary.append(x))
                summary_text = "\n".join(model_summary)
                self.text_edit.setPlainText(summary_text)
            else:
                self.text_edit.setPlainText(str(model))
        else:
            self.text_edit.setVisible(False)

    @staticmethod
    def load_model_name_from_db():
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
        if query_code == error_code.OK:
            for idx, name in enumerate(query_result):
                query_result_idx = query_result[idx]
                model_list.append(query_result_idx[0])
        return model_list

    def load_model_structure(self, selected_model):
        query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
        model = None
        if query_code == error_code.OK:
            self.model_path, config_path = query_result[0]
            kwargs = {"config_path": config_path}
            model = init_model_from_config(**kwargs)
            model.load_model(self.model_path)
        return model

    def save_model_path_to_config(self):
        dir_path = 'ui_config'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        file_path = dir_path + "/" + "model_path.txt"
        with open(file_path, 'w') as f:
            f.write(self.model_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AiWindow()
    window.show()
    window.exec()




