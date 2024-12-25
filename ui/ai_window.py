import os
import sys
from PyQt5.QtCore import QEventLoop, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QTextCursor
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QGroupBox
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QSplitter, QTextEdit, QVBoxLayout, QWidget

from base.training_model_management import TrainingModelManagement
from consts import error_code, ui_style_const
from main import evaluate, init_model_from_config, train


class AiWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.train_dir = self.load_default_train_test_path("train")
        self.test_dir = self.load_default_train_test_path("evaluate")
        self.model_path = self.load_model_path_from_config()

        self.init_ui()
        self.th = TrainEvaluateThread(train_dir=self.train_dir,
                                      test_dir=self.test_dir,
                                      model_path=self.model_path,
                                      mode="train")
        self.th.signalForText.connect(self.on_update_text)
        sys.stdout = self.th

    def on_update_text(self, text):
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        if text == "\r":
            cursor.insertText("66666")
            # cursor.movePosition(QTextCursor.StartOfLine)
            # cursor.movePosition(QTextCursor.PreviousRow)
            # cursor.movePosition(QTextCursor.StartOfLine)
        else:
            cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def init_ui(self):
        self.setWindowTitle("AI训练窗口")
        # self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        # self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        ai_layout = QGridLayout()

        base_model_wdiget = BaseModel()
        base_model_layout = QHBoxLayout()
        base_model_layout.addWidget(base_model_wdiget)

        self.process = QTextEdit(self, readOnly=True)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(800)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.process.setMinimumSize(800, 600)

        train_group_box = QGroupBox("训练")
        train_btn_layout = QGridLayout()
        train_dir_label = QLabel("训练数据路径：")
        # train_dir_btn.clicked.connect(self.train_dir_btn_clicked)
        self.train_dir_box = QLineEdit()
        train_dir_icon = QIcon("./ui_pic/ai_window_pic/folder-s.png")
        train_dir_action = self.train_dir_box.addAction(train_dir_icon, QLineEdit.TrailingPosition)
        train_dir_action.setToolTip("添加训练数据")
        train_dir_action.triggered.connect(self.train_dir_btn_clicked)
        self.train_dir_box.setText(self.train_dir)
        self.train_btn = QPushButton(" 开始训练 ")
        self.train_btn.setStyleSheet("padding: 5px")
        self.train_btn.clicked.connect(self.train_btn_clicked)
        train_btn_layout.addWidget(train_dir_label, 0, 0)
        train_btn_layout.addWidget(self.train_dir_box, 0, 1)
        train_btn_layout.addWidget(self.train_btn, 0, 2)
        train_btn_layout.setSpacing(20)
        train_group_box.setLayout(train_btn_layout)

        evaluate_group_box = QGroupBox("评估")
        evaluate_btn_layout = QGridLayout()
        evaluate_dir_label = QLabel("测试数据路径：")
        self.evaluate_dir_box = QLineEdit()
        self.evaluate_dir_box.setPlaceholderText("请选择测试数据路径")
        evaluate_dir_icon = QIcon("./ui_pic/ai_window_pic/folder-s.png")
        evaluate_dir_action = self.evaluate_dir_box.addAction(evaluate_dir_icon, QLineEdit.TrailingPosition)
        evaluate_dir_action.setToolTip("添加测试数据")
        evaluate_dir_action.triggered.connect(self.evaluate_dir_btn_clicked)
        self.evaluate_dir_box.setText(self.test_dir)
        self.evaluate_btn = QPushButton(" 开始评估 ")
        self.evaluate_btn.setStyleSheet("padding: 5px")
        self.evaluate_btn.clicked.connect(self.evaluate_btn_clicked)
        evaluate_btn_layout.addWidget(evaluate_dir_label, 0, 0)
        evaluate_btn_layout.addWidget(self.evaluate_dir_box, 0, 1)
        evaluate_btn_layout.addWidget(self.evaluate_btn, 0, 2)
        evaluate_btn_layout.setSpacing(20)
        evaluate_group_box.setLayout(evaluate_btn_layout)

        btn_function_layout = QVBoxLayout()
        btn_function_layout.addWidget(train_group_box)
        btn_function_layout.addWidget(evaluate_group_box)
        btn_function_layout.addWidget(self.process)

        splitter = QSplitter(Qt.Horizontal)
        base_model_frame = QFrame()
        base_model_frame.setLayout(base_model_layout)
        base_model_layout.setContentsMargins(0,0,0,0)
        btn_function_frame = QFrame()
        btn_function_frame.setLayout(btn_function_layout)
        splitter.addWidget(base_model_frame)
        splitter.addWidget(btn_function_frame)
        splitter.setHandleWidth(0)
        ai_layout.addWidget(splitter)

        self.setLayout(ai_layout)
        self.setStyleSheet(ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qcombobox_stytle)

    def train_dir_btn_clicked(self):
        path = QFileDialog.getExistingDirectory(self,
                                                "选择训练数据目录",
                                                "../audio_data",
                                                options=QFileDialog.DontUseNativeDialog)
        if path:
            self.train_dir = path
            self.save_default_train_test_path(path, "train")
            self.train_dir_box.setText(path)

    def evaluate_dir_btn_clicked(self):
        path = QFileDialog.getExistingDirectory(self,
                                                "选择评估数据目录",
                                                "../audio_data",
                                                options=QFileDialog.DontUseNativeDialog)
        if path:
            self.test_dir = path
            self.save_default_train_test_path(path, "evaluate")
            self.evaluate_dir_box.setText(path)

    def train_btn_clicked(self):
        print("start training model...")
        try:
            self.t = TrainEvaluateThread(train_dir=self.train_dir,
                                         test_dir=self.test_dir,
                                         model_path=self.model_path,
                                         mode="train")
            self.t.start()
        except Exception as e:
            raise e

        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()

    def evaluate_btn_clicked(self):
        print("start evaluating model...")
        try:
            self.t = TrainEvaluateThread(train_dir=self.train_dir,
                                         test_dir=self.test_dir,
                                         model_path=self.model_path,
                                         mode="evaluate")
            self.t.start()
        except Exception as e:
            raise e

        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(174, 171, 162, 123))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        super().paintEvent(event)

    @staticmethod
    def load_model_path_from_config():
        file_path = "ui_config/model_path.txt"
        with open(file_path, 'r') as f:
            model_path = f.read()
        return model_path

    @staticmethod
    def save_default_train_test_path(path, mode="train"):
        file_path = "ui_config/%s_data_path.txt" % mode
        with open(file_path, 'w') as f:
            f.write(path)

    @staticmethod
    def load_default_train_test_path(mode="train"):
        path_file = "ui_config/%s_data_path.txt" % mode
        if not os.path.exists(path_file):
            return ""
        with open(path_file, 'r') as f:
            path = f.read()
        return path


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
        base_model_box.setMinimumSize(500, 500)
        base_model_label = QLabel("基础模型:")
        self.base_model_combo_box = QComboBox(self)
        for model_name in self.load_model:
            self.base_model_combo_box.addItem(model_name)
        self.base_model_combo_box.currentIndexChanged.connect(self.combobox_clicked)
        new_model_btn = QPushButton(" 新建模型 ")
        new_model_btn.setStyleSheet("padding: 5px")
        self.text_edit = QTextEdit(self)
        self.text_edit.setMinimumSize(450, 400)
        self.text_edit.setVisible(True)
        self.text_edit.setReadOnly(True)
        self.text_edit.setAlignment(Qt.AlignCenter)
        self.set_default_model()
        base_model_combo_layout = QHBoxLayout()

        base_model_combo_layout.addWidget(base_model_label)
        base_model_combo_layout.addWidget(self.base_model_combo_box)
        base_model_combo_layout.addWidget(new_model_btn)

        base_model_layout = QVBoxLayout()
        base_model_layout.addLayout(base_model_combo_layout)
        base_model_layout.addWidget(self.text_edit)
        base_model_layout.setSpacing(20)
        base_model_box.setLayout(base_model_layout)
        return base_model_box

    def combobox_clicked(self):
        selected_model = self.base_model_combo_box.currentText()
        if selected_model:
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

    def set_default_model(self):
        default_model_path = self.load_default_model_path()
        if default_model_path:
            default_model_name = os.path.basename(default_model_path)
            default_model_name = os.path.splitext(default_model_name)[0]
            if default_model_name in self.load_model:
                default_index = self.load_model.index(default_model_name)
                self.base_model_combo_box.setCurrentIndex(default_index)
                self.combobox_clicked()

    @staticmethod
    def load_default_model_path():
        file_path = "ui_config/model_path.txt"
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return ""
        try:
            with open(file_path, 'r') as f:
                model_path = f.read().strip()
                return model_path
        except Exception as e:
            return ""

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


class TrainEvaluateThread(QThread):

    signalForText = pyqtSignal(str)

    def __init__(self, data=None, parent=None,
                 train_dir=None, test_dir=None, model_path=None, mode="train"):
        super().__init__(parent)
        self.data = data
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_path = model_path
        self.mode = mode

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def flush(self):
        self.signalForText.emit("\r")

    def run(self):
        try:
            if self.mode == "train":
                train(self.train_dir, self.model_path, self.test_dir)
            elif self.mode == "evaluate":
                evaluate(self.test_dir, self.model_path, verbose=7)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AiWindow()
    window.show()
    window.exec()
