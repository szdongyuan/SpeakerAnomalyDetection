import time
import importlib

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRect
from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel
from consts.running_consts import DEFAULT_DIR
from PyQt5.QtGui import QPixmap


class Splash(QWidget):
    splashClose = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def setupUi(self):
        self.setObjectName("Form")
        self.resize(633, 426)
        self.prg = QProgressBar(self)
        self.prg.setGeometry(QRect(20, 360, 591, 8))
        self.prg.setProperty("value", 0)
        self.prg.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self.prg.setTextVisible(False)
        self.prg.setObjectName("prg")

        self.label = QLabel(self)
        self.label.setGeometry(QRect(20, 0, 591, 411))
        img_path = DEFAULT_DIR + "ui/ui_pic/logo_pic/ting_splash.ico"
        pixmap = QPixmap(img_path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.lab = QLabel(self)
        self.lab.setGeometry(QRect(30, 340, 200, 16))
        self.lab.setStyleSheet("color: rgb(0, 0, 0);")
        self.lab.setObjectName("lab")
        self.lab.setText("正在初始化...0%")

        self.label.raise_()
        self.prg.raise_()
        self.lab.raise_()

    def update_progress(self, message: str, step: int):
        self.prg.setValue(step)
        self.lab.setText(f"{message}... {step}%")


class LoaderThread(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.modules_to_load = [
            ("加载格式", "consts.ui_style_const"),
            ("加载常量", "consts.model_consts"),
            ("加载路径", "consts.running_consts"),
            ("加载日志模块", "base.log_manager"),
            ("加载数据库模块", "base.db_manager"),
            ("加载 AI 模型训练窗口", "ui.ai_window"),
            ("加载 校准窗口", "ui.calibration_window"),
            ("加载 硬件选择窗口", "ui.hardware_window"),
            ("加载 分析流程窗口", "ui.sequence_widget"),
            ("加载 激励信号窗口", "ui.stimulus_window"),
            ("加载 模型选择窗口", "ui.analysis_model_sellect_dialog"),
            ("加载 登录窗口", "ui.login_window"),
            ("加载完成", None)
        ]

    def run(self):
        try:
            total = len(self.modules_to_load)
            for idx, (desc, module_path) in enumerate(self.modules_to_load):
                if module_path:
                    try:
                        importlib.import_module(module_path)
                    except Exception as mod_err:
                        raise ImportError(f"模块{module_path}加载失败：{mod_err}")
                percent = int((idx + 1) / total * 100)
                self.progress.emit(desc, percent)
                time.sleep(0.1)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"{str(e)}")

