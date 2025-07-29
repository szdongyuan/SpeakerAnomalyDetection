import time
import importlib

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRect
from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel
from consts.running_consts import DEFAULT_DIR, MODULES_LOAD
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
        img_path = DEFAULT_DIR + "ui/ui_pic/splash_screen/splash_screen.png"
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
        self.modules_to_load = MODULES_LOAD

    def run(self):
        try:
            total = len(self.modules_to_load)
            percent = 0
            for idx, (desc, module_path) in enumerate(self.modules_to_load):
                self.progress.emit(desc, percent)
                if module_path:
                    try:
                        importlib.import_module(module_path)
                    except Exception as mod_err:
                        raise ImportError(f"模块{module_path}加载失败：{mod_err}")
                percent = int((idx + 1) / total * 100)
                time.sleep(0.01)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"{str(e)}")
