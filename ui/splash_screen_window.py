import time
import importlib

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRect
from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel
from consts import ui_style_const
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
        self.prg.setGeometry(QRect(20, 366, 591, 8))
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

        self.product_name_label = QLabel(self)
        self.product_name_label.setGeometry(QRect(30, 292, 350, 42))
        self.product_name_label.setStyleSheet(
            'background-color: white; color: black; '
            f'font-family: {ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; '
            'font-size: 22px;'
        )
        self.product_name_label.setText("欢迎使用希听异音检测系统")
        self.product_name_label.setObjectName("productNameLabel")

        self.lab = QLabel(self)
        self.lab.setGeometry(QRect(30, 344, 320, 20))
        self.lab.setStyleSheet(
            'color: rgb(0, 0, 0); '
            f'font-family: {ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; '
            'font-size: 12px;'
        )
        self.lab.setObjectName("lab")
        self.lab.setText("正在初始化...0%")

        self.label.raise_()
        self.product_name_label.raise_()
        self.prg.raise_()
        self.lab.raise_()

    def update_progress(self, message: str, step: int):
        self.prg.setValue(step)
        self.lab.setText(f"{message}... {step}%")

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            # Move the window
            self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mousePressEvent(self, event):
        # If the mouse is pressed, record mouse move data, start the window resizing
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()


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
