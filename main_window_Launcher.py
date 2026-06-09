import os
import sys
from pathlib import Path


def ensure_qt_platform_plugin_path():
    qt_plugins_path = (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "PyQt5"
        / "Qt5"
        / "plugins"
    )
    qt_platforms_path = qt_plugins_path / "platforms"
    if qt_plugins_path.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins_path))
    if qt_platforms_path.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms_path))


ensure_qt_platform_plugin_path()

from PyQt5.QtCore import QThread, QFile, QTextStream
from PyQt5.QtWidgets import QApplication
from ui.splash_screen_window import Splash, LoaderThread

from ui.custom_ui_widget.widgets import MessageBox
from ui.ui_src import ui_resources


class MainWindowLauncher(object):
    def __init__(self):
        self.app = QApplication(sys.argv)
        qss = self.load_qss()
        self.app.setStyleSheet(qss)
        self.splash = Splash()
        self.splash.show()
        self.app.processEvents()

        self.loader = LoaderThread()
        self.loader_thread = QThread()
        self.loader.moveToThread(self.loader_thread)

        self.loader.progress.connect(self.splash.update_progress)
        self.loader.error.connect(self.on_error)
        self.loader.finished.connect(self.on_loader_finished)

        self.loader_thread.started.connect(self.loader.run)
        self.loader.finished.connect(self.loader_thread.quit)
        self.loader.finished.connect(self.loader.deleteLater)
        self.loader.finished.connect(self.loader_thread.deleteLater)

    def load_qss(self):
        #path = ":/ui/style/dongyuan_style.qss"
        path = ":/ui/style/jingcheng_style.qss"
        file = QFile(path)
        if not file.open(QFile.ReadOnly | QFile.Text):
            raise RuntimeError(f"Failed to open QSS: {path}")
        stream = QTextStream(file)
        qss = stream.readAll()
        file.close()
        return qss

    def run(self):
        self.loader_thread.start()
        sys.exit(self.app.exec())

    def on_loader_finished(self):
        try:
            self.splash.close()
            from main_window import MainWindow

            self.window = MainWindow()
            self.window.show()
        except Exception as e:
            msg = f"主界面启动失败：{str(e)}"
            MessageBox.critical(None, "错误", msg)
            sys.exit(1)

    def on_error(self, msg):
        self.splash.lab.setText(f"启动失败")
        MessageBox.critical(self.splash, "错误", msg)
        sys.exit(1)


if __name__ == "__main__":
    mwl = MainWindowLauncher()
    mwl.run()
