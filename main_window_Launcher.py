import sys

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMessageBox
from ui.splash_screen_window import Splash, LoaderThread


class MainWindowLauncher(object):
    def __init__(self):
        self.app = QApplication(sys.argv)
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
            QMessageBox.critical(None, "错误", msg)
            sys.exit(1)

    def on_error(self, msg):
        self.splash.lab.setText(f"启动失败")
        QMessageBox.critical(self.splash, "错误", msg)
        sys.exit(1)


if __name__ == "__main__":
    mwl = MainWindowLauncher()
    mwl.run()
