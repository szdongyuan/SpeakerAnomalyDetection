import sys

class MainWindowLauncher(object):
    def __init__(self):
        from PyQt5.QtCore import QThread
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import QApplication
        from consts import ui_style_const
        from ui.splash_screen_window import Splash, LoaderThread
        from base.recording_service import RecordingService
        from ui.recording_service_bridge import RecordingServiceBridge

        self.app = QApplication(sys.argv)
        self.app.setFont(QFont(ui_style_const.UI_FONT_FAMILY_NAME))
        self.recording_service = RecordingService()
        self.recording_bridge = RecordingServiceBridge(self.recording_service, self.app)
        self.app.aboutToQuit.connect(self.recording_bridge.shutdown)
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
        from PyQt5.QtWidgets import QMessageBox
        try:
            self.splash.close()
            from main_window import MainWindow
            self.window = MainWindow(recording_bridge=self.recording_bridge)
            self.window.show()
        except Exception as e:
            msg = f"主界面启动失败：{str(e)}"
            QMessageBox.critical(None, "错误", msg)
            self.recording_bridge.shutdown(lambda: self.app.exit(1))

    def on_error(self, msg):
        from PyQt5.QtWidgets import QMessageBox
        self.splash.lab.setText(f"启动失败")
        QMessageBox.critical(self.splash, "错误", msg)
        self.recording_bridge.shutdown(lambda: self.app.exit(1))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    mwl = MainWindowLauncher()
    mwl.run()
