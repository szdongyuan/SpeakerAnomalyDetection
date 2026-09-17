"""Standalone --usb preview/recording tool, or explicit --simulate UI demo."""

import argparse
import multiprocessing
import os
from pathlib import Path
import sys


def configure_demo_font(app):
    from PyQt5.QtGui import QFont, QFontDatabase

    # Windows offscreen Qt may report an empty font database; use an installed font.
    if not QFontDatabase().families():
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei", 9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="fake backend, no video files")
    mode.add_argument("--usb", action="store_true", help="real USB camera and video files")
    parser.add_argument("--config", help="USB settings JSON path (default: project/exe ui/ui_config directory)")
    parser.add_argument("--disconnect-after", type=float, help="simulate one disconnect after recording N seconds")
    args = parser.parse_args(argv)

    # Keep spawn imports lightweight; only the GUI process imports Qt.
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from base.video.service import VideoService
    from base.video.worker import SimulationOptions, simulated_video_worker
    from ui.video_monitor_widget import VideoMonitorWidget
    from ui.video_controller import VideoServiceBridge

    class DemoWindow(VideoMonitorWidget):
        closing_requested = False

        def closeEvent(self, event):
            if getattr(self.bridge, "is_shutdown_complete", self.bridge.service.is_closed):
                event.accept()
            else:
                event.ignore()
                self.closing_requested = True
                self.record_button.setEnabled(False)
                self.bridge.shutdown()

        def service_closed(self):
            # A failed worker should leave the error visible, not close the GUI.
            if self.closing_requested:
                self.close()

    app = QApplication(sys.argv[:1])
    configure_demo_font(app)
    if args.usb:
        from ui.video_controller import VideoController
        bridge = VideoController(config_path=args.config)
    else:
        service = VideoService(
            worker_target=simulated_video_worker,
            worker_options=SimulationOptions(disconnect_after=args.disconnect_after),
        )
        bridge = VideoServiceBridge(service)
    window = DemoWindow(bridge, simulation=args.simulate)
    window.setWindowTitle("USB 摄像头预览与录像" if args.usb else "USB 视频交互演示（模拟模式，不保存视频）")
    window.resize(560, 350)
    if args.usb:
        bridge.attach_widget(window)
    else:
        window.settings_requested.connect(lambda: QMessageBox.information(
            window, "模拟模式", "本窗口用于验证人工启停与状态显示。\n"
            "不读取实际摄像头、不生成录像文件。\n真实模式请使用 --usb 启动。",
        ))
    bridge.closed.connect(window.service_closed)
    app.aboutToQuit.connect(bridge.shutdown)
    window.show()
    if args.simulate:
        service.start()
    return app.exec_()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
