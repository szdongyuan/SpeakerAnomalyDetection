"""Qt video bridge and application controller, including settings and device probing."""

import multiprocessing
from pathlib import Path
import sys
import threading
import time

from PyQt5.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImage
from PyQt5.QtWidgets import QMessageBox

from base.video.config import VideoConfig, load_config, migrate_config, save_config
from base.video.runtime import usb_video_worker
from base.video.service import VideoService
from base.video.usb_capture import probe_worker


class VideoServiceBridge(QObject):
    status_changed = pyqtSignal(object, int)
    preview_changed = pyqtSignal(QImage)
    closed = pyqtSignal()

    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self._last_status = None
        self._last_seconds = -1
        self._frame_sequence = 0
        self._closed_emitted = False
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.poll)
        self._timer.start()

    def poll(self):
        status = self.service.status
        seconds = status.elapsed(time.monotonic())
        if status != self._last_status or seconds != self._last_seconds:
            self._last_status, self._last_seconds = status, seconds
            self.status_changed.emit(status, seconds)
        frame = self.service.latest_preview(self._frame_sequence)
        if frame is not None:
            self._frame_sequence = frame.sequence
            # Own the bytes before returning; shared storage can change immediately.
            image = QImage(frame.rgb, frame.width, frame.height, frame.width * 3, QImage.Format_RGB888).copy()
            self.preview_changed.emit(image)
        if self.service.is_closed and not self._closed_emitted:
            self._closed_emitted = True
            self._timer.stop()
            self.closed.emit()

    def start_recording(self):
        return self.service.start_recording()

    def stop_recording(self):
        return self.service.stop_recording()

    def shutdown(self):
        self.service.shutdown()


def default_config_path():
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    return root / "ui" / "ui_config" / "video_settings.json"


class VideoController(VideoServiceBridge):
    devices_ready = pyqtSignal(object, str)
    configuration_saved = pyqtSignal(object, str)

    def __init__(self, parent=None, *, config_path=None, can_configure=lambda: True, start_automatically=True):
        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self.can_configure = can_configure
        error = ""
        try:
            if config_path is None:
                root = self.config_path.parents[2]
                migrate_config(root / "configs" / "video_settings.json", self.config_path)
            self.config = load_config(self.config_path)
        except (OSError, ValueError) as exc:
            self.config, error = VideoConfig(), f"视频配置读取或迁移失败：{exc}"
        self._generation = 1
        self._shutdown_requested = False
        self._pending_config = None
        self._probe_running = False
        self._saving = False
        self.dialog = None
        self.panel = None
        self.recent_directory = ""
        super().__init__(self._make_service(), parent)
        if error:
            self.service._fail(error)
        self.closed.connect(self._service_closed)
        self.devices_ready.connect(self._devices_ready)
        self.configuration_saved.connect(self._configuration_saved)
        self.status_changed.connect(self._remember_directory)
        if start_automatically:
            QTimer.singleShot(0, self.start_preview)

    def start_preview(self):
        if self.config.enabled and not self._shutdown_requested:
            self.service.start()

    def _make_service(self):
        return VideoService(
            worker_target=usb_video_worker, worker_options=self.config, generation=self._generation,
            initial_connection="connecting" if self.config.enabled else "disabled",
            heartbeat_timeout=8, command_timeout=20, shutdown_timeout=120, drain_timeout=30,
        )

    @property
    def is_shutdown_complete(self):
        return self.service.is_closed and not self._saving and not self._probe_running

    def attach_panel(self, placeholder):
        self.attach_widget(placeholder.bind_bridge(self))

    def attach_widget(self, widget):
        self.panel = widget
        self.panel.settings_requested.connect(self.show_settings)
        self.panel.directory_requested.connect(self.open_directory)
        self.panel.directory_action.setEnabled(bool(self.config.recording_root))

    def _remember_directory(self, status, seconds):
        if status.directory:
            self.recent_directory = status.directory

    def show_settings(self):
        from ui.video_settings_dialog import VideoSettingsDialog

        if self.dialog is not None:
            self.dialog.show()
            self.dialog.raise_()
            return
        active = self.service.status.recording in {"starting", "recording", "recovering", "stopping"}
        self.dialog = VideoSettingsDialog(self.config, self.parent(), read_only=active or not self.can_configure())
        self.dialog.probe_requested.connect(self.probe_devices)
        self.dialog.configuration_requested.connect(self.apply_config)
        self.dialog.finished.connect(self._dialog_closed)
        self.dialog.show()
        self.probe_devices()

    def _dialog_closed(self, result):
        dialog, self.dialog = self.dialog, None
        dialog.deleteLater()

    def probe_devices(self):
        if self._probe_running or self._shutdown_requested:
            return
        self._probe_running = True
        if self.dialog is not None:
            self.dialog.refresh_button.setEnabled(False)
        threading.Thread(target=self._probe, name="VideoDeviceProbe", daemon=True).start()

    def _probe(self):
        process = parent = child = None
        result = ((), "摄像头枚举超时")
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=probe_worker, args=(child,), name="VideoDeviceProbe")
            process.start()
            child.close()
            if parent.poll(8):
                result = parent.recv()
        except (OSError, EOFError, RuntimeError) as exc:
            result = ((), str(exc))
        finally:
            if process is not None and process.pid is not None:
                process.join(.5)
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if not process.is_alive():
                    process.close()
            if parent is not None:
                parent.close()
            if child is not None:
                child.close()
            self.devices_ready.emit(*result)

    def _devices_ready(self, devices, error):
        self._probe_running = False
        if self.dialog is not None:
            self.dialog.set_devices(devices, error)
        if self._shutdown_requested and self.is_shutdown_complete:
            self.closed.emit()

    def apply_config(self, config):
        active = self.service.status.recording in {"starting", "recording", "recovering", "stopping"}
        if active or not self.can_configure() or self._shutdown_requested or self._saving:
            QMessageBox.warning(self.parent(), "不能应用设置", "请先停止录像，并使用工程师或管理员权限。")
            return
        self._saving = True
        self._pending_config = config
        if self.dialog is not None:
            self.dialog.setEnabled(False)
        if self.service.is_closed:
            self._service_closed()
        else:
            self.service.shutdown()

    def _service_closed(self):
        if self._shutdown_requested and self.service.forced_termination:
            QMessageBox.warning(
                self.parent(), "录像未正常收尾",
                "视频进程已被强制终止，尾段可能不完整。已有文件保留，请检查录像目录。",
            )
        if self._pending_config is not None and not self._shutdown_requested:
            config, self._pending_config = self._pending_config, None
            threading.Thread(target=self._save, args=(config,), name="VideoSettingsSave", daemon=True).start()

    def _save(self, config):
        error = ""
        try:
            save_config(self.config_path, config)
        except (OSError, ValueError) as exc:
            error = str(exc)
        self.configuration_saved.emit(config, error)

    def _configuration_saved(self, config, error):
        self._saving = False
        if not error:
            if config.recording_root != self.config.recording_root:
                self.recent_directory = ""
            self.config = config
        if self._shutdown_requested:
            self.closed.emit()
            return
        self._generation += 1
        self.service = self._make_service()
        self._last_status = None
        self._last_seconds = -1
        self._frame_sequence = 0
        self._closed_emitted = False
        self._timer.start()
        if self.config.enabled:
            self.service.start()
        if self.panel is not None:
            self.panel.directory_action.setEnabled(bool(self.config.recording_root))
        if self.dialog is not None:
            self.dialog.setEnabled(True)
            if not error:
                self.dialog.accept()
        if error:
            QMessageBox.warning(self.parent(), "视频设置保存失败", error)

    def start_recording(self):
        if self._saving or self._shutdown_requested:
            return False
        return super().start_recording()

    def open_directory(self):
        directory = self.recent_directory
        if (not directory or not Path(directory).is_dir()) and self.config.recording_root:
            directory = Path(self.config.recording_root) / "video"
        if not directory or not Path(directory).is_dir():
            QMessageBox.information(self.parent(), "录像目录", "录像目录尚未创建或已不可用。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(directory).resolve())))

    def shutdown(self):
        self._shutdown_requested = True
        if self._pending_config is not None:
            self._pending_config = None
            self._saving = False
        super().shutdown()
