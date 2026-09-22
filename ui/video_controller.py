"""Qt video bridge and application controller, including settings and device probing."""

import logging
import multiprocessing
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time

from PyQt5.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImage
from PyQt5.QtWidgets import QMessageBox

from base.log_manager import LogManager
from base.log_exit import ProcessLogDrain, run_with_log_drain
from base.video.config import VideoConfig, load_config, migrate_config, save_config
from base.video.runtime import usb_video_worker
from base.video.service import VideoService
from base.video.usb_capture import probe_worker


RECORD_CONNECT_TIMEOUT = 20.0
PREVIEW_ACK_TIMEOUT = 2.0
ACTIVE_RECORDING_STATES = {"starting", "recording", "recovering", "stopping"}


@dataclass
class PendingRecording:
    deadline: float
    generation: int | None = None


class VideoServiceBridge(QObject):
    status_changed = pyqtSignal(object, int)
    preview_changed = pyqtSignal(QImage)
    closed = pyqtSignal()
    controls_changed = pyqtSignal()

    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self._last_status = None
        self._last_seconds = -1
        self._frame_sequence = 0
        self._closed_emitted = False
        self.preview_requested = True
        self._preview_revision = 0
        self._last_controls = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.poll)
        self._timer.start()

    def poll(self):
        self._advance()
        status = self.service.status
        seconds = status.elapsed(time.monotonic())
        if status != self._last_status or seconds != self._last_seconds:
            self._last_status, self._last_seconds = status, seconds
            self.status_changed.emit(status, seconds)
        controls = (self.preview_requested, self.record_start_pending, self.can_start_recording, self.control_message)
        if controls != self._last_controls:
            self._last_controls = controls
            self.controls_changed.emit()
        frame = self.service.latest_preview(self._frame_sequence) if self.preview_requested else None
        if frame is not None and frame.revision == self._preview_revision:
            self._frame_sequence = frame.sequence
            # Own the bytes before returning; shared storage can change immediately.
            image = QImage(frame.rgb, frame.width, frame.height, frame.width * 3, QImage.Format_RGB888).copy()
            self.preview_changed.emit(image)
        if self.service.is_closed and not self._closed_emitted:
            self._closed_emitted = True
            self._service_closed()

    def _advance(self):
        pass

    def _service_closed(self):
        self._timer.stop()
        self.closed.emit()

    @property
    def record_start_pending(self):
        return False

    @property
    def record_start_timed_out(self):
        return False

    @property
    def can_start_recording(self):
        return self.service.status.connection == "ready"

    @property
    def control_message(self):
        return ""

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
        self._unreaped_probe = None
        self._logger = LogManager.set_log_handler("core")
        self._saving = False
        self._save_preview_only = False
        self._activated = False
        self._pending_record_start = None
        self._record_start_timed_out = False
        self._restart_requested = False
        self._final_closed_emitted = False
        self._control_error = ""
        self._preview_error = ""
        self._preview_wait_since = None
        self._service_config = self.config
        self.dialog = None
        self._applying_dialog = None
        self.panel = None
        self.recent_directory = ""
        super().__init__(self._make_service(), parent)
        self.preview_requested = self.config.enabled
        if error:
            self.service._fail(error)
        self.devices_ready.connect(self._devices_ready)
        self.configuration_saved.connect(self._configuration_saved)
        self.status_changed.connect(self._remember_directory)
        if start_automatically:
            QTimer.singleShot(0, self.start_preview)

    def start_preview(self):
        if not self._activated and not self._shutdown_requested:
            self._activated = True
            self._restart_requested = self.preview_requested
            self._reconcile_capture()

    def _make_service(self):
        return VideoService(
            worker_target=usb_video_worker, worker_options=self.config, generation=self._generation,
            initial_connection="connecting" if self.config.enabled else "disabled",
            initial_preview_enabled=self.config.enabled,
            heartbeat_timeout=8, command_timeout=20, shutdown_timeout=120, drain_timeout=30,
        )

    @property
    def is_shutdown_complete(self):
        return (not self.service.is_started or self.service.is_closed) and not (
            self._saving or self._probe_running or self._pending_record_start or self._restart_requested
        )

    @property
    def record_start_pending(self):
        return self._pending_record_start is not None

    @property
    def record_start_timed_out(self):
        return self._record_start_timed_out

    @property
    def can_start_recording(self):
        return (not self._shutdown_requested and not self._saving and not self._recording_busy()
                and bool(self.config.device_id.strip() and self.config.recording_root.strip()))

    @property
    def control_message(self):
        if self.record_start_pending:
            return "正在连接摄像头…"
        return self._control_error

    def _recording_busy(self):
        return self.record_start_pending or self.service.status.recording in ACTIVE_RECORDING_STATES

    def _replace_service(self):
        awaiting_preview = self._preview_wait_since is not None and self.preview_requested
        self._generation += 1
        self.service = self._make_service()
        self._service_config = self.config
        self._last_status = self._last_controls = None
        self._last_seconds = -1
        self._frame_sequence = self._preview_revision = 0
        self._preview_wait_since = time.monotonic() if awaiting_preview else None
        self._preview_error = ""
        self._closed_emitted = False
        self._timer.start()

    def _reconcile_capture(self):
        if self._shutdown_requested or (self._saving and not self._save_preview_only):
            return
        needed = (self._activated and self.preview_requested) or self._recording_busy()
        if not needed:
            self._restart_requested = False
            if self.service.is_started and not self.service.is_closing:
                self.service.shutdown()
            return
        if self.service.is_closing and not self.service.is_closed:
            return
        if self.service.is_closed:
            if not self._restart_requested:
                return  # A crashed worker must not become an automatic restart loop.
            self._replace_service()
        elif self.service.status.connection == "unavailable" and self.service.is_started:
            if self._restart_requested:
                self.service.shutdown()
            return
        if not self.service.is_started:
            if self._service_config != self.config:
                self._replace_service()
            self.service.start()
        if self._pending_record_start is not None:
            self._pending_record_start.generation = self._generation
        self._restart_requested = False

    def _advance(self):
        now = time.monotonic()
        pending = self._pending_record_start
        if pending is not None:
            if now >= pending.deadline:
                self._pending_record_start = None
                self._record_start_timed_out = True
                self._control_error = "连接超时，请连接摄像头后重试。"
            elif (pending.generation == self._generation and not self.service.is_closing
                  and (self.service.status.connection == "unavailable" or self.service.is_closed)):
                self._pending_record_start = None
                self._control_error = self.service.status.connection_detail or "视频服务已退出，未开始录像。"
            elif (pending.generation == self._generation and self.service.status.connection == "ready"
                  and not self.service.is_closing):
                self._pending_record_start = None
                if not self.service.start_recording():
                    self._control_error = "录像启动请求未被接受，请重试。"
        if self._preview_wait_since is not None:
            status = self.service.status
            if (status.preview_revision == self._preview_revision
                    and status.preview_enabled == self.preview_requested
                    and not self.service.is_closing
                    and (self._preview_revision > 0 or status.connection == "ready")):
                self._preview_wait_since = None
                self._preview_error = ""
                if not self._saving:
                    self._finish_settings()
            elif now - self._preview_wait_since > PREVIEW_ACK_TIMEOUT:
                self._preview_error = "预览未生效，请重试。"
        self._reconcile_capture()
        self._update_dialog_access()

    def _update_dialog_access(self):
        if self.dialog is not None:
            self.dialog.set_access(read_only=not self.can_configure(),
                                   preview_only=self._recording_busy(), saving=self._saving)
            if self._preview_error:
                self.dialog.show_apply_status(self._preview_error)

    def _notify_shutdown_complete(self):
        if self._shutdown_requested and self.is_shutdown_complete and not self._final_closed_emitted:
            self._final_closed_emitted = True
            self._timer.stop()
            self.closed.emit()

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
        self.dialog = VideoSettingsDialog(self.config, self.parent(), read_only=not self.can_configure(),
                                          preview_only=self._recording_busy())
        self.dialog.probe_requested.connect(self.probe_devices)
        self.dialog.configuration_requested.connect(self.apply_config)
        self.dialog.finished.connect(self._dialog_closed)
        self._update_dialog_access()
        self.dialog.show()
        if not self._recording_busy():
            self.probe_devices()

    def _dialog_closed(self, result):
        dialog, self.dialog = self.dialog, None
        if self._applying_dialog is dialog:
            self._applying_dialog = None
        dialog.deleteLater()

    def probe_devices(self):
        if self._probe_running or self._shutdown_requested or self._recording_busy():
            return
        self._probe_running = True
        if self.dialog is not None:
            self.dialog.refresh_button.setEnabled(False)
        threading.Thread(target=self._probe, name="VideoDeviceProbe", daemon=True).start()

    def _reap_probe(self):
        if self._unreaped_probe is None:
            return True
        process, drain = self._unreaped_probe
        if process.is_alive():
            return False
        process.join(0)
        process.close()
        drain.close()
        self._unreaped_probe = None
        return True

    def _probe(self):
        process = parent = child = log_drain = None
        result = ((), "摄像头枚举超时")
        try:
            if not self._reap_probe():
                result = ((), "上次摄像头探测进程尚未退出")
                return
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            log_drain = ProcessLogDrain.create(context)
            process = context.Process(
                target=run_with_log_drain,
                args=(probe_worker, (child,), log_drain.child_endpoint), name="VideoDeviceProbe")
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
                    log_drain.begin("camera probe retirement")
                    self._report_probe_log_drain(process.pid, log_drain, log_drain.wait(is_alive=process.is_alive))
                    if process.is_alive():
                        process.terminate()
                    process.join(1)
                else:
                    self._report_probe_log_drain(process.pid, log_drain, log_drain.poll(already_dead=True))
                if not process.is_alive():
                    process.close()
                    log_drain.close()
                else:
                    self._unreaped_probe = (process, log_drain)
                    result = ((), "摄像头探测进程尚未退出")
            elif log_drain is not None:
                log_drain.close()
            if parent is not None:
                parent.close()
            if child is not None:
                child.close()
            self.devices_ready.emit(*result)

    def _report_probe_log_drain(self, pid, drain, result):
        if (result.status in ("timeout", "drained-with-errors")
                or (result.status == "already-dead" and result.detail is not None)):
            self._logger.error(
                "Camera probe log drain pid=%s reason=%s status=%s pending=%s stats=%s detail=%s",
                pid, drain.reason or "self-exit", result.status,
                "unknown" if result.stats is None else result.stats["pending"],
                result.stats, result.detail)

    def _devices_ready(self, devices, error):
        self._probe_running = False
        if self.dialog is not None:
            self.dialog.set_devices(devices, error)
        self._notify_shutdown_complete()

    def apply_config(self, config):
        # Replacing the old preview value into an incomplete disabled candidate
        # would run validation before it can be classified as a hardware change.
        preview_only = replace(self.config, enabled=False) == replace(config, enabled=False)
        if ((self._recording_busy() and not preview_only) or not self.can_configure()
                or self._shutdown_requested or self._saving):
            QMessageBox.warning(self.parent(), "不能应用设置", "请先停止录像，并使用工程师或管理员权限。")
            return
        self._applying_dialog = self.dialog
        if config == self.config:
            if (self._preview_error or self._preview_wait_since is not None
                    or (config.enabled and (not self.service.is_started or self.service.is_closed
                                            or self.service.status.connection == "unavailable"))):
                self._apply_preview()
            self._finish_settings()
            return
        self._saving = True
        self._save_preview_only = preview_only
        self._pending_config = config
        self._update_dialog_access()
        self._timer.start()
        if preview_only or self.service.is_closed:
            self._begin_save()
        else:
            self.service.shutdown()

    def _service_closed(self):
        if self._shutdown_requested and self.service.forced_termination:
            QMessageBox.warning(
                self.parent(), "录像未正常收尾",
                "视频进程已被强制终止，尾段可能不完整。已有文件保留，请检查录像目录。",
            )
        if self._pending_config is not None and not self._shutdown_requested:
            self._begin_save()
        if not self.preview_requested:
            self._preview_wait_since = None
            self._preview_error = ""
            if not self._saving and not self._shutdown_requested:
                self._finish_settings()
        self._notify_shutdown_complete()

    def _begin_save(self):
        config, self._pending_config = self._pending_config, None
        threading.Thread(target=self._save, args=(config,), name="VideoSettingsSave", daemon=True).start()

    def _save(self, config):
        error = ""
        try:
            save_config(self.config_path, config)
        except PermissionError:
            logging.getLogger("core.video").warning("Video settings save denied", exc_info=True)
            error = "无法更新配置文件，请检查文件占用或写入权限。"
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
            self._notify_shutdown_complete()
            return
        if not self._save_preview_only:
            self._replace_service()
        if not error or not self._save_preview_only:
            self._apply_preview()
        if self.panel is not None:
            self.panel.directory_action.setEnabled(bool(self.config.recording_root))
        self._update_dialog_access()
        if not error:
            self._finish_settings()
        if error:
            self._applying_dialog = None
            QMessageBox.warning(self.parent(), "视频设置保存失败", error)

    def _apply_preview(self):
        self.preview_requested = self.config.enabled
        self._activated = True
        self._preview_error = ""
        self._preview_wait_since = None
        self._timer.start()
        if self.service.is_started and not self.service.is_closing and not self.service.is_closed:
            if self.service.set_preview(self.preview_requested):
                self._preview_revision = self.service.preview_revision
                self._preview_wait_since = time.monotonic()
            else:
                self._preview_error = "预览切换失败，请重试。"
        self._restart_requested = self.preview_requested or self.record_start_pending
        self._reconcile_capture()
        if self.preview_requested and self._preview_wait_since is None and not self._preview_error:
            # A newly acquired camera has no SET_PREVIEW acknowledgement; READY
            # confirms its initial preference. Do not report an unopened device as applied.
            self._preview_wait_since = time.monotonic()

    def _finish_settings(self):
        if not self._preview_error and self._preview_wait_since is None:
            dialog, self._applying_dialog = self._applying_dialog, None
            if dialog is not None and dialog is self.dialog:
                dialog.accept()

    def start_recording(self):
        if self._saving or self._shutdown_requested or self._recording_busy():
            return False
        self._record_start_timed_out = False
        try:
            self.config.validate_capture()
        except ValueError as exc:
            self._control_error = str(exc)
            return False
        self._control_error = ""
        self._pending_record_start = PendingRecording(time.monotonic() + RECORD_CONNECT_TIMEOUT)
        self._restart_requested = True
        self._activated = True
        self._timer.start()
        self._reconcile_capture()
        self._advance()
        return True

    def stop_recording(self):
        if self._pending_record_start is not None:
            self._pending_record_start = None
            self._restart_requested = self.preview_requested and self.service.is_closing
            self._reconcile_capture()
            return True
        return super().stop_recording()

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
        self._pending_record_start = None
        self._restart_requested = False
        if self._pending_config is not None:
            self._pending_config = None
            self._saving = False
        super().shutdown()
        self._notify_shutdown_complete()
