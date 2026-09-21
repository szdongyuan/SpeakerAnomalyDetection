import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PyQt5.QtWidgets import QComboBox, QDialogButtonBox, QLabel, QMessageBox, QSpinBox

from base.video.config import VideoConfig, load_config, save_config
import ui.video_controller as video_controller_module
from base.video.usb_capture import CameraDevice
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel
from ui.video_controller import VideoController
from ui.video_settings_dialog import VideoSettingsDialog
from ui.video_demo import configure_demo_font


def simulated_probe_worker(channel):
    """Keep controller integration tests off the live DirectShow device list."""
    channel.send(((CameraDevice("test-device", "合成摄像头"),), ""))
    channel.close()


@pytest.fixture(autouse=True)
def isolate_video_project_logging(tmp_path, monkeypatch):
    from unit_test.logging_test_support import isolated_project_logger
    with isolated_project_logger(tmp_path, monkeypatch):
        yield


@pytest.fixture(autouse=True)
def isolate_camera_probe(monkeypatch):
    monkeypatch.setattr(video_controller_module, "probe_worker", simulated_probe_worker)


def spin(app, predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("video UI operation timed out")


@pytest.mark.parametrize("frozen", [False, True])
def test_default_video_config_path_is_under_ui_config_not_cwd_or_bundle(tmp_path, monkeypatch, frozen):
    project = tmp_path / "application"
    monkeypatch.setattr(video_controller_module, "__file__", str(project / "ui" / "video_controller.py"))
    monkeypatch.setattr(video_controller_module, "sys", SimpleNamespace(
        frozen=frozen, executable=str(project / "app.exe"), _MEIPASS=str(tmp_path / "bundle"),
    ))
    monkeypatch.chdir(tmp_path)
    assert video_controller_module.default_config_path() == project / "ui" / "ui_config" / "video_settings.json"


def test_default_controller_migrates_then_saves_and_reloads_only_new_location(ui_qapp, tmp_path, monkeypatch):
    path = tmp_path / "ui" / "ui_config" / "video_settings.json"
    legacy = tmp_path / "configs" / "video_settings.json"
    config = VideoConfig(fps_num=30, device_id="usb1", device_name="USB Cam")
    save_config(legacy, config)
    original = legacy.read_bytes()
    monkeypatch.setattr(video_controller_module, "default_config_path", lambda: path)
    controller = VideoController(start_automatically=False)
    try:
        assert controller.config_path == path
        assert controller.config == config
        assert controller.service.process_id is None
        updated = replace(config, target_bitrate_bps=6_000_000)
        controller.apply_config(updated)
        spin(ui_qapp, lambda: not controller._saving)
        assert load_config(path) == updated
        assert legacy.read_bytes() == original
    finally:
        controller.shutdown()
        controller._timer.stop()
    reopened = VideoController(start_automatically=False)
    try:
        assert reopened.config == updated
        assert legacy.read_bytes() == original
    finally:
        reopened.shutdown()
        reopened._timer.stop()


def test_explicit_config_path_does_not_trigger_default_migration(ui_qapp, tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("custom settings must not migrate default files")

    monkeypatch.setattr(video_controller_module, "migrate_config", unexpected)
    controller = VideoController(config_path=tmp_path / "custom.json", start_automatically=False)
    try:
        assert controller.config == VideoConfig()
        assert not controller.config_path.exists()
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_migration_error_is_visible_and_does_not_start_camera(ui_qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(video_controller_module, "default_config_path", lambda: tmp_path / "ui/ui_config/video_settings.json")

    def denied(*args):
        raise OSError("视频配置迁移失败，旧文件已保留")

    monkeypatch.setattr(video_controller_module, "migrate_config", denied)
    controller = VideoController(start_automatically=False)
    try:
        assert "迁移失败" in controller.service.status.detail
        assert controller.service.status.connection == "unavailable"
        assert not controller.config.enabled
        assert controller.service.process_id is None
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_disabled_defaults_no_process_or_settings_file_and_binds_main_card(ui_qapp, tmp_path):
    controller = VideoController(config_path=tmp_path / "video.json")
    panel = MotorVideoMonitorPanel()
    try:
        controller.attach_panel(panel)
        controller.poll()
        assert controller.service.process_id is None
        assert not (tmp_path / "video.json").exists()
        assert not panel.live_panel.record_button.isEnabled()
        assert panel.live_panel.more_button.isEnabled()
        assert panel.card.isHidden()
    finally:
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        controller._timer.stop()
        panel.close()


def test_save_settings_and_reload_without_automatic_recording(ui_qapp, tmp_path):
    path = tmp_path / "video.json"
    controller = VideoController(config_path=path)
    config = VideoConfig(recording_root=str(tmp_path / "独立录像"), width=640, height=480)
    try:
        controller.apply_config(config)
        spin(ui_qapp, lambda: not controller._saving)
        assert load_config(path) == config
        assert controller.config == config
        assert controller.service.process_id is None
        assert not controller.service.status.record_intent
    finally:
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        controller._timer.stop()


@pytest.mark.parametrize("recent", ["", "existing", "missing"])
def test_open_directory_uses_recent_session_or_video_child(ui_qapp, tmp_path, monkeypatch, recent):
    video = tmp_path / "video"
    video.mkdir()
    session = video / "2026-09-09_10-23-30"
    session.mkdir()
    opened = []
    monkeypatch.setattr(video_controller_module.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    try:
        controller.config = VideoConfig(recording_root=str(tmp_path))
        controller.recent_directory = str(session) if recent == "existing" else str(video / "missing") if recent else ""
        controller.open_directory()
        target = session if recent == "existing" else video
        assert opened == [target.resolve().as_posix()]
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_changed_root_clears_recent_session_and_missing_video_only_notifies(ui_qapp, tmp_path, monkeypatch):
    opened, messages = [], []
    monkeypatch.setattr(video_controller_module.QDesktopServices, "openUrl", lambda url: opened.append(url))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    try:
        controller.config = VideoConfig(recording_root=str(tmp_path / "old"))
        controller.recent_directory = str(tmp_path / "old/video/recent")
        controller._configuration_saved(VideoConfig(recording_root=str(tmp_path / "new")), "")
        assert not controller.recent_directory
        controller.open_directory()
        assert len(messages) == 1 and not opened
        assert not (tmp_path / "new").exists()
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_operator_cannot_apply_configuration(ui_qapp, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    controller = VideoController(config_path=tmp_path / "video.json", can_configure=lambda: False)
    try:
        controller.apply_config(VideoConfig())
        assert messages
        assert not (tmp_path / "video.json").exists()
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_settings_keep_same_named_device_identities_and_readonly_guard(ui_qapp):
    config = VideoConfig(device_id="id2", device_name="USB Camera")
    dialog = VideoSettingsDialog(config, read_only=True)
    try:
        dialog.set_devices((CameraDevice("id1", "USB Camera"), CameraDevice("id2", "USB Camera")))
        assert dialog.devices.currentData() == "id2"
        assert not dialog.devices.isEnabled()
        assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    finally:
        dialog.close()


def test_simplified_settings_preserve_hidden_encoding_and_storage_policy(ui_qapp, tmp_path):
    config = VideoConfig(
        device_id="usb1", device_name="USB Camera", input_format="nv12", codec="h265",
        min_free_bytes=7 * 1024**3 + 123, recording_root=str(tmp_path),
    )
    dialog = VideoSettingsDialog(config)
    requested = []
    dialog.configuration_requested.connect(requested.append)
    try:
        assert len(dialog.findChildren(QComboBox)) == 3
        assert len(dialog.findChildren(QSpinBox)) == 1
        labels = {label.text() for label in dialog.findChildren(QLabel)}
        assert not labels & {"输入格式", "录像编码", "最低保留空间", "文件分段"}
        dialog.enabled.setChecked(True)
        dialog.bitrate.setValue(6000)
        dialog.submit()
        assert len(requested) == 1
        saved = requested[0]
        assert saved.enabled
        assert saved.target_bitrate_bps == 6_000_000
        assert saved.input_format == config.input_format
        assert saved.codec == config.codec
        assert saved.min_free_bytes == config.min_free_bytes
        assert saved.segment_duration_seconds == config.segment_duration_seconds
    finally:
        dialog.close()


@pytest.mark.parametrize("device_id,folder,expected", [
    ("usb1", "", "请选择录像保存文件夹。"),
    ("", "", "请选择摄像头"),
    ("usb1", "relative-folder", "录像保存位置必须为完整路径"),
])
def test_video_validation_dialog_uses_actionable_chinese(ui_qapp, monkeypatch, device_id, folder, expected):
    dialog = VideoSettingsDialog(VideoConfig(device_id=device_id, device_name="USB Cam"))
    messages, requested = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: messages.append((title, text)))
    dialog.configuration_requested.connect(requested.append)
    try:
        dialog.enabled.setChecked(True)
        dialog.folder.setText(folder)
        dialog.submit()
        assert len(messages) == 1
        assert messages[0][0] == "配置无效"
        assert expected in messages[0][1]
        assert not requested
    finally:
        dialog.close()


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("width", [560, 680])
def test_simplified_settings_layout_and_persistent_readonly_notice(ui_qapp, tmp_path, read_only, width):
    configure_demo_font(ui_qapp)
    dialog = VideoSettingsDialog(VideoConfig(), read_only=read_only)
    try:
        dialog.set_devices(())
        dialog.resize(width, dialog.sizeHint().height())
        dialog.show()
        ui_qapp.processEvents()
        assert dialog.read_only_label.isVisible() == read_only
        assert dialog.buttons.button(QDialogButtonBox.Save).isEnabled() != read_only
        assert dialog.devices.font().pixelSize() == 13
        assert dialog.devices.height() >= 28
        assert dialog.folder.width() >= 250
        inputs = (dialog.devices, dialog.resolution, dialog.rate, dialog.bitrate, dialog.folder)
        assert len({field.width() for field in inputs}) == 1
        assert len({field.geometry().left() for field in inputs}) == 1
        assert len({field.geometry().right() for field in inputs}) == 1
        assert dialog.refresh_button.geometry().left() == dialog.browse_button.geometry().left()
        assert dialog.message.geometry().bottom() < dialog.buttons.geometry().top()
        assert dialog.buttons.geometry().bottom() < dialog.height()
        screenshot = tmp_path / f"video-settings-simple-{'readonly' if read_only else 'editable'}.png"
        assert dialog.grab().save(str(screenshot))
        print(f"\nSimplified settings screenshot: {screenshot}")
        dialog.set_devices((CameraDevice("usb1", "USB Camera"),))
        assert dialog.message.isVisible()
        assert not dialog.message.text()
        assert dialog.read_only_label.isVisible() == read_only
        assert "不会自动开始" not in {label.text() for label in dialog.findChildren(QLabel)}
        assert "2小时" in dialog.folder.toolTip()
        assert "video" in dialog.folder.placeholderText()
        assert "上一级目录" in dialog.folder.toolTip()
    finally:
        dialog.close()


@pytest.mark.parametrize("read_only", [False, True])
def test_settings_geometry_stays_fixed_across_device_refresh(ui_qapp, tmp_path, read_only):
    configure_demo_font(ui_qapp)
    dialog = VideoSettingsDialog(VideoConfig(device_id="usb1", device_name="USB Cam"), read_only=read_only)
    dialog.show()
    ui_qapp.processEvents()
    widgets = (
        dialog.enabled, dialog.devices, dialog.resolution, dialog.rate,
        dialog.bitrate, dialog.folder, dialog.refresh_button, dialog.browse_button,
        dialog.message, dialog.buttons,
    )
    original = [widget.geometry() for widget in widgets]
    original_window = dialog.geometry()
    long_error = "摄像头检测失败，请检查设备连接。" * 30
    try:
        for devices, error in (
            ((CameraDevice("usb1", "USB Cam"),), ""),
            ((), ""),
            ((), long_error),
            ((CameraDevice("usb1", "很长的USB摄像头名称" * 20),), ""),
            ((CameraDevice("usb1", "USB Cam"),), ""),
        ):
            dialog.set_devices(devices, error)
            for _ in range(3):
                ui_qapp.processEvents()
            assert dialog.geometry() == original_window
            assert [widget.geometry() for widget in widgets] == original
            if error:
                assert dialog.message.toolTip() == error
        screenshot = tmp_path / "video-settings-stable.png"
        assert dialog.grab().save(str(screenshot))
        print(f"\nStable settings screenshot: {screenshot}")
    finally:
        dialog.close()


def test_device_probe_is_async_and_shutdown_waits_for_it(ui_qapp, tmp_path):
    controller = VideoController(config_path=tmp_path / "video.json")
    try:
        start = time.monotonic()
        controller.probe_devices()
        assert time.monotonic() - start < .2
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete, timeout=12)
        assert not controller._probe_running
    finally:
        controller.shutdown()
        controller._timer.stop()


def test_main_close_waits_for_video_without_starting_audio_shutdown(ui_qapp):
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtGui import QCloseEvent
    from main_window import MainWindow

    class ClosingVideo(QObject):
        closed = pyqtSignal()
        is_shutdown_complete = False
        shutdown = Mock()

    video = ClosingVideo()
    host = SimpleNamespace(video_controller=video, sequence_window=None, setEnabled=Mock(), close=Mock())
    event = QCloseEvent()
    MainWindow.closeEvent(host, event)
    assert not event.isAccepted()
    video.shutdown.assert_called_once()
    host.setEnabled.assert_called_once_with(False)
    video.closed.emit()
    host.close.assert_called_once()


def test_live_card_real_files_and_settings_screenshot(ui_qapp, tmp_path, monkeypatch):
    from base.video.config import save_config
    from base.video.service import VideoService
    from ui.video_demo import configure_demo_font
    from unit_test.video.test_recording import config_for, decoded
    from unit_test.video.test_runtime import synthetic_video_worker

    configure_demo_font(ui_qapp)
    path = tmp_path / "settings.json"
    config = config_for(tmp_path / "videos")
    save_config(path, config)
    monkeypatch.setattr(VideoController, "_make_service", lambda self: VideoService(
        worker_target=synthetic_video_worker, worker_options=config, heartbeat_timeout=8, command_timeout=10,
    ))
    controller = VideoController(config_path=path)
    placeholder = MotorVideoMonitorPanel()
    controller.attach_panel(placeholder)
    placeholder.resize(560, 350)
    placeholder.show()
    try:
        spin(ui_qapp, lambda: controller.panel._status.connection == "ready")
        controller.panel.record_button.click()
        spin(ui_qapp, lambda: controller.panel._status.recording == "recording")
        ui_qapp.processEvents()
        screenshot = tmp_path / "live-video-test-frames.png"
        assert placeholder.grab().save(str(screenshot))
        print(f"\nLive card (test input, real recording): {screenshot}")
        controller.show_settings()
        assert controller.dialog.read_only
        ui_qapp.processEvents()
        settings_screenshot = tmp_path / "video-settings.png"
        assert controller.dialog.grab().save(str(settings_screenshot))
        print(f"Video settings: {settings_screenshot}")
        controller.dialog.close()
        controller.panel.record_button.click()
        spin(ui_qapp, lambda: controller.panel._status.recording == "completed")
        assert decoded(next((tmp_path / "videos").glob("*/*/*.mp4")))
        assert controller.recent_directory
    finally:
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        controller._timer.stop()
        placeholder.close()


@pytest.mark.parametrize("status", ["no-runtime", "timeout", "drained-with-errors", "already-dead", "exit-after-ack"])
@pytest.mark.parametrize("start_failure", [False, True])
def test_probe_log_drain_background_and_start_failure(ui_qapp, tmp_path, monkeypatch, start_failure, status):
    import threading
    from base.log_exit import DrainResult, run_with_log_drain
    entered, release = threading.Event(), threading.Event()
    calls, processes = [], []
    endpoint = object()
    class Drain:
        child_endpoint = endpoint
        reason = None
        def begin(self, reason):
            self.reason = reason
            calls.append("begin")
        def wait(self, *, is_alive=None):
            assert is_alive == processes[0].is_alive
            assert is_alive()
            assert threading.current_thread().name == "VideoDeviceProbe"
            entered.set()
            assert release.wait(3)
            calls.append("ack")
            if status in ("already-dead", "exit-after-ack"):
                processes[0].alive = False
                assert not is_alive()
            return DrainResult("no-runtime" if status == "exit-after-ack" else status,
                               detail="log drain unconfirmed" if status == "already-dead" else None)
        def poll(self, **kwargs):
            return DrainResult("already-dead")
        def close(self):
            calls.append("drain-close")
    class Process:
        pid = None
        alive = True
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            processes.append(self)
        def start(self):
            if start_failure:
                raise OSError("probe start denied")
            self.pid = 123
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append("terminate")
            self.alive = False
        def join(self, timeout):
            pass
        def close(self):
            calls.append("process-close")
    channel = SimpleNamespace(close=lambda: None, poll=lambda _: False)
    context = SimpleNamespace(Process=Process, Pipe=lambda **kwargs: (channel, channel))
    monkeypatch.setattr(video_controller_module, "ProcessLogDrain", SimpleNamespace(create=lambda _: Drain()), raising=False)
    monkeypatch.setattr(video_controller_module.multiprocessing, "get_context", lambda _: context)
    diagnostic_logger = Mock()
    monkeypatch.setattr(video_controller_module.LogManager, "set_log_handler", lambda _: diagnostic_logger)
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    results = []
    controller.devices_ready.connect(lambda devices, error: results.append((devices, error)))
    try:
        started = time.monotonic()
        controller.probe_devices()
        assert time.monotonic() - started < .2
        if not start_failure:
            assert entered.wait(1), "probe terminated without requesting log drain"
            assert calls == ["begin"]
        started = time.monotonic()
        controller.shutdown()
        assert time.monotonic() - started < .2
        release.set()
        spin(ui_qapp, lambda: controller.is_shutdown_complete and bool(results))
        assert len(results) == 1
        assert "drain-close" in calls
        if start_failure:
            assert "probe start denied" in results[0][1]
            assert calls == ["drain-close"]
        elif status in ("already-dead", "exit-after-ack"):
            assert calls == ["begin", "ack", "process-close", "drain-close"]
        else:
            assert calls == ["begin", "ack", "terminate", "process-close", "drain-close"]
        if not start_failure and status not in ("no-runtime", "exit-after-ack"):
            diagnostic_logger.error.assert_called_once()
            assert status in diagnostic_logger.error.call_args.args
            assert "unknown" in diagnostic_logger.error.call_args.args
            if status == "already-dead":
                assert "unconfirmed" in diagnostic_logger.error.call_args.args[-1]
        else:
            diagnostic_logger.error.assert_not_called()
        assert processes[0].kwargs["target"] is run_with_log_drain
        assert processes[0].kwargs["args"] == (simulated_probe_worker, (channel,), endpoint)
    finally:
        release.set()
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        controller._timer.stop()


def no_runtime_blocked_probe(channel):
    import threading
    from base.log_manager import LogManager
    assert LogManager._runtime is None
    channel.send(((), "test probe result"))
    threading.Event().wait()


def test_real_probe_no_runtime_drain_creates_no_logs(ui_qapp, tmp_path, monkeypatch):
    import multiprocessing
    from base.log_exit import ProcessLogDrain
    drains = []
    def create(context):
        drain = ProcessLogDrain.create(context)
        drains.append(drain)
        return drain
    monkeypatch.setattr(video_controller_module, "ProcessLogDrain", SimpleNamespace(create=create))
    monkeypatch.setattr(video_controller_module, "probe_worker", no_runtime_blocked_probe)
    monkeypatch.chdir(tmp_path)
    children_before = {child.pid for child in multiprocessing.active_children()}
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    results = []
    controller.devices_ready.connect(lambda devices, error: results.append((devices, error)))
    # Controller owns the parent logger; the child must add no logging paths.
    paths_before_probe = set(tmp_path.rglob("*"))
    try:
        controller.probe_devices()
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete and bool(results))
        assert results == [((), "test probe result")]
        assert drains[0].poll().status == "no-runtime"
        assert drains[0].child_endpoint is None
        assert set(tmp_path.rglob("*")) == paths_before_probe
        assert not any(path.is_file() for path in tmp_path.rglob("*"))
        assert {child.pid for child in multiprocessing.active_children()} <= children_before
    finally:
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        controller._timer.stop()


def test_probe_retains_unconfirmed_child_and_reaps_before_next_launch(ui_qapp, tmp_path, monkeypatch):
    from base.log_exit import DrainResult
    calls, processes, drains = [], [], []
    class Drain:
        child_endpoint = object()
        reason = None
        def begin(self, reason):
            self.reason = reason
            calls.append("begin")
        def wait(self, *, is_alive):
            return DrainResult("no-runtime")
        def poll(self, **kwargs):
            return DrainResult("no-runtime")
        def close(self):
            calls.append("drain-close")
    class Process:
        pid = 123
        alive = True
        def __init__(self, **kwargs):
            processes.append(self)
        def start(self):
            pass
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append("terminate")
        def join(self, timeout):
            pass
        def close(self):
            calls.append("process-close")
    def create(context):
        drain = Drain()
        drains.append(drain)
        return drain
    channel = SimpleNamespace(close=lambda: None, poll=lambda _: False)
    context = SimpleNamespace(Process=Process, Pipe=lambda **kwargs: (channel, channel))
    monkeypatch.setattr(video_controller_module, "ProcessLogDrain", SimpleNamespace(create=create))
    monkeypatch.setattr(video_controller_module.multiprocessing, "get_context", lambda _: context)
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    results = []
    controller.devices_ready.connect(lambda devices, error: results.append((devices, error)))
    try:
        controller.probe_devices()
        spin(ui_qapp, lambda: len(results) == 1 and not controller._probe_running)
        assert controller._unreaped_probe == (processes[0], drains[0])
        controller.probe_devices()
        spin(ui_qapp, lambda: len(results) == 2 and not controller._probe_running)
        assert len(processes) == 1 and calls == ["begin", "terminate"]
        assert results[1][1]
        processes[0].alive = False
        controller.probe_devices()
        spin(ui_qapp, lambda: len(results) == 3 and not controller._probe_running)
        assert len(processes) == 2
        assert calls == ["begin", "terminate", "process-close", "drain-close", "begin", "terminate"]
        processes[1].alive = False
        controller._reap_probe()
        assert controller._unreaped_probe is None
    finally:
        controller.shutdown()
        controller._timer.stop()


@pytest.mark.parametrize("parent_state", ["closing", "sealed", "stopped"])
def test_probe_teardown_reuses_logger_when_parent_logging_is_closing(ui_qapp, tmp_path, monkeypatch, parent_state):
    import threading
    from base import log_manager
    from base.log_exit import DrainResult
    manager = log_manager.LogManager
    logger = manager.set_log_handler("core")
    runtime = manager._runtime
    entered, release = threading.Event(), threading.Event()
    calls, acquisitions, results = [], [], []
    original_acquire = manager.set_log_handler
    def acquire(name):
        acquisitions.append(name)
        return original_acquire(name)
    monkeypatch.setattr(manager, "set_log_handler", acquire)
    # Ensure fixture cleanup restores sealing even if the RED assertion raises.
    monkeypatch.setattr(manager, "_forced_exit", False)
    if parent_state == "closing":
        original_write = log_manager._BatchFileHandler.do_write
        def gated_write(sink, message):
            assert sink.is_locked
            entered.set()
            assert release.wait(3)
            return original_write(sink, message)
        monkeypatch.setattr(log_manager._BatchFileHandler, "do_write", gated_write)
    class Drain:
        child_endpoint = object()
        reason = None
        def begin(self, reason):
            self.reason = reason
            if parent_state == "closing":
                logger.error("hold-parent-log-consumer")
                assert entered.wait(2)
                assert not manager.shutdown_all(0)
            elif parent_state == "sealed":
                manager.seal_for_forced_exit()
            else:
                assert manager.shutdown_all(1)
        def wait(self, *, is_alive):
            return DrainResult("timeout")
        def close(self):
            calls.append("drain-close")
    class Process:
        pid = 123
        alive = True
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append("terminate")
            self.alive = False
        def join(self, timeout):
            pass
        def close(self):
            calls.append("process-close")
    channel = SimpleNamespace(close=lambda: calls.append("channel-close"), poll=lambda _: False)
    context = SimpleNamespace(Process=Process, Pipe=lambda **kwargs: (channel, channel))
    monkeypatch.setattr(video_controller_module, "ProcessLogDrain", SimpleNamespace(create=lambda _: Drain()))
    monkeypatch.setattr(video_controller_module.multiprocessing, "get_context", lambda _: context)
    controller = VideoController(config_path=tmp_path / "settings.json", start_automatically=False)
    controller.devices_ready.connect(lambda devices, error: results.append((devices, error)))
    acquired_before_exit = list(acquisitions)
    try:
        controller._probe_running = True
        controller._probe()
        assert acquisitions == acquired_before_exit
        assert manager._runtime is runtime
        assert calls == ["channel-close", "terminate", "process-close", "drain-close", "channel-close", "channel-close"]
        assert len(results) == 1 and not controller._probe_running
    finally:
        release.set()
        assert manager.shutdown_all(2)
        controller.shutdown()
        controller._timer.stop()
