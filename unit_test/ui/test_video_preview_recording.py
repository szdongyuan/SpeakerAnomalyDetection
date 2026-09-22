"""Camera input is synthetic; control IPC, Qt and encoded files are real."""

from dataclasses import replace
import threading
import time
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialogButtonBox, QMessageBox

from base.video.config import load_config, save_config
from base.video.service import VideoService
from base.video.worker import SimulationOptions, simulated_video_worker
from ui.video_controller import VideoController
from ui.video_monitor_widget import VideoMonitorWidget
from ui.video_settings_dialog import VideoSettingsDialog
from ui.video_demo import configure_demo_font
from unit_test.ui.test_video_service_bridge import spin
from unit_test.video.test_recording import config_for, decoded
from unit_test.video.test_runtime import synthetic_video_worker


@pytest.fixture
def camera(ui_qapp, tmp_path, monkeypatch):
    controllers = []
    monkeypatch.setattr(QMessageBox, "warning", Mock())

    def make(*, preview=False, simulation=False, ready_delay=0, start_automatically=True, **faults):
        root = tmp_path / str(len(controllers))
        path = root / "settings.json"
        config = replace(config_for(root / "recordings"), enabled=preview)
        save_config(path, config)

        class Controller(VideoController):
            def _make_service(self):
                return VideoService(
                    worker_target=simulated_video_worker if simulation else synthetic_video_worker,
                    worker_options=SimulationOptions(preview_enabled=self.config.enabled,
                                                     ready_delay=ready_delay, **faults) if simulation else self.config,
                    generation=self._generation, initial_preview_enabled=self.config.enabled,
                    initial_connection="connecting" if self.config.enabled else "disabled",
                    heartbeat_timeout=8, command_timeout=8, shutdown_timeout=10,
                )

        controller = Controller(config_path=path, start_automatically=start_automatically)
        controller.probe_devices = Mock()
        configure_demo_font(ui_qapp)
        panel = VideoMonitorWidget(controller)
        controller.attach_widget(panel)
        panel.resize(560, 350)
        panel.show()
        controllers.append(controller)
        ui_qapp.processEvents()
        return controller

    yield make
    for controller in controllers:
        controller.shutdown()
        spin(ui_qapp, lambda: controller.is_shutdown_complete, timeout=15)
        controller._timer.stop()
        if controller.dialog is not None:
            controller.dialog.close()
        controller.panel.close()


def start(app, controller):
    assert controller.start_recording()
    spin(app, lambda: controller.service.status.recording == "recording", timeout=12)


def save_preview(app, controller, enabled):
    controller.show_settings()
    controller.dialog.enabled.setChecked(enabled)
    controller.dialog.submit()
    spin(app, lambda: not controller._saving)
    assert load_config(controller.config_path).enabled is enabled
    spin(app, lambda: controller.dialog is None, timeout=8)


def test_no_preview_records_real_video_then_releases_and_restarts(camera, ui_qapp):
    controller = camera()
    assert not controller.service.is_started
    assert controller.panel.record_button.isEnabled()
    assert not list(controller.config_path.parent.rglob("*.mp4"))
    closed = []
    controller.closed.connect(lambda: closed.append(True))
    start(ui_qapp, controller)
    generation = controller._generation
    spin(ui_qapp, lambda: controller.service.status.elapsed(time.monotonic()) >= 1)
    assert controller.panel.canvas.image.isNull()
    assert "预览已关闭" in controller.panel.canvas.message
    assert controller.service.latest_preview() is None
    controller.poll()
    assert controller.panel.grab().save(str(controller.config_path.parent / "background-recording.png"))
    assert controller.stop_recording()
    spin(ui_qapp, lambda: controller.service.is_closed)
    assert controller.service.status.recording == "completed"
    assert not closed  # Releasing an idle camera is not application shutdown.
    videos = list(controller.config_path.parent.rglob("*.mp4"))
    assert len(videos) == 1 and len(decoded(videos[0])) >= 5
    start(ui_qapp, controller)
    assert controller._generation > generation
    assert controller.stop_recording()
    spin(ui_qapp, lambda: controller.service.is_closed)


@pytest.mark.parametrize("initial", [False, True])
def test_save_preview_during_real_recording_preserves_session(camera, ui_qapp, initial):
    controller = camera(preview=initial)
    start(ui_qapp, controller)
    service = controller.service
    pid, session, began = service.process_id, service.status.session_id, service.status.started_at
    for enabled in (not initial, initial, not initial):
        save_preview(ui_qapp, controller, enabled)
        assert controller.service is service and service.process_id == pid
        assert service.status.session_id == session and service.status.started_at == began
        assert service.status.recording == "recording"
        if enabled:
            spin(ui_qapp, lambda: not controller.panel.canvas.image.isNull())
        else:
            assert controller.panel.canvas.image.isNull()
        assert not controller.probe_devices.called
    controller.stop_recording()
    spin(ui_qapp, lambda: service.status.recording == "completed")
    if controller.preview_requested:
        assert not service.is_closing
    else:
        spin(ui_qapp, lambda: service.is_closed)
    assert all(decoded(path) for path in controller.config_path.parent.rglob("*.mp4"))


@pytest.mark.parametrize("cancel", [True, False])
@pytest.mark.parametrize("preview", [False, True])
def test_pending_connection_cancel_or_timeout_never_late_starts(camera, ui_qapp, cancel, preview, monkeypatch):
    import ui.video_controller as module
    monkeypatch.setattr(module, "RECORD_CONNECT_TIMEOUT", 0.15 if not cancel else 5)
    controller = camera(preview=preview, simulation=True, ready_delay=1)
    assert controller.start_recording()
    controller.poll()
    assert controller.panel.record_button.text() == "取消录像"
    if cancel:
        assert controller.stop_recording()
    spin(ui_qapp, lambda: not controller.record_start_pending)
    if preview:
        spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    else:
        spin(ui_qapp, lambda: controller.service.is_closed)
    controller.poll()
    assert not controller.record_start_pending
    assert not controller.service.status.session_id
    assert "超时" in controller.control_message if not cancel else not controller.control_message
    assert controller.record_start_timed_out is (not cancel)
    assert controller.panel.record_button.text() == "开始录像"
    if not cancel:
        assert controller.panel.canvas.message == "连接超时，请连接摄像头后重试。"
        assert controller.panel.grab().save(str(controller.config_path.parent / "camera-connect-timeout.png"))
        assert controller.start_recording()
        controller.poll()
        assert not controller.record_start_timed_out
        assert "连接超时" not in controller.panel.canvas.message
        controller.stop_recording()


def test_preview_release_then_record_uses_new_generation(camera, ui_qapp):
    controller = camera(preview=True, simulation=True)
    spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    old = controller.service
    controller.apply_config(replace(controller.config, enabled=False))
    spin(ui_qapp, lambda: not controller._saving)
    assert old.is_closing
    assert controller.start_recording()
    spin(ui_qapp, lambda: controller.service.status.recording == "recording")
    assert old.is_closed and controller.service is not old


@pytest.mark.parametrize("error", [OSError("write denied"), PermissionError("locked settings path")])
def test_preview_save_failure_preserves_file_display_and_recording(camera, ui_qapp, monkeypatch, caplog, error):
    controller = camera(preview=True, simulation=True)
    start(ui_qapp, controller)
    before = controller.config_path.read_bytes()
    service, session = controller.service, controller.service.status.session_id
    monkeypatch.setattr("ui.video_controller.save_config", Mock(side_effect=error))
    controller.apply_config(replace(controller.config, enabled=False))
    spin(ui_qapp, lambda: not controller._saving)
    assert controller.config_path.read_bytes() == before
    assert controller.preview_requested
    assert controller.service is service and service.status.session_id == session
    assert service.status.recording == "recording" and not service.is_closing
    if isinstance(error, PermissionError):
        assert QMessageBox.warning.call_args.args[2] == "无法更新配置文件，请检查文件占用或写入权限。"
        assert "locked settings path" in caplog.text
    else:
        assert QMessageBox.warning.call_args.args[2] == str(error)


def test_saved_preview_command_failure_can_retry_same_value(camera, ui_qapp, monkeypatch):
    controller = camera(preview=True, simulation=True)
    start(ui_qapp, controller)
    service = controller.service
    original = service.set_preview
    monkeypatch.setattr(service, "set_preview", lambda value: False)
    controller.show_settings()
    controller.apply_config(replace(controller.config, enabled=False))
    spin(ui_qapp, lambda: not controller._saving)
    assert not load_config(controller.config_path).enabled
    assert controller.dialog.message.text() == "预览切换失败，请重试。"
    controller.poll()
    assert not controller.control_message
    assert controller.panel.canvas.message == "预览已关闭，正在录像"
    assert service.status.recording == "recording"
    monkeypatch.setattr(service, "set_preview", original)
    controller.apply_config(controller.config)
    spin(ui_qapp, lambda: service.status.preview_revision == service.preview_revision)
    controller.poll()
    assert not controller.control_message


def test_recording_locks_parameters_but_allows_preview_without_probe(camera, ui_qapp):
    controller = camera(simulation=True)
    start(ui_qapp, controller)
    controller.show_settings()
    dialog = controller.dialog
    assert dialog.preview_only and dialog.enabled.isEnabled()
    assert not dialog.message.text()
    assert dialog.read_only_label.text() == "当前仅可修改预览开关，设备参数已锁定。"
    assert dialog.enabled.toolTip() == "保存后生效，下次启动沿用；关闭预览不影响录像。"
    assert dialog.grab().save(str(controller.config_path.parent / "settings-single-note.png"))
    assert not dialog.devices.isEnabled() and not dialog.refresh_button.isEnabled()
    assert dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    assert not controller.probe_devices.called
    original = controller.config
    controller.apply_config(replace(original, width=640))
    assert controller.config == original and not controller._saving
    controller.can_configure = lambda: False
    controller.poll()
    assert not dialog.enabled.isEnabled()
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    controller.apply_config(replace(original, enabled=True))
    assert load_config(controller.config_path) == original


@pytest.mark.parametrize("exit_app", [False, True])
def test_stop_or_exit_during_preview_save_cannot_restart(camera, ui_qapp, monkeypatch, exit_app):
    controller = camera(simulation=True)
    start(ui_qapp, controller)
    entered, release = threading.Event(), threading.Event()
    original = save_config

    def slow_save(path, config):
        entered.set()
        assert release.wait(5)
        original(path, config)

    monkeypatch.setattr("ui.video_controller.save_config", slow_save)
    controller.apply_config(replace(controller.config, enabled=True))
    spin(ui_qapp, entered.is_set)
    service = controller.service
    try:
        if exit_app:
            controller.shutdown()
        else:
            assert controller.stop_recording()
        spin(ui_qapp, lambda: service.status.recording == "completed")
    finally:
        release.set()
    spin(ui_qapp, lambda: not controller._saving)
    if exit_app:
        spin(ui_qapp, lambda: controller.is_shutdown_complete)
        assert controller.service is service
    else:
        spin(ui_qapp, lambda: controller.service.status.connection == "ready")
        assert not controller.service.status.record_intent


def test_preview_only_submit_keeps_unenumerated_device_and_hidden_fields(ui_qapp, tmp_path):
    config = config_for(tmp_path)
    dialog = VideoSettingsDialog(config, preview_only=True)
    changes = []
    dialog.configuration_requested.connect(changes.append)
    try:
        dialog.devices.clear()
        dialog.enabled.setChecked(False)
        dialog.submit()
        assert changes == [replace(config, enabled=False)]
    finally:
        dialog.close()


def test_unsaved_preview_is_discarded_and_saved_preference_survives_restart(camera, ui_qapp):
    controller = camera(simulation=True)
    controller.show_settings()
    controller.dialog.enabled.setChecked(True)
    controller.dialog.close()
    assert not controller.preview_requested and not load_config(controller.config_path).enabled
    save_preview(ui_qapp, controller, True)
    spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    controller.shutdown()
    spin(ui_qapp, lambda: controller.is_shutdown_complete)
    reopened = type(controller)(config_path=controller.config_path)
    try:
        spin(ui_qapp, lambda: reopened.service.status.connection == "ready")
        assert reopened.preview_requested and not reopened.service.status.record_intent
    finally:
        reopened.shutdown()
        spin(ui_qapp, lambda: reopened.is_shutdown_complete)


def test_idle_release_does_not_close_unsaved_settings(camera, ui_qapp):
    controller = camera(simulation=True)
    start(ui_qapp, controller)
    controller.stop_recording()
    controller.show_settings()
    dialog = controller.dialog
    dialog.enabled.setChecked(True)
    spin(ui_qapp, lambda: controller.service.is_closed)
    controller.poll()
    assert controller.dialog is dialog and dialog.enabled.isChecked()
    assert not load_config(controller.config_path).enabled


def test_old_save_callback_does_not_close_reopened_settings(camera, ui_qapp, monkeypatch):
    controller = camera(simulation=True)
    entered, release = threading.Event(), threading.Event()

    def slow_save(path, config):
        entered.set()
        assert release.wait(5)
        save_config(path, config)

    monkeypatch.setattr("ui.video_controller.save_config", slow_save)
    controller.show_settings()
    controller.dialog.enabled.setChecked(True)
    controller.dialog.submit()
    spin(ui_qapp, entered.is_set)
    controller.dialog.close()
    controller.show_settings()
    reopened = controller.dialog
    release.set()
    spin(ui_qapp, lambda: not controller._saving)
    spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    assert controller.dialog is reopened
    assert not reopened.enabled.isChecked()  # Preserve this new, unsaved draft.


def test_login_gate_defers_saved_preview_and_duplicate_start_is_ignored(camera, ui_qapp):
    controller = camera(preview=True, simulation=True, start_automatically=False)
    controller.poll()
    assert not controller.service.is_started
    controller.start_preview()
    controller.start_preview()
    service = controller.service
    spin(ui_qapp, lambda: service.status.connection == "ready")
    assert not service.status.session_id
    assert controller.start_recording()
    assert not controller.start_recording()
    spin(ui_qapp, lambda: service.status.recording == "recording")
    assert controller.service is service


def test_cancel_starting_keeps_preview_without_late_recording(camera, ui_qapp):
    controller = camera(preview=True, simulation=True, start_delay=.5)
    spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    assert controller.start_recording()
    assert controller.service.status.recording == "starting"
    controller.poll()
    assert controller.service.status.elapsed(time.monotonic()) == 0
    controller.panel.record_button.click()
    spin(ui_qapp, lambda: controller.service.status.recording == "completed")
    QTest.qWait(600)
    assert not controller.service.status.record_intent
    assert controller.service.status.started_at is None
    assert not controller.service.is_closing
    assert not controller.panel.canvas.image.isNull()


def test_preview_toggle_during_recovery_does_not_resume_cancelled_recording(camera, ui_qapp):
    controller = camera(simulation=True, disconnect_after=.1, reconnect_after=.8)
    start(ui_qapp, controller)
    service = controller.service
    session, began = service.status.session_id, service.status.started_at
    spin(ui_qapp, lambda: service.status.recording == "recovering")
    save_preview(ui_qapp, controller, True)
    assert service.status.session_id == session and service.status.started_at == began
    assert controller.stop_recording()
    spin(ui_qapp, lambda: service.status.recording == "interrupted")
    spin(ui_qapp, lambda: service.status.connection == "ready")
    assert not service.status.record_intent and service.status.recording == "interrupted"
    assert not service.is_closing and controller.service is service


def test_device_draft_is_not_silently_discarded_when_recording_starts(camera, ui_qapp):
    controller = camera(simulation=True)
    controller.show_settings()
    dialog = controller.dialog
    dialog.bitrate.setValue(dialog.bitrate.value() + 100)
    before = load_config(controller.config_path)
    start(ui_qapp, controller)
    controller.poll()
    assert dialog.preview_only and not dialog.bitrate.isEnabled()
    dialog.enabled.setChecked(True)
    dialog.submit()
    assert not controller._saving and load_config(controller.config_path) == before
    assert QMessageBox.warning.call_args.args[1] == "设备参数已锁定"


def test_pending_cancel_remains_effective_during_save_and_duplicate_enter(camera, ui_qapp, monkeypatch):
    controller = camera(simulation=True, ready_delay=.8)
    assert controller.start_recording()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_save(path, config):
        calls.append(config)
        entered.set()
        assert release.wait(5)
        save_config(path, config)

    monkeypatch.setattr("ui.video_controller.save_config", slow_save)
    controller.show_settings()
    dialog = controller.dialog
    dialog.enabled.setChecked(True)
    QTest.keyClick(dialog, Qt.Key_Return)
    spin(ui_qapp, entered.is_set)
    try:
        dialog.submit()
        QTest.keyClick(dialog, Qt.Key_Return)
        assert not controller.start_recording()
        assert controller.stop_recording()
        assert not controller.record_start_pending
    finally:
        release.set()
    spin(ui_qapp, lambda: not controller._saving)
    spin(ui_qapp, lambda: controller.service.status.connection == "ready")
    assert len(calls) == 1
    assert not controller.service.status.session_id
    assert controller.preview_requested


def test_unconfirmed_preview_timeout_leaves_recording_active_and_retryable(camera, ui_qapp, monkeypatch):
    monkeypatch.setattr("ui.video_controller.PREVIEW_ACK_TIMEOUT", .05)
    controller = camera(preview=True, simulation=True)
    start(ui_qapp, controller)
    service = controller.service
    original = service.set_preview

    def lost_command(enabled):
        service._preview_revision += 1
        return True

    monkeypatch.setattr(service, "set_preview", lost_command)
    controller.show_settings()
    controller.dialog.enabled.setChecked(False)
    controller.dialog.submit()
    spin(ui_qapp, lambda: controller.dialog.message.text() == "预览未生效，请重试。")
    controller.poll()
    assert controller.panel.canvas.message == "预览已关闭，正在录像"
    assert "保存" not in controller.panel.record_button.toolTip()
    assert controller.dialog.grab().save(str(controller.config_path.parent / "preview-retry-settings.png"))
    assert controller.panel.grab().save(str(controller.config_path.parent / "preview-retry-panel.png"))
    assert service.status.recording == "recording" and not service.is_closing
    assert not load_config(controller.config_path).enabled
    assert controller.dialog is not None
    monkeypatch.setattr(service, "set_preview", original)
    controller.dialog.submit()
    spin(ui_qapp, lambda: controller.dialog is None)
    assert not controller.control_message and service.status.recording == "recording"


def test_enabling_idle_preview_waits_for_camera_ready_before_confirming(camera, ui_qapp, monkeypatch):
    monkeypatch.setattr("ui.video_controller.PREVIEW_ACK_TIMEOUT", .05)
    controller = camera(simulation=True, ready_delay=.5)
    controller.show_settings()
    controller.dialog.enabled.setChecked(True)
    controller.dialog.submit()
    spin(ui_qapp, lambda: controller.dialog.message.text() == "预览未生效，请重试。")
    controller.poll()
    assert not controller.control_message
    assert "保存" not in controller.panel.canvas.message
    assert load_config(controller.config_path).enabled
    assert controller.dialog is not None
    spin(ui_qapp, lambda: controller.dialog is None)
    assert controller.service.status.connection == "ready"
    assert not controller.control_message and not controller.service.status.session_id
