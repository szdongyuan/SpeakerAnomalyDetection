"""Camera input is synthetic; control IPC and Qt are real."""

import threading
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QMessageBox

from base.video.config import VideoConfig, load_config, save_config
from base.video.service import VideoService
from base.video.worker import SimulationOptions, simulated_video_worker
from ui.video_controller import VideoController
from ui.video_monitor_widget import VideoMonitorWidget
from ui.video_demo import configure_demo_font
from unit_test.ui.test_video_service_bridge import spin
from unit_test.video.test_runtime import synthetic_video_worker


@pytest.fixture
def camera(ui_qapp, tmp_path, monkeypatch):
    controllers = []
    monkeypatch.setattr(QMessageBox, "warning", Mock())

    def make(*, preview=False, simulation=False, ready_delay=0, start_automatically=True, **faults):
        root = tmp_path / str(len(controllers))
        path = root / "settings.json"
        config = VideoConfig(enabled=preview, device_id="synthetic", width=160, height=90)
        save_config(path, config)

        class Controller(VideoController):
            def _make_service(self):
                return VideoService(
                    worker_target=simulated_video_worker if simulation else synthetic_video_worker,
                    worker_options=SimulationOptions(preview_enabled=self.config.enabled,
                                                     ready_delay=ready_delay, **faults) if simulation else self.config,
                    generation=self._generation, initial_preview_enabled=self.config.enabled,
                    initial_connection="connecting" if self.config.enabled else "disabled",
                    heartbeat_timeout=8, shutdown_timeout=10,
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


def save_preview(app, controller, enabled):
    controller.show_settings()
    controller.dialog.enabled.setChecked(enabled)
    controller.dialog.submit()
    spin(app, lambda: not controller._saving)
    assert load_config(controller.config_path).enabled is enabled
    spin(app, lambda: controller.dialog is None, timeout=8)


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
        assert reopened.preview_requested
    finally:
        reopened.shutdown()
        spin(ui_qapp, lambda: reopened.is_shutdown_complete)


@pytest.fixture
def delayed_save(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_save(path, config):
        entered.set()
        assert release.wait(5)
        save_config(path, config)

    monkeypatch.setattr("ui.video_controller.save_config", slow_save)
    yield entered, release
    release.set()


def test_old_save_callback_does_not_close_reopened_settings(camera, ui_qapp, delayed_save):
    controller = camera(simulation=True)
    entered, release = delayed_save
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


def test_failed_save_keeps_live_preview_and_saved_preference(camera, ui_qapp, monkeypatch):
    controller = camera(preview=True, simulation=True)
    service = controller.service
    spin(ui_qapp, lambda: service.latest_preview() is not None)
    before = controller.config_path.read_bytes()
    monkeypatch.setattr("ui.video_controller.save_config", Mock(side_effect=PermissionError("locked")))
    controller.show_settings()
    controller.dialog.enabled.setChecked(False)
    controller.dialog.submit()
    spin(ui_qapp, lambda: not controller._saving)
    QMessageBox.warning.assert_called_once()
    assert controller.config_path.read_bytes() == before
    assert controller.preview_requested and controller.service is service
    sequence = service.latest_preview().sequence
    spin(ui_qapp, lambda: service.latest_preview(sequence) is not None)


def test_shutdown_waits_for_save_without_starting_preview(camera, ui_qapp, delayed_save):
    controller = camera(simulation=True)
    service = controller.service
    entered, release = delayed_save
    controller.show_settings()
    controller.dialog.enabled.setChecked(True)
    controller.dialog.submit()
    spin(ui_qapp, entered.is_set)
    controller.shutdown()
    assert not controller.is_shutdown_complete
    release.set()
    spin(ui_qapp, lambda: controller.is_shutdown_complete)
    assert load_config(controller.config_path).enabled
    assert controller.service is service and not service.is_started


def test_login_gate_defers_saved_preview_and_duplicate_start_is_ignored(camera, ui_qapp):
    controller = camera(preview=True, simulation=True, start_automatically=False)
    controller.poll()
    assert not controller.service.is_started
    controller.start_preview()
    controller.start_preview()
    service = controller.service
    spin(ui_qapp, lambda: service.status.connection == "ready")
    assert controller.service is service


def test_enabling_idle_preview_waits_for_camera_ready_before_confirming(camera, ui_qapp, monkeypatch):
    monkeypatch.setattr("ui.video_controller.PREVIEW_ACK_TIMEOUT", .05)
    controller = camera(simulation=True, ready_delay=.5)
    controller.show_settings()
    controller.dialog.enabled.setChecked(True)
    controller.dialog.submit()
    spin(ui_qapp, lambda: controller.dialog.message.text() == "预览未生效，请重试。")
    controller.poll()
    assert "保存" not in controller.panel.canvas.message
    assert load_config(controller.config_path).enabled
    assert controller.dialog is not None
    spin(ui_qapp, lambda: controller.dialog is None)
    assert controller.service.status.connection == "ready"

def test_disable_releases_camera_and_reenable_recreates_preview(camera, ui_qapp):
    controller = camera(preview=True)
    spin(ui_qapp, lambda: not controller.panel.canvas.image.isNull())
    previous = controller.service
    save_preview(ui_qapp, controller, False)
    spin(ui_qapp, lambda: previous.is_closed)
    assert controller.panel.canvas.image.isNull()
    save_preview(ui_qapp, controller, True)
    spin(ui_qapp, lambda: not controller.panel.canvas.image.isNull())
    assert controller.service is not previous
