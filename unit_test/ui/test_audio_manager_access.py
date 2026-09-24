from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMenu


def activate_audio_manager(window):
    # Exercise menu interaction: QAction.trigger() bypasses disabled state in Qt5.
    menu = window.audio_test_menu
    menu.popup(QPoint(0, 0))
    QApplication.processEvents()
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(window.function_audio_manager).center())
    menu.hide()


@pytest.fixture
def audio_manager_window(ui_qapp, monkeypatch):
    import main_window as module
    from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
    from ui import video_controller

    monkeypatch.setattr(module, "restore_or_default", lambda **kwargs: ({}, {}, [], []))
    monkeypatch.setattr(module.LogManager, "set_log_handler", Mock())
    monkeypatch.setattr(video_controller, "VideoController", Mock())
    monkeypatch.setattr(module.MainWindow, "_init_ve_hardware_runtime", lambda self: None)
    monkeypatch.setattr(module.MainWindow, "closeEvent", lambda self, event: event.accept())

    def init_ui(window):
        window.audio_test_menu = QMenu(window)
        window.audio_test_menu.addAction(window.function_audio_manager)
        window.function_audio_manager.triggered.connect(window.on_audio_manager_init)
        window.on_access_lvl_changed()

    monkeypatch.setattr(module.MainWindow, "init_ui", init_ui)
    lifetime = VePrewarmLifetime()
    monkeypatch.setattr(ui_qapp, "_ve_prewarm_lifetime", lifetime, raising=False)
    recording_bridge = SimpleNamespace(service=object(), shutdown=lambda: None)
    raw_audio_csv_bridge = SimpleNamespace(service=object())
    dialog_factory = Mock()
    monkeypatch.setattr(module, "ArchiveAudioDataDialog", dialog_factory)
    login_factory = Mock()
    login_factory.return_value.on_exec.return_value = (None, None)
    monkeypatch.setattr(module, "LoginWindow", login_factory)

    window = module.MainWindow(
        ve_prewarm_lifetime=lifetime,
        recording_bridge=recording_bridge,
        raw_audio_csv_bridge=raw_audio_csv_bridge,
        ve_profile_store=object(),
        ve_calibration_store=object(),
    )
    yield SimpleNamespace(window=window, dialog=dialog_factory, login=login_factory)
    ui_qapp.aboutToQuit.disconnect(recording_bridge.shutdown)
    ui_qapp.aboutToQuit.disconnect(window.video_controller.shutdown)
    window.close()
    window.deleteLater()


@pytest.mark.parametrize(
    "role, enabled",
    [(None, False), ("unknown", False), ("Operator", True), ("Engineer", True), ("Admin", True)],
)
def test_audio_manager_matches_report_export_permissions(audio_manager_window, role, enabled):
    harness = audio_manager_window
    window = harness.window
    window.access_lvl = role
    window.on_access_lvl_changed()

    activate_audio_manager(window)

    assert harness.dialog.call_count == int(enabled)
    assert window.function_audio_manager.isEnabled() is enabled
    assert window.function_action_report_export.isEnabled() is enabled
    if enabled:
        assert harness.dialog.call_args.kwargs == {
            "raw_audio_csv_service": window.raw_audio_csv_service,
            "recording_service": window.recording_bridge.service,
        }
        harness.dialog.return_value.exec.assert_called_once_with()


def test_audio_manager_is_disabled_before_login_and_after_cancellation(audio_manager_window):
    harness = audio_manager_window
    window = harness.window

    assert window.access_lvl is None
    assert not window.function_audio_manager.isEnabled()
    activate_audio_manager(window)
    harness.dialog.assert_not_called()

    window.on_login_window_init()
    harness.login.return_value.on_exec.assert_called_once_with()
    assert window.access_lvl is None
    assert not window.function_audio_manager.isEnabled()
    assert not window.function_action_report_export.isEnabled()
    activate_audio_manager(window)
    harness.dialog.assert_not_called()


def test_audio_manager_refreshes_on_role_transitions(audio_manager_window):
    harness = audio_manager_window
    window = harness.window
    for role in ("Admin", "Operator", "Engineer", "unknown", "Operator", None):
        window.access_lvl = role
        window.on_access_lvl_changed()
        enabled = role in ("Operator", "Engineer", "Admin")
        harness.dialog.reset_mock()

        activate_audio_manager(window)

        assert harness.dialog.call_count == int(enabled)
        assert window.function_audio_manager.isEnabled() is enabled
        assert window.function_action_report_export.isEnabled() is enabled
