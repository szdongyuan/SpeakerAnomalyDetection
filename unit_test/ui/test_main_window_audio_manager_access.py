import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent

import main_window as main_window_module


@pytest.fixture
def window(ui_qapp, monkeypatch):
    monkeypatch.setattr(main_window_module.LogManager, "set_log_handler", logging.getLogger)
    monkeypatch.setattr(
        main_window_module, "restore_or_default", lambda **kwargs: (None, None, [], [])
    )
    monkeypatch.setattr(main_window_module.MainWindow, "init_ui", lambda self: None)
    monkeypatch.setattr(
        main_window_module.MainWindow, "_init_ve_hardware_runtime", lambda self: None
    )
    instance = main_window_module.MainWindow(
        recording_bridge=SimpleNamespace(shutdown=lambda: None, service=object())
    )
    instance.function_audio_manager.triggered.connect(instance.on_audio_manager_init)
    instance.on_access_lvl_changed()
    yield instance
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(instance, QEvent.DeferredDelete)


@pytest.mark.parametrize("access_lvl", [None, "", "Unknown"])
def test_audio_manager_menu_is_disabled_without_valid_role(window, access_lvl):
    window.access_lvl = access_lvl
    window.on_access_lvl_changed()

    assert not window.function_audio_manager.isEnabled()


@pytest.mark.parametrize("access_lvl", ["Operator", "Engineer", "Admin"])
def test_audio_manager_opens_for_authenticated_roles(window, monkeypatch, access_lvl):
    dialog = Mock()
    monkeypatch.setattr(main_window_module, "ArchiveAudioDataDialog", dialog)
    window.access_lvl = access_lvl
    window.on_access_lvl_changed()

    assert window.function_audio_manager.isEnabled()
    window.function_audio_manager.trigger()

    dialog.assert_called_once()
    dialog.return_value.exec.assert_called_once_with()


def test_cancelled_initial_login_keeps_audio_manager_disabled(window, monkeypatch):
    login = Mock()
    login.return_value.on_exec.return_value = (None, None)
    monkeypatch.setattr(main_window_module, "LoginWindow", login)

    window.on_login_window_init()

    assert window.access_lvl is None
    assert not window.function_audio_manager.isEnabled()


def test_audio_manager_is_disabled_when_access_is_cleared(window):
    window.access_lvl = "Admin"
    window.on_access_lvl_changed()
    assert window.function_audio_manager.isEnabled()

    window.access_lvl = None
    window.on_access_lvl_changed()

    assert not window.function_audio_manager.isEnabled()
