"""Actual system dialogs use their explicit Enter target, preserving editors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractButton, QComboBox, QDialogButtonBox, QLineEdit, QPushButton

from base.video.config import VideoConfig
from ui.acquisition_config_window import BaseConfigWindow, RecordConfigWindow
from ui.calibration_window import CalibrationWindow, OutputCalibration
from ui.hardware_window import HardwareSelectionView
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
from ui.serial_discrete_input_config_dialog import SerialDiscreteInputConfigDialog
from ui.video_settings_dialog import VideoSettingsDialog


@pytest.fixture(params=[(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier)],
                ids=["return", "keypad-enter"])
def enter(request):
    def press(widget):
        widget.setFocus()
        QTest.keyClick(widget, *request.param)
    return press


@pytest.fixture
def opened(ui_qapp, monkeypatch):
    windows = []
    monkeypatch.setattr("base.log_manager.LogManager.set_log_handler", lambda *args: Mock())

    def show(window):
        windows.append(window)
        window.show()
        window.activateWindow()
        ui_qapp.processEvents()
        return window

    yield show
    for window in reversed(windows):
        window.close()
        window.deleteLater()
    ui_qapp.processEvents()


def button_named(window, text):
    return next(button for button in window.findChildren(QPushButton)
                if button.text().replace(" ", "") == text)


def watch_buttons(window):
    clicks = []
    for button in window.findChildren(QAbstractButton):
        button.clicked.connect(lambda checked=False, button=button: clicks.append(button))
    return clicks


@pytest.fixture(params=["record", "hardware", "serial", "video", "video-read-only",
                        "login", "add-account", "change-password"])
def system_dialog(request, opened, monkeypatch):
    callbacks = []

    def observe(cls, method):
        monkeypatch.setattr(cls, method, lambda *args: callbacks.append(method))

    kind = request.param
    if kind == "record":
        observe(RecordConfigWindow, "on_click_ok_btn")
        window = RecordConfigWindow({}, mic={"name": "Soundcard"}, speaker={"name": "Output"})
        target = button_named(window, "确认")
        auxiliary = button_named(window, "取消")
        editors = [window.time_input.lineEdit(), window.samplerate_combo, window.input_device_display]
    elif kind == "hardware":
        # This view has no discovery or persistence until a controller is attached.
        window = HardwareSelectionView()
        target, auxiliary = window.ok_btn, window.refresh_btn
        target.clicked.connect(lambda: callbacks.append("selected"))
        window.driver_combo.addItems(["Test driver", "Other driver"])
        editors = [window.driver_combo]
    elif kind == "serial":
        monkeypatch.setattr("ui.serial_discrete_input_config_dialog.list_ports", SimpleNamespace(
            comports=lambda: [SimpleNamespace(device="COM3", description="Test port")]))
        observe(SerialDiscreteInputConfigDialog, "_on_ok_btn_clicked")
        observe(SerialDiscreteInputConfigDialog, "_on_test_btn_clicked")
        window = SerialDiscreteInputConfigDialog({})
        target, auxiliary = window.ok_btn, window.test_btn
        editors = [window.device_model_lineedit, window.port_combobox.lineEdit(),
                   window.baudrate_combobox.lineEdit()]
    elif kind.startswith("video"):
        observe(VideoSettingsDialog, "choose_directory")
        window = VideoSettingsDialog(VideoConfig(), read_only=kind == "video-read-only")
        window.configuration_requested.connect(lambda config: callbacks.append(config))
        window.probe_requested.connect(lambda: callbacks.append("probe"))
        target = window.buttons.button(QDialogButtonBox.Close if window.read_only else QDialogButtonBox.Save)
        auxiliary = window.refresh_button
        # Read-only mode disables every parameter; editable mode exercises the editors.
        editors = [] if window.read_only else [window.folder, window.bitrate.lineEdit(), window.resolution]
    elif kind == "login":
        observe(LoginWindow, "login_click")
        observe(LoginWindow, "add_account_click")
        observe(LoginWindow, "change_pwd_click")
        window = LoginWindow()
        target, auxiliary = window.login_button, window.add_account_botton
        editors = [window.username_input, window.password_input, window.access_selection]
    elif kind == "add-account":
        observe(AddAccountWindow, "add_user_click")
        window = AddAccountWindow(Mock())
        target, auxiliary = button_named(window, "添加账号"), button_named(window, "退出")
        editors = [window.username_input, window.password_input, window.access_selection]
    else:
        observe(ChangePwdWindow, "change_pwd_click")
        window = ChangePwdWindow("test-user", Mock())
        target, auxiliary = button_named(window, "修改密码"), window.info
        auxiliary.setFocusPolicy(Qt.StrongFocus)
        editors = [window.password_input, window.confirm_password_input]
    return opened(window), target, auxiliary, editors, callbacks


def test_system_editors_preserve_values_without_clicking(system_dialog, enter):
    window, target, auxiliary, editors, callbacks = system_dialog
    clicks = watch_buttons(window)
    for editor in editors:
        before = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
        finished = []
        if isinstance(editor, QLineEdit):
            editor.editingFinished.connect(lambda: finished.append(True))
        enter(editor)
        assert clicks == []
        assert callbacks == []
        after = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
        assert after == before
        if isinstance(editor, QLineEdit):
            assert finished
    assert window.isVisible()


def test_system_auxiliary_focus_clicks_only_confirm(system_dialog, enter):
    window, target, auxiliary, editors, callbacks = system_dialog
    clicks = watch_buttons(window)
    enter(auxiliary)
    assert clicks == [target]
    if isinstance(window, VideoSettingsDialog) and window.read_only:
        assert not window.isVisible()
        assert callbacks == []
    else:
        assert len(callbacks) == 1


@pytest.mark.parametrize("tab_index", [0, 1], ids=["output", "input"])
def test_calibration_enter_never_clicks_buttons(opened, monkeypatch, enter, tab_index, ui_qapp):
    callbacks = []
    for cls, methods in (
        (CalibrationWindow, ["clicked_calibration_button", "clicked_reset_button", "clicked_close_button"]),
        (OutputCalibration, ["play_btn_clicked", "save_btn_clicked", "test_calibration"]),
    ):
        for method in methods:
            monkeypatch.setattr(cls, method, lambda *args, method=method: callbacks.append(method))
    monkeypatch.setattr("ui.calibration_window.load_mic_channel_v2pa_factors", lambda device: {})
    window = opened(CalibrationWindow(
        input_device={"index": 7, "name": "Test Microphone", "hostapi": 3, "max_input_channels": 2},
        input_channels=[0, 1]))
    window.tabwidget.setCurrentIndex(tab_index)
    # Include the output Save button's enabled state without starting playback.
    window.output_cal_wnd.save_btn.setEnabled(True)
    ui_qapp.processEvents()
    clicks = watch_buttons(window)
    editors = (list(window.output_cal_wnd.findChildren(QLineEdit)) if tab_index == 0 else
               [window.input_cal_wnd.channel_combo_box, window.input_cal_wnd.v2pa_factor_lineedit])
    buttons = [button for button in window.findChildren(QAbstractButton)
               if button.isVisible() and button.isEnabled()]
    assert buttons and editors
    for widget in editors + buttons + [window.tabwidget.tabBar()]:
        enter(widget)
        assert clicks == []
        assert callbacks == []
        assert window.isVisible()
    QTest.mouseClick(window.cal_btn, Qt.LeftButton)
    assert clicks == [window.cal_btn]
    assert callbacks == ["clicked_calibration_button"]


def test_record_base_does_not_inherit_policy(opened):
    window = opened(BaseConfigWindow(mic={"name": "Soundcard"}))
    window.main_layout.addLayout(window.create_cancel_ok_buttons())
    assert not hasattr(window, "_enter_policy")


@pytest.mark.parametrize("kind", ["login", "add-account", "change-password", "serial", "record"])
def test_enter_keeps_existing_validation(opened, monkeypatch, enter, kind):
    warnings, writes = [], []
    monkeypatch.setattr("ui.login_window.QMessageBox.warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr("ui.login_window.DataSave", lambda *args: writes.append(args))
    if kind == "login":
        monkeypatch.setattr(LoginWindow, "get_user_info_from_db", lambda *args: {})
        monkeypatch.setattr("ui.login_window.get_mac_address", lambda: "test-mac")
        window = opened(LoginWindow())
        auxiliary = window.add_account_botton
    elif kind == "add-account":
        window = opened(AddAccountWindow(Mock()))
        auxiliary = button_named(window, "退出")
    elif kind == "change-password":
        window = opened(ChangePwdWindow("test-user", Mock()))
        auxiliary = window.info
        auxiliary.setFocusPolicy(Qt.StrongFocus)
    elif kind == "serial":
        monkeypatch.setattr("ui.serial_discrete_input_config_dialog.list_ports", SimpleNamespace(comports=lambda: []))
        window = opened(SerialDiscreteInputConfigDialog({}))
        auxiliary = window.test_btn
    else:
        window = opened(RecordConfigWindow({"sample_rate": 123},
                        mic={"name": "Soundcard"}, speaker={"name": "Output"}))
        auxiliary = button_named(window, "取消")
    enter(auxiliary)
    assert window.isVisible()
    assert writes == []
    if kind == "add-account":
        assert window.info.text() == "添加账号失败"
    else:
        assert len(warnings) == 1
