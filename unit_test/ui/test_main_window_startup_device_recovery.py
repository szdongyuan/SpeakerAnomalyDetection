import importlib
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QWidget


def _install_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeAction:
    def __init__(self, *args, **kwargs):
        self.triggered = types.SimpleNamespace(connect=lambda *a, **k: None, disconnect=lambda *a, **k: None)

    def setDisabled(self, value):
        self.disabled = bool(value)

    def setEnabled(self, value):
        self.enabled = bool(value)


class _FakeMessageBox:
    @staticmethod
    def warning(*args, **kwargs):
        return None


@pytest.fixture
def main_window_module(monkeypatch):
    stubs = {
        "base.log_manager": types.SimpleNamespace(
            LogManager=types.SimpleNamespace(set_log_handler=lambda *args, **kwargs: None)
        ),
        "base.db_manager": types.SimpleNamespace(
            DataSave=object,
            ensure_system_database_ready=lambda: None,
        ),
        "ui.ai_window": types.SimpleNamespace(AiWindow=object),
        "ui.archive_audio_data_dialog": types.SimpleNamespace(ArchiveAudioDataDialog=object),
        "ui.calibration_window": types.SimpleNamespace(CalibrationWindow=object),
        "ui.login_window": types.SimpleNamespace(
            AddAccountWindow=object,
            ChangePwdWindow=object,
            LoginWindow=object,
        ),
        "ui.operation_sequence": types.SimpleNamespace(AnalysisModelSelect=object),
        "ui.sequence.sequence_widget": types.SimpleNamespace(SequenceWindow=object),
        "ui.custom_ui_widget.traypopuppanel": types.SimpleNamespace(TrayPopupButton=object),
        "ui.ui_src": types.SimpleNamespace(ui_resources=object()),
        "ui.hardware_window": types.SimpleNamespace(open_hardware_selection_window=lambda **kwargs: (None, None, [])),
        "ui.custom_ui_widget.widgets": types.SimpleNamespace(
            PushButton=QWidget,
            MenuBar=QWidget,
            Label=QWidget,
            Action=_FakeAction,
            MessageBox=_FakeMessageBox,
        ),
    }
    previous_main_window = sys.modules.pop("main_window", None)
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, _install_module(name, **vars(attrs)))

    try:
        yield importlib.import_module("main_window")
    finally:
        sys.modules.pop("main_window", None)
        if previous_main_window is not None:
            sys.modules["main_window"] = previous_main_window


class FakeSequenceWindow:
    def __init__(self):
        self.mic = "old-mic"
        self.speaker = "old-speaker"
        self.mic_channels = [9]
        self.available_calls = []
        self.refresh_calls = 0
        self.player_status_flag = False

    def set_audio_devices_available(self, available, message=""):
        self.available_calls.append((available, message))

    def refresh_channel_windows(self):
        self.refresh_calls += 1


def _window(main_window_module):
    window = main_window_module.MainWindow.__new__(main_window_module.MainWindow)
    window.sequence_window = FakeSequenceWindow()
    window.statusbar_updates = 0
    window.update_statusbar = lambda: setattr(window, "statusbar_updates", window.statusbar_updates + 1)
    window.mic = None
    window.speaker = None
    window.mic_channels = []
    window.startup_device_error_reason = ""
    window.startup_device_notice_message = ""
    window.startup_can_retry_saved_devices = False
    window.device_workflow_available = False
    window.tray_popup_button = object()
    return window


def test_apply_audio_devices_unavailable_disables_sequence_workflow(main_window_module):
    window = _window(main_window_module)

    window._apply_audio_devices(None, None, [], available=False, message="设备不可用")

    assert window.mic is None
    assert window.speaker is None
    assert window.mic_channels == []
    assert window.sequence_window.mic is None
    assert window.sequence_window.speaker is None
    assert window.sequence_window.mic_channels == []
    assert window.sequence_window.available_calls[-1] == (False, "设备不可用")
    assert window.statusbar_updates == 1


def test_startup_unavailable_state_schedules_recovery_prompt(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.startup_device_notice_message = "设备不可用"
    window.startup_device_error_reason = "设备不可用"
    window.device_workflow_available = False
    scheduled = []
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda delay, callback: scheduled.append((delay, callback))),
    )

    window._schedule_startup_device_recovery_if_needed()

    assert scheduled
    assert scheduled[-1][0] == 0
    assert scheduled[-1][1] == window.show_startup_device_warning


def test_apply_startup_audio_devices_syncs_unavailable_state_before_prompt(main_window_module):
    window = _window(main_window_module)
    window.startup_device_error_reason = "设备不可用"
    window.device_workflow_available = False

    window._apply_startup_audio_devices_to_sequence()

    assert window.sequence_window.mic is None
    assert window.sequence_window.speaker is None
    assert window.sequence_window.mic_channels == []
    assert window.sequence_window.available_calls[-1] == (False, "设备不可用")


def test_recovery_retry_success_applies_saved_devices(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.startup_can_retry_saved_devices = True
    mic = {"name": "Mic", "index": 1}
    speaker = {"name": "Speaker", "index": 2}

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": True,
                "mic": mic,
                "speaker": speaker,
                "mic_channels": [0],
                "startup_device_error_reason": None,
                "can_retry_saved_devices": True,
            }

    hardware_calls = []
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: hardware_calls.append(True))

    window._retry_or_select_startup_devices()

    assert window.mic == mic
    assert window.speaker == speaker
    assert window.mic_channels == [0]
    assert window.sequence_window.available_calls[-1] == (True, "")
    assert hardware_calls == []


def test_recovery_retry_failure_opens_hardware_selection(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.startup_can_retry_saved_devices = True

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": False,
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "startup_device_error_reason": "仍不可用",
                "can_retry_saved_devices": True,
            }

    hardware_calls = []
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: hardware_calls.append(True))

    window._retry_or_select_startup_devices()

    assert hardware_calls == [True]


def test_recovery_without_retryable_devices_opens_hardware_selection_after_refresh(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.startup_can_retry_saved_devices = False

    class FakeManager:
        def __init__(self):
            self.refresh_calls = 0

        def refresh_available_device(self):
            self.refresh_calls += 1

    manager = FakeManager()
    hardware_calls = []
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: manager)
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: hardware_calls.append(True))

    window._retry_or_select_startup_devices()

    assert manager.refresh_calls == 1
    assert hardware_calls == [True]


def test_hardware_recovery_success_enables_workflow(main_window_module, monkeypatch):
    window = _window(main_window_module)
    mic = {"name": "New Mic", "index": 10, "hostapi": 0}
    speaker = {"name": "New Speaker", "index": 11, "hostapi": 0}
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kwargs: (speaker, mic, [1]))

    window._open_hardware_selection_for_recovery()

    assert window.mic == mic
    assert window.speaker == speaker
    assert window.mic_channels == [1]
    assert window.sequence_window.available_calls[-1] == (True, "")


def test_hardware_recovery_cancel_keeps_workflow_disabled(main_window_module, monkeypatch):
    window = _window(main_window_module)
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kwargs: (None, None, []))
    warnings = []
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    window._open_hardware_selection_for_recovery()

    assert window.device_workflow_available is False
    assert window.sequence_window.available_calls[-1][0] is False
    assert warnings


def test_menu_hardware_success_restores_unavailable_workflow(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.device_workflow_available = False
    window.sequence_window.player_status_flag = False
    mic = {"name": "New Mic", "index": 10, "hostapi": 0}
    speaker = {"name": "New Speaker", "index": 11, "hostapi": 0}
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kwargs: (speaker, mic, [1]))

    window.on_hardware_window_init()

    assert window.device_workflow_available is True
    assert window.mic == mic
    assert window.speaker == speaker
    assert window.mic_channels == [1]
    assert window.sequence_window.available_calls[-1] == (True, "")


def test_menu_hardware_cancel_preserves_available_devices(main_window_module, monkeypatch):
    window = _window(main_window_module)
    old_mic = {"name": "Old Mic", "index": 1, "hostapi": 0}
    old_speaker = {"name": "Old Speaker", "index": 2, "hostapi": 0}
    window.mic = old_mic
    window.speaker = old_speaker
    window.mic_channels = [0]
    window.device_workflow_available = True
    window.sequence_window.player_status_flag = False
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kwargs: (old_speaker, old_mic, [0]))

    window.on_hardware_window_init()

    assert window.mic == old_mic
    assert window.speaker == old_speaker
    assert window.mic_channels == [0]
    assert window.sequence_window.available_calls[-1] == (True, "")


def test_menu_hardware_cancel_keeps_unavailable_state(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.device_workflow_available = False
    window.sequence_window.player_status_flag = False
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kwargs: (None, None, []))

    window.on_hardware_window_init()

    assert window.device_workflow_available is False
    assert window.sequence_window.available_calls[-1][0] is False
