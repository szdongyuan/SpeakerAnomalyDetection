import importlib
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QWidget


def _install_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeAction:
    def __init__(self, *args, **kwargs):
        self.text = args[0] if args else ""
        self.triggered_callback = None
        self.triggered = types.SimpleNamespace(connect=self._connect, disconnect=self._disconnect)

    def _connect(self, callback):
        self.triggered_callback = callback

    def _disconnect(self):
        self.triggered_callback = None

    def trigger(self):
        if self.triggered_callback is not None:
            self.triggered_callback()

    def setDisabled(self, value):
        self.disabled = bool(value)

    def setEnabled(self, value):
        self.enabled = bool(value)


class _FakeMenu:
    def __init__(self, text=""):
        self.text = text
        self.actions = []

    def addAction(self, action):
        self.actions.append(action)
        return action

    def addSeparator(self):
        self.actions.append(None)


class _FakeMenuBar:
    def __init__(self, *args, **kwargs):
        self.menus = []
        self.actions = []

    def addMenu(self, text):
        menu = _FakeMenu(text)
        self.menus.append(menu)
        return menu

    def addAction(self, text):
        action = _FakeAction(text)
        self.actions.append(action)
        return action


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
        "ui.hardware_management_window": types.SimpleNamespace(
            open_hardware_management_window=lambda **kwargs: None
        ),
        "ui.custom_ui_widget.widgets": types.SimpleNamespace(
            PushButton=QWidget,
            MenuBar=_FakeMenuBar,
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
        self.init_calls = 0
        self.player_status_flag = False
        self._record_workflow_busy = False
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "PLAY_AND_RECORD",
                        "detail": {"stimulus_info": {"sample_rate": 44100}},
                    }
                }
            }
        ]
        self.data_struct = types.SimpleNamespace(
            sample_rate=None,
            stimulus_data=None,
            stimulus_info=None,
        )

    def set_audio_devices_available(self, available, message=""):
        self.available_calls.append((available, message))

    def refresh_channel_windows(self):
        self.refresh_calls += 1

    def init_data_struct_stimulus_config(self):
        self.init_calls += 1
        if not self.mic or not self.speaker:
            return
        mic_rate = self.mic.get("samplerate")
        speaker_rate = self.speaker.get("samplerate")
        if mic_rate is None or mic_rate != speaker_rate:
            return
        self.data_struct.sample_rate = mic_rate
        self.data_struct.stimulus_data = [0.0]
        self.data_struct.stimulus_info = {"sample_rate": mic_rate}


def _window(main_window_module):
    window = main_window_module.MainWindow.__new__(main_window_module.MainWindow)
    window.sequence_window = FakeSequenceWindow()
    window.statusbar_updates = 0
    window.update_statusbar = lambda: setattr(window, "statusbar_updates", window.statusbar_updates + 1)
    window.mic = None
    window.speaker = None
    window.mic_channels = []
    window.access_lvl = "Engineer"
    window.startup_device_error_reason = ""
    window.startup_device_notice_message = ""
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = None
    window.device_workflow_available = False
    window.tray_popup_button = object()
    return window


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_hardware_management_menu_action_matches_hardware_role_and_opens_window(qapp, main_window_module, monkeypatch):
    class FakeManager:
        def get_startup_devices(self):
            return {
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "device_available": False,
            }

    opened = []
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(main_window_module.MainWindow, "init_ui", lambda self: None)
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: opened.append(kwargs),
    )

    window = main_window_module.MainWindow()
    window.access_lvl = "Engineer"
    menu_bar = window.init_menu()

    assert window.hardware_action_management.text == "硬件管理"
    assert window.hardware_action_management in window.widget_list_engineer
    assert window.hardware_action_selection in window.widget_list_engineer
    assert window.hardware_action_calibration in window.widget_list_engineer
    assert window.hardware_action_management in window.widget_list_admin
    assert window.hardware_action_selection in window.widget_list_admin
    assert window.hardware_action_calibration in window.widget_list_admin

    hardware_menu = next(menu for menu in menu_bar.menus if menu.text == "硬件")
    assert window.hardware_action_management in hardware_menu.actions

    window.hardware_action_management.trigger()

    assert len(opened) == 1
    assert opened[0]["parent"] is window
    assert opened[0]["audio_workflow_active_provider"]() is False


def test_init_sequence_widget_reloads_play_record_stimulus_after_attaching_devices(
    qapp, main_window_module, monkeypatch
):
    from base.audio_sample_rate import resolve_duplex_sample_rate

    class AttachOrderSequenceWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.sequence_config = [
                {
                    "seq1": {
                        "acq": {
                            "mode": "PLAY_AND_RECORD",
                            "detail": {
                                "sample_rate": 44100,
                                "stimulus_info": {
                                    "sample_rate": 44100,
                                    "total_time": 0.01,
                                },
                            },
                        }
                    }
                }
            ]
            self.data_struct = types.SimpleNamespace(sample_rate=None)
            self.using_config_path = "sequence.json"
            self.failed_init_messages = []
            self.stimulus_setup_calls = []
            self.available_calls = []
            self.init_data_struct_stimulus_config()

        def init_data_struct_stimulus_config(self):
            detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
            result = resolve_duplex_sample_rate(getattr(self, "mic", None), getattr(self, "speaker", None))
            if not result.ok:
                self.failed_init_messages.append(result.message)
                return
            self.data_struct.sample_rate = result.sample_rate
            self.stimulus_setup_calls.append(
                {
                    "detail_config_rate": detail["stimulus_info"]["sample_rate"],
                    "runtime_sample_rate": result.sample_rate,
                }
            )

        def set_audio_devices_available(self, available, message=""):
            self.available_calls.append((available, message))

    monkeypatch.setattr(main_window_module, "SequenceWindow", AttachOrderSequenceWindow)
    monkeypatch.setattr(main_window_module.MainWindow, "init_menu", lambda self: QWidget())
    monkeypatch.setattr(main_window_module.MainWindow, "set_title", lambda self: QWidget())
    monkeypatch.setattr(main_window_module.MainWindow, "init_ui", lambda self: None)

    window = main_window_module.MainWindow()
    window.mic = {"name": "Mic", "samplerate": 48000}
    window.speaker = {"name": "Speaker", "samplerate": 48000}
    window.mic_channels = [0]
    window.device_workflow_available = True
    window.startup_device_error_reason = ""

    main_window_module.MainWindow.init_sequence_widget(window)

    assert window.sequence_window.failed_init_messages
    assert window.sequence_window.mic == window.mic
    assert window.sequence_window.speaker == window.speaker
    assert window.sequence_window.mic_channels == [0]
    assert window.sequence_window.stimulus_setup_calls == [
        {"detail_config_rate": 44100, "runtime_sample_rate": 48000}
    ]
    assert window.sequence_window.data_struct.sample_rate == 48000


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


def test_apply_audio_devices_unavailable_clears_play_record_runtime_stimulus(main_window_module):
    window = _window(main_window_module)
    window.sequence_window.data_struct.sample_rate = 48000
    window.sequence_window.data_struct.stimulus_data = [1.0]
    window.sequence_window.data_struct.stimulus_info = {"sample_rate": 48000}
    window.sequence_window.data_struct.alignment_sample_count = 5

    window._apply_audio_devices(None, None, [], available=False, message="设备不可用")

    assert window.sequence_window.init_calls == 0
    assert window.sequence_window.data_struct.sample_rate is None
    assert window.sequence_window.data_struct.stimulus_data is None
    assert window.sequence_window.data_struct.stimulus_info is None
    assert not hasattr(window.sequence_window.data_struct, "alignment_sample_count")
    assert window.sequence_window.available_calls[-1] == (False, "设备不可用")


def test_deleted_selected_hardware_callback_clears_main_window_devices(main_window_module):
    window = _window(main_window_module)
    window.mic = {"hardware_id": "hw-1", "name": "Old Mic", "samplerate": 48000}
    window.speaker = {"hardware_id": "hw-2", "name": "Old Speaker", "samplerate": 48000}
    window.mic_channels = [0]
    window.device_workflow_available = True
    window.sequence_window.mic = window.mic
    window.sequence_window.speaker = window.speaker
    window.sequence_window.mic_channels = [0]

    window.on_selected_audio_hardware_deleted("hw-1")

    assert window.mic is None
    assert window.speaker is None
    assert window.mic_channels == []
    assert window.sequence_window.mic is None
    assert window.sequence_window.speaker is None
    assert window.sequence_window.mic_channels == []
    assert window.sequence_window.available_calls[-1][0] is False
    assert "已删除" in window.sequence_window.available_calls[-1][1]
    assert window.statusbar_updates == 1


def test_registered_hardware_update_refreshes_selected_mic(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.mic = {
        "hardware_id": "hw-1",
        "name": "Runtime Mic",
        "device_name": "Runtime Mic",
        "index": 11,
        "hostapi": 0,
        "hostapi_name": "API",
        "default_samplerate": 48000.0,
        "samplerate": 44100,
        "max_input_channels": 2,
    }
    window.speaker = {
        "hardware_id": "hw-2",
        "name": "Runtime Speaker",
        "device_name": "Runtime Speaker",
        "hostapi": 0,
        "hostapi_name": "API",
        "default_samplerate": 48000.0,
        "samplerate": 48000,
        "max_output_channels": 2,
    }
    window.mic_channels = [0]
    window.device_workflow_available = True
    saved = []
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda mic, speaker, channels: saved.append((dict(mic), dict(speaker), list(channels)))),
    )

    window.on_registered_audio_hardware_updated(
        "hw-1",
        {
            "hardware_id": "hw-1",
            "display_name": "Edited Mic",
            "device_name": "Runtime Mic",
            "hardware_type": "microphone",
            "hostapi_name": "API",
            "samplerate": 48000,
            "bit_depth": 32,
            "latency_ms": 100,
        },
    )

    assert window.mic["samplerate"] == 48000
    assert window.mic["display_name"] == "Edited Mic"
    assert window.mic["index"] == 11
    assert window.mic["name"] == "Runtime Mic"
    assert window.mic["hostapi"] == 0
    assert window.mic["default_samplerate"] == 48000.0
    assert window.mic["max_input_channels"] == 2
    assert window.sequence_window.mic["samplerate"] == 48000
    assert saved[-1][0]["samplerate"] == 48000
    assert saved[-1][2] == [0]
    assert window.sequence_window.init_calls == 1


def test_registered_hardware_update_refreshes_selected_speaker(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.mic = {
        "hardware_id": "hw-1",
        "name": "Runtime Mic",
        "device_name": "Runtime Mic",
        "hostapi": 0,
        "hostapi_name": "API",
        "default_samplerate": 48000.0,
        "samplerate": 48000,
        "max_input_channels": 2,
    }
    window.speaker = {
        "hardware_id": "hw-2",
        "name": "Runtime Speaker",
        "device_name": "Runtime Speaker",
        "index": 12,
        "hostapi": 0,
        "hostapi_name": "API",
        "default_samplerate": 48000.0,
        "samplerate": 44100,
        "max_output_channels": 2,
    }
    window.mic_channels = [0]
    window.device_workflow_available = True
    saved = []
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda mic, speaker, channels: saved.append((dict(mic), dict(speaker), list(channels)))),
    )

    window.on_registered_audio_hardware_updated(
        "hw-2",
        {
            "hardware_id": "hw-2",
            "display_name": "Edited Speaker",
            "device_name": "Runtime Speaker",
            "hardware_type": "speaker",
            "hostapi_name": "API",
            "samplerate": 48000,
            "bit_depth": 24,
            "latency_ms": 50,
        },
    )

    assert window.speaker["samplerate"] == 48000
    assert window.speaker["display_name"] == "Edited Speaker"
    assert window.speaker["index"] == 12
    assert window.speaker["name"] == "Runtime Speaker"
    assert window.speaker["hostapi"] == 0
    assert window.speaker["default_samplerate"] == 48000.0
    assert window.speaker["max_output_channels"] == 2
    assert window.sequence_window.speaker["samplerate"] == 48000
    assert saved[-1][1]["samplerate"] == 48000
    assert saved[-1][2] == [0]
    assert window.sequence_window.init_calls == 1


def test_registered_hardware_update_refreshes_mic_and_speaker_when_same_hardware_id(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    window.mic = {
        "hardware_id": "interface-1",
        "name": "Interface",
        "device_name": "Interface",
        "index": 3,
        "hostapi": 1,
        "hostapi_name": "ASIO",
        "default_samplerate": 48000.0,
        "samplerate": 44100,
        "max_input_channels": 4,
        "max_output_channels": 4,
    }
    window.speaker = dict(window.mic)
    window.mic_channels = [0, 1]
    saved = []
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda mic, speaker, channels: saved.append((dict(mic), dict(speaker), list(channels)))),
    )

    window.on_registered_audio_hardware_updated(
        "interface-1",
        {
            "hardware_id": "interface-1",
            "display_name": "Edited Interface",
            "device_name": "Interface",
            "hardware_type": "audio_interface",
            "hostapi_name": "ASIO",
            "samplerate": 48000,
            "bit_depth": 32,
            "latency_ms": 100,
        },
    )

    assert window.mic["samplerate"] == 48000
    assert window.speaker["samplerate"] == 48000
    assert window.sequence_window.mic["samplerate"] == 48000
    assert window.sequence_window.speaker["samplerate"] == 48000
    assert saved[-1][0]["samplerate"] == 48000
    assert saved[-1][1]["samplerate"] == 48000
    assert saved[-1][2] == [0, 1]


def test_registered_hardware_update_ignores_unselected_hardware(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.mic = {"hardware_id": "hw-1", "samplerate": 44100}
    window.speaker = {"hardware_id": "hw-2", "samplerate": 44100}
    saved = []
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda *args: saved.append(args)),
    )

    window.on_registered_audio_hardware_updated("hw-3", {"hardware_id": "hw-3", "samplerate": 48000})

    assert saved == []
    assert window.mic["samplerate"] == 44100
    assert window.speaker["samplerate"] == 44100


@pytest.mark.parametrize("busy_attr", ["player_status_flag", "_record_workflow_busy"])
def test_registered_hardware_update_during_active_run_preserves_runtime_state(
    main_window_module, monkeypatch, busy_attr
):
    window = _window(main_window_module)
    mic = {"hardware_id": "hw-1", "name": "Mic", "samplerate": 44100}
    speaker = {"hardware_id": "hw-2", "name": "Speaker", "samplerate": 44100}
    window.mic = mic
    window.speaker = speaker
    window.mic_channels = [0]
    window.device_workflow_available = True
    window.sequence_window.mic = mic
    window.sequence_window.speaker = speaker
    window.sequence_window.mic_channels = [0]
    window.sequence_window.data_struct.sample_rate = 44100
    window.sequence_window.data_struct.stimulus_data = [1.0]
    window.sequence_window.data_struct.stimulus_info = {"sample_rate": 44100}
    setattr(window.sequence_window, busy_attr, True)
    warnings = []
    saved = []
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda *args: saved.append(args)),
    )

    window.on_registered_audio_hardware_updated("hw-1", {"hardware_id": "hw-1", "samplerate": 48000})

    assert window.mic is mic
    assert window.speaker is speaker
    assert window.sequence_window.mic is mic
    assert window.sequence_window.speaker is speaker
    assert window.sequence_window.data_struct.sample_rate == 44100
    assert window.sequence_window.data_struct.stimulus_data == [1.0]
    assert window.sequence_window.data_struct.stimulus_info == {"sample_rate": 44100}
    assert saved == []
    assert warnings
    assert "播放或录音进行中" in warnings[-1][2]


def test_registered_hardware_update_save_failure_marks_devices_unavailable(main_window_module, monkeypatch):
    window = _window(main_window_module)
    window.mic = {"hardware_id": "hw-1", "samplerate": 44100}
    window.speaker = {"hardware_id": "hw-2", "samplerate": 48000}
    window.mic_channels = [0]
    warnings = []
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    monkeypatch.setattr(
        main_window_module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda *args: (_ for _ in ()).throw(OSError("disk denied"))),
    )

    window.on_registered_audio_hardware_updated("hw-1", {"hardware_id": "hw-1", "samplerate": 48000})

    assert warnings
    assert window.mic is None
    assert window.speaker is None
    assert window.sequence_window.mic is None
    assert window.device_workflow_available is False
    assert "重新选择" in window.sequence_window.available_calls[-1][1]


@pytest.mark.parametrize("busy_attr", ["player_status_flag", "_record_workflow_busy"])
def test_deleted_selected_hardware_callback_during_active_run_preserves_runtime_state(
    main_window_module, monkeypatch, busy_attr
):
    window = _window(main_window_module)
    mic = {"hardware_id": "hw-1", "name": "Old Mic", "samplerate": 48000}
    speaker = {"hardware_id": "hw-2", "name": "Old Speaker", "samplerate": 48000}
    window.mic = mic
    window.speaker = speaker
    window.mic_channels = [0]
    window.device_workflow_available = True
    window.sequence_window.mic = mic
    window.sequence_window.speaker = speaker
    window.sequence_window.mic_channels = [0]
    window.sequence_window.data_struct.sample_rate = 48000
    window.sequence_window.data_struct.stimulus_data = [1.0]
    window.sequence_window.data_struct.stimulus_info = {"sample_rate": 48000, "stimulus_id": "stim-1"}
    setattr(window.sequence_window, busy_attr, True)
    warnings = []
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    window.on_selected_audio_hardware_deleted("hw-1")

    assert window.mic == mic
    assert window.speaker == speaker
    assert window.mic_channels == [0]
    assert window.device_workflow_available is True
    assert window.sequence_window.mic == mic
    assert window.sequence_window.speaker == speaker
    assert window.sequence_window.mic_channels == [0]
    assert window.sequence_window.data_struct.sample_rate == 48000
    assert window.sequence_window.data_struct.stimulus_data == [1.0]
    assert window.sequence_window.data_struct.stimulus_info == {"sample_rate": 48000, "stimulus_id": "stim-1"}
    assert window.sequence_window.available_calls == []
    assert window.statusbar_updates == 0
    assert warnings
    assert "播放或录音进行中" in warnings[-1][2]


@pytest.mark.parametrize("busy_attr", ["player_status_flag", "_record_workflow_busy"])
def test_hardware_management_menu_blocks_active_run(main_window_module, monkeypatch, busy_attr):
    window = _window(main_window_module)
    setattr(window.sequence_window, busy_attr, True)
    opened = []
    warnings = []
    monkeypatch.setattr(main_window_module, "open_hardware_management_window", lambda **kwargs: opened.append(kwargs))
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    window.on_hardware_management_window_init()

    assert opened == []
    assert warnings
    assert "播放或录音进行中" in warnings[-1][2]


def test_hardware_management_menu_passes_active_run_provider(main_window_module, monkeypatch):
    window = _window(main_window_module)
    opened = []
    monkeypatch.setattr(main_window_module, "open_hardware_management_window", lambda **kwargs: opened.append(kwargs))

    window.on_hardware_management_window_init()

    assert len(opened) == 1
    provider = opened[0]["audio_workflow_active_provider"]
    assert provider() is False
    window.sequence_window._record_workflow_busy = True
    assert provider() is True


def test_apply_audio_devices_success_reinitializes_play_record_stimulus(main_window_module):
    window = _window(main_window_module)
    mic = {"name": "Mic", "samplerate": 48000}
    speaker = {"name": "Speaker", "samplerate": 48000}

    window._apply_audio_devices(mic, speaker, [0], available=True)

    assert window.sequence_window.init_calls == 1
    assert window.sequence_window.data_struct.sample_rate == 48000
    assert window.sequence_window.data_struct.stimulus_info == {"sample_rate": 48000}


def test_apply_audio_devices_failed_reinit_clears_stale_play_record_stimulus(main_window_module):
    window = _window(main_window_module)
    window.sequence_window.data_struct.sample_rate = 48000
    window.sequence_window.data_struct.stimulus_data = [1.0]
    window.sequence_window.data_struct.stimulus_info = {"sample_rate": 48000}
    window.sequence_window.data_struct.alignment_sample_count = 5

    window._apply_audio_devices(
        {"name": "Mic", "samplerate": 44100},
        {"name": "Speaker", "samplerate": 48000},
        [0],
        available=True,
    )

    assert window.sequence_window.init_calls == 1
    assert window.sequence_window.data_struct.sample_rate is None
    assert window.sequence_window.data_struct.stimulus_data is None
    assert window.sequence_window.data_struct.stimulus_info is None
    assert not hasattr(window.sequence_window.data_struct, "alignment_sample_count")


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


def test_register_first_startup_recovery_opens_hardware_management_not_selection(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    warnings = []
    selection_calls = []
    management_calls = []

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": False,
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "startup_device_error_reason": "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。",
                "can_retry_saved_devices": False,
                "startup_recovery_action": "register_hardware",
            }

    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs) or object(),
    )

    window.show_startup_device_warning()

    assert warnings
    warning_text = warnings[-1][2]
    assert "请先在硬件管理中注册硬件" in warning_text
    assert "重新选择设备" not in warning_text
    assert "重新扫描设备" not in warning_text
    assert selection_calls == []
    assert len(management_calls) == 1
    assert management_calls[0]["parent"] is window
    assert management_calls[0]["audio_workflow_active_provider"]() is False


def test_register_first_recovery_rechecks_startup_state_after_hardware_management_closes(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    warnings = []
    selection_calls = []
    management_calls = []

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": False,
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "startup_device_error_reason": "已保存的注册硬件已删除或不存在，请在硬件管理中重新选择设备。",
                "startup_notice_message": "请重新选择设备。",
                "can_retry_saved_devices": False,
                "startup_recovery_action": None,
            }

    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs) or object(),
    )

    window.show_startup_device_warning()

    assert len(management_calls) == 1
    assert selection_calls == [True]
    assert window.startup_recovery_action is None
    assert window.startup_can_retry_saved_devices is False
    assert window.startup_device_error_reason == "已保存的注册硬件已删除或不存在，请在硬件管理中重新选择设备。"
    assert window.startup_device_notice_message == "请重新选择设备。"


def test_register_first_recovery_applies_devices_after_hardware_management_closes(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    register_first_message = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = register_first_message
    window.startup_device_notice_message = register_first_message
    mic = {"name": "Mic", "index": 1, "samplerate": 48000}
    speaker = {"name": "Speaker", "index": 2, "samplerate": 48000}
    selection_calls = []
    management_window = object()
    management_calls = []

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": True,
                "mic": mic,
                "speaker": speaker,
                "mic_channels": [0],
                "startup_device_error_reason": None,
                "startup_notice_message": None,
                "can_retry_saved_devices": True,
                "startup_recovery_action": None,
            }

    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs) or management_window,
    )

    window.on_hardware_management_window_init(startup_register_recovery=True)

    assert len(management_calls) == 1
    assert management_calls[0]["parent"] is window
    assert selection_calls == []
    assert window.mic == mic
    assert window.speaker == speaker
    assert window.mic_channels == [0]
    assert window.startup_recovery_action is None
    assert window.startup_device_error_reason is None
    assert window.startup_device_notice_message is None
    assert window.sequence_window.available_calls[-1] == (True, "")
    assert window.sequence_window.init_calls == 1


def test_register_first_recovery_stays_active_after_hardware_management_closes_without_devices(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    old_message = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    refreshed_message = "已保存的注册硬件仍不存在，请先在硬件管理中注册硬件。"
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = old_message
    window.startup_device_notice_message = old_message
    selection_calls = []
    management_window = object()
    management_calls = []

    class FakeManager:
        def get_startup_devices(self):
            return {
                "device_available": False,
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "startup_device_error_reason": refreshed_message,
                "startup_notice_message": refreshed_message,
                "can_retry_saved_devices": False,
                "startup_recovery_action": "register_hardware",
            }

    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: FakeManager())
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs) or management_window,
    )

    window.on_hardware_management_window_init(startup_register_recovery=True)

    assert len(management_calls) == 1
    assert management_calls[0]["parent"] is window
    assert selection_calls == []
    assert window.startup_recovery_action == "register_hardware"
    assert window.startup_can_retry_saved_devices is False
    assert window.startup_device_error_reason == refreshed_message
    assert window.startup_device_notice_message == refreshed_message
    assert window.mic is None
    assert window.speaker is None
    assert window.mic_channels == []
    assert window.sequence_window.available_calls == []
    assert window.sequence_window.init_calls == 0


def test_register_first_recovery_preserves_state_when_hardware_management_fails_to_open(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    register_first_message = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = register_first_message
    window.startup_device_notice_message = register_first_message
    selection_calls = []
    management_calls = []

    class FakeManager:
        def __init__(self):
            self.startup_calls = 0

        def get_startup_devices(self):
            self.startup_calls += 1
            return {
                "device_available": False,
                "mic": None,
                "speaker": None,
                "mic_channels": [],
                "startup_device_error_reason": "硬件管理未打开时不应重新计算启动设备。",
                "startup_notice_message": "硬件管理未打开时不应重新选择设备。",
                "can_retry_saved_devices": False,
                "startup_recovery_action": None,
            }

    manager = FakeManager()
    monkeypatch.setattr(main_window_module, "SoundDeviceManager", lambda: manager)
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs) or None,
    )

    window.on_hardware_management_window_init(startup_register_recovery=True)

    assert len(management_calls) == 1
    assert management_calls[0]["parent"] is window
    assert selection_calls == []
    assert manager.startup_calls == 0
    assert window.startup_recovery_action == "register_hardware"
    assert window.startup_device_error_reason == register_first_message
    assert window.startup_device_notice_message == register_first_message
    assert window.mic is None
    assert window.speaker is None
    assert window.mic_channels == []
    assert window.sequence_window.available_calls == []
    assert window.sequence_window.init_calls == 0


def test_register_first_startup_recovery_blocks_hardware_management_for_operator(
    main_window_module, monkeypatch
):
    window = _window(main_window_module)
    window.access_lvl = "Operator"
    window.startup_can_retry_saved_devices = False
    window.startup_recovery_action = "register_hardware"
    window.startup_device_error_reason = "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    warnings = []
    selection_calls = []
    management_calls = []
    monkeypatch.setattr(
        main_window_module.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    monkeypatch.setattr(window, "_open_hardware_selection_for_recovery", lambda: selection_calls.append(True))
    monkeypatch.setattr(
        main_window_module,
        "open_hardware_management_window",
        lambda **kwargs: management_calls.append(kwargs),
    )

    window.show_startup_device_warning()

    assert selection_calls == []
    assert management_calls == []
    assert warnings
    assert len(warnings) == 1
    assert "点击确认后将打开硬件管理" not in warnings[0][2]
    assert "当前用户无硬件管理权限，请联系工程师或管理员注册硬件。" in warnings[0][2]


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
