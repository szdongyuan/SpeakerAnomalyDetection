import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.acquisition_recording_defaults import load_acquisition_defaults, save_acquisition_default
from consts.running_consts import DEFAULT_PLAY_AND_RECORD_DETAIL, DEFAULT_RECORD_ONLY_DETAIL


def test_load_acquisition_defaults_missing_file_uses_false(tmp_path):
    cfg = load_acquisition_defaults(tmp_path / "missing.json")
    assert cfg["PLAY_AND_RECORD"]["use_streaming_recording"] is False
    assert cfg["RECORD_ONLY"]["use_streaming_recording"] is False


def test_save_acquisition_default_preserves_unknown_keys(tmp_path):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        '{"OTHER": {"x": 1}, "PLAY_AND_RECORD": {"legacy": 2}}',
        encoding="utf-8",
    )
    ok = save_acquisition_default("PLAY_AND_RECORD", {"use_streaming_recording": True}, path=path)
    assert ok is True
    cfg = load_acquisition_defaults(path)
    assert cfg["PLAY_AND_RECORD"]["use_streaming_recording"] is True
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["OTHER"] == {"x": 1}
    assert raw["PLAY_AND_RECORD"]["legacy"] == 2


def test_load_acquisition_defaults_malformed_json_falls_back(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")
    cfg = load_acquisition_defaults(path)
    assert cfg["PLAY_AND_RECORD"]["use_streaming_recording"] is False
    assert cfg["RECORD_ONLY"]["total_time"] == DEFAULT_RECORD_ONLY_DETAIL["total_time"]


def test_load_acquisition_defaults_wrong_top_level_type_falls_back(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    cfg = load_acquisition_defaults(path)
    assert cfg["RECORD_ONLY"]["sample_rate"] == DEFAULT_RECORD_ONLY_DETAIL["sample_rate"]


def test_load_acquisition_defaults_partial_invalid_values_fall_back_per_field(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(
        '{"RECORD_ONLY": {"total_time": "bad", "sample_rate": 48000, "use_streaming_recording": true}}',
        encoding="utf-8",
    )
    cfg = load_acquisition_defaults(path)
    assert cfg["RECORD_ONLY"]["total_time"] == DEFAULT_RECORD_ONLY_DETAIL["total_time"]
    assert cfg["RECORD_ONLY"]["sample_rate"] == 48000
    assert cfg["RECORD_ONLY"]["use_streaming_recording"] is True
    assert cfg["RECORD_ONLY"]["monitor_playback"] is False


def test_acquisition_defaults_include_recording_start_delay_ms():
    assert DEFAULT_PLAY_AND_RECORD_DETAIL["recording_start_delay_ms"] == 100.0
    assert DEFAULT_RECORD_ONLY_DETAIL["recording_start_delay_ms"] == 100.0


@pytest.mark.parametrize("value", [None, "bad", True, False, -1, float("nan"), float("inf"), float("-inf")])
def test_recording_start_delay_ms_invalid_values_default_to_100(tmp_path, value):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        json.dumps(
            {
                "PLAY_AND_RECORD": {"recording_start_delay_ms": value},
                "RECORD_ONLY": {"recording_start_delay_ms": value},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_acquisition_defaults(path)

    assert cfg["PLAY_AND_RECORD"]["recording_start_delay_ms"] == 100.0
    assert cfg["RECORD_ONLY"]["recording_start_delay_ms"] == 100.0


def test_recording_start_delay_ms_zero_and_clamp_rules(tmp_path):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        json.dumps(
            {
                "PLAY_AND_RECORD": {"recording_start_delay_ms": 0},
                "RECORD_ONLY": {"recording_start_delay_ms": 5000},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_acquisition_defaults(path)

    assert cfg["PLAY_AND_RECORD"]["recording_start_delay_ms"] == 0.0
    assert cfg["RECORD_ONLY"]["recording_start_delay_ms"] == 1000.0


def test_save_acquisition_default_persists_ms_but_not_runtime_frames(tmp_path):
    path = tmp_path / "acquisition_default_config.json"

    assert save_acquisition_default(
        "RECORD_ONLY",
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "recording_start_delay_ms": 250,
            "recording_start_delay_frames": 12000,
        },
        path=path,
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["RECORD_ONLY"]["recording_start_delay_ms"] == 250.0
    assert "recording_start_delay_frames" not in raw["RECORD_ONLY"]


def test_acquisition_defaults_persist_recording_start_delay_ms(tmp_path):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        json.dumps(
            {
                "PLAY_AND_RECORD": {
                    "legacy": 2,
                    "recording_start_delay_ms": 100,
                },
                "RECORD_ONLY": {
                    "total_time": 1.0,
                    "sample_rate": 48000,
                    "recording_start_delay_frames": 4800,
                },
            }
        ),
        encoding="utf-8",
    )

    assert save_acquisition_default(
        "PLAY_AND_RECORD",
        {
            "use_streaming_recording": True,
            "recording_start_delay_ms": 250,
            "recording_start_delay_frames": 4800,
        },
        path=path,
    )
    assert save_acquisition_default(
        "RECORD_ONLY",
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "recording_start_delay_ms": 300,
            "recording_start_delay_frames": 12000,
        },
        path=path,
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["PLAY_AND_RECORD"]["legacy"] == 2
    assert raw["PLAY_AND_RECORD"]["recording_start_delay_ms"] == 250.0
    assert raw["RECORD_ONLY"]["recording_start_delay_ms"] == 300.0
    assert "recording_start_delay_frames" not in raw["PLAY_AND_RECORD"]
    assert "recording_start_delay_frames" not in raw["RECORD_ONLY"]


def test_save_acquisition_default_scrubs_runtime_delay_frames_from_unsaved_mode(tmp_path):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        json.dumps(
            {
                "PLAY_AND_RECORD": {
                    "legacy": 2,
                    "recording_start_delay_ms": 100,
                    "nested": {
                        "recording_start_delay_frames": 4800,
                        "keep": "play",
                    },
                },
                "RECORD_ONLY": {
                    "total_time": 1.0,
                    "sample_rate": 48000,
                    "recording_start_delay_frames": 4800,
                    "nested": {
                        "recording_start_delay_custom": "forbidden",
                        "keep": "record",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert save_acquisition_default(
        "PLAY_AND_RECORD",
        {"use_streaming_recording": True},
        path=path,
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["PLAY_AND_RECORD"]["legacy"] == 2
    assert raw["PLAY_AND_RECORD"]["recording_start_delay_ms"] == 100.0
    assert raw["PLAY_AND_RECORD"]["nested"]["keep"] == "play"
    assert raw["RECORD_ONLY"]["nested"]["keep"] == "record"
    assert "recording_start_delay_frames" not in raw["PLAY_AND_RECORD"]["nested"]
    assert "recording_start_delay_frames" not in raw["RECORD_ONLY"]


def test_existing_acquisition_defaults_without_recording_delay_still_load(tmp_path):
    path = tmp_path / "acquisition_default_config.json"
    path.write_text(
        json.dumps(
            {
                "PLAY_AND_RECORD": {"use_streaming_recording": True},
                "RECORD_ONLY": {
                    "total_time": 3.0,
                    "sample_rate": 48000,
                    "monitor_playback": False,
                    "use_streaming_recording": True,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_acquisition_defaults(path)

    assert cfg["PLAY_AND_RECORD"]["use_streaming_recording"] is True
    assert cfg["RECORD_ONLY"]["sample_rate"] == 48000
    assert cfg["RECORD_ONLY"]["use_streaming_recording"] is True
    assert cfg["PLAY_AND_RECORD"]["recording_start_delay_ms"] == 100.0
    assert cfg["RECORD_ONLY"]["recording_start_delay_ms"] == 100.0
    assert "recording_start_delay_frames" not in cfg["PLAY_AND_RECORD"]
    assert "recording_start_delay_frames" not in cfg["RECORD_ONLY"]


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_record_config_window_returns_streaming_flag(qapp):
    from ui.acquisition_config_window import RecordConfigWindow

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": False,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": True,
        },
        mic={"name": "mic", "samplerate": 48000},
        speaker={"name": "speaker", "samplerate": 48000, "max_output_channels": 2},
        available_channels=[0],
    )
    assert window.streaming_recording_checkbox.isChecked() is True
    window.streaming_recording_checkbox.setChecked(False)
    window.on_click_ok_btn()
    assert window.final_data["use_streaming_recording"] is False


def test_record_config_window_returns_recording_start_delay_ms(qapp):
    from ui.acquisition_config_window import RecordConfigWindow

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": False,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": True,
            "recording_start_delay_ms": 230.0,
        },
        mic={"name": "mic", "samplerate": 48000},
        speaker={"name": "speaker", "samplerate": 48000, "max_output_channels": 2},
        available_channels=[0],
    )
    assert window.recording_start_delay_ms_input.value() == 230.0
    window.recording_start_delay_ms_input.setValue(0.0)
    window.on_click_ok_btn()
    assert window.final_data["recording_start_delay_ms"] == 0.0
    assert "recording_start_delay_frames" not in window.final_data


def test_play_record_config_window_preserves_delay_through_stimulus_window(qapp, monkeypatch):
    from ui import acquisition_config_window
    from ui.acquisition_config_window import PlayRecordConfigWindow

    class FakeStimulusWindow:
        def __init__(self, stimulus_config_data, speaker=None):
            assert "use_streaming_recording" not in stimulus_config_data
            assert "recording_start_delay_ms" not in stimulus_config_data
            self.final_save_data = dict(stimulus_config_data)
            self.final_save_data["stimulus_info"] = {"total_time": 1.5, "sample_rate": 48000}
            self.stimulus_data = None

        def on_exec(self):
            return True

    monkeypatch.setattr(acquisition_config_window, "StimulusWindow", FakeStimulusWindow)

    window = PlayRecordConfigWindow(
        {
            "stimulus_info": {"total_time": 1.0, "sample_rate": 48000},
            "use_streaming_recording": True,
            "recording_start_delay_ms": 340.0,
        },
        mic={"name": "mic", "samplerate": 48000},
        speaker={"name": "speaker", "samplerate": 48000},
    )

    assert window.recording_start_delay_ms_input.value() == 340.0
    window.recording_start_delay_ms_input.setValue(120.0)
    window.open_stimulus_window()
    window.on_click_ok_btn()

    assert window.final_data["recording_start_delay_ms"] == 120.0
    assert "recording_start_delay_frames" not in window.final_data


def test_play_record_config_window_returns_streaming_flag(qapp):
    from ui.acquisition_config_window import PlayRecordConfigWindow

    detail = {
        "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
        "use_streaming_recording": True,
    }
    window = PlayRecordConfigWindow(
        detail,
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100},
    )
    assert window.streaming_recording_checkbox.isChecked() is True
    window.streaming_recording_checkbox.setChecked(False)
    window.on_click_ok_btn()
    assert window.final_data["use_streaming_recording"] is False


def test_base_config_window_does_not_query_default_devices(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import BaseConfigWindow

    class FailingSoundDeviceManager:
        def get_default_device(self, *args, **kwargs):
            raise AssertionError("must not query default devices")

    monkeypatch.setattr(module, "SoundDeviceManager", FailingSoundDeviceManager, raising=False)

    window = BaseConfigWindow()

    assert window.mic is None
    assert window.speaker is None


def test_device_display_name_treats_malformed_payloads_as_missing(qapp):
    from ui.acquisition_config_window import BaseConfigWindow

    window = BaseConfigWindow()

    assert window._device_display_name(None, "empty") == "empty"
    assert window._device_display_name({}, "empty") == "empty"
    assert window._device_display_name({"name": ""}, "empty") == "empty"
    assert window._device_display_name({"name": "   "}, "empty") == "empty"
    assert window._device_display_name(object(), "empty") == "empty"
    assert window._device_display_name({"name": " Mic "}, "empty") == "Mic"


def test_play_record_config_window_missing_devices_disables_confirm_without_warning(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import PlayRecordConfigWindow

    warnings = []
    saved = []
    infos = []
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))
    monkeypatch.setattr(module.MessageBox, "information", lambda *args, **kwargs: infos.append((args, kwargs)))
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: saved.append(args) or True)

    detail = {
        "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
        "use_streaming_recording": True,
    }
    window = PlayRecordConfigWindow(detail, mic=None, speaker=None)
    accepted = []
    window.accept = lambda: accepted.append(True)

    assert warnings == []
    assert window.input_device_display.text() == "未选择输入设备"
    assert window.output_device_display.text() == "未选择输出设备"
    assert window.ok_btn.isEnabled() is False
    assert window.cancel_btn.isEnabled() is True

    window.on_click_ok_btn()
    assert accepted == []
    assert window.final_data is None

    window.on_click_cancel_btn()
    assert accepted == []
    assert window.final_data is None

    window.on_default_btn_clicked()
    assert saved == []
    assert warnings
    assert infos == []
    assert accepted == []
    assert window.final_data is None


def test_record_config_window_missing_mic_disables_confirm_and_skips_default_lookup(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    class FailingSoundDeviceManager:
        def get_default_device(self, *args, **kwargs):
            raise AssertionError("must not query default devices")

    monkeypatch.setattr(module, "SoundDeviceManager", FailingSoundDeviceManager, raising=False)

    record_input_data = {
        "total_time": 2.0,
        "sample_rate": 44100,
        "monitor_playback": False,
        "monitor_input_channel": 0,
        "monitor_gain_db": 0.0,
        "use_streaming_recording": False,
        "recording_start_delay_ms": 100.0,
    }
    window = RecordConfigWindow(
        record_input_data,
        mic=None,
        speaker={"name": "speaker", "samplerate": 44100, "max_output_channels": 2},
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    assert window.input_device_display.text() == "未选择输入设备"
    assert window.ok_btn.isEnabled() is False
    window.on_click_ok_btn()
    assert accepted == []
    assert window.final_data is None


def test_import_audio_config_window_missing_mic_disables_confirm_and_skips_default_lookup(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import ImportAudioConfigWindow

    class FailingSoundDeviceManager:
        def get_default_device(self, *args, **kwargs):
            raise AssertionError("must not query default devices")

    monkeypatch.setattr(module, "SoundDeviceManager", FailingSoundDeviceManager, raising=False)

    window = ImportAudioConfigWindow({"sample_rate": 44100}, mic=None)
    accepted = []
    window.accept = lambda: accepted.append(True)

    assert window.input_device_display.text() == "未选择输入设备"
    assert window.ok_btn.isEnabled() is False
    window.on_click_ok_btn()
    assert accepted == []
    assert window.final_data is None


def test_import_stimulus_audio_config_window_missing_speaker_can_confirm_offline_reference_config(
    qapp, monkeypatch
):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import ImportStimulusAudioConfigWindow

    class FailingSoundDeviceManager:
        def get_default_device(self, *args, **kwargs):
            raise AssertionError("must not query default devices")

    monkeypatch.setattr(module, "SoundDeviceManager", FailingSoundDeviceManager, raising=False)

    stimulus_config_data = {"stimulus_info": {"total_time": 1.0, "sample_rate": 44100}}
    window = ImportStimulusAudioConfigWindow(stimulus_config_data, speaker=None)
    accepted = []
    window.accept = lambda: accepted.append(True)

    assert window.ok_btn.isEnabled() is True
    window.on_click_ok_btn()
    assert accepted == [True]
    assert window.final_data == stimulus_config_data


def test_record_config_window_missing_mic_default_save_blocks_stale_sample_rate(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    saved = []
    infos = []
    warnings = []
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: saved.append(args) or True)
    monkeypatch.setattr(module.MessageBox, "information", lambda *args, **kwargs: infos.append((args, kwargs)))
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": False,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": False,
            "recording_start_delay_ms": 100.0,
        },
        mic=None,
        speaker={"name": "speaker", "samplerate": 44100, "max_output_channels": 2},
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    window.on_default_btn_clicked()

    assert saved == []
    assert infos == []
    assert warnings
    assert window.final_data is None
    assert accepted == []


def test_play_record_config_window_valid_devices_confirm_normally(qapp):
    from ui.acquisition_config_window import PlayRecordConfigWindow

    window = PlayRecordConfigWindow(
        {
            "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
            "use_streaming_recording": True,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100},
    )

    assert window.ok_btn.isEnabled() is True
    window.on_click_ok_btn()
    assert window.final_data is not None


def test_record_config_window_valid_mic_confirms_normally(qapp):
    from ui.acquisition_config_window import RecordConfigWindow

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": False,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": False,
            "recording_start_delay_ms": 100.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100, "max_output_channels": 2},
        available_channels=[0],
    )

    assert window.ok_btn.isEnabled() is True
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == 44100


def test_record_config_plain_record_only_without_monitor_playback_allows_missing_speaker(qapp):
    from ui.acquisition_config_window import RecordConfigWindow

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": False,
            "use_streaming_recording": False,
            "recording_start_delay_ms": 100.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker=None,
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    window.on_click_ok_btn()

    assert accepted == [True]
    assert window.final_data["monitor_playback"] is False
    assert window.final_data["sample_rate"] == 44100


def test_record_config_missing_speaker_preserves_monitor_state_and_allows_explicit_disable(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": True,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": True,
            "recording_start_delay_ms": 100.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker=None,
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    assert window.monitor_checkbox.isChecked() is True
    assert window.monitor_checkbox.isEnabled() is True

    window.monitor_checkbox.setChecked(False)
    window.on_click_ok_btn()

    assert warnings == []
    assert accepted == [True]
    assert window.final_data["monitor_playback"] is False


def test_record_config_missing_speaker_monitor_guard_uses_exact_prompt(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": True,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": True,
            "recording_start_delay_ms": 100.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker=None,
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    window.on_click_ok_btn()

    assert window.monitor_checkbox.isChecked() is True
    assert window.monitor_checkbox.isEnabled() is True
    assert accepted == []
    assert warnings[-1][2] == "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"


def test_record_config_zero_output_capacity_preserves_saved_monitor_state(qapp):
    from ui.acquisition_config_window import RecordConfigWindow

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "monitor_playback": True,
            "monitor_input_channel": 0,
            "monitor_gain_db": 0.0,
            "use_streaming_recording": True,
            "recording_start_delay_ms": 100.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100, "max_output_channels": 0},
        available_channels=[0],
    )

    assert window.monitor_checkbox.isChecked() is True
    assert window.monitor_checkbox.isEnabled() is True


def test_import_audio_config_window_valid_mic_confirms_normally(qapp):
    from ui.acquisition_config_window import ImportAudioConfigWindow

    window = ImportAudioConfigWindow({"sample_rate": 44100}, mic={"name": "mic", "samplerate": 48000})

    assert window.ok_btn.isEnabled() is True
    window.on_click_ok_btn()
    assert window.final_data == {}


def test_import_stimulus_audio_config_window_valid_speaker_confirms_normally(qapp):
    from ui.acquisition_config_window import ImportStimulusAudioConfigWindow

    stimulus_config_data = {
        "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
    }
    window = ImportStimulusAudioConfigWindow(stimulus_config_data, speaker={"name": "speaker", "samplerate": 48000})

    assert window.ok_btn.isEnabled() is True
    window.on_click_ok_btn()
    assert window.final_data == stimulus_config_data


def test_record_default_save_failure_warns_and_keeps_dialog_open(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    warnings = []
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: False)
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))

    window = RecordConfigWindow(
        {"total_time": 2.0, "sample_rate": 44100},
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100, "max_output_channels": 2},
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)

    window.on_default_btn_clicked()

    assert warnings
    assert accepted == []
    assert window.final_data is None


def test_record_default_save_saves_visible_values_and_keeps_dialog_open(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    saved = []
    infos = []
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: saved.append(args) or True)
    monkeypatch.setattr(module.MessageBox, "information", lambda *args, **kwargs: infos.append((args, kwargs)))

    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            "use_streaming_recording": True,
            "recording_start_delay_ms": 150.0,
        },
        mic={"name": "mic", "samplerate": 48000},
        speaker={"name": "speaker", "samplerate": 48000, "max_output_channels": 2},
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)
    window.time_input.setValue(3.5)
    assert window.samplerate_lineedit.text() == "48000"
    assert window.samplerate_lineedit.isReadOnly() is True
    assert not hasattr(window, "samplerate_combo")
    window.streaming_recording_checkbox.setChecked(False)
    window.recording_start_delay_ms_input.setValue(250.0)

    window.on_default_btn_clicked()

    assert saved == [
        (
            "RECORD_ONLY",
            {
                "total_time": 3.5,
                "sample_rate": 48000,
                "monitor_playback": False,
                "monitor_gain_db": 0.0,
                "monitor_input_channel": 0,
                "use_streaming_recording": False,
                "recording_start_delay_ms": 250.0,
            },
        )
    ]
    assert infos
    assert accepted == []
    assert window.final_data is None


def test_play_record_default_save_saves_visible_acquisition_values_and_keeps_dialog_open(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import PlayRecordConfigWindow

    saved = []
    infos = []
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: saved.append(args) or True)
    monkeypatch.setattr(module.MessageBox, "information", lambda *args, **kwargs: infos.append((args, kwargs)))

    window = PlayRecordConfigWindow(
        {
            "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
            "stimulus_signal_path": "keep.wav",
            "use_streaming_recording": True,
            "recording_start_delay_ms": 180.0,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100},
    )
    accepted = []
    window.accept = lambda: accepted.append(True)
    window.streaming_recording_checkbox.setChecked(False)
    window.recording_start_delay_ms_input.setValue(300.0)

    window.on_default_btn_clicked()

    assert saved == [
        ("PLAY_AND_RECORD", {"use_streaming_recording": False, "recording_start_delay_ms": 300.0})
    ]
    assert infos
    assert accepted == []
    assert window.final_data is None


def test_play_record_stimulus_window_does_not_receive_streaming_flag(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import PlayRecordConfigWindow

    seen = {}

    class FakeStimulusWindow:
        def __init__(self, stimulus_config_data, speaker=None):
            seen.update(stimulus_config_data)
            self.final_save_data = {
                "stimulus_info": {"total_time": 2.0, "sample_rate": 48000},
            }
            self.stimulus_data = [0, 1]

        def on_exec(self):
            return True

    monkeypatch.setattr(module, "StimulusWindow", FakeStimulusWindow)
    window = PlayRecordConfigWindow(
        {
            "stimulus_info": {"total_time": 1.0, "sample_rate": 44100, "use_custom_stimulus": True},
            "use_streaming_recording": True,
        },
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 44100},
    )

    window.open_stimulus_window()

    assert "use_streaming_recording" not in seen
    assert window.stimulus_config_data["use_streaming_recording"] is True


def _build_option_list_for_new_item(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ui.signal_analysis_window",
        types.SimpleNamespace(get_class_mapping=lambda: {}),
    )
    from ui.operation_sequence import OptionList

    option_list = OptionList.__new__(OptionList)
    option_list.config = []
    option_list.model = lambda: type("M", (), {"insertRow": lambda *args: None})()
    option_list.signal_len = 0
    option_list.mic = {"name": "mic", "samplerate": 48000}
    option_list.speaker = {"name": "speaker", "samplerate": 48000}
    return option_list


def test_record_only_new_item_uses_acquisition_default(qapp, monkeypatch):
    option_list = _build_option_list_for_new_item(monkeypatch)
    monkeypatch.setattr(
        "ui.operation_sequence.load_acquisition_defaults",
        lambda: {
            "PLAY_AND_RECORD": {"use_streaming_recording": False},
            "RECORD_ONLY": {
                "total_time": 7.5,
                "sample_rate": 48000,
                "monitor_playback": True,
                "monitor_input_channel": 1,
                "monitor_gain_db": -3.0,
                "use_streaming_recording": True,
                "recording_start_delay_ms": 220.0,
            },
        },
    )

    option_list.set_sound_item("录制音频")

    assert option_list.config[0].detail["total_time"] == 7.5
    assert option_list.config[0].detail["sample_rate"] == 48000
    assert option_list.config[0].detail["monitor_playback"] is True
    assert option_list.config[0].detail["monitor_input_channel"] == 1
    assert option_list.config[0].detail["monitor_gain_db"] == -3.0
    assert option_list.config[0].detail["use_streaming_recording"] is True
    assert option_list.config[0].detail["recording_start_delay_ms"] == 220.0
    assert "recording_start_delay_frames" not in option_list.config[0].detail


def test_play_record_new_item_merges_acquisition_default_without_dropping_stimulus(qapp, monkeypatch):
    option_list = _build_option_list_for_new_item(monkeypatch)
    stimulus_detail = {
        "stimulus_info": {"total_time": 1.5, "sample_rate": 44100},
        "stimulus_signal_path": "stimulus.wav",
        "load_stimulus_signal_path": "loaded.wav",
        "custom_key": {"keep": True},
    }
    option_list.load_stimulus_config = lambda: (True, dict(stimulus_detail))
    monkeypatch.setattr(
        "ui.operation_sequence.load_acquisition_defaults",
        lambda: {
            "PLAY_AND_RECORD": {
                "use_streaming_recording": True,
                "recording_start_delay_ms": 320.0,
            },
            "RECORD_ONLY": {},
        },
    )

    option_list.set_sound_item("播放与录制")

    detail = option_list.config[0].detail
    assert detail["use_streaming_recording"] is True
    assert detail["stimulus_info"] == stimulus_detail["stimulus_info"]
    assert detail["stimulus_signal_path"] == "stimulus.wav"
    assert detail["load_stimulus_signal_path"] == "loaded.wav"
    assert detail["custom_key"] == {"keep": True}
    assert detail["recording_start_delay_ms"] == 320.0
    assert "recording_start_delay_frames" not in detail


def test_new_items_missing_acquisition_default_use_false(qapp, monkeypatch):
    record_option_list = _build_option_list_for_new_item(monkeypatch)
    play_option_list = _build_option_list_for_new_item(monkeypatch)
    monkeypatch.setattr(
        "ui.operation_sequence.load_acquisition_defaults",
        lambda: {
            "PLAY_AND_RECORD": {},
            "RECORD_ONLY": {},
        },
    )

    record_option_list.set_sound_item("录制音频")
    assert record_option_list.config[0].detail["use_streaming_recording"] is False
    assert record_option_list.config[0].detail["recording_start_delay_ms"] == 100.0

    play_option_list.load_stimulus_config = lambda: (
        True,
        {"stimulus_info": {"total_time": 1.0, "sample_rate": 44100}},
    )
    play_option_list.set_sound_item("播放与录制")
    assert play_option_list.config[0].detail["use_streaming_recording"] is False
    assert play_option_list.config[0].detail["recording_start_delay_ms"] == 100.0
