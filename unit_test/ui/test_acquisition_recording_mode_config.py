import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.acquisition_recording_defaults import load_acquisition_defaults, save_acquisition_default
from consts.running_consts import DEFAULT_RECORD_ONLY_DETAIL


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
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
        available_channels=[0],
    )
    assert window.streaming_recording_checkbox.isChecked() is True
    window.streaming_recording_checkbox.setChecked(False)
    window.on_click_ok_btn()
    assert window.final_data["use_streaming_recording"] is False


def test_play_record_config_window_returns_streaming_flag(qapp):
    from ui.acquisition_config_window import PlayRecordConfigWindow

    detail = {
        "stimulus_info": {"total_time": 1.0, "sample_rate": 44100},
        "use_streaming_recording": True,
    }
    window = PlayRecordConfigWindow(
        detail,
        mic={"name": "mic"},
        speaker={"name": "speaker"},
    )
    assert window.streaming_recording_checkbox.isChecked() is True
    window.streaming_recording_checkbox.setChecked(False)
    window.on_click_ok_btn()
    assert window.final_data["use_streaming_recording"] is False


def test_record_default_save_failure_warns_and_keeps_dialog_open(qapp, monkeypatch):
    from ui import acquisition_config_window as module
    from ui.acquisition_config_window import RecordConfigWindow

    warnings = []
    monkeypatch.setattr(module, "save_acquisition_default", lambda *args, **kwargs: False)
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: warnings.append((args, kwargs)))

    window = RecordConfigWindow(
        {"total_time": 2.0, "sample_rate": 44100},
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
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
        {"total_time": 2.0, "sample_rate": 44100, "use_streaming_recording": True},
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
        available_channels=[0],
    )
    accepted = []
    window.accept = lambda: accepted.append(True)
    window.time_input.setValue(3.5)
    window.samplerate_combo.setCurrentText("48000")
    window.streaming_recording_checkbox.setChecked(False)

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
            },
        )
    ]
    assert infos
    assert accepted == []
    assert window.final_data is None


def test_play_record_default_save_saves_only_streaming_flag_and_keeps_dialog_open(qapp, monkeypatch):
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
        },
        mic={"name": "mic"},
        speaker={"name": "speaker"},
    )
    accepted = []
    window.accept = lambda: accepted.append(True)
    window.streaming_recording_checkbox.setChecked(False)

    window.on_default_btn_clicked()

    assert saved == [("PLAY_AND_RECORD", {"use_streaming_recording": False})]
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
        mic={"name": "mic"},
        speaker={"name": "speaker"},
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
            "PLAY_AND_RECORD": {"use_streaming_recording": True},
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

    play_option_list.load_stimulus_config = lambda: (
        True,
        {"stimulus_info": {"total_time": 1.0, "sample_rate": 44100}},
    )
    play_option_list.set_sound_item("播放与录制")
    assert play_option_list.config[0].detail["use_streaming_recording"] is False


def test_delete_item_removes_excel_and_pdf_save_items(qapp, monkeypatch):
    from ui.operation_sequence import OptionList

    option_list = OptionList.__new__(OptionList)
    option_list.config = [
        types.SimpleNamespace(
            analysis_list={
                "SPL1": {"type": "SPL"},
                "Excel1": {"type": "Excel", "save_items": ["SPL1", "FR1"]},
                "PDF1": {"type": "PDF", "save_items": ["SPL1", "FR1"]},
            }
        )
    ]

    option_list.delete_item_config("SPL1")

    assert "SPL1" not in option_list.config[0].analysis_list
    assert option_list.config[0].analysis_list["Excel1"]["save_items"] == ["FR1"]
    assert option_list.config[0].analysis_list["PDF1"]["save_items"] == ["FR1"]


def test_rename_item_updates_excel_and_pdf_save_items(qapp, monkeypatch):
    from ui.operation_sequence import OptionList

    option_list = OptionList.__new__(OptionList)
    display_sequence = ["SPL1", "Excel1", "PDF1"]
    option_list.config = [
        types.SimpleNamespace(
            analysis_list={
                "SPL1": {"type": "SPL"},
                "Excel1": {"type": "Excel", "save_items": ["SPL1", "FR1"]},
                "PDF1": {"type": "PDF", "save_items": ["SPL1", "FR1"]},
            }
        )
    ]

    option_list.update_config_data("SPL1", "SPL_A", display_sequence)

    assert "SPL1" not in option_list.config[0].analysis_list
    assert option_list.config[0].analysis_list["SPL_A"] == {"type": "SPL"}
    assert display_sequence == ["SPL_A", "Excel1", "PDF1"]
    assert option_list.config[0].analysis_list["Excel1"]["save_items"] == ["SPL_A", "FR1"]
    assert option_list.config[0].analysis_list["PDF1"]["save_items"] == ["SPL_A", "FR1"]
