import os

import pytest

from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from ui.acquisition_config_window import (
    ImportAudioConfigWindow,
    ImportStimulusAudioConfigWindow,
    PlayRecordConfigWindow,
    RecordConfigWindow,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_record_config_sample_rate_is_read_only_from_mic(qapp):
    window = RecordConfigWindow({"sample_rate": 44100, "total_time": 1.0}, mic={"name": "mic", "samplerate": 48000})

    assert window.samplerate_lineedit.text() == "48000"
    assert window.samplerate_lineedit.isReadOnly() is True
    assert not hasattr(window, "samplerate_combo")
    detail = window._collect_record_detail()
    assert detail["sample_rate"] == 48000


def test_import_audio_config_does_not_use_mic_samplerate_as_analysis_rate(qapp):
    window = ImportAudioConfigWindow({"sample_rate": 44100}, mic={"name": "mic", "samplerate": 48000})

    assert window.samplerate_lineedit.text() == "导入文件解码后确定"
    assert window.samplerate_lineedit.isReadOnly() is True
    assert not hasattr(window, "samplerate_combo")


def test_play_record_config_blocks_mismatched_samplerates(qapp, monkeypatch):
    detail = {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}}
    window = PlayRecordConfigWindow(
        detail,
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 48000},
    )
    warnings = []
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.on_click_ok_btn()

    assert warnings
    assert getattr(window, "final_data", None) is None


def test_play_record_stimulus_config_blocks_missing_output_before_opening_window(qapp, monkeypatch):
    detail = {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}}
    window = PlayRecordConfigWindow(
        detail,
        mic={"name": "mic", "samplerate": 44100},
        speaker=None,
    )
    warnings = []
    created_windows = []
    before_click_data = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in window.stimulus_config_data.items()
    }

    class FakeStimulusWindow:
        def __init__(self, *args, **kwargs):
            created_windows.append((args, kwargs))
            self.final_save_data = {"stimulus_info": {"sample_rate": 48000, "total_time": 2.0}}
            self.stimulus_data = [0, 1]

        def on_exec(self):
            return True

    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr("ui.acquisition_config_window.StimulusWindow", FakeStimulusWindow)

    window.open_stimulus_window()

    assert created_windows == []
    assert len(warnings) == 1
    assert warnings[0][1:] == ("采样率配置", "未选择输出设备，请在硬件管理中选择设备。")
    assert window.stimulus_config_data == before_click_data
    assert not hasattr(window, "stimulus_window")


def test_play_record_config_rebuilds_frequency_stepped_payload_at_resolved_duplex_samplerate(qapp):
    stale = generate_frequency_stepped(
        sample_rate=44100,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 2000.0],
        generate_waveform=False,
    ).metadata
    expected = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 2000.0],
        generate_waveform=False,
    ).metadata
    assert stale["segments"] != expected["segments"]
    detail = {"stimulus_info": dict(stale)}
    window = PlayRecordConfigWindow(
        detail,
        mic={"name": "mic", "samplerate": 48000},
        speaker={"name": "speaker", "samplerate": 48000},
    )

    window.on_click_ok_btn()

    final_info = window.final_data["stimulus_info"]
    assert window.final_data["sample_rate"] == 48000
    assert final_info["sample_rate"] == 48000
    assert final_info["schedule_sample_rate"] == 48000
    assert final_info["schedule_provenance"] == expected["schedule_provenance"]
    assert final_info["segments"] == expected["segments"]
    assert final_info["playback_sample_count"] == expected["playback_sample_count"]


def test_play_record_default_save_blocks_mismatched_samplerates(qapp, monkeypatch):
    saved = []
    warnings = []
    monkeypatch.setattr(
        "ui.acquisition_config_window.save_acquisition_default",
        lambda *args, **kwargs: saved.append(args) or True,
    )
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.information", lambda *args: None)
    window = PlayRecordConfigWindow(
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 48000},
    )

    window.on_default_btn_clicked()

    assert saved == []
    assert warnings
    assert "不一致" in warnings[-1][-1]


def test_record_config_invalid_mic_samplerate_keeps_confirmation_unavailable(qapp):
    window = RecordConfigWindow({"sample_rate": 44100, "total_time": 1.0}, mic={"name": "mic", "samplerate": 96000})

    assert not window.ok_btn.isEnabled()
    assert "采样率" in window.samplerate_lineedit.text()
    assert window.samplerate_lineedit.text() != "44100"
    assert window.samplerate_lineedit.isReadOnly() is True


def test_record_config_invalid_mic_does_not_collect_stale_sample_rate(qapp):
    window = RecordConfigWindow({"sample_rate": 44100, "total_time": 1.0}, mic={"name": "mic", "samplerate": 96000})

    detail = window._collect_record_detail()

    assert "sample_rate" not in detail


def test_record_config_invalid_mic_default_save_blocks_stale_sample_rate(qapp, monkeypatch):
    saved = []
    warnings = []
    monkeypatch.setattr(
        "ui.acquisition_config_window.save_acquisition_default",
        lambda *args, **kwargs: saved.append(args) or True,
    )
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.information", lambda *args: None)
    window = RecordConfigWindow({"sample_rate": 44100, "total_time": 1.0}, mic={"name": "mic", "samplerate": 96000})

    window.on_default_btn_clicked()

    assert saved == []
    assert warnings


def test_record_config_monitor_default_save_blocks_mismatched_samplerates(qapp, monkeypatch):
    saved = []
    warnings = []
    monkeypatch.setattr(
        "ui.acquisition_config_window.save_acquisition_default",
        lambda *args, **kwargs: saved.append(args) or True,
    )
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr("ui.acquisition_config_window.MessageBox.information", lambda *args: None)
    window = RecordConfigWindow(
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": True},
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 48000, "max_output_channels": 2},
    )

    window.on_default_btn_clicked()

    assert saved == []
    assert warnings
    assert "不一致" in warnings[-1][-1]


def test_import_stimulus_audio_config_does_not_present_speaker_samplerate_as_analysis_rate(qapp):
    window = ImportStimulusAudioConfigWindow(
        {"stimulus_info": {"sample_rate": 44100}},
        mic={"name": "mic", "samplerate": 44100},
        speaker={"name": "speaker", "samplerate": 48000},
    )

    assert not hasattr(window, "samplerate_combo")
    assert "48000" not in window.windowTitle()


def test_import_stimulus_audio_opens_stimulus_window_for_offline_reference_authoring(qapp, monkeypatch):
    seen = {}

    class FakeStimulusWindow:
        def __init__(self, stimulus_config_data, speaker=None, offline_reference_authoring=False):
            seen["stimulus_config_data"] = stimulus_config_data
            seen["speaker"] = speaker
            seen["offline_reference_authoring"] = offline_reference_authoring
            self.final_save_data = {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}}
            self.stimulus_data = [0, 1]

        def on_exec(self):
            return True

    monkeypatch.setattr("ui.acquisition_config_window.StimulusWindow", FakeStimulusWindow)
    window = ImportStimulusAudioConfigWindow(
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        speaker={"name": "speaker", "sample_rate": 48000},
    )

    window.open_stimulus_window()

    assert seen["speaker"] == {"name": "speaker", "sample_rate": 48000}
    assert seen["offline_reference_authoring"] is True
    assert window.stimulus_config_data == {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}}
