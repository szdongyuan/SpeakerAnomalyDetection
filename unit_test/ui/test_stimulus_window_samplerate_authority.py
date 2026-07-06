import os
from types import SimpleNamespace

import numpy as np
import pytest

from ui.stimulus_window import StimulusWindow


def _stimulus_config(sample_rate=44100):
    return {
        "stimulus_info": {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "sample_rate": sample_rate,
            "start_freq": 100,
            "stop_freq": 1000,
            "total_time": 0.01,
            "repeat_times": 1,
            "amplitude": 1.0,
            "voltage": 1.0,
            "voltage_type": "RMS",
            "use_custom_stimulus": True,
        }
    }


def _frequency_stimulus_config(sample_rate=44100):
    return {
        "stimulus_info": {
            "stimulus_method": "frequency_stepped",
            "stimulus_label": "step(sc)",
            "frequency_mode": "custom_linear",
            "stimulus_type": "custom_linear",
            "start_freq": 100,
            "stop_freq": 400,
            "num_steps": 3,
            "frequencies": [100, 250, 400],
            "min_duration": 0.02,
            "min_cycles": 4.0,
            "repeat_times": 1,
            "sample_rate": sample_rate,
            "voltage_type": "RMS",
            "voltage": 1.0,
            "amplitude": 1.0,
            "use_custom_stimulus": True,
        }
    }


def _set_legacy_external_wav_branch(window, path="external.wav"):
    window.stimulus_info["use_custom_stimulus"] = False
    previous_signal_state = window.custom_chk_box.blockSignals(True)
    try:
        window.custom_chk_box.setChecked(False)
    finally:
        window.custom_chk_box.blockSignals(previous_signal_state)
    window.load_wav_path = path
    window.load_stimulus_signal_path = path
    window._legacy_external_wav_loaded_by_user = True


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_stimulus_window_displays_speaker_samplerate_as_read_only(qapp):
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )

    assert window.stimulus_info["sample_rate"] == 48000
    assert window.sample_rate_lineedit.text() == "48000"
    assert window.sample_rate_lineedit.isReadOnly()


def test_missing_speaker_samplerate_display_shows_unresolved_text_without_numeric_fallback(qapp, monkeypatch):
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: None)
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "sample_rate": 48000, "index": 7},
    )

    display_text = window.sample_rate_lineedit.text()

    assert display_text
    assert "采样率" in display_text
    assert display_text not in {"44100", "48000"}
    assert "44100" not in display_text
    assert "48000" not in display_text
    assert window.sample_rate_lineedit.isReadOnly()


def test_invalid_speaker_samplerate_display_does_not_show_numeric_fallback(qapp, monkeypatch):
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: None)
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=48000),
        speaker={"name": "speaker", "samplerate": "not-a-number", "index": 7},
    )

    display_text = window.sample_rate_lineedit.text()

    assert display_text
    assert "采样率" in display_text
    assert display_text not in {"44100", "48000"}
    assert "44100" not in display_text
    assert "48000" not in display_text
    assert window.sample_rate_lineedit.isReadOnly()


def test_offline_reference_authoring_display_explains_definition_rate_not_hardware(qapp):
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker=None,
        offline_reference_authoring=True,
    )

    tooltip = window.sample_rate_lineedit.toolTip()

    assert window.sample_rate_lineedit.text() == "44100"
    assert window.sample_rate_lineedit.isReadOnly()
    assert "定义" in tooltip
    assert "分析" in tooltip
    assert "导入录音" in tooltip
    assert "硬件管理" not in tooltip
    assert "输出设备" not in tooltip


def test_stimulus_window_load_wav_uses_speaker_samplerate(qapp, monkeypatch):
    calls = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    monkeypatch.setattr("ui.stimulus_window.QFileDialog.getOpenFileName", lambda *args, **kwargs: ("stim.wav", ""))
    monkeypatch.setattr(
        "ui.stimulus_window.load_audio_simple",
        lambda path, sr: calls.append((path, sr)) or (np.zeros(4), np.arange(4)),
    )
    window.graph_stimulus = lambda: None

    window.load_wav_btn_clicked()

    assert calls == [("stim.wav", 48000)]


def test_stimulus_window_preview_uses_speaker_samplerate(qapp, monkeypatch):
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    window.stimulus_data = np.zeros(8, dtype=np.float32)
    calls = []
    monkeypatch.setattr(
        "ui.stimulus_window.SoundcardAudioProcessor.sd_play",
        lambda self, params: calls.append(params) or (0, "ok"),
    )

    window.play_btn_clicked()

    assert calls[-1]["sr"] == 48000
    assert calls[-1]["device"] == 7


def test_offline_reference_authoring_preview_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    play_calls = []
    warnings = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "sample_rate": 48000, "index": 7},
        offline_reference_authoring=True,
    )
    window.stimulus_data = np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(
        "ui.stimulus_window.SoundcardAudioProcessor.sd_play",
        lambda self, params: play_calls.append(params) or (0, "ok"),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.play_btn_clicked()

    assert play_calls == []
    assert warnings


def test_offline_reference_authoring_preview_restores_authoring_waveform_and_sample_rate(qapp, monkeypatch):
    generated_by_rate = {}
    play_calls = []

    def fake_generate_chirps(self, **kwargs):
        sample_rate = int(kwargs["sample_rate"])
        data = np.full(int(sample_rate * kwargs["total_time"]), sample_rate, dtype=np.float32)
        generated_by_rate[sample_rate] = data.copy()
        return data, sample_rate

    monkeypatch.setattr("ui.stimulus_window.StimulusSignal.generate_chirps", fake_generate_chirps)
    monkeypatch.setattr(
        "ui.stimulus_window.SoundcardAudioProcessor.sd_play",
        lambda self, params: play_calls.append({**params, "data": np.asarray(params["data"]).copy()}) or (0, "ok"),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
        offline_reference_authoring=True,
    )
    original_info = window.stimulus_info.copy()
    original_config_info = window.stimulus_config_data["stimulus_info"].copy()
    original_data = window.stimulus_data.copy()
    original_data_sample_rate = window._stimulus_data_sample_rate
    original_display_text = window.sample_rate_lineedit.text()
    original_display_tooltip = window.sample_rate_lineedit.toolTip()

    window.play_btn_clicked()

    assert play_calls
    assert play_calls[-1]["sr"] == 48000
    assert np.array_equal(play_calls[-1]["data"], generated_by_rate[48000])
    assert window.stimulus_info == original_info
    assert window.stimulus_config_data["stimulus_info"] == original_config_info
    assert np.array_equal(window.stimulus_data, original_data)
    assert window._stimulus_data_sample_rate == original_data_sample_rate == 44100
    assert window.stimulus_info["sample_rate"] == 44100
    assert window.sample_rate_lineedit.text() == original_display_text
    assert window.sample_rate_lineedit.toolTip() == original_display_tooltip


def test_preview_regenerates_generated_waveform_after_selected_speaker_samplerate_change(qapp, monkeypatch):
    generated_sample_rates = []
    play_calls = []

    def fake_generate_chirps(self, **kwargs):
        sample_rate = int(kwargs["sample_rate"])
        generated_sample_rates.append(sample_rate)
        return np.ones(int(sample_rate * kwargs["total_time"]), dtype=np.float32), sample_rate

    monkeypatch.setattr("ui.stimulus_window.StimulusSignal.generate_chirps", fake_generate_chirps)
    monkeypatch.setattr(
        "ui.stimulus_window.SoundcardAudioProcessor.sd_play",
        lambda self, params: play_calls.append(params) or (0, "ok"),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    assert len(window.stimulus_data) == 480

    window.speaker = {"name": "speaker", "samplerate": 44100, "index": 8}
    window.play_btn_clicked()

    assert generated_sample_rates[-1] == 44100
    assert play_calls[-1]["sr"] == 44100
    assert play_calls[-1]["device"] == 8
    assert len(play_calls[-1]["data"]) == 441


def test_confirm_regenerates_generated_waveform_after_selected_speaker_samplerate_change(qapp, monkeypatch):
    saved_audio = []

    def fake_generate_chirps(self, **kwargs):
        sample_rate = int(kwargs["sample_rate"])
        return np.ones(int(sample_rate * kwargs["total_time"]), dtype=np.float32), sample_rate

    monkeypatch.setattr("ui.stimulus_window.StimulusSignal.generate_chirps", fake_generate_chirps)
    monkeypatch.setattr(
        "ui.stimulus_window.save_audio_simple",
        lambda path, data, sr: saved_audio.append((path, np.asarray(data).copy(), sr)),
    )
    monkeypatch.setattr(StimulusWindow, "set_ai_popup", lambda self: None)

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    assert len(window.stimulus_data) == 480

    window.speaker = {"name": "speaker", "samplerate": 44100, "index": 8}
    window.ok_btn_clicked()

    assert saved_audio
    assert saved_audio[-1][2] == 44100
    assert len(saved_audio[-1][1]) == 441
    assert window.final_save_data["stimulus_info"]["sample_rate"] == 44100


def test_save_wav_regenerates_generated_waveform_after_selected_speaker_samplerate_change(qapp, monkeypatch):
    saved_audio = []

    def fake_generate_chirps(self, **kwargs):
        sample_rate = int(kwargs["sample_rate"])
        return np.ones(int(sample_rate * kwargs["total_time"]), dtype=np.float32), sample_rate

    monkeypatch.setattr("ui.stimulus_window.StimulusSignal.generate_chirps", fake_generate_chirps)
    monkeypatch.setattr("ui.stimulus_window.QFileDialog.getSaveFileName", lambda *args, **kwargs: ("out.wav", ""))
    monkeypatch.setattr(
        "ui.stimulus_window.save_audio_simple",
        lambda path, data, sr: saved_audio.append((path, np.asarray(data).copy(), sr)),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    assert len(window.stimulus_data) == 480

    window.speaker = {"name": "speaker", "samplerate": 44100, "index": 8}
    window.save_wav_btn_clicked()

    assert len(saved_audio) == 1
    assert saved_audio[0][0] == "out.wav"
    assert saved_audio[0][2] == 44100
    assert len(saved_audio[0][1]) == 441


def test_stimulus_generation_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    calls = []
    warnings = []

    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignal.generate_chirps",
        lambda self, **kwargs: calls.append(kwargs) or (np.zeros(4), np.arange(4)),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()
    previous_data = window.stimulus_data
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100

    result = window.create_signal_from_stimulus_info()

    assert result is False
    assert calls == []
    assert window.stimulus_data is previous_data
    assert window.stimulus_info["sample_rate"] == 44100
    assert warnings


def test_stimulus_window_legacy_external_load_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    calls = []
    warnings = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    monkeypatch.setattr(
        "ui.stimulus_window.load_audio_simple",
        lambda path, sr: calls.append((path, sr)) or (np.zeros(4), np.arange(4)),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    previous_data = window.stimulus_data
    previous_path = window.load_wav_path
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100
    window.load_stimulus_signal_path = "external.wav"

    window.change_custom_chk_box(False)

    assert calls == []
    assert window.stimulus_data is previous_data
    assert window.load_wav_path == previous_path
    assert window.stimulus_info["sample_rate"] == 44100
    assert warnings


def test_stimulus_window_save_wav_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    saves = []
    warnings = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100
    monkeypatch.setattr("ui.stimulus_window.QFileDialog.getSaveFileName", lambda *args, **kwargs: ("out.wav", ""))
    monkeypatch.setattr(
        "ui.stimulus_window.save_audio_simple",
        lambda path, data, sr: saves.append((path, sr)),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.save_wav_btn_clicked()

    assert saves == []
    assert window.stimulus_info["sample_rate"] == 44100
    assert warnings


def test_frequency_stepped_generation_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    calls = []
    warnings = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window = StimulusWindow(
        stimulus_config_data=_frequency_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()
    previous_data = window.stimulus_data
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100

    result = window.create_signal_from_stimulus_info()

    assert result is False
    assert calls == []
    assert window.stimulus_data is previous_data
    assert window.stimulus_info["sample_rate"] == 44100
    assert warnings


def test_load_frequency_stepped_config_uses_speaker_samplerate_not_config_sample_rate(qapp, monkeypatch):
    calls = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()

    window.load_stimulus_config_data(_frequency_stimulus_config(sample_rate=44100))

    assert calls
    assert all(call["sample_rate"] == 48000 for call in calls)
    assert window.stimulus_info["sample_rate"] == 48000
    assert window.stimulus_info["schedule_sample_rate"] == 48000


def test_load_frequency_stepped_config_blocks_invalid_speaker_samplerate_without_config_fallback(
    qapp, monkeypatch
):
    calls = []
    warnings = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    window.load_stimulus_config_data(_frequency_stimulus_config(sample_rate=44100))

    assert calls == []
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert warnings


def test_load_live_non_frequency_config_blocks_invalid_speaker_samplerate_atomically(
    qapp, monkeypatch
):
    warnings = []
    graph_calls = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    previous_load_wav_path = window.load_wav_path
    previous_load_stimulus_signal_path = window.load_stimulus_signal_path
    previous_external_loaded = window._legacy_external_wav_loaded_by_user
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(window, "graph_stimulus", lambda: graph_calls.append(window.stimulus_info.copy()))

    window.load_stimulus_config_data(
        {
            **_stimulus_config(sample_rate=44100),
            "stimulus_signal_path": "new-live-config.wav",
            "load_stimulus_signal_path": "new-live-load.wav",
        }
    )

    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.load_wav_path == previous_load_wav_path
    assert window.load_stimulus_signal_path == previous_load_stimulus_signal_path
    assert window._legacy_external_wav_loaded_by_user == previous_external_loaded
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert graph_calls == []
    assert warnings


def test_load_live_unsupported_config_blocks_invalid_speaker_samplerate_atomically(qapp, monkeypatch):
    warnings = []
    graph_calls = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    window.load_wav_path = "previous-live.wav"
    window.load_stimulus_signal_path = "previous-load.wav"
    window._legacy_external_wav_loaded_by_user = True
    previous_load_wav_path = window.load_wav_path
    previous_load_stimulus_signal_path = window.load_stimulus_signal_path
    previous_external_loaded = window._legacy_external_wav_loaded_by_user
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(window, "graph_stimulus", lambda: graph_calls.append(window.stimulus_info.copy()))

    window.load_stimulus_config_data(
        {
            "stimulus_info": {
                **_stimulus_config(sample_rate=44100)["stimulus_info"],
                "stimulus_method": "unsupported_legacy_method",
                "sample_rate": 44100,
            },
            "stimulus_signal_path": "unsupported-live.wav",
            "load_stimulus_signal_path": "unsupported-load.wav",
        }
    )

    assert window.stimulus_info == previous_info
    assert window.stimulus_info["sample_rate"] == 48000
    assert window.stimulus_info["stimulus_type"] == "linear"
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.load_wav_path == previous_load_wav_path
    assert window.load_stimulus_signal_path == previous_load_stimulus_signal_path
    assert window._legacy_external_wav_loaded_by_user == previous_external_loaded
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert graph_calls == []
    assert warnings


def test_load_live_legacy_config_blocks_invalid_speaker_samplerate_atomically(qapp, monkeypatch):
    warnings = []
    graph_calls = []

    class FakeLoadStimulusDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return {
                **_stimulus_config(sample_rate=44100)["stimulus_info"],
                "start_freq": 250,
                "stop_freq": 2000,
                "total_time": 0.02,
            }

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    previous_load_wav_path = window.load_wav_path
    previous_load_stimulus_signal_path = window.load_stimulus_signal_path
    previous_external_loaded = window._legacy_external_wav_loaded_by_user
    previous_retained_state = window._step_sc_retained_frequency_state
    previous_retained_frequencies = window._step_sc_retained_frequencies
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    monkeypatch.setattr("ui.stimulus_window.LoadStimulusDialog", FakeLoadStimulusDialog)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(window, "graph_stimulus", lambda: graph_calls.append(window.stimulus_info.copy()))

    window.load_config_btn_clicked()

    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.load_wav_path == previous_load_wav_path
    assert window.load_stimulus_signal_path == previous_load_stimulus_signal_path
    assert window._legacy_external_wav_loaded_by_user == previous_external_loaded
    assert window._step_sc_retained_frequency_state == previous_retained_state
    assert window._step_sc_retained_frequencies == previous_retained_frequencies
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert graph_calls == []
    assert warnings


def test_default_frequency_stepped_config_uses_speaker_samplerate_not_config_sample_rate(qapp, monkeypatch):
    calls = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusWindow.load_stimulus_info_from_json",
        staticmethod(
            lambda default_config_flag=False: (
                0,
                {
                    **_frequency_stimulus_config(sample_rate=44100),
                    "stimulus_signal_path": "default-step-sc.wav",
                    "load_stimulus_signal_path": None,
                },
            )
        ),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()

    window.default_config_btn_clicked()

    assert calls
    assert all(call["sample_rate"] == 48000 for call in calls)
    assert window.stimulus_info["sample_rate"] == 48000
    assert window.stimulus_info["schedule_sample_rate"] == 48000


def test_default_frequency_stepped_config_blocks_invalid_speaker_samplerate_without_config_fallback(
    qapp, monkeypatch
):
    calls = []
    warnings = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusWindow.load_stimulus_info_from_json",
        staticmethod(
            lambda default_config_flag=False: (
                0,
                {
                    **_frequency_stimulus_config(sample_rate=44100),
                    "stimulus_signal_path": "default-step-sc.wav",
                    "load_stimulus_signal_path": None,
                },
            )
        ),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    calls.clear()
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    window.default_config_btn_clicked()

    assert calls == []
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert warnings


def test_default_live_legacy_config_blocks_invalid_speaker_samplerate_atomically(qapp, monkeypatch):
    warnings = []
    graph_calls = []

    monkeypatch.setattr(
        "ui.stimulus_window.StimulusWindow.load_stimulus_info_from_json",
        staticmethod(
            lambda default_config_flag=False: (
                0,
                {
                    **_stimulus_config(sample_rate=44100),
                    "stimulus_signal_path": "default-legacy.wav",
                    "load_stimulus_signal_path": None,
                },
            )
        ),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    previous_info = window.stimulus_info.copy()
    previous_data = window.stimulus_data.copy()
    previous_load_wav_path = window.load_wav_path
    previous_load_stimulus_signal_path = window.load_stimulus_signal_path
    previous_external_loaded = window._legacy_external_wav_loaded_by_user
    previous_retained_state = window._step_sc_retained_frequency_state
    previous_retained_frequencies = window._step_sc_retained_frequencies
    previous_sample_rate_text = window.sample_rate_lineedit.text()
    previous_sample_rate_tooltip = window.sample_rate_lineedit.toolTip()
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}

    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(window, "graph_stimulus", lambda: graph_calls.append(window.stimulus_info.copy()))

    window.default_config_btn_clicked()

    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window.load_wav_path == previous_load_wav_path
    assert window.load_stimulus_signal_path == previous_load_stimulus_signal_path
    assert window._legacy_external_wav_loaded_by_user == previous_external_loaded
    assert window._step_sc_retained_frequency_state == previous_retained_state
    assert window._step_sc_retained_frequencies == previous_retained_frequencies
    assert window.sample_rate_lineedit.text() == previous_sample_rate_text
    assert window.sample_rate_lineedit.toolTip() == previous_sample_rate_tooltip
    assert graph_calls == []
    assert warnings


def test_offline_load_frequency_stepped_config_uses_loaded_candidate_sample_rate_not_stale_window_rate(
    qapp, monkeypatch
):
    calls = []
    loaded_config = _frequency_stimulus_config(sample_rate=48000)["stimulus_info"]

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    class FakeLoadStimulusDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return loaded_config.copy()

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr("ui.stimulus_window.LoadStimulusDialog", FakeLoadStimulusDialog)

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker=None,
        offline_reference_authoring=True,
    )
    calls.clear()

    window.load_config_btn_clicked()

    assert calls
    assert all(call["sample_rate"] == 48000 for call in calls)
    assert window.stimulus_info["sample_rate"] == 48000
    assert window.stimulus_info["schedule_sample_rate"] == 48000


def test_offline_default_frequency_stepped_config_uses_default_candidate_sample_rate_not_stale_window_rate(
    qapp, monkeypatch
):
    calls = []

    def fake_generate_frequency_stepped(**kwargs):
        calls.append(kwargs)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": kwargs["sample_rate"],
            "schedule_sample_rate": kwargs["sample_rate"],
        }
        return SimpleNamespace(data=np.zeros(4), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusWindow.load_stimulus_info_from_json",
        staticmethod(
            lambda default_config_flag=False: (
                0,
                {
                    **_frequency_stimulus_config(sample_rate=48000),
                    "stimulus_signal_path": "default-step-sc.wav",
                    "load_stimulus_signal_path": None,
                },
            )
        ),
    )

    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker=None,
        offline_reference_authoring=True,
    )
    calls.clear()

    window.default_config_btn_clicked()

    assert calls
    assert all(call["sample_rate"] == 48000 for call in calls)
    assert window.stimulus_info["sample_rate"] == 48000
    assert window.stimulus_info["schedule_sample_rate"] == 48000


def test_save_config_uses_speaker_samplerate_before_db_save(qapp, monkeypatch):
    saved_infos = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    window.stimulus_info["sample_rate"] = 44100

    monkeypatch.setattr("ui.stimulus_window.SetConfigName", lambda: SimpleNamespace(exec=lambda: "saved_stimulus"))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignalManagement",
        lambda: SimpleNamespace(save_stimulus_info_to_db=lambda info: saved_infos.append(info.copy()) or (0, "ok")),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.information", lambda *args: None)

    window.save_config_btn_clicked()

    assert saved_infos
    assert saved_infos[0]["stimulus_name"] == "saved_stimulus"
    assert saved_infos[0]["sample_rate"] == 48000
    assert window.stimulus_info["sample_rate"] == 48000


def test_confirm_external_wav_syncs_total_time_from_reloaded_waveform(qapp, monkeypatch):
    loaded_waveform = np.ones(9600, dtype=np.float32)
    load_calls = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    _set_legacy_external_wav_branch(window)
    window.stimulus_info["total_time"] = 9.9
    window.stimulus_data = np.zeros(10, dtype=np.float32)
    window._stimulus_data_sample_rate = 44100

    monkeypatch.setattr(
        "ui.stimulus_window.load_audio_simple",
        lambda path, sr: load_calls.append((path, sr)) or (loaded_waveform, sr),
    )
    monkeypatch.setattr(StimulusWindow, "set_ai_popup", lambda self: None)

    window.ok_btn_clicked()

    assert load_calls == [("external.wav", 48000)]
    assert window.final_save_data is not None
    assert window.final_save_data["stimulus_info"]["sample_rate"] == 48000
    assert window.final_save_data["stimulus_info"]["total_time"] == pytest.approx(0.2)
    assert window.stimulus_info["total_time"] == pytest.approx(0.2)


def test_save_config_external_wav_syncs_total_time_before_db_save(qapp, monkeypatch):
    saved_infos = []
    loaded_waveform = np.ones(7200, dtype=np.float32)
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    _set_legacy_external_wav_branch(window)
    window.stimulus_info["total_time"] = 4.0
    window.stimulus_data = np.zeros(10, dtype=np.float32)
    window._stimulus_data_sample_rate = 44100

    monkeypatch.setattr(
        "ui.stimulus_window.load_audio_simple",
        lambda path, sr: (loaded_waveform, sr),
    )
    monkeypatch.setattr("ui.stimulus_window.SetConfigName", lambda: SimpleNamespace(exec=lambda: "saved_external"))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignalManagement",
        lambda: SimpleNamespace(save_stimulus_info_to_db=lambda info: saved_infos.append(info.copy()) or (0, "ok")),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.information", lambda *args: None)

    window.save_config_btn_clicked()

    assert saved_infos
    assert saved_infos[0]["stimulus_name"] == "saved_external"
    assert saved_infos[0]["sample_rate"] == 48000
    assert saved_infos[0]["total_time"] == pytest.approx(0.15)
    assert window.stimulus_info["total_time"] == pytest.approx(0.15)


def test_save_config_rebuilds_frequency_stepped_metadata_after_selected_speaker_samplerate_change(
    qapp, monkeypatch
):
    saved_infos = []
    generated_sample_rates = []

    def fake_generate_frequency_stepped(**kwargs):
        sample_rate = int(kwargs["sample_rate"])
        generated_sample_rates.append(sample_rate)
        metadata = {
            "frequency_mode": kwargs["frequency_mode"],
            "frequencies": kwargs["frequencies"] or [100, 250, 400],
            "sample_rate": sample_rate,
            "schedule_sample_rate": sample_rate,
            "segments": [{"sample_rate": sample_rate}],
        }
        return SimpleNamespace(data=np.ones(sample_rate // 100, dtype=np.float32), metadata=metadata)

    monkeypatch.setattr("ui.stimulus_window.generate_frequency_stepped", fake_generate_frequency_stepped)
    monkeypatch.setattr("ui.stimulus_window.SetConfigName", lambda: SimpleNamespace(exec=lambda: "saved_step_sc"))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignalManagement",
        lambda: SimpleNamespace(save_stimulus_info_to_db=lambda info: saved_infos.append(info.copy()) or (0, "ok")),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.information", lambda *args: None)

    window = StimulusWindow(
        stimulus_config_data=_frequency_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    assert generated_sample_rates[-1] == 48000
    assert window.stimulus_info["schedule_sample_rate"] == 48000

    window.speaker = {"name": "speaker", "samplerate": 44100, "index": 8}
    window.stimulus_info["sample_rate"] = 48000
    window.stimulus_info["schedule_sample_rate"] = 48000
    window.stimulus_info["segments"] = [{"sample_rate": 48000}]

    window.save_config_btn_clicked()

    assert saved_infos
    assert generated_sample_rates[-1] == 44100
    assert saved_infos[0]["stimulus_name"] == "saved_step_sc"
    assert saved_infos[0]["sample_rate"] == 44100
    assert saved_infos[0]["schedule_sample_rate"] == 44100
    assert saved_infos[0]["segments"] == [{"sample_rate": 44100}]
    assert len(window.stimulus_data) == 441


def test_save_config_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    saves = []
    warnings = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100

    monkeypatch.setattr("ui.stimulus_window.SetConfigName", lambda: SimpleNamespace(exec=lambda: "saved_stimulus"))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignalManagement",
        lambda: SimpleNamespace(save_stimulus_info_to_db=lambda info: saves.append(info.copy()) or (0, "ok")),
    )
    monkeypatch.setattr("ui.stimulus_window.MessageBox.information", lambda *args: None)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.save_config_btn_clicked()

    assert saves == []
    assert window.stimulus_info["sample_rate"] == 44100
    assert warnings


def test_confirm_blocks_without_valid_speaker_samplerate(qapp, monkeypatch):
    saves = []
    warnings = []
    window = StimulusWindow(
        stimulus_config_data=_stimulus_config(sample_rate=44100),
        speaker={"name": "speaker", "samplerate": 48000, "index": 7},
    )
    window.speaker = {"name": "speaker", "sample_rate": 44100, "index": 7}
    window.stimulus_info["sample_rate"] = 44100
    monkeypatch.setattr("ui.stimulus_window.save_audio_simple", lambda *args: saves.append(args))
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.ok_btn_clicked()

    assert saves == []
    assert window.final_save_data is None
    assert warnings


def test_offline_reference_authoring_save_and_confirm_do_not_require_speaker_samplerate(qapp, monkeypatch):
    db_saves = []
    audio_saves = []
    warnings = []
    config = _stimulus_config(sample_rate=44100)
    window = StimulusWindow(
        stimulus_config_data=config,
        speaker={"name": "speaker", "sample_rate": 48000, "index": 7},
        offline_reference_authoring=True,
    )

    monkeypatch.setattr("ui.stimulus_window.SetConfigName", lambda: SimpleNamespace(exec=lambda: "offline_ref"))
    monkeypatch.setattr(
        "ui.stimulus_window.StimulusSignalManagement",
        lambda: SimpleNamespace(save_stimulus_info_to_db=lambda info: db_saves.append(info.copy()) or (0, "ok")),
    )
    monkeypatch.setattr("ui.stimulus_window.save_audio_simple", lambda *args: audio_saves.append(args))
    monkeypatch.setattr("ui.stimulus_window.MessageBox.information", lambda *args: None)
    monkeypatch.setattr("ui.stimulus_window.MessageBox.warning", lambda *args: warnings.append(args))

    window.save_config_btn_clicked()
    window.ok_btn_clicked()

    assert db_saves
    assert db_saves[0]["stimulus_name"] == "offline_ref"
    assert db_saves[0]["sample_rate"] == 44100
    assert audio_saves
    assert window.final_save_data is not None
    assert window.final_save_data["stimulus_info"]["sample_rate"] == 44100
    assert not [args for args in warnings if args[1] == "采样率配置"]
