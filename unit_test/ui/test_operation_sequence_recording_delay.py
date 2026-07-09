import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui import operation_sequence


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_record_golden_sample_converts_configured_recording_start_delay(qapp, monkeypatch):
    captured = {}

    class FakeSoundcardAudioProcessor:
        def sd_play_rec(self, recorded_dict, stimulus_dict, path, calibration_metadata=None):
            captured["recorded_dict"] = dict(recorded_dict)
            captured["stimulus_dict"] = dict(stimulus_dict)
            captured["path"] = path
            return 0, np.array([0.1, 0.2], dtype=np.float32)

    data_struct = SimpleNamespace(sample_rate=48000, store_wave_data=None)
    seq = SimpleNamespace(
        detail={"recording_start_delay_ms": 250.0},
        analysis_list={
            "display_sequence": ["fr"],
            "fr": {"type": "FR", "golden_sample_checked": True},
        },
    )
    dialog = operation_sequence.AnalysisModelSelect.__new__(operation_sequence.AnalysisModelSelect)
    dialog.select_list = SimpleNamespace(config=[seq], data_struct=data_struct)
    dialog.using_config_path = "unused.json"
    dialog.default_logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    dialog.set_data_struct_stimulus_signal = lambda *args, **kwargs: None
    dialog.mic = {"samplerate": 48000}
    dialog.speaker = {"samplerate": 48000}

    def fake_get_dict(data_struct, recording_start_delay_ms=None, total_time=None):
        recorded = {"sr": data_struct.sample_rate, "num_frames": 2}
        if recording_start_delay_ms is not None:
            recorded["recording_start_delay_frames"] = int(
                round(recording_start_delay_ms * data_struct.sample_rate / 1000.0)
            )
        return {
            "data": np.array([0.1, 0.2], dtype=np.float32),
            "amplitude": 1.0,
            "sr": data_struct.sample_rate,
        }, recorded

    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        fake_get_dict,
    )
    monkeypatch.setattr(operation_sequence, "SoundcardAudioProcessor", FakeSoundcardAudioProcessor)
    monkeypatch.setattr(operation_sequence, "get_class_mapping", lambda: {})
    monkeypatch.setattr(operation_sequence.QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))

    dialog.record_golden_sample_btn_clicked()

    assert captured["recorded_dict"]["recording_start_delay_frames"] == 12000
    assert "recording_start_delay_ms" not in captured["recorded_dict"]


def test_golden_sample_runtime_rate_requires_selected_duplex_devices():
    dialog = SimpleNamespace(
        mic=None,
        speaker=None,
        select_list=SimpleNamespace(mic=None, speaker=None),
    )
    data_struct = SimpleNamespace(sample_rate=48000)

    result = operation_sequence._resolve_golden_sample_runtime_sample_rate(dialog, data_struct)

    assert result.ok is False
    assert result.sample_rate is None
    assert "输入设备" in result.message


def test_record_golden_sample_stimulus_setup_exception_restores_runtime_stimulus_state(qapp, monkeypatch):
    seeded_stimulus = np.array([1.0], dtype=np.float32)
    seeded_info = {"sample_rate": 44100}
    data_struct = SimpleNamespace(
        sample_rate=44100,
        stimulus_data=seeded_stimulus,
        stimulus_info=seeded_info,
        alignment_sample_count=1,
        store_wave_data=None,
    )
    seq = SimpleNamespace(
        detail={"recording_start_delay_ms": 0.0},
        analysis_list={
            "display_sequence": ["fr"],
            "fr": {"type": "FR", "golden_sample_checked": True},
        },
    )
    dialog = operation_sequence.AnalysisModelSelect.__new__(operation_sequence.AnalysisModelSelect)
    dialog.select_list = SimpleNamespace(config=[seq], data_struct=data_struct)
    dialog.using_config_path = "unused.json"
    dialog.default_logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    dialog.mic = {"samplerate": 48000}
    dialog.speaker = {"samplerate": 48000}
    dialog.set_data_struct_stimulus_signal = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("setup failed")
    )
    warnings = []

    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stale stimulus must not be reused")),
    )

    dialog.record_golden_sample_btn_clicked()

    assert warnings
    assert data_struct.sample_rate == 44100
    assert data_struct.stimulus_data is seeded_stimulus
    assert data_struct.stimulus_info is seeded_info
    assert data_struct.alignment_sample_count == 1


def test_record_golden_sample_runtime_rate_failure_preserves_runtime_stimulus_state(qapp, monkeypatch):
    seeded_stimulus = np.array([1.0], dtype=np.float32)
    seeded_info = {"sample_rate": 44100}
    data_struct = SimpleNamespace(
        sample_rate=44100,
        stimulus_data=seeded_stimulus,
        stimulus_info=seeded_info,
        alignment_sample_count=1,
        store_wave_data=None,
    )
    seq = SimpleNamespace(
        detail={"recording_start_delay_ms": 0.0},
        analysis_list={
            "display_sequence": ["fr"],
            "fr": {"type": "FR", "golden_sample_checked": True},
        },
    )
    dialog = operation_sequence.AnalysisModelSelect.__new__(operation_sequence.AnalysisModelSelect)
    dialog.select_list = SimpleNamespace(config=[seq], data_struct=data_struct, mic=None, speaker=None)
    dialog.using_config_path = "unused.json"
    dialog.default_logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    dialog.mic = None
    dialog.speaker = None
    dialog.set_data_struct_stimulus_signal = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("stimulus setup must not run without a resolved sample rate")
    )
    warnings = []

    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("play/record dict setup must not run")),
    )

    dialog.record_golden_sample_btn_clicked()

    assert warnings
    assert data_struct.sample_rate == 44100
    assert data_struct.stimulus_data is seeded_stimulus
    assert data_struct.stimulus_info is seeded_info
    assert data_struct.alignment_sample_count == 1


def test_record_golden_sample_dict_generation_failure_restores_runtime_stimulus_state(qapp, monkeypatch):
    seeded_stimulus = np.array([1.0], dtype=np.float32)
    seeded_info = {"sample_rate": 44100}
    data_struct = SimpleNamespace(
        sample_rate=44100,
        stimulus_data=seeded_stimulus,
        stimulus_info=seeded_info,
        alignment_sample_count=1,
        store_wave_data=None,
    )
    seq = SimpleNamespace(
        detail={"recording_start_delay_ms": 0.0},
        analysis_list={
            "display_sequence": ["fr"],
            "fr": {"type": "FR", "golden_sample_checked": True},
        },
    )
    dialog = operation_sequence.AnalysisModelSelect.__new__(operation_sequence.AnalysisModelSelect)
    dialog.select_list = SimpleNamespace(config=[seq], data_struct=data_struct)
    dialog.using_config_path = "unused.json"
    dialog.default_logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    dialog.mic = {"samplerate": 48000}
    dialog.speaker = {"samplerate": 48000}
    dialog.set_data_struct_stimulus_signal = lambda *args, **kwargs: None
    warnings = []

    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dict generation failed")),
    )

    dialog.record_golden_sample_btn_clicked()

    assert warnings
    assert data_struct.sample_rate == 44100
    assert data_struct.stimulus_data is seeded_stimulus
    assert data_struct.stimulus_info is seeded_info
    assert data_struct.alignment_sample_count == 1
