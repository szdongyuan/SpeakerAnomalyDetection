import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import stimulus_resolver
from ui.sequence import sequence_widget


@pytest.fixture(autouse=True)
def _valid_wav_calibration_metadata(monkeypatch):
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: metadata)


class _Button:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = enabled


def _chirp_detail(config_rate=44100):
    return {
        "sample_rate": config_rate,
        "stimulus_info": {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "sample_rate": config_rate,
            "start_freq": 100,
            "stop_freq": 1000,
            "total_time": 0.01,
            "repeat_times": 1,
            "amplitude": 0.5,
            "voltage": 1.0,
            "voltage_type": "RMS",
        },
    }


def _window(mode, detail=None):
    if detail is None:
        detail = {
            "sample_rate": 44100,
            "stimulus_info": {
                "stimulus_method": "chirp",
                "sample_rate": 44100,
                "total_time": 0.01,
            },
        }
    return SimpleNamespace(
        sequence_config=[
            {
                "seq1": {
                    "acq": {
                        "mode": mode,
                        "detail": detail,
                    }
                }
            }
        ],
        data_struct=SimpleNamespace(sample_rate=None),
        analysis_config={"auto_analysis": False},
        recorded_path=None,
        recorded_signal_info=None,
        _clear_plot_area=lambda: None,
        plot_waveform_to_workspace=lambda *args, **kwargs: None,
        data_btn=_Button(),
        run=lambda: None,
        using_config_path="configs/sequence.json",
    )


def _native_audio_loader(calls, decoded_rate=32000):
    def fake_load_audio_preserve_rate(path, mono=True):
        calls.append((path, mono))
        return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32), decoded_rate

    return fake_load_audio_preserve_rate


def test_import_stimulus_audio_uses_decoded_rate_for_analysis_reference(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO")
    load_calls = []
    reference_calls = []

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("recording.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader(load_calls), raising=False)

    def fake_reference(data_struct, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        reference_calls.append((detail, using_config_path, runtime_sample_rate))
        data_struct.stimulus_data = np.zeros(3, dtype=np.float32)
        data_struct.stimulus_info = {"sample_rate": runtime_sample_rate}
        return True

    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        fake_reference,
        raising=False,
    )

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert load_calls == [("recording.wav", False)]
    assert win.data_struct.sample_rate == 32000
    assert win.data_struct.store_wave_data_multi.shape == (3, 2)
    assert reference_calls == [(win.sequence_config[0]["seq1"]["acq"]["detail"], "configs/sequence.json", 32000)]
    assert win.sequence_config[0]["seq1"]["acq"]["detail"]["sample_rate"] == 44100
    assert win.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"]["sample_rate"] == 44100


def test_import_stimulus_audio_generated_config_regenerates_reference_at_decoded_rate(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_signal_path"] = "stored/generated-artifact.wav"
    before = {k: v.copy() if isinstance(v, dict) else v for k, v in detail.items()}
    win = _window("IMPORT_STIMULUS_AUDIO", detail=detail)
    warnings = []
    original_exists = stimulus_resolver.os.path.exists

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("recording.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader([], 32000), raising=False)
    monkeypatch.setattr(
        stimulus_resolver.os.path,
        "exists",
        lambda path: False if str(path).replace("\\", "/").endswith("stored/generated-artifact.wav") else original_exists(path),
    )
    monkeypatch.setattr(
        stimulus_resolver,
        "load_audio_simple",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generated artifact must not be loaded")),
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert warnings == []
    assert win.data_struct.sample_rate == 32000
    assert win.data_struct.stimulus_info["sample_rate"] == 32000
    assert len(win.data_struct.stimulus_data) == int(0.01 * 32000)
    assert win.sequence_config[0]["seq1"]["acq"]["detail"] == before


def test_import_stimulus_audio_external_reference_length_check_uses_loaded_metadata(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_info"]["total_time"] = 0.5
    detail["load_stimulus_signal_path"] = "external-reference.wav"
    before = {k: v.copy() if isinstance(v, dict) else v for k, v in detail.items()}
    win = _window("IMPORT_STIMULUS_AUDIO", detail=detail)
    run_completed = []
    warnings = []

    class _Size:
        def width(self):
            return 1000

        def height(self):
            return 800

    class _Screen:
        def size(self):
            return _Size()

    win.analysis_config = {"auto_analysis": True, "display_sequence": []}
    win.analysis_window = []
    win._analysis_result_summary_window = None
    win.mode = "IMPORT_STIMULUS_AUDIO"
    win.count_board = SimpleNamespace(mode="normal")
    win.data_struct.analysis_result_dict = {}
    win.screen = lambda: _Screen()
    win._handle_post_analysis_exports = lambda *args, **kwargs: None
    win._maybe_show_analysis_result_summary = lambda *args, **kwargs: run_completed.append("summary")
    win._send_tcp_analysis_result_callback = lambda *args, **kwargs: None
    win.run = lambda: sequence_widget.SequenceWindow.run(win)

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("recording.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader([], 32000), raising=False)
    monkeypatch.setattr(stimulus_resolver.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        stimulus_resolver,
        "load_audio_simple",
        lambda path, sr: (np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32)),
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert warnings == []
    assert run_completed == ["summary"]
    assert win.data_struct.audio_lenth == 3
    assert win.data_struct.stimulus_info["sample_rate"] == 32000
    assert win.data_struct.stimulus_info["total_time"] == 3 / 32000
    assert win.sequence_config[0]["seq1"]["acq"]["detail"] == before


def test_import_stimulus_audio_reference_false_clears_state_and_disables_analysis(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO")
    win.data_struct.sample_rate = 44100
    win.data_struct.store_wave_data_multi = np.array([[1.0, 2.0]], dtype=np.float32)
    win.data_struct.store_wave_data = np.array([1.5], dtype=np.float32)
    win.data_struct.audio_lenth = 1
    win.data_struct.stimulus_data = np.array([1.0], dtype=np.float32)
    win.data_struct.stimulus_info = {"sample_rate": 44100}
    win.data_struct.alignment_sample_count = 1
    win.recorded_path = "old.wav"
    win.recorded_signal_info = {"file_path": "old.wav"}
    win.data_btn.setEnabled(True)
    warnings = []

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("new.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader([], 32000), raising=False)

    def fake_reference(data_struct, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        data_struct.stimulus_data = np.zeros(3, dtype=np.float32)
        data_struct.stimulus_info = {"sample_rate": runtime_sample_rate}
        data_struct.alignment_sample_count = 3
        return False

    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        fake_reference,
        raising=False,
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert warnings
    assert win.recorded_path is None
    assert win.recorded_signal_info is None
    assert win.data_struct.sample_rate is None
    assert win.data_struct.store_wave_data_multi is None
    assert win.data_struct.store_wave_data is None
    assert win.data_struct.audio_lenth is None
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert win.data_btn.enabled is False


def test_import_audio_preserves_decoded_rate_without_reference_generation(monkeypatch):
    win = _window("IMPORT_AUDIO")
    stale_stimulus = np.array([9.0, 8.0], dtype=np.float32)
    stale_info = {"sample_rate": 44100, "stimulus_method": "stale"}
    win.data_struct.stimulus_data = stale_stimulus
    win.data_struct.stimulus_info = stale_info
    win.data_struct.alignment_sample_count = 99
    load_calls = []
    reference_calls = []

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("recording.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader(load_calls, 22050), raising=False)
    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        lambda *args, **kwargs: reference_calls.append((args, kwargs)),
        raising=False,
    )

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert load_calls == [("recording.wav", False)]
    assert win.data_struct.sample_rate == 22050
    assert win.data_struct.audio_lenth == 3
    assert win.data_struct.store_wave_data_multi.shape == (3, 2)
    assert np.allclose(win.data_struct.store_wave_data, np.array([0.25, 0.35, 0.45], dtype=np.float32))
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert reference_calls == []


def test_import_stimulus_audio_reference_failure_does_not_commit_new_recording(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO")
    old_multi = np.array([[9.0, 8.0]], dtype=np.float32)
    old_mono = np.array([8.5], dtype=np.float32)
    old_stimulus = np.array([1.0, 2.0], dtype=np.float32)
    old_info = {"sample_rate": 44100, "stimulus_method": "chirp"}
    win.data_struct.sample_rate = 44100
    win.data_struct.store_wave_data_multi = old_multi
    win.data_struct.store_wave_data = old_mono
    win.data_struct.audio_lenth = 1
    win.data_struct.stimulus_data = old_stimulus
    win.data_struct.stimulus_info = old_info
    win.data_struct.alignment_sample_count = 2
    win.recorded_path = "old.wav"
    win.recorded_signal_info = {"file_path": "old.wav"}
    win.data_btn.setEnabled(True)

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("new.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", _native_audio_loader([], 32000), raising=False)
    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reference failed")),
        raising=False,
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: None)

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert win.recorded_path is None
    assert win.recorded_signal_info is None
    assert win.data_struct.sample_rate is None
    assert win.data_struct.store_wave_data_multi is None
    assert win.data_struct.store_wave_data is None
    assert win.data_struct.audio_lenth is None
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert win.data_btn.enabled is False


def test_import_audio_decode_exception_clears_stale_state_and_disables_analysis(monkeypatch):
    win = _window("IMPORT_AUDIO")
    win.data_struct.sample_rate = 44100
    win.data_struct.store_wave_data_multi = np.array([[1.0, 2.0]], dtype=np.float32)
    win.data_struct.store_wave_data = np.array([1.5], dtype=np.float32)
    win.data_struct.audio_lenth = 1
    win.data_struct.stimulus_data = np.array([1.0], dtype=np.float32)
    win.data_struct.stimulus_info = {"sample_rate": 44100}
    win.data_struct.alignment_sample_count = 1
    win.recorded_path = "old.wav"
    win.recorded_signal_info = {"file_path": "old.wav"}
    win.data_btn.setEnabled(True)
    warnings = []

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("broken.wav", ""))
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
        raising=False,
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert warnings
    assert win.recorded_path is None
    assert win.recorded_signal_info is None
    assert win.data_struct.sample_rate is None
    assert win.data_struct.store_wave_data_multi is None
    assert win.data_struct.store_wave_data is None
    assert win.data_struct.audio_lenth is None
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert win.data_btn.enabled is False


def test_import_stimulus_audio_none_load_clears_stale_state_and_disables_analysis(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO")
    win.data_struct.sample_rate = 44100
    win.data_struct.store_wave_data_multi = np.array([[1.0, 2.0]], dtype=np.float32)
    win.data_struct.store_wave_data = np.array([1.5], dtype=np.float32)
    win.data_struct.audio_lenth = 1
    win.data_struct.stimulus_data = np.array([1.0], dtype=np.float32)
    win.data_struct.stimulus_info = {"sample_rate": 44100}
    win.data_struct.alignment_sample_count = 1
    win.recorded_path = "old.wav"
    win.recorded_signal_info = {"file_path": "old.wav"}
    win.data_btn.setEnabled(True)
    warnings = []

    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("missing.wav", ""))
    monkeypatch.setattr(sequence_widget, "load_audio_preserve_rate", lambda *args, **kwargs: (None, None), raising=False)
    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference setup must not run")),
        raising=False,
    )
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert warnings
    assert win.recorded_path is None
    assert win.recorded_signal_info is None
    assert win.data_struct.sample_rate is None
    assert win.data_struct.store_wave_data_multi is None
    assert win.data_struct.store_wave_data is None
    assert win.data_struct.audio_lenth is None
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert win.data_btn.enabled is False
