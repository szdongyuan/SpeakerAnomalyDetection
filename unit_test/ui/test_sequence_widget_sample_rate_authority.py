import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import play_and_record
from ui.sequence import sequence_widget


_DEFAULT_DEVICE = object()


def _window(mode, detail, mic=_DEFAULT_DEVICE, speaker=_DEFAULT_DEVICE):
    obj = SimpleNamespace()
    obj.sequence_config = [{"seq1": {"acq": {"mode": mode, "detail": detail}}}]
    obj.mic = {"samplerate": 48000, "name": "mic", "hostapi": 1, "index": 1} if mic is _DEFAULT_DEVICE else mic
    obj.speaker = (
        {"samplerate": 48000, "name": "speaker", "hostapi": 1, "index": 2}
        if speaker is _DEFAULT_DEVICE
        else speaker
    )
    obj.mic_channels = [0]
    obj.data_struct = SimpleNamespace(
        sample_rate=None,
        stimulus_data=np.zeros(10),
        stimulus_info={"total_time": 0.01, "sample_rate": 44100},
        clear_data=lambda: None,
    )
    obj._excel_export_cache = None
    obj.lineedit_type = SimpleNamespace(text=lambda: "model")
    obj.lineedit_count = SimpleNamespace(text=lambda: "1")
    obj.lineedit_s_or_n = SimpleNamespace(text=lambda: "")
    obj._active_input_channels = []
    obj._build_current_wav_calibration_metadata = lambda: {"recorded_channels": []}
    obj.using_config_path = "sequence.json"
    return obj


def test_record_only_uses_mic_samplerate_not_detail_sample_rate(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0},
        mic={"samplerate": 48000, "bit_depth": 64, "name": "mic", "hostapi": 1, "index": 1},
    )
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))
    monkeypatch.setattr(
        sequence_widget.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda ds, total_time, recording_start_delay_ms=None: ({}, {"sample_rate": ds.sample_rate}),
    )

    _, recorded_dict, sample_rate = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert sample_rate == 48000
    assert win.data_struct.sample_rate == 48000
    assert recorded_dict["sample_rate"] == 48000
    assert recorded_dict["bit_depth"] == 64


def test_play_and_record_blocks_mismatched_samplerates(monkeypatch):
    win = _window(
        "PLAY_AND_RECORD",
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        mic={"samplerate": 44100, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
    )
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))

    result = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert result == (None, None, None)
    assert warnings


def test_play_and_record_uses_device_samplerate_not_stimulus_config_rate(monkeypatch):
    win = _window(
        "PLAY_AND_RECORD",
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker", "index": 2},
    )
    win.data_struct.sample_rate = 48000
    win.data_struct.stimulus_info = {"sample_rate": 48000, "total_time": 0.01}
    win.data_struct.stimulus_data = np.zeros(480, dtype=np.float32)
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))
    monkeypatch.setattr(
        sequence_widget.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda ds, total_time, recording_start_delay_ms=None: (
            {"sr": ds.sample_rate, "data": ds.stimulus_data},
            {"sample_rate": ds.sample_rate, "sr": ds.sample_rate},
        ),
    )

    stimulus_dict, recorded_dict, sample_rate = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert sample_rate == 48000
    assert stimulus_dict["sr"] == 48000
    assert recorded_dict["sample_rate"] == 48000


def test_record_only_monitor_blocks_mismatched_samplerates(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": True},
        mic={"samplerate": 44100, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
    )
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))

    result = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert result == (None, None, None)
    assert warnings


def test_record_only_without_monitor_does_not_require_speaker_samplerate(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": False},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker={"hostapi": 1, "name": "speaker"},
    )
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))
    monkeypatch.setattr(
        sequence_widget.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda ds, total_time, recording_start_delay_ms=None: ({}, {"sample_rate": ds.sample_rate}),
    )

    _, recorded_dict, sample_rate = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert warnings == []
    assert sample_rate == 48000
    assert recorded_dict["output_device"] is None


def test_preflight_record_only_without_monitor_does_not_require_speaker(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": False},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    assert sequence_widget.SequenceWindow.checked_work_status_message(win) is False
    assert warnings == []


def test_preflight_record_only_string_false_monitor_does_not_require_speaker(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": "False"},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    assert sequence_widget.SequenceWindow.checked_work_status_message(win) is False
    assert warnings == []


def test_preflight_record_only_monitor_requires_speaker(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": True},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    assert sequence_widget.SequenceWindow.checked_work_status_message(win) is True
    assert warnings
    assert "输出设备" in str(warnings[0])


def test_preflight_play_and_record_requires_speaker(monkeypatch):
    win = _window(
        "PLAY_AND_RECORD",
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    assert sequence_widget.SequenceWindow.checked_work_status_message(win) is True
    assert warnings
    assert "输出设备" in str(warnings[0])


def test_record_only_invalid_mic_samplerate_warns_without_exception(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0},
        mic={"samplerate": 96000, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
    )
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {"labels": "not_labeled"}))

    result = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert result == (None, None, None)
    assert warnings
    assert "采样率" in str(warnings[0])


def test_start_streaming_recording_passes_real_reset_rate_to_record_writer_and_processor(monkeypatch):
    win = _window(
        "RECORD_ONLY",
        {"use_streaming_recording": True, "sample_rate": 44100, "total_time": 1.0},
        mic={"samplerate": 48000, "name": "mic", "hostapi": 1, "index": 1},
        speaker={"samplerate": 44100, "name": "speaker", "hostapi": 1, "index": 2},
    )
    win.data_struct.stimulus_data = np.array([], dtype=np.float32)
    win.mic_channels = [0, 1]
    win.recorded_path = "record.wav"
    win.recorded_signal_info = {}
    win.streaming_poll_timer = SimpleNamespace(start=lambda interval: None)
    writer_calls = []
    stream_calls = []
    processor_calls = []
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("record.wav", {}))
    monkeypatch.setattr(
        sequence_widget,
        "StreamingWavWriter",
        lambda path, sample_rate, channels=1: writer_calls.append((path, sample_rate, channels)) or object(),
    )

    class FakeProcessor:
        def start_streaming_rec(self, **kwargs):
            processor_calls.append(kwargs)
            return play_and_record.error_code.OK, "ok"

    monkeypatch.setattr(play_and_record, "StreamingAudioProcessor", FakeProcessor)
    play_and_record.data_struct.sample_rate = 32000

    stimulus_dict, recorded_dict, sample_rate = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")
    stream_calls.append((recorded_dict.get("sr"), recorded_dict.get("sample_rate"), sample_rate))
    sequence_widget.SequenceWindow._start_streaming_recording(win, stimulus_dict, recorded_dict, sample_rate)

    assert writer_calls == [("record.wav", 48000, 2)]
    assert stream_calls == [(48000, 48000, 48000)]
    assert processor_calls[0]["sample_rate"] == 48000


def test_start_streaming_recording_passes_resolved_rate_to_playrec_writer_and_processor(monkeypatch):
    win = _window("PLAY_AND_RECORD", {"use_streaming_recording": True})
    win.recorded_path = "record.wav"
    win.recorded_signal_info = {}
    win._active_input_channels = [0]
    win.streaming_poll_timer = SimpleNamespace(start=lambda interval: None)
    writer_calls = []
    stream_calls = []
    monkeypatch.setattr(
        sequence_widget,
        "StreamingWavWriter",
        lambda path, sample_rate, channels=1: writer_calls.append((path, sample_rate, channels)) or object(),
    )
    monkeypatch.setattr(
        sequence_widget,
        "stream_play_and_record",
        lambda stimulus_dict, recorded_dict, recorded_path, info: stream_calls.append(
            (stimulus_dict["sr"], recorded_path)
        )
        or (object(), stimulus_dict["data"], stimulus_dict["sr"]),
    )
    stimulus_dict = {"sr": 48000, "data": np.zeros(4)}

    sequence_widget.SequenceWindow._start_streaming_recording(win, stimulus_dict, {"sample_rate": 48000}, 48000)

    assert writer_calls == [("record_temp.wav", 48000, 1)]
    assert stream_calls == [(48000, "record.wav")]


def test_import_audio_preserves_file_native_samplerate(monkeypatch):
    win = _window("IMPORT_AUDIO", {"sample_rate": 44100})
    win.recorded_path = None
    win.recorded_signal_info = {}
    win.plot_waveform_to_workspace = lambda *args: None
    win._clear_plot_area = lambda: None
    win.data_btn = SimpleNamespace(setEnabled=lambda enabled: None)
    win.analysis_config = {"auto_analysis": False}
    load_calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("input.wav", ""))
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda path, mono=False: load_calls.append((path, mono))
        or (np.zeros((2, 320), dtype=np.float32), 32000),
    )

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert win.data_struct.sample_rate == 32000
    assert load_calls[0] == ("input.wav", False)


def test_import_stimulus_audio_generates_temporary_reference_at_recording_rate(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO", {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}})
    win.recorded_path = None
    win.recorded_signal_info = {}
    win.plot_waveform_to_workspace = lambda *args: None
    win._clear_plot_area = lambda: None
    win.data_btn = SimpleNamespace(setEnabled=lambda enabled: None)
    win.analysis_config = {"auto_analysis": False}
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("input.wav", ""))
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.zeros((2, 320), dtype=np.float32), 32000),
    )
    monkeypatch.setattr(
        sequence_widget,
        "set_data_struct_analysis_reference_signal",
        lambda data_struct, detail, using_config_path=None, *, runtime_sample_rate, **kwargs: calls.append(
            (detail, runtime_sample_rate)
        )
        or True,
    )

    sequence_widget.SequenceWindow.import_audio_and_analyze(win)

    assert win.data_struct.sample_rate == 32000
    assert calls == [(win.sequence_config[0]["seq1"]["acq"]["detail"], 32000)]
