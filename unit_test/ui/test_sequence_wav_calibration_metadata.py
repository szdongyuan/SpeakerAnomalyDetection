import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.data_struct.data_deal_struct import DataDealStruct
from base.hardware_management import HardwareManagementRepository
from base import recording_management
from base import soundcard_audio_processor as soundcard
from consts import error_code
from ui.sequence import sequence_widget


@pytest.fixture(autouse=True)
def reset_data_deal_struct_singleton():
    DataDealStruct._instance = None
    yield
    DataDealStruct._instance = None


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


class FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, value):
        self.enabled = value

    def setDisabled(self, value):
        self.enabled = not value


def _metadata(factor=2.5):
    return {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": factor, "standard_spl": 94.0, "calibrated": True}
        ]
    }


def _record_window(mode, metadata=None):
    data_struct = SimpleNamespace(
        store_wave_data=None,
        store_wave_data_multi=None,
        stimulus_info={"repeat_times": 1},
        sample_rate=48000,
    )
    window = SimpleNamespace(
        sequence_config=[{"seq1": {"acq": {"mode": mode}}}],
        recorded_path="record.wav",
        recorded_signal_info={},
        data_struct=data_struct,
        _active_input_channels=[0],
        _build_current_wav_calibration_metadata=lambda: metadata or _metadata(),
        plot_waveform_to_workspace=lambda *args: None,
        _finish_recording_success=lambda sample_rate: None,
        _finish_recording_failure=lambda error: (_ for _ in ()).throw(error),
        _recording_output_path=lambda: "record.wav",
        _finalize_successful_replay_output=lambda: None,
        _delete_successful_replay_backup=lambda: None,
        _clear_recording_output_attempt=lambda: None,
        _run_post_recording_followup=lambda action_name, callback, *args, **kwargs: callback(*args, **kwargs),
        default_logger=FakeLogger(),
    )
    window._normalize_blocking_recorded_data = lambda recorded_data: sequence_widget.SequenceWindow._normalize_blocking_recorded_data(
        window, recorded_data
    )
    return window


def _import_window(mode="IMPORT_AUDIO"):
    return SimpleNamespace(
        sequence_config=[{"seq1": {"acq": {"mode": mode, "detail": {"sample_rate": 44100}}}}],
        data_struct=SimpleNamespace(sample_rate=None),
        analysis_config={"auto_analysis": False},
        recorded_path=None,
        recorded_signal_info=None,
        _clear_plot_area=lambda: None,
        plot_waveform_to_workspace=lambda *args, **kwargs: None,
        data_btn=SimpleNamespace(setEnabled=lambda enabled: None),
        run=lambda: None,
        using_config_path="configs/sequence.json",
        default_logger=FakeLogger(),
    )


def _bind_current_metadata_input_channels(window):
    window._current_metadata_input_channels = sequence_widget.SequenceWindow._current_metadata_input_channels.__get__(
        window, type(window)
    )
    return window


def test_build_current_wav_calibration_metadata_uses_selected_mic_and_active_channels(monkeypatch):
    calls = []
    expected = _metadata()

    monkeypatch.setattr(
        sequence_widget,
        "build_recording_wav_calibration_metadata",
        lambda input_channels, hardware_id=None, logger=None: calls.append((list(input_channels), hardware_id, logger))
        or expected,
    )

    window = _bind_current_metadata_input_channels(
        SimpleNamespace(mic={"hardware_id": "mic-1"}, _active_input_channels=[2, 4], default_logger=FakeLogger())
    )

    assert sequence_widget.SequenceWindow._build_current_wav_calibration_metadata(window) is expected
    assert calls == [([2, 4], "mic-1", window.default_logger)]


def test_play_and_record_wav_metadata_uses_only_saved_mono_input_channel(monkeypatch):
    calls = []
    expected = _metadata()

    monkeypatch.setattr(
        sequence_widget,
        "build_recording_wav_calibration_metadata",
        lambda input_channels, hardware_id=None, logger=None: calls.append((list(input_channels), hardware_id))
        or expected,
    )

    window = _bind_current_metadata_input_channels(
        SimpleNamespace(
            mic={"hardware_id": "mic-1"},
            _active_input_channels=[2, 4],
            sequence_config=[{"seq1": {"acq": {"mode": "PLAY_AND_RECORD"}}}],
            default_logger=FakeLogger(),
        )
    )

    assert sequence_widget.SequenceWindow._build_current_wav_calibration_metadata(window) is expected
    assert calls == [([2], "mic-1")]


def test_record_only_wav_metadata_keeps_all_saved_input_channels(monkeypatch):
    calls = []
    expected = _metadata()

    monkeypatch.setattr(
        sequence_widget,
        "build_recording_wav_calibration_metadata",
        lambda input_channels, hardware_id=None, logger=None: calls.append((list(input_channels), hardware_id))
        or expected,
    )

    window = _bind_current_metadata_input_channels(
        SimpleNamespace(
            mic={"hardware_id": "mic-1"},
            _active_input_channels=[2, 4],
            sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
            default_logger=FakeLogger(),
        )
    )

    assert sequence_widget.SequenceWindow._build_current_wav_calibration_metadata(window) is expected
    assert calls == [([2, 4], "mic-1")]


def test_data_struct_initializes_and_clears_wav_calibration_runtime_fields():
    data_struct = DataDealStruct()
    assert data_struct.wav_calibration_metadata is None
    assert data_struct.wav_calibration_metadata_authoritative is False
    assert data_struct.wav_calibration_warning_shown is False

    data_struct.wav_calibration_metadata = _metadata()
    data_struct.wav_calibration_metadata_authoritative = True
    data_struct.wav_calibration_warning_shown = True

    data_struct.clear_data()

    assert data_struct.wav_calibration_metadata is None
    assert data_struct.wav_calibration_metadata_authoritative is False
    assert data_struct.wav_calibration_warning_shown is False


def test_import_audio_reads_wav_calibration_metadata(monkeypatch):
    window = _import_window("IMPORT_AUDIO")
    warnings = []
    metadata = _metadata()
    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("input.wav", ""))
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 48000),
    )
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    sequence_widget.SequenceWindow.import_audio_and_analyze(window)

    assert window.data_struct.wav_calibration_metadata == metadata
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.wav_calibration_warning_shown is False
    assert warnings == []


def test_import_audio_missing_metadata_warns_once_and_marks_authoritative(monkeypatch):
    window = _import_window("IMPORT_AUDIO")
    warnings = []
    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("input.wav", ""))
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 48000),
    )
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: None)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    sequence_widget.SequenceWindow.import_audio_and_analyze(window)

    assert window.data_struct.wav_calibration_metadata is None
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.wav_calibration_warning_shown is True
    assert warnings == ["该音频文件未包含有效校准数据，分析结果仅供参考。"]


def test_import_stimulus_audio_reads_metadata_and_preserves_decoded_rate_reference(monkeypatch):
    window = _import_window("IMPORT_STIMULUS_AUDIO")
    metadata = _metadata()
    reference_calls = []
    monkeypatch.setattr(sequence_widget.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("input.wav", ""))
    monkeypatch.setattr(
        sequence_widget,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 32000),
    )
    monkeypatch.setattr(sequence_widget, "read_wav_calibration_metadata", lambda path, logger=None: metadata)

    def fake_reference(data_struct, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        reference_calls.append(runtime_sample_rate)
        data_struct.stimulus_data = np.zeros(2, dtype=np.float32)
        data_struct.stimulus_info = {"sample_rate": runtime_sample_rate}
        return True

    monkeypatch.setattr(sequence_widget, "set_data_struct_analysis_reference_signal", fake_reference, raising=False)

    sequence_widget.SequenceWindow.import_audio_and_analyze(window)

    assert window.data_struct.wav_calibration_metadata == metadata
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.sample_rate == 32000
    assert reference_calls == [32000]


def test_blocking_record_only_saves_metadata(monkeypatch):
    calls = []
    metadata = _metadata()
    window = _record_window("RECORD_ONLY", metadata)

    monkeypatch.setattr(
        sequence_widget.SoundcardAudioProcessor,
        "sd_rec",
        staticmethod(lambda recorded_dict: (error_code.OK, np.array([0.1, 0.2], dtype=np.float32))),
    )
    monkeypatch.setattr(
        sequence_widget,
        "save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None, bit_depth=None: calls.append(
            (path, sr, calibration_metadata)
        ),
    )
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus: (error_code.OK, "ok"),
    )

    sequence_widget.SequenceWindow._start_blocking_recording(window, {}, {"sample_rate": 48000}, 48000)

    assert calls == [("record.wav", 48000, metadata)]


def test_blocking_play_and_record_passes_metadata(monkeypatch):
    calls = []
    metadata = _metadata()
    window = _record_window("PLAY_AND_RECORD", metadata)

    def fake_sd_play_rec(self, recorded_dict, stimulus_dict, path, calibration_metadata=None):
        calls.append((path, calibration_metadata, recorded_dict.get("wav_calibration_metadata")))
        return error_code.OK, np.array([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(sequence_widget.SoundcardAudioProcessor, "sd_play_rec", fake_sd_play_rec)
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus: (error_code.OK, "ok"),
    )

    sequence_widget.SequenceWindow._start_blocking_recording(
        window,
        {"sr": 48000, "data": np.zeros(2, dtype=np.float32), "amplitude": 1.0},
        {"sample_rate": 48000, "sr": 48000},
        48000,
    )

    assert calls == [("record.wav", metadata, metadata)]


@pytest.fixture
def full_input_record_window(monkeypatch):
    def deny_database_connection(*args, **kwargs):
        pytest.fail("Blocking integration tests must not open SQLite connections")

    monkeypatch.setattr(sqlite3, "connect", deny_database_connection)
    monkeypatch.setattr(recording_management, "ensure_audio_database_ready", lambda: (error_code.OK, "isolated"))

    def calibration_rows(self, hardware_id, direction, calibration_type):
        assert (hardware_id, direction, calibration_type) == ("six-input-mic", "input", "mic_v2pa")
        return [
            {"channel_index": 2, "factor_value": 2.5, "standard_spl": 94.0},
            {"channel_index": 4, "factor_value": 4.5, "standard_spl": 96.0},
        ]

    monkeypatch.setattr(HardwareManagementRepository, "list_channel_calibrations", calibration_rows)

    def build_window(mode):
        window = _bind_current_metadata_input_channels(_record_window(mode))
        window.mic = {"index": 18, "hardware_id": "six-input-mic"}
        window._active_input_channels = [2, 4]
        window._build_current_wav_calibration_metadata = lambda: (
            sequence_widget.SequenceWindow._build_current_wav_calibration_metadata(window)
        )
        return window

    return build_window


@pytest.mark.parametrize("mode", ["RECORD_ONLY", "PLAY_AND_RECORD"])
@pytest.mark.parametrize("bit_depth", [32, 64])
def test_real_blocking_capture_retains_selected_audio_and_calibration(
    monkeypatch, full_input_record_window, mode, bit_depth
):
    window = full_input_record_window(mode)
    dtype = np.dtype(f"float{bit_depth}")
    delay_frames, prepare_frames, prolong_frames = 3, 2, 2
    stimulus = np.linspace(0.1, 0.7, 7, dtype=dtype)
    amplitude = 0.25
    recorded = {
        "device": window.mic,
        "output_device": {"index": 16},
        "input_channels": [2, 4],
        "sr": 48000,
        "sample_rate": 48000,
        "num_frames": len(stimulus) + prolong_frames,
        "recording_start_delay_frames": delay_frames,
        "prepare_frames": prepare_frames,
        "prolong_frames": prolong_frames,
        "bit_depth": bit_depth,
    }
    frame_count = (
        recorded["num_frames"] + delay_frames
        if mode == "RECORD_ONLY"
        else delay_frames + prepare_frames + len(stimulus) + prolong_frames
    )
    raw = np.arange(frame_count * 6, dtype=dtype).reshape(frame_count, 6) / 1000
    queries, captures, alignment_inputs, saved, database_calls, successes, failures = [], [], [], [], [], [], []

    def query_devices(device, kind):
        queries.append((device, kind))
        return {"max_input_channels": 6}

    def capture_record(**kwargs):
        captures.append(("record", kwargs))
        return raw.copy()

    def capture_playrec(data, **kwargs):
        captures.append(("playrec", kwargs))
        np.testing.assert_array_equal(
            data,
            np.concatenate([
                np.zeros(delay_frames + prepare_frames, dtype=dtype),
                stimulus * amplitude,
                np.zeros(prolong_frames, dtype=dtype),
            ]),
        )
        return raw.copy()

    def align(reference, captured):
        alignment_inputs.append((reference.copy(), captured.copy()))
        return prepare_frames

    def save(path, data, sample_rate, calibration_metadata=None, logger=None, bit_depth=32):
        saved.append((path, data.copy(), sample_rate, calibration_metadata, bit_depth))

    monkeypatch.setattr(soundcard.sd, "query_devices", query_devices)
    monkeypatch.setattr(soundcard.sd, "rec", capture_record)
    monkeypatch.setattr(soundcard.sd, "playrec", capture_playrec)
    monkeypatch.setattr(soundcard.SoundcardAudioProcessor, "calculate_alignment", staticmethod(align))
    monkeypatch.setattr(soundcard, "save_audio_with_calibration_metadata", save)
    monkeypatch.setattr(sequence_widget, "save_audio_with_calibration_metadata", save)
    monkeypatch.setattr(soundcard, "save_audio_simple", lambda *args, **kwargs: pytest.fail("missing calibration"))
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus_info: database_calls.append((info.copy(), stimulus_info)) or (error_code.OK, "ok"),
    )
    window._finish_recording_success = successes.append
    window._finish_recording_failure = failures.append

    sequence_widget.SequenceWindow._start_blocking_recording(
        window, {"sr": 48000, "data": stimulus, "amplitude": amplitude}, recorded, 48000
    )

    assert failures == []
    assert successes == [48000]
    assert queries == [(18, "input")]
    assert len(captures) == 1
    capture_mode, capture_kwargs = captures[0]
    assert capture_mode == ("record" if mode == "RECORD_ONLY" else "playrec")
    assert capture_kwargs["channels"] == 6
    assert capture_kwargs["samplerate"] == 48000
    assert capture_kwargs["dtype"] == dtype.name
    assert capture_kwargs["blocking"] is True
    expected_metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    if mode == "RECORD_ONLY":
        assert capture_kwargs["frames"] == recorded["num_frames"] + delay_frames
        assert capture_kwargs["device"] == 18
        expected = raw[delay_frames + prolong_frames:, [2, 4]]
        expected_mono = expected.mean(axis=1)
        expected_metadata["recorded_channels"].append(
            {"wav_channel_index": 1, "v2pa_factor": 4.5, "standard_spl": 96.0, "calibrated": True}
        )
        assert alignment_inputs == []
    else:
        assert capture_kwargs["device"] == (18, 16)
        assert len(alignment_inputs) == 1
        np.testing.assert_array_equal(alignment_inputs[0][0], stimulus * amplitude)
        np.testing.assert_array_equal(alignment_inputs[0][1], raw[delay_frames:, 2])
        expected = raw[delay_frames + prepare_frames:delay_frames + prepare_frames + len(stimulus), 2]
        expected_mono = expected
    assert len(saved) == 1
    path, saved_data, saved_rate, saved_metadata, saved_bit_depth = saved[0]
    assert (path, saved_rate, saved_bit_depth) == ("record.wav", 48000, bit_depth)
    assert saved_metadata == expected_metadata
    assert recorded["wav_calibration_metadata"] == expected_metadata
    assert saved_data.shape == expected.shape
    assert saved_data.dtype == dtype
    np.testing.assert_array_equal(saved_data, expected)
    np.testing.assert_array_equal(window.data_struct.store_wave_data, expected_mono)
    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, expected.reshape(len(stimulus), -1))
    assert len(saved_data) == len(stimulus)
    assert len(database_calls) == 1
    assert database_calls[0][0]["sample_rate"] == 48000
    assert database_calls[0][1] == (None if mode == "RECORD_ONLY" else window.data_struct.stimulus_info)


@pytest.mark.parametrize("mode", ["RECORD_ONLY", "PLAY_AND_RECORD"])
@pytest.mark.parametrize("failure_boundary", ["query", "capture"])
def test_real_blocking_endpoint_failure_reaches_ui_without_persistence(
    monkeypatch, full_input_record_window, mode, failure_boundary
):
    window = full_input_record_window(mode)
    original_error = RuntimeError(f"{failure_boundary} failed")
    queries, captures, alignment_calls, saves, database_calls, successes, failures = [], [], [], [], [], [], []

    def query_devices(device, kind):
        queries.append((device, kind))
        if failure_boundary == "query":
            raise original_error
        return {"max_input_channels": 6}

    def capture(*args, **kwargs):
        captures.append(kwargs)
        raise original_error

    monkeypatch.setattr(soundcard.sd, "query_devices", query_devices)
    monkeypatch.setattr(soundcard.sd, "rec", capture)
    monkeypatch.setattr(soundcard.sd, "playrec", capture)
    monkeypatch.setattr(
        soundcard.SoundcardAudioProcessor, "calculate_alignment",
        staticmethod(lambda *args: alignment_calls.append(args) or 0),
    )
    monkeypatch.setattr(soundcard, "save_audio_simple", lambda *args, **kwargs: saves.append(args))
    monkeypatch.setattr(soundcard, "save_audio_with_calibration_metadata", lambda *args, **kwargs: saves.append(args))
    monkeypatch.setattr(sequence_widget, "save_audio_with_calibration_metadata", lambda *args, **kwargs: saves.append(args))
    monkeypatch.setattr(
        sequence_widget.RecordingManager, "save_signal_info_to_db",
        lambda *args: database_calls.append(args) or (error_code.OK, "ok"),
    )
    window._finish_recording_success = successes.append
    window._finish_recording_failure = failures.append

    sequence_widget.SequenceWindow._start_blocking_recording(
        window,
        {"sr": 48000, "data": np.ones(7, dtype=np.float32), "amplitude": 0.25},
        {"device": window.mic, "input_channels": [2, 4], "sr": 48000, "num_frames": 7},
        48000,
    )

    assert queries == [(18, "input")]
    assert len(captures) == (0 if failure_boundary == "query" else 1)
    if captures:
        assert captures[0]["channels"] == 6
    assert len(failures) == 1
    assert failures[0] is original_error
    assert alignment_calls == []
    assert saves == []
    assert database_calls == []
    assert successes == []
    assert window.data_struct.store_wave_data is None
    assert window.data_struct.store_wave_data_multi is None


def test_streaming_record_only_appends_metadata_after_finalize(monkeypatch):
    calls = []
    metadata = _metadata()
    writer = SimpleNamespace(finalize=lambda: calls.append(("finalize", None)))
    processor = SimpleNamespace(
        target_samples=2,
        get_recorded_data=lambda: np.array([0.1, 0.2], dtype=np.float32),
        get_recorded_data_multi=lambda: np.array([[0.1], [0.2]], dtype=np.float32),
    )
    window = _stream_window("record_only", processor, writer, metadata)

    monkeypatch.setattr(
        sequence_widget,
        "append_wav_calibration_metadata",
        lambda path, calibration_metadata, logger=None: calls.append((path, calibration_metadata)) or True,
    )
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus: (error_code.OK, "ok"),
    )

    sequence_widget.SequenceWindow._on_streaming_complete(window)

    assert ("finalize", None) in calls
    assert ("record.wav", metadata) in calls


def test_streaming_play_and_record_rewrites_final_wav_with_metadata(monkeypatch):
    calls = []
    metadata = _metadata()
    writer = SimpleNamespace(finalize=lambda: None)
    processor = SimpleNamespace(
        target_samples=2,
        get_recorded_data=lambda: np.array([0.1, 0.2], dtype=np.float32),
    )
    window = _stream_window("play_record", processor, writer, metadata)
    window.streaming_stimulus_data = np.array([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(
        sequence_widget.AlignmentProcessing,
        "align_play_and_rec_data_using_gccphat",
        staticmethod(lambda stimulus, recorded: np.array([0.1, 0.2], dtype=np.float32)),
    )
    monkeypatch.setattr(
        sequence_widget,
        "save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None, bit_depth=None: calls.append(
            (path, sr, calibration_metadata)
        ),
    )
    monkeypatch.setattr(sequence_widget.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus: (error_code.OK, "ok"),
    )

    sequence_widget.SequenceWindow._on_streaming_complete(window)

    assert calls == [("record.wav", 48000, metadata)]


def test_streaming_play_and_record_saves_first_retained_channel_for_single_channel_metadata(monkeypatch):
    calls = []
    alignment_inputs = []
    metadata = _metadata(7.0)
    writer = SimpleNamespace(finalize=lambda: None)
    processor = SimpleNamespace(
        target_samples=2,
        get_recorded_data=lambda: np.array([2.0, 2.5], dtype=np.float32),
        get_recorded_data_multi=lambda: np.array([[1.0, 3.0], [1.5, 3.5]], dtype=np.float32),
    )
    window = _stream_window("play_record", processor, writer, metadata)
    window._active_input_channels = [2, 4]
    window.streaming_stimulus_data = np.array([0.0, 0.0], dtype=np.float32)

    def fake_align(stimulus, recorded):
        alignment_inputs.append(np.asarray(recorded, dtype=np.float32).copy())
        return np.asarray(recorded, dtype=np.float32)

    monkeypatch.setattr(
        sequence_widget.AlignmentProcessing,
        "align_play_and_rec_data_using_gccphat",
        staticmethod(fake_align),
    )
    monkeypatch.setattr(
        sequence_widget,
        "save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None, bit_depth=None: calls.append(
            (path, np.asarray(data, dtype=np.float32).copy(), sr, calibration_metadata)
        ),
    )
    monkeypatch.setattr(sequence_widget.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        sequence_widget.RecordingManager,
        "save_signal_info_to_db",
        lambda self, info, stimulus: (error_code.OK, "ok"),
    )

    sequence_widget.SequenceWindow._on_streaming_complete(window)

    np.testing.assert_allclose(alignment_inputs[0], np.array([1.0, 1.5], dtype=np.float32))
    assert calls[0][0] == "record.wav"
    np.testing.assert_allclose(calls[0][1], np.array([1.0, 1.5], dtype=np.float32))
    assert calls[0][2] == 48000
    assert calls[0][3] == metadata
    assert len(calls[0][3]["recorded_channels"]) == 1


def _stream_window(mode, processor, writer, metadata):
    return SimpleNamespace(
        streaming_processor=processor,
        streaming_mode=mode,
        streaming_wav_writer=writer,
        streaming_stimulus_data=None,
        streaming_plot_item=None,
        streaming_temp_path="record_temp.wav",
        recorded_path="record.wav",
        recorded_signal_info={},
        data_struct=SimpleNamespace(
            sample_rate=48000,
            store_wave_data=None,
            store_wave_data_multi=None,
            stimulus_info={"repeat_times": 1},
        ),
        _active_input_channels=[0],
        _build_current_wav_calibration_metadata=lambda: metadata,
        _recording_output_path=lambda: "record.wav",
        _finalize_successful_replay_output=lambda: None,
        _commit_pending_recorded_count=lambda: None,
        _clear_pending_recorded_count=lambda: None,
        _delete_successful_replay_backup=lambda: None,
        _clear_recording_output_attempt=lambda: None,
        _delete_failed_streaming_outputs=lambda: None,
        _run_post_recording_followup=lambda action_name, callback, *args, **kwargs: callback(*args, **kwargs),
        default_logger=FakeLogger(),
        data_btn=FakeButton(),
        replayer_btn=FakeButton(),
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
        analysis_config={"auto_analysis": False},
        _set_sn_input_recording_read_only=lambda value: None,
        update_player_btn_is_paused=lambda: None,
        run=lambda: None,
        _awaiting_ok_ng=False,
        _sn_clear_on_next_scan=False,
        player_status_flag=True,
        _record_workflow_busy=True,
    )
