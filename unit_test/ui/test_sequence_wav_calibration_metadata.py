import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.data_struct.data_deal_struct import DataDealStruct
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
