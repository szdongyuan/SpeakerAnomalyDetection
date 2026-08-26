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
from ui.sequence import sequence_recording_import_service as import_service_module
from ui.sequence.sequence_recording_import_owner import (
    SequenceRecordingImportController,
)
from ui.sequence.sequence_recording_import_service import SequenceImportedAudioService
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    ConfigurationSnapshot,
    LoadImportedAudioRequested,
)
from ui.sequence import sequence_recording_controller
from ui.sequence.sequence_recording_controller import (
    BlockingRecordingAdapter,
    PreparedRecordingSession,
)
from ui.sequence.sequence_recording_model import (
    RecordingSessionSnapshot,
    thaw_recording_session_value,
)
from ui.sequence.sequence_recording_worker import _build_staged_result


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


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


def _execute_import(window, selected_path="input.wav"):
    snapshot = ConfigurationSnapshot(
        window.sequence_config,
        window.analysis_config,
        using_config_path=window.using_config_path,
    )
    command = LoadImportedAudioRequested(
        "compat-command",
        "compat-import",
        window.sequence_config[0]["seq1"]["acq"]["mode"],
        selected_path,
        snapshot,
    )
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda title, message: sequence_widget.MessageBox.warning(
            window, title, message
        ),
        clear_import_projection=window._clear_plot_area,
        show_imported_audio=window.plot_waveform_to_workspace,
        set_import_data_enabled=window.data_btn.setEnabled,
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    service = SequenceImportedAudioService(
        audio_loader=import_service_module.load_audio_preserve_rate,
        metadata_reader=import_service_module.read_wav_calibration_metadata,
        reference_builder=import_service_module.set_data_struct_analysis_reference_signal,
    )
    controller = SequenceRecordingImportController(
        RecordingModel(), view, bus=bus, runtime=window, import_service=service
    )
    return controller.handle_load_imported_audio_requested(command)


def _prepare_recording_metadata(tmp_path, monkeypatch, *, mode, input_channels):
    calls = []
    expected = _metadata()
    monkeypatch.setattr(
        sequence_recording_controller,
        "build_recording_wav_calibration_metadata",
        lambda channels, hardware_id=None, logger=None: calls.append(
            (list(channels), hardware_id, logger)
        )
        or expected,
    )
    detail = {"total_time": 2 / 48_000, "use_streaming_recording": False}
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": mode, "detail": detail}}}],
        {"auto_analysis": False},
        mic={
            "name": "mic",
            "hardware_id": "mic-1",
            "hostapi": 1,
            "samplerate": 48_000,
        },
        speaker={"name": "speaker", "hostapi": 1, "samplerate": 48_000},
        mic_channels=tuple(input_channels),
    )
    admission = {
        "workflow_generation": 1,
        "configuration": configuration,
        "recorded_path": str(tmp_path / "record.wav"),
        "recorded_signal_info": {},
        "record_id": "record-1",
    }
    if mode == "PLAY_AND_RECORD":
        admission.update(
            stimulus_data=np.zeros(2, dtype=np.float32),
            stimulus_info={"amplitude": 1.0, "repeat_times": 1},
        )
    logger = FakeLogger()
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda _info, _stimulus: None,
        commit_count=lambda _count: None,
        persist_count=lambda _count, _context: None,
        logger=logger,
    )

    prepared = adapter.prepare(
        BeginRecordingRequested("command-1", "session-1", False, admission)
    )

    return prepared, calls, expected, logger


def test_formal_recording_owner_uses_selected_mic_and_active_channels(
    tmp_path, monkeypatch
):
    prepared, calls, expected, logger = _prepare_recording_metadata(
        tmp_path,
        monkeypatch,
        mode="RECORD_ONLY",
        input_channels=(2, 4),
    )

    frozen = prepared.acquisition_context["wav_calibration_metadata"]
    assert frozen is not expected
    assert thaw_recording_session_value(frozen) == expected
    assert calls == [([2, 4], "mic-1", logger)]


def test_formal_play_and_record_metadata_uses_only_saved_mono_input_channel(
    tmp_path, monkeypatch
):
    prepared, calls, expected, logger = _prepare_recording_metadata(
        tmp_path,
        monkeypatch,
        mode="PLAY_AND_RECORD",
        input_channels=(2, 4),
    )

    frozen = prepared.acquisition_context["wav_calibration_metadata"]
    assert frozen is not expected
    assert thaw_recording_session_value(frozen) == expected
    assert calls == [([2], "mic-1", logger)]


def test_formal_record_only_metadata_keeps_all_saved_input_channels(
    tmp_path, monkeypatch
):
    prepared, calls, expected, logger = _prepare_recording_metadata(
        tmp_path,
        monkeypatch,
        mode="RECORD_ONLY",
        input_channels=(2, 4),
    )

    frozen = prepared.acquisition_context["wav_calibration_metadata"]
    assert frozen is not expected
    assert thaw_recording_session_value(frozen) == expected
    assert calls == [([2, 4], "mic-1", logger)]


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
    monkeypatch.setattr(
        import_service_module,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 48000),
    )
    monkeypatch.setattr(import_service_module, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    _execute_import(window)

    assert window.data_struct.wav_calibration_metadata == metadata
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.wav_calibration_warning_shown is False
    assert warnings == []


def test_import_audio_missing_metadata_warns_once_and_marks_authoritative(monkeypatch):
    window = _import_window("IMPORT_AUDIO")
    warnings = []
    monkeypatch.setattr(
        import_service_module,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 48000),
    )
    monkeypatch.setattr(import_service_module, "read_wav_calibration_metadata", lambda path, logger=None: None)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    _execute_import(window)

    assert window.data_struct.wav_calibration_metadata is None
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.wav_calibration_warning_shown is True
    assert warnings == ["该音频文件未包含有效校准数据，分析结果仅供参考。"]


def test_import_stimulus_audio_reads_metadata_and_preserves_decoded_rate_reference(monkeypatch):
    window = _import_window("IMPORT_STIMULUS_AUDIO")
    metadata = _metadata()
    reference_calls = []
    monkeypatch.setattr(
        import_service_module,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.array([0.1, 0.2], dtype=np.float32), 32000),
    )
    monkeypatch.setattr(import_service_module, "read_wav_calibration_metadata", lambda path, logger=None: metadata)

    def fake_reference(data_struct, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        reference_calls.append(runtime_sample_rate)
        data_struct.stimulus_data = np.zeros(2, dtype=np.float32)
        data_struct.stimulus_info = {"sample_rate": runtime_sample_rate}
        return True

    monkeypatch.setattr(import_service_module, "set_data_struct_analysis_reference_signal", fake_reference, raising=False)

    _execute_import(window)

    assert window.data_struct.wav_calibration_metadata == metadata
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.sample_rate == 32000
    assert reference_calls == [32000]


def test_blocking_record_only_saves_metadata():
    expected = _metadata()
    prepared = SimpleNamespace(
        acquisition_context={"wav_calibration_metadata": expected}
    )
    calls = []
    facade = SimpleNamespace(
        blocking_recording_adapter=SimpleNamespace(
            acquire=lambda received: calls.append(received) or "staged"
        )
    )

    assert sequence_widget.SequenceWindow._start_blocking_recording(
        facade, prepared
    ) == "staged"
    assert calls == [prepared]
    assert calls[0].acquisition_context["wav_calibration_metadata"] is expected


def test_blocking_play_and_record_passes_metadata():
    expected = _metadata(3.0)
    prepared = SimpleNamespace(
        acquisition_context={"wav_calibration_metadata": expected}
    )
    calls = []
    facade = SimpleNamespace(
        blocking_recording_adapter=SimpleNamespace(
            acquire=lambda received: calls.append(received) or "staged"
        )
    )

    assert sequence_widget.SequenceWindow._start_blocking_recording(
        facade, prepared
    ) == "staged"
    assert calls == [prepared]
    assert calls[0].acquisition_context["wav_calibration_metadata"] is expected








def _formal_prepared(tmp_path, mode, metadata, input_channels=(0,)):
    snapshot = RecordingSessionSnapshot.create(
        session_id="session-1",
        workflow_generation=1,
        mode=mode,
        sample_rate=48000,
        bit_depth=32,
        input_channels=input_channels,
        input_device={"name": "mic"},
        output_device={"name": "speaker"} if mode == "PLAY_AND_RECORD" else None,
        stimulus_snapshot={},
        target_samples=2,
        acquisition_sample_count=2,
        output_path=tmp_path / "record.wav",
        temp_path=tmp_path / "record.tmp.wav",
        backup_path=None,
        record_id="record-1",
    )
    return PreparedRecordingSession(
        snapshot,
        {
            "recorded_dict": {},
            "recorded_signal_info": {},
            "stimulus_dict": {"data": np.array([0.0, 0.0], dtype=np.float32)},
            "stimulus_info": {},
            "stimulus_data": np.array([0.0, 0.0], dtype=np.float32),
            "wav_calibration_metadata": metadata,
            "count_persistence": {},
        },
    )


def _formal_adapter():
    return BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: (error_code.OK, "ok"),
        commit_count=lambda count: None,
        persist_count=lambda count, context: None,
    )


def test_streaming_record_only_appends_metadata_after_finalize(tmp_path, monkeypatch):
    calls = []
    metadata = _metadata()
    prepared = _formal_prepared(tmp_path, "RECORD_ONLY", metadata)
    monkeypatch.setattr(
        sequence_recording_controller,
        "append_wav_calibration_metadata",
        lambda path, payload, logger=None: calls.append((path, payload)) or True,
    )

    _formal_adapter().transaction(prepared).finalize_metadata(
        SimpleNamespace(snapshot=prepared.snapshot)
    )

    assert calls == [(prepared.snapshot.temp_path, metadata)]


def test_streaming_play_and_record_rewrites_final_wav_with_metadata(tmp_path, monkeypatch):
    calls = []
    metadata = _metadata()
    prepared = _formal_prepared(tmp_path, "PLAY_AND_RECORD", metadata)
    monkeypatch.setattr(
        sequence_recording_controller,
        "append_wav_calibration_metadata",
        lambda path, payload, logger=None: calls.append((path, payload)) or True,
    )

    _formal_adapter().transaction(prepared).finalize_metadata(
        SimpleNamespace(snapshot=prepared.snapshot)
    )

    assert calls == [(prepared.snapshot.temp_path, metadata)]


def test_streaming_play_and_record_saves_first_retained_channel_for_single_channel_metadata(tmp_path):
    saved = []
    metadata = _metadata(7.0)
    prepared = _formal_prepared(tmp_path, "PLAY_AND_RECORD", metadata, (2, 4))
    staged = _build_staged_result(
        prepared,
        lambda stimulus, recorded: np.asarray(recorded, dtype=np.float32),
        lambda path, data, sr, bit_depth: saved.append(
            (path, np.asarray(data).copy(), sr, bit_depth)
        ),
        np.array([2.0, 2.5], dtype=np.float32),
        np.array([[1.0, 3.0], [1.5, 3.5]], dtype=np.float32),
    )

    np.testing.assert_allclose(staged.data_struct_fields["store_wave_data"], [1.0, 1.5])
    np.testing.assert_allclose(staged.data_struct_fields["store_wave_data_multi"], [[1.0], [1.5]])
    assert saved[0][0] == str(prepared.snapshot.temp_path)
    np.testing.assert_allclose(saved[0][1], [1.0, 1.5])
    assert saved[0][2:] == (48000, 32)
    frozen = prepared.acquisition_context["wav_calibration_metadata"]
    assert frozen is not metadata
    assert thaw_recording_session_value(frozen) == metadata


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
        _sn_clear_on_next_scan=False,
        player_status_flag=True,
        _record_workflow_busy=True,
    )
