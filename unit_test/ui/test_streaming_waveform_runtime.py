import inspect
import logging
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from ui.sequence import sequence_widget_streaming_ops
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _Writer:
    def __init__(self):
        self.chunks = []
        self.finalized = False

    def write_chunk(self, chunk):
        self.chunks.append(chunk)

    def finalize(self):
        self.finalized = True


class _RuntimeHost(SequenceWidgetStreamingOpsMixin):
    def __init__(self, channels=(0,)):
        self.data_btn = mock.Mock()
        self.streaming_wav_writer = _Writer()
        self.default_logger = logging.getLogger(__name__)
        self._recording_input_channels = tuple(channels)
        self._active_input_channels = list(channels)
        self.sequence_config = [{"seq1": {"acq": {"detail": {}}}}]
        self._streaming_first_chunk_logged = True
        self._streaming_chunk_contract_failed = False
        self.streaming_processor = object()
        self.scheduled = []
        # Isolated legacy host has no current-product retry workflow.
        self._recover_current_product_recording = lambda reason: False


    def _finalize_recording_channel_selection(self):
        self._recording_input_channels = None

    def _abort_recording_channel_selection(self):
        self._finalize_recording_channel_selection()


class _Button:
    def setDisabled(self, disabled):
        self.disabled = disabled


class _RecordingBoundaryHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self, *, streaming=True):
        from unit_test.ui.test_recording_process_integration import CapturingBridge
        self.recording_bridge = CapturingBridge()
        self._recording_input_channels = (0, 1)
        self.use_streaming = streaming
        self.events = []
        self._record_workflow_busy = False
        self.last_play_count = None
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.player_status_flag = False
        self.replayer_btn = _Button()
        self.data_btn = _Button()
        self.streaming_processor = None
        self.streaming_wav_writer = None
        self._streaming_completion_processor = None
        self._active_input_channels = [0, 1]
        self.recorded_path = "record.wav"
        self.recorded_signal_info = {"name": "capture"}
        self.default_logger = mock.Mock()
        self._current_trigger_direction = "forward"
        self.captured_direction = ""

    def checked_work_status_message(self):
        return False

    def _close_analysis_windows(self):
        return None


    def _cleanup_streaming_resources(self):
        self.streaming_processor = None
        self.streaming_wav_writer = None

    def update_player_btn_is_playing(self):
        return None

    def update_player_btn_is_paused(self):
        return None

    def reset_work_pram(self, label, count=None):
        from unit_test.base.recording_process_fakes import device_info
        return {"device": device_info(), "num_frames": 4800}, 48_000

    def _capture_recording_wav_calibration_metadata(self):
        return None

    def _should_use_streaming_recording(self):
        return self.use_streaming

    def _resolve_recording_acq_detail(self):
        return {"startup_trim_ms": 25}

    def _resolve_active_recording_waveform_direction(self, fallback="forward"):
        self.events.append("resolve-direction")
        return self._current_trigger_direction or fallback


    def _begin_recent_session_for_current_run(self):
        self.events.append("recent")

    def _start_blocking_recording(self, recorded_dict, sample_rate):
        self.events.append(("blocking", sample_rate))

    def _unlock_sn_after_recording_if_needed(self):
        return None

    def _on_serial_product_runtime_error(self, reason):
        return True


def test_judge_snapshots_trim_and_direction_before_async_service_start():
    host = _RecordingBoundaryHost(streaming=True)
    host.judge_play_and_record()
    host._current_trigger_direction = "reverse"

    assert host.recording_bridge.request.trim_samples == 1200
    assert host._recording_process_direction == "forward"
    assert host.events == ["resolve-direction", "recent"]


def test_judge_ordinary_path_invalidates_live_session_before_async_capture():
    host = _RecordingBoundaryHost(streaming=False)
    host.captured_direction = "forward"
    host.judge_play_and_record()

    assert host.events == ["resolve-direction", "recent"]
    assert host._record_workflow_busy is True


def test_judge_service_start_failure_restores_controls(monkeypatch):
    host = _RecordingBoundaryHost(streaming=True)
    host.recording_bridge.start = mock.Mock(side_effect=RuntimeError("device unavailable"))
    host.judge_play_and_record()

    assert host._record_workflow_busy is False
    assert host.player_status_flag is False
    assert host._recording_process_id is None


def test_payload_normalization_appends_exact_multichannel_chunk_and_ignores_mono():
    host = _RuntimeHost(channels=(0, 2))
    multi = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float64)
    payload = {
        "multi": multi,
        "mono": np.array([99.0, 98.0], dtype=np.float32),
    }

    host.on_audio_chunk_received(payload)

    appended = host.streaming_wav_writer.chunks[0]
    np.testing.assert_array_equal(appended, multi.astype(np.float32))
    assert appended.dtype == np.float32
    assert len(host.streaming_wav_writer.chunks) == 1


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"multi": []},
        {"multi": np.empty((0, 2), dtype=np.float32)},
        {"multi": np.zeros((1, 1, 1), dtype=np.float32)},
        {"multi": object()},
    ],
)
def test_invalid_payload_is_rejected_before_wav_write(payload):
    host = _RuntimeHost()
    writer = host.streaming_wav_writer
    host._handle_invalid_recording = mock.Mock()

    host.on_audio_chunk_received(payload)

    assert writer.chunks == []
    host._handle_invalid_recording.assert_called_once()


def test_column_mismatch_routes_once_to_invalid_recording_cleanup_before_persistence():
    host = _RuntimeHost(channels=(0, 2))
    processor = mock.Mock()
    writer = host.streaming_wav_writer
    host.streaming_processor = processor
    host._handle_invalid_recording = mock.Mock()
    mismatch = np.ones((3, 1), dtype=np.float32)

    host.on_audio_chunk_received({"multi": mismatch})
    host.on_audio_chunk_received({"multi": mismatch})

    assert writer.chunks == []
    host._handle_invalid_recording.assert_called_once()
    reason = host._handle_invalid_recording.call_args.args[0]
    assert "expected 2 channels" in reason
    assert "actual shape (3, 1)" in reason
    processor.stop_streaming.assert_called_once_with()
    assert writer.finalized is True
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None


def test_one_dimensional_payload_is_accepted_only_for_single_channel():
    single = _RuntimeHost(channels=(3,))
    values = np.array([1.0, 2.0], dtype=np.float32)

    single.on_audio_chunk_received(values)

    np.testing.assert_array_equal(single.streaming_wav_writer.chunks[0], values[:, None])

    multi = _RuntimeHost(channels=(0, 2))
    writer = multi.streaming_wav_writer
    multi._handle_invalid_recording = mock.Mock()
    multi.on_audio_chunk_received(values)
    assert writer.chunks == []
    multi._handle_invalid_recording.assert_called_once()


def test_each_valid_payload_reaches_wav_writer_exactly_once():
    host = _RuntimeHost(channels=(0, 2))
    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.array([[5.0, 6.0]], dtype=np.float32)

    host.on_audio_chunk_received({"multi": first})
    host.on_audio_chunk_received({"multi": second})

    assert len(host.streaming_wav_writer.chunks) == 2
    np.testing.assert_array_equal(host.streaming_wav_writer.chunks[0], first)
    np.testing.assert_array_equal(host.streaming_wav_writer.chunks[1], second)


@pytest.mark.parametrize("error", [TypeError("bad"), ValueError("bad"), OverflowError("bad")])
def test_optional_mono_is_never_materialized(error):
    class _BadMono:
        def __array__(self, *args, **kwargs):
            raise error

    host = _RuntimeHost(channels=(0, 2))
    host.default_logger = mock.Mock()
    multi = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)

    host.on_audio_chunk_received({"multi": multi, "mono": _BadMono()})

    appended = host.streaming_wav_writer.chunks[0]
    np.testing.assert_array_equal(appended, multi)
    assert len(host.streaming_wav_writer.chunks) == 1
    host.default_logger.error.assert_not_called()


def test_payload_handler_contains_no_historical_concatenation():
    source = inspect.getsource(SequenceWidgetStreamingOpsMixin.on_audio_chunk_received)

    assert "np.concatenate" not in source
    assert "streaming_buffer_multi" not in source


def test_finished_drain_contract_failure_skips_normal_completion_and_terminates_once():
    host = _RuntimeHost(channels=(0, 2))
    writer = host.streaming_wav_writer
    bad_chunk = np.ones((3, 1), dtype=np.float32)
    processor = mock.Mock()
    processor.process_queue.side_effect = lambda: host.on_audio_chunk_received(
        {"multi": bad_chunk}
    )
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host._serial_product_condition_executing = True
    host._on_serial_product_runtime_error = mock.Mock(return_value=True)
    host._handle_invalid_recording = mock.Mock(wraps=host._handle_invalid_recording)
    host._on_streaming_complete = mock.Mock()
    host._abort_recording_channel_selection = mock.Mock(
        wraps=host._abort_recording_channel_selection
    )
    writer.finalize = mock.Mock(wraps=writer.finalize)

    host._on_streaming_recording_finished(processor)
    host._on_streaming_recording_finished(processor)

    processor.process_queue.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    host._on_streaming_complete.assert_not_called()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._abort_recording_channel_selection.assert_called_once_with()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None


def test_finished_drain_exception_uses_invalid_terminal_path_once():
    host = _RuntimeHost(channels=(0, 2))
    writer = host.streaming_wav_writer
    writer.finalize = mock.Mock(wraps=writer.finalize)
    processor = mock.Mock()
    processor.process_queue.side_effect = RuntimeError("final drain failed")
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host._record_workflow_busy = True
    host._on_streaming_complete = mock.Mock()
    host._handle_invalid_recording = mock.Mock(
        wraps=host._handle_invalid_recording
    )
    host._abort_recording_channel_selection = mock.Mock(
        side_effect=lambda: setattr(host, "_recording_input_channels", None)
    )
    host._finalize_recording_channel_selection = mock.Mock(
        side_effect=lambda: setattr(host, "_recording_input_channels", None)
    )

    host._on_streaming_recording_finished(processor)
    host._on_streaming_recording_finished(processor)

    processor.process_queue.assert_called_once_with()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    assert "final drain failed" in host._handle_invalid_recording.call_args.args[0]
    host._abort_recording_channel_selection.assert_called_once_with()
    host._finalize_recording_channel_selection.assert_not_called()
    host._on_streaming_complete.assert_not_called()
    assert host.streaming_processor is None
    assert host._streaming_completion_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert host._record_workflow_busy is False


def _configure_invalid_terminal_recovery(host, tmp_path, filename):
    failed_path = tmp_path / filename
    failed_path.write_bytes(b"incomplete wav")
    host.recorded_path = str(failed_path)
    host._discard_current_recent_session = mock.Mock()
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host.update_player_btn_is_paused = mock.Mock()
    host._record_workflow_busy = True
    host.player_status_flag = True
    host._handle_invalid_recording = mock.Mock(
        wraps=host._handle_invalid_recording
    )
    host._abort_recording_channel_selection = mock.Mock(
        side_effect=lambda: setattr(host, "_recording_input_channels", None)
    )
    host._finalize_recording_channel_selection = mock.Mock(
        side_effect=lambda: setattr(host, "_recording_input_channels", None)
    )
    return failed_path


def test_queue_ready_exception_uses_invalid_terminal_path_once(tmp_path):
    host = _RuntimeHost(channels=(0, 2))
    failed_path = _configure_invalid_terminal_recovery(
        host, tmp_path, "queue-failed.wav"
    )
    writer = host.streaming_wav_writer
    writer.finalize = mock.Mock(wraps=writer.finalize)
    processor = mock.Mock()
    processor.process_queue.side_effect = RuntimeError("queue drain failed")
    host.streaming_processor = processor
    host._streaming_completion_processor = None

    host._on_streaming_queue_ready(processor)
    host._on_streaming_queue_ready(processor)
    host._on_streaming_recording_finished(processor)
    host._cleanup_streaming_resources()

    processor.process_queue.assert_called_once_with()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    assert "queue drain failed" in host._handle_invalid_recording.call_args.args[0]
    host._abort_recording_channel_selection.assert_called_once_with()
    host._finalize_recording_channel_selection.assert_not_called()
    host._discard_current_recent_session.assert_called_once_with()
    assert not failed_path.exists()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert host._record_workflow_busy is False


def test_chunk_write_failure_uses_invalid_terminal_path_once(
    tmp_path, monkeypatch
):
    host = _RuntimeHost(channels=(0, 2))
    failed_path = _configure_invalid_terminal_recovery(
        host, tmp_path, "chunk-write-failed.wav"
    )
    writer = host.streaming_wav_writer
    writer.write_chunk = mock.Mock(side_effect=RuntimeError("disk full"))
    writer.finalize = mock.Mock(wraps=writer.finalize)
    processor = mock.Mock()
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host._on_streaming_complete = mock.Mock()
    validation = mock.Mock()
    database = mock.Mock()
    monkeypatch.setattr(
        sequence_widget_streaming_ops, "validate_recorded_audio", validation
    )
    monkeypatch.setattr(
        sequence_widget_streaming_ops, "RecordingManager", lambda: database
    )
    chunk = np.ones((4, 2), dtype=np.float32)

    host.on_audio_chunk_received(chunk)
    host.on_audio_chunk_received(chunk)
    host._on_streaming_queue_ready(processor)
    host._on_streaming_recording_finished(processor)
    host._cleanup_streaming_resources()

    writer.write_chunk.assert_called_once_with(chunk)
    processor.process_queue.assert_not_called()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    assert "disk full" in host._handle_invalid_recording.call_args.args[0]
    host._abort_recording_channel_selection.assert_called_once_with()
    host._finalize_recording_channel_selection.assert_not_called()
    host._discard_current_recent_session.assert_called_once_with()
    host._on_streaming_complete.assert_not_called()
    validation.assert_not_called()
    database.save_signal_info_to_db.assert_not_called()
    assert not failed_path.exists()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert host._record_workflow_busy is False


def test_normal_completion_finalize_failure_uses_invalid_terminal_once(
    tmp_path, monkeypatch
):
    host = _RuntimeHost()
    failed_path = _configure_invalid_terminal_recovery(
        host, tmp_path, "finalize-failed.wav"
    )
    writer = host.streaming_wav_writer
    writer.finalize = mock.Mock(side_effect=RuntimeError("finalize failed"))
    processor = mock.Mock()
    processor.target_samples = 4
    processor.get_recorded_data.return_value = np.ones(4, dtype=np.float32)
    processor.get_recorded_data_multi.return_value = np.ones(
        (4, 1), dtype=np.float32
    )
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host.data_struct = SimpleNamespace(
        sample_rate=48_000,
        store_wave_data=None,
        store_wave_data_multi=None,
    )
    host.streaming_stimulus_data = object()
    host.streaming_mode = "record_only"
    host._awaiting_ok_ng = True
    host._sn_clear_on_next_scan = True
    host._pending_recent_session_append = True
    host._on_serial_product_runtime_error = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()
    validation = mock.Mock()
    database = mock.Mock()
    monkeypatch.setattr(
        sequence_widget_streaming_ops, "validate_recorded_audio", validation
    )
    monkeypatch.setattr(
        sequence_widget_streaming_ops, "RecordingManager", lambda: database
    )

    host._on_streaming_recording_finished(processor)
    host._on_streaming_recording_finished(processor)
    host._cleanup_streaming_resources()

    writer.finalize.assert_called_once_with()
    processor.process_queue.assert_called_once_with()
    processor.stop_streaming.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    assert "finalize failed" in host._handle_invalid_recording.call_args.args[0]
    host._abort_recording_channel_selection.assert_called_once_with()
    host._finalize_recording_channel_selection.assert_not_called()
    host._discard_current_recent_session.assert_called_once_with()
    validation.assert_not_called()
    database.save_signal_info_to_db.assert_not_called()
    assert not failed_path.exists()
    assert host.streaming_wav_writer is None
    assert host.streaming_processor is None
    assert host._streaming_completion_processor is None
    assert host.streaming_stimulus_data is None
    assert host.streaming_mode is None
    assert host.player_status_flag is False
    assert host._record_workflow_busy is False
    host.data_btn.setEnabled.assert_called_once_with(True)
    host.replayer_btn.setEnabled.assert_called_once_with(True)
    host._on_serial_product_runtime_error.assert_not_called()


def test_later_completion_error_does_not_delete_recording_as_invalid(
    tmp_path, monkeypatch
):
    host = _RuntimeHost()
    recorded_path = tmp_path / "database-failed.wav"
    recorded_path.write_bytes(b"valid finalized wav")
    host.recorded_path = str(recorded_path)
    writer = host.streaming_wav_writer
    writer.finalize = mock.Mock(wraps=writer.finalize)
    host.data_struct = SimpleNamespace(
        store_wave_data=None,
        store_wave_data_multi=None,
    )
    host.recorded_signal_info = {}
    host.streaming_stimulus_data = object()
    host.streaming_mode = "record_only"
    host.player_status_flag = True
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host._record_workflow_busy = True
    host.update_player_btn_is_paused = mock.Mock()
    host._on_serial_product_runtime_error = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()
    host._resolve_recording_acq_detail = lambda: {}
    host._append_recording_wav_calibration_metadata = mock.Mock(return_value=True)
    host._resolve_active_recording_waveform_direction = lambda fallback="": "forward"
    host.plot_waveform_to_workspace = mock.Mock()
    host._handle_invalid_recording = mock.Mock()
    host._discard_current_recent_session = mock.Mock()
    monkeypatch.setattr(
        sequence_widget_streaming_ops,
        "validate_recorded_audio",
        mock.Mock(return_value=(True, "", {})),
    )
    database = mock.Mock()
    database.save_signal_info_to_db.side_effect = RuntimeError("database failed")
    monkeypatch.setattr(
        sequence_widget_streaming_ops, "RecordingManager", lambda: database
    )

    host._on_streaming_complete(
        recorded_mono=np.ones(4, dtype=np.float32),
        recorded_multi=np.ones((4, 1), dtype=np.float32),
        sample_rate=48_000,
    )

    writer.finalize.assert_called_once_with()
    host._handle_invalid_recording.assert_not_called()
    host._discard_current_recent_session.assert_not_called()
    assert recorded_path.exists()


def test_manual_cleanup_invalidates_pending_live_callback_before_resources():
    host = _RuntimeHost()
    events = []
    processor = mock.Mock()
    processor.stop_streaming.side_effect = lambda: events.append("processor")
    host.streaming_wav_writer.finalize = mock.Mock(
        side_effect=lambda: events.append("writer")
    )
    host.streaming_processor = processor
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    host._cleanup_streaming_resources()
    assert host._recording_input_channels is None
    assert events == ["processor", "writer"]
    processor.stop_streaming.assert_called_once_with()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None


def test_cleanup_supports_minimal_host_without_live_or_channel_capabilities():
    host = SimpleNamespace(
        streaming_processor=None,
        streaming_wav_writer=None,
        streaming_stimulus_data=object(),
        streaming_mode="record_only",
    )

    SequenceWidgetStreamingOpsMixin._cleanup_streaming_resources(host)

    assert host.streaming_stimulus_data is None
    assert host.streaming_mode is None


@pytest.mark.parametrize(
    "failure_point",
    ["channel-query", "channel-abort"],
)
def test_contract_cleanup_capability_failure_does_not_block_core_resources(
    failure_point,
):
    host = _RuntimeHost(channels=(0, 2))
    host.default_logger = mock.Mock()
    processor = mock.Mock()
    writer = host.streaming_wav_writer
    writer.finalize = mock.Mock(wraps=writer.finalize)
    host.streaming_processor = processor
    host._handle_invalid_recording = mock.Mock()
    failure = RuntimeError(f"{failure_point} failed")
    if failure_point == "channel-query":
        host._recording_channel_selection_is_active = mock.Mock(side_effect=failure)
    else:
        host._abort_recording_channel_selection = mock.Mock(side_effect=failure)

    host._reject_streaming_chunk_contract("bad shape")
    host._cleanup_streaming_resources()

    host._handle_invalid_recording.assert_called_once_with("bad shape")
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert any(
        failure_point in str(call.args[0])
        for call in host.default_logger.error.call_args_list
    )


def test_contract_rejection_calls_invalid_handler_when_cleanup_capability_raises():
    host = _RuntimeHost()
    host.default_logger = mock.Mock()
    host._cleanup_streaming_resources = mock.Mock(
        side_effect=RuntimeError("cleanup capability failed")
    )
    host._handle_invalid_recording = mock.Mock()

    host._reject_streaming_chunk_contract("bad shape")

    host._handle_invalid_recording.assert_called_once_with("bad shape")
    assert "cleanup capability failed" in host.default_logger.error.call_args.args[0]


@pytest.mark.parametrize(
    ("processor_raises", "writer_raises"),
    [(True, False), (False, True), (True, True)],
)
def test_contract_cleanup_releases_failing_core_resources_once(
    processor_raises,
    writer_raises,
):
    host = _RuntimeHost()
    host.default_logger = mock.Mock()
    host.streaming_stimulus_data = object()
    host.streaming_mode = "record_only"
    host._handle_invalid_recording = mock.Mock()
    processor = mock.Mock()
    writer = host.streaming_wav_writer
    processor_ownership = []
    writer_ownership = []

    def stop_streaming():
        processor_ownership.append(host.streaming_processor)
        if processor_raises:
            raise RuntimeError("processor stop failed")

    def finalize():
        writer_ownership.append(host.streaming_wav_writer)
        if writer_raises:
            raise RuntimeError("writer finalize failed")

    processor.stop_streaming.side_effect = stop_streaming
    writer.finalize = mock.Mock(side_effect=finalize)
    host.streaming_processor = processor

    host._reject_streaming_chunk_contract("bad shape")
    host._cleanup_streaming_resources()

    host._handle_invalid_recording.assert_called_once_with("bad shape")
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    assert processor_ownership == [None]
    assert writer_ownership == [None]
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host.streaming_stimulus_data is None
    assert host.streaming_mode is None
