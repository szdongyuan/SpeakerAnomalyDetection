import gc
import inspect
import logging
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from ui.sequence import sequence_widget_analysis_ops, sequence_widget_streaming_ops
from ui.sequence.multichannel_waveform_session import MultichannelWaveformSession
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


class _PlotWindow:
    def __init__(self, channel, calls):
        self.channel_index = channel
        self._calls = calls

    def set_data(self, time_axis, amplitude):
        self._calls.append(
            SimpleNamespace(
                channel=self.channel_index,
                x=np.array(time_axis, copy=True),
                y=np.array(amplitude, copy=True),
            )
        )


class _Workspace:
    def __init__(self, channels):
        self.calls = []
        self._windows = [_PlotWindow(channel, self.calls) for channel in channels]

    def all_subwindows(self):
        return list(self._windows)

    def set_direction_data(self, direction, time_axis, amplitude):
        self.calls.append(
            SimpleNamespace(
                direction=direction,
                x=np.array(time_axis, copy=True),
                y=np.array(amplitude, copy=True),
            )
        )


class _RuntimeHost(SequenceWidgetStreamingOpsMixin):
    def __init__(self, channels=(0,)):
        self.streaming_wav_writer = _Writer()
        self.default_logger = logging.getLogger(__name__)
        self._recording_input_channels = tuple(channels)
        self._active_input_channels = list(channels)
        self.channel_workspace = _Workspace(channels)
        self._streaming_first_chunk_logged = True
        self._streaming_waveform_session = MultichannelWaveformSession(
            max_points=self._WAVEFORM_DISPLAY_MAX_POINTS
        )
        self._streaming_waveform_generation = 0
        self._streaming_waveform_refresh_scheduled = False
        self._streaming_waveform_pending = False
        self._streaming_waveform_live_enabled = False
        self._streaming_waveform_failure_logged = False
        self._streaming_chunk_contract_failed = False
        self._direction_waveform_cache = {}
        self.streaming_processor = object()
        self.scheduled = []

    def _schedule_streaming_waveform_callback(self, callback):
        self.scheduled.append(callback)

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

    def _clear_plot_area(self):
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

    def _begin_streaming_waveform_session(
        self, sample_rate, startup_trim_samples, direction
    ):
        self.events.append(
            ("begin", sample_rate, startup_trim_samples, direction)
        )
        self.captured_direction = str(direction or "")

    def _end_streaming_waveform_session(self):
        self.events.append("end")
        self.captured_direction = ""

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
    assert host.events == ["end", "resolve-direction", "recent"]
    assert host._recording_process_preview_enabled is True


def test_judge_ordinary_path_invalidates_live_session_before_async_capture():
    host = _RecordingBoundaryHost(streaming=False)
    host.captured_direction = "forward"
    host.judge_play_and_record()

    assert host.events == ["end", "resolve-direction", "recent"]
    assert host._recording_process_preview_enabled is False
    assert host.captured_direction == ""
    assert host._record_workflow_busy is True


def test_judge_service_start_failure_restores_controls(monkeypatch):
    host = _RecordingBoundaryHost(streaming=True)
    host.recording_bridge.start = mock.Mock(side_effect=RuntimeError("device unavailable"))
    host.judge_play_and_record()

    assert host.captured_direction == ""
    assert host._record_workflow_busy is False
    assert host.player_status_flag is False
    assert host._recording_process_id is None


def test_payload_normalization_appends_exact_multichannel_chunk_and_ignores_mono():
    host = _RuntimeHost(channels=(0, 2))
    host._begin_streaming_waveform_session(48_000, 0, "forward")
    host._streaming_waveform_session.append = mock.Mock()
    multi = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float64)
    payload = {
        "multi": multi,
        "mono": np.array([99.0, 98.0], dtype=np.float32),
    }

    host.on_audio_chunk_received(payload)

    appended = host._streaming_waveform_session.append.call_args.args[0]
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
    host._begin_streaming_waveform_session(48_000, 0, "forward")
    host._streaming_waveform_session.append = mock.Mock()

    host.on_audio_chunk_received(payload)

    assert writer.chunks == []
    host._streaming_waveform_session.append.assert_not_called()
    host._handle_invalid_recording.assert_called_once()


def test_column_mismatch_routes_once_to_invalid_recording_cleanup_before_persistence():
    host = _RuntimeHost(channels=(0, 2))
    processor = mock.Mock()
    writer = host.streaming_wav_writer
    host.streaming_processor = processor
    host._handle_invalid_recording = mock.Mock()
    host._begin_streaming_waveform_session(48_000, 0, "forward")
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
    single._begin_streaming_waveform_session(48_000, 0, "forward")
    values = np.array([1.0, 2.0], dtype=np.float32)

    single.on_audio_chunk_received(values)

    np.testing.assert_array_equal(single.streaming_wav_writer.chunks[0], values[:, None])

    multi = _RuntimeHost(channels=(0, 2))
    writer = multi.streaming_wav_writer
    multi._handle_invalid_recording = mock.Mock()
    multi._begin_streaming_waveform_session(48_000, 0, "forward")
    multi.on_audio_chunk_received(values)
    assert writer.chunks == []
    multi._handle_invalid_recording.assert_called_once()


def test_each_valid_payload_reaches_wav_writer_exactly_once():
    host = _RuntimeHost(channels=(0, 2))
    host._begin_streaming_waveform_session(48_000, 0, "forward")
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
    host._begin_streaming_waveform_session(48_000, 0, "forward")
    host._streaming_waveform_session.append = mock.Mock()
    multi = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)

    host.on_audio_chunk_received({"multi": multi, "mono": _BadMono()})

    appended = host._streaming_waveform_session.append.call_args.args[0]
    np.testing.assert_array_equal(appended, multi)
    assert host._streaming_waveform_live_enabled is True
    assert len(host.streaming_wav_writer.chunks) == 1
    host.default_logger.error.assert_not_called()


def test_payload_handler_contains_no_historical_concatenation():
    source = inspect.getsource(SequenceWidgetStreamingOpsMixin.on_audio_chunk_received)

    assert "np.concatenate" not in source
    assert "streaming_buffer_multi" not in source


def test_refresh_is_coalesced_and_projects_latest_snapshot():
    host = _RuntimeHost(channels=(0, 2))
    host._begin_streaming_waveform_session(10, 0, "forward")

    host.on_audio_chunk_received(np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32))
    host.on_audio_chunk_received(np.array([[3.0, 30.0], [4.0, 40.0]], dtype=np.float32))

    assert len(host.scheduled) == 1
    assert host.channel_workspace.calls == []
    host.scheduled.pop()()

    assert [call.channel for call in host.channel_workspace.calls] == [0, 2]
    assert all(call.x[-1] == pytest.approx(0.3) for call in host.channel_workspace.calls)
    np.testing.assert_array_equal(host.channel_workspace.calls[0].y, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(host.channel_workspace.calls[1].y, [10.0, 20.0, 30.0, 40.0])


def test_stale_generation_callback_is_a_no_op():
    host = _RuntimeHost()
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    stale_callback = host.scheduled.pop()

    host._recording_input_channels = (1,)
    host.channel_workspace = _Workspace((1,))
    host._begin_streaming_waveform_session(10, 0, "reverse")
    stale_callback()

    assert host.channel_workspace.calls == []


def test_flush_clears_flags_before_snapshot_materialization():
    host = _RuntimeHost()
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    original_snapshots = host._streaming_waveform_session.snapshots

    def snapshots():
        assert host._streaming_waveform_pending is False
        assert host._streaming_waveform_refresh_scheduled is False
        return original_snapshots()

    host._streaming_waveform_session.snapshots = snapshots
    host.scheduled.pop()()

    assert len(host.channel_workspace.calls) == 1


def test_append_memory_failure_disables_only_live_projection_and_releases_buffer():
    host = _RuntimeHost()
    host.default_logger = mock.Mock()
    processor = host.streaming_processor
    completion_owner = object()
    host._streaming_completion_processor = completion_owner
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    accumulator = next(iter(host._streaming_waveform_session._accumulators.values()))
    live_allocation = weakref.ref(accumulator)
    del accumulator
    host._streaming_waveform_session.append = mock.Mock(
        side_effect=MemoryError("live allocation failed")
    )

    host.on_audio_chunk_received(np.array([3.0, 4.0], dtype=np.float32))
    host.on_audio_chunk_received(np.array([5.0, 6.0], dtype=np.float32))
    gc.collect()

    assert len(host.streaming_wav_writer.chunks) == 3
    host._streaming_waveform_session.append.assert_called_once()
    host.default_logger.error.assert_called_once()
    assert "live allocation failed" in host.default_logger.error.call_args.args[0]
    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.channels == ()
    assert host.streaming_processor is processor
    assert host._streaming_completion_processor is completion_owner
    assert host._direction_waveform_cache == {}
    assert live_allocation() is None


@pytest.mark.parametrize("failure_point", ["snapshot", "workspace"])
def test_deferred_failure_disables_only_live_projection_and_releases_buffer(
    failure_point,
):
    host = _RuntimeHost()
    host.default_logger = mock.Mock()
    processor = host.streaming_processor
    completion_owner = object()
    host._streaming_completion_processor = completion_owner
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    accumulator = next(iter(host._streaming_waveform_session._accumulators.values()))
    live_allocation = weakref.ref(accumulator)
    del accumulator
    if failure_point == "snapshot":
        host._streaming_waveform_session.snapshots = mock.Mock(
            side_effect=RuntimeError("snapshot failed")
        )
    else:
        host.channel_workspace._windows[0].set_data = mock.Mock(
            side_effect=RuntimeError("workspace failed")
        )

    host.scheduled.pop()()
    host.on_audio_chunk_received(np.array([3.0, 4.0], dtype=np.float32))
    gc.collect()

    assert len(host.streaming_wav_writer.chunks) == 2
    host.default_logger.error.assert_called_once()
    assert f"{failure_point} failed" in host.default_logger.error.call_args.args[0]
    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.channels == ()
    assert host.streaming_processor is processor
    assert host._streaming_completion_processor is completion_owner
    assert host._direction_waveform_cache == {}
    assert live_allocation() is None
    if failure_point == "snapshot":
        host._streaming_waveform_session.snapshots.assert_called_once_with()
    else:
        host.channel_workspace._windows[0].set_data.assert_called_once()


def test_natural_completion_flushes_tail_before_authoritative_projection_then_releases():
    host = _RuntimeHost()
    events = []

    class _Processor:
        def process_queue(self):
            events.append("processor.process_queue")

    processor = _Processor()
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    stale_callback = host.scheduled.pop()
    accumulator = next(iter(host._streaming_waveform_session._accumulators.values()))
    live_allocation = weakref.ref(accumulator)
    del accumulator
    original_flush = host._flush_streaming_waveform_refresh
    original_end = host._end_streaming_waveform_session
    final_waveform = np.array([7.0, 8.0], dtype=np.float32)

    def flush(generation):
        events.append("live tail flush")
        original_flush(generation)

    def complete():
        events.append("existing _on_streaming_complete final projection")
        host._direction_waveform_cache["forward"] = (final_waveform, 10.0)

    def end():
        events.append("live generation invalidation and buffer release")
        original_end()

    host._flush_streaming_waveform_refresh = flush
    host._on_streaming_complete = complete
    host._end_streaming_waveform_session = end

    host._on_streaming_recording_finished(processor)
    gc.collect()

    assert events == [
        "processor.process_queue",
        "live tail flush",
        "existing _on_streaming_complete final projection",
        "live generation invalidation and buffer release",
    ]
    assert len(host.channel_workspace.calls) == 1
    assert host._direction_waveform_cache["forward"][0] is final_waveform
    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.snapshots() == {}
    assert host._recording_input_channels is None
    assert live_allocation() is None
    stale_callback()
    assert len(host.channel_workspace.calls) == 1


def test_natural_completion_releases_live_state_when_completion_raises():
    host = _RuntimeHost()
    processor = mock.Mock()
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    host._on_streaming_complete = mock.Mock(
        side_effect=RuntimeError("completion escaped")
    )

    with pytest.raises(RuntimeError, match="completion escaped"):
        host._on_streaming_recording_finished(processor)

    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.snapshots() == {}
    assert host._recording_input_channels is None


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
    host._end_streaming_waveform_session = mock.Mock(
        wraps=host._end_streaming_waveform_session
    )
    host._abort_recording_channel_selection = mock.Mock(
        wraps=host._abort_recording_channel_selection
    )
    writer.finalize = mock.Mock(wraps=writer.finalize)
    host._begin_streaming_waveform_session(48_000, 0, "forward")

    host._on_streaming_recording_finished(processor)
    host._on_streaming_recording_finished(processor)

    processor.process_queue.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    host._on_streaming_complete.assert_not_called()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._end_streaming_waveform_session.assert_called_once_with()
    host._abort_recording_channel_selection.assert_called_once_with()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert host._streaming_waveform_session.channels == ()
    assert host.channel_workspace.calls == []


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
    host._begin_streaming_waveform_session(48_000, 0, "forward")

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
    host.clear_all_direction_waveforms = mock.Mock()
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
    host._begin_streaming_waveform_session(48_000, 0, "forward")

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
    host._begin_streaming_waveform_session(48_000, 0, "forward")
    live_append = mock.Mock(wraps=host._streaming_waveform_session.append)
    host._streaming_waveform_session.append = live_append
    chunk = np.ones((4, 2), dtype=np.float32)

    host.on_audio_chunk_received(chunk)
    host.on_audio_chunk_received(chunk)
    host._on_streaming_queue_ready(processor)
    host._on_streaming_recording_finished(processor)
    host._cleanup_streaming_resources()

    writer.write_chunk.assert_called_once_with(chunk)
    live_append.assert_not_called()
    processor.process_queue.assert_not_called()
    processor.stop_streaming.assert_called_once_with()
    writer.finalize.assert_called_once_with()
    host._handle_invalid_recording.assert_called_once()
    assert "disk full" in host._handle_invalid_recording.call_args.args[0]
    host._abort_recording_channel_selection.assert_called_once_with()
    host._finalize_recording_channel_selection.assert_not_called()
    host._discard_current_recent_session.assert_called_once_with()
    host.clear_all_direction_waveforms.assert_called_once_with()
    host._on_streaming_complete.assert_not_called()
    validation.assert_not_called()
    database.save_signal_info_to_db.assert_not_called()
    assert not failed_path.exists()
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert host._streaming_waveform_session.channels == ()
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
    host._begin_streaming_waveform_session(48_000, 0, "forward")

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
    host.clear_all_direction_waveforms.assert_called_once_with()
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
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    stale_callback = host.scheduled.pop()
    original_end = host._end_streaming_waveform_session

    def end():
        events.append("live")
        original_end()

    host._end_streaming_waveform_session = end

    host._cleanup_streaming_resources()
    stale_callback()

    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.snapshots() == {}
    assert host._recording_input_channels is None
    assert host.channel_workspace.calls == []
    assert events == ["live", "processor", "writer"]
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
    ["live-query", "live-end", "channel-query", "channel-abort"],
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
    host._begin_streaming_waveform_session(48_000, 0, "forward")
    failure = RuntimeError(f"{failure_point} failed")
    if failure_point == "live-query":
        host._streaming_waveform_session_is_active = mock.Mock(side_effect=failure)
    elif failure_point == "live-end":
        host._end_streaming_waveform_session = mock.Mock(side_effect=failure)
    elif failure_point == "channel-query":
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


def test_invalid_recording_invalidates_pending_live_callback():
    host = _RuntimeHost()
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    stale_callback = host.scheduled.pop()
    host._serial_product_condition_executing = True
    host._on_serial_product_runtime_error = mock.Mock(return_value=True)

    host._handle_invalid_recording("empty audio")
    stale_callback()

    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.snapshots() == {}
    assert host._recording_input_channels is None
    assert host.channel_workspace.calls == []


def test_close_invalidates_pending_live_callback():
    class _CloseBase:
        def closeEvent(self, event):
            self.closed_event = event

    class _CloseHost(_RuntimeHost, _CloseBase):
        def isVisible(self):
            return False

    host = _CloseHost()
    events = []
    processor = mock.Mock()
    processor.stop_streaming.side_effect = lambda: events.append("processor")
    host.streaming_processor = processor
    host.streaming_wav_writer.finalize = mock.Mock(
        side_effect=lambda: events.append("writer")
    )
    host._begin_streaming_waveform_session(10, 0, "forward")
    host.on_audio_chunk_received(np.array([1.0, 2.0], dtype=np.float32))
    stale_callback = host.scheduled.pop()
    active_generation = host._streaming_waveform_generation
    original_end = host._end_streaming_waveform_session
    original_abort = host._abort_recording_channel_selection

    def end():
        original_end()
        events.append(
            (
                "invalidate-live",
                host._streaming_waveform_generation,
                host._streaming_waveform_session.channels,
                host._recording_input_channels,
            )
        )

    def abort():
        events.append(
            (
                "clear-run-snapshot",
                host._streaming_waveform_generation,
                host._streaming_waveform_session.channels,
            )
        )
        original_abort()

    host._end_streaming_waveform_session = end
    host._abort_recording_channel_selection = abort
    event = object()

    host.closeEvent(event)
    stale_callback()

    assert host.closed_event is event
    assert host._streaming_waveform_live_enabled is False
    assert host._streaming_waveform_session.snapshots() == {}
    assert host._recording_input_channels is None
    assert host.channel_workspace.calls == []
    assert events == [
        ("invalidate-live", active_generation + 1, (), (0,)),
        ("clear-run-snapshot", active_generation + 1, ()),
        "processor",
        "writer",
    ]
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None


def test_coalesced_600_second_delivery_reaches_live_and_final_endpoints(
    monkeypatch,
):
    sample_rate = 1_000
    trim = 137
    display_samples = sample_rate * 600
    total = display_samples + trim
    expected_endpoint = (display_samples - 1) / sample_rate
    host = _RuntimeHost()
    host.default_logger = mock.Mock()
    host._begin_streaming_waveform_session(sample_rate, trim, "forward")
    raw_mono = np.sin(np.arange(total, dtype=np.float32) / 17.0)
    raw_multi = raw_mono.reshape(-1, 1)
    source_chunks = []
    allow_authoritative_reduction = False
    original_prepare = host._prepare_waveform_display_data

    def reject_live_full_history_reduction(waveform, rate):
        if not allow_authoritative_reduction:
            raise AssertionError("live refresh used the legacy full-history reducer")
        return original_prepare(waveform, rate)

    legacy_prepare = mock.Mock(side_effect=reject_live_full_history_reduction)
    host._prepare_waveform_display_data = legacy_prepare

    for chunk_index, start in enumerate(range(0, total, 2_048), start=1):
        stop = min(start + 2_048, total)
        mono = raw_mono[start:stop]
        multi = raw_multi[start:stop]
        source_chunks.append(multi)

        assert len(host.scheduled) <= 1
        host.on_audio_chunk_received({"mono": mono, "multi": multi})
        assert len(host.scheduled) == 1
        if chunk_index % 19 == 0:
            host.scheduled.pop()()
            assert len(host.scheduled) == 0
            assert host._streaming_waveform_live_enabled is True
            legacy_prepare.assert_not_called()

    if host.scheduled:
        host.scheduled.pop()()
    assert host._streaming_waveform_live_enabled is True
    assert host._streaming_waveform_pending is False
    legacy_prepare.assert_not_called()
    live_call = host.channel_workspace.calls[-1]
    assert live_call.x[-1] == pytest.approx(expected_endpoint)
    assert len(live_call.x) <= host._WAVEFORM_DISPLAY_MAX_POINTS

    writer = host.streaming_wav_writer
    assert len(writer.chunks) == len(source_chunks)
    for written, source in zip(writer.chunks, source_chunks):
        np.testing.assert_array_equal(written, source)
    assert all(
        len(call.x) <= host._WAVEFORM_DISPLAY_MAX_POINTS
        for call in host.channel_workspace.calls
    )

    class _Processor:
        target_samples = total

        def __init__(self):
            self.process_queue_calls = 0

        def process_queue(self):
            self.process_queue_calls += 1

        def get_recorded_data(self):
            return raw_mono

        def get_recorded_data_multi(self):
            return raw_multi

    processor = _Processor()
    host.streaming_processor = processor
    host._streaming_completion_processor = None
    host.data_struct = SimpleNamespace(
        sample_rate=sample_rate,
        store_wave_data=None,
        store_wave_data_multi=None,
    )
    host.recorded_signal_info = {"labels": "not_labeled"}
    host.recorded_path = "synthetic-600s.wav"
    host.streaming_stimulus_data = object()
    host.streaming_mode = "record_only"
    host.player_status_flag = True
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host.count_board = SimpleNamespace(mode="")
    host.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
    host._condition_record_cache = {}
    host._waveform_condition_keys = lambda: ["forward"]
    host._resolve_active_recording_waveform_direction = (
        lambda fallback="": "forward"
    )
    host._resolve_recording_acq_detail = lambda: {"startup_trim_ms": trim}
    host._rewrite_recorded_wav = mock.Mock()
    host._append_recording_wav_calibration_metadata = mock.Mock(
        return_value=True
    )
    host._update_current_recent_session_result = mock.Mock()
    host._should_run_silent_analysis_after_recording = lambda: False
    host.update_player_btn_is_paused = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()

    validation = mock.Mock(return_value=(True, "", ""))
    database = mock.Mock()
    database.save_signal_info_to_db.return_value = (
        sequence_widget_streaming_ops.error_code.OK,
        "saved",
    )
    monkeypatch.setattr(
        sequence_widget_streaming_ops,
        "validate_recorded_audio",
        validation,
    )
    monkeypatch.setattr(
        sequence_widget_streaming_ops,
        "RecordingManager",
        lambda: database,
    )

    calls_before_completion = len(host.channel_workspace.calls)
    allow_authoritative_reduction = True

    host._on_streaming_recording_finished(processor)

    assert processor.process_queue_calls == 1
    assert len(host.channel_workspace.calls) == calls_before_completion + 1
    authoritative_call = host.channel_workspace.calls[-1]
    assert authoritative_call.x[-1] == pytest.approx(expected_endpoint)
    assert 600 - authoritative_call.x[-1] <= 1 / sample_rate
    assert all(
        len(call.x) <= host._WAVEFORM_DISPLAY_MAX_POINTS
        for call in host.channel_workspace.calls
    )
    assert legacy_prepare.call_count == 1
    assert host._direction_waveform_cache == {}
    assert host._condition_record_cache["forward"] == {
        "recorded_path": "synthetic-600s.wav",
        "recorded_signal_info": {
            "labels": "not_labeled",
            "sample_rate": sample_rate,
        },
        "session_id": "",
    }
    expected_time, expected_display = original_prepare(
        raw_mono[trim:],
        sample_rate,
    )
    np.testing.assert_array_equal(authoritative_call.x, expected_time)
    np.testing.assert_array_equal(authoritative_call.y, expected_display)
    assert host.data_struct.store_wave_data_multi.shape == (display_samples, 1)
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi[:, 0],
        raw_mono[trim:],
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data,
        raw_mono[trim:],
    )
    validation.assert_called_once()
    assert validation.call_args.args[0].shape == (display_samples, 1)
    host._rewrite_recorded_wav.assert_called_once()
    host._append_recording_wav_calibration_metadata.assert_called_once_with()
    database.save_signal_info_to_db.assert_called_once_with(
        host.recorded_signal_info,
        None,
    )
    assert len(writer.chunks) == len(source_chunks)
    assert writer.finalized is True
    assert host.streaming_wav_writer is None
    assert host.streaming_processor is None
    assert host.recorded_signal_info["sample_rate"] == sample_rate
    assert host._awaiting_ok_ng is True
    assert any(
        "Streaming recording completed successfully" in call.args[0]
        for call in host.default_logger.info.call_args_list
    )
    assert not any(
        "Error in streaming completion" in call.args[0]
        for call in host.default_logger.error.call_args_list
    )
    assert host.scheduled == []
    assert host._streaming_waveform_session.snapshots() == {}
