import queue
import threading
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pytest

from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code


@pytest.fixture(autouse=True)
def isolated_logger():
    logger = SimpleNamespace(
        info=mock.Mock(),
        warning=mock.Mock(),
        error=mock.Mock(),
    )
    with mock.patch(
        "base.streaming_audio_processor.LogManager.set_log_handler",
        return_value=logger,
    ):
        yield logger


class _SignalProbe:
    def __init__(self):
        self.emit = mock.Mock()


def _fake_streaming_signals():
    return SimpleNamespace(
        stream_audio_chunk_signal=_SignalProbe(),
        stream_audio_queue_ready_signal=_SignalProbe(),
        stream_audio_recording_finished_signal=_SignalProbe(),
    )


class _FakeInputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _ImmediateThread:
    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


def test_each_successful_enqueue_emits_one_source_only_wakeup():
    processor = StreamingAudioProcessor()
    processor.target_samples = 10
    fake_sign = _fake_streaming_signals()

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        first_payload, first_reached = processor._queue_chunk_and_maybe_stop(
            np.ones((2, 2), dtype=np.float32)
        )
        second_payload, second_reached = processor._queue_chunk_and_maybe_stop(
            np.full((2, 2), 2.0, dtype=np.float32)
        )

    assert first_reached is False
    assert second_reached is False
    assert set(first_payload) == {"mono", "multi"}
    assert set(second_payload) == {"mono", "multi"}
    queued_payloads = [
        processor.audio_queue.get_nowait(),
        processor.audio_queue.get_nowait(),
    ]
    assert queued_payloads[0] is first_payload
    assert queued_payloads[1] is second_payload
    fake_sign.stream_audio_queue_ready_signal.emit.assert_has_calls(
        [mock.call(processor), mock.call(processor)]
    )
    assert fake_sign.stream_audio_queue_ready_signal.emit.call_count == 2


def test_full_queue_drops_chunk_without_wakeup(isolated_logger):
    processor = StreamingAudioProcessor()
    processor.target_samples = 10
    processor.audio_queue = queue.Queue(maxsize=1)
    processor.audio_queue.put_nowait({"existing": True})
    fake_sign = _fake_streaming_signals()

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor._queue_chunk_and_maybe_stop(np.ones((2, 1), dtype=np.float32))

    fake_sign.stream_audio_queue_ready_signal.emit.assert_not_called()
    assert any(
        "Audio queue full" in call.args[0]
        for call in isolated_logger.warning.call_args_list
    )


def test_record_only_callback_emits_queue_ready_for_processor():
    processor = StreamingAudioProcessor()
    processor.target_samples = 10
    fake_sign = _fake_streaming_signals()

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor._audio_callback(
            np.array([[1.0], [2.0]], dtype=np.float32), 2, None, None
        )

    fake_sign.stream_audio_queue_ready_signal.emit.assert_called_once_with(processor)


def test_monitored_duplex_callback_emits_queue_ready_for_processor():
    created_streams = []

    def create_stream(**kwargs):
        stream = _FakeInputStream(**kwargs)
        created_streams.append(stream)
        return stream

    processor = StreamingAudioProcessor()
    fake_sign = _fake_streaming_signals()

    with (
        mock.patch("base.streaming_audio_processor.sign", fake_sign),
        mock.patch("base.streaming_audio_processor.sd.Stream", side_effect=create_stream),
    ):
        code, _ = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device={"index": 7, "max_input_channels": 1},
            input_channels=[0],
            output_device={"index": 8, "max_output_channels": 2},
            output_channels=[0, 1],
            monitor_playback=True,
        )
        callback = created_streams[0].kwargs["callback"]
        callback(
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.empty((2, 2), dtype=np.float32),
            2,
            None,
            None,
        )

    assert code == error_code.OK
    fake_sign.stream_audio_queue_ready_signal.emit.assert_called_once_with(processor)


def test_automatic_completion_stops_closes_then_emits_exactly_once():
    events = []
    processor = StreamingAudioProcessor()
    processor.target_samples = 2
    processor.is_recording = True
    processor.stream = SimpleNamespace(
        stop=lambda: events.append("stop"),
        close=lambda: events.append("close"),
    )
    fake_sign = _fake_streaming_signals()
    fake_sign.stream_audio_recording_finished_signal.emit.side_effect = (
        lambda source: events.append("finished")
    )

    with (
        mock.patch("base.streaming_audio_processor.sign", fake_sign),
        mock.patch("base.streaming_audio_processor.threading.Thread", _ImmediateThread),
    ):
        processor._queue_chunk_and_maybe_stop(np.ones((2, 1), dtype=np.float32))
        processor._queue_chunk_and_maybe_stop(np.ones((1, 1), dtype=np.float32))

    assert events == ["stop", "close", "finished"]
    fake_sign.stream_audio_recording_finished_signal.emit.assert_called_once_with(
        processor
    )
    assert processor.is_recording is False
    assert processor.stream is None


def test_manual_stop_does_not_emit_normal_recording_finished():
    processor = StreamingAudioProcessor()
    processor.stream = _FakeInputStream()
    processor.is_recording = True
    fake_sign = _fake_streaming_signals()

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor.stop_streaming()

    fake_sign.stream_audio_recording_finished_signal.emit.assert_not_called()
    assert processor.is_recording is False
    assert processor.stream is None


def test_cleanup_attempts_all_owned_operations_when_input_stop_fails(
    isolated_logger,
):
    events = []
    processor = StreamingAudioProcessor()

    def failing_input_stop():
        events.append("input_stop")
        raise OSError("input stop failed")

    processor.stream = SimpleNamespace(
        stop=failing_input_stop,
        close=lambda: events.append("input_close"),
    )
    processor.output_stream = SimpleNamespace(
        stop=lambda: events.append("output_stop"),
        close=lambda: events.append("output_close"),
    )

    processor.stop_streaming()

    assert events == [
        "input_stop",
        "input_close",
        "output_stop",
        "output_close",
    ]
    assert processor.stream is None
    assert processor.output_stream is None
    assert any(
        "input stop failed" in call.args[0]
        for call in isolated_logger.error.call_args_list
    )


def test_manual_stop_waits_for_in_progress_automatic_cleanup():
    events = []
    automatic_stop_entered = threading.Event()
    release_automatic_stop = threading.Event()
    manual_returned = threading.Event()
    processor = StreamingAudioProcessor()
    fake_sign = _fake_streaming_signals()

    def blocking_stop():
        events.append("stop")
        automatic_stop_entered.set()
        assert release_automatic_stop.wait(timeout=2)

    processor.stream = SimpleNamespace(
        stop=blocking_stop,
        close=lambda: events.append("close"),
    )

    def manual_stop():
        processor.stop_streaming()
        manual_returned.set()

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        automatic_thread = threading.Thread(target=processor._stop_after_target)
        automatic_thread.start()
        assert automatic_stop_entered.wait(timeout=2)
        manual_thread = threading.Thread(target=manual_stop)
        manual_thread.start()
        try:
            assert manual_returned.wait(timeout=0.2) is False
        finally:
            release_automatic_stop.set()
            automatic_thread.join(timeout=2)
            manual_thread.join(timeout=2)

    assert automatic_thread.is_alive() is False
    assert manual_thread.is_alive() is False
    assert manual_returned.is_set()
    assert events == ["stop", "close"]
    fake_sign.stream_audio_recording_finished_signal.emit.assert_not_called()


def test_manual_stop_cancels_completion_after_close_before_completion_lock():
    processor = StreamingAudioProcessor()
    fake_sign = _fake_streaming_signals()
    close_returned = threading.Event()

    class _CompletionBoundaryLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.automatic_thread = None
            self.automatic_waiting = threading.Event()
            self.release_automatic = threading.Event()

        def __enter__(self):
            if (
                threading.current_thread() is self.automatic_thread
                and close_returned.is_set()
            ):
                self.automatic_waiting.set()
                assert self.release_automatic.wait(timeout=2)
            self.lock.acquire()

        def __exit__(self, exc_type, exc_value, traceback):
            self.lock.release()

    completion_lock = _CompletionBoundaryLock()
    processor._completion_lock = completion_lock

    def controlled_close():
        close_returned.set()

    processor._close_stream_resources = controlled_close

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        automatic_thread = threading.Thread(target=processor._stop_after_target)
        completion_lock.automatic_thread = automatic_thread
        automatic_thread.start()
        assert completion_lock.automatic_waiting.wait(timeout=2)
        processor.stop_streaming()
        completion_lock.release_automatic.set()
        automatic_thread.join(timeout=2)

    assert automatic_thread.is_alive() is False
    fake_sign.stream_audio_recording_finished_signal.emit.assert_not_called()


def test_single_physical_input_channel_is_selected_and_accumulated():
    created_streams = []

    def create_stream(**kwargs):
        stream = _FakeInputStream(**kwargs)
        created_streams.append(stream)
        return stream

    processor = StreamingAudioProcessor()
    input_device = {
        "index": 7,
        "name": "Two Channel Input",
        "hostapi": 1,
        "max_input_channels": 2,
    }

    with mock.patch(
        "base.streaming_audio_processor.sd.InputStream",
        side_effect=create_stream,
    ):
        code, _ = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device=input_device,
            input_channels=[1],
        )

    assert code == error_code.OK
    assert processor._rec_in_sel == [1]
    assert created_streams[0].kwargs["channels"] == 2
    assert created_streams[0].kwargs["device"] == 7

    callback = created_streams[0].kwargs["callback"]
    callback(
        np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        2,
        None,
        None,
    )
    emit = mock.Mock()
    fake_sign = SimpleNamespace(
        stream_audio_chunk_signal=SimpleNamespace(emit=emit)
    )
    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor.process_queue(emit_signal=False)

    np.testing.assert_array_equal(
        processor.get_recorded_data(),
        np.array([10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        processor.get_recorded_data_multi(),
        np.array([[10.0], [20.0]], dtype=np.float32),
    )
    emit.assert_not_called()
    processor.stop_streaming()


def test_process_queue_keeps_legacy_array_payload_compatible():
    processor = StreamingAudioProcessor()
    processor.audio_queue.put_nowait(np.array([1.0, 2.0], dtype=np.float32))

    processor.process_queue(emit_signal=False)

    np.testing.assert_array_equal(
        processor.get_recorded_data(),
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert processor.get_recorded_data_multi().shape == (2, 1)


def test_process_queue_still_emits_waveform_payload_by_default():
    processor = StreamingAudioProcessor()
    processor.audio_queue.put_nowait(np.array([1.0, 2.0], dtype=np.float32))
    emit = mock.Mock()
    fake_sign = SimpleNamespace(
        stream_audio_chunk_signal=SimpleNamespace(emit=emit)
    )

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor.process_queue()

    emitted_payload = emit.call_args.args[0]
    np.testing.assert_array_equal(
        emitted_payload["mono"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert emitted_payload["multi"].shape == (2, 1)


def test_invalid_selected_channel_is_rejected_before_opening_stream():
    processor = StreamingAudioProcessor()
    input_device = {
        "index": 7,
        "name": "Two Channel Input",
        "hostapi": 1,
        "max_input_channels": 2,
    }

    with mock.patch("base.streaming_audio_processor.sd.InputStream") as input_stream:
        code, message = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device=input_device,
            input_channels=[2],
        )

    assert code == error_code.INVALID_RECORD
    assert "Invalid input_channels" in message
    assert processor.is_recording is False
    input_stream.assert_not_called()


def test_stream_open_failure_resets_recording_state():
    processor = StreamingAudioProcessor()
    fake_sign = _fake_streaming_signals()

    with (
        mock.patch("base.streaming_audio_processor.sign", fake_sign),
        mock.patch(
            "base.streaming_audio_processor.sd.InputStream",
            side_effect=OSError("device unavailable"),
        ),
    ):
        code, message = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device={"index": 7, "max_input_channels": 1},
            input_channels=[0],
        )
        processor.stop_streaming()

    assert code == error_code.INVALID_RECORD
    assert "device unavailable" in message
    assert processor.is_recording is False
    assert processor.error_occurred is True
    fake_sign.stream_audio_queue_ready_signal.emit.assert_not_called()
    fake_sign.stream_audio_recording_finished_signal.emit.assert_not_called()


@pytest.mark.parametrize(
    ("monitor_playback", "stream_factory_name", "extra_kwargs"),
    [
        (False, "InputStream", {}),
        (
            True,
            "Stream",
            {
                "output_device": {"index": 8, "max_output_channels": 2},
                "output_channels": [0, 1],
            },
        ),
    ],
    ids=["input", "duplex"],
)
def test_stream_start_failure_synchronously_cleans_partial_stream_without_messages(
    monitor_playback,
    stream_factory_name,
    extra_kwargs,
):
    events = []
    fake_sign = _fake_streaming_signals()
    processor = StreamingAudioProcessor()

    class _StartFailingStream:
        def start(self):
            events.append("start")
            raise OSError(f"{stream_factory_name} start failed")

        def stop(self):
            events.append("stop")

        def close(self):
            events.append("close")

    stream = _StartFailingStream()
    with (
        mock.patch("base.streaming_audio_processor.sign", fake_sign),
        mock.patch(
            f"base.streaming_audio_processor.sd.{stream_factory_name}",
            return_value=stream,
        ),
    ):
        code, message = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device={"index": 7, "max_input_channels": 1},
            input_channels=[0],
            monitor_playback=monitor_playback,
            **extra_kwargs,
        )

    assert code == error_code.INVALID_RECORD
    assert f"{stream_factory_name} start failed" in message
    assert events == ["start", "stop", "close"]
    assert processor.stream is None
    assert processor.is_recording is False
    assert processor.error_occurred is True
    assert processor.error_message == f"{stream_factory_name} start failed"
    fake_sign.stream_audio_queue_ready_signal.emit.assert_not_called()
    fake_sign.stream_audio_recording_finished_signal.emit.assert_not_called()
