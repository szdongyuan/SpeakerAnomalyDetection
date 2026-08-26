import math
import threading
import time

import numpy as np
import pytest

from base import streaming_audio_processor
from base.streaming_audio_processor import (
    AudioFinalizationPending,
    ProducerGateState,
    StreamingAudioProcessor,
)
from consts import error_code
from ui.sequence.sequence_messages import (
    AudioBatch,
    AudioCancelled,
    AudioCompleted,
    AudioFailed,
    AudioBatch as CanonicalAudioBatch,
    AudioCancelled as CanonicalAudioCancelled,
    AudioCompleted as CanonicalAudioCompleted,
    AudioFailed as CanonicalAudioFailed,
)


class _HostileBoundaryError(BaseException):
    def __str__(self):
        raise KeyboardInterrupt("hostile __str__")


class _HostileException(Exception):
    def __str__(self):
        raise KeyboardInterrupt("hostile __str__")


def _configured_processor(*, target_samples=4, channels=(1, 3), block_size=2):
    processor = StreamingAudioProcessor()
    processor.configure_event_session(
        session_id="session-1",
        sample_rate=48_000,
        target_samples=target_samples,
        callback_block_size=block_size,
        channel_order=channels,
    )
    return processor


def test_streaming_processor_exposes_only_canonical_audio_fifo_message_classes():
    assert streaming_audio_processor.AudioBatch is CanonicalAudioBatch
    assert streaming_audio_processor.AudioCompleted is CanonicalAudioCompleted
    assert streaming_audio_processor.AudioFailed is CanonicalAudioFailed
    assert streaming_audio_processor.AudioCancelled is CanonicalAudioCancelled

    processor = _configured_processor(target_samples=2, channels=(0,))
    processor._audio_callback(
        np.array([[1.0], [2.0]], dtype=np.float32), 2, None, None
    )
    batch = processor.audio_queue.get(timeout=1)
    terminal = processor.audio_queue.get(timeout=1)

    assert type(batch) is CanonicalAudioBatch
    assert type(terminal) is CanonicalAudioCompleted
    assert batch["multi"].flags.writeable is False
    assert batch["mono"].flags.writeable is False


def test_callback_enqueues_copied_read_only_monotonic_batches_before_completion():
    processor = _configured_processor()
    first = np.array([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float32)
    second = np.array([[2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]], dtype=np.float32)

    processor._audio_callback(first, 2, None, None)
    processor._audio_callback(second, 2, None, None)

    first[:, :] = 99
    second[:, :] = 88
    messages = [processor.audio_queue.get(timeout=1) for _ in range(3)]
    assert [type(message) for message in messages] == [AudioBatch, AudioBatch, AudioCompleted]
    assert [(message.sequence_no, message.sample_start, message.sample_stop) for message in messages[:2]] == [
        (0, 0, 2),
        (1, 2, 4),
    ]
    assert messages[0].channel_order == (1, 3)
    np.testing.assert_allclose(messages[0].multi, [[0.1, 0.3], [1.1, 1.3]])
    assert messages[0].multi.flags.writeable is False
    assert messages[0].mono.flags.writeable is False
    assert messages[2] == AudioCompleted("session-1", 1, 4)
    assert processor.gate_state is ProducerGateState.TERMINAL_ENQUEUED


def test_terminal_claim_waits_for_active_callback_and_rejects_post_terminal_audio(monkeypatch):
    processor = _configured_processor(target_samples=10)
    entered = threading.Event()
    release = threading.Event()
    original_factory = processor._audio_batch_from_callback

    def blocking_factory(**payload):
        entered.set()
        assert release.wait(1)
        return original_factory(**payload)

    monkeypatch.setattr(
        processor, "_audio_batch_from_callback", blocking_factory
    )
    callback = threading.Thread(
        target=processor._audio_callback,
        args=(np.ones((2, 4), dtype=np.float32), 2, None, None),
    )
    callback.start()
    assert entered.wait(1)

    terminal_done = threading.Event()
    terminal = threading.Thread(
        target=lambda: (processor.complete_streaming(), terminal_done.set())
    )
    terminal.start()
    time.sleep(0.02)
    assert terminal_done.is_set() is False
    assert processor.gate_state is ProducerGateState.QUIESCING

    release.set()
    callback.join(1)
    terminal.join(1)
    assert terminal_done.is_set()
    assert isinstance(processor.audio_queue.get_nowait(), AudioBatch)
    assert isinstance(processor.audio_queue.get_nowait(), AudioCompleted)

    processor._audio_callback(np.ones((2, 4), dtype=np.float32), 2, None, None)
    assert processor.audio_queue.empty()


def test_complete_fail_cancel_race_enqueues_exactly_one_terminal_message():
    processor = _configured_processor(target_samples=10)
    barrier = threading.Barrier(4)
    outcomes = []

    def claim(callback):
        barrier.wait()
        outcomes.append(callback())

    threads = [
        threading.Thread(target=claim, args=(processor.complete_streaming,)),
        threading.Thread(target=claim, args=(lambda: processor.fail_streaming("device", "gone"),)),
        threading.Thread(target=claim, args=(lambda: processor.cancel_streaming("closed"),)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    assert sum(result is True for result in outcomes) == 1
    assert isinstance(
        processor.audio_queue.get_nowait(),
        (AudioCompleted, AudioFailed, AudioCancelled),
    )
    assert processor.audio_queue.empty()


def test_concurrent_callback_entries_still_assign_monotonic_nonoverlapping_ranges():
    processor = _configured_processor(target_samples=8, channels=(0,), block_size=2)
    barrier = threading.Barrier(5)

    def callback(value):
        barrier.wait()
        processor._audio_callback(
            np.full((2, 1), value, dtype=np.float32), 2, None, None
        )

    threads = [threading.Thread(target=callback, args=(value,)) for value in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    messages = [processor.audio_queue.get(timeout=1) for _ in range(5)]
    batches = messages[:4]
    assert [batch.sequence_no for batch in batches] == [0, 1, 2, 3]
    assert [(batch.sample_start, batch.sample_stop) for batch in batches] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    assert messages[-1] == AudioCompleted("session-1", 3, 8)


@pytest.mark.parametrize(
    ("sample_rate", "target_samples", "duration", "block_size"),
    [
        (0, 8, None, 2),
        (math.inf, 8, None, 2),
        (48_000, 0, None, 2),
        (48_000, None, 0, 2),
        (48_000, None, math.nan, 2),
        (48_000, 8, None, 0),
    ],
)
def test_invalid_finite_admission_is_rejected_before_device_open(
    monkeypatch, sample_rate, target_samples, duration, block_size
):
    opened = []
    monkeypatch.setattr(
        streaming_audio_processor.sd,
        "InputStream",
        lambda **_kwargs: opened.append(True),
    )
    processor = StreamingAudioProcessor()

    code, _message = processor.start_streaming_rec(
        sample_rate=sample_rate,
        target_samples=target_samples,
        duration=duration,
        device={"index": 1, "max_input_channels": 1},
        callback_block_size=block_size,
        session_id="invalid",
    )

    assert code == error_code.INVALID_RECORD
    assert opened == []
    assert processor.is_recording is False


def test_callback_allocation_failure_is_a_fatal_terminal_failure(monkeypatch):
    processor = _configured_processor(target_samples=10, channels=(0,))

    def fail_factory(**_payload):
        raise MemoryError("allocation exhausted")

    monkeypatch.setattr(processor, "_audio_batch_from_callback", fail_factory)
    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    failure = processor.audio_queue.get(timeout=1)
    assert isinstance(failure, AudioFailed)
    assert failure.error_code == "allocation-failed"
    assert "allocation exhausted" in failure.message
    assert processor.error_occurred is True
    assert processor.is_recording is False
    assert processor.audio_queue.empty()


def test_event_batches_preserve_selected_channel_order_and_float64_dtype():
    processor = _configured_processor(target_samples=2)
    processor.sample_dtype = np.dtype("float64")
    payload = np.array(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float32
    )

    processor._audio_callback(payload, 2, None, None)

    batch = processor.audio_queue.get(timeout=1)
    assert batch.multi.dtype == np.float64
    assert batch.mono.dtype == np.float64
    assert batch.channel_order == (1, 3)
    np.testing.assert_allclose(batch.multi, [[0.1, 0.3], [1.1, 1.3]])


def test_target_callback_closes_gate_before_delayed_finalizer_is_scheduled(monkeypatch):
    processor = _configured_processor(target_samples=2, channels=(0,), block_size=2)
    finalizers = []
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
        raising=False,
    )

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    assert processor.gate_state is ProducerGateState.QUIESCING
    assert processor.samples_captured == 2
    assert len(finalizers) == 1
    processor._audio_callback(np.full((2, 1), 9.0, dtype=np.float32), 2, None, None)
    assert processor.samples_captured == 2
    assert len(finalizers) == 1
    batch = processor.audio_queue.get_nowait()
    assert (batch.sequence_no, batch.sample_start, batch.sample_stop) == (0, 0, 2)
    assert processor.audio_queue.empty()

    finalizers.pop()()
    assert processor.audio_queue.get_nowait() == AudioCompleted("session-1", 0, 2)


def test_concurrent_entered_callbacks_crop_to_exact_remaining_target(monkeypatch):
    processor = _configured_processor(target_samples=3, channels=(0,), block_size=2)
    finalizers = []
    selected = threading.Barrier(3)
    original_select = processor._select_multi

    def delayed_select(*args, **kwargs):
        value = original_select(*args, **kwargs)
        selected.wait(timeout=1)
        return value

    monkeypatch.setattr(processor, "_select_multi", delayed_select)
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
        raising=False,
    )
    callbacks = [
        threading.Thread(
            target=processor._audio_callback,
            args=(np.full((2, 1), value, dtype=np.float32), 2, None, None),
        )
        for value in (1.0, 2.0)
    ]
    for callback in callbacks:
        callback.start()
    selected.wait(timeout=1)
    for callback in callbacks:
        callback.join(timeout=1)

    assert processor.samples_captured == 3
    assert processor.gate_state is ProducerGateState.QUIESCING
    assert len(finalizers) == 1
    batches = [processor.audio_queue.get_nowait(), processor.audio_queue.get_nowait()]
    assert [(item.sequence_no, item.sample_start, item.sample_stop) for item in batches] == [
        (0, 0, 2),
        (1, 2, 3),
    ]
    assert processor.audio_queue.empty()
    finalizers.pop()()
    assert processor.audio_queue.get_nowait() == AudioCompleted("session-1", 1, 3)


def test_fatal_allocation_claims_gate_synchronously_before_delayed_finalizer(monkeypatch):
    processor = _configured_processor(target_samples=4, channels=(0,), block_size=2)
    finalizers = []
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
        raising=False,
    )
    monkeypatch.setattr(
        processor,
        "_audio_batch_from_callback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("oom")),
    )

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    assert processor.gate_state is ProducerGateState.QUIESCING
    assert processor.samples_captured == 0
    assert len(finalizers) == 1
    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)
    assert len(finalizers) == 1
    assert processor.audio_queue.empty()
    finalizers.pop()()
    failure = processor.audio_queue.get_nowait()
    assert isinstance(failure, AudioFailed)
    assert failure.error_code == "allocation-failed"
    assert failure.last_sequence_no == -1


def test_fatal_allocation_claim_linearizes_before_already_entered_callback(monkeypatch):
    processor = _configured_processor(target_samples=4, channels=(0,), block_size=2)
    copy_entered = threading.Event()
    release_copy = threading.Event()
    request_entered = threading.Event()
    release_request = threading.Event()
    finalizers = []
    original_request = processor._request_callback_failure
    original_factory = processor._audio_batch_from_callback
    first_copy = True

    def fail_first_copy(*args, **kwargs):
        nonlocal first_copy
        if not first_copy:
            return original_factory(*args, **kwargs)
        first_copy = False
        copy_entered.set()
        assert release_copy.wait(1)
        raise MemoryError("oom")

    def delay_outer_failure_request(code, error):
        request_entered.set()
        assert release_request.wait(1)
        original_request(code, error)

    monkeypatch.setattr(
        processor, "_audio_batch_from_callback", fail_first_copy
    )
    monkeypatch.setattr(
        processor, "_request_callback_failure", delay_outer_failure_request
    )
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
    )
    first = threading.Thread(
        target=processor._audio_callback,
        args=(np.ones((2, 1), dtype=np.float32), 2, None, None),
    )
    second = threading.Thread(
        target=processor._audio_callback,
        args=(np.ones((2, 1), dtype=np.float32), 2, None, None),
    )
    first.start()
    assert copy_entered.wait(1)
    second.start()
    deadline = time.monotonic() + 1
    while processor._active_callback_count != 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert processor._active_callback_count == 2
    release_copy.set()
    assert request_entered.wait(1)
    assert processor.gate_state is ProducerGateState.QUIESCING
    second.join(timeout=1)
    assert not second.is_alive()
    release_request.set()
    first.join(timeout=1)
    assert not first.is_alive()

    assert processor.samples_captured == 2
    assert processor.gate_state is ProducerGateState.QUIESCING
    batch = processor.audio_queue.get_nowait()
    assert (batch.sequence_no, batch.sample_start, batch.sample_stop) == (0, 0, 2)
    assert processor.audio_queue.empty()
    assert len(finalizers) == 1
    finalizers.pop()()
    failure = processor.audio_queue.get_nowait()
    assert isinstance(failure, AudioFailed)
    assert failure.last_sequence_no == 0


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf])
def test_non_none_duration_is_validated_even_with_explicit_target(duration):
    processor = StreamingAudioProcessor()

    with pytest.raises(ValueError, match="duration must be positive and finite"):
        processor.configure_event_session(
            session_id="session-1",
            sample_rate=48_000,
            target_samples=4,
            duration=duration,
            callback_block_size=2,
            channel_order=(0,),
        )


class _StopCloseStream:
    def __init__(self, *, stop_errors=(), close_errors=()):
        self.stop_errors = list(stop_errors)
        self.close_errors = list(close_errors)
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        if self.stop_errors:
            raise self.stop_errors.pop(0)

    def close(self):
        self.calls.append("close")
        if self.close_errors:
            raise self.close_errors.pop(0)


def test_stop_failure_with_successful_close_is_truthfully_quiesced():
    processor = _configured_processor(target_samples=10, channels=(0,))
    stream = _StopCloseStream(stop_errors=(RuntimeError("stop failed"),))
    processor.stream = stream

    assert processor.cancel_streaming("operator") is True
    assert stream.calls == ["stop", "close"]
    assert processor.stream is None
    assert processor.gate_state is ProducerGateState.TERMINAL_ENQUEUED
    assert isinstance(processor.audio_queue.get_nowait(), AudioCancelled)


def test_close_failure_retains_handle_and_retry_can_finish_quiescence():
    processor = _configured_processor(target_samples=10, channels=(0,))
    stream = _StopCloseStream(
        stop_errors=(RuntimeError("stop failed"), RuntimeError("stop failed again")),
        close_errors=(RuntimeError("close failed"),),
    )
    processor.stream = stream

    assert processor.cancel_streaming("operator") is False
    assert processor.stream is stream
    assert processor.gate_state is ProducerGateState.QUIESCING
    assert processor.audio_queue.empty()

    assert processor.finish_cancel_streaming("operator") is True
    assert processor.stream is None
    assert processor.gate_state is ProducerGateState.TERMINAL_ENQUEUED
    assert isinstance(processor.audio_queue.get_nowait(), AudioCancelled)


def test_base_exception_from_close_retains_recoverable_handle():
    processor = _configured_processor(target_samples=10, channels=(0,))
    stream = _StopCloseStream(close_errors=(KeyboardInterrupt("interrupted"),))
    processor.stream = stream

    assert processor.cancel_streaming("operator") is False
    assert processor.stream is stream
    assert processor.gate_state is ProducerGateState.QUIESCING
    assert processor.audio_queue.empty()


def test_hostile_stop_close_and_logger_cannot_break_retryable_terminal():
    processor = _configured_processor(target_samples=10, channels=(0,))

    class Logger:
        def warning(self, _message):
            raise _HostileBoundaryError()

    class Stream:
        def __init__(self):
            self.close_attempts = 0

        def stop(self):
            raise _HostileBoundaryError()

        def close(self):
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise _HostileBoundaryError()

    processor.logger = Logger()
    stream = Stream()
    processor.stream = stream

    assert processor.cancel_streaming("operator") is False
    assert processor.stream is stream
    assert type(processor.hardware_quiescence_diagnostic) is str
    assert processor.retry_terminal_quiescence() is True
    assert processor.stream is None
    assert isinstance(processor.audio_queue.get_nowait(), AudioCancelled)


@pytest.mark.parametrize("failure_phase", ["construct", "start"])
def test_auto_completed_finalizer_thread_failure_is_observable_and_retryable(
    failure_phase,
):
    processor = _configured_processor(target_samples=2, channels=(0,))
    controls = []
    processor.set_terminal_finalization_pending_callback(
        lambda event: controls.append(event) or True
    )

    class FailedThread:
        def start(self):
            raise KeyboardInterrupt("thread start interrupted")

    def thread_factory(**_kwargs):
        if failure_phase == "construct":
            raise KeyboardInterrupt("thread construction interrupted")
        return FailedThread()

    processor._terminal_thread_factory = thread_factory

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    batch = processor.audio_queue.get_nowait()
    assert isinstance(batch, AudioBatch)
    assert processor.audio_queue.empty()
    assert len(controls) == 1
    pending = controls[0]
    assert isinstance(pending, AudioFinalizationPending)
    assert pending.terminal_kind == "completed"
    assert processor.gate_state is ProducerGateState.QUIESCING
    assert processor.terminal_finalization_pending is False
    assert processor.retry_terminal_quiescence() is True
    assert processor.audio_queue.get_nowait() == AudioCompleted("session-1", 0, 2)
    assert processor.audio_queue.empty()


def test_auto_fatal_finalizer_thread_failure_preserves_failed_terminal(monkeypatch):
    processor = _configured_processor(target_samples=2, channels=(0,))
    controls = []
    processor.set_terminal_finalization_pending_callback(
        lambda event: controls.append(event) or True
    )
    processor._terminal_thread_factory = lambda **_kwargs: (_ for _ in ()).throw(
        SystemExit("thread unavailable")
    )
    monkeypatch.setattr(
        processor,
        "_audio_batch_from_callback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("oom")),
    )

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    assert processor.audio_queue.empty()
    assert len(controls) == 1
    pending = controls[0]
    assert isinstance(pending, AudioFinalizationPending)
    assert pending.terminal_kind == "failed"
    assert processor.retry_terminal_quiescence() is True
    failure = processor.audio_queue.get_nowait()
    assert isinstance(failure, AudioFailed)
    assert failure.error_code == "allocation-failed"
    assert processor.audio_queue.empty()


def test_auto_close_failure_allows_concurrent_duplicate_retries_without_duplicate_terminal(
    monkeypatch,
):
    retry_close_entered = threading.Event()
    release_retry_close = threading.Event()

    class RetryStream(_StopCloseStream):
        def __init__(self):
            super().__init__(close_errors=(RuntimeError("close failed"),))
            self.close_count = 0

        def close(self):
            self.close_count += 1
            if self.close_count == 1:
                return super().close()
            retry_close_entered.set()
            assert release_retry_close.wait(1)
            self.calls.append("close")

    processor = _configured_processor(target_samples=2, channels=(0,))
    controls = []
    processor.set_terminal_finalization_pending_callback(
        lambda event: controls.append(event) or True
    )
    processor.stream = RetryStream()
    finalizers = []
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
    )
    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)
    assert isinstance(processor.audio_queue.get_nowait(), AudioBatch)
    finalizers.pop()()
    assert processor.audio_queue.empty()
    assert len(controls) == 1
    pending = controls[0]
    assert isinstance(pending, AudioFinalizationPending)
    assert processor.stream is not None

    outcomes = []
    retries = [
        threading.Thread(
            target=lambda: outcomes.append(processor.retry_terminal_quiescence())
        )
        for _ in range(2)
    ]
    retries[0].start()
    assert retry_close_entered.wait(1)
    retries[1].start()
    release_retry_close.set()
    for retry in retries:
        retry.join(timeout=1)

    assert outcomes == [True, True]
    assert processor.audio_queue.get_nowait() == AudioCompleted("session-1", 0, 2)
    assert processor.audio_queue.empty()


def test_first_claimed_failure_payload_survives_losing_failure_and_retry():
    processor = _configured_processor(target_samples=4, channels=(0,))
    processor.stream = _StopCloseStream(
        close_errors=(RuntimeError("close failed"),)
    )

    assert processor.fail_streaming("first-code", "first-message") is False
    assert processor.fail_streaming("second-code", "second-message") is False
    assert processor.error_message == "first-message"
    assert processor.retry_terminal_quiescence() is True

    failure = processor.audio_queue.get_nowait()
    assert failure == AudioFailed("session-1", -1, "first-code", "first-message")
    assert processor.audio_queue.empty()


def test_failed_pending_delivery_is_retained_for_deterministic_take():
    processor = _configured_processor(target_samples=2, channels=(0,))
    processor._terminal_thread_factory = lambda **_kwargs: (_ for _ in ()).throw(
        SystemExit("thread unavailable")
    )
    processor.set_terminal_finalization_pending_callback(
        lambda _event: (_ for _ in ()).throw(KeyboardInterrupt("delivery failed"))
    )

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    assert isinstance(processor.audio_queue.get_nowait(), AudioBatch)
    assert processor.audio_queue.empty()
    assert processor.terminal_finalization_pending is True
    pending = processor.take_terminal_finalization_pending()
    assert isinstance(pending, AudioFinalizationPending)
    assert pending.terminal_kind == "completed"
    assert processor.take_terminal_finalization_pending() is None
    assert processor.terminal_finalization_pending is False


def test_fatal_pending_is_out_of_band_while_admitted_callback_finishes(monkeypatch):
    processor = _configured_processor(target_samples=4, channels=(0,))
    controls = []
    first_copy_entered = threading.Event()
    release_first_copy = threading.Event()
    second_copy_entered = threading.Event()
    release_second_copy = threading.Event()
    original_factory = processor._audio_batch_from_callback
    copy_index = 0

    def controlled_copy(*args, **kwargs):
        nonlocal copy_index
        copy_index += 1
        if copy_index == 1:
            first_copy_entered.set()
            assert release_first_copy.wait(1)
            raise MemoryError("oom")
        if copy_index == 2:
            second_copy_entered.set()
            assert release_second_copy.wait(1)
        return original_factory(*args, **kwargs)

    processor.set_terminal_finalization_pending_callback(
        lambda event: controls.append(event) or True
    )
    processor._terminal_thread_factory = lambda **_kwargs: (_ for _ in ()).throw(
        SystemExit("thread unavailable")
    )
    monkeypatch.setattr(
        processor, "_audio_batch_from_callback", controlled_copy
    )
    first = threading.Thread(
        target=processor._audio_callback,
        args=(np.ones((2, 1), dtype=np.float32), 2, None, None),
    )
    second = threading.Thread(
        target=processor._audio_callback,
        args=(np.ones((2, 1), dtype=np.float32), 2, None, None),
    )
    first.start()
    assert first_copy_entered.wait(1)
    second.start()
    deadline = time.monotonic() + 1
    while processor._active_callback_count != 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    release_first_copy.set()
    assert second_copy_entered.wait(1)
    first.join(timeout=1)

    assert len(controls) == 1
    assert processor.audio_queue.empty()
    release_second_copy.set()
    second.join(timeout=1)
    batch = processor.audio_queue.get_nowait()
    assert (batch.sequence_no, batch.sample_start, batch.sample_stop) == (0, 0, 2)
    assert processor.audio_queue.empty()
    assert processor.retry_terminal_quiescence() is True
    failure = processor.audio_queue.get_nowait()
    assert isinstance(failure, AudioFailed)
    assert failure.last_sequence_no == 0
    assert processor.audio_queue.empty()


def test_callback_baseexception_is_normalized_and_closes_gate_synchronously(monkeypatch):
    processor = _configured_processor(target_samples=4, channels=(0,))
    finalizers = []
    monkeypatch.setattr(
        processor,
        "_launch_terminal_finalizer",
        lambda callback: finalizers.append(callback),
    )
    monkeypatch.setattr(
        processor,
        "_select_multi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("callback interrupted")
        ),
    )

    processor._audio_callback(np.ones((2, 1), dtype=np.float32), 2, None, None)

    assert processor.gate_state is ProducerGateState.QUIESCING
    assert len(finalizers) == 1
    finalizers.pop()()
    failure = processor.audio_queue.get_nowait()
    assert isinstance(failure, AudioFailed)
    assert failure.error_code == "callback-failed"
    assert "callback interrupted" in failure.message


def test_startup_close_failure_does_not_clear_recoverable_stream_handle(monkeypatch):
    class StartupStream(_StopCloseStream):
        def start(self):
            raise RuntimeError("start failed")

    stream = StartupStream(close_errors=(RuntimeError("close failed"),))
    monkeypatch.setattr(
        streaming_audio_processor.sd,
        "InputStream",
        lambda **_kwargs: stream,
    )
    processor = StreamingAudioProcessor()

    code, _message = processor.start_streaming_rec(
        sample_rate=48_000,
        target_samples=4,
        device={"index": 1, "max_input_channels": 1},
        input_channels=(0,),
        session_id="session-startup",
    )

    assert code == error_code.INVALID_RECORD
    assert processor.stream is stream
    assert "close failed" in processor.hardware_quiescence_diagnostic


def test_hardware_start_hostile_exception_text_is_normalized_after_cleanup(monkeypatch):
    class StartupStream(_StopCloseStream):
        def start(self):
            raise _HostileException()

    stream = StartupStream()
    monkeypatch.setattr(
        streaming_audio_processor.sd,
        "InputStream",
        lambda **_kwargs: stream,
    )
    processor = StreamingAudioProcessor()

    code, message = processor.start_streaming_rec(
        sample_rate=48_000,
        target_samples=4,
        device={"index": 1, "max_input_channels": 1},
        input_channels=(0,),
        session_id="session-hostile-start",
    )

    assert code == error_code.INVALID_RECORD
    assert type(message) is str
    assert processor.stream is None
