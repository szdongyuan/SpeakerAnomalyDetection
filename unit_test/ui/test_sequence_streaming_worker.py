import queue
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

from base.save_data import save_audio_simple
from base.streaming_audio_processor import (
    AudioFinalizationPending,
    StreamingAudioProcessor,
)
from ui.sequence.sequence_messages import (
    AudioBatch,
    AudioCancelled,
    AudioCompleted,
    AudioFailed,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView
from ui.sequence.sequence_recording_worker import (
    SequenceStreamingRecordingService,
    StreamingRecordingCancellation,
    StreamingRecordingFailure,
    StreamingRecordingResult,
    StreamingRecordingWorker,
    _build_staged_result,
)
from ui.sequence.sequence_recording_controller import (
    PreparedRecordingSession,
    _recording_sample_counts,
)
from ui.sequence.sequence_recording_model import RecordingSessionSnapshot, StagedRecording
import ui.sequence.sequence_recording_worker as recording_worker_module
from consts import error_code


_QT_APP_HOLDER = None


class _HostileBoundaryError(BaseException):
    def __str__(self):
        raise KeyboardInterrupt("hostile __str__")


def _batch(sequence_no, start, values, *, session_id="session-1", channels=(0,)):
    multi = np.array(values, dtype=np.float32)
    if multi.ndim == 1:
        multi = multi.reshape(-1, 1)
    mono = multi.mean(axis=1)
    multi.setflags(write=False)
    mono.setflags(write=False)
    return AudioBatch.from_callback(
        session_id=session_id,
        sequence_no=sequence_no,
        sample_start=start,
        channel_order=channels,
        mono=mono,
        multi=multi,
    )


class FakeWriter:
    def __init__(self, calls, *, fail_on_write=False):
        self.calls = calls
        self.fail_on_write = fail_on_write

    def write_chunk(self, chunk):
        self.calls.append(("write", np.array(chunk, copy=True)))
        if self.fail_on_write:
            raise OSError("disk full")

    def finalize(self):
        self.calls.append(("finalize",))

    def rollback(self):
        self.calls.append(("rollback",))
        return {"restored": True, "errors": ()}


def _worker(message_queue, writer, events):
    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=message_queue,
        writer=writer,
        channel_order=(0,),
        target_samples=4,
    )
    worker.batch_ready.connect(
        lambda value: events.append(("batch", value)), Qt.DirectConnection
    )
    worker.completed.connect(
        lambda value: events.append(("completed", value)), Qt.DirectConnection
    )
    worker.failed.connect(
        lambda value: events.append(("failed", value)), Qt.DirectConnection
    )
    worker.cancelled.connect(
        lambda value: events.append(("cancelled", value)), Qt.DirectConnection
    )
    return worker


def test_worker_blocks_then_accepts_every_sample_once_and_completes_after_writer():
    messages = queue.SimpleQueue()
    calls = []
    events = []
    worker = _worker(messages, FakeWriter(calls), events)
    thread = threading.Thread(target=worker.run)
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()
    assert calls == []

    messages.put(_batch(0, 0, [[1.0], [2.0]]))
    messages.put(_batch(1, 2, [[3.0], [4.0]]))
    messages.put(AudioCompleted("session-1", 1, 4))
    thread.join(1)

    assert thread.is_alive() is False
    assert [name for name, *_rest in calls] == ["write", "write", "finalize"]
    assert [name for name, _value in events] == ["batch", "batch", "completed"]
    assert calls[0][1].tolist() == [[1.0], [2.0]]
    assert calls[1][1].tolist() == [[3.0], [4.0]]
    result = events[-1][1]
    assert isinstance(result, StreamingRecordingResult)
    assert result.last_sequence_no == 1
    assert result.sample_count == 4
    np.testing.assert_allclose(result.multi, [[1.0], [2.0], [3.0], [4.0]])
    assert result.multi.flags.writeable is False


def test_worker_emits_progress_only_after_writer_acceptance():
    messages = queue.SimpleQueue()
    calls = []
    events = []

    class OrderingWriter(FakeWriter):
        def write_chunk(self, chunk):
            assert events == []
            super().write_chunk(chunk)

    worker = _worker(messages, OrderingWriter(calls), events)
    messages.put(_batch(0, 0, [[1.0], [2.0]]))
    messages.put(AudioCancelled("session-1", 0, "stop"))
    worker.run()

    assert [name for name, _value in events] == ["batch", "cancelled"]
    assert calls[-1] == ("rollback",)


def test_worker_rejects_gap_and_rolls_back_without_later_progress():
    messages = queue.SimpleQueue()
    calls = []
    events = []
    worker = _worker(messages, FakeWriter(calls), events)
    messages.put(_batch(1, 2, [[3.0], [4.0]]))
    worker.run()

    assert [name for name, _value in events] == ["failed"]
    failure = events[0][1]
    assert isinstance(failure, StreamingRecordingFailure)
    assert failure.code == "sequence-mismatch"
    assert "expected sequence 0" in failure.message
    assert calls == [("rollback",)]


def test_worker_rejects_completed_sequence_and_sample_count_mismatches():
    for terminal, expected in (
        (AudioCompleted("session-1", 9, 4), "terminal-sequence-mismatch"),
        (AudioCompleted("session-1", 1, 99), "terminal-sample-mismatch"),
    ):
        messages = queue.SimpleQueue()
        calls = []
        events = []
        worker = _worker(messages, FakeWriter(calls), events)
        messages.put(_batch(0, 0, [[1.0], [2.0]]))
        messages.put(_batch(1, 2, [[3.0], [4.0]]))
        messages.put(terminal)
        worker.run()

        assert [name for name, _value in events] == ["batch", "batch", "failed"]
        assert events[-1][1].code == expected
        assert calls[-1] == ("rollback",)


def test_writer_failure_is_normalized_and_rolls_back():
    messages = queue.SimpleQueue()
    calls = []
    events = []
    worker = _worker(messages, FakeWriter(calls, fail_on_write=True), events)
    messages.put(_batch(0, 0, [[1.0], [2.0]]))
    worker.run()

    assert [name for name, _value in events] == ["failed"]
    assert events[0][1].code == "writer-failed"
    assert events[0][1].exception is None
    assert calls[-1] == ("rollback",)


def test_writer_baseexception_is_normalized_and_shuts_down_producer():
    messages = queue.SimpleQueue()
    events = []
    shutdown = []

    class InterruptedWriter(FakeWriter):
        def write_chunk(self, _chunk):
            raise SystemExit("writer interrupted")

    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=InterruptedWriter([]),
        channel_order=(0,),
        target_samples=4,
        shutdown_producer=lambda code, message: shutdown.append((code, message))
        or {"quiesced": True},
    )
    worker.failed.connect(events.append, Qt.DirectConnection)
    messages.put(_batch(0, 0, [[1.0], [2.0]]))

    worker.run()

    assert shutdown == [("writer-failed", "writer interrupted")]
    assert len(events) == 1
    assert events[0].exception is None


@pytest.mark.parametrize("failure_phase", ["write", "finalize", "queue"])
def test_hostile_consumer_boundaries_still_emit_plain_failure_and_cleanup_once(
    failure_phase,
):
    backing = queue.SimpleQueue()
    writer_calls = []
    shutdown_calls = []
    failures = []

    class HostileQueue:
        def get(self):
            if failure_phase == "queue":
                raise _HostileBoundaryError()
            return backing.get()

    class Writer:
        def write_chunk(self, _chunk):
            writer_calls.append("write")
            if failure_phase == "write":
                raise _HostileBoundaryError()

        def finalize(self):
            writer_calls.append("finalize")
            if failure_phase == "finalize":
                raise _HostileBoundaryError()

        def rollback(self):
            writer_calls.append("rollback")
            raise _HostileBoundaryError()

    class Logger:
        def error(self, _message):
            raise _HostileBoundaryError()

    def shutdown(*_args):
        shutdown_calls.append("shutdown")
        raise _HostileBoundaryError()

    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=HostileQueue(),
        writer=Writer(),
        channel_order=(0,),
        target_samples=2,
        shutdown_producer=shutdown,
        logger=Logger(),
    )
    worker.failed.connect(failures.append, Qt.DirectConnection)
    if failure_phase != "queue":
        backing.put(_batch(0, 0, [[1.0], [2.0]]))
    if failure_phase == "finalize":
        backing.put(AudioCompleted("session-1", 0, 2))

    worker.run()

    assert shutdown_calls == ["shutdown"]
    assert writer_calls.count("rollback") == 1
    assert len(failures) == 1
    assert type(failures[0].message) is str
    assert failures[0].exception is None
    assert failures[0].producer_quiesced is False
    assert type(failures[0].shutdown_diagnostic) is str


def test_failed_and_cancelled_sentinels_are_first_terminal_and_rollback():
    for terminal, event_name, code in (
        (AudioFailed("session-1", -1, "allocation-failed", "oom"), "failed", "allocation-failed"),
        (AudioCancelled("session-1", -1, "operator"), "cancelled", None),
    ):
        messages = queue.SimpleQueue()
        calls = []
        events = []
        worker = _worker(messages, FakeWriter(calls), events)
        messages.put(terminal)
        messages.put(_batch(0, 0, [[1.0], [2.0]]))
        worker.run()

        assert [name for name, _value in events] == [event_name]
        if code is not None:
            assert events[0][1].code == code
        assert calls == [("rollback",)]


def test_recording_view_coalesces_pending_continuous_ranges_without_gaps():
    scheduled = []
    paints = []
    view = SequenceRecordingView(
        plot_recording=lambda signal, sample_rate: paints.append(
            (np.array(signal, copy=True), sample_rate)
        ),
        schedule_waveform_refresh=lambda callback: scheduled.append(callback),
    )
    view.begin_streaming_session("session-1", 10.0)

    view.queue_recording_batch(_batch(0, 0, [[1.0], [2.0]]))
    assert len(scheduled) == 1
    scheduled.pop(0)()
    view.queue_recording_batch(_batch(1, 2, [[3.0]]))
    view.queue_recording_batch(_batch(2, 3, [[4.0]]))
    view.queue_recording_batch(_batch(3, 4, [[5.0]]))
    assert len(scheduled) == 1
    scheduled.pop(0)()

    assert len(paints) == 2
    np.testing.assert_allclose(paints[0][0], [1.0, 2.0])
    np.testing.assert_allclose(paints[1][0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert view.waveform_display_cursor == 5
    assert view.pending_waveform_range is None


def test_stale_waveform_callback_cannot_flush_the_next_session():
    scheduled = []
    paints = []
    view = SequenceRecordingView(
        plot_recording=lambda signal, _rate: paints.append(np.array(signal)),
        schedule_waveform_refresh=scheduled.append,
    )
    view.begin_streaming_session("session-a", 10.0)
    view.queue_recording_batch(_batch(0, 0, [[1.0]], session_id="session-a"))
    stale_callback = scheduled.pop(0)
    view.begin_streaming_session("session-b", 10.0)
    view.queue_recording_batch(_batch(0, 0, [[2.0]], session_id="session-b"))
    current_callback = scheduled.pop(0)

    stale_callback()
    assert paints == []
    assert view.pending_waveform_range == (0, 1)
    current_callback()
    assert len(paints) == 1
    np.testing.assert_allclose(paints[0], [2.0])


def test_waveform_coalescing_preserves_multichannel_display_order():
    scheduled = []
    paints = []
    view = SequenceRecordingView(
        plot_recording=lambda signal, _rate: paints.append(np.array(signal)),
        schedule_waveform_refresh=scheduled.append,
    )
    view.begin_streaming_session("session-1", 10.0)
    worker_events = []
    messages = queue.SimpleQueue()
    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=FakeWriter([]),
        channel_order=(1, 3),
        target_samples=2,
    )
    worker.batch_ready.connect(worker_events.append, Qt.DirectConnection)
    messages.put(_batch(0, 0, [[1.0, 10.0], [2.0, 20.0]], channels=(1, 3)))
    messages.put(AudioCompleted("session-1", 0, 2))
    worker.run()

    assert view.queue_recording_batch(worker_events[0])
    scheduled.pop(0)()
    assert paints[0].shape == (2, 2)
    np.testing.assert_allclose(paints[0], [[1.0, 10.0], [2.0, 20.0]])


def test_sequence_window_has_no_streaming_poll_timer_or_queue_poll_call_site():
    source = Path("ui/sequence/sequence_widget.py").read_text(encoding="utf-8")
    assert "streaming_poll_timer" not in source
    assert "def _poll_streaming_queue" not in source
    assert ".process_queue()" not in source


def _prepared_streaming_session(
    tmp_path,
    *,
    session_id="session-1",
    mode="RECORD_ONLY",
    target_samples=4,
    acquisition_sample_count=None,
    delay_frames=0,
    prepare_frames=0,
    prolong_frames=0,
    stimulus_data=None,
    alignment_sample_count=None,
    input_channels=(0,),
):
    if acquisition_sample_count is None:
        acquisition_sample_count = target_samples
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=1,
        configuration_generation=1,
        mode=mode,
        sample_rate=10.0,
        bit_depth=32,
        input_channels=input_channels,
        input_device={"index": 1, "max_input_channels": len(input_channels)},
        output_device=None,
        stimulus_snapshot={
            "data": stimulus_data,
            "info": None,
            "alignment_sample_count": alignment_sample_count,
        },
        target_samples=target_samples,
        acquisition_sample_count=acquisition_sample_count,
        output_path=tmp_path / "record.wav",
        temp_path=tmp_path / "record.tmp.wav",
        backup_path=None,
        record_id="record-1",
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    return PreparedRecordingSession(
        snapshot,
        {
            "recorded_dict": {
                "monitor_playback": False,
                "recording_start_delay_frames": delay_frames,
                "prepare_frames": prepare_frames,
                "prolong_frames": prolong_frames,
                "num_frames": target_samples + prolong_frames,
            },
            "recorded_signal_info": {"file_path": str(snapshot.output_path)},
            "stimulus_info": None,
            "stimulus_data": stimulus_data,
            "stimulus_dict": {
                "data": stimulus_data,
                "amplitude": 1.0,
                "alignment_sample_count": alignment_sample_count,
            },
            "alignment_sample_count": alignment_sample_count,
        },
    )


def _nonblocking_streaming_worker(**kwargs):
    worker = StreamingRecordingWorker(**kwargs)
    worker.run = lambda: None
    return worker


class _ImmediateStreamingProcessor:
    def __init__(self, hardware_starts=None):
        self.audio_queue = queue.SimpleQueue()
        self.hardware_starts = hardware_starts

    def start_streaming_rec(self, **_kwargs):
        if self.hardware_starts is not None:
            self.hardware_starts.append(True)
        return error_code.OK, "started"

    def begin_cancel_streaming(self):
        return True

    def finish_cancel_streaming(self, _reason):
        return True


def test_start_publication_is_atomic_with_idle_close_and_quiesce(
    tmp_path, monkeypatch
):
    event_construction_entered = threading.Event()
    release_event_construction = threading.Event()
    processor_factory_entered = threading.Event()
    release_processor_factory = threading.Event()
    close_attempted = threading.Event()
    close_admitted = threading.Event()
    real_event = threading.Event
    event_calls = []
    hardware_starts = []
    start_results = []
    quiesce_results = []
    prepared = _prepared_streaming_session(tmp_path)

    def first_blocking_event():
        event_calls.append(True)
        if len(event_calls) == 1:
            event_construction_entered.set()
            assert release_event_construction.wait(1)
        return real_event()

    def processor_factory():
        processor_factory_entered.set()
        assert release_processor_factory.wait(1)
        return _ImmediateStreamingProcessor(hardware_starts)

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=processor_factory,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=_nonblocking_streaming_worker,
        queued_delivery=False,
    )
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, object()))
    )
    monkeypatch.setattr(recording_worker_module.threading, "Event", first_blocking_event)
    starter.start()
    assert event_construction_entered.wait(1)

    def close_and_quiesce():
        close_attempted.set()
        handle = service.close_admission(prepared)
        close_admitted.set()
        quiesce_results.append(service.quiesce(prepared, "operator", handle))

    stopper = threading.Thread(target=close_and_quiesce)
    stopper.start()
    assert close_attempted.wait(1)
    release_event_construction.set()
    assert processor_factory_entered.wait(1)
    assert close_admitted.wait(1)
    release_processor_factory.set()
    starter.join(1)
    stopper.join(1)

    assert starter.is_alive() is False
    assert stopper.is_alive() is False
    assert start_results == [False]
    assert quiesce_results == [{"quiesced": True}]
    assert hardware_starts == []
    assert service.active_session_id is None


def test_final_state_commit_rereads_close_intent_before_thread_start(tmp_path):
    final_check_entered = threading.Event()
    release_final_check = threading.Event()
    start_results = []
    quiesce_results = []
    observed_thread_start_states = []
    lifecycle = []
    prepared = _prepared_streaming_session(tmp_path)

    class Processor(_ImmediateStreamingProcessor):
        def begin_cancel_streaming(self):
            lifecycle.append("cancel-owned")
            return True

    class ObservingThread:
        def __init__(self, target):
            self.target = target

        def start(self):
            observed_thread_start_states.append(service._session.state)
            lifecycle.append("thread-start")
            self.target()

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=_nonblocking_streaming_worker,
        thread_factory=lambda **kwargs: ObservingThread(kwargs["target"]),
        queued_delivery=False,
    )
    original_require = service._require_reservation
    require_calls = []

    def block_after_final_require(reservation, **kwargs):
        result = original_require(reservation, **kwargs)
        require_calls.append(True)
        if len(require_calls) == 10:
            final_check_entered.set()
            assert release_final_check.wait(1)
        return result

    service._require_reservation = block_after_final_require
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, object()))
    )
    starter.start()
    assert final_check_entered.wait(1)
    handle = service.close_admission(prepared)
    stopper = threading.Thread(
        target=lambda: quiesce_results.append(
            service.quiesce(prepared, "operator", handle)
        )
    )
    stopper.start()
    release_final_check.set()
    starter.join(1)
    stopper.join(1)

    assert starter.is_alive() is False
    assert stopper.is_alive() is False
    assert start_results == [True]
    assert observed_thread_start_states == ["STARTUP_CANCELLING"]
    assert lifecycle == ["cancel-owned", "thread-start"]
    assert quiesce_results == [{"quiesced": True}]
    assert service.active_session_id is None


def test_idle_close_intents_are_exact_bounded_and_reject_same_session_start(
    tmp_path,
):
    hardware_starts = []
    processor_factory_calls = []

    def processor_factory():
        processor_factory_calls.append(True)
        return _ImmediateStreamingProcessor(hardware_starts)

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=processor_factory,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=_nonblocking_streaming_worker,
        queued_delivery=False,
    )
    closed = _prepared_streaming_session(tmp_path, session_id="closed-session")
    duplicate_handles = [
        service.close_admission(closed),
        service.close_admission(closed),
    ]
    assert [service.quiesce(closed, "idle", item) for item in duplicate_handles] == [
        {"quiesced": True},
        {"quiesced": True},
    ]
    assert service.start(closed, object()) is False
    assert processor_factory_calls == []
    assert hardware_starts == []

    fresh = _prepared_streaming_session(tmp_path, session_id="fresh-session")
    assert service.start(fresh, object()) is True
    handle = service.close_admission(fresh)
    assert service.quiesce(fresh, "done", handle) == {"quiesced": True}
    assert processor_factory_calls == [True]
    assert hardware_starts == [True]

    for index in range(service._MAX_IDLE_CLOSE_INTENTS * 2):
        stale = _prepared_streaming_session(
            tmp_path, session_id=f"stale-session-{index}"
        )
        service.close_admission(stale)
    assert len(service._idle_close_intents) == service._MAX_IDLE_CLOSE_INTENTS


def test_event_construction_baseexception_releases_start_lock_without_corruption(
    tmp_path, monkeypatch
):
    real_event = threading.Event
    hardware_starts = []
    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=lambda: _ImmediateStreamingProcessor(hardware_starts),
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=_nonblocking_streaming_worker,
        queued_delivery=False,
    )

    def interrupted_event():
        raise SystemExit("event construction interrupted")

    monkeypatch.setattr(recording_worker_module.threading, "Event", interrupted_event)
    with pytest.raises(SystemExit, match="event construction interrupted"):
        service.start(prepared, object())
    assert service.active_session_id is None

    monkeypatch.setattr(recording_worker_module.threading, "Event", real_event)
    assert service.start(prepared, object()) is True
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "done", handle) == {"quiesced": True}
    assert hardware_starts == [True]


@pytest.mark.parametrize("close_wins", [False, True])
def test_start_and_close_barrier_has_one_linearization_without_deadlock(
    tmp_path, close_wins
):
    for index in range(8):
        factory_entered = threading.Event()
        release_factory = threading.Event()
        starts = []
        hardware_starts = []
        prepared = _prepared_streaming_session(
            tmp_path, session_id=f"session-{close_wins}-{index}"
        )

        def processor_factory():
            factory_entered.set()
            assert release_factory.wait(1)
            return _ImmediateStreamingProcessor(hardware_starts)

        service = SequenceStreamingRecordingService(
            view=SequenceRecordingView(
                schedule_waveform_refresh=lambda _callback: None
            ),
            processor_factory=processor_factory,
            writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
            worker_factory=_nonblocking_streaming_worker,
            queued_delivery=False,
        )
        if close_wins:
            handle = service.close_admission(prepared)
            assert service.quiesce(prepared, "close-wins", handle) == {
                "quiesced": True
            }
            assert service.start(prepared, object()) is False
            assert factory_entered.is_set() is False
            assert hardware_starts == []
            continue

        starter = threading.Thread(
            target=lambda: starts.append(service.start(prepared, object()))
        )
        starter.start()
        assert factory_entered.wait(1)
        handle = service.close_admission(prepared)
        release_factory.set()
        assert service.quiesce(prepared, "start-wins", handle) == {
            "quiesced": True
        }
        starter.join(1)
        assert starter.is_alive() is False
        assert handle["session"] is not None
        assert starts == [False]
        assert hardware_starts == []
        assert service.active_session_id is None


def test_record_only_trim_rewrite_preserves_multichannel_file_and_result(tmp_path):
    prepared = _prepared_streaming_session(
        tmp_path,
        target_samples=4,
        acquisition_sample_count=7,
        delay_frames=2,
        prolong_frames=1,
        input_channels=(0, 1),
    )
    multi = np.arange(14, dtype=np.float32).reshape(7, 2) / 20
    mono = multi.mean(axis=1)

    staged = _build_staged_result(
        prepared,
        lambda *_args: (_ for _ in ()).throw(AssertionError("no alignment")),
        save_audio_simple,
        mono,
        multi,
    )

    rate, written = wavfile.read(prepared.snapshot.temp_path)
    expected = multi[2:6]
    assert rate == 10
    assert written.shape == (4, 2)
    np.testing.assert_allclose(written, expected)
    result = staged.data_struct_fields["store_wave_data_multi"]
    np.testing.assert_allclose(result, expected)
    assert result.flags.writeable is False
    assert staged.data_struct_fields["store_wave_data"].shape == (4,)


def test_record_only_trim_rewrite_keeps_single_channel_mono_file_contract(tmp_path):
    prepared = _prepared_streaming_session(
        tmp_path,
        target_samples=3,
        acquisition_sample_count=5,
        delay_frames=1,
        prolong_frames=1,
    )
    multi = np.arange(5, dtype=np.float32).reshape(-1, 1) / 10

    _build_staged_result(
        prepared,
        lambda *_args: (_ for _ in ()).throw(AssertionError("no alignment")),
        save_audio_simple,
        multi[:, 0],
        multi,
    )

    _rate, written = wavfile.read(prepared.snapshot.temp_path)
    assert written.shape == (3,)
    np.testing.assert_allclose(written, multi[1:4, 0])


def test_recording_sample_counts_separate_physical_capture_from_result_frames():
    assert _recording_sample_counts(
        "RECORD_ONLY",
        {
            "num_frames": 20,
            "prolong_frames": 3,
            "recording_start_delay_frames": 4,
        },
        {},
    ) == (24, 17)
    assert _recording_sample_counts(
        "PLAY_AND_RECORD",
        {
            "prepare_frames": 3,
            "prolong_frames": 4,
            "recording_start_delay_frames": 2,
        },
        {"data": np.arange(5), "amplitude": 1.0},
    ) == (14, 5)


def test_frequency_stepped_capture_uses_full_stimulus_but_result_uses_alignment_count():
    assert _recording_sample_counts(
        "PLAY_AND_RECORD",
        {
            "prepare_frames": 2,
            "prolong_frames": 3,
            "recording_start_delay_frames": 4,
        },
        {
            "data": np.arange(10),
            "amplitude": 1.0,
            "alignment_sample_count": 6,
        },
    ) == (19, 6)


def test_snapshot_defaults_acquisition_count_for_blocking_compatibility(tmp_path):
    prepared = _prepared_streaming_session(tmp_path)

    assert prepared.snapshot.acquisition_sample_count == 4
    assert prepared.snapshot.target_samples == 4


def test_streaming_service_passes_acquisition_count_without_discarding_delay(tmp_path):
    prepared = _prepared_streaming_session(
        tmp_path,
        acquisition_sample_count=6,
        delay_frames=2,
    )
    captured = {}

    class Processor:
        def start_streaming_rec(self, **kwargs):
            captured.update(kwargs)
            return error_code.OK, "started"

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
    )

    assert service._start_processor(Processor(), prepared)[0] == error_code.OK
    assert captured["target_samples"] == 6
    assert captured["discard_initial_samples"] == 0


def test_consumer_failure_requests_producer_shutdown_before_rollback_and_signal():
    messages = queue.SimpleQueue()
    order = []

    class Writer(FakeWriter):
        def rollback(self):
            order.append("rollback")
            return super().rollback()

    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=Writer([]),
        channel_order=(0,),
        target_samples=4,
        shutdown_producer=lambda code, message: order.append(
            ("shutdown", code, message)
        )
        or {"quiesced": True},
    )
    worker.failed.connect(lambda _failure: order.append("failed"), Qt.DirectConnection)
    messages.put(_batch(1, 2, [[3.0], [4.0]]))

    worker.run()

    assert order[0][0:2] == ("shutdown", "sequence-mismatch")
    assert order[1:] == ["rollback", "failed"]


def test_auto_completed_close_failure_routes_task5_pending_then_retry_succeeds(
    tmp_path,
):
    processors = []
    writer_events = []
    terminal_failures = []
    terminal_notified = threading.Event()

    class RecordingQueue(queue.SimpleQueue):
        def __init__(self):
            super().__init__()
            self.put_types = []

        def put(self, item, *args, **kwargs):
            self.put_types.append(type(item))
            return super().put(item, *args, **kwargs)

    class Stream:
        def __init__(self):
            self.close_attempts = 0

        def stop(self):
            return None

        def close(self):
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("close failed")

    class Processor(StreamingAudioProcessor):
        def __init__(self):
            super().__init__()
            processors.append(self)

        def start_streaming_rec(self, **kwargs):
            self.configure_event_session(
                session_id=kwargs["session_id"],
                sample_rate=kwargs["sample_rate"],
                target_samples=kwargs["target_samples"],
                callback_block_size=kwargs["callback_block_size"],
                channel_order=kwargs["input_channels"],
            )
            self.audio_queue = RecordingQueue()
            self.stream = Stream()
            self.is_recording = True
            self._audio_callback(
                np.ones((kwargs["target_samples"], 1), dtype=np.float32),
                kwargs["target_samples"],
                None,
                None,
            )
            return error_code.OK, "started"

    class Terminal:
        def streaming_consumer_failed(
            self, message, rollback, producer_quiesced, diagnostic
        ):
            terminal_failures.append(
                (message, rollback, producer_quiesced, diagnostic)
            )
            terminal_notified.set()
            return True

    prepared = _prepared_streaming_session(tmp_path, target_samples=4)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(writer_events),
        queued_delivery=False,
    )

    global _QT_APP_HOLDER
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance() or QApplication([])
    _QT_APP_HOLDER = app

    assert service.start(prepared, Terminal())
    deadline = time.monotonic() + 1
    while not terminal_notified.is_set() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert terminal_notified.is_set()
    assert terminal_failures[0][2] is False
    assert "close failed" in terminal_failures[0][3]
    assert service.active_session_id == "session-1"
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "retry", handle) == {"quiesced": True}
    assert service.active_session_id is None
    assert processors[0].stream is None
    assert processors[0].audio_queue.put_types.count(AudioCompleted) == 1
    assert processors[0].audio_queue.put_types.count(AudioFinalizationPending) == 0
    assert processors[0].audio_queue.put_types == [AudioBatch, AudioCompleted]
    assert writer_events.count(("rollback",)) == 1


def test_streaming_service_reserves_start_atomically_before_processor_factory(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    processor_factory_calls = []
    results = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            entered.set()
            assert release.wait(1)
            self.audio_queue.put(AudioCancelled("session-1", -1, "done"))
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            return False

        def retry_terminal_quiescence(self):
            return True

    def processor_factory():
        processor_factory_calls.append(True)
        return Processor()

    class Terminal:
        def recording_cancelled(self, _reason):
            return True

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=processor_factory,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    first = threading.Thread(
        target=lambda: results.append(service.start(prepared, Terminal()))
    )
    first.start()
    assert entered.wait(1)

    assert service.start(prepared, Terminal()) is False
    assert len(processor_factory_calls) == 1
    release.set()
    first.join(1)

    assert results == [True]
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "done", handle) == {"quiesced": True}


def test_concurrent_losing_start_cannot_orphan_first_close_failure_recovery(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    factory_calls = []
    first_errors = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.hardware_quiescence_diagnostic = "close failed"
            self.claimed = False

        def start_streaming_rec(self, **_kwargs):
            entered.set()
            assert release.wait(1)
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            if self.claimed:
                return False
            self.claimed = True
            return True

        def finish_cancel_streaming(self, _reason):
            return False

        def retry_terminal_quiescence(self):
            return True

    def processor_factory():
        factory_calls.append(True)
        return Processor()

    def fail_worker(**_kwargs):
        raise _HostileBoundaryError()

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=processor_factory,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=fail_worker,
        queued_delivery=False,
    )

    def run_first():
        try:
            service.start(prepared, object())
        except BaseException as error:
            first_errors.append(type(error))

    first = threading.Thread(target=run_first)
    first.start()
    assert entered.wait(1)
    assert service.start(prepared, object()) is False
    release.set()
    first.join(1)

    assert first_errors == [_HostileBoundaryError]
    assert factory_calls == [True]
    assert service.active_session_id == "session-1"
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "retry", handle) == {"quiesced": True}
    assert service.active_session_id is None


def test_close_during_processor_factory_is_not_lost_and_prevents_hardware_start(
    tmp_path,
):
    factory_entered = threading.Event()
    release_factory = threading.Event()
    start_calls = []
    start_results = []
    quiesce_results = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            start_calls.append(True)
            self.audio_queue.put(AudioCancelled("session-1", -1, "cleanup"))
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            return True

        def finish_cancel_streaming(self, reason):
            self.audio_queue.put(AudioCancelled("session-1", -1, reason))
            return True

    def processor_factory():
        factory_entered.set()
        assert release_factory.wait(1)
        return Processor()

    class Terminal:
        def recording_cancelled(self, _reason):
            return True

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=processor_factory,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, Terminal()))
    )
    starter.start()
    assert factory_entered.wait(1)
    handle = service.close_admission(prepared)
    stopper = threading.Thread(
        target=lambda: quiesce_results.append(
            service.quiesce(prepared, "operator", handle)
        )
    )
    stopper.start()
    release_factory.set()
    starter.join(1)
    stopper.join(1)

    try:
        assert starter.is_alive() is False
        assert stopper.is_alive() is False
        assert start_results == [False]
        assert quiesce_results == [{"quiesced": True}]
        assert start_calls == []
        assert service.active_session_id is None
    finally:
        if service.active_session_id is not None:
            cleanup = service.close_admission(prepared)
            service.quiesce(prepared, "cleanup", cleanup)


def test_close_before_configure_reset_is_reapplied_after_hardware_start(
    tmp_path,
):
    start_entered = threading.Event()
    release_start = threading.Event()
    start_results = []
    quiesce_results = []
    processor_holder = []

    class Stream:
        def __init__(self):
            self.closed = False

        def stop(self):
            return None

        def close(self):
            self.closed = True

    class Processor(StreamingAudioProcessor):
        def __init__(self):
            super().__init__()
            self.live_stream = None
            processor_holder.append(self)

        def start_streaming_rec(self, **kwargs):
            start_entered.set()
            assert release_start.wait(1)
            self.configure_event_session(
                session_id=kwargs["session_id"],
                sample_rate=kwargs["sample_rate"],
                target_samples=kwargs["target_samples"],
                callback_block_size=kwargs["callback_block_size"],
                channel_order=kwargs["input_channels"],
            )
            self.live_stream = Stream()
            self.stream = self.live_stream
            return error_code.OK, "started"

    def nonblocking_worker(**kwargs):
        worker = StreamingRecordingWorker(**kwargs)
        worker.run = lambda: None
        return worker

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        worker_factory=nonblocking_worker,
        queued_delivery=False,
    )
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, object()))
    )
    starter.start()
    assert start_entered.wait(1)
    handle = service.close_admission(prepared)
    stopper = threading.Thread(
        target=lambda: quiesce_results.append(
            service.quiesce(prepared, "operator", handle)
        )
    )
    stopper.start()
    release_start.set()
    starter.join(1)
    stopper.join(1)

    try:
        assert starter.is_alive() is False
        assert stopper.is_alive() is False
        assert start_results == [True]
        assert quiesce_results == [{"quiesced": True}]
        assert processor_holder[0].live_stream.closed is True
        assert service.active_session_id is None
    finally:
        if service.active_session_id is not None:
            cleanup = service.close_admission(prepared)
            service.quiesce(prepared, "cleanup", cleanup)


def test_close_during_hardware_start_drains_batches_and_duplicate_quiesce_once(
    tmp_path,
):
    hardware_entered = threading.Event()
    release_hardware = threading.Event()
    start_results = []
    quiesce_results = []
    processors = []
    writer_events = []

    class RecordingQueue(queue.SimpleQueue):
        def __init__(self):
            super().__init__()
            self.put_types = []

        def put(self, item, *args, **kwargs):
            self.put_types.append(type(item))
            return super().put(item, *args, **kwargs)

    class Processor(StreamingAudioProcessor):
        def __init__(self):
            super().__init__()
            self.begin_calls = 0
            self.finish_calls = 0
            processors.append(self)

        def start_streaming_rec(self, **kwargs):
            self.configure_event_session(
                session_id=kwargs["session_id"],
                sample_rate=kwargs["sample_rate"],
                target_samples=kwargs["target_samples"],
                callback_block_size=kwargs["callback_block_size"],
                channel_order=kwargs["input_channels"],
            )
            self.audio_queue = RecordingQueue()
            self._audio_callback(
                np.array([[1.0], [2.0]], dtype=np.float32), 2, None, None
            )
            hardware_entered.set()
            assert release_hardware.wait(1)
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            self.begin_calls += 1
            return super().begin_cancel_streaming()

        def finish_cancel_streaming(self, reason):
            self.finish_calls += 1
            return super().finish_cancel_streaming(reason)

    prepared = _prepared_streaming_session(tmp_path, target_samples=4)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(writer_events),
        queued_delivery=False,
    )
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, object()))
    )
    starter.start()
    assert hardware_entered.wait(1)
    handles = [service.close_admission(prepared), service.close_admission(prepared)]
    stoppers = [
        threading.Thread(
            target=lambda handle=handle: quiesce_results.append(
                service.quiesce(prepared, "operator", handle)
            )
        )
        for handle in handles
    ]
    for stopper in stoppers:
        stopper.start()
    release_hardware.set()
    starter.join(1)
    for stopper in stoppers:
        stopper.join(1)

    assert start_results == [True]
    assert quiesce_results == [{"quiesced": True}, {"quiesced": True}]
    assert processors[0].begin_calls == 1
    assert processors[0].finish_calls == 1
    assert processors[0].audio_queue.put_types == [AudioBatch, AudioCancelled]
    assert [event[0] for event in writer_events] == ["write", "rollback"]
    assert service.active_session_id is None


def test_starting_close_failure_stays_recoverable_for_next_quiesce(tmp_path):
    hardware_entered = threading.Event()
    release_hardware = threading.Event()
    start_results = []
    processors = []

    class Stream:
        def __init__(self):
            self.close_attempts = 0

        def stop(self):
            return None

        def close(self):
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("close failed")

    class Processor(StreamingAudioProcessor):
        def __init__(self):
            super().__init__()
            processors.append(self)

        def start_streaming_rec(self, **kwargs):
            self.configure_event_session(
                session_id=kwargs["session_id"],
                sample_rate=kwargs["sample_rate"],
                target_samples=kwargs["target_samples"],
                callback_block_size=kwargs["callback_block_size"],
                channel_order=kwargs["input_channels"],
            )
            self.stream = Stream()
            hardware_entered.set()
            assert release_hardware.wait(1)
            return error_code.OK, "started"

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    starter = threading.Thread(
        target=lambda: start_results.append(service.start(prepared, object()))
    )
    starter.start()
    assert hardware_entered.wait(1)
    handle = service.close_admission(prepared)
    release_hardware.set()

    first = service.quiesce(prepared, "operator", handle)
    starter.join(1)

    assert start_results == [True]
    assert first["quiesced"] is False
    assert "close failed" in first["diagnostic"]
    assert service.active_session_id == "session-1"
    assert service.quiesce(prepared, "operator", handle) == {"quiesced": True}
    assert processors[0].stream is None
    assert service.active_session_id is None


def test_service_retries_retained_pending_notification_without_audio_fifo_item(
    tmp_path,
):
    notifications = []
    attempts = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.pending_callback = None

        def set_terminal_finalization_pending_callback(self, callback):
            self.pending_callback = callback

        def take_terminal_finalization_pending(self):
            return None

        def start_streaming_rec(self, **_kwargs):
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            return False

        def retry_terminal_quiescence(self):
            self.audio_queue.put(AudioCancelled("session-1", -1, "done"))
            return True

    class Terminal:
        def streaming_consumer_failed(self, *values):
            attempts.append(values)
            if len(attempts) == 1:
                raise _HostileBoundaryError()
            notifications.append(values)
            return True

    processor = Processor()
    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None),
        processor_factory=lambda: processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    terminal = Terminal()
    assert service.start(prepared, terminal)
    control_emissions = []
    service.producer_finalization_pending.connect(
        control_emissions.append, Qt.DirectConnection
    )
    descriptor = AudioFinalizationPending(
        "session-1", -1, 0, "completed", "terminal-finalization-failed", "close failed"
    )

    assert processor.pending_callback(descriptor) is True
    assert processor.pending_callback(
        AudioFinalizationPending(
            "session-1", -1, 0, "failed", "other", "later message"
        )
    ) is True
    assert control_emissions == [descriptor]
    service._on_producer_finalization_pending(descriptor)
    assert notifications == []
    assert service.retry_pending_notification("session-1") is True
    assert len(notifications) == 1
    assert processor.audio_queue.empty()

    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "done", handle) == {"quiesced": True}


def test_start_event_allocation_failure_retires_reservation_before_hardware(
    tmp_path, monkeypatch
):
    processor_factory_calls = []

    def fail_event():
        raise _HostileBoundaryError()

    monkeypatch.setattr(
        "ui.sequence.sequence_recording_worker.threading.Event", fail_event
    )
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=lambda: processor_factory_calls.append(True),
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )

    with pytest.raises(_HostileBoundaryError):
        service.start(_prepared_streaming_session(tmp_path), object())

    assert processor_factory_calls == []
    assert service.active_session_id is None


@pytest.mark.parametrize("first_close_fails", [False, True])
def test_hostile_queue_access_after_hardware_is_guarded_and_retryable(
    tmp_path, first_close_fails
):
    lifecycle = []

    class Processor:
        def __init__(self):
            self._queue = queue.SimpleQueue()
            self.queue_reads = 0
            self.hardware_quiescence_diagnostic = "close failed"
            self.claimed = False

        @property
        def audio_queue(self):
            self.queue_reads += 1
            if self.queue_reads == 1:
                raise _HostileBoundaryError()
            return self._queue

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            if self.claimed:
                return False
            self.claimed = True
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("close-attempt")
            return not first_close_fails

        def retry_terminal_quiescence(self):
            lifecycle.append("retry-close")
            return True

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        queued_delivery=False,
    )

    with pytest.raises(_HostileBoundaryError):
        service.start(prepared, object())

    assert lifecycle.count(("rollback",)) == 1
    if first_close_fails:
        assert service.active_session_id == "session-1"
        handle = service.close_admission(prepared)
        assert service.quiesce(prepared, "retry", handle) == {"quiesced": True}
        assert lifecycle.count("retry-close") == 1
    assert service.active_session_id is None


def test_hostile_start_result_is_rejected_without_running_user_hooks(tmp_path):
    lifecycle = []

    class HostileResult:
        def __iter__(self):
            raise _HostileBoundaryError()

        def __bool__(self):
            raise _HostileBoundaryError()

    class Processor:
        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return HostileResult()

        def begin_cancel_streaming(self):
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("hardware-closed")
            return True

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        queued_delivery=False,
    )

    with pytest.raises(RuntimeError, match="invalid result"):
        service.start(_prepared_streaming_session(tmp_path), object())

    assert lifecycle == [
        "hardware-started",
        "gate-closed",
        "hardware-closed",
        ("rollback",),
    ]
    assert service.active_session_id is None


def test_alignment_file_write_and_full_result_build_stay_off_qt_thread(tmp_path):
    global _QT_APP_HOLDER
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance() or QApplication([])
    _QT_APP_HOLDER = app
    main_thread = threading.get_ident()
    alignment_threads = []
    save_threads = []
    terminal_threads = []
    done = threading.Event()
    prepared = _prepared_streaming_session(
        tmp_path,
        mode="PLAY_AND_RECORD",
        stimulus_data=np.arange(4, dtype=np.float32),
    )

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_playrec(self, **_kwargs):
            self.audio_queue.put(_batch(0, 0, [[1.0], [2.0], [3.0], [4.0]]))
            self.audio_queue.put(AudioCompleted("session-1", 0, 4))
            return error_code.OK, "started"

    class Terminal:
        def staged_recording_ready(self, staged):
            terminal_threads.append(threading.get_ident())
            assert staged.data_struct_fields["store_wave_data"].flags.writeable is False
            done.set()
            return True

    def align(_reference, recorded):
        alignment_threads.append(threading.get_ident())
        return np.array(recorded, copy=True)

    def save(*_args, **_kwargs):
        save_threads.append(threading.get_ident())

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=True,
        alignment=align,
        save_aligned_audio=save,
    )

    assert service.start(prepared, Terminal())
    deadline = time.monotonic() + 2
    while not done.is_set() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)

    assert done.is_set()
    assert alignment_threads and alignment_threads[0] != main_thread
    assert save_threads and save_threads[0] != main_thread
    assert terminal_threads == [main_thread]


def test_waveform_long_stream_avoids_cumulative_concatenate(monkeypatch):
    scheduled = []
    paints = []
    view = SequenceRecordingView(
        plot_recording=lambda signal, _rate: paints.append(np.array(signal)),
        schedule_waveform_refresh=scheduled.append,
    )
    view.begin_streaming_session("session-1", 10.0)
    monkeypatch.setattr(
        np,
        "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("waveform refresh must not concatenate full history")
        ),
    )

    for index in range(200):
        assert view.queue_recording_batch(_batch(index, index, [[float(index)]]))
    assert len(scheduled) == 1
    scheduled.pop()()

    assert len(paints) == 1
    np.testing.assert_allclose(paints[0], np.arange(200, dtype=np.float32))
    assert view.waveform_display_cursor == 200


def test_waveform_reentrant_paint_and_session_end_are_safe():
    scheduled = []
    paints = []
    view = None

    def paint(signal, _rate):
        paints.append(np.array(signal))
        if len(paints) == 1:
            assert view.queue_recording_batch(_batch(1, 1, [[2.0]]))

    view = SequenceRecordingView(
        plot_recording=paint,
        schedule_waveform_refresh=scheduled.append,
    )
    view.begin_streaming_session("session-1", 10.0)
    view.queue_recording_batch(_batch(0, 0, [[1.0]]))
    scheduled.pop(0)()
    assert len(scheduled) == 1
    stale = scheduled.pop(0)
    view.end_streaming_session("session-1")
    stale()

    assert len(paints) == 1


def test_service_logs_all_stale_event_kinds_before_ignoring_new_session(tmp_path):
    logs = []
    cancelled = threading.Event()

    class Logger:
        def debug(self, message):
            logs.append(message)

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            return True

        def finish_cancel_streaming(self, reason):
            self.audio_queue.put(AudioCancelled("session-new", -1, reason))
            return True

    class Terminal:
        def recording_cancelled(self, _reason):
            cancelled.set()
            return True

    prepared = _prepared_streaming_session(tmp_path, session_id="session-new")
    view = SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None)
    service = SequenceStreamingRecordingService(
        view=view,
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
        logger=Logger(),
    )
    assert service.start(prepared, Terminal())

    service._on_batch_ready(_batch(0, 0, [[1.0]], session_id="session-old"))
    service._on_completed(
        StreamingRecordingResult(
            "session-old",
            -1,
            0,
            (0,),
            np.empty(0),
            np.empty((0, 1)),
        )
    )
    service._on_failed(
        StreamingRecordingFailure(
            "session-old", "old", "old failure", 0, {}
        )
    )
    service._on_cancelled(
        StreamingRecordingCancellation("session-old", "old cancel", 0, {})
    )

    assert service.active_session_id == "session-new"
    assert view.waveform_display_cursor == 0
    assert all("session-old" in message for message in logs)
    assert {kind for kind in ("batch", "completed", "failed", "cancelled") if any(kind in message for message in logs)} == {
        "batch",
        "completed",
        "failed",
        "cancelled",
    }
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "done", handle) == {"quiesced": True}
    assert cancelled.wait(1)


def test_service_retains_session_when_close_fails_then_retires_after_retry(tmp_path):
    finish_results = iter((False, True))
    terminal = threading.Event()

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.hardware_quiescence_diagnostic = "close failed"

        def start_streaming_rec(self, **_kwargs):
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            return True

        def finish_cancel_streaming(self, reason):
            result = next(finish_results)
            if result:
                self.audio_queue.put(AudioCancelled("session-1", -1, reason))
            return result

    class Terminal:
        def recording_cancelled(self, _reason):
            terminal.set()
            return True

    prepared = _prepared_streaming_session(tmp_path)
    view = SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None)
    service = SequenceStreamingRecordingService(
        view=view,
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    assert service.start(prepared, Terminal())
    handle = service.close_admission(prepared)

    first = service.quiesce(prepared, "operator", handle)
    assert first["quiesced"] is False
    assert "close failed" in first["diagnostic"]
    assert service.active_session_id == "session-1"
    assert service.quiesce(prepared, "operator", handle) == {"quiesced": True}
    assert terminal.wait(1)
    assert service.active_session_id is None


def test_hostile_queue_drain_keeps_quiesced_session_retryable(tmp_path):
    class HostileDrainQueue:
        def __init__(self):
            self.backing = queue.SimpleQueue()
            self.fail_drain_once = True

        def put(self, value):
            self.backing.put(value)

        def get(self):
            return self.backing.get()

        def get_nowait(self):
            if self.fail_drain_once:
                self.fail_drain_once = False
                raise _HostileBoundaryError()
            return self.backing.get_nowait()

    class Processor:
        def __init__(self):
            self.audio_queue = HostileDrainQueue()
            self.claimed = False

        def start_streaming_rec(self, **_kwargs):
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            self.claimed = True
            return True

        def finish_cancel_streaming(self, reason):
            self.audio_queue.put(AudioCancelled("session-1", -1, reason))
            return True

        def retry_terminal_quiescence(self):
            return True

    class Terminal:
        def recording_cancelled(self, _reason):
            return True

    prepared = _prepared_streaming_session(tmp_path)
    view = SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None)
    service = SequenceStreamingRecordingService(
        view=view,
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
    )
    assert service.start(prepared, Terminal())
    handle = service.close_admission(prepared)

    first = service.quiesce(prepared, "operator", handle)
    assert first["quiesced"] is False
    assert type(first["diagnostic"]) is str
    assert service.active_session_id == "session-1"
    assert service.quiesce(prepared, "operator", handle) == {
        "quiesced": True
    }
    assert service.active_session_id is None


def test_consumer_thread_start_failure_quiesces_started_hardware_and_rolls_back(
    tmp_path,
):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("hardware-closed")
            return True

    class FailedThread:
        def start(self):
            raise RuntimeError("thread start failed")

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        thread_factory=lambda **_kwargs: FailedThread(),
        queued_delivery=False,
    )

    try:
        service.start(_prepared_streaming_session(tmp_path), object())
    except RuntimeError as error:
        assert str(error) == "thread start failed"
    else:
        raise AssertionError("consumer thread start failure must escape startup")
    assert lifecycle[0:3] == [
        "hardware-started",
        "gate-closed",
        "hardware-closed",
    ]
    assert lifecycle[-1] == ("rollback",)
    assert service.active_session_id is None


def test_worker_construction_failure_after_hardware_start_is_cleaned(tmp_path):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("hardware-closed")
            return True

    def fail_worker(**_kwargs):
        raise RuntimeError("worker construction failed")

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        worker_factory=fail_worker,
        queued_delivery=False,
    )

    try:
        service.start(_prepared_streaming_session(tmp_path), object())
    except RuntimeError as error:
        assert str(error) == "worker construction failed"
    else:
        raise AssertionError("worker construction failure must escape startup")
    assert lifecycle == [
        "hardware-started",
        "gate-closed",
        "hardware-closed",
        ("rollback",),
    ]
    assert service.active_session_id is None


def test_processor_start_failure_retains_unclosed_hardware_until_task5_retry(
    tmp_path,
):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.hardware_quiescence_diagnostic = "close failed"
            self._claimed = False

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            raise RuntimeError("device start failed")

        def begin_cancel_streaming(self):
            if self._claimed:
                return False
            self._claimed = True
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("initial-close-failed")
            return False

        def retry_terminal_quiescence(self):
            lifecycle.append("retry-close-succeeded")
            return True

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        queued_delivery=False,
    )

    try:
        service.start(prepared, object())
    except RuntimeError as error:
        assert str(error) == "device start failed"
    else:
        raise AssertionError("processor start failure must escape startup")

    assert service.active_session_id == "session-1"
    assert service.start(prepared, object()) is False
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "start failed", handle) == {
        "quiesced": True
    }
    assert service.active_session_id is None
    assert lifecycle == [
        "hardware-started",
        "gate-closed",
        "initial-close-failed",
        ("rollback",),
        "retry-close-succeeded",
    ]


def test_processor_non_ok_hostile_message_cannot_skip_startup_cleanup(tmp_path):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-open")
            return error_code.INVALID_RECORD, _HostileBoundaryError()

        def begin_cancel_streaming(self):
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("hardware-closed")
            return True

    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        queued_delivery=False,
    )

    with pytest.raises(RuntimeError):
        service.start(_prepared_streaming_session(tmp_path), object())

    assert lifecycle == [
        "hardware-open",
        "gate-closed",
        "hardware-closed",
        ("rollback",),
    ]
    assert service.active_session_id is None


def test_thread_factory_construction_failure_retains_unclosed_started_hardware(
    tmp_path,
):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.hardware_quiescence_diagnostic = "close failed"
            self.claimed = False

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            if self.claimed:
                return False
            self.claimed = True
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("initial-close-failed")
            return False

        def retry_terminal_quiescence(self):
            lifecycle.append("retry-close-succeeded")
            return True

    def fail_thread_factory(**_kwargs):
        raise SystemExit("thread construction interrupted")

    prepared = _prepared_streaming_session(tmp_path)
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        thread_factory=fail_thread_factory,
        queued_delivery=False,
    )

    with pytest.raises(SystemExit, match="thread construction interrupted"):
        service.start(prepared, object())

    assert service.active_session_id == "session-1"
    assert lifecycle == [
        "hardware-started",
        "gate-closed",
        "initial-close-failed",
        ("rollback",),
    ]
    handle = service.close_admission(prepared)
    assert service.quiesce(prepared, "retry", handle) == {"quiesced": True}
    assert service.active_session_id is None
    assert lifecycle[-1] == "retry-close-succeeded"


@pytest.mark.parametrize(
    "failure_stage", ["worker", "signal", "thread-factory", "thread-start"]
)
@pytest.mark.parametrize("first_close_fails", [False, True])
def test_post_hardware_start_transaction_contains_hostile_stage_failure(
    tmp_path,
    failure_stage,
    first_close_fails,
):
    lifecycle = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.hardware_quiescence_diagnostic = "close interrupted"
            self.claimed = False

        def start_streaming_rec(self, **_kwargs):
            lifecycle.append("hardware-started")
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            if self.claimed:
                return False
            self.claimed = True
            lifecycle.append("gate-closed")
            return True

        def finish_cancel_streaming(self, _reason):
            lifecycle.append("close-attempt")
            return not first_close_fails

        def retry_terminal_quiescence(self):
            lifecycle.append("retry-close")
            return True

    class FailedThread:
        def start(self):
            raise _HostileBoundaryError()

    def worker_factory(**kwargs):
        if failure_stage == "worker":
            raise _HostileBoundaryError()
        return StreamingRecordingWorker(**kwargs)

    def thread_factory(**kwargs):
        if failure_stage == "thread-factory":
            raise _HostileBoundaryError()
        if failure_stage == "thread-start":
            return FailedThread()
        return threading.Thread(**kwargs)

    prepared = _prepared_streaming_session(tmp_path)
    view = SequenceRecordingView(schedule_waveform_refresh=lambda _callback: None)
    service = SequenceStreamingRecordingService(
        view=view,
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(lifecycle),
        worker_factory=worker_factory,
        thread_factory=thread_factory,
        queued_delivery=False,
    )
    if failure_stage == "signal":
        service._connect_worker = lambda _worker: (_ for _ in ()).throw(
            _HostileBoundaryError()
        )

    with pytest.raises(_HostileBoundaryError):
        service.start(prepared, object())

    assert lifecycle.count(("rollback",)) == 1
    if first_close_fails:
        assert service.active_session_id == "session-1"
        handle = service.close_admission(prepared)
        assert service.quiesce(prepared, "retry", handle) == {
            "quiesced": True
        }
        assert lifecycle.count("retry-close") == 1
    assert service.active_session_id is None
    assert view.queue_recording_batch(_batch(0, 0, [[1.0]])) is False


def test_streaming_service_owns_one_consumer_and_submits_staged_result(tmp_path):
    class FakeProcessor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()
            self.cancelled = False

        def start_streaming_rec(self, **kwargs):
            assert kwargs["session_id"] == "session-1"
            self.audio_queue.put(_batch(0, 0, [[1.0], [2.0]]))
            self.audio_queue.put(_batch(1, 2, [[3.0], [4.0]]))
            self.audio_queue.put(AudioCompleted("session-1", 1, 4))
            return error_code.OK, "started"

        def begin_cancel_streaming(self):
            self.cancelled = True
            return True

        def finish_cancel_streaming(self, _reason):
            return True

        def wait_for_terminal(self):
            return None

    class Terminal:
        def __init__(self):
            self.staged = []
            self.failures = []
            self.cancelled = []
            self.done = threading.Event()

        def staged_recording_ready(self, staged):
            self.staged.append(staged)
            self.done.set()
            return True

        def recording_failed(self, reason):
            self.failures.append(reason)
            self.done.set()
            return True

        def recording_cancelled(self, reason):
            self.cancelled.append(reason)
            self.done.set()
            return True

    writes = []
    view = SequenceRecordingView(
        schedule_waveform_refresh=lambda callback: callback()
    )
    terminal = Terminal()
    service = SequenceStreamingRecordingService(
        view=view,
        processor_factory=FakeProcessor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(writes),
        queued_delivery=False,
    )

    assert service.start(_prepared_streaming_session(tmp_path), terminal) is True
    assert terminal.done.wait(1)
    assert terminal.failures == []
    assert len(terminal.staged) == 1
    staged = terminal.staged[0]
    assert staged.snapshot.session_id == "session-1"
    assert staged.sample_count == 4
    np.testing.assert_allclose(staged.data_struct_fields["store_wave_data"], [1, 2, 3, 4])
    np.testing.assert_allclose(
        staged.data_struct_fields["store_wave_data_multi"], [[1], [2], [3], [4]]
    )
    assert [name for name, *_rest in writes] == ["write", "write", "finalize"]
    assert service.active_session_id is None


def test_streaming_consumer_thread_uses_only_frozen_session_values(tmp_path):
    source_samples = np.arange(4, dtype=np.float32)
    source_context = {
        "detail": {"callback_block_size": 64},
        "recorded_dict": {
            "prepare_frames": 0,
            "prolong_frames": 0,
            "recording_start_delay_frames": 0,
        },
        "recorded_signal_info": {
            "file_path": str(tmp_path / "record.wav"),
            "metadata": {"serial": "SN-FROZEN"},
        },
        "stimulus_info": {"steps": [{"hz": 1_000}]},
        "stimulus_data": source_samples,
        "stimulus_dict": {
            "data": source_samples,
            "steps": [{"hz": 1_000}],
            "amplitude": 1.0,
        },
    }
    base = _prepared_streaming_session(
        tmp_path,
        mode="PLAY_AND_RECORD",
        stimulus_data=source_samples,
    )
    prepared = PreparedRecordingSession(base.snapshot, source_context)
    source_samples[:] = -1
    source_context["stimulus_dict"]["steps"][0]["hz"] = 9_999
    source_context["recorded_signal_info"]["metadata"]["serial"] = "MUTATED"
    processor_inputs = []

    class Processor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_playrec(self, **kwargs):
            processor_inputs.append(kwargs)
            kwargs["stimulus_dict"]["data"][:] = 77
            kwargs["stimulus_dict"]["steps"][0]["hz"] = 8_888
            kwargs["input_device"]["index"] = 999
            returned = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
            self.audio_queue.put(_batch(0, 0, returned))
            returned[:] = -500
            self.audio_queue.put(AudioCompleted("session-1", 0, 4))
            return error_code.OK, "started"

    class Terminal:
        def __init__(self):
            self.staged = None
            self.done = threading.Event()

        def staged_recording_ready(self, staged):
            self.staged = staged
            self.done.set()
            return True

    terminal = Terminal()
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(
            schedule_waveform_refresh=lambda callback: callback()
        ),
        processor_factory=Processor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter([]),
        queued_delivery=False,
        alignment=lambda _reference, recorded: np.array(recorded, copy=True),
        save_aligned_audio=lambda *_args, **_kwargs: None,
    )

    assert service.start(prepared, terminal) is True
    assert terminal.done.wait(1)
    assert processor_inputs[0]["stimulus_dict"] is not prepared.acquisition_context[
        "stimulus_dict"
    ]
    assert prepared.snapshot.input_device["index"] == 1
    assert prepared.acquisition_context["stimulus_dict"]["data"].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert prepared.acquisition_context["stimulus_dict"]["steps"][0]["hz"] == 1_000
    assert prepared.acquisition_context["recorded_signal_info"]["metadata"][
        "serial"
    ] == "SN-FROZEN"
    np.testing.assert_allclose(
        terminal.staged.data_struct_fields["store_wave_data"],
        [1.0, 2.0, 3.0, 4.0],
    )
    with pytest.raises(ValueError):
        terminal.staged.data_struct_fields["store_wave_data"][0] = 12


def test_streaming_worker_canonicalizes_forged_finalizer_result_before_emit(tmp_path):
    prepared = _prepared_streaming_session(tmp_path)
    source = np.arange(4, dtype=np.float32)
    nested = {"labels": ["original"], "samples": source}
    info = {"metadata": {"serial": "SN-1"}}
    forged = object.__new__(StagedRecording)
    object.__setattr__(forged, "snapshot", prepared.snapshot)
    object.__setattr__(forged, "sample_count", 4)
    object.__setattr__(forged, "data_struct_fields", nested)
    object.__setattr__(forged, "recorded_signal_info", info)
    object.__setattr__(forged, "stimulus_info", None)
    object.__setattr__(forged, "warnings", ())
    messages = queue.SimpleQueue()
    messages.put(_batch(0, 0, [[1.0], [2.0], [3.0], [4.0]]))
    messages.put(AudioCompleted("session-1", 0, 4))
    completed = []
    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=FakeWriter([]),
        channel_order=(0,),
        target_samples=4,
        finalize_result=lambda *_args: forged,
    )
    worker.completed.connect(completed.append, Qt.DirectConnection)

    worker.run()
    source[:] = -1
    nested["labels"].append("mutated")
    info["metadata"]["serial"] = "MUTATED"

    staged = completed[0].staged
    assert staged is not forged
    assert staged.data_struct_fields["labels"] == ("original",)
    assert staged.data_struct_fields["samples"].tolist() == [0, 1, 2, 3]
    assert staged.recorded_signal_info["metadata"]["serial"] == "SN-1"


def test_streaming_worker_rebuilds_poisoned_factory_result_before_emit(tmp_path):
    prepared = _prepared_streaming_session(tmp_path)
    source = np.arange(4, dtype=np.float32)
    fields = {"labels": ["original"], "samples": source}
    poisoned = StagedRecording.create(
        snapshot=prepared.snapshot,
        sample_count=4,
        data_struct_fields={},
        recorded_signal_info={},
    )
    object.__setattr__(poisoned, "data_struct_fields", fields)
    messages = queue.SimpleQueue()
    messages.put(_batch(0, 0, [[1.0], [2.0], [3.0], [4.0]]))
    messages.put(AudioCompleted("session-1", 0, 4))
    completed = []
    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=FakeWriter([]),
        channel_order=(0,),
        target_samples=4,
        finalize_result=lambda *_args: poisoned,
    )
    worker.completed.connect(completed.append, Qt.DirectConnection)

    worker.run()
    source[:] = -1
    fields["labels"].append("mutated")

    rebuilt = completed[0].staged
    assert rebuilt is not poisoned
    assert rebuilt.data_struct_fields["labels"] == ("original",)
    assert rebuilt.data_struct_fields["samples"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("returned", [False, object()])
def test_streaming_worker_rejects_non_staged_finalizer_result(returned):
    messages = queue.SimpleQueue()
    messages.put(_batch(0, 0, [[1.0], [2.0], [3.0], [4.0]]))
    messages.put(AudioCompleted("session-1", 0, 4))
    completed = []
    failures = []
    worker = StreamingRecordingWorker(
        session_id="session-1",
        message_queue=messages,
        writer=FakeWriter([]),
        channel_order=(0,),
        target_samples=4,
        finalize_result=lambda *_args: returned,
    )
    worker.completed.connect(completed.append, Qt.DirectConnection)
    worker.failed.connect(failures.append, Qt.DirectConnection)

    worker.run()

    assert completed == []
    assert len(failures) == 1
    assert failures[0].code == "finalization-failed"


def test_sequence_window_composes_formal_streaming_service_ports():
    source = Path("ui/sequence/sequence_widget.py").read_text(encoding="utf-8")
    assert "SequenceStreamingRecordingService(" in source
    assert "streaming_adapter=self.streaming_recording_service.start" in source
    assert "self._start_admitted_legacy_recording(admission, terminal)" not in source


def test_streaming_service_forwards_fatal_rollback_outcome_to_controller_port(tmp_path):
    class FailedProcessor:
        def __init__(self):
            self.audio_queue = queue.SimpleQueue()

        def start_streaming_rec(self, **_kwargs):
            self.audio_queue.put(
                AudioFailed("session-1", -1, "allocation-failed", "oom")
            )
            return error_code.OK, "started"

    class Terminal:
        def __init__(self):
            self.failure = None
            self.done = threading.Event()

        def recording_failed(self, reason, rollback_outcome=None):
            self.failure = (reason, rollback_outcome)
            self.done.set()
            return True

    writes = []
    terminal = Terminal()
    service = SequenceStreamingRecordingService(
        view=SequenceRecordingView(),
        processor_factory=FailedProcessor,
        writer_factory=lambda *_args, **_kwargs: FakeWriter(writes),
        queued_delivery=False,
    )

    assert service.start(_prepared_streaming_session(tmp_path), terminal)
    assert terminal.done.wait(1)
    assert terminal.failure == ("oom", {"restored": True, "errors": ()})
    assert writes == [("rollback",)]
