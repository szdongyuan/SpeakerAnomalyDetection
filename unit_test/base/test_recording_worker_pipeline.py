from types import SimpleNamespace
import multiprocessing
from pathlib import Path
import queue
import threading
import time

import pytest

from base.recording_process_protocol import RecordingEvent, RecordingResult, WorkerFatal
from base.recording_worker import _send_loop, recording_worker
from unit_test.base.ve3668n_fakes import capture_request


def capture(request_id):
    return SimpleNamespace(request=SimpleNamespace(request_id=request_id))


def test_capacity_two_moves_released_capture_to_request_keyed_finalizer():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    capture_a = capture("A")
    capture_b = capture("B")
    pipeline.start("A", capture_a)
    pipeline.capture_released("A")
    pipeline.start("B", capture_b)

    assert pipeline.active.request_id == "B"
    assert pipeline.active.capture is capture_b
    assert set(pipeline.finalizers) == {"A"}
    assert pipeline.finalizers["A"].capture is capture_a


def test_duplicate_and_third_request_are_rejected_without_changing_state():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture("A"))
    with pytest.raises(ValueError, match="duplicate request ID"):
        pipeline.start("A", capture("A"))
    pipeline.capture_released("A")
    pipeline.start("B", capture("B"))

    with pytest.raises(RuntimeError, match="capacity"):
        pipeline.start("C", capture("C"))
    assert pipeline.active.request_id == "B"
    assert set(pipeline.finalizers) == {"A"}


def test_result_ack_for_finalizer_never_clears_new_active_capture():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture("A"))
    pipeline.capture_released("A")
    pending = pipeline.mark_terminal("A")
    pipeline.start("B", capture("B"))

    acknowledged = pipeline.result_ack("A", "accepted")

    assert not hasattr(pending, "capture")
    assert acknowledged.request_id == "A"
    assert pipeline.active.request_id == "B"
    assert not pipeline.finalizers
    assert not pipeline.pending_result_acks


def test_terminal_is_retained_until_matching_ack_and_ack_requires_terminal():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture("A"))
    pipeline.capture_released("A")

    with pytest.raises(RuntimeError, match="before terminal"):
        pipeline.result_ack("A", "accepted")
    terminal = pipeline.mark_terminal("A")
    assert terminal.request_id == "A"
    assert not hasattr(terminal, "capture")
    assert "A" not in pipeline.finalizers
    assert pipeline.pending_result_acks["A"] is terminal
    with pytest.raises(KeyError, match="unknown request ID"):
        pipeline.result_ack("B", "accepted")


def test_shutdown_snapshot_is_bounded_and_does_not_expose_mutable_table():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture("A"))
    pipeline.capture_released("A")
    pipeline.start("B", capture("B"))

    snapshot = pipeline.shutdown_snapshot()

    assert isinstance(snapshot, tuple)
    assert [state.request_id for state in snapshot] == ["B", "A"]
    pipeline.mark_terminal("A")
    assert [state.request_id for state in pipeline.shutdown_snapshot()] == ["B"]
    pipeline.result_ack("A", "rejected")
    assert [state.request_id for state in snapshot] == ["B", "A"]


def test_sound_card_terminal_can_remain_active_until_matching_ack():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("sound", capture("sound"))
    pipeline.mark_terminal("sound")

    assert pipeline.active is None
    assert set(pipeline.pending_result_acks) == {"sound"}
    pipeline.result_ack("sound", "rejected")
    assert pipeline.active is None
    assert pipeline.empty


def test_pending_ack_counts_toward_capacity_without_retaining_capture():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    capture_a = capture("A")
    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture_a)
    pending = pipeline.mark_terminal("A")
    pipeline.start("B", capture("B"))

    assert not hasattr(pending, "capture")
    assert all(value is not capture_a for value in vars(pending).values())
    with pytest.raises(RuntimeError, match="capacity"):
        pipeline.start("C", capture("C"))


def test_duplicate_capture_release_is_a_protocol_error_and_keeps_finalizer():
    from base.recording_worker_pipeline import WorkerCapturePipeline

    pipeline = WorkerCapturePipeline()
    pipeline.start("A", capture("A"))
    pipeline.capture_released("A")
    with pytest.raises(KeyError, match="not owned"):
        pipeline.capture_released("A")
    assert set(pipeline.finalizers) == {"A"}


def _spawn_worker(tmp_path, *, target=recording_worker, worker_cancel_timeout=.5, **options):
    context = multiprocessing.get_context("spawn")
    control, child_control = context.Pipe(duplex=True)
    preview, child_preview = context.Pipe(duplex=False)
    trace_dir = tmp_path / "worker-trace"
    process = context.Process(
        target=target,
        args=(child_control, child_preview, 1,
              "unit_test.base.recording_process_fakes:persistent_ve_worker_dependencies",
              {"trace_dir": str(trace_dir), **options}, worker_cancel_timeout, .05),
    )
    process.start()
    child_control.close()
    child_preview.close()
    return process, control, preview, trace_dir


def _receive_until(control, predicate, *, timeout=10):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if control.poll(.05):
            event = control.recv()
            events.append(event)
            if predicate(event):
                return events
    raise AssertionError(f"worker event did not arrive; received {[item.kind for item in events]}")


def _stop_worker(process, control, preview):
    if process.is_alive():
        try:
            control.send(RecordingEvent(1, "", "shutdown"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        process.join(3)
    if process.is_alive():
        process.terminate()
        process.join(3)
    control.close()
    preview.close()
    process.close()


def _wait_for_path(path, *, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists()


def test_control_sender_prioritizes_generation_fatal_after_blocked_send():
    first_send_entered = threading.Event()
    release_first_send = threading.Event()
    sent = []
    normal = queue.Queue()
    fatal = queue.Queue(maxsize=1)
    progress = queue.Queue(maxsize=1)
    wake = threading.Event()
    broken = threading.Event()

    class BlockedConnection:
        def send(self, event):
            if not sent:
                first_send_entered.set()
                assert release_first_send.wait(2)
            sent.append(event)

    first = RecordingEvent(1, "A", "started", 1.0)
    second = RecordingEvent(1, "A", "finalizing")
    generation_fatal = RecordingEvent(
        1, "", "worker_fatal", WorkerFatal(
            1, "worker/control_queue", "control queue saturated (Full)"))
    normal.put(first)
    sender = threading.Thread(
        target=_send_loop,
        args=(BlockedConnection(), normal, broken, progress, wake, fatal),
        daemon=True,
    )
    sender.start()
    assert first_send_entered.wait(2)
    normal.put(second)
    fatal.put(generation_fatal)
    normal.put(None)
    wake.set()

    release_first_send.set()
    sender.join(2)

    assert not sender.is_alive()
    assert sent == [first, generation_fatal, second]
    assert not broken.is_set()


def test_generation_fatal_diagnostic_names_empty_exception_type():
    from base import recording_worker as module

    event = module._generation_fatal_event(
        3, "worker/control_queue", queue.Full())

    assert event.request_id == ""
    assert event.kind == "worker_fatal"
    assert event.payload.stage == "worker/control_queue"
    assert event.payload.message == "worker/control_queue failed (Full)"


def test_two_request_lifecycle_burst_delivers_one_fatal_when_control_queue_saturates(
        tmp_path, monkeypatch):
    from base import recording_worker as module
    from base.ve3668n_capture_timing import VeCaptureProgress

    requests = (
        capture_request(tmp_path / "A.wav", request_id="A"),
        capture_request(tmp_path / "B.wav", request_id="B"),
    )
    commands = queue.Queue()
    for request in requests:
        commands.put(RecordingEvent(1, request.request_id, "start", request))
    first_send_entered = threading.Event()
    release_first_send = threading.Event()
    fatal_constructed = threading.Event()
    commands_enabled = threading.Event()
    sent = []

    class FinishedCapture:
        def __init__(self, request, **_dependencies):
            self.request = request
            self.started = threading.Event()
            self.done = threading.Event()
            self.capture_slot_released = threading.Event()
            self.started_at = time.monotonic()
            self.raw_frames = request.target_samples
            self.capture_slot = SimpleNamespace(
                target_reached_at=self.started_at,
                raw_frames=request.target_samples,
                adapter_released=True,
                writer_released=True,
            )
            self.outcome = RecordingResult(
                request.request_id, request.purpose, request.path, request.sample_rate,
                request.channels, request.target_samples, request.target_samples - 2, True)

        def start(self):
            self.started.set()
            self.capture_slot_released.set()
            self.done.set()

        def progress_snapshot(self):
            return VeCaptureProgress(
                self.started_at, self.request.target_samples, self.started_at)

        def snapshot(self, **_kwargs):
            return None

        def cancel(self):
            self.done.set()

    class BlockedControl:
        def poll(self, _timeout):
            return commands_enabled.is_set() and not commands.empty()

        def recv(self):
            return commands.get_nowait()

        def send(self, event):
            if not sent:
                first_send_entered.set()
                assert release_first_send.wait(2)
            sent.append(event)

        def close(self):
            pass

    class PreviewSink:
        def send(self, _event):
            pass

        def close(self):
            pass

    original_fatal_event = module._generation_fatal_event

    def observe_fatal(generation, stage, source):
        event = original_fatal_event(generation, stage, source)
        fatal_constructed.set()
        return event

    monkeypatch.setattr(module, "RecordingCapture", FinishedCapture)
    monkeypatch.setattr(module, "_generation_fatal_event", observe_fatal)
    worker = threading.Thread(
        target=module.recording_worker,
        args=(BlockedControl(), PreviewSink(), 1, None, {}, .5, .05),
        daemon=True,
    )
    worker.start()
    try:
        assert first_send_entered.wait(2)
        commands_enabled.set()
        assert fatal_constructed.wait(2), "two lifecycles must saturate normal control traffic"
        release_first_send.set()
        worker.join(2)

        assert not worker.is_alive(), "queue saturation must enter bounded retirement"
        fatals = [event for event in sent if event.kind == "worker_fatal"]
        assert len(fatals) == 1
        assert fatals[0].request_id == ""
        assert fatals[0].payload.stage == "worker/control_queue"
        assert fatals[0].payload.message == "worker/control_queue failed (Full)"
        assert sent[0].kind == "ready"
        assert sent[1] is fatals[0]
        assert [event.kind for event in sent[2:7]] == [
            "started", "finalizing", "progress", "capture_slot_released", "completed"]
        assert [event.request_id for event in sent[2:7]] == ["A"] * 5
    finally:
        release_first_send.set()
        if worker.is_alive():
            commands.put(RecordingEvent(1, "", "shutdown"))
            worker.join(2)


def test_spawn_overlap_routes_finalizer_ack_without_clearing_active_capture(tmp_path):
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, pause_finalizers=("A",), pause_writers=("B",))
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request_a = capture_request(tmp_path / "A.wav", request_id="A")
        control.send(RecordingEvent(1, "A", "start", request_a))
        events_a = _receive_until(
            control, lambda event: event.kind == "capture_slot_released" and event.request_id == "A")
        final_progress = next(index for index, event in enumerate(events_a)
                              if event.kind == "progress" and event.request_id == "A"
                              and event.payload.frames == request_a.target_samples)
        slot_index = next(index for index, event in enumerate(events_a)
                          if event.kind == "capture_slot_released" and event.request_id == "A")
        slot = events_a[slot_index].payload
        assert final_progress < slot_index
        assert slot.target_reached_at == events_a[final_progress].payload.last_frame_at
        assert slot.adapter_released and slot.writer_released
        assert slot.lifecycle_counts.task_stop == slot.lifecycle_counts.task_clear == 0
        assert slot.lifecycle_counts.sdk_close == 0
        _wait_for_path(trace_dir / "finalizer-A-entered")

        request_b = capture_request(tmp_path / "B.wav", request_id="B")
        control.send(RecordingEvent(1, "B", "start", request_b))
        _receive_until(control, lambda event: event.kind == "started" and event.request_id == "B")
        _wait_for_path(trace_dir / "writer-B-entered")
        _receive_until(control, lambda event: event.kind == "finalizing" and event.request_id == "B")

        control.send(RecordingEvent(1, "A", "cancel"))
        (trace_dir / "release-finalizer-A").touch()
        _receive_until(control, lambda event: event.kind == "completed" and event.request_id == "A")
        control.send(RecordingEvent(1, "A", "result_ack", "accepted"))
        (trace_dir / "release-writer-B").touch()
        _receive_until(
            control, lambda event: event.kind == "capture_slot_released" and event.request_id == "B")
        _receive_until(control, lambda event: event.kind == "completed" and event.request_id == "B")
        control.send(RecordingEvent(1, "B", "result_ack", "accepted"))
    finally:
        _stop_worker(process, control, preview)


def test_spawn_idle_fatal_from_native_failure_emits_worker_fatal(tmp_path):
    process, control, preview, trace_dir = _spawn_worker(tmp_path, idle_fail_after_reads=1)
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "idle-fatal.wav", request_id="idle-fatal")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(control, lambda event: event.kind == "capture_slot_released")
        (trace_dir / "trigger-idle-failure").touch()
        events = _receive_until(control, lambda event: event.kind == "worker_fatal")
        fatal = events[-1]
        assert fatal.request_id == ""
        assert fatal.payload.generation == 1
        assert fatal.payload.stage == "read_task_data"
        assert "idle native read failure" in fatal.payload.message
    finally:
        _stop_worker(process, control, preview)


@pytest.mark.parametrize("failure", [None, "stop_task", "clear_task"])
def test_spawn_release_ve_reports_typed_success_or_ordered_failure(tmp_path, failure):
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, fail_release_operation=failure)
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "release.wav", request_id="release")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(control, lambda event: event.kind == "completed")
        control.send(RecordingEvent(1, request.request_id, "result_ack", "accepted"))
        control.send(RecordingEvent(1, "", "release_ve"))
        expected = "ve_released" if failure is None else "ve_release_failed"
        outcome = _receive_until(control, lambda event: event.kind == expected)[-1].payload
        assert outcome.generation == 1
        assert outcome.released_signature is not None
        assert bool(outcome.diagnostics) is (failure is not None)
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations[-3:] == ["stop_task", "clear_task", "close"]
    finally:
        _stop_worker(process, control, preview)


def test_spawn_shutdown_with_active_and_finalizer_is_bounded_and_releases_once(tmp_path):
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, pause_finalizers=("A",), pause_writers=("B",))
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request_a = capture_request(tmp_path / "A.wav", request_id="A")
        control.send(RecordingEvent(1, "A", "start", request_a))
        _receive_until(control, lambda event: event.kind == "capture_slot_released")
        _wait_for_path(trace_dir / "finalizer-A-entered")
        request_b = capture_request(tmp_path / "B.wav", request_id="B")
        control.send(RecordingEvent(1, "B", "start", request_b))
        _receive_until(control, lambda event: event.kind == "started" and event.request_id == "B")
        _wait_for_path(trace_dir / "writer-B-entered")

        control.send(RecordingEvent(1, "", "shutdown"))
        (trace_dir / "release-finalizer-A").touch()
        (trace_dir / "release-writer-B").touch()
        process.join(3)
        assert not process.is_alive()
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        _stop_worker(process, control, preview)


def test_spawn_control_eof_performs_bounded_controller_cleanup(tmp_path):
    process, control, preview, trace_dir = _spawn_worker(tmp_path)
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "eof.wav", request_id="eof")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(control, lambda event: event.kind == "started")
        control.close()
        preview.close()
        process.join(3)
        assert not process.is_alive()
        assert process.exitcode == 0
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
        process.close()


def test_spawn_control_eof_forces_exit_when_capture_finalizer_misses_deadline(
        tmp_path):
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, worker_cancel_timeout=.2, pause_finalizers=("eof-finalizer",))
    release = trace_dir / "release-finalizer-eof-finalizer"
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(
            tmp_path / "eof-finalizer.wav", request_id="eof-finalizer")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(
            control, lambda event: event.kind == "capture_slot_released")
        _wait_for_path(trace_dir / "finalizer-eof-finalizer-entered")

        control.close()
        preview.close()
        process.join(3)

        assert not process.is_alive(), "capture finalizer cleanup must remain bounded"
        assert process.exitcode == 1
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        release.touch()
        if process.is_alive():
            process.terminate()
            process.join(3)
        process.close()


def test_spawn_outer_exception_forces_exit_when_capture_finalizer_misses_deadline(
        tmp_path):
    from unit_test.base.recording_process_fakes import outer_exception_control_worker

    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, target=outer_exception_control_worker,
        worker_cancel_timeout=.2, pause_finalizers=("outer-finalizer",),
        fail_recv_kind="shutdown")
    release = trace_dir / "release-finalizer-outer-finalizer"
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(
            tmp_path / "outer-finalizer.wav", request_id="outer-finalizer")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(
            control, lambda event: event.kind == "capture_slot_released")
        _wait_for_path(trace_dir / "finalizer-outer-finalizer-entered")

        control.send(RecordingEvent(1, "", "shutdown"))
        fatal = _receive_until(
            control, lambda event: event.kind == "worker_fatal")[-1]
        assert fatal.payload.stage == "worker"
        process.join(3)

        assert not process.is_alive(), "outer-exception cleanup must remain bounded"
        assert process.exitcode == 1
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        release.touch()
        _stop_worker(process, control, preview)


def test_spawn_control_eof_forces_exit_when_controller_cleanup_misses_deadline(tmp_path):
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path,
        worker_cancel_timeout=.2,
        pause_release_operation="stop_task",
    )
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "eof-timeout.wav", request_id="eof-timeout")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _receive_until(control, lambda event: event.kind == "started")
        started_at = time.monotonic()
        control.close()
        preview.close()
        _wait_for_path(trace_dir / "release-stop_task-entered")
        process.join(3)
        elapsed = time.monotonic() - started_at

        assert not process.is_alive(), "control EOF cleanup must remain bounded"
        assert elapsed < 3
        assert process.exitcode == 1
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == operations.count("close") == 0
    finally:
        (trace_dir / "release-stop_task").touch()
        if process.is_alive():
            process.terminate()
            process.join(3)
        process.close()


def test_spawn_unexpected_sender_failure_sets_broken_and_releases_controller(tmp_path):
    from unit_test.base.recording_process_fakes import failing_control_worker

    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, target=failing_control_worker, fail_control_kind="started")
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "sender-failure.wav", request_id="sender-failure")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        process.join(3)
        assert not process.is_alive(), "sender failure must enter bounded stopping"
        assert process.exitcode == 0
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        _stop_worker(process, control, preview)


def test_spawn_sender_failure_forces_exit_when_controller_cleanup_misses_deadline(tmp_path):
    from unit_test.base.recording_process_fakes import failing_control_worker

    process, control, preview, trace_dir = _spawn_worker(
        tmp_path,
        target=failing_control_worker,
        worker_cancel_timeout=.2,
        fail_control_kind="started",
        pause_release_operation="stop_task",
    )
    try:
        _receive_until(control, lambda event: event.kind == "ready")
        request = capture_request(tmp_path / "sender-timeout.wav", request_id="sender-timeout")
        control.send(RecordingEvent(1, request.request_id, "start", request))
        _wait_for_path(trace_dir / "release-stop_task-entered")
        process.join(3)
        assert not process.is_alive(), "broken sender must force bounded worker exit"
        assert process.exitcode == 1, "cleanup deadline exhaustion must force the defined worker exit"
        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == operations.count("close") == 0
    finally:
        (trace_dir / "release-stop_task").touch()
        _stop_worker(process, control, preview)


@pytest.mark.parametrize("mode", [
    "duplicate_active",
    "active_slot_conflict",
    "duplicate_finalizer",
    "duplicate_pending_ack",
    "capacity_exhausted",
])
def test_spawn_invalid_start_is_one_generation_fatal_without_spurious_terminals(
        tmp_path, mode):
    pause_writers = ("A",) if mode in ("duplicate_active", "active_slot_conflict") else (
        ("B",) if mode == "capacity_exhausted" else ())
    pause_finalizers = ("A",) if mode in ("duplicate_finalizer", "capacity_exhausted") else ()
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, worker_cancel_timeout=2.0,
        pause_writers=pause_writers, pause_finalizers=pause_finalizers)
    observed = []

    def until(predicate):
        received = _receive_until(control, predicate)
        observed.extend(received)
        return received[-1]

    release_paths = [trace_dir / f"release-writer-{identity}" for identity in pause_writers]
    release_paths += [trace_dir / f"release-finalizer-{identity}" for identity in pause_finalizers]
    registered = ["A"]
    invalid_id = "A"
    try:
        until(lambda event: event.kind == "ready")
        request_a = capture_request(tmp_path / "A.wav", request_id="A")
        control.send(RecordingEvent(1, "A", "start", request_a))

        if mode in ("duplicate_active", "active_slot_conflict"):
            until(lambda event: event.kind == "started" and event.request_id == "A")
            _wait_for_path(trace_dir / "writer-A-entered")
            invalid_id = "A" if mode == "duplicate_active" else "B"
        elif mode in ("duplicate_finalizer", "capacity_exhausted"):
            until(lambda event: event.kind == "capture_slot_released" and event.request_id == "A")
            _wait_for_path(trace_dir / "finalizer-A-entered")
            if mode == "capacity_exhausted":
                request_b = capture_request(tmp_path / "B.wav", request_id="B")
                control.send(RecordingEvent(1, "B", "start", request_b))
                until(lambda event: event.kind == "started" and event.request_id == "B")
                _wait_for_path(trace_dir / "writer-B-entered")
                registered.append("B")
                invalid_id = "C"
        else:
            until(lambda event: event.kind == "completed" and event.request_id == "A")

        invalid = capture_request(tmp_path / f"invalid-{invalid_id}.wav", request_id=invalid_id)
        control.send(RecordingEvent(1, invalid_id, "start", invalid))
        fatal = until(lambda event: event.kind == "worker_fatal")
        assert fatal.request_id == ""
        assert fatal.payload.stage == "protocol/start"

        for path in release_paths:
            path.touch()
        for request_id in registered:
            if not any(event.request_id == request_id
                       and event.kind in ("completed", "failed", "cancelled")
                       for event in observed):
                until(lambda event, identity=request_id:
                      event.request_id == identity
                      and event.kind in ("completed", "failed", "cancelled"))
        process.join(3)
        assert not process.is_alive(), "protocol-fatal start rejection must retire boundedly"
        while True:
            try:
                if not control.poll(.05):
                    break
                observed.append(control.recv())
            except (BrokenPipeError, EOFError, OSError):
                break

        fatals = [event for event in observed if event.kind == "worker_fatal"]
        assert len(fatals) == 1
        assert not [event for event in observed if event.kind == "failed"]
        if invalid_id not in registered:
            assert not [event for event in observed
                        if event.request_id == invalid_id
                        and event.kind in ("completed", "failed", "cancelled")]
        terminals = [event for event in observed
                     if event.kind in ("completed", "failed", "cancelled")]
        for request_id in registered:
            matching = [event for event in terminals if event.request_id == request_id]
            assert len(matching) == 1
            expected = "completed" if mode == "duplicate_pending_ack" else "cancelled"
            assert matching[0].kind == expected
        if mode != "duplicate_pending_ack":
            fatal_index = observed.index(fatals[0])
            assert all(observed.index(event) > fatal_index for event in terminals)

        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        for path in release_paths:
            path.touch()
        _stop_worker(process, control, preview)


@pytest.mark.parametrize("mode, fatal_stage", [
    ("early_result_ack", "protocol/result_ack"),
    ("duplicate_result_ack", "protocol/result_ack"),
    ("unknown_result_ack", "protocol/result_ack"),
    ("unknown_cancel", "protocol/cancel"),
    ("unknown_preview_ack", "protocol/preview_ack"),
])
def test_spawn_invalid_command_transition_retires_generation_without_blame_shift(
        tmp_path, mode, fatal_stage):
    pause_finalizers = ("A",) if mode == "early_result_ack" else ()
    process, control, preview, trace_dir = _spawn_worker(
        tmp_path, worker_cancel_timeout=2.0,
        pause_writers=("B",), pause_finalizers=pause_finalizers)
    observed = []
    release_paths = [trace_dir / "release-writer-B"]
    if pause_finalizers:
        release_paths.append(trace_dir / "release-finalizer-A")

    def until(predicate, *, timeout=3):
        received = _receive_until(control, predicate, timeout=timeout)
        observed.extend(received)
        return received[-1]

    expected_terminals = {"B": "cancelled"}
    try:
        until(lambda event: event.kind == "ready")
        if mode in ("early_result_ack", "duplicate_result_ack"):
            request_a = capture_request(tmp_path / "A.wav", request_id="A")
            control.send(RecordingEvent(1, "A", "start", request_a))
            if mode == "early_result_ack":
                until(lambda event: event.kind == "capture_slot_released"
                      and event.request_id == "A")
                _wait_for_path(trace_dir / "finalizer-A-entered")
                expected_terminals["A"] = "cancelled"
            else:
                until(lambda event: event.kind == "completed" and event.request_id == "A")
                control.send(RecordingEvent(1, "A", "result_ack", "accepted"))
                expected_terminals["A"] = "completed"

        request_b = capture_request(tmp_path / "B.wav", request_id="B")
        control.send(RecordingEvent(1, "B", "start", request_b))
        until(lambda event: event.kind == "started" and event.request_id == "B")
        _wait_for_path(trace_dir / "writer-B-entered")

        if mode in ("early_result_ack", "duplicate_result_ack"):
            invalid = RecordingEvent(1, "A", "result_ack", "accepted")
        elif mode == "unknown_result_ack":
            invalid = RecordingEvent(1, "unknown", "result_ack", "accepted")
        elif mode == "unknown_cancel":
            invalid = RecordingEvent(1, "unknown", "cancel")
        else:
            invalid = RecordingEvent(1, "unknown", "preview_ack", 1)
        control.send(invalid)

        fatal = until(lambda event: event.kind == "worker_fatal")
        assert fatal.request_id == ""
        assert fatal.payload.stage == fatal_stage
        assert not [event for event in observed
                    if event.request_id == "B"
                    and event.kind in ("completed", "failed", "cancelled")]

        for path in release_paths:
            path.touch()
        for request_id in expected_terminals:
            if not any(event.request_id == request_id
                       and event.kind in ("completed", "failed", "cancelled")
                       for event in observed):
                until(lambda event, identity=request_id:
                      event.request_id == identity
                      and event.kind in ("completed", "failed", "cancelled"))
        process.join(3)
        assert not process.is_alive(), "protocol command failure must retire boundedly"
        while True:
            try:
                if not control.poll(.05):
                    break
                observed.append(control.recv())
            except (BrokenPipeError, EOFError, OSError):
                break

        fatals = [event for event in observed if event.kind == "worker_fatal"]
        assert len(fatals) == 1
        assert not [event for event in observed if event.kind == "failed"]
        terminals = [event for event in observed
                     if event.kind in ("completed", "failed", "cancelled")]
        for request_id, kind in expected_terminals.items():
            matching = [event for event in terminals if event.request_id == request_id]
            assert len(matching) == 1
            assert matching[0].kind == kind
        fatal_index = observed.index(fatals[0])
        b_terminal = next(event for event in terminals if event.request_id == "B")
        assert observed.index(b_terminal) > fatal_index

        operations = [__import__("json").loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text().splitlines()]
        assert operations.count("stop_task") == 1
        assert operations.count("clear_task") == 1
        assert operations.count("close") == 1
    finally:
        for path in release_paths:
            path.touch()
        _stop_worker(process, control, preview)


def test_parent_watch_never_inspects_pipeline_during_active_to_finalizer_transition(
        tmp_path, monkeypatch):
    from base import recording_worker as module
    from base.recording_worker_pipeline import WorkerCapturePipeline
    from base.ve3668n_capture_timing import VeCaptureProgress

    request = capture_request(tmp_path / "parent-death.wav", request_id="parent-death")
    commands = queue.Queue()
    commands.put(RecordingEvent(1, request.request_id, "start", request))
    transition_entered = threading.Event()
    release_transition = threading.Event()
    death_confirmed = threading.Event()
    watcher_inspected_pipeline = threading.Event()
    forced_exit = threading.Event()

    class TransitionPipeline(WorkerCapturePipeline):
        def capture_released(self, request_id):
            transition_entered.set()
            assert release_transition.wait(2)
            return super().capture_released(request_id)

        def shutdown_snapshot(self):
            if threading.current_thread().name == "recording-parent-watch":
                watcher_inspected_pipeline.set()
                raise RuntimeError("parent watcher crossed the pipeline ownership boundary")
            return super().shutdown_snapshot()

    class DeadParent:
        def is_alive(self):
            assert transition_entered.wait(2)
            death_confirmed.set()
            return False

    class FinishedCapture:
        def __init__(self, request, **_dependencies):
            self.request = request
            self.started = threading.Event()
            self.done = threading.Event()
            self.capture_slot_released = threading.Event()
            self.started_at = time.monotonic()
            self.raw_frames = request.target_samples
            self.capture_slot = SimpleNamespace(
                target_reached_at=self.started_at,
                raw_frames=request.target_samples,
                adapter_released=True,
                writer_released=True,
            )
            self.outcome = RecordingResult(
                request.request_id, request.purpose, request.path, request.sample_rate,
                request.channels, request.target_samples, request.target_samples - 2, True)

        def start(self):
            self.started.set()
            self.capture_slot_released.set()
            self.done.set()

        def progress_snapshot(self):
            return VeCaptureProgress(
                self.started_at, self.request.target_samples, self.started_at)

        def snapshot(self, **_kwargs):
            return None

        def cancel(self):
            self.done.set()

    class Connection:
        def poll(self, _timeout):
            return not commands.empty()

        def recv(self):
            return commands.get_nowait()

        def send(self, _event):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "WorkerCapturePipeline", TransitionPipeline)
    monkeypatch.setattr(module, "RecordingCapture", FinishedCapture)
    monkeypatch.setattr(module.multiprocessing, "parent_process", lambda: DeadParent())
    monkeypatch.setattr(module.os, "_exit", lambda _code: forced_exit.set())
    worker = threading.Thread(
        target=module.recording_worker,
        args=(Connection(), Connection(), 1, None, {}, 1.0, .05), daemon=True)
    worker.start()
    try:
        assert transition_entered.wait(2)
        assert death_confirmed.wait(2)
        # Let the watcher either set broken or violate ownership before the
        # main thread completes its transition.
        threading.Event().wait(.05)
        release_transition.set()
        worker.join(2)
        assert not watcher_inspected_pipeline.is_set()
        assert not worker.is_alive()
        assert not forced_exit.is_set()
    finally:
        release_transition.set()
        if worker.is_alive():
            commands.put(RecordingEvent(1, "", "shutdown"))
            worker.join(2)


def test_worker_lost_slot_snapshot_is_fatal_instead_of_opening_capacity(
        tmp_path, monkeypatch):
    from base import recording_worker as module
    from base.ve3668n_capture_timing import VeCaptureProgress

    request = capture_request(tmp_path / "lost.wav", request_id="lost")
    commands = queue.Queue()
    commands.put(RecordingEvent(1, request.request_id, "start", request))
    sent = []

    class LostSlotCapture:
        def __init__(self, request, **_dependencies):
            self.request = request
            self.started = threading.Event()
            self.done = threading.Event()
            self.capture_slot_released = threading.Event()
            self.capture_slot = None
            self.raw_frames = request.target_samples
            self.started_at = time.monotonic()
            self.outcome = None

        def start(self):
            self.started.set()
            self.capture_slot_released.set()

        def progress_snapshot(self):
            return VeCaptureProgress(
                self.started_at, self.request.target_samples, self.started_at)

        def snapshot(self, **_kwargs):
            return None

        def cancel(self):
            self.done.set()

    class Connection:
        def poll(self, _timeout):
            return not commands.empty()

        def recv(self):
            return commands.get_nowait()

        def send(self, event):
            sent.append(event)

        def close(self):
            pass

    monkeypatch.setattr(module, "RecordingCapture", LostSlotCapture)
    worker = threading.Thread(
        target=module.recording_worker,
        args=(Connection(), Connection(), 1, None, {}, .2, .05), daemon=True)
    worker.start()
    worker.join(2)

    assert not worker.is_alive()
    fatals = [event for event in sent if event.kind == "worker_fatal"]
    assert len(fatals) == 1
    assert fatals[0].request_id == ""
    assert fatals[0].payload.stage == "worker"
    assert "slot state was lost" in fatals[0].payload.message
    assert not [event for event in sent if event.kind == "failed"]
