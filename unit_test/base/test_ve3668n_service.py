"""Preview-independent VE supervision: real spawn and explicit fake clocks."""
from dataclasses import replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from base.recording_process_protocol import (
    RecordingCancelled, RecordingEvent, RecordingFailure, RecordingPreview, RecordingProgress, RecordingResult,
    VeLifecycleCounts, VeReleaseOutcome,
    WorkerFatal,
)
from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
from base.ve3668n_capture_timing import VeCaptureProgress
from base.recording_service import RecordingCallbacks, RecordingService, _Worker
from unit_test.base.test_recording_service import Events, eventually
from unit_test.base.ve3668n_fakes import DiscoveryClock, capture_request


class _ServiceProcess:
    def __init__(self, pid=123, *, alive=True):
        self.pid = pid
        self.alive = alive
        self.terminates = 0
        self.kills = 0
        self.closed = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminates += 1

    def kill(self):
        self.kills += 1

    def join(self, timeout=0):
        return None

    def close(self):
        self.closed += 1


def read_trace(path):
    try:
        # A writer can still be appending the last line.
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.rpartition("\n")[0].splitlines()]


def test_service_starts_b_while_a_result_finalizer_is_paused(tmp_path):
    trace_dir = tmp_path / "persistent-parent"
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:persistent_ve_worker_dependencies",
        backend_options={"trace_dir": str(trace_dir), "pause_finalizers": ("A",)})
    events_a, events_b = Events(), Events()
    try:
        first = service.start(capture_request(tmp_path / "A.wav", request_id="A"), events_a.callbacks)
        eventually(lambda: (trace_dir / "finalizer-A-entered").exists())
        eventually(lambda: service.can_start_recording)
        assert service.busy and not first.released.is_set()

        second = service.start(capture_request(tmp_path / "B.wav", request_id="B"), events_b.callbacks)
        events_b.started.get(timeout=10)
        assert second.worker_pid == first.worker_pid and second.generation == first.generation
        (trace_dir / "release-finalizer-A").touch()
        events_a.results.get(timeout=10)
        events_b.results.get(timeout=10)
        first.accept_result()
        second.accept_result()
        assert first.released.wait(5) and second.released.wait(5)
        operations = [item["operation"] for item in read_trace(trace_dir / "native.jsonl")]
        assert operations.count("create_task") == operations.count("start_task") == 1
        assert not any(item in operations for item in ("stop_task", "clear_task", "close"))
    finally:
        (trace_dir / "release-finalizer-A").touch()
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics


@pytest.fixture
def services(tmp_path, monkeypatch):
    owned = []

    def make(options=None, **kwargs):
        trace = tmp_path / f"native-{len(owned)}.jsonl"
        service = RecordingService(
            backend_factory="unit_test.base.ve3668n_fakes:capture_dependencies",
            backend_options=dict(trace_path=str(trace), **(options or {})), **kwargs)
        owned.append(service)
        received = []
        dispatch = service._event

        def inspect(worker, event):
            received.append((time.monotonic(), event))
            dispatch(worker, event)

        monkeypatch.setattr(service, "_event", inspect)
        return service, received, trace

    yield make
    for service in owned:
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics
        eventually(lambda: not any(thread.is_alive() for thread in service.threads))
        assert service.worker_pid is None


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """Exercise the real supervisor methods with no OS I/O or five-second wait.

    Both sent/native timestamps explicitly share this clock's origin (100).
    Only supervisor launch and the process/pipe boundary are replaced.
    """
    from base import recording_service as module

    clock = DiscoveryClock()
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService()
    process = SimpleNamespace(pid=123, is_alive=lambda: True,
                              terminate=lambda: None, kill=lambda: None)
    worker = _Worker(1, process, None, None, None)
    worker.ready = True
    service._worker = worker
    events = Events()
    session = service.start(capture_request(tmp_path / "probe.wav", target_samples=51200), events.callbacks)
    service._dispatch(service._inbox.get_nowait())
    assert worker.outgoing.get_nowait().kind == "start"

    def event(kind, payload=None):
        service._event(worker, RecordingEvent(1, session.request.request_id, kind, payload))

    def progress(frames, at):
        event("progress", RecordingProgress(session.request.request_id, 1, frames, at))

    return SimpleNamespace(service=service, worker=worker, session=session, clock=clock,
                           events=events, event=event, progress=progress)


def test_watchdog_native_start_not_receipt_time_and_actual_cancel(probe):
    p = probe
    p.clock.advance(.4)
    p.event("started", 100.1)
    assert p.session.state == "recording"
    assert p.session._deadline is None
    p.clock.advance(4.6)
    p.service._tick()
    assert p.events.failed.empty()
    p.clock.advance(.11)
    p.service._tick()
    failure = p.events.failed.get_nowait()
    assert failure.stage == "capture_timeout" and "no progress" in failure.message
    assert not failure.handles_released
    assert p.worker.outgoing.get_nowait().kind == "cancel", "_fail alone never sends cancel"
    assert not p.session.released.is_set()
    # A late cancellation result releases ownership but cannot replace failure.
    p.event("cancelled", RecordingCancelled(p.session.request.request_id, p.session.request.path, 0, 0))
    assert p.session.state == "failed" and p.session.failure is failure
    assert p.session.released.is_set() and p.events.cancelled.empty()
    assert p.events.failed.empty()


@pytest.mark.parametrize("frames", [0, 1])
def test_same_count_never_extends_watchdog_even_with_new_timestamp(probe, frames):
    p = probe
    p.event("started", 100.0)
    p.progress(frames, 100.0)
    p.clock.advance(4.99)
    p.progress(frames, p.clock())
    p.service._tick()
    assert p.events.failed.empty()
    p.clock.advance(.02)
    p.service._tick()
    assert "no progress" in p.events.failed.get_nowait().message


def test_trickle_progress_cannot_extend_overall_capture_deadline(probe):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(4)
    p.progress(1, p.clock())
    p.clock.advance(1.9)
    p.progress(2, p.clock())
    p.service._tick()
    assert p.events.failed.empty()
    p.clock.advance(.1)
    p.service._tick()
    assert "total deadline" in p.events.failed.get_nowait().message


@pytest.mark.parametrize("target", [False, True])
def test_finalizing_keeps_watchdog_until_trusted_target(probe, target):
    p = probe
    p.event("started", 100.0)
    p.event("finalizing")
    assert p.session.state == "finalizing"
    if target:
        p.progress(p.session.request.target_samples, 100.0)
    p.clock.advance(100)
    p.service._tick()
    if target:
        assert p.events.failed.get_nowait().stage == "capture_release_timeout"
        assert p.worker.retiring
    else:
        assert p.events.failed.get_nowait().stage == "capture_timeout"
        # The existing five-second cancellation budget precedes retirement.
        assert not p.worker.retiring
        p.clock.advance(5)
        p.service._tick()
        assert p.worker.retiring


def test_progress_before_native_start_cannot_mark_started_or_remove_start_timeout(probe):
    p = probe
    p.progress(51200, 100.0)
    p.event("finalizing")
    assert p.session.state == "starting" and p.session._deadline == 110.0
    p.clock.advance(10)
    p.service._tick()
    assert p.events.failed.get_nowait().stage == "start_timeout"


def forged(event, **changes):
    """Actual pickle roundtrip, which does NOT run dataclass __post_init__."""
    for name, value in changes.items():
        object.__setattr__(event, name, value)
    return pickle.loads(pickle.dumps(event))


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 1.0), ("generation", True), ("generation", 1.0),
    ("generation", 2), ("request_id", "previous"), ("request_id", True),
    ("payload", None), ("payload", True), ("payload", -1), ("payload", float("nan")),
    ("payload", float("inf")), ("payload", "100"), ("payload", 99.9), ("payload", 102.1),
])
def test_unpickled_started_requires_strict_current_request_native_time(probe, field, value):
    p = probe
    p.clock.advance(2)
    event = RecordingEvent(1, p.session.request.request_id, "started", 100.0)
    p.service._event(p.worker, forged(event, **{field: value}))
    assert p.session.state == "starting"
    assert p.session._deadline == 110.0
    assert p.events.started.empty()


@pytest.mark.parametrize("field", ["generation", "request_id", "kind"])
def test_unpickled_missing_envelope_fields_are_rejected_without_marking_start(probe, field):
    p = probe
    event = RecordingEvent(1, p.session.request.request_id, "started", 100.0)
    object.__delattr__(event, field)
    p.service._event(p.worker, pickle.loads(pickle.dumps(event)))
    assert p.session.state == "starting" and p.session._deadline == 110.0


@pytest.mark.parametrize("field,value", [
    ("frames", True), ("frames", 20.0), ("frames", -1), ("frames", 9), ("frames", 51201),
    ("last_frame_at", True), ("last_frame_at", None), ("last_frame_at", "102"),
    ("last_frame_at", -1), ("last_frame_at", float("nan")), ("last_frame_at", float("inf")),
    ("last_frame_at", 99.9), ("last_frame_at", 100.9), ("last_frame_at", 103),
    ("generation", True), ("generation", 1.0), ("generation", 2),
    ("request_id", True), ("request_id", "previous"),
])
def test_unpickled_progress_admission_rechecks_every_fact(probe, field, value):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(1)
    p.progress(10, 101.0)
    p.clock.advance(1)
    payload = forged(RecordingProgress(p.session.request.request_id, 1, 20, 102.0), **{field: value})
    event = forged(RecordingEvent(1, p.session.request.request_id, "progress",
                                 RecordingProgress(p.session.request.request_id, 1, 20, 102.0)), payload=payload)
    p.service._event(p.worker, event)
    assert p.session._capture_deadline.snapshot().frames == 10
    assert p.session._capture_deadline.snapshot().last_frame_at == 101.0
    p.clock.advance(4)
    p.service._tick()
    assert "no progress" in p.events.failed.get_nowait().message


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 1.0), ("generation", True), ("generation", 1.0),
    ("generation", 2), ("request_id", "previous"), ("payload", None), ("payload", {}),
])
def test_progress_envelope_and_previous_worker_are_not_heartbeat(probe, field, value):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(4)
    event = RecordingEvent(1, p.session.request.request_id, "progress",
                           RecordingProgress(p.session.request.request_id, 1, 100, 104.0))
    p.service._event(SimpleNamespace(generation=1), event)
    p.service._event(p.worker, forged(event, **{field: value}))
    assert p.session._capture_deadline.snapshot().frames == 0
    p.clock.advance(1)
    p.service._tick()
    assert "no progress" in p.events.failed.get_nowait().message


def test_valid_zero_progress_start_timestamp_and_duplicate_start_do_not_extend(probe):
    p = probe
    p.event("started", 100.0)
    p.progress(0, 100.0)
    p.clock.advance(4)
    p.event("started", 104.0)
    p.progress(0, 104.0)
    assert p.session._capture_deadline.snapshot().started_at == 100.0
    assert p.session._capture_deadline.snapshot().last_frame_at == 100.0
    assert p.events.started.qsize() == 1
    p.clock.advance(1)
    p.service._tick()
    assert p.events.failed.get_nowait().stage == "capture_timeout"


def test_preview_frame_count_is_never_capture_heartbeat(probe):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(4.9)
    waveform = StreamingWaveformSnapshot(np.array([0.0]), np.array([1], dtype=np.float32), 51200)
    p.event("preview", RecordingPreview(p.session.request.request_id, 1, 1, 51200,
                                        p.session.request.channels, (waveform, waveform)))
    assert p.events.preview.qsize() == 1 and p.session._last_sample_stop == 51200
    assert not p.session._capture_deadline.complete
    p.clock.advance(.2)
    p.service._tick()
    assert p.events.failed.get_nowait().stage == "capture_timeout"


def test_watchdog_preserves_first_failure_through_close_failure_and_kill_deadline(probe):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(5)
    p.service._tick()
    first = p.events.failed.get_nowait()
    assert p.worker.outgoing.get_nowait().kind == "cancel"
    p.event("failed", RecordingFailure(p.session.request.request_id, "close_stream", p.session.request.path,
                                       "native owner did not exit", handles_released=False))
    assert p.worker.retiring and not p.session.released.is_set()
    assert p.session.failure is first
    assert p.events.failed.empty() and p.events.cancelled.empty()
    p.clock.advance(2)
    p.service._tick()
    assert p.worker.kill_reported
    assert p.service.busy and p.service.is_path_leased(p.session.request.path)


def test_user_cancel_uses_existing_cancel_deadline_not_capture_watchdog(probe):
    p = probe
    p.event("started", 100.0)
    p.clock.advance(4)
    p.service._request_cancel(p.session)
    assert p.session._deadline == 109.0
    p.clock.advance(1.1)
    p.service._tick()
    assert p.events.failed.empty()
    p.clock.advance(3.9)
    p.service._tick()
    assert p.events.failed.get_nowait().stage == "cancel_timeout"


@pytest.mark.parametrize("change", [
    {"raw_frames": 51199}, {"final_frames": 51200}, {"sample_rate": 48000}, {"channels": (1, 7)},
])
def test_trusted_progress_does_not_weaken_completed_descriptor_guard(probe, change):
    p = probe
    p.event("started", 100.0)
    p.progress(51200, 100.0)
    values = dict(request_id=p.session.request.request_id, purpose="main", path=p.session.request.path,
                  sample_rate=51200, channels=(7, 1), raw_frames=51200, final_frames=51198,
                  metadata_appended=True)
    values.update(change)
    p.event("completed", RecordingResult(**values))
    assert p.events.failed.get_nowait().stage == "protocol"
    assert p.events.results.empty() and p.session.reader is None


@pytest.mark.parametrize("already_done", [False, True])
def test_worker_final_progress_survives_done_snapshot_race_and_immediate_ack(tmp_path, monkeypatch, already_done):
    """Deterministic control scheduling; real capture/spawn is covered separately."""
    from base import recording_worker as module

    req = capture_request(tmp_path / "fast.wav")
    sent = []
    commands = queue.Queue()
    commands.put(RecordingEvent(1, req.request_id, "start", req))

    class Capture:
        def __init__(self, request, **_dependencies):
            self.request = request
            self.started = threading.Event()
            self.done = threading.Event()
            self.capture_slot_released = threading.Event()
            self.capture_slot = None
            self.raw_frames = 0
            self.started_at = time.monotonic()
            self.outcome = RecordingResult(req.request_id, "main", req.path, req.sample_rate,
                                           req.channels, req.target_samples, req.target_samples - 2, True)

        def start(self):
            self.started.set()
            if already_done:
                self.raw_frames = req.target_samples
                self.done.set()

        def progress_snapshot(self):
            if not self.done.is_set():
                # Completion happens just AFTER sampling an older progress value.
                self.raw_frames = req.target_samples
                self.done.set()
                return VeCaptureProgress(self.started_at, 1, self.started_at)
            return VeCaptureProgress(self.started_at, req.target_samples, self.started_at)

        def snapshot(self, **kwargs):
            return None

        def cancel(self):
            self.done.set()

    class Control:
        def poll(self, timeout):
            return not commands.empty()

        def recv(self):
            return commands.get_nowait()

        def send(self, event):
            sent.append(event)
            if event.kind == "completed":
                commands.put(RecordingEvent(1, req.request_id, "result_ack", "accepted"))
                commands.put(RecordingEvent(1, "", "shutdown"))

        def close(self):
            pass

    monkeypatch.setattr(module, "RecordingCapture", Capture)
    thread = threading.Thread(target=module.recording_worker,
                              args=(Control(), Control(), 1, None, {}), daemon=True)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    kinds = [event.kind for event in sent]
    final = [index for index, event in enumerate(sent)
             if event.kind == "progress" and event.payload.frames == req.target_samples]
    assert len(final) == 1, "final progress must survive terminal ACK clearing capture"
    assert kinds.index("started") < final[0] < kinds.index("completed")


@pytest.mark.parametrize("streaming", [False, True])
def test_spawn_broken_preview_reducer_cannot_hide_progress_or_lose_audio(tmp_path, services, streaming):
    service, received, _ = services(dict(broken_preview=True))
    events = Events()
    session = service.start(capture_request(tmp_path / "preview.wav", target_samples=8192,
                                            streaming=streaming), events.callbacks)
    audio = events.results.get(timeout=10)
    assert audio.multi.shape == (8190, 2)
    assert session._capture_deadline.complete
    assert any(event.kind == "progress" for _, event in received)
    assert any("preview disabled" in warning for warning in session.descriptor.warnings) is streaming
    assert events.failed.empty()
    session.accept_result()
    assert session.released.wait(5)


def test_spawn_progress_rate_is_bounded_and_final_is_forced(tmp_path, services):
    service, received, trace_path = services(dict(read_delay=.06))
    events = Events()
    session = service.start(capture_request(tmp_path / "coalesced.wav", target_samples=40960), events.callbacks)
    events.results.get(timeout=10)
    progress = [(at, event.payload) for at, event in received if event.kind == "progress"]
    assert progress[-1][1].frames == 40960
    ordinary = progress[:-1]
    native_start = next(event.payload for _, event in received if event.kind == "started")
    elapsed = progress[-1][1].last_frame_at - native_start
    # Windows waits and SDK-fake buffer preparation can take longer than the
    # requested read delay. Bound rate against actual elapsed native time.
    assert 2 <= len(ordinary) <= math.floor(elapsed / .2) + 1
    # Local delivery may jitter; no per-block (~17 Hz) or preview (~20 Hz) sends.
    assert all(later[0] - earlier[0] >= .15 for earlier, later in zip(ordinary, ordinary[1:]))
    assert len([item for item in read_trace(trace_path)
                if item["operation"] == "read_task_data"]) >= 20
    assert session._capture_deadline.complete
    assert session.descriptor.raw_frames == 40960 and session.descriptor.final_frames == 40958
    session.accept_result()
    assert session.released.wait(5)


def test_spawn_ordinary_fast_capture_forces_target_and_reuses_healthy_worker(tmp_path, services):
    service, received, _ = services(dict(read_delay=0))
    identity = None
    for index in range(2):
        events = Events()
        req = capture_request(tmp_path / f"fast-{index}.wav", request_id=f"fast-{index}")
        session = service.start(req, events.callbacks)
        events.results.get(timeout=10)
        assert session._capture_deadline.complete
        progress = [event for _, event in received if event.kind == "progress" and event.request_id == req.request_id]
        assert len(progress) == 1 and progress[0].payload.frames == req.target_samples
        if identity is not None:
            assert (session.worker_pid, session.generation) == identity
        identity = session.worker_pid, session.generation
        session.accept_result()
        assert session.released.wait(5)
        assert session.state == "completed" and events.failed.empty()


@pytest.mark.parametrize("phase", ["starting", "recording", "finalizing", "delivering"])
@pytest.mark.parametrize("streaming", [False, True])
def test_spawn_cooperative_cancel_releases_each_phase_without_success(tmp_path, services, phase, streaming):
    options = dict(zero_reads=True) if phase == "recording" else (
        dict(block_operation="stop_task") if phase == "finalizing" else {})
    service, _, trace_path = services(options)
    events = Events()
    session = service.start(capture_request(tmp_path / "cooperative.wav", streaming=streaming), events.callbacks)
    if phase == "recording":
        events.started.get(timeout=10)
    elif phase == "finalizing":
        eventually(lambda: session.state == "finalizing")
    elif phase == "delivering":
        events.results.get(timeout=10)
    session.cancel()
    if phase == "finalizing":
        Path(str(trace_path) + ".release").touch()
    events.cancelled.get(timeout=5)
    assert session.released.wait(5)
    assert session.state == "cancelled" and not service.is_path_leased(session.request.path)
    assert events.failed.empty() and events.accepted.empty() and events.cancelled.empty()


@pytest.mark.parametrize("streaming", [False, True])
def test_spawn_full_control_pipe_does_not_block_native_audio_or_drop_final_count(
        tmp_path, services, monkeypatch, streaming):
    from base import recording_service as module
    from unit_test.base.ve3668n_fakes import blocked_control_worker
    monkeypatch.setattr(module, "recording_worker", blocked_control_worker)
    service, received, trace_path = services(dict(read_delay=.025))
    events = Events()
    session = service.start(capture_request(tmp_path / "full-pipe.wav", target_samples=40960,
                                            streaming=streaming), events.callbacks)
    try:
        def final_wav_on_disk():
            try:
                return sf.info(session.request.path).frames == session.request.target_samples - 2
            except (OSError, sf.LibsndfileError):
                # Native close precedes writer finalization/atomic trim replace.
                return False

        eventually(final_wav_on_disk)
        assert events.started.empty(), "control sender is still blocked"
    finally:
        Path(str(trace_path) + ".ipc-release").touch()
    audio = events.results.get(timeout=5)
    assert session._capture_deadline.complete
    progress = [event for _, event in received if event.kind == "progress"]
    assert progress[-1].payload.frames == 40960
    assert len(progress) <= 2  # Coalescing keeps one pending update, plus forced final.
    np.testing.assert_array_equal(audio.multi, np.tile([8.25, 2.5], (40958, 1)))
    session.accept_result()
    assert session.released.wait(5)
    assert events.failed.empty()


@pytest.mark.parametrize("operation", ["read_task_data"])
def test_spawn_hung_native_calls_retire_before_healthy_new_request(
        tmp_path, services, monkeypatch, operation):
    from base import recording_service as module

    service, received, trace_path = services(dict(block_operation=operation, first_only=True),
                                            terminate_timeout=.2)
    events = Events()
    session = service.start(capture_request(tmp_path / "hung.wav"), events.callbacks)
    events.started.get(timeout=10)
    eventually(lambda: any(item["operation"] == operation for item in read_trace(trace_path)))
    old_worker = service._worker
    if operation == "read_task_data":
        # The real owner is now stuck and cannot generate another timestamp.
        # Advance only the parent's elapsed-time probe, on the SAME monotonic
        # origin as the retained native start; no real five-second test sleep.
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + 5.1))
    failure = events.failed.get(timeout=5)
    assert failure.stage == ("capture_timeout" if operation == "read_task_data" else "close_stream")
    assert not session.released.is_set() or service.worker_pid is None
    assert session.released.wait(5)
    assert service.worker_pid is None and not service.is_path_leased(session.request.path)
    assert not any(thread.is_alive() for thread in old_worker.threads)
    assert events.results.empty() and events.accepted.empty() and events.cancelled.empty()
    assert events.failed.empty()
    trace = read_trace(trace_path)
    native = [item for item in trace if item["operation"] != "owner_join"]
    assert {item["pid"] for item in trace} == {session.worker_pid}
    assert len({item["thread_id"] for item in native}) == 1
    assert all(item["thread_name"].startswith("VE_") for item in native)
    assert not any(item["operation"] in ("stop_task", "clear_task", "close") for item in trace)
    monkeypatch.setattr(module, "time", time)
    fresh_events = Events()
    fresh = service.start(capture_request(tmp_path / "healthy.wav", request_id="healthy"), fresh_events.callbacks)
    fresh_events.results.get(timeout=10)
    assert fresh.worker_pid != session.worker_pid and fresh.generation > session.generation
    # Late previous-generation events cannot overwrite the new request.
    previous = [event for _, event in received if event.request_id == session.request.request_id]
    for event in previous:
        service._inbox.put(("event", old_worker, event))
    fresh.accept_result()
    assert fresh.released.wait(5) and fresh.state == "completed"
    assert fresh_events.failed.empty()


@pytest.mark.parametrize("operation", ["get_devices", "start_task", "read_task_data"])
def test_spawn_cancel_and_window_close_all_native_phases_are_asynchronous(
        tmp_path, services, operation):
    service, _, trace_path = services(dict(block_operation=operation), cancel_timeout=.15,
                                      shutdown_timeout=.15, terminate_timeout=.15)
    events = Events()
    session = service.start(capture_request(tmp_path / "cancel.wav"), events.callbacks)
    eventually(lambda: any(item["operation"] == operation for item in read_trace(trace_path)))
    before = time.monotonic()
    session.cancel()
    service.shutdown()
    assert time.monotonic() - before < .1
    assert service.closed.wait(5)
    assert session.released.is_set()
    assert service.worker_pid is None
    assert events.results.empty() and events.accepted.empty()
    assert events.failed.qsize() + events.cancelled.qsize() == 1
    assert all(item["pid"] == session.worker_pid != os.getpid() for item in read_trace(trace_path))


def test_spawn_native_process_crash_is_one_failure_then_releases(tmp_path, services):
    service, _, _ = services(dict(crash_operation="read_task_data"))
    events = Events()
    session = service.start(capture_request(tmp_path / "crash.wav"), events.callbacks)
    assert events.failed.get(timeout=10).stage == "worker"
    assert session.released.wait(5)
    assert service.worker_pid is None and events.failed.empty() and events.results.empty()


@pytest.mark.skipif(os.name != "nt", reason="Windows process-handle liveness acceptance")
@pytest.mark.parametrize("operation", ["read_task_data", "stop_task", "clear_task"])
def test_spawn_parent_death_reaps_blocked_native_owner(tmp_path, operation):
    from unit_test.base.recording_process_fakes import open_process_observer
    from unit_test.base.ve3668n_fakes import capture_orphan_parent

    context = mp.get_context("spawn")
    incoming, outgoing = context.Pipe(duplex=False)
    parent = context.Process(target=capture_orphan_parent, args=(outgoing, dict(
        trace_path=str(tmp_path / "orphan.jsonl"), block_operation=operation)))
    kernel = handle = None
    try:
        parent.start()
        outgoing.close()
        assert incoming.poll(10)
        worker_pid = incoming.recv()
        kernel, handle = open_process_observer(worker_pid)
        assert kernel.WaitForSingleObject(handle, 0) == 258
        parent.terminate()
        parent.join(5)
        assert not parent.is_alive()
        eventually(lambda: kernel.WaitForSingleObject(handle, 0) == 0, timeout=5)
    finally:
        if parent.is_alive():
            parent.terminate()
        parent.join(5)
        parent.close()
        incoming.close()
        outgoing.close()
        if handle:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                assert kernel.WaitForSingleObject(handle, 5000) == 0
            kernel.CloseHandle(handle)
