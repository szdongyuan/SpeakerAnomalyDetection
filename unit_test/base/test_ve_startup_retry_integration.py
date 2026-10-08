"""VK startup policy integration using fake owners and injected clocks."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import json
import os
import queue
import time

import numpy as np
import soundfile as sf

import pytest

from base.recording_process_protocol import (
    CaptureSlotReleased, RecordingEvent, RecordingProgress, VeLifecycleCounts,
    VeResourceRetryProof, VeReleaseOutcome, VE_RESOURCE_RETRY,
    VE_STARTUP_RECOVERY_TERMINAL_STAGES,
)
from base.recording_service import RecordingService, _Worker, _PendingVeRelease
from base.ve_startup_policy import VeStartupBudget
from unit_test.base.ve3668n_fakes import DiscoveryClock, capture_request
from unit_test.base.test_ve3668n_service import (
    _ServiceProcess, _service_prewarm_probe, _prewarm_started, _prewarm_detaching,
    _prewarm_result,
)
from unit_test.base.test_recording_service import Events, eventually
from unit_test.base.test_ve3668n_service import read_trace


@pytest.fixture
def spawned_retry_service(monkeypatch, tmp_path):
    from base import recording_service as module
    from unit_test.base.vk_startup_retry_fakes import gated_startup_worker

    # Only the parent constructs the authoritative budget; the real child gets
    # the same immutable value through its normal command envelope.
    monkeypatch.setattr(module, "VeStartupBudget", SimpleNamespace(create=lambda start, total:
        VeStartupBudget.create(start, total, first_attempt_timeout=.5, cleanup_timeout=.8)))
    monkeypatch.setattr(module, "recording_worker", gated_startup_worker)
    owned = []

    def make(**options):
        trace = tmp_path / f"worker-{len(owned)}"
        trace.mkdir()
        service = RecordingService(
            backend_factory="unit_test.base.vk_startup_retry_fakes:startup_retry_dependencies",
            backend_options=dict(trace_dir=str(trace), **options), start_timeout=4,
            cancel_timeout=.5, shutdown_timeout=1, terminate_timeout=.3, retry_delay=.05)
        received, phases, exits = [], [], []
        dispatch, tick, dead = service._event, service._tick, service._dead

        def inspect(worker, event):
            received.append((time.monotonic(), event))
            dispatch(worker, event)

        def inspect_tick():
            pending = service._pending_ve_prewarm
            if pending is not None:
                phases.append(pending.phase)
            tick()

        def inspect_dead(worker):
            exits.append(worker.process.exitcode)
            dead(worker)

        monkeypatch.setattr(service, "_event", inspect)
        monkeypatch.setattr(service, "_tick", inspect_tick)
        monkeypatch.setattr(service, "_dead", inspect_dead)
        probe = SimpleNamespace(service=service, trace=trace, received=received,
                                phases=phases, exits=exits)
        owned.append(probe)
        return probe

    yield make
    for probe in owned:
        for name in ("release-native-1", "release-native-2", "release-cleanup",
                     "release-retry-send", "release-finalizer-A"):
            (probe.trace / name).touch()
        probe.service.shutdown()
        assert probe.service.closed.wait(12), probe.service.diagnostics
        eventually(lambda: not any(t.is_alive() for t in probe.service.threads))
        assert probe.service.worker_pid is None


def _admit_spawned(probe, tmp_path, prewarm):
    from base.recording_process_protocol import VePrewarmRequest

    req = capture_request(tmp_path / "recovered.wav", request_id="recovered",
                          target_samples=8192, trim_samples=0)
    if prewarm:
        req = VePrewarmRequest.create("recovered", req.device, req.channels,
                                      req.sample_rate, attempt=1)
        results = queue.Queue()
        assert probe.service.prewarm_ve(req, results.put) == "accepted"
        state = probe.service._pending_ve_prewarm
        return req, state, results
    events = Events()
    session = probe.service.start(req, events.callbacks)
    return req, session, events


def _release_first_after_deadline(probe, state, prewarm):
    eventually(lambda: (probe.trace / "native-1-entered").exists())
    budget = state.startup_budget if prewarm else state._startup_budget
    pid, generation = probe.service.worker_pid, probe.service._worker.generation
    eventually(lambda: time.monotonic() >= budget.first_attempt_deadline)
    (probe.trace / "release-native-1").touch()
    return budget, pid, generation


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("backpressure", [False, True])
def test_spawn_retry_preserves_owner_writer_and_reliable_order(
        spawned_retry_service, tmp_path, prewarm, backpressure):
    probe = spawned_retry_service(gate_retry_send=backpressure)
    req, state, events = _admit_spawned(probe, tmp_path, prewarm)
    budget, pid, generation = _release_first_after_deadline(probe, state, prewarm)
    if backpressure:
        eventually(lambda: (probe.trace / "backlog-ready").exists())
        backlog = json.loads((probe.trace / "backlog.json").read_text())
        kinds = backlog["ordered" if prewarm else "outgoing"]
        expected = ([VE_RESOURCE_RETRY, "ve_prewarm_started", "ve_prewarm_progress", "ve_prewarm_detaching",
                     "ve_prewarm_terminal"] if prewarm else
                    [VE_RESOURCE_RETRY, "started", "progress", "capture_slot_released", "completed"])
        assert all(kind in kinds for kind in expected), backlog
        assert [kinds.index(kind) for kind in expected] == sorted(kinds.index(kind) for kind in expected)
        assert not any(event.kind == VE_RESOURCE_RETRY for _, event in probe.received)
        (probe.trace / "release-retry-send").touch()
    if prewarm:
        result = events.get(timeout=10)
        assert result.success and result.ownership_safe
        assert len(result.attempts) == 1 and result.attempts[0].generation == generation
        assert result.attempts[0].frames_per_channel == req.frames_per_channel
        assert result.lifecycle_counts == VeLifecycleCounts(2, 2, 1, 1, 1, 1)
        assert not list(tmp_path.rglob("*.wav"))
    else:
        audio = events.results.get(timeout=10)
        saved, rate = sf.read(req.path, dtype="float32", always_2d=True)
        expected_audio = np.tile(np.array([.125, -.25], dtype=np.float32), (req.target_samples, 1))
        np.testing.assert_allclose(saved, expected_audio, rtol=0, atol=2 ** -23)
        np.testing.assert_allclose(audio.multi, expected_audio, rtol=0, atol=2 ** -23)
        assert rate == req.sample_rate and state.descriptor.raw_frames == req.target_samples
        assert state.worker_pid == pid and state.generation == generation
        assert events.failed.empty() and events.started.qsize() == 1
        state.accept_result()
        assert state.released.wait(5)
    assert pid != os.getpid() and probe.service.worker_pid == pid
    assert probe.service._worker.generation == generation
    trace = read_trace(probe.trace / "native.jsonl")
    creates = [row for row in trace if row["operation"] == "sdk_create"]
    assert len(creates) == 2 and creates[1]["previous_owners_alive"] == [False]
    assert {row["pid"] for row in trace} == {pid}
    tasks = [row["args"][0] for row in trace if row["operation"] == "create_task"]
    assert len(tasks) == len(set(tasks)) == 2
    assert not any(row["operation"] == "read_task_data" and row["instance"] == 1 for row in trace)
    writers = read_trace(probe.trace / "writer.jsonl")
    assert len([row for row in writers if row["operation"] == "open"]) == (0 if prewarm else 1)
    delivered = [event for _, event in probe.received if event.request_id == req.request_id] if not prewarm else [
        event for _, event in probe.received if event.request_id == req.warmup_id]
    kinds = [event.kind for event in delivered]
    assert kinds.count(VE_RESOURCE_RETRY) == 1
    proof = next(event.payload for event in delivered if event.kind == VE_RESOURCE_RETRY)
    assert proof.startup_budget == budget and proof.old_task_id != proof.new_task_id
    assert kinds.index(VE_RESOURCE_RETRY) < kinds.index("ve_prewarm_started" if prewarm else "started")
    started = next(event.payload for event in delivered
                   if event.kind == ("ve_prewarm_started" if prewarm else "started"))
    started_at = started.started_at if prewarm else started
    native_start = next(row["at"] for row in trace if row["operation"] == "start_task")
    verified = next(row["at"] for row in trace if row["operation"] == "verify_actual_sample_rate")
    assert native_start <= started_at <= verified
    for instance in (1, 2):
        assert len({row["thread_id"] for row in trace if row["instance"] == instance}) == 1
    assert "retry_wait" not in probe.phases
    probe.service.shutdown()
    assert probe.service.closed.wait(12)
    assert probe.exits == [0]


@pytest.mark.parametrize("release_before_terminal", [False, True])
def test_spawn_cancelled_cold_start_preserves_older_successful_finalizer(
        spawned_retry_service, tmp_path, release_before_terminal):
    probe = spawned_retry_service(block_instance=2, pause_finalizers=("A",))
    first_events, second_events = Events(), Events()
    first_request = capture_request(tmp_path / "A.wav", request_id="A", trim_samples=0)
    first = probe.service.start(first_request, first_events.callbacks)
    eventually(lambda: (probe.trace / "finalizer-A-entered").exists())
    eventually(lambda: probe.service.can_start_recording)
    pid, generation = first.worker_pid, first.generation
    # A rate change requires a cold SDK while A's completed audio is finalizing.
    second = probe.service.start(capture_request(tmp_path / "B.wav", request_id="B",
                                                 sample_rate=44100), second_events.callbacks)
    eventually(lambda: (probe.trace / "native-2-entered").exists())
    second.cancel()
    eventually(lambda: second.cancel_requested)
    if release_before_terminal:
        (probe.trace / "release-native-2").touch()
    eventually(lambda: not second_events.failed.empty() or not second_events.cancelled.empty())
    assert second_events.failed.empty(), read_trace(probe.trace / "events.jsonl")
    second_events.cancelled.get(timeout=5)
    eventually(lambda: any(event.kind == "cancelled" and event.request_id == "B"
                           for _, event in probe.received))
    assert not probe.service.can_start_recording
    assert probe.service.worker_pid == pid and not probe.service._worker.retiring
    assert first_events.failed.empty() and first_events.results.empty()
    assert first.failure is None and not first.released.is_set()
    if not release_before_terminal:
        assert second.descriptor.handles_released is False
        assert not second._child_released and not second.released.is_set()
        assert probe.service.is_path_leased(second.request.path)
    (probe.trace / "release-native-2").touch()
    eventually(lambda: len([row for row in read_trace(probe.trace / "native.jsonl")
                            if row["operation"] == "cleanup_complete"]) == 2)
    (probe.trace / "release-finalizer-A").touch()
    audio = first_events.results.get(timeout=5)
    first.accept_result()
    assert first.released.wait(5) and second.released.wait(5)
    eventually(lambda: probe.service.worker_pid is None)
    saved, _ = sf.read(first_request.path, dtype="float32", always_2d=True)
    expected = np.tile(np.array([.125, -.25], dtype=np.float32), (first_request.target_samples, 1))
    np.testing.assert_allclose(saved, expected, rtol=0, atol=2 ** -23)
    np.testing.assert_allclose(audio.multi, expected, rtol=0, atol=2 ** -23)
    assert first.state == "completed" and first.failure is None
    assert second.state == "cancelled" and second_events.started.empty()
    assert probe.service._generation == generation
    assert not any(event.kind == VE_RESOURCE_RETRY for _, event in probe.received)
    trace = read_trace(probe.trace / "native.jsonl")
    assert len([row for row in trace if row["operation"] == "sdk_create"]) == 2


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("cleanup", ["failure", "timeout"])
def test_spawn_retry_cleanup_failure_never_restarts_request(
        spawned_retry_service, tmp_path, prewarm, cleanup):
    probe = spawned_retry_service(cleanup_mode=cleanup)
    req, state, events = _admit_spawned(probe, tmp_path, prewarm)
    budget, pid, generation = _release_first_after_deadline(probe, state, prewarm)
    eventually(lambda: (probe.trace / "cleanup-entered").exists())
    failure = events.get(timeout=5) if prewarm else events.failed.get(timeout=5)
    notified_at = time.monotonic()
    assert (not failure.success) if prewarm else state.state == "failed"
    assert notified_at < budget.cleanup_deadline + .5
    if prewarm:
        assert failure.stage == ("startup_cleanup_failed" if cleanup == "failure" else "startup_cleanup_timeout")
        assert failure.code == (-17 if cleanup == "failure" else None)
        assert len(failure.attempts) == 1
    else:
        assert events.results.empty() and events.started.empty()
    # Let an outstanding SDK operation return after the terminal notification.
    # Retirement must remain final rather than creating SDK/worker number two.
    (probe.trace / "release-cleanup").touch()
    eventually(lambda: probe.service.worker_pid is None)
    if not prewarm:
        assert state.released.wait(5)
        assert state.state == "failed" and events.failed.empty()
    else:
        assert events.empty() and probe.service._pending_ve_prewarm is None
    assert probe.service._generation == generation
    trace = read_trace(probe.trace / "native.jsonl")
    assert {row["pid"] for row in trace} == {pid}
    assert len([row for row in trace if row["operation"] == "sdk_create"]) == 1
    assert not any(event.kind in (VE_RESOURCE_RETRY, "started", "ve_prewarm_started")
                   for _, event in probe.received)
    assert "retry_wait" not in probe.phases


def test_spawn_cancelled_cold_start_still_reports_real_cleanup_failure(
        spawned_retry_service, tmp_path):
    probe = spawned_retry_service(block_instance=2, pause_finalizers=("A",),
                                  cleanup_mode="failure", cleanup_instance=2)
    first_events, second_events = Events(), Events()
    first = probe.service.start(capture_request(tmp_path / "A.wav", request_id="A"),
                                first_events.callbacks)
    eventually(lambda: (probe.trace / "finalizer-A-entered").exists())
    eventually(lambda: probe.service.can_start_recording)
    second = probe.service.start(capture_request(tmp_path / "B.wav", request_id="B",
                                                 sample_rate=44100), second_events.callbacks)
    eventually(lambda: (probe.trace / "native-2-entered").exists())
    second.cancel()
    second_events.cancelled.get(timeout=5)
    assert first.failure is None and not probe.service._worker.retiring
    (probe.trace / "release-native-2").touch()
    # Cancellation does not hide a subsequent real error from revoked cleanup.
    failure = first_events.failed.get(timeout=5)
    assert failure.stage == "worker_retired_during_finalization"
    fatal = next(event.payload for _, event in probe.received if event.kind == "worker_fatal")
    assert fatal.stage == "startup_cleanup_failed" and "-17" in fatal.message
    (probe.trace / "release-cleanup").touch()
    (probe.trace / "release-finalizer-A").touch()
    eventually(lambda: probe.service.worker_pid is None)
    assert second.state == "cancelled" and second_events.failed.empty()
    trace = read_trace(probe.trace / "native.jsonl")
    assert len([row for row in trace if row["operation"] == "sdk_create"]) == 2


def recording_probe(monkeypatch, tmp_path, **options):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **kw: None)
    clock = DiscoveryClock()
    service = RecordingService(monotonic=clock, **options)
    worker = _Worker(1, _ServiceProcess(), SimpleNamespace(close=lambda: None),
                     SimpleNamespace(close=lambda: None), None)
    worker.ready = True
    service._worker = worker
    session = service.start(capture_request(tmp_path / "retry.wav"))
    service._dispatch(service._inbox.get_nowait())
    return SimpleNamespace(service=service, worker=worker, session=session, clock=clock,
                           command=worker.outgoing.get_nowait())


def proof_for(probe, *, prewarm=False, before=None, **changes):
    state = probe.service._pending_ve_prewarm if prewarm else probe.session
    budget = state.startup_budget if prewarm else state._startup_budget
    identity = probe.request.warmup_id if prewarm else probe.session.request.request_id
    signature = probe.request.signature if prewarm else probe.service._request_signature(probe.session.request)
    before = before or VeLifecycleCounts(0, 0, 0, 0, 0, 0)
    after = VeLifecycleCounts(*(getattr(before, field) + 1 for field in before.__dataclass_fields__))
    values = dict(request_id=identity, generation=1, signature=signature, attempt=2,
                  old_task_id="old", new_task_id="new", startup_budget=budget,
                  cleanup_confirmed_at=budget.first_attempt_deadline,
                  before_counts=before, after_cleanup_counts=after)
    values.update(changes)
    return VeResourceRetryProof(**values)


@pytest.mark.parametrize("invalid", [None, "not_cancelled", "hot", "started", "progress", "frames", "failure"])
def test_pending_cancel_cleanup_exception_keeps_lease_and_rejects_other_terminals(
        monkeypatch, tmp_path, invalid):
    from base.recording_process_protocol import RecordingCancelled, RecordingFailure
    from base.recording_service import RecordingSession, RecordingCallbacks

    probe = recording_probe(monkeypatch, tmp_path)
    service, session = probe.service, probe.session
    older = RecordingSession(service, capture_request(tmp_path / "older.wav", request_id="older"),
                             RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    service._sessions["older"] = older
    if invalid != "not_cancelled":
        service._request_cancel(session)
    if invalid == "hot":
        session._cold_start = False
    if invalid == "started":
        service._event(probe.worker, RecordingEvent(1, session.request.request_id, "started", probe.clock()))
    if invalid == "progress":
        session._target_reached_at = probe.clock()
    descriptor = RecordingCancelled(session.request.request_id, session.request.path,
                                    1 if invalid == "frames" else 0, 0, handles_released=False)
    kind = "cancelled"
    if invalid == "failure":
        descriptor = RecordingFailure(session.request.request_id, "clear_task", session.request.path,
                                      "native cleanup failed", handles_released=False)
        kind = "failed"
    service._event(probe.worker, RecordingEvent(1, session.request.request_id, kind, descriptor))
    if invalid is not None:
        assert probe.worker.retiring
        assert session.state == "failed"
    else:
        assert session.state == "cancelled" and session.descriptor == descriptor
        assert session._deadline is None and not session._child_released
        assert not session.released.is_set() and service.is_path_leased(session.request.path)
        assert not service.can_start_recording and not probe.worker.retiring
        probe.clock.advance(20)
        service._tick()
        assert not probe.worker.retiring and older.failure is None
        older._trusted_terminal = True
        service._tick()
        assert probe.worker.retiring and not session.released.is_set()


@pytest.mark.parametrize("timeout,expected", [(2.0, 2.0), (10.0, 10.0), (30.0, 10.0)])
def test_recording_and_prewarm_commands_share_authority_budget(monkeypatch, tmp_path, timeout, expected):
    recording = recording_probe(monkeypatch, tmp_path, start_timeout=timeout)
    session = recording.session
    budget = recording.command.startup_budget
    assert budget is not None
    assert budget is session._startup_budget
    assert budget.started_at == recording.clock()
    assert session._deadline == budget.deadline == recording.clock() + expected
    warm = _service_prewarm_probe(monkeypatch, tmp_path, start_timeout=timeout)
    pending = warm.service._pending_ve_prewarm
    assert warm.command.startup_budget is pending.startup_budget
    assert pending.start_deadline == warm.clock() + expected


def test_recording_retry_updates_counts_without_refreshing_deadline(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    proof = proof_for(probe)
    probe.clock.advance(proof.cleanup_confirmed_at - probe.clock())
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert probe.session._startup_retry_proof == proof
    assert probe.session._deadline == proof.startup_budget.deadline
    expected = VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    assert probe.service._expected_next_lifecycle_counts == expected
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "started", probe.clock()))
    assert probe.session.state == "recording"
    assert probe.service._expected_next_lifecycle_counts == expected
    target = probe.session.request.target_samples
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "progress",
        RecordingProgress(proof.request_id, 1, target, probe.clock())))
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "capture_slot_released",
        CaptureSlotReleased(proof.request_id, 1, probe.clock(), target, True, True, expected)))
    assert not probe.worker.retiring
    assert probe.service._retained_lifecycle_counts == expected
    assert probe.service._actual_lifecycle_counts == expected


@pytest.mark.parametrize("invalid", ["budget", "baseline", "signature", "duplicate", "started", "future_time"])
def test_current_request_contradictory_proof_retires_generation(monkeypatch, tmp_path, invalid):
    probe = recording_probe(monkeypatch, tmp_path)
    proof = proof_for(probe)
    probe.clock.advance(proof.cleanup_confirmed_at - probe.clock())
    if invalid == "budget":
        proof = replace(proof, startup_budget=VeStartupBudget.create(99.9))
    elif invalid == "baseline":
        proof = proof_for(probe, before=VeLifecycleCounts(1, 1, 1, 1, 1, 1))
    elif invalid == "signature":
        proof = replace(proof, signature=(*proof.signature[:2], (1, 7), *proof.signature[3:]))
    elif invalid == "started":
        probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "started", probe.clock()))
    elif invalid == "future_time":
        probe.clock.advance(-.1)
    event = RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof)
    probe.service._event(probe.worker, event)
    if invalid == "duplicate":
        probe.service._event(probe.worker, event)
    assert probe.worker.retiring
    assert probe.session.failure.stage == "protocol"


def test_stale_retry_proof_cannot_change_current_session(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    proof = replace(proof_for(probe), generation=2)
    # Current worker advances; stale proof is well formed for its old generation.
    probe.worker.generation = probe.session.generation = 3
    probe.service._event(probe.worker, RecordingEvent(2, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert not probe.worker.retiring
    assert probe.session._startup_retry_proof is None


def test_retry_proof_cannot_authorize_rebuilding_a_retained_resource(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    proof = proof_for(probe)
    probe.clock.advance(3.5)
    probe.service._retained_ve_signature = proof.signature
    probe.session._cold_start = False
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert probe.worker.retiring
    assert probe.session._startup_retry_proof is None


def test_vk_cancel_and_failure_deadlines_use_service_clock(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    probe.service._request_cancel(probe.session)
    assert probe.session._deadline == probe.clock() + probe.service._cancel_timeout
    probe.service._fail(probe.session, "startup_cleanup_timeout", "cleanup unconfirmed")
    assert probe.session._deadline == probe.clock() + probe.service._cancel_timeout


@pytest.mark.parametrize("stage", sorted(VE_STARTUP_RECOVERY_TERMINAL_STAGES))
def test_prewarm_recovery_failure_notifies_before_death_without_outer_retry(monkeypatch, tmp_path, stage):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    result = _prewarm_result(probe, success=False, stage=stage, released=False)
    probe.service._event(probe.worker, RecordingEvent(1, probe.request.warmup_id, "ve_prewarm_terminal", result))
    assert probe.process.alive
    assert len(probe.completions) == 1
    assert probe.completions[0].stage == stage
    assert not probe.completions[0].ownership_safe
    probe.process.alive = False
    probe.service._tick()
    assert probe.service._pending_ve_prewarm is None
    assert probe.service._worker is None


@pytest.mark.parametrize("failure", ["timeout", "fatal", "death", "broken"])
def test_expired_prewarm_budget_never_gets_outer_retry(monkeypatch, tmp_path, failure):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    probe.clock.advance(probe.command.startup_budget.deadline - probe.clock())
    if failure == "fatal":
        from base.recording_process_protocol import WorkerFatal
        probe.service._event(probe.worker, RecordingEvent(1, "", "worker_fatal", WorkerFatal(1, "worker", "lost")))
    elif failure == "broken":
        probe.service._dispatch(("broken", probe.worker, "closed"))
    elif failure == "death":
        probe.process.alive = False
    probe.service._tick()
    probe.process.alive = False
    probe.service._tick()
    assert len(probe.completions) == 1
    assert probe.service._pending_ve_prewarm is None


@pytest.mark.parametrize("failure", ["fatal", "death", "broken"])
def test_lost_prewarm_owner_during_cleanup_does_not_reopen_budget(monkeypatch, tmp_path, failure):
    from base.recording_process_protocol import WorkerFatal

    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    probe.clock.advance(3.5)
    if failure == "fatal":
        probe.service._event(probe.worker, RecordingEvent(1, "", "worker_fatal", WorkerFatal(1, "worker", "lost")))
    elif failure == "broken":
        probe.service._dispatch(("broken", probe.worker, "closed during cleanup"))
    probe.process.alive = False
    probe.service._tick()
    assert len(probe.completions) == 1
    assert probe.service._pending_ve_prewarm is None


def test_prewarm_retry_success_validates_adjusted_counts(monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    proof = proof_for(probe, prewarm=True)
    probe.clock.advance(proof.cleanup_confirmed_at - probe.clock())
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert probe.service._pending_ve_prewarm.startup_retry_proof == proof
    _prewarm_started(probe)
    _prewarm_detaching(probe)
    result = replace(_prewarm_result(probe, success=True), lifecycle_counts=VeLifecycleCounts(2, 2, 2, 1, 1, 1))
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "ve_prewarm_terminal", result))
    assert len(probe.completions) == 1 and probe.completions[0].success
    assert probe.service._actual_lifecycle_counts == result.lifecycle_counts


@pytest.mark.parametrize("prewarm", [False, True])
def test_replacement_started_cannot_inherit_first_attempt_timestamp(monkeypatch, tmp_path, prewarm):
    probe = (_service_prewarm_probe(monkeypatch, tmp_path) if prewarm
             else recording_probe(monkeypatch, tmp_path))
    proof = proof_for(probe, prewarm=prewarm)
    probe.clock.advance(3.5)
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    if prewarm:
        _prewarm_started(probe, started_at=proof.startup_budget.started_at)
        assert probe.worker.retiring
    else:
        probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, "started", proof.startup_budget.started_at))
        assert probe.session.state == "starting"
        assert probe.session._capture_deadline is None


def test_full_release_updates_actual_baseline_once_and_empty_release_does_not(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    service = probe.service
    signature = service._request_signature(probe.session.request)
    service._retained_ve_signature = signature
    service._retained_lifecycle_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    service._actual_lifecycle_counts = service._retained_lifecycle_counts
    for released_signature in (signature, None):
        service._pending_ve_release = _PendingVeRelease(None, generation=1, sent=True)
        service._event(probe.worker, RecordingEvent(1, "", "ve_released", VeReleaseOutcome(1, released_signature)))
        assert service._actual_lifecycle_counts == VeLifecycleCounts(1, 1, 1, 1, 1, 1)
        assert service._expected_next_lifecycle_counts == VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    proof = proof_for(probe, before=service._actual_lifecycle_counts)
    probe.clock.advance(proof.cleanup_confirmed_at - probe.clock())
    service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert not probe.worker.retiring
    assert service._expected_next_lifecycle_counts == VeLifecycleCounts(3, 3, 3, 2, 2, 2)


def test_capture_forwards_original_budget_after_writer_delay(tmp_path):
    from base.recording_capture import RecordingCapture

    clock = DiscoveryClock()
    budget = VeStartupBudget.create(clock())
    calls, writers = [], []
    on_retry = lambda proof: None

    def writer(*args, **kwargs):
        clock.advance(4)
        writers.append(args)
        return object()

    def stream(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(start=lambda: False)

    capture = RecordingCapture(capture_request(tmp_path / "delayed.wav"),
        writer_factory=writer, ve_stream_factory=stream, startup_budget=budget,
        on_retry=on_retry)
    capture._open()
    assert len(writers) == len(calls) == 1
    assert calls[0]["startup_budget"] is budget
    assert calls[0]["on_retry"] is on_retry
    assert calls[0]["stop_event"] is capture._stop_requested
    assert budget.first_attempt_remaining(clock()) == 0


def test_capture_legacy_vk_factory_receives_no_new_optional_kwargs(tmp_path):
    from base.recording_capture import RecordingCapture

    calls = []
    def stream(*, request, callback, fail, stop_event):
        calls.append(request)
        return SimpleNamespace(start=lambda: False)
    capture = RecordingCapture(capture_request(tmp_path / "legacy.wav"),
        writer_factory=lambda *a, **kw: None, ve_stream_factory=stream)
    capture._open()
    assert calls == [capture.request]


def test_controller_default_cold_budget_is_created_once_at_bind(monkeypatch, tmp_path):
    from base.ve3668n_resource import VeResourceController

    clock = DiscoveryClock()
    controller = VeResourceController(sdk_factory=lambda: pytest.fail("no SDK needed"), clock=clock)
    observed = []
    monkeypatch.setattr(controller, "_bind_startup", lambda adapter: observed.append(adapter.startup_budget) or False)
    adapter = controller.stream(request=capture_request(tmp_path / "cold.wav"),
        callback=lambda *a: None, fail=lambda *a: None, stop_event=threading.Event())
    clock.advance(2)
    assert not adapter.start()
    assert observed == [VeStartupBudget.create(102)]


def test_prewarm_start_does_not_block_worker_cancel(monkeypatch, tmp_path):
    import queue
    import time
    from base import recording_worker as module
    from base.recording_process_protocol import VePrewarmRequest
    from base.ve3668n_capture_timing import VeCaptureProgress

    commands, sent = queue.Queue(), queue.Queue()
    entered, cancelled, finish = threading.Event(), threading.Event(), threading.Event()
    adapters = []

    class Adapter:
        def __init__(self):
            self.started = threading.Event()
            self.completed = threading.Event()
            self.stop_event = threading.Event()
            self.failure_snapshot = None
            self.progress_snapshot = VeCaptureProgress(None, 0, None)
            self.handles_released = True
            self.diagnostics = ()

        def start(self):
            entered.set()
            assert self.stop_event.wait(2), "control loop did not deliver cancel during startup"
            finish.set()
            self.completed.set()
            return False

        def stop(self):
            self.stop_event.set()
            cancelled.set()

    class Controller:
        lifecycle_counts = VeLifecycleCounts(0, 0, 0, 0, 0, 0)
        def __init__(self, **kwargs):
            pass
        def prewarm(self, **kwargs):
            assert kwargs["startup_budget"] is budget
            adapter = Adapter()
            adapters.append(adapter)
            return adapter
        def close(self, timeout):
            for adapter in adapters:
                adapter.stop()
            return SimpleNamespace(success=True)

    class Connection:
        def poll(self, timeout):
            time.sleep(.001)
            return not commands.empty()
        def recv(self):
            return commands.get_nowait()
        def send(self, event):
            sent.put(event)
        def close(self):
            pass

    monkeypatch.setattr(module, "VeResourceController", Controller)
    monkeypatch.setattr(module.multiprocessing, "parent_process", lambda: None)
    monkeypatch.setattr(module, "exit_with_log_drain", lambda *a: (_ for _ in ()).throw(SystemExit()))
    request = capture_request(tmp_path / "unused.wav")
    warm = VePrewarmRequest.create("warm", request.device, request.channels, request.sample_rate, attempt=1)
    budget = VeStartupBudget.create(time.monotonic())
    commands.put(RecordingEvent(1, "warm", "prewarm_ve", warm, startup_budget=budget))
    worker = threading.Thread(target=module.recording_worker,
        args=(Connection(), Connection(), 1, None, {}, .3))
    worker.start()
    try:
        assert entered.wait(1)
        commands.put(RecordingEvent(1, "warm", "cancel"))
        assert cancelled.wait(.5)
        assert finish.wait(.5)
    finally:
        for adapter in adapters:
            adapter.stop()
        commands.put(RecordingEvent(1, "", "shutdown"))
        worker.join(3)
        assert not worker.is_alive()


def test_cancelled_cold_start_blocks_reuse_but_waits_for_older_finalizer(monkeypatch, tmp_path):
    from base.recording_service import RecordingSession, RecordingCallbacks
    from base.recording_process_protocol import RecordingCancelled

    probe = recording_probe(monkeypatch, tmp_path)
    service, session = probe.service, probe.session
    older = RecordingSession(service, capture_request(tmp_path / "older.wav", request_id="older"), RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    service._sessions["older"] = older
    session.cancel_requested = True
    service._event(probe.worker, RecordingEvent(1, session.request.request_id, "cancelled",
        RecordingCancelled(session.request.request_id, session.request.path, 0, 0)))
    assert not service.can_start_recording
    service._tick()
    assert not probe.worker.retiring
    assert older.failure is None
    # Once the old finalizer has published a trusted terminal it no longer
    # needs the process; retirement must preserve its reader/file ownership.
    older._trusted_terminal = True
    service._tick()
    assert probe.worker.retiring
    assert older.failure is None


@pytest.mark.parametrize("prewarm", [False, True])
def test_worker_retry_proof_precedes_started_on_real_capture_path(monkeypatch, tmp_path, prewarm):
    import queue
    import time
    from base import recording_worker as module
    from base.recording_process_protocol import VePrewarmRequest
    from base.ve3668n_resource import VeResourceController
    from unit_test.base.ve3668n_fakes import CaptureSDK

    commands, sent = queue.Queue(), queue.Queue()
    entered, gate = threading.Event(), threading.Event()
    clock = DiscoveryClock()
    sdk_instances, controllers, owners, captures = [], [], [], []
    original_capture = module.RecordingCapture

    def block(*args):
        entered.set()
        assert gate.wait(3)

    def sdk_factory():
        sdk = CaptureSDK(hooks={"create_task": block} if not sdk_instances else {})
        sdk.values = (.125, -.25)
        sdk_instances.append(sdk)
        owners.append(threading.current_thread())
        if len(owners) == 2:
            assert not owners[0].is_alive()
        return sdk

    def controller_factory(**kwargs):
        kwargs.update(sdk_factory=sdk_factory, clock=clock)
        controller = VeResourceController(**kwargs)
        controllers.append(controller)
        return controller

    def capture_factory(*args, **kwargs):
        capture = original_capture(*args, **kwargs)
        captures.append(capture)
        return capture

    class Connection:
        def poll(self, timeout):
            time.sleep(.001)
            return not commands.empty()
        def recv(self):
            return commands.get_nowait()
        def send(self, event):
            sent.put(event)
        def close(self):
            pass

    monkeypatch.setattr(module, "VeResourceController", controller_factory)
    monkeypatch.setattr(module, "RecordingCapture", capture_factory)
    monkeypatch.setattr(module.multiprocessing, "parent_process", lambda: None)
    exits = []
    monkeypatch.setattr(module, "exit_with_log_drain", lambda code: exits.append(code))
    request = capture_request(tmp_path / "one-writer.wav", trim_samples=0)
    if prewarm:
        request = VePrewarmRequest.create("warm", request.device, request.channels, request.sample_rate, attempt=1)
    identity = request.warmup_id if prewarm else request.request_id
    budget = VeStartupBudget.create(clock())
    commands.put(RecordingEvent(1, identity, "prewarm_ve" if prewarm else "start", request,
                               startup_budget=budget))
    worker = threading.Thread(target=module.recording_worker,
        args=(Connection(), Connection(), 1, None, {}, .5))
    worker.start()
    events = []
    try:
        assert entered.wait(1)
        clock.advance(3.6)
        assert controllers[0]._startup_attempt.revoked.wait(1)
        gate.set()
        terminal_kind = "ve_prewarm_terminal" if prewarm else "completed"
        while not events or events[-1].kind != terminal_kind:
            events.append(sent.get(timeout=3))
            assert events[-1].kind not in ("failed", "worker_fatal")
        kinds = [event.kind for event in events]
        assert kinds.count(VE_RESOURCE_RETRY) == 1
        started = "ve_prewarm_started" if prewarm else "started"
        assert kinds.count(started) == 1
        assert kinds.index(VE_RESOURCE_RETRY) < kinds.index(started) < kinds.index(terminal_kind)
        proof = next(event.payload for event in events if event.kind == VE_RESOURCE_RETRY)
        assert proof.startup_budget is budget
        assert proof.before_counts == VeLifecycleCounts(0, 0, 0, 0, 0, 0)
        assert proof.after_cleanup_counts == VeLifecycleCounts(1, 1, 0, 1, 1, 1)
        assert len(sdk_instances) == 2
        assert sdk_instances[0].calls("read_task_data") == 0
        assert len(captures) == (0 if prewarm else 1)
        if prewarm:
            assert events[-1].payload.success
            assert not (tmp_path / "one-writer.wav").exists()
        else:
            commands.put(RecordingEvent(1, identity, "result_ack", "accepted"))
            assert events[-1].payload.raw_frames == request.target_samples
    finally:
        gate.set()
        commands.put(RecordingEvent(1, "", "shutdown"))
        worker.join(4)
        assert not worker.is_alive()
        for owner in owners:
            owner.join(1)
            assert not owner.is_alive()
    assert exits == []


def test_native_start_failure_waits_bounded_cleanup_before_terminal_snapshot(tmp_path):
    from base.ve3668n_resource import VeResourceController
    from unit_test.base.ve3668n_fakes import CaptureSDK

    entered, gate, returned = threading.Event(), threading.Event(), threading.Event()
    def stop(*args):
        entered.set()
        assert gate.wait(2)
    sdk = CaptureSDK(failures=("start_task",), hooks={"stop_task": stop})
    controller = VeResourceController(sdk_factory=lambda: sdk)
    failures = []
    adapter = controller.stream(request=capture_request(tmp_path / "unused.wav"),
        callback=lambda *a: None, fail=lambda *a: failures.append(a), stop_event=threading.Event())
    def start():
        assert not adapter.start()
        returned.set()
    runner = threading.Thread(target=start)
    runner.start()
    try:
        assert entered.wait(1)
        assert not returned.wait(.02)
        gate.set()
        assert returned.wait(1)
        assert adapter.handles_released
        assert failures[0][0] == "start_task"
        assert sdk.closed
    finally:
        gate.set()
        runner.join(2)
        if controller._owner is not None:
            controller._owner.join(1)


@pytest.mark.parametrize("late_delivery", [False, True])
def test_cancel_before_retry_proof_delivery_preserves_older_finalizer(monkeypatch, tmp_path, late_delivery):
    from base.recording_service import RecordingSession, RecordingCallbacks
    from base.recording_process_protocol import RecordingCancelled

    probe = recording_probe(monkeypatch, tmp_path)
    service, session = probe.service, probe.session
    proof = proof_for(probe)
    older = RecordingSession(service, capture_request(tmp_path / "older.wav", request_id="older"), RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    service._sessions["older"] = older
    notifications = []
    session.callbacks = RecordingCallbacks(
        started=lambda *a: notifications.append("started"),
        cancelled=lambda *a: notifications.append("cancelled"),
        failed=lambda *a: notifications.append("failed"))
    probe.clock.advance(3.5)
    service.cancel(session.request.request_id)
    service._dispatch(service._inbox.get_nowait())
    deadline = session._deadline
    if late_delivery:
        probe.clock.advance(7)
    # The reliable FIFO may already contain this proof when parent cancel is
    # requested. It is evidence of past cleanup, not a new request to retry.
    service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert not probe.worker.retiring
    assert older.failure is None and session.failure is None
    assert session.cancel_requested and session.state == "starting"
    assert session._startup_retry_proof == proof
    assert session._deadline == deadline
    assert notifications == []
    service._event(probe.worker, RecordingEvent(1, proof.request_id, "started", proof.cleanup_confirmed_at))
    assert session.state == "starting"
    assert session._capture_deadline.snapshot().started_at == proof.cleanup_confirmed_at
    assert session._deadline == deadline and notifications == []
    terminal = RecordingEvent(1, proof.request_id, "cancelled",
        RecordingCancelled(proof.request_id, session.request.path, 0, 0))
    service._event(probe.worker, terminal)
    service._event(probe.worker, terminal)
    assert notifications == ["cancelled"]
    service._tick()
    assert not probe.worker.retiring and older.failure is None
    assert not service.can_start_recording
    older._trusted_terminal = True
    service._tick()
    assert probe.worker.retiring and older.failure is None


@pytest.mark.parametrize("invalid", ["budget", "signature", "counts", "duplicate"])
def test_cancel_does_not_weaken_retry_proof_validation(monkeypatch, tmp_path, invalid):
    probe = recording_probe(monkeypatch, tmp_path)
    proof = proof_for(probe)
    probe.clock.advance(3.5)
    probe.service._request_cancel(probe.session)
    if invalid == "budget":
        proof = replace(proof, startup_budget=VeStartupBudget.create(99.9))
    elif invalid == "signature":
        proof = replace(proof, signature=(*proof.signature[:2], (1, 7), *proof.signature[3:]))
    elif invalid == "counts":
        proof = proof_for(probe, before=VeLifecycleCounts(1, 1, 1, 1, 1, 1))
    event = RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof)
    probe.service._event(probe.worker, event)
    if invalid == "duplicate":
        assert not probe.worker.retiring
        probe.service._event(probe.worker, event)
    assert probe.worker.retiring
    assert probe.session.failure.stage == "protocol"


@pytest.mark.parametrize("prewarm", [False, True])
def test_shutdown_reconciles_recording_proof_and_ignores_removed_prewarm(monkeypatch, tmp_path, prewarm):
    probe = (_service_prewarm_probe(monkeypatch, tmp_path) if prewarm
             else recording_probe(monkeypatch, tmp_path))
    proof = proof_for(probe, prewarm=prewarm)
    probe.clock.advance(3.5)
    probe.service.shutdown()
    probe.service._dispatch(probe.service._inbox.get_nowait())
    deadline = probe.service._shutdown_deadline
    before_counts = probe.service._actual_lifecycle_counts
    probe.service._event(probe.worker, RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof))
    assert not probe.worker.retiring
    assert probe.service._shutdown_deadline == deadline
    if prewarm:
        assert probe.service._actual_lifecycle_counts == before_counts
        assert probe.service._pending_ve_prewarm is None
        assert probe.completions == []
    else:
        assert probe.service._actual_lifecycle_counts == proof.after_cleanup_counts
        assert probe.session.cancel_requested and probe.session.failure is None


def queued_retry_lifecycle(probe):
    from base.recording_process_protocol import RecordingResult

    proof = proof_for(probe)
    request = probe.session.request
    frames = request.target_samples
    counts = VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    trim = request.trim_samples if request.trim_samples < frames else 0
    return [
        RecordingEvent(1, proof.request_id, VE_RESOURCE_RETRY, proof),
        RecordingEvent(1, proof.request_id, "started", proof.cleanup_confirmed_at),
        RecordingEvent(1, proof.request_id, "finalizing"),
        RecordingEvent(1, proof.request_id, "progress",
                       RecordingProgress(proof.request_id, 1, frames, proof.cleanup_confirmed_at)),
        RecordingEvent(1, proof.request_id, "capture_slot_released",
                       CaptureSlotReleased(proof.request_id, 1, proof.cleanup_confirmed_at,
                                           frames, True, True, counts)),
        RecordingEvent(1, proof.request_id, "completed", RecordingResult(
            proof.request_id, request.purpose, request.path, request.sample_rate,
            request.channels, frames, frames - trim, True)),
    ]


@pytest.mark.parametrize("boundary", range(6), ids=[
    "before-proof", "before-started", "before-finalizing", "before-progress",
    "before-slot", "before-terminal",
])
@pytest.mark.parametrize("shutdown", [False, True], ids=["cancel", "shutdown"])
def test_cancel_reconciles_complete_queued_lifecycle_without_reviving_recording(
        monkeypatch, tmp_path, boundary, shutdown):
    from base.recording_service import RecordingSession, RecordingCallbacks

    probe = recording_probe(monkeypatch, tmp_path)
    service, session = probe.service, probe.session
    older = RecordingSession(service, capture_request(tmp_path / "older.wav", request_id="older"), RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    service._sessions["older"] = older
    notifications = []
    session.callbacks = RecordingCallbacks(
        started=lambda *a: notifications.append("started"),
        finalizing=lambda *a: notifications.append("finalizing"),
        cancelled=lambda *a: notifications.append("cancelled"),
        failed=lambda *a: notifications.append("failed"),
        result_ready=lambda *a: notifications.append("result_ready"))
    events = queued_retry_lifecycle(probe)
    probe.clock.advance(3.5)
    for event in events[:boundary]:
        service._event(probe.worker, event)
    if shutdown:
        service.shutdown()
    else:
        service.cancel(session.request.request_id)
    service._dispatch(service._inbox.get_nowait())
    deadline = session._deadline
    state_at_cancel = session.state
    notified_before_cancel = list(notifications)
    for event in events[boundary:-1]:
        service._event(probe.worker, event)
        assert not probe.worker.retiring and older.failure is None
        assert session.cancel_requested
        assert session._deadline == (None if session._slot_released_at is not None else deadline)
        assert session.state == state_at_cancel
        assert notifications == notified_before_cancel
    assert session._capture_deadline.snapshot().frames == session.request.target_samples
    assert session._slot_lifecycle_counts == VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    service._event(probe.worker, events[-1])
    service._event(probe.worker, events[-1])
    assert notifications == notified_before_cancel + ["cancelled"]
    assert session.state == "cancelled" and session.failure is None
    assert session.reader is None and session.released.is_set()
    assert not probe.worker.retiring and older.failure is None
    assert service._actual_lifecycle_counts == VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    assert not service._counts_retirement_pending
    service._tick()
    assert not probe.worker.retiring and older.failure is None


@pytest.mark.parametrize("invalid", ["slot-counts", "missing-progress", "future-progress", "missing-started", "terminal-before-slot"])
def test_cancel_keeps_full_queued_lifecycle_validation_strict(monkeypatch, tmp_path, invalid):
    from base.recording_service import RecordingSession, RecordingCallbacks

    probe = recording_probe(monkeypatch, tmp_path)
    older = RecordingSession(probe.service, capture_request(tmp_path / "older.wav", request_id="older"), RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    probe.service._sessions["older"] = older
    events = queued_retry_lifecycle(probe)
    probe.clock.advance(3.5)
    probe.service._request_cancel(probe.session)
    if invalid == "slot-counts":
        events[4] = replace(events[4], payload=replace(events[4].payload,
                            lifecycle_counts=VeLifecycleCounts(3, 3, 3, 1, 1, 1)))
    elif invalid == "missing-progress":
        del events[3]
    elif invalid == "future-progress":
        events[3] = replace(events[3], payload=replace(events[3].payload, last_frame_at=110))
    elif invalid == "missing-started":
        del events[1]
    else:
        events[4], events[5] = events[5], events[4]
    for event in events:
        probe.service._event(probe.worker, event)
    assert probe.worker.retiring
    assert probe.session.failure.stage == "protocol"
    assert older.failure.stage == "worker_retired_during_finalization"


@pytest.mark.parametrize("cancel_after_slot", [False, True])
def test_confirmed_slot_retires_capture_cancel_timer_while_terminal_is_delayed(
        monkeypatch, tmp_path, cancel_after_slot):
    from base.recording_service import RecordingSession, RecordingCallbacks

    probe = recording_probe(monkeypatch, tmp_path)
    service, session = probe.service, probe.session
    older = RecordingSession(service, capture_request(tmp_path / "older.wav", request_id="older"), RecordingCallbacks())
    older.generation = 1
    older.state = "finalizing"
    older._slot_released_at = probe.clock()
    service._sessions["older"] = older
    cancelled = []
    session.callbacks = RecordingCallbacks(cancelled=lambda *args: cancelled.append(args))
    events = queued_retry_lifecycle(probe)
    probe.clock.advance(3.5)
    if not cancel_after_slot:
        service._request_cancel(session)
    for event in events[:-1]:
        service._event(probe.worker, event)
    if cancel_after_slot:
        service._request_cancel(session)
    probe.clock.advance(service._cancel_timeout + .01)
    service._tick()
    assert session.failure is None and older.failure is None
    assert session._deadline is None and not probe.worker.retiring
    assert cancelled == []
    assert not session.released.is_set()
    service._event(probe.worker, events[-1])
    service._event(probe.worker, events[-1])
    assert session.state == "cancelled" and len(cancelled) == 1
    assert session.released.is_set() and session.reader is None
    assert older.failure is None and not probe.worker.retiring


def test_unconfirmed_slot_still_enforces_capture_cancel_timer(monkeypatch, tmp_path):
    probe = recording_probe(monkeypatch, tmp_path)
    probe.clock.advance(3.5)
    probe.service._request_cancel(probe.session)
    for event in queued_retry_lifecycle(probe)[:2]:
        probe.service._event(probe.worker, event)
    deadline = probe.session._deadline
    probe.clock.advance(probe.service._cancel_timeout + .01)
    probe.service._tick()
    assert deadline < probe.clock()
    assert probe.session.failure.stage == "cancel_timeout"
    assert probe.worker.retiring
