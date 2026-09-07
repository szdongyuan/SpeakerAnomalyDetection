"""Parent capacity-two admission and generation-retirement contracts."""
from dataclasses import replace
from types import SimpleNamespace
import queue

import pytest

from base.recording_process_protocol import (
    CaptureSlotReleased, RecordingCancelled, RecordingEvent, RecordingFailure,
    RecordingProgress, RecordingResult, VeLifecycleCounts, VeReleaseOutcome,
    WorkerFatal,
)
from base.recording_service import RecordingService, _Worker
from unit_test.base.test_recording_service import request
from unit_test.base.ve3668n_fakes import DiscoveryClock, capture_request


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    return RecordingService()


def test_admission_is_additive_and_reserved_atomically(service, tmp_path):
    assert service.can_start_recording
    first = service.start(request(tmp_path, request_id="A", path=str(tmp_path / "A.wav")))
    assert service.busy
    assert not service.can_start_recording

    first._child_released = True
    first._sent = True
    first.state = "delivering"
    service._capture_session = None
    assert service.busy
    assert service.can_start_recording

    second = service.start(request(tmp_path, request_id="B", path=str(tmp_path / "B.wav")))
    assert service._capture_session is second
    assert not service.can_start_recording
    with pytest.raises(RuntimeError, match="busy|capacity"):
        service.start(request(tmp_path, request_id="C", path=str(tmp_path / "C.wav")))


@pytest.mark.parametrize("condition", ["closing", "pending", "retiring", "uncertain"])
def test_admission_rejects_global_exclusion_states(service, condition):
    if condition == "closing":
        service._closing = True
    elif condition == "pending":
        service._pending_ve_release = object()
    elif condition == "uncertain":
        service._ownership_uncertain = True
    else:
        process = SimpleNamespace(pid=1, is_alive=lambda: True)
        service._worker = _Worker(1, process, None, None, None)
        service._worker.retiring = True
    assert not service.can_start_recording


@pytest.mark.parametrize("capacity", [True, False, 0, -1, 3, 99])
def test_pipeline_capacity_accepts_only_strict_one_or_two(monkeypatch, capacity):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="1 or 2"):
        RecordingService(pipeline_capacity=capacity)


@pytest.mark.parametrize("capacity", [1, 2])
def test_pipeline_capacity_allows_strict_supported_values(monkeypatch, capacity):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    assert RecordingService(pipeline_capacity=capacity)._pipeline_capacity == capacity


def ve_probe(monkeypatch, tmp_path):
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
    failures = queue.Queue()
    from base.recording_service import RecordingCallbacks
    session = service.start(capture_request(tmp_path / "A.wav"),
                            RecordingCallbacks(failed=lambda session, failure: failures.put(failure)))
    service._dispatch(service._inbox.get_nowait())
    worker.outgoing.get_nowait()
    service._event(worker, RecordingEvent(1, session.request.request_id, "started", 100.0))
    return SimpleNamespace(service=service, worker=worker, session=session,
                           clock=clock, failures=failures)


def slot_payload(probe, **changes):
    values = dict(request_id=probe.session.request.request_id, generation=1,
                  target_reached_at=100.1, raw_frames=probe.session.request.target_samples,
                  adapter_released=True, writer_released=True,
                  lifecycle_counts=VeLifecycleCounts(1, 1, 1, 0, 0, 0))
    values.update(changes)
    return CaptureSlotReleased(**values)


def release_probe_slot(probe):
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released", slot_payload(probe)))


def test_valid_slot_release_opens_admission_but_retains_busy_session(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released", slot_payload(probe)))
    evidence = probe.service.session_diagnostics(probe.session.request.request_id)
    assert probe.service.busy and probe.service.can_start_recording
    assert evidence["target_reached_at"] == 100.1
    assert evidence["capture_slot_released_at"] == 100.1
    assert evidence["lifecycle_counts"] == VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    assert evidence["admission_reason"] is None


def test_capacity_full_slot_release_records_backpressure_evidence(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    second_request = replace(
        capture_request(tmp_path / "B.wav"), request_id="ve-capture-B")
    second = probe.service.start(second_request)
    probe.service._dispatch(probe.service._inbox.get_nowait())
    probe.worker.outgoing.get_nowait()
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "started", 100.1))
    probe.clock.advance(.1)
    target_reached_at = probe.clock()
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "progress",
        RecordingProgress(second.request.request_id, 1,
                          second.request.target_samples, target_reached_at)))
    payload = CaptureSlotReleased(
        request_id=second.request.request_id, generation=1,
        target_reached_at=target_reached_at, raw_frames=second.request.target_samples,
        adapter_released=True, writer_released=True,
        lifecycle_counts=VeLifecycleCounts(1, 1, 1, 0, 0, 0))
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "capture_slot_released", payload))

    assert not probe.service.can_start_recording
    assert probe.service.session_diagnostics(second.request.request_id)[
        "admission_reason"] == "CAPACITY_BACKPRESSURE"


@pytest.mark.parametrize("case", [
    "completed_path", "completed_count", "completed_channels", "completed_rate",
    "completed_purpose", "completed_handles",
    "failed_path", "failed_count", "failed_order", "failed_handles",
    "cancelled_path", "cancelled_count", "cancelled_order", "cancelled_handles",
])
def test_invalid_terminal_after_slot_retires_before_trusting_release_or_ack(
    monkeypatch, tmp_path, case,
):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    request = probe.session.request
    kind = case.split("_", 1)[0]
    if kind == "completed":
        descriptor = RecordingResult(
            request.request_id, request.purpose, request.path,
            request.sample_rate, request.channels,
            request.target_samples, request.target_samples - request.trim_samples,
            False, handles_released=True)
    elif kind == "failed":
        descriptor = RecordingFailure(
            request.request_id, "metadata", request.path, "metadata failed",
            raw_frames=request.target_samples,
            written_frames=request.target_samples,
            handles_released=True)
    else:
        descriptor = RecordingCancelled(
            request.request_id, request.path,
            request.target_samples, request.target_samples,
            handles_released=True)
    fault = case.split("_", 1)[1]
    if fault == "path":
        descriptor = replace(descriptor, path=str(tmp_path / "other.wav"))
    elif fault == "count":
        descriptor = replace(descriptor, raw_frames=request.target_samples + 1)
    elif fault == "channels":
        descriptor = replace(descriptor, channels=(1, 7))
    elif fault == "rate":
        descriptor = replace(descriptor, sample_rate=request.sample_rate + 1)
    elif fault == "purpose":
        descriptor = replace(descriptor, purpose="calibration")
    elif fault == "order":
        field = "written_frames" if kind == "failed" else "final_frames"
        descriptor = replace(descriptor, raw_frames=request.target_samples - 1,
                             **{field: request.target_samples})
    else:
        descriptor = replace(descriptor, handles_released=False)

    probe.service._event(probe.worker, RecordingEvent(
        1, request.request_id, kind, descriptor))

    assert probe.worker.retiring
    assert probe.session.descriptor is None
    assert probe.session._child_released is False
    assert not probe.session.acknowledged
    assert probe.worker.outgoing.empty()
    assert request.request_id in probe.service._sessions
    assert probe.service.is_path_leased(request.path)
    assert not probe.session.released.is_set()
    assert not probe.service.can_start_recording
    if case == "failed_handles":
        failure = probe.failures.get_nowait()
        assert failure is descriptor
        assert failure.stage == "metadata" and not failure.handles_released

    probe.worker.process.is_alive = lambda: False
    probe.worker.process.join = lambda timeout=0: None
    probe.worker.process.close = lambda: None
    probe.worker.control = SimpleNamespace(close=lambda: None)
    probe.worker.preview = SimpleNamespace(close=lambda: None)
    probe.service._dead(probe.worker)
    assert probe.session._child_released is True
    assert probe.session.released.is_set()
    assert not probe.service.is_path_leased(request.path)
    assert probe.service.can_start_recording


@pytest.mark.parametrize("kind", ["completed", "failed", "cancelled"])
def test_constructor_bypassed_terminal_request_contradictions_retire(
    monkeypatch, tmp_path, kind,
):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    request = probe.session.request
    if kind == "completed":
        descriptor = RecordingResult(
            "other", request.purpose, request.path, request.sample_rate,
            request.channels, request.target_samples,
            request.target_samples - request.trim_samples, False)
    elif kind == "failed":
        descriptor = RecordingFailure(
            "other", "metadata", request.path, "metadata failed",
            request.target_samples, request.target_samples)
    else:
        descriptor = RecordingCancelled(
            "other", request.path, request.target_samples, request.target_samples)
    event = RecordingEvent.__new__(RecordingEvent)
    object.__setattr__(event, "generation", 1)
    object.__setattr__(event, "request_id", request.request_id)
    object.__setattr__(event, "kind", kind)
    object.__setattr__(event, "payload", descriptor)
    object.__setattr__(event, "version", 1)

    probe.service._event(probe.worker, event)

    assert probe.worker.retiring
    assert probe.session.descriptor is None
    assert probe.session._child_released is False
    assert probe.service.is_path_leased(request.path)


def test_terminal_payload_type_retires_even_when_event_kind_is_corrupted(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    request = probe.session.request
    descriptor = RecordingResult(
        request.request_id, request.purpose, request.path, request.sample_rate,
        request.channels, request.target_samples,
        request.target_samples - request.trim_samples, False)
    event = RecordingEvent.__new__(RecordingEvent)
    object.__setattr__(event, "generation", 1)
    object.__setattr__(event, "request_id", request.request_id)
    object.__setattr__(event, "kind", "progress")
    object.__setattr__(event, "payload", descriptor)
    object.__setattr__(event, "version", 1)

    probe.service._event(probe.worker, event)

    assert probe.worker.retiring
    assert probe.session.descriptor is None
    assert probe.session._child_released is False


def test_reader_constructor_failure_isolates_only_finalizer_and_preserves_active_capture(
    monkeypatch, tmp_path,
):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    probe.service._reader_factory = lambda descriptor, completed: (_ for _ in ()).throw(
        RuntimeError("injected reader construction failure"))
    from base.recording_service import RecordingCallbacks
    second_failures = queue.Queue()
    second = probe.service.start(
        capture_request(tmp_path / "B.wav", request_id="B"),
        RecordingCallbacks(failed=lambda session, failure: second_failures.put(failure)))
    probe.service._dispatch(probe.service._inbox.get_nowait())
    start = probe.worker.outgoing.get_nowait()
    assert start.kind == "start" and start.request_id == "B"
    probe.service._event(probe.worker, RecordingEvent(1, "B", "started", probe.clock()))
    request = probe.session.request
    descriptor = RecordingResult(
        request.request_id, request.purpose, request.path, request.sample_rate,
        request.channels, request.target_samples,
        request.target_samples - request.trim_samples, False)

    probe.service._event(probe.worker, RecordingEvent(
        1, request.request_id, "completed", descriptor))

    failure = probe.failures.get_nowait()
    assert failure.stage == "service" and "construction failure" in failure.message
    assert probe.session.released.is_set()
    assert not probe.service.is_path_leased(request.path)
    assert not probe.worker.retiring and not probe.service._ownership_uncertain
    assert probe.service._worker is probe.worker
    assert second.state == "recording" and second_failures.empty()
    ack = probe.worker.outgoing.get_nowait()
    assert ack.kind == "result_ack" and ack.request_id == request.request_id


@pytest.mark.parametrize("fault", ["before_progress", "wrong_id", "false_flag", "late"])
def test_current_generation_slot_contradictions_retire_without_opening_admission(
    monkeypatch, tmp_path, fault,
):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    if fault != "before_progress":
        probe.service._event(probe.worker, RecordingEvent(
            1, probe.session.request.request_id, "progress",
            RecordingProgress(probe.session.request.request_id, 1,
                              probe.session.request.target_samples, 100.1)))
    if fault == "late":
        probe.clock.advance(.51)
    payload = slot_payload(probe, writer_released=False) if fault == "false_flag" else slot_payload(probe)
    request_id = "wrong" if fault == "wrong_id" else probe.session.request.request_id
    if fault == "wrong_id":
        object.__setattr__(payload, "request_id", "wrong")
    probe.service._event(probe.worker, RecordingEvent(
        1, request_id, "capture_slot_released", payload))
    assert probe.worker.retiring and not probe.service.can_start_recording
    assert probe.service._capture_session is probe.session


def test_generation_retirement_fans_out_active_and_finalizer_once(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released", slot_payload(probe)))
    second_failures = queue.Queue()
    from base.recording_service import RecordingCallbacks
    second = probe.service.start(capture_request(tmp_path / "B.wav", request_id="B"),
                                 RecordingCallbacks(failed=lambda session, failure: second_failures.put(failure)))
    probe.service._dispatch(probe.service._inbox.get_nowait())
    probe.worker.outgoing.get_nowait()
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "started", probe.clock()))
    fatal = WorkerFatal(1, "native/read", "idle read failed")
    probe.service._event(probe.worker, RecordingEvent(1, "", "worker_fatal", fatal))
    assert probe.session.failure.stage == "worker_retired_during_finalization"
    assert second_failures.get_nowait().stage == "native/read"
    assert probe.service.is_path_leased(probe.session.request.path)
    assert probe.service.is_path_leased(second.request.path)
    assert not probe.service.can_start_recording


def test_finalizer_handle_uncertainty_retires_newer_active_capture_without_blame_shift(
        monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    from base.recording_service import RecordingCallbacks
    second_failures = queue.Queue()
    second = probe.service.start(
        capture_request(tmp_path / "B.wav", request_id="B"),
        RecordingCallbacks(failed=lambda session, failure: second_failures.put(failure)),
    )
    probe.service._dispatch(probe.service._inbox.get_nowait())
    probe.worker.outgoing.get_nowait()
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "started", probe.clock()))
    request = probe.session.request
    primary = RecordingFailure(
        request.request_id,
        "close_wav",
        request.path,
        "PRIMARY-WRITER-CLOSE-ERROR; SECONDARY-CLEANUP-ERROR",
        raw_frames=request.target_samples,
        written_frames=request.target_samples,
        handles_released=False,
    )

    probe.service._event(probe.worker, RecordingEvent(
        1, request.request_id, "failed", primary))

    assert probe.worker.retiring and probe.service._ownership_uncertain
    assert probe.session.failure is primary
    assert probe.session.failure.stage == "close_wav"
    assert probe.session.failure.message.split("; ") == [
        "PRIMARY-WRITER-CLOSE-ERROR", "SECONDARY-CLEANUP-ERROR"]
    newer_failure = second_failures.get_nowait()
    assert newer_failure.stage == "close"
    assert newer_failure.message == "worker retained file/device handles"
    assert not probe.service.can_start_recording
    assert probe.service.is_path_leased(request.path)
    assert probe.service.is_path_leased(second.request.path)


def test_trusted_terminal_survives_newer_active_fatal_and_unconfirmed_death_keeps_capacity(
        monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    release_probe_slot(probe)
    request = probe.session.request
    trusted = RecordingResult(
        request.request_id,
        request.purpose,
        request.path,
        request.sample_rate,
        request.channels,
        request.target_samples,
        request.target_samples - request.trim_samples,
        False,
    )
    probe.service._event(probe.worker, RecordingEvent(
        1, request.request_id, "completed", trusted))
    assert probe.session._trusted_terminal and probe.session.state == "delivering"

    from base.recording_service import RecordingCallbacks
    second_failures = queue.Queue()
    second = probe.service.start(
        capture_request(tmp_path / "B.wav", request_id="B"),
        RecordingCallbacks(failed=lambda session, failure: second_failures.put(failure)),
    )
    probe.service._dispatch(probe.service._inbox.get_nowait())
    probe.worker.outgoing.get_nowait()
    probe.service._event(probe.worker, RecordingEvent(
        1, second.request.request_id, "started", probe.clock()))

    kills = []
    probe.worker.process.kill = lambda: kills.append("kill")
    probe.service._event(probe.worker, RecordingEvent(
        1, "", "worker_fatal", WorkerFatal(1, "native/read", "ACTIVE-SDK-FATAL")))

    assert probe.session.descriptor is trusted
    assert probe.session.failure is None and probe.session.state == "delivering"
    active_failure = second_failures.get_nowait()
    assert (active_failure.stage, active_failure.message) == (
        "native/read", "ACTIVE-SDK-FATAL")
    assert not probe.service.can_start_recording
    probe.clock.advance(2)
    probe.service._tick()
    assert kills == ["kill"]
    assert probe.service._worker is probe.worker
    assert probe.service._ownership_uncertain
    assert not probe.service.can_start_recording
    assert probe.service.is_path_leased(request.path)
    assert probe.service.is_path_leased(second.request.path)


def test_duplicate_target_progress_cannot_replace_t0_or_extend_slot_deadline(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    assert probe.session._target_reached_at == 100.1
    assert probe.session._slot_release_deadline == 100.6
    probe.clock.advance(.39)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.49)))
    assert probe.session._target_reached_at == 100.1
    assert probe.session._slot_release_deadline == 100.6
    probe.clock.advance(.12)
    probe.service._tick()
    assert probe.worker.retiring
    assert probe.session.failure.stage == "capture_release_timeout"


@pytest.mark.parametrize("field", [
    "sdk_open", "task_create", "task_start", "task_stop", "task_clear", "sdk_close",
])
def test_first_slot_rejects_every_unauthorized_lifecycle_field(monkeypatch, tmp_path, field):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    baseline = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    invalid = replace(baseline, **{field: getattr(baseline, field) + 1})
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released",
        slot_payload(probe, lifecycle_counts=invalid)))
    assert probe.worker.retiring and probe.service._capture_session is probe.session


@pytest.mark.parametrize("field", [
    "sdk_open", "task_create", "task_start", "task_stop", "task_clear", "sdk_close",
])
def test_compatible_slot_requires_every_lifecycle_field_to_match_previous_snapshot(
        monkeypatch, tmp_path, field):
    probe = ve_probe(monkeypatch, tmp_path)
    baseline = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    probe.service._retained_lifecycle_counts = baseline
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    invalid = replace(baseline, **{field: getattr(baseline, field) + 1})
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released",
        slot_payload(probe, lifecycle_counts=invalid)))
    assert probe.worker.retiring and probe.service._capture_session is probe.session


@pytest.mark.parametrize("field", [
    "sdk_open", "task_create", "task_start", "task_stop", "task_clear", "sdk_close",
])
@pytest.mark.parametrize("delta", [-1, 1])
def test_reinitialized_slot_rejects_lifecycle_regressions_and_jumps(
        monkeypatch, tmp_path, field, delta):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.service._retained_lifecycle_counts = VeLifecycleCounts(1, 1, 1, 1, 1, 1)
    probe.service._retained_ve_signature = None
    probe.service._expected_next_lifecycle_counts = VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))
    expected = VeLifecycleCounts(2, 2, 2, 1, 1, 1)
    invalid = replace(expected, **{field: getattr(expected, field) + delta})
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released",
        slot_payload(probe, lifecycle_counts=invalid)))
    assert probe.worker.retiring and probe.service._capture_session is probe.session


@pytest.mark.parametrize("field,value", [
    ("version", True), ("generation", True), ("generation", 1.0), ("request_id", True),
])
@pytest.mark.parametrize("kind", [
    "capture_slot_released", "ve_released", "ve_release_failed", "worker_fatal",
])
def test_malformed_lifecycle_envelopes_from_current_connection_retire(
        monkeypatch, tmp_path, kind, field, value):
    probe = ve_probe(monkeypatch, tmp_path)
    if kind == "capture_slot_released":
        payload = slot_payload(probe)
        request_id = probe.session.request.request_id
    elif kind in ("ve_released", "ve_release_failed"):
        payload = VeReleaseOutcome(1, None)
        request_id = ""
    else:
        payload = WorkerFatal(1, "native/read", "failed")
        request_id = ""
    event = RecordingEvent(1, request_id, kind, payload)
    object.__setattr__(event, field, value)
    probe.service._event(probe.worker, event)
    assert probe.worker.retiring and not probe.service.can_start_recording


@pytest.mark.parametrize("kind", [
    "capture_slot_released", "ve_released", "ve_release_failed", "worker_fatal",
])
def test_constructor_bypassed_malformed_lifecycle_payloads_retire(monkeypatch, tmp_path, kind):
    probe = ve_probe(monkeypatch, tmp_path)
    if kind == "capture_slot_released":
        payload = slot_payload(probe)
        object.__setattr__(payload, "raw_frames", True)
        request_id = probe.session.request.request_id
    elif kind in ("ve_released", "ve_release_failed"):
        payload = VeReleaseOutcome(1, None)
        object.__setattr__(payload, "diagnostics", ["not immutable"])
        request_id = ""
    else:
        payload = WorkerFatal(1, "native/read", "failed")
        object.__setattr__(payload, "stage", "")
        request_id = ""
    event = RecordingEvent.__new__(RecordingEvent)
    object.__setattr__(event, "generation", 1)
    object.__setattr__(event, "request_id", request_id)
    object.__setattr__(event, "kind", kind)
    object.__setattr__(event, "payload", payload)
    object.__setattr__(event, "version", 1)
    probe.service._event(probe.worker, event)
    assert probe.worker.retiring and not probe.service.can_start_recording


@pytest.mark.parametrize("corrupted_kind", [True, "progress"])
@pytest.mark.parametrize("payload_type", ["slot", "release", "fatal"])
def test_lifecycle_payload_type_retires_even_when_event_kind_is_corrupted(
        monkeypatch, tmp_path, payload_type, corrupted_kind):
    probe = ve_probe(monkeypatch, tmp_path)
    if payload_type == "slot":
        payload = slot_payload(probe)
        request_id = probe.session.request.request_id
    elif payload_type == "release":
        payload = VeReleaseOutcome(1, None)
        request_id = ""
    else:
        payload = WorkerFatal(1, "native/read", "failed")
        request_id = ""
    event = RecordingEvent.__new__(RecordingEvent)
    object.__setattr__(event, "generation", 1)
    object.__setattr__(event, "request_id", request_id)
    object.__setattr__(event, "kind", corrupted_kind)
    object.__setattr__(event, "payload", payload)
    object.__setattr__(event, "version", 1)
    probe.service._event(probe.worker, event)
    assert probe.worker.retiring and not probe.service.can_start_recording


def test_authoritative_t1_after_atomic_slot_change_cannot_open_late_admission(monkeypatch, tmp_path):
    from base import recording_service as module

    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1,
                          probe.session.request.target_samples, 100.1)))

    def transition_clock():
        return 100.61 if probe.service._capture_session is None else 100.59

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=transition_clock))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.session.request.request_id, "capture_slot_released", slot_payload(probe)))
    assert probe.worker.retiring
    assert probe.service._capture_session is probe.session
    assert probe.session._slot_released_at is None
    assert probe.session.failure.stage == "capture_release_timeout"
    assert not probe.service.can_start_recording


def test_delayed_old_generation_preview_ack_is_cleared_without_reaching_new_worker(
        service, tmp_path):
    old = service.start(request(tmp_path, request_id="old", path=str(tmp_path / "old.wav")))
    old.generation = 1
    old.state = "delivering"
    old._trusted_terminal = True
    old._preview_pending = 7
    old._preview_ack_requested = 7
    service._capture_session = None
    process = SimpleNamespace(pid=2, is_alive=lambda: True,
                              terminate=lambda: None, kill=lambda: None)
    replacement = _Worker(2, process, None, None, None)
    replacement.ready = True
    service._worker = replacement

    service._dispatch(("preview_ack", old, 7))

    assert old._preview_pending is None and old._preview_ack_requested is None
    assert replacement.outgoing.empty()
    assert not replacement.retiring
