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
    VE_PREWARM_DETACHING, VE_PREWARM_PROGRESS,
    VeLifecycleCounts, VePrewarmRequest, VePrewarmResult, VeReleaseOutcome,
    VePrewarmProgress, VePrewarmStarted,
    WorkerFatal,
)
from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
from base.ve3668n_capture_timing import VeCaptureProgress
from base.recording_service import RecordingCallbacks, RecordingService, _Worker
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_LOWER_BOUND_TOLERANCE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from unit_test.base.test_recording_service import Events, eventually
from unit_test.base.ve3668n_fakes import DiscoveryClock, capture_request


def _prewarm_request(tmp_path, warmup_id="warmup-1", *, attempt=1):
    recording = capture_request(tmp_path / "must-not-exist.wav")
    return VePrewarmRequest.create(
        warmup_id, recording.device, recording.channels, recording.sample_rate,
        attempt=attempt,
    )


def test_prewarm_service_admission_is_atomic_and_uses_no_recording_pipeline(
        monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService()
    request = _prewarm_request(tmp_path)
    completions = []

    assert service.prewarm_ve(request, completions.append) == "accepted"
    assert not service.can_start_recording
    assert service._capture_session is None
    assert service._sessions == {}
    assert not service.is_path_leased(tmp_path / "must-not-exist.wav")
    assert service.release_ve(None) == "busy"
    assert completions == []


@pytest.mark.parametrize("state, expected", [("busy", "busy"), ("closing", "closing")])
def test_prewarm_service_admission_rejects_without_mutating_state(
        monkeypatch, tmp_path, state, expected):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService()
    if state == "busy":
        service._ownership_uncertain = True
    else:
        service._closing = True
    completions = []

    assert service.prewarm_ve(_prewarm_request(tmp_path), completions.append) == expected
    assert getattr(service, "_pending_ve_prewarm", None) is None
    assert completions == []


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


def _service_prewarm_probe(monkeypatch, tmp_path, *, ready=True, retained=None,
                           **service_options):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    clock = DiscoveryClock()
    service = RecordingService(monotonic=clock, **service_options)
    process = _ServiceProcess()
    worker = _Worker(1, process, SimpleNamespace(close=lambda: None),
                     SimpleNamespace(close=lambda: None),
                     clock() + service._ready_timeout)
    worker.ready = ready
    if ready:
        worker.deadline = None
    service._worker = worker
    service._generation = 1
    service._retained_ve_signature = retained
    completions = []
    request = _prewarm_request(tmp_path)
    assert service.prewarm_ve(request, completions.append) == "accepted"
    service._dispatch(service._inbox.get_nowait())
    command = None if worker.outgoing.empty() else worker.outgoing.get_nowait()
    return SimpleNamespace(service=service, worker=worker, process=process,
                           clock=clock, request=request,
                           completions=completions, command=command)


def _prewarm_result(probe, *, success, attempt=1, stage=None, code=None,
                    detail=None, frames=None, released=True,
                    diagnostics=(), generation=None):
    request = replace(probe.request, attempt=attempt)
    return VePrewarmResult(
        request.warmup_id, generation or probe.worker.generation, attempt,
        request.signature, success, "completed" if success else (stage or "read"),
        None if success else code, "" if success else (detail or "native failure"),
        request.frames_per_channel if frames is None and success else (frames or 0),
        released, diagnostics,
        VeLifecycleCounts(1, 1, 1, 0, 0, 0),
    )


def _prewarm_started(probe, *, started_at=None):
    pending = probe.service._pending_ve_prewarm
    started_at = probe.clock() if started_at is None else started_at
    payload = VePrewarmStarted(
        probe.request.warmup_id, probe.worker.generation, pending.attempt,
        probe.request.signature, started_at)
    probe.service._event(probe.worker, RecordingEvent(
        probe.worker.generation, probe.request.warmup_id,
        "ve_prewarm_started", payload))


def _prewarm_detaching(probe):
    pending = probe.service._pending_ve_prewarm
    started_at = pending.capture_deadline.snapshot().started_at
    payload = VePrewarmProgress(
        probe.request.warmup_id, probe.worker.generation, pending.attempt,
        probe.request.signature, started_at,
        probe.request.frames_per_channel, probe.clock())
    probe.service._event(probe.worker, RecordingEvent(
        probe.worker.generation, probe.request.warmup_id,
        VE_PREWARM_PROGRESS, payload))
    probe.service._event(probe.worker, RecordingEvent(
        probe.worker.generation, probe.request.warmup_id,
        VE_PREWARM_DETACHING, payload))


def _prewarm_progress(probe, frames, *, attempt=None, at=None, generation=None):
    pending = probe.service._pending_ve_prewarm
    attempt = pending.attempt if attempt is None else attempt
    at = probe.clock() if at is None else at
    return VePrewarmProgress(
        probe.request.warmup_id, generation or probe.worker.generation,
        attempt, probe.request.signature, probe.clock(), frames, at)


def test_prewarm_service_success_validates_identity_and_completes_once(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    assert probe.command.kind == "prewarm_ve"
    _prewarm_started(probe)
    _prewarm_detaching(probe)
    result = _prewarm_result(probe, success=True)

    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", result))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", result))

    assert len(probe.completions) == 1
    completion = probe.completions[0]
    assert completion.success and completion.ownership_safe
    assert completion.signature == probe.request.signature
    assert probe.service.retained_ve_signature == probe.request.signature
    assert probe.service._capture_session is None and probe.service._sessions == {}
    assert probe.service.can_start_recording


def test_prewarm_first_native_failure_waits_for_death_and_exact_retry_deadline(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    first = _prewarm_result(
        probe, success=False, stage="start_task", code=-12001,
        detail="iio_device_create_multi_buffer: invalid argument",
        released=False, diagnostics=("owner exited before binding",))
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", first))
    probe.service._event(probe.worker, RecordingEvent(
        1, "", "worker_fatal",
        WorkerFatal(1, "bind", "VE resource owner exited before binding")))
    assert probe.process.terminates == 1 and probe.completions == []
    assert probe.service._pending_ve_prewarm.first_fault == first

    probe.process.alive = False
    probe.service._tick()
    pending = probe.service._pending_ve_prewarm
    assert pending.phase == "retry_wait" and pending.retry_deadline == 100.75

    spawned = []

    def spawn():
        process = _ServiceProcess(pid=456)
        worker = _Worker(2, process, SimpleNamespace(close=lambda: None),
                         SimpleNamespace(close=lambda: None), None)
        worker.ready = True
        probe.service._worker = worker
        probe.service._generation = 2
        spawned.append(worker)

    monkeypatch.setattr(probe.service, "_spawn", spawn)
    probe.clock.advance(.749)
    probe.service._tick()
    assert spawned == []
    probe.clock.advance(.001)
    probe.service._tick()
    assert len(spawned) == 1
    second_worker = spawned[0]
    command = second_worker.outgoing.get_nowait()
    assert command.payload.attempt == 2 and command.payload.signature == first.signature

    probe.worker = second_worker
    _prewarm_started(probe)
    _prewarm_detaching(probe)
    second = _prewarm_result(probe, success=True, attempt=2, generation=2)
    probe.service._event(second_worker, RecordingEvent(
        2, probe.request.warmup_id, "ve_prewarm_terminal", second))
    assert len(probe.completions) == 1 and probe.completions[0].success
    assert probe.completions[0].attempts == (first, second)


def test_prewarm_two_failures_preserve_first_cause_and_never_create_third_attempt(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    pending = probe.service._pending_ve_prewarm
    first = _prewarm_result(
        probe, success=False, stage="start_task", code=-12001,
        detail="first native failure", released=False,
        diagnostics=("secondary bind diagnostic",))
    pending.first_fault = first
    pending.results = (first,)
    pending.attempt = 2
    pending.current_request = replace(probe.request, attempt=2)
    _prewarm_started(probe)
    second = _prewarm_result(
        probe, success=False, attempt=2, stage="read", code=-7,
        detail="second native failure", released=False)

    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", second))
    assert probe.process.terminates == 1 and probe.completions == []
    probe.process.alive = False
    probe.service._tick()

    assert len(probe.completions) == 1
    completion = probe.completions[0]
    assert not completion.success and completion.ownership_safe
    assert (completion.stage, completion.code, completion.detail) == (
        "start_task", -12001, "first native failure")
    assert completion.attempts == (first, second)
    assert probe.service._pending_ve_prewarm is None
    probe.clock.advance(100)
    probe.service._tick()
    assert probe.service.generation == 1


def test_prewarm_deterministic_request_validation_finishes_without_native_retry(
        monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService(monotonic=DiscoveryClock())
    valid = _prewarm_request(tmp_path)
    invalid = object.__new__(VePrewarmRequest)
    for name in ("warmup_id", "device", "channels", "sample_rate", "frames_per_channel"):
        object.__setattr__(invalid, name, getattr(valid, name))
    object.__setattr__(invalid, "attempt", 3)
    completions = []

    assert service.prewarm_ve(invalid, completions.append) == "accepted"
    service._dispatch(service._inbox.get_nowait())

    assert len(completions) == 1
    assert not completions[0].success and completions[0].stage == "validation"
    assert completions[0].ownership_safe and completions[0].attempts == ()
    assert service._worker is None and service.generation == 0


@pytest.mark.parametrize("stage", ["capture_timeout", "detach"])
def test_prewarm_read_and_detach_failures_retire_before_retry(
        monkeypatch, tmp_path, stage):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    result = _prewarm_result(
        probe, success=False, stage=stage, detail=f"{stage} failed",
        released=stage != "detach")
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", result))

    assert probe.process.terminates == 1
    assert probe.service._pending_ve_prewarm.phase == "retiring_retry"
    assert probe.completions == []


def test_prewarm_terminate_boundary_kills_and_reports_ownership_uncertain_once(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path, terminate_timeout=2)
    _prewarm_started(probe)
    failure = _prewarm_result(
        probe, success=False, stage="start_task", code=-12001,
        detail="native failed", released=False)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", failure))

    probe.clock.advance(1.999)
    probe.service._tick()
    assert probe.process.kills == 0 and probe.completions == []
    probe.clock.advance(.001)
    probe.service._tick()
    assert probe.process.kills == 1 and len(probe.completions) == 1
    assert not probe.completions[0].ownership_safe
    assert not probe.service.can_start_recording

    probe.process.alive = False
    probe.service._tick()
    assert len(probe.completions) == 1 and probe.service.can_start_recording


def test_prewarm_incompatible_retained_signature_releases_before_dispatch(
        monkeypatch, tmp_path):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(monkeypatch, tmp_path, retained=old)
    assert probe.command.kind == "release_ve"
    assert probe.service._pending_ve_prewarm.phase == "releasing"

    probe.service._event(probe.worker, RecordingEvent(
        1, "", "ve_released", VeReleaseOutcome(1, old)))

    command = probe.worker.outgoing.get_nowait()
    assert command.kind == "prewarm_ve" and command.payload.attempt == 1
    assert probe.service._pending_ve_release is None


def test_prewarm_release_admission_busy_creates_no_prewarms(monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService(monotonic=DiscoveryClock())
    service._pending_ve_release = object()
    completions = []
    assert service.prewarm_ve(_prewarm_request(tmp_path), completions.append) == "busy"
    assert service._pending_ve_prewarm is None and completions == []


@pytest.mark.parametrize("death_confirmed", [True, False])
def test_prewarm_prerequisite_release_failure_never_dispatches_capture(
        monkeypatch, tmp_path, death_confirmed):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(
        monkeypatch, tmp_path, retained=old, terminate_timeout=2)
    outcome = VeReleaseOutcome(1, old, ("clear task failed",))
    probe.service._event(probe.worker, RecordingEvent(
        1, "", "ve_release_failed", outcome))
    assert probe.worker.outgoing.empty()
    if death_confirmed:
        probe.process.alive = False
        probe.service._tick()
    else:
        probe.clock.advance(2)
        probe.service._tick()

    assert len(probe.completions) == 1
    completion = probe.completions[0]
    assert not completion.success and completion.stage == "release_ve"
    assert completion.signature == probe.request.signature
    assert completion.ownership_safe is death_confirmed
    assert not any(result.success for result in completion.attempts)


def test_prewarm_release_timeout_fires_at_five_seconds_and_is_not_busy_discard(
        monkeypatch, tmp_path):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(
        monkeypatch, tmp_path, retained=old, release_timeout=5)
    probe.clock.advance(4.999)
    probe.service._tick()
    assert not probe.worker.retiring and probe.completions == []
    probe.clock.advance(.001)
    probe.service._tick()
    assert probe.worker.retiring and probe.completions == []
    probe.process.alive = False
    probe.service._tick()
    assert probe.completions[0].stage == "release_ve"
    assert probe.completions[0].ownership_safe


def test_prewarm_release_timeout_without_confirmed_death_never_dispatches_attempt(
        monkeypatch, tmp_path):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(
        monkeypatch, tmp_path, retained=old,
        release_timeout=5, terminate_timeout=2)
    probe.clock.advance(5)
    probe.service._tick()
    assert probe.service._pending_ve_prewarm.phase == "retiring_release"
    assert probe.worker.outgoing.empty()
    probe.clock.advance(2)
    probe.service._tick()
    assert len(probe.completions) == 1
    assert probe.completions[0].stage == "release_ve"
    assert not probe.completions[0].ownership_safe
    assert probe.service.generation == 1


def test_prewarm_shutdown_deadline_cancels_silently_without_retry_or_resurrection(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path, shutdown_timeout=5)
    probe.service.shutdown()
    probe.service._dispatch(probe.service._inbox.get_nowait())
    assert probe.service._pending_ve_prewarm is None
    assert probe.completions == []
    commands = [probe.worker.outgoing.get_nowait().kind
                for _ in range(probe.worker.outgoing.qsize())]
    assert commands == ["cancel", "shutdown"]

    probe.clock.advance(4.999)
    probe.service._tick()
    assert not probe.worker.retiring
    probe.clock.advance(.001)
    probe.service._tick()
    assert probe.worker.retiring and probe.completions == []
    terminal = _prewarm_result(
        probe, success=False, stage="cancelled", detail="shutdown")
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", terminal))
    assert probe.completions == []


def test_prewarm_ready_timeout_fires_at_ten_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    clock = DiscoveryClock()
    service = RecordingService(monotonic=clock, ready_timeout=10)
    process = _ServiceProcess()

    def spawn():
        worker = _Worker(
            1, process, SimpleNamespace(close=lambda: None),
            SimpleNamespace(close=lambda: None), clock() + 10)
        service._worker = worker
        service._generation = 1

    monkeypatch.setattr(service, "_spawn", spawn)
    completions = []
    assert service.prewarm_ve(_prewarm_request(tmp_path), completions.append) == "accepted"
    service._dispatch(service._inbox.get_nowait())
    assert service._pending_ve_prewarm.phase == "waiting_ready"

    clock.advance(9.999)
    service._tick()
    assert not service._worker.retiring
    clock.advance(.001)
    service._tick()
    assert service._worker.retiring
    assert service._pending_ve_prewarm.first_fault.stage == "ready_timeout"


def test_prewarm_start_timeout_fires_at_ten_seconds_but_bind_at_three_owns_cause(
        monkeypatch, tmp_path):
    timeout = _service_prewarm_probe(monkeypatch, tmp_path, start_timeout=10)
    timeout.clock.advance(9.999)
    timeout.service._tick()
    assert not timeout.worker.retiring
    timeout.clock.advance(.001)
    timeout.service._tick()
    assert timeout.worker.retiring
    assert timeout.service._pending_ve_prewarm.first_fault.stage == "start_timeout"

    bind = _service_prewarm_probe(monkeypatch, tmp_path / "bind", start_timeout=10)
    bind.clock.advance(3)
    result = _prewarm_result(
        bind, success=False, stage="bind", detail="controller bind deadline exceeded",
        released=False)
    bind.service._event(bind.worker, RecordingEvent(
        1, bind.request.warmup_id, "ve_prewarm_terminal", result))
    bind.clock.advance(7)
    bind.service._tick()
    assert bind.completions[0].stage == "bind"
    assert bind.completions[0].detail == (
        "controller bind deadline exceeded")


def test_prewarm_terminal_attempt_or_signature_mismatch_retires_as_protocol_fault(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    mismatched = _prewarm_result(
        probe, success=False, attempt=2, stage="read", detail="wrong attempt")
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", mismatched))

    assert probe.worker.retiring
    assert probe.service._pending_ve_prewarm.first_fault.stage == "protocol"
    assert probe.service._pending_ve_prewarm.phase == "retiring_final"


@pytest.mark.parametrize("failure", ["broken", "dead"])
def test_prewarm_prerequisite_release_owner_loss_never_becomes_capture_retry(
        monkeypatch, tmp_path, failure):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(monkeypatch, tmp_path, retained=old)
    if failure == "broken":
        probe.service._dispatch(("broken", probe.worker, "release IPC closed"))
    else:
        probe.process.alive = False
        probe.service._tick()

    pending = probe.service._pending_ve_prewarm
    if pending is not None:
        assert pending.phase == "retiring_release"
        assert pending.first_fault.stage == "release_ve"
    if failure == "broken":
        probe.process.alive = False
        probe.service._tick()
    assert len(probe.completions) == 1
    assert probe.completions[0].stage == "release_ve"
    assert probe.completions[0].failed_signature == probe.request.signature
    assert probe.completions[0].ownership_safe
    assert not any(item.success for item in probe.completions[0].attempts)


def test_prewarm_prerequisite_release_owner_loss_without_death_is_unsafe_terminal(
        monkeypatch, tmp_path):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(
        monkeypatch, tmp_path, retained=old, terminate_timeout=2)
    probe.service._dispatch(("broken", probe.worker, "release IPC closed"))
    probe.clock.advance(2)
    probe.service._tick()

    assert len(probe.completions) == 1
    assert probe.completions[0].stage == "release_ve"
    assert not probe.completions[0].ownership_safe
    assert not probe.service.can_start_recording


@pytest.mark.parametrize("failure", ["malformed", "future", "service"])
def test_prewarm_prerequisite_release_all_retirement_paths_are_release_failures(
        monkeypatch, tmp_path, failure):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(monkeypatch, tmp_path, retained=old)
    if failure == "malformed":
        event = object.__new__(RecordingEvent)
        object.__setattr__(event, "generation", 1)
        object.__setattr__(event, "request_id", "")
        object.__setattr__(event, "kind", "ve_released")
        object.__setattr__(event, "payload", "not-an-outcome")
        object.__setattr__(event, "version", 1)
        probe.service._event(probe.worker, event)
    elif failure == "future":
        probe.service._event(probe.worker, RecordingEvent(
            2, "", "ve_release_failed",
            VeReleaseOutcome(2, old, ("future release failure",))))
    else:
        probe.service._handle_supervisor_exception(
            RuntimeError("injected service failure during release"))

    pending = probe.service._pending_ve_prewarm
    assert pending.phase == "retiring_release"
    assert pending.first_fault.stage == "release_ve"
    assert not any(event.kind == "prewarm_ve"
                   for event in tuple(probe.worker.outgoing.queue))
    probe.process.alive = False
    probe.service._tick()
    assert len(probe.completions) == 1
    assert probe.completions[0].stage == "release_ve"


def test_prewarm_prerequisite_release_send_exception_cannot_hang_or_retry(
        monkeypatch, tmp_path):
    old = ("vkinging", "old-machine", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    probe = _service_prewarm_probe(monkeypatch, tmp_path, retained=old)
    pending_release = probe.service._pending_ve_release
    pending_release.sent = False
    monkeypatch.setattr(
        probe.service, "_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected release send failure")))
    probe.service._send_ve_release(pending_release)
    pending = probe.service._pending_ve_prewarm
    assert pending.phase == "retiring_release"
    assert pending.first_fault.stage == "release_ve"
    probe.clock.advance(2)
    probe.service._tick()
    assert len(probe.completions) == 1
    assert not probe.completions[0].ownership_safe


def test_prewarm_retry_delay_is_validated_injected_and_exact(monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    for invalid in (0, -1, math.inf, math.nan, "0.75", True):
        with pytest.raises(ValueError, match="retry"):
            RecordingService(retry_delay=invalid)

    probe = _service_prewarm_probe(monkeypatch, tmp_path, retry_delay=.125)
    _prewarm_started(probe)
    failure = _prewarm_result(
        probe, success=False, stage="read", detail="retry me", released=False)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", failure))
    probe.process.alive = False
    probe.service._tick()
    assert probe.service._pending_ve_prewarm.retry_deadline == 100.125


def test_prewarm_partial_spawn_success_thread_failure_retires_registered_generation(
        monkeypatch, tmp_path):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    clock = DiscoveryClock()
    service = RecordingService(monotonic=clock)
    process = _ServiceProcess()

    def partial_spawn():
        worker = _Worker(
            1, process, SimpleNamespace(close=lambda: None),
            SimpleNamespace(close=lambda: None), clock() + 10)
        service._worker = worker
        service._generation = 1
        raise RuntimeError("sender thread failed after process start")

    monkeypatch.setattr(service, "_spawn", partial_spawn)
    completions = []
    assert service.prewarm_ve(_prewarm_request(tmp_path), completions.append) == "accepted"
    service._dispatch(service._inbox.get_nowait())

    pending = service._pending_ve_prewarm
    assert pending.generation == 1 and pending.phase == "retiring_retry"
    assert pending.first_fault.stage == "service"
    assert process.terminates == 1 and completions == []
    process.alive = False
    service._tick()
    monkeypatch.setattr(
        service, "_spawn",
        lambda: (_ for _ in ()).throw(RuntimeError("second spawn failed")))
    clock.advance(.75)
    service._tick()
    assert len(completions) == 1
    assert not completions[0].success and completions[0].ownership_safe
    assert service._pending_ve_prewarm is None and service.can_start_recording


def test_prewarm_real_spawn_post_start_thread_failure_is_bounded(
        monkeypatch, tmp_path):
    real_start_thread = RecordingService._start_thread
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    service = RecordingService(retry_delay=.01)
    completions = []
    injected = False

    def start_thread(thread, *, start=None, worker=None):
        nonlocal injected
        if worker is not None and not injected:
            injected = True
            try:
                (start or thread.start)()
            finally:
                service.threads.append(thread)
                worker.threads.append(thread)
            raise RuntimeError("injected parent IPC thread startup failure")
        return real_start_thread(service, thread, start=start, worker=worker)

    service._start_thread = start_thread
    assert service.prewarm_ve(
        _prewarm_request(tmp_path, "real-partial-spawn"),
        completions.append) == "accepted"
    service._dispatch(service._inbox.get_nowait())
    worker = service._worker
    assert worker is not None and worker.process.pid is not None
    assert worker.retiring and worker.process.is_alive()
    eventually(lambda: not worker.process.is_alive(), timeout=5)
    service._tick()
    monkeypatch.setattr(
        service, "_spawn",
        lambda: (_ for _ in ()).throw(RuntimeError("retry spawn failed")))
    eventually(lambda: (service._tick() or bool(completions)), timeout=2)
    assert len(completions) == 1 and completions[0].ownership_safe
    assert service._pending_ve_prewarm is None and service.can_start_recording


def test_prewarm_success_terminal_before_started_is_protocol_failure(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    success = _prewarm_result(probe, success=True)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, "ve_prewarm_terminal", success))
    assert probe.worker.retiring
    assert probe.service._pending_ve_prewarm.first_fault.stage == "protocol"
    assert probe.service.retained_ve_signature is None


def test_prewarm_authenticated_progress_drives_independent_deadlines(
        monkeypatch, tmp_path):
    stalled = _service_prewarm_probe(monkeypatch, tmp_path / "stalled")
    _prewarm_started(stalled)
    started_at = stalled.clock()
    stalled.clock.advance(.1)
    progress = VePrewarmProgress(
        stalled.request.warmup_id, 1, 1, stalled.request.signature,
        started_at, 1, stalled.clock())
    stalled.service._event(stalled.worker, RecordingEvent(
        1, stalled.request.warmup_id, VE_PREWARM_PROGRESS, progress))
    stalled.clock.advance(4.999)
    stalled.service._tick()
    assert not stalled.worker.retiring
    stalled.clock.advance(.001)
    stalled.service._tick()
    assert "no progress" in stalled.service._pending_ve_prewarm.first_fault.detail

    total = _service_prewarm_probe(monkeypatch, tmp_path / "total")
    _prewarm_started(total)
    started_at = total.clock()
    total.clock.advance(1)
    progress = VePrewarmProgress(
        total.request.warmup_id, 1, 1, total.request.signature,
        started_at, 1, total.clock())
    total.service._event(total.worker, RecordingEvent(
        1, total.request.warmup_id, VE_PREWARM_PROGRESS, progress))
    total.clock.advance(4.499)
    total.service._tick()
    assert not total.worker.retiring
    total.clock.advance(.001)
    total.service._tick()
    assert "total deadline" in total.service._pending_ve_prewarm.first_fault.detail


def test_prewarm_zero_progress_deadline_originates_at_authenticated_native_start(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    probe.clock.advance(.5)
    _prewarm_started(probe, started_at=100.0)
    assert probe.service._pending_ve_prewarm.capture_deadline.snapshot().started_at == 100.0
    probe.clock.advance(4.499)
    probe.service._tick()
    assert not probe.worker.retiring
    probe.clock.advance(.001)
    probe.service._tick()
    assert probe.worker.retiring
    assert "no progress" in probe.service._pending_ve_prewarm.first_fault.detail


def test_prewarm_progress_regressing_observation_time_is_local_protocol_fault(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    probe.clock.advance(1)
    first = VePrewarmProgress(
        probe.request.warmup_id, 1, 1, probe.request.signature,
        100.0, 1, 101.0)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, VE_PREWARM_PROGRESS, first))
    regressed = replace(first, frames_per_channel=2, observed_at=100.5)
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, VE_PREWARM_PROGRESS, regressed))
    pending = probe.service._pending_ve_prewarm
    assert pending.first_fault.stage == "protocol"
    assert pending.phase == "retiring_final"
    assert pending.attempt == 1 and pending.retry_deadline is None


def test_prewarm_authenticated_detaching_event_arms_half_second_deadline(
        monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    _prewarm_detaching(probe)
    pending = probe.service._pending_ve_prewarm
    assert pending.phase == "detaching"
    assert pending.detach_deadline == 100.5
    probe.clock.advance(.499)
    probe.service._tick()
    assert not probe.worker.retiring
    probe.clock.advance(.001)
    probe.service._tick()
    assert pending.first_fault.stage == "detach"


def test_prewarm_progress_attempt_identity_is_authenticated(monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    payload = VePrewarmProgress(
        probe.request.warmup_id, 1, 2, probe.request.signature,
        probe.clock(), 1, probe.clock())
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, VE_PREWARM_PROGRESS, payload))
    assert probe.worker.retiring
    assert probe.service._pending_ve_prewarm.first_fault.stage == "protocol"


def test_prewarm_detaching_requires_ordered_target_progress(monkeypatch, tmp_path):
    probe = _service_prewarm_probe(monkeypatch, tmp_path)
    _prewarm_started(probe)
    payload = VePrewarmProgress(
        probe.request.warmup_id, 1, 1, probe.request.signature,
        probe.clock(), probe.request.frames_per_channel, probe.clock())
    probe.service._event(probe.worker, RecordingEvent(
        1, probe.request.warmup_id, VE_PREWARM_DETACHING, payload))
    assert probe.worker.retiring
    assert probe.service._pending_ve_prewarm.first_fault.stage == "protocol"


def test_prewarm_service_real_worker_retries_once_in_fresh_generation(tmp_path):
    trace_path = tmp_path / "service-prewarm-native.jsonl"
    service = RecordingService(
        backend_factory="unit_test.base.ve3668n_fakes:capture_dependencies",
        backend_options={
            "trace_path": str(trace_path),
            "read_delay": 0,
            "fail_operation": "start_task",
            "first_only": True,
        })
    completions = queue.Queue()
    request = _prewarm_request(tmp_path, "service-retry")
    try:
        assert service.prewarm_ve(request, completions.put) == "accepted"
        completion = completions.get(timeout=20)
        assert completion.success and completion.ownership_safe
        assert len(completion.attempts) == 2
        assert not completion.attempts[0].success
        assert completion.attempts[1].success
        assert completion.attempts[0].generation != completion.attempts[1].generation
        assert service.retained_ve_signature == request.signature
        assert service.can_start_recording
        assert not (tmp_path / "must-not-exist.wav").exists()
    finally:
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics


def _spawn_recording_worker(tmp_path, *, generation=1, options=None):
    from base.recording_worker import recording_worker

    tmp_path.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    parent_control, child_control = context.Pipe()
    parent_preview, child_preview = context.Pipe()
    trace_path = tmp_path / "prewarm-native.jsonl"
    backend_options = dict(trace_path=str(trace_path), read_delay=0)
    backend_options.update(options or {})
    process = context.Process(
        target=recording_worker,
        args=(child_control, child_preview, generation,
              "unit_test.base.ve3668n_fakes:capture_dependencies",
              backend_options, .25, .01),
    )
    process.start()
    child_control.close()
    child_preview.close()
    assert parent_control.poll(10), "worker did not publish ready"
    ready = parent_control.recv()
    assert ready.kind == "ready" and ready.generation == generation
    return SimpleNamespace(
        generation=generation, process=process, control=parent_control,
        preview=parent_preview, trace_path=trace_path,
    )


def _blocked_prewarm_started_worker(
        control, preview, generation, backend_factory, backend_options,
        cancel_timeout, preview_interval):
    """Hold the sender on prewarm-started while terminal/fatal are produced."""
    from base.recording_worker import recording_worker

    class Connection:
        def __getattr__(self, name):
            return getattr(control, name)

        def send(self, event):
            if event.kind == "ve_prewarm_started":
                marker = Path(backend_options["trace_path"] + ".started-blocked")
                release = Path(backend_options["trace_path"] + ".ipc-release")
                marker.touch()
                deadline = time.monotonic() + 10
                while not release.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test did not release prewarm sender")
                    threading.Event().wait(.005)
            control.send(event)

    recording_worker(Connection(), preview, generation, backend_factory,
                     backend_options, cancel_timeout, preview_interval)


def _spawn_blocked_prewarm_worker(tmp_path, *, generation=1, options=None):
    context = mp.get_context("spawn")
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent_control, child_control = context.Pipe()
    parent_preview, child_preview = context.Pipe()
    trace_path = tmp_path / "prewarm-native.jsonl"
    backend_options = dict(trace_path=str(trace_path), read_delay=0)
    backend_options.update(options or {})
    process = context.Process(
        target=_blocked_prewarm_started_worker,
        args=(child_control, child_preview, generation,
              "unit_test.base.ve3668n_fakes:capture_dependencies",
              backend_options, 2.0, .01),
    )
    process.start()
    child_control.close()
    child_preview.close()
    assert parent_control.poll(10)
    assert parent_control.recv().kind == "ready"
    return SimpleNamespace(
        generation=generation, process=process, control=parent_control,
        preview=parent_preview, trace_path=trace_path,
    )


def _recv_worker_event(worker, *, timeout=10):
    assert worker.control.poll(timeout), "worker event timed out"
    return worker.control.recv()


def _recv_prewarm_terminal_sequence(worker, *, timeout=10):
    events = []
    while not events or events[-1].kind != "ve_prewarm_terminal":
        events.append(_recv_worker_event(worker, timeout=timeout))
    return events


def _stop_spawned_worker(worker):
    if worker.process.is_alive():
        try:
            worker.control.send(RecordingEvent(worker.generation, "", "shutdown"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        worker.process.join(3)
    if worker.process.is_alive():
        worker.process.terminate()
        worker.process.join(3)
    worker.process.close()
    worker.control.close()
    worker.preview.close()


def test_prewarm_worker_reads_target_with_only_prewarm_events_and_no_file_pipeline(tmp_path):
    worker = _spawn_recording_worker(tmp_path)
    request = _prewarm_request(tmp_path)
    try:
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "prewarm_ve", request))
        events = _recv_prewarm_terminal_sequence(worker)
        kinds = [event.kind for event in events]
        assert kinds[0] == "ve_prewarm_started"
        assert kinds[-2:] == ["ve_prewarm_detaching", "ve_prewarm_terminal"]
        assert "ve_prewarm_progress" in kinds[1:-2]
        assert all(event.request_id == request.warmup_id for event in events)
        assert events[0].payload.warmup_id == request.warmup_id
        assert events[0].payload.attempt == request.attempt
        assert events[0].payload.signature == request.signature
        terminal = events[-1].payload
        assert isinstance(terminal, VePrewarmResult)
        assert terminal.success and terminal.stage == "completed"
        assert terminal.frames_per_channel == request.frames_per_channel
        assert terminal.handles_released
        assert terminal.lifecycle_counts.task_create == 1
        assert terminal.lifecycle_counts.task_start == 1
        assert not worker.preview.poll(.1)
        assert not (tmp_path / "must-not-exist.wav").exists()
        assert {path.name for path in tmp_path.iterdir()} == {
            worker.trace_path.name,
        }
        forbidden = {
            "started", "progress", "preview", "finalizing", "completed",
            "failed", "cancelled", "capture_slot_released",
        }
        assert forbidden.isdisjoint(event.kind for event in events)

        # A second unique worker-only attempt proves no recording/result slot,
        # finalizer, descriptor or acknowledgement capacity was consumed.
        second = _prewarm_request(tmp_path, "warmup-2", attempt=2)
        worker.control.send(RecordingEvent(
            worker.generation, second.warmup_id, "prewarm_ve", second))
        second_events = _recv_prewarm_terminal_sequence(worker)
        assert second_events[0].kind == "ve_prewarm_started"
        assert second_events[-2].kind == "ve_prewarm_detaching"
        assert second_events[-1].kind == "ve_prewarm_terminal"
        assert second_events[-1].payload.success
        assert second_events[-1].payload.lifecycle_counts.task_create == 1
    finally:
        _stop_spawned_worker(worker)


def test_prewarm_worker_first_structured_fault_precedes_generation_fatal(tmp_path):
    worker = _spawn_recording_worker(tmp_path, options={"fail_operation": "start_task"})
    request = _prewarm_request(tmp_path)
    try:
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "prewarm_ve", request))
        events = [_recv_worker_event(worker), _recv_worker_event(worker)]
        assert [event.kind for event in events] == [
            "ve_prewarm_terminal", "worker_fatal",
        ]
        terminal = events[0].payload
        assert not terminal.success
        assert terminal.stage == "start_task"
        assert terminal.code is None
        assert terminal.detail == "injected native start_task"
        assert terminal.handles_released
        assert any("owner exited before binding" in item
                   for item in terminal.diagnostics)
        assert events[1].payload.stage == "start_task"
        if worker.control.poll(.2):
            with pytest.raises(EOFError):
                worker.control.recv()
    finally:
        _stop_spawned_worker(worker)


def test_prewarm_worker_rejects_active_recording_and_duplicate_warmup_ids(tmp_path):
    worker = _spawn_recording_worker(tmp_path)
    first = _prewarm_request(tmp_path)
    try:
        worker.control.send(RecordingEvent(
            worker.generation, first.warmup_id, "prewarm_ve", first))
        first_events = _recv_prewarm_terminal_sequence(worker)
        assert first_events[0].kind == "ve_prewarm_started"
        assert first_events[-1].kind == "ve_prewarm_terminal"
        worker.control.send(RecordingEvent(
            worker.generation, first.warmup_id, "prewarm_ve", first))
        fatal = _recv_worker_event(worker)
        assert fatal.kind == "worker_fatal"
        assert fatal.payload.stage == "protocol/prewarm_ve"
        assert "duplicate" in fatal.payload.message
    finally:
        _stop_spawned_worker(worker)

    blocked = _spawn_recording_worker(
        tmp_path / "blocked", options={"zero_reads": True})
    recording = capture_request(tmp_path / "blocked.wav")
    warmup = _prewarm_request(tmp_path, "overlap")
    try:
        blocked.control.send(RecordingEvent(
            blocked.generation, recording.request_id, "start", recording))
        assert _recv_worker_event(blocked).kind == "started"
        blocked.control.send(RecordingEvent(
            blocked.generation, warmup.warmup_id, "prewarm_ve", warmup))
        fatal = _recv_worker_event(blocked)
        assert fatal.kind == "worker_fatal"
        assert fatal.payload.stage == "protocol/prewarm_ve"
        assert "active capture" in fatal.payload.message
    finally:
        _stop_spawned_worker(blocked)

    warming = _spawn_recording_worker(
        tmp_path / "warming", options={"zero_reads": True})
    recording = capture_request(tmp_path / "overlap-recording.wav", request_id="overlap-recording")
    warmup = _prewarm_request(tmp_path, "active-prewarm")
    try:
        warming.control.send(RecordingEvent(
            warming.generation, warmup.warmup_id, "prewarm_ve", warmup))
        assert _recv_worker_event(warming).kind == "ve_prewarm_started"
        warming.control.send(RecordingEvent(
            warming.generation, recording.request_id, "start", recording))
        terminal = _recv_worker_event(warming)
        fatal = _recv_worker_event(warming)
        assert terminal.kind == "ve_prewarm_terminal"
        assert not terminal.payload.success
        assert fatal.kind == "worker_fatal"
        assert fatal.payload.stage == "protocol/start"
        assert not Path(recording.path).exists()
    finally:
        _stop_spawned_worker(warming)


def test_prewarm_worker_ignores_old_generation_and_shutdown_is_bounded(tmp_path):
    worker = _spawn_recording_worker(
        tmp_path, generation=2, options={"zero_reads": True})
    stale = _prewarm_request(tmp_path, "stale")
    current = _prewarm_request(tmp_path, "current")
    try:
        worker.control.send(RecordingEvent(1, stale.warmup_id, "prewarm_ve", stale))
        assert not worker.control.poll(.1)
        worker.control.send(RecordingEvent(
            worker.generation, current.warmup_id, "prewarm_ve", current))
        started = _recv_worker_event(worker)
        assert started.kind == "ve_prewarm_started"
        before = time.monotonic()
        worker.control.send(RecordingEvent(worker.generation, "", "shutdown"))
        terminal = _recv_worker_event(worker)
        assert terminal.kind == "ve_prewarm_terminal"
        assert not terminal.payload.success
        assert terminal.payload.stage == "cancelled"
        worker.process.join(3)
        assert time.monotonic() - before < 3
        assert not worker.process.is_alive()
    finally:
        _stop_spawned_worker(worker)


def test_prewarm_worker_cancel_is_one_terminal_and_does_not_consume_pipeline(tmp_path):
    worker = _spawn_recording_worker(tmp_path, options={"zero_reads": True})
    request = _prewarm_request(tmp_path, "cancel-me")
    try:
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "prewarm_ve", request))
        assert _recv_worker_event(worker).kind == "ve_prewarm_started"
        before = time.monotonic()
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "cancel"))
        terminal = _recv_worker_event(worker)
        assert time.monotonic() - before < 2
        assert terminal.kind == "ve_prewarm_terminal"
        assert terminal.payload.stage == "cancelled"
        assert terminal.payload.handles_released

        second = _prewarm_request(tmp_path, "after-cancel", attempt=2)
        worker.control.send(RecordingEvent(
            worker.generation, second.warmup_id, "prewarm_ve", second))
        assert _recv_worker_event(worker).kind == "ve_prewarm_started"
        worker.control.send(RecordingEvent(
            worker.generation, second.warmup_id, "cancel"))
        assert _recv_worker_event(worker).kind == "ve_prewarm_terminal"
        assert not worker.control.poll(.1)
    finally:
        _stop_spawned_worker(worker)


def test_prewarm_worker_late_cancel_for_terminal_id_is_idempotent_and_reusable(tmp_path):
    worker = _spawn_recording_worker(tmp_path)
    request = _prewarm_request(tmp_path, "already-terminal")
    try:
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "prewarm_ve", request))
        first_events = _recv_prewarm_terminal_sequence(worker)
        assert first_events[0].kind == "ve_prewarm_started"
        assert first_events[-1].kind == "ve_prewarm_terminal"

        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "cancel"))
        assert not worker.control.poll(.15), "late cancel must be an idempotent no-op"

        next_request = _prewarm_request(tmp_path, "still-healthy", attempt=2)
        worker.control.send(RecordingEvent(
            worker.generation, next_request.warmup_id, "prewarm_ve", next_request))
        next_events = _recv_prewarm_terminal_sequence(worker)
        assert next_events[0].kind == "ve_prewarm_started"
        terminal = next_events[-1]
        assert terminal.kind == "ve_prewarm_terminal" and terminal.payload.success
        assert worker.process.is_alive()
    finally:
        _stop_spawned_worker(worker)


def test_prewarm_worker_detach_fault_terminal_precedes_fatal_under_send_backpressure(tmp_path):
    worker = _spawn_blocked_prewarm_worker(
        tmp_path, generation=4,
        options={"block_operation": "read_task_data",
                 "fail_operation": "read_task_data"},
    )
    request = _prewarm_request(tmp_path, "detach-race")
    ipc_release = Path(str(worker.trace_path) + ".ipc-release")
    native_release = Path(str(worker.trace_path) + ".release")
    try:
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "prewarm_ve", request))
        eventually(lambda: Path(str(worker.trace_path) + ".started-blocked").exists())
        worker.control.send(RecordingEvent(
            worker.generation, request.warmup_id, "cancel"))
        # The adapter detach deadline is 0.5 seconds. Keep both send and native
        # boundaries blocked until the worker has produced the race outcomes.
        time.sleep(.7)
        assert not worker.control.poll(.05)
        ipc_release.touch()

        started = _recv_worker_event(worker)
        first = _recv_worker_event(worker)
        second = _recv_worker_event(worker)
        assert started.kind == "ve_prewarm_started"
        assert [first.kind, second.kind] == [
            "ve_prewarm_terminal", "worker_fatal",
        ]
        terminal = first.payload
        assert first.generation == second.generation == terminal.generation == 4
        assert terminal.warmup_id == request.warmup_id
        assert not terminal.success and terminal.stage == "detach"
        assert terminal.detail == "VE adapter detach confirmation timed out"
        assert not terminal.handles_released
        assert second.payload.stage == "detach"
        assert not worker.control.poll(.1), "terminal/fatal pair must be exactly once"

        native_release.touch()
        worker.process.join(3)
        try:
            has_more = worker.control.poll(.2)
        except (BrokenPipeError, OSError):
            has_more = False
        if has_more:
            with pytest.raises(EOFError):
                worker.control.recv()
    finally:
        ipc_release.touch()
        native_release.touch()
        _stop_spawned_worker(worker)


def test_prewarm_worker_protocol_events_bind_warmup_identity_and_payload(tmp_path):
    request = _prewarm_request(tmp_path, "wire-id")
    command = RecordingEvent(3, request.warmup_id, "prewarm_ve", request)
    started = RecordingEvent(
        3, request.warmup_id, "ve_prewarm_started",
        VePrewarmStarted(
            request.warmup_id, 3, request.attempt, request.signature, 10.0))
    progress_payload = VePrewarmProgress(
        request.warmup_id, 3, request.attempt, request.signature,
        10.0, 1, 10.1)
    progress = RecordingEvent(
        3, request.warmup_id, VE_PREWARM_PROGRESS, progress_payload)
    detaching_payload = replace(
        progress_payload, frames_per_channel=request.frames_per_channel)
    detaching = RecordingEvent(
        3, request.warmup_id, VE_PREWARM_DETACHING, detaching_payload)
    wire = (command, started, progress, detaching)
    assert pickle.loads(pickle.dumps(wire)) == wire

    with pytest.raises(ValueError, match="warmup ID"):
        RecordingEvent(3, "different", "prewarm_ve", request)
    with pytest.raises(ValueError, match="payload"):
        RecordingEvent(3, request.warmup_id, "ve_prewarm_terminal", request)


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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("preview", ["none", "withheld"])
@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_spawn_progress_without_preview_consumption(tmp_path, services, streaming, preview, rate):
    service, received, trace_path = services()
    events = Events()
    if preview == "none":
        events.callbacks = RecordingCallbacks(
            result_ready=lambda session, audio: events.results.put(audio))
    req = capture_request(tmp_path / "volts.wav", target_samples=12288, streaming=streaming, sample_rate=rate)
    session = service.start(req, events.callbacks)
    audio = events.results.get(timeout=10)
    session.accept_result()
    assert session.released.wait(5)
    progress = [event for _, event in received if event.kind == "progress"]
    assert progress, "ordinary and streaming capture must report progress without previews"
    assert progress[-1].payload.frames == req.target_samples
    assert all(event.generation == session.generation and event.request_id == req.request_id
               and event.payload.generation == session.generation
               and event.payload.request_id == req.request_id for event in progress)
    counts = [event.payload.frames for event in progress]
    assert counts == sorted(set(counts))
    started = next(event.payload for _, event in received if event.kind == "started")
    trace = [item for item in read_trace(trace_path) if item["operation"] != "owner_join"]
    before = next(item["at"] for item in trace if item["operation"] == "start_task")
    after = next(item["at"] for item in trace if item["operation"] == "verify_actual_sample_rate")
    assert before <= started <= after
    assert all(started <= event.payload.last_frame_at <= at
               for at, event in received if event.kind == "progress")
    assert len({item["thread_id"] for item in trace}) == 1
    assert {item["pid"] for item in trace} == {session.worker_pid}
    assert session.worker_pid != os.getpid()
    assert not any(item["operation"] in ("stop_task", "clear_task", "close")
                   for item in trace), "compatible native ownership remains retained"
    expected = np.tile(np.array([8.25, 2.5], dtype=np.float32), (req.target_samples - 2, 1))
    np.testing.assert_array_equal(audio.multi, expected)
    saved, rate = sf.read(req.path, dtype="float32", always_2d=True)
    assert rate == req.sample_rate and session.descriptor.raw_frames == req.target_samples
    np.testing.assert_array_equal(saved, expected)
    assert session.state == "completed"
    assert events.failed.empty()
    if not streaming or preview == "none":
        assert events.preview.empty()
    else:
        assert events.preview.qsize() == 1  # Credit was never returned.
        rolling = events.preview.get_nowait()
        assert rolling.time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        assert 0 < rolling.sample_stop <= req.target_samples - 2
        for waveform in rolling.waveforms:
            assert waveform.time[-1] == 0.0
            assert waveform.time[0] >= (
                -MAIN_RECORDING_LIVE_WINDOW_SECONDS - PREVIEW_TIME_LOWER_BOUND_TOLERANCE
            )
        assert max(rolling.waveforms[0].amplitude) == 8.25
        assert max(rolling.waveforms[1].amplitude) == 2.5


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


def test_spawn_all_preview_fault_phases_preserve_ve_progress_audio_and_success(
        tmp_path, services):
    def run(preview_fault):
        options = dict(read_delay=.06)
        if preview_fault is not None:
            options["preview_fault"] = preview_fault
        service, received, trace_path = services(options)
        events = Events()
        suffix = "baseline" if preview_fault is None else preview_fault
        request = capture_request(
            tmp_path / f"preview-{suffix}.wav",
            target_samples=8192,
            streaming=True,
            preview_time_mode=PREVIEW_TIME_MODE_CUMULATIVE,
        )
        session = service.start(request, events.callbacks)
        audio = events.results.get(timeout=10)

        session_events = [
            event for _, event in received
            if event.request_id == request.request_id
        ]
        final_progress = [
            event for event in session_events
            if event.kind == "progress" and event.payload.frames == request.target_samples
        ]
        assert len(final_progress) == 1
        assert session_events.index(final_progress[0]) < next(
            index for index, event in enumerate(session_events)
            if event.kind == "completed"
        )
        assert session._capture_deadline.complete
        assert session._capture_deadline.snapshot().frames == request.target_samples

        saved, rate = sf.read(request.path, dtype="float32", always_2d=True)
        trace = read_trace(trace_path)
        session.accept_result()
        assert events.accepted.get(timeout=5) is audio
        assert session.released.wait(5)
        assert session.state == "completed"
        assert events.failed.empty() and events.cancelled.empty()
        return dict(
            progress_frames=tuple(
                event.payload.frames for event in session_events
                if event.kind == "progress"
            ),
            multi=audio.multi.copy(),
            mono=audio.mono.copy(),
            descriptor=audio.descriptor,
            saved=saved,
            rate=rate,
            trace=trace,
        )

    baseline = run(None)
    for preview_fault in ("construct", "begin", "append", "snapshot"):
        actual = run(preview_fault)
        assert actual["progress_frames"][-1] == baseline["progress_frames"][-1] == 8192
        np.testing.assert_array_equal(actual["multi"], baseline["multi"])
        np.testing.assert_array_equal(actual["mono"], baseline["mono"])
        np.testing.assert_array_equal(actual["saved"], baseline["saved"])
        assert actual["rate"] == baseline["rate"] == 51200
        assert (
            actual["descriptor"].purpose,
            actual["descriptor"].sample_rate,
            actual["descriptor"].channels,
            actual["descriptor"].raw_frames,
            actual["descriptor"].final_frames,
            actual["descriptor"].metadata_appended,
            actual["descriptor"].handles_released,
            actual["descriptor"].cleanup_paths,
        ) == (
            baseline["descriptor"].purpose,
            baseline["descriptor"].sample_rate,
            baseline["descriptor"].channels,
            baseline["descriptor"].raw_frames,
            baseline["descriptor"].final_frames,
            baseline["descriptor"].metadata_appended,
            baseline["descriptor"].handles_released,
            baseline["descriptor"].cleanup_paths,
        )
        preview_trace = [
            item for item in actual["trace"]
            if item["operation"].startswith("preview_")
        ]
        assert sum(item["operation"] == "preview_session_construct" for item in preview_trace) == 1
        assert next(
            item for item in preview_trace
            if item["operation"] == "preview_session_construct"
        )["rolling_window_seconds"] is None
        assert any(
            item["operation"] == "preview_fault"
            and item["phase"] == preview_fault
            for item in preview_trace
        )
        assert not any(item["operation"] == "preview_fallback" for item in preview_trace)


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
