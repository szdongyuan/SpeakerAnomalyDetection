"""Parent-side retained VE release and preparing-start supervision."""
from types import SimpleNamespace
import json
import queue
import threading
import time

import pytest

from base.recording_process_protocol import RecordingEvent, VeLifecycleCounts, VeReleaseOutcome
from base.recording_service import RecordingService, _Worker
from base.ve3668n_input import ve_acquisition_signature
from unit_test.base.ve3668n_fakes import capture_request
from unit_test.base.test_recording_service import Events, eventually, request as ordinary_request


def changed_request(tmp_path, change):
    from unit_test.base.ve3668n_fakes import device_info, input_config, wav_metadata

    if change == "rate":
        return capture_request(tmp_path / "B.wav", request_id="B", sample_rate=48000)
    config = input_config(range_min=-.5, range_max=.5)
    metadata = wav_metadata(("none", "none"), sample_rate=51200)
    metadata["acquisition"].update(config, machine_id="test-machine-1")
    return capture_request(tmp_path / "B.wav", request_id="B",
                           device=device_info(input_config=config), calibration_metadata=metadata)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *args, **kwargs: None)
    result = RecordingService(start_timeout=1, release_timeout=5)
    process = SimpleNamespace(pid=123, is_alive=lambda: True,
                              terminate=lambda: None, kill=lambda: None)
    result._worker = _Worker(1, process, None, None, None)
    result._worker.ready = True
    return result


def test_release_semantics_cover_uninitialized_same_busy_pending_and_closing(service, tmp_path):
    signature = ve_acquisition_signature(
        capture_request(tmp_path / "A.wav").device, (7, 1), 51200)
    assert service.release_ve(signature) == "released"
    service._retained_ve_signature = signature
    assert service.release_ve(signature) == "unchanged"
    session = service.start(capture_request(tmp_path / "A.wav"))
    assert service.release_ve(None) == "busy"
    service._capture_session = None
    assert service.release_ve(None) == "pending"
    assert service.release_ve(None) == "pending"
    service._pending_ve_release = None
    service._closing = True
    assert service.release_ve(None) == "closing"
    assert session.request.request_id in service._sessions


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("retirement", ["uncertain", "retiring"])
def test_release_rejects_retirement_before_immediate_or_pending_paths(
    service, retained, retirement,
):
    if retained:
        service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    if retirement == "uncertain":
        service._ownership_uncertain = True
        service._worker = None
    else:
        service._worker.retiring = True

    callbacks = []
    assert service.release_ve(None, lambda *result: callbacks.append(result)) == "busy"
    assert service._pending_ve_release is None
    assert callbacks == []


@pytest.mark.parametrize("retained_at_retirement", [False, True])
@pytest.mark.parametrize("cleanup", ["retire", "dead"])
def test_pending_release_racing_retirement_is_consumed_once_and_admission_recovers(
    service, retained_at_retirement, cleanup,
):
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    callbacks = []
    assert service.release_ve(None, lambda *result: callbacks.append(result)) == "pending"
    pending = service._pending_ve_release
    if not retained_at_retirement:
        service._retained_ve_signature = None

    worker = service._worker
    worker.process.join = lambda timeout=0: None
    worker.process.close = lambda: None
    worker.control = SimpleNamespace(close=lambda: None)
    worker.preview = SimpleNamespace(close=lambda: None)
    if cleanup == "retire":
        service._retire_generation(worker)
    else:
        worker.retiring = True
        service._ownership_uncertain = True
    worker.process.is_alive = lambda: False
    service._dead(worker)

    assert service._pending_ve_release is None
    assert len(callbacks) == 1
    status, diagnostics = callbacks[0]
    assert status == "failed"
    assert diagnostics and "retir" in " ".join(diagnostics).lower()
    assert service.can_start_recording
    assert pending is not service._pending_ve_release


@pytest.mark.parametrize("change", ["rate", "range"])
def test_incompatible_start_prepares_then_arms_start_deadline_only_after_release(service, tmp_path, change):
    first = capture_request(tmp_path / "A.wav")
    service._retained_ve_signature = ve_acquisition_signature(
        first.device, first.channels, first.sample_rate)
    second = changed_request(tmp_path, change)
    session = service.start(second)
    service._dispatch(service._inbox.get_nowait())
    release_command = service._worker.outgoing.get_nowait()
    assert session.state == "preparing" and session._deadline is None
    assert release_command.kind == "release_ve"
    assert service._pending_ve_release.deadline is not None

    outcome = VeReleaseOutcome(1, service._retained_ve_signature)
    service._event(service._worker, RecordingEvent(1, "", "ve_released", outcome))
    start = service._worker.outgoing.get_nowait()
    assert start.kind == "start" and start.request_id == "B"
    assert session.state == "starting" and session._deadline is not None


@pytest.mark.parametrize("change", ["rate", "range"])
def test_release_failure_preserves_first_release_stage_and_retires(service, tmp_path, change):
    request = changed_request(tmp_path, change)
    first = capture_request(tmp_path / "A.wav")
    service._retained_ve_signature = ve_acquisition_signature(
        first.device, first.channels, first.sample_rate)
    session = service.start(request)
    service._dispatch(service._inbox.get_nowait())
    service._worker.outgoing.get_nowait()
    outcome = VeReleaseOutcome(1, service._retained_ve_signature, ("clear failed",))
    service._event(service._worker, RecordingEvent(1, "", "ve_release_failed", outcome))
    assert service._worker.retiring and service._ownership_uncertain
    assert session.failure.stage == "release_ve"
    assert "clear failed" in session.failure.message
    assert all(command.kind != "start" for command in list(service._worker.outgoing.queue))


@pytest.mark.parametrize("failure", ["terminal", "timeout"])
def test_release_failure_closes_admission_before_reentrant_callback(
    service, tmp_path, failure,
):
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    callback_observations = []

    def callback(status, diagnostics):
        try:
            service.start(capture_request(tmp_path / "reentrant.wav", request_id="reentrant"))
        except RuntimeError:
            admission = "rejected"
        else:
            admission = "admitted"
        callback_observations.append((status, diagnostics, admission,
                                      service._worker.retiring,
                                      service._ownership_uncertain))

    assert service.release_ve(None, callback) == "pending"
    service._dispatch(service._inbox.get_nowait())
    assert service._worker.outgoing.get_nowait().kind == "release_ve"
    if failure == "terminal":
        service._event(service._worker, RecordingEvent(
            1, "", "ve_release_failed",
            VeReleaseOutcome(1, service._retained_ve_signature, ("clear failed",))))
    else:
        service._pending_ve_release.deadline = time.monotonic() - .01
        service._tick()

    assert callback_observations
    status, diagnostics, admission, retiring, uncertain = callback_observations[0]
    assert status == "failed" and diagnostics
    assert admission == "rejected"
    assert retiring and uncertain
    assert service._pending_ve_release is None
    assert not service.can_start_recording

    worker = service._worker
    worker.process.is_alive = lambda: False
    worker.process.join = lambda timeout=0: None
    worker.process.close = lambda: None
    worker.control = SimpleNamespace(close=lambda: None)
    worker.preview = SimpleNamespace(close=lambda: None)
    service._dead(worker)
    assert service.can_start_recording


def test_preparing_release_timeout_never_consumes_start_timeout(service, tmp_path):
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    session = service.start(capture_request(tmp_path / "B.wav"))
    service._dispatch(service._inbox.get_nowait())
    service._worker.outgoing.get_nowait()
    assert session._deadline is None
    service._pending_ve_release.deadline = time.monotonic() - .01
    service._tick()
    assert session.failure.stage == "release_ve"
    assert service._worker.retiring and not service.can_start_recording


def test_cancel_before_preparing_begin_clears_unsent_release_and_restores_admission(
    service, tmp_path,
):
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    session = service.start(capture_request(tmp_path / "B.wav"))
    session.cancel()
    assert service._pending_ve_release is None

    service._dispatch(service._inbox.get_nowait())

    assert service._pending_ve_release is None
    assert session.state == "cancelled" and session.released.is_set()
    assert service._capture_session is None
    assert service._worker.outgoing.empty()
    assert service.can_start_recording


def test_cancel_during_slow_begin_never_transfers_child_or_path_ownership(
    service, monkeypatch, tmp_path,
):
    from base import recording_service as module

    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    entered = threading.Event()
    release_setup = threading.Event()
    real_mkdtemp = module.tempfile.mkdtemp

    def gated_mkdtemp(*args, **kwargs):
        entered.set()
        assert release_setup.wait(5)
        return real_mkdtemp(*args, dir=tmp_path, **kwargs)

    monkeypatch.setattr(module.tempfile, "mkdtemp", gated_mkdtemp)
    setup_request = ordinary_request(
        tmp_path, request_id="slow-setup", purpose="calibration", channels=(0,))
    session = service.start(setup_request)
    begin = threading.Thread(target=service._dispatch, args=(service._inbox.get_nowait(),))
    begin.start()
    assert entered.wait(5)

    session.cancel()
    assert service._pending_ve_release is None
    release_setup.set()
    begin.join(5)

    assert not begin.is_alive()
    assert service._worker.outgoing.empty()
    assert session.state == "cancelled" and session.released.is_set()
    assert session._child_released and not session._sent
    assert service._capture_session is None and not service._leases
    assert session._temporary_dir is not None
    assert not module.os.path.exists(session._temporary_dir)
    assert service.can_start_recording


def test_cancel_preparing_while_waiting_ready_clears_unsent_release(
    service, monkeypatch, tmp_path,
):
    worker = service._worker
    service._worker = None
    worker.ready = False
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    monkeypatch.setattr(service, "_spawn", lambda: setattr(service, "_worker", worker))
    session = service.start(capture_request(tmp_path / "B.wav"))
    service._dispatch(service._inbox.get_nowait())
    assert service._pending_ve_release is not None
    assert not service._pending_ve_release.sent

    session.cancel()
    assert service._pending_ve_release is None
    # Ready can already be queued when the public cancel flag becomes visible;
    # it must observe that flag before the queued cancel command is dispatched.
    service._event(worker, RecordingEvent(1, "", "ready"))
    assert service._pending_ve_release is None
    assert session.state == "cancelled" and session.released.is_set()
    assert service._capture_session is None

    service._dispatch(service._inbox.get_nowait())
    assert worker.outgoing.empty()
    assert service.can_start_recording


def test_cancel_after_preparing_release_sent_keeps_release_terminal_but_never_starts(
    service, tmp_path,
):
    service._retained_ve_signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    session = service.start(capture_request(tmp_path / "B.wav"))
    service._dispatch(service._inbox.get_nowait())
    release = service._worker.outgoing.get_nowait()
    assert release.kind == "release_ve"
    pending = service._pending_ve_release
    assert pending.sent and pending.deadline is not None

    session.cancel()
    service._dispatch(service._inbox.get_nowait())
    assert service._pending_ve_release is pending
    assert session.state == "cancelled" and session.released.is_set()
    assert not service.can_start_recording

    service._event(service._worker, RecordingEvent(
        1, "", "ve_released", VeReleaseOutcome(1, service._retained_ve_signature)))
    assert service._pending_ve_release is None
    assert service._worker.outgoing.empty()
    assert session.state == "cancelled"
    assert service.can_start_recording


def test_unexpected_current_release_terminal_retires_but_old_generation_is_ignored(service):
    signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    service._retained_ve_signature = signature
    service._worker.generation = 2
    service._event(service._worker, RecordingEvent(1, "", "ve_released",
                                                   VeReleaseOutcome(1, signature)))
    assert not service._worker.retiring
    service._event(service._worker, RecordingEvent(2, "", "ve_released",
                                                   VeReleaseOutcome(2, signature)))
    assert service._worker.retiring and not service.can_start_recording


def test_release_before_first_slot_advances_all_six_expected_lifecycle_counts(service):
    signature = ("vkinging", "old", (0,), 51200, "IEPE", "V", -10.0, 10.0)
    service._retained_ve_signature = signature
    service._expected_next_lifecycle_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    assert service.release_ve(None) == "pending"
    service._dispatch(service._inbox.get_nowait())
    service._worker.outgoing.get_nowait()
    service._event(service._worker, RecordingEvent(
        1, "", "ve_released", VeReleaseOutcome(1, signature)))
    assert service._expected_next_lifecycle_counts == VeLifecycleCounts(2, 2, 2, 1, 1, 1)


def test_spawn_idle_release_reinitializes_lazily_in_same_worker(tmp_path):
    trace_dir = tmp_path / "release-trace"
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:persistent_ve_worker_dependencies",
        backend_options={"trace_dir": str(trace_dir)})
    callbacks = queue.Queue()
    try:
        first_events = Events()
        first = service.start(capture_request(tmp_path / "A.wav", request_id="A"), first_events.callbacks)
        first_events.results.get(timeout=10)
        first.accept_result()
        assert first.released.wait(5)
        pid = first.worker_pid

        assert service.release_ve(None, lambda status, diagnostics: callbacks.put((status, diagnostics))) == "pending"
        assert callbacks.get(timeout=10) == ("released", ())
        assert service.retained_ve_signature is None and service.worker_pid == pid
        assert service.can_start_recording

        second_events = Events()
        second = service.start(capture_request(
            tmp_path / "B.wav", request_id="B", sample_rate=48000), second_events.callbacks)
        eventually(lambda: not second_events.results.empty() or not second_events.failed.empty())
        assert second_events.failed.empty(), (second_events.failed.get_nowait(), service.diagnostics)
        second_events.results.get_nowait()
        second.accept_result()
        assert second.released.wait(5) and second.worker_pid == pid
    finally:
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics


@pytest.mark.parametrize("failure", [None, "clear_task"])
def test_spawn_range_change_releases_before_new_native_task(tmp_path, failure):
    trace_dir = tmp_path / "range-trace"
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:persistent_ve_worker_dependencies",
        backend_options={"trace_dir": str(trace_dir), "fail_release_operation": failure})
    try:
        first_events = Events()
        first = service.start(capture_request(tmp_path / "A.wav", request_id="A"), first_events.callbacks)
        first_events.results.get(timeout=10)
        first.accept_result()
        assert first.released.wait(5)
        second_events = Events()
        second = service.start(changed_request(tmp_path, "range"), second_events.callbacks)
        eventually(lambda: not second_events.results.empty() or not second_events.failed.empty())
        if failure is None:
            assert second_events.failed.empty()
            second_events.results.get_nowait()
            second.accept_result()
            assert second.released.wait(5)
            assert second.worker_pid == first.worker_pid
        else:
            assert second_events.results.empty()
            assert second_events.failed.get_nowait().stage == "release_ve"
        operations = [json.loads(line)["operation"]
                      for line in (trace_dir / "native.jsonl").read_text(encoding="utf-8").splitlines()]
        creations = [i for i, operation in enumerate(operations) if operation == "create_task"]
        assert len(creations) == (2 if failure is None else 1)
        if failure is None:
            between = operations[creations[0] + 1:creations[1]]
            assert between.index("stop_task") < between.index("clear_task") < between.index("close")
    finally:
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics
