"""Parent evidence uses real queue/callback boundaries without changing admission."""
import json
import logging
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from base.recording_service import RecordingService, RecordingCallbacks
from base.recording_process_protocol import RecordingEvent, RecordingProgress
from unit_test.base.test_recording_service_pipeline import ve_probe, slot_payload


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    return RecordingService()


def messages(service):
    return [item[1].getMessage() for item in list(service._timing_logger._queue.queue)]


def test_inbox_commits_before_consumer_and_preserves_tuple(service):
    original = ("shutdown",)
    service._inbox.put_nowait(original)
    item = service._inbox.get_nowait()
    assert item == original and item[:] == original
    assert item.enqueue_started_ns <= item.enqueued_ns <= item.enqueue_ended_ns
    assert service._inbox.maxsize == 64
    for _ in range(64):
        service._inbox.put_nowait(original)
    with pytest.raises(queue.Full):
        service._inbox.put_nowait(original)


def test_blocked_receive_put_retains_observation_before_commit(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    service, worker = probe.service, probe.worker
    entered, release = threading.Event(), threading.Event()
    original = service._inbox.put
    event = RecordingEvent(1, probe.session.request.request_id, "capture_slot_released", slot_payload(probe))
    def gated(item, *args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(item, *args, **kwargs)
    service._inbox.put = gated
    connection = SimpleNamespace(poll=lambda _: True, recv=lambda: event)
    thread = threading.Thread(target=service._receive, args=(worker, connection))
    thread.start()
    try:
        assert entered.wait(1)
        evidence = probe.session._critical_observations["capture_slot_released"]
        assert evidence["received_ns"] <= evidence["enqueue_started_ns"]
        assert "enqueued_ns" not in evidence
        assert any("stage=parent_enqueue_begin" in m for m in messages(service))
    finally:
        worker.stop.set()
        release.set()
        thread.join(3)
    item = service._inbox.get_nowait()
    assert evidence["received_ns"] < item.enqueued_ns
    assert evidence["enqueue_ended_ns"] >= item.enqueued_ns


def test_dispatch_true_kind_recent_ring_and_no_ordinary_message_logs(service, monkeypatch):
    monkeypatch.setattr(service, "_event", lambda *a: None)
    for _ in range(100):
        service._inbox.put(("event", None, RecordingEvent(1, "A", "progress", RecordingProgress("A", 1, 1, 1.0))))
        service._dispatch(service._inbox.get_nowait())
    state = service._recording_diagnostics.snapshot()
    assert len(state["recent"]) == 32
    assert all(r["stage"] == "dispatch_progress" and r["request"] == "A" for r in state["recent"])
    assert state["categories"]["dispatch_progress"]["count"] == 100
    assert not messages(service)


def test_timeout_snapshot_uses_original_decision_clock(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    probe.clock.advance(.1)
    probe.service._event(probe.worker, RecordingEvent(1, probe.session.request.request_id, "progress",
        RecordingProgress(probe.session.request.request_id, 1, probe.session.request.target_samples, 100.1)))
    calls = []
    def clock():
        calls.append(True)
        return 101.1
    probe.service._clock = clock
    probe.service._tick()
    assert probe.session.failure.stage == "capture_release_timeout"
    assert len(calls) == 2  # Original tick + terminate deadline only.
    record = next(m for m in messages(probe.service) if "stage=parent_release_timeout" in m)
    fields = json.loads(record.split(" details=", 1)[1].split(" summary=", 1)[0])
    assert (fields["t0"], fields["deadline"], fields["decision_now"]) == (100.1, 101.1, 101.1)
    assert "log_delivery" in record and "queue_depth" in record


def test_preview_callback_block_is_measured_with_request(service, tmp_path):
    from unit_test.base.test_recording_service import request
    gate, release = threading.Event(), threading.Event()
    session = service.start(request(tmp_path), RecordingCallbacks(preview=lambda *a: (gate.set(), release.wait(3))))
    thread = threading.Thread(target=service._notify, args=(session, "preview", None))
    thread.start()
    try:
        assert gate.wait(1)
        state = service._recording_diagnostics.snapshot()
        assert state["active"][0]["stage"] == "callback_preview"
        assert state["active"][0]["request"] == session.request.request_id
    finally:
        release.set()
        thread.join(3)
    assert service._recording_diagnostics.snapshot()["categories"]["callback_preview"]["count"] == 1


def test_old_cleanup_is_identified_while_new_capture_is_active(service, tmp_path):
    from unit_test.base.test_recording_service import request
    old = service.start(request(tmp_path, request_id="old", path=str(tmp_path / "old.wav")))
    service._capture_session = None
    new = service.start(request(tmp_path, request_id="new", path=str(tmp_path / "new.wav")))
    entered, release = threading.Event(), threading.Event()
    service.defer_path_cleanup(old.request.path, lambda _: (entered.set(), release.wait(3)))
    old._terminal = True
    old.state = "completed"
    thread = threading.Thread(target=service._release, args=(old,))
    thread.start()
    try:
        assert entered.wait(1)
        record = next(m for m in messages(service) if "stage=cleanup_custom_begin" in m)
        assert "request=old" in record and '"active_request":"new"' in record
        assert service._capture_session is new
        assert not old.released.is_set()
    finally:
        release.set()
        thread.join(3)
    assert old.released.is_set() and service._capture_session is new
    recent = service._recording_diagnostics.snapshot()["recent"]
    cleanup = next(item for item in recent if item["stage"] == "cleanup_custom")
    assert cleanup["active_request"] == "new"
    end = next(m for m in messages(service) if "stage=cleanup_custom_end" in m)
    assert '"success":true' in end


def test_retire_evidence_precedes_external_failure_and_termination(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    seen = []
    def failed(*_):
        seen.extend(messages(probe.service))
    probe.session.callbacks = RecordingCallbacks(failed=failed)
    probe.service._retire_generation(probe.worker, "test_timeout", "blocked", probe.session)
    record = next(m for m in seen if "stage=worker_retire_requested" in m)
    assert '"cause":"test_timeout"' in record and '"worker_pid":123' in record
    assert any("stage=terminate_requested" in m for m in messages(probe.service))
    assert not any("stage=kill_requested" in m for m in messages(probe.service))


def test_normal_summary_is_compact_and_has_all_task_categories(service, tmp_path):
    from unit_test.base.test_recording_service import request
    session = service.start(request(tmp_path))
    session._terminal = True
    session.state = "completed"
    service._release(session)
    record = next(m for m in messages(service) if "stage=parent_session_summary" in m)
    summary = json.loads(record.split(" summary=", 1)[1])
    assert not summary["recent"]
    assert "log_delivery" in summary
    assert len(record.encode()) < 4096
    assert "schedule_gap" in service._recording_diagnostics.snapshot()["categories"]


def test_extended_timeout_summary_preserves_32_task_records(service):
    diag = service._recording_diagnostics
    for _ in range(32):
        token = diag.begin("dispatch_preview", request="request-name-long-enough", generation=1)
        diag.end(token, recent=True)
    diag.summary("test_timeout", max_chars=32768)
    summary = json.loads(messages(service)[-1].split(" summary=", 1)[1])
    assert len(summary["recent"]) == 32
    assert summary["omitted_recent"] == 0


@pytest.mark.parametrize("name,invoke", [
    ("callback_release", lambda s: s._invoke_release_callback(lambda *a: None, "released", ())),
    ("callback_shutdown", lambda s: s._invoke_shutdown(lambda: None)),
])
def test_lifecycle_callback_durations(service, name, invoke):
    invoke(service)
    assert service._recording_diagnostics.snapshot()["categories"][name]["count"] == 1


def test_slow_callback_keeps_capture_at_start_even_if_callback_changes_it(service, tmp_path):
    from unit_test.base.test_recording_service import request
    old = service.start(request(tmp_path, request_id="old"))
    service._capture_session = None
    new = service.start(request(tmp_path, request_id="new", path=str(tmp_path / "new.wav")))
    tick = [1_000_000_000]
    service._recording_diagnostics._perf_ns = lambda: tick[0]
    def callback(*_):
        service._capture_session = None
        tick[0] += 110_000_000
    old.callbacks = RecordingCallbacks(preview=callback)
    service._notify(old, "preview", None)
    record = next(m for m in messages(service) if "stage=callback_preview" in m)
    assert "request=old" in record and '"active_request":"new"' in record
    stats = service._recording_diagnostics.snapshot()["categories"]["callback_preview"]
    assert stats["max_active_request"] == new.request.request_id


def test_delayed_empty_queue_wait_is_separate_from_tick(service, monkeypatch):
    from base import recording_service as module
    tick = [1_000_000_000]
    monkeypatch.setattr(module, "perf_counter_ns", lambda: tick[0])
    service._recording_diagnostics._perf_ns = lambda: tick[0]
    def empty(timeout):
        assert timeout == .02
        tick[0] += 220_000_000
        service._closing = True
        raise queue.Empty
    monkeypatch.setattr(service._inbox, "get", empty)
    service._run()
    state = service._recording_diagnostics.snapshot()
    assert state["categories"]["schedule_gap"]["max_ns"] == 200_000_000
    assert state["categories"]["tick"]["max_ns"] == 0


def test_timeout_copies_mutating_observation_dictionaries(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    class MutationOnItems(dict):
        def items(self):
            iterator = super().items()
            for pair in iterator:
                self["enqueued_ns"] = 99
                yield pair
    probe.session._critical_observations = MutationOnItems(
        progress=MutationOnItems(received_ns=1, enqueue_started_ns=2))
    probe.service._release_timeout_evidence(probe.session, 101.1)
    assert any("stage=parent_release_timeout" in m for m in messages(probe.service))


def test_malformed_kind_does_not_run_equality_during_observation(service, monkeypatch):
    class BrokenEquality:
        def __eq__(self, other):
            raise ValueError("diagnostics must not compare non-string kinds")
    event = object.__new__(RecordingEvent)
    object.__setattr__(event, "kind", BrokenEquality())
    seen = []
    monkeypatch.setattr(service, "_event", lambda worker, value: seen.append(value))
    service._dispatch(("event", None, event))
    assert seen == [event]


def test_invalid_slot_protocol_boundary_records_rejection(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    payload = slot_payload(probe)
    object.__setattr__(payload, "raw_frames", -1)
    event = object.__new__(RecordingEvent)
    for name, value in dict(kind="capture_slot_released", payload=payload,
                            request_id=probe.session.request.request_id, generation=1, version=1).items():
        object.__setattr__(event, name, value)
    probe.service._event(probe.worker, event)
    assert probe.session.failure.stage == "protocol"
    assert any('stage=parent_slot_outcome' in m and '"outcome":"invalid"' in m
               for m in messages(probe.service))


def test_put_return_race_keeps_commit_and_full_producer_duration(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = queue.Queue.put
    def delayed_return(inbox, item, block=True, timeout=None):
        result = original(inbox, item, block, timeout)
        entered.set()
        assert release.wait(3)
        return result
    monkeypatch.setattr(queue.Queue, "put", delayed_return)
    thread = threading.Thread(target=service._inbox.put, args=(("cancel", None),))
    thread.start()
    try:
        assert entered.wait(1)
        item = service._inbox.get_nowait()
        assert item.enqueued_ns is not None and item.enqueue_ended_ns is None
        assert item.enqueued_ns >= item.enqueue_started_ns
    finally:
        release.set()
        thread.join(3)
    state = service._recording_diagnostics.snapshot()
    assert state["categories"]["inbox_put"]["count"] == 1
    assert state["categories"]["inbox_put"]["max_ns"] == item.enqueue_ended_ns - item.enqueue_started_ns


def test_normal_ve_parent_event_budget(tmp_path):
    from unit_test.base.ve3668n_fakes import capture_request
    records, finished = [], threading.Event()
    class Capture(logging.Handler):
        def emit(self, record):
            if record.msg.startswith("Recording diagnostic"):
                records.append(record)
                if "stage=parent_session_summary" in record.msg:
                    finished.set()
    logger = logging.Logger("parent-diagnostic-budget", logging.INFO)
    logger.addHandler(Capture())
    service = RecordingService(
        backend_factory="unit_test.base.ve3668n_fakes:capture_dependencies",
        backend_options={"trace_path": str(tmp_path / "sdk.jsonl"), "read_delay": .005})
    service._logger = logger
    try:
        session = service.start(capture_request(tmp_path / "normal.wav", target_samples=8192),
            RecordingCallbacks(started=lambda s: None, finalizing=lambda s: None,
                preview=lambda s, p: s.release_preview(p.sequence),
                result_ready=lambda s, a: s.accept_result(), accepted=lambda *a: None,
                released=lambda s: None))
        assert finished.wait(15)
        assert session.state == "completed" and session.released.is_set()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        size = sum(len(formatter.format(r).encode("utf-8")) + 1 for r in records)
        assert len(records) <= 21 and size <= 9 * 1024, (len(records), size)
        print(f"parent_normal_diagnostics: events={len(records)} formatted_bytes={size}")
    finally:
        service.shutdown()
        assert service.closed.wait(10)


@pytest.mark.parametrize("kind", ["start", "cancel", "accept", "reject", "preview_ack",
    "prewarm_ve", "release_ve", "release_callback", "shutdown", "read", "event", "broken"])
def test_every_producer_kind_uses_same_bounded_timed_queue(service, kind):
    session_or_worker = None
    item = (kind, session_or_worker, None)
    service._inbox.put_nowait(item)
    received = service._inbox.get_nowait()
    assert received == item and received[1:] == item[1:]
    assert received.enqueue_started_ns <= received.enqueued_ns <= received.enqueue_ended_ns
    assert service._recording_diagnostics.snapshot()["categories"]["inbox_put"]["count"] == 1


def test_kill_and_exit_evidence_match_real_lifecycle_calls(monkeypatch, tmp_path):
    probe = ve_probe(monkeypatch, tmp_path)
    service, worker = probe.service, probe.worker
    calls = []
    worker.process.terminate = lambda: calls.append("terminate")
    worker.process.kill = lambda: calls.append("kill")
    service._retire_generation(worker, "capture_release_timeout", "blocked", probe.session)
    assert calls == ["terminate"]
    probe.clock.advance(3)
    service._tick()
    assert calls == ["terminate", "kill"]
    record = next(m for m in messages(service) if "stage=kill_requested" in m)
    assert '"cause":"capture_release_timeout"' in record and '"worker_pid":123' in record
    worker.process.is_alive = lambda: False
    worker.process.exitcode = -15
    worker.process.join = lambda timeout: None
    worker.control = worker.preview = SimpleNamespace(close=lambda: None)
    def close():
        record = next(m for m in messages(service) if "stage=worker_exit_observed" in m)
        assert '"exitcode":-15' in record and "request=ve-capture" in record
        calls.append("close")
    worker.process.close = close
    service._dead(worker)
    assert calls == ["terminate", "kill", "close"]
    assert sum("stage=kill_requested" in m for m in messages(service)) == 1


def test_retire_is_visible_before_pending_release_callback(monkeypatch, tmp_path):
    from base.recording_service import _PendingVeRelease
    probe = ve_probe(monkeypatch, tmp_path)
    seen = []
    probe.service._pending_ve_release = _PendingVeRelease(None,
        callback=lambda *_: seen.extend(messages(probe.service)))
    probe.service._retire_generation(probe.worker, "release_ve", "blocked", probe.session)
    assert any("stage=worker_retire_requested" in m for m in seen)
