"""Parent service startup boundaries using the real spawn/fake-device path."""
import logging
import pytest

from base.recording_service import RecordingCallbacks, RecordingService
from unit_test.base.test_recording_service import request


def test_startup_continues_with_blocked_project_sink_and_bounded_queue(tmp_path, monkeypatch):
    """Integration characterization: real spawn starts while the sink is gated."""
    import threading
    from base import log_manager
    from unit_test.logging_test_support import isolated_project_logger
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(log_manager, 'LOG_QUEUE_CAPACITY', 2)
    native_write = log_manager._BatchFileHandler.write_batch

    def blocked(sink, entries):
        entered.set()
        assert release.wait(20)
        return native_write(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, 'write_batch', blocked)
    with isolated_project_logger(tmp_path, monkeypatch):
        logger = log_manager.LogManager.set_log_handler('core')
        service = RecordingService(backend_factory='unit_test.base.recording_process_fakes:process_dependencies',
                                   backend_options={'trace_dir': str(tmp_path)})
        try:
            logger.error('gate startup sink')
            assert entered.wait(2)
            session = service.start(request(tmp_path), RecordingCallbacks(
                started=lambda s: started.set(), result_ready=lambda s, audio: s.accept_result()))
            assert started.wait(12), 'startup waited for the file sink'
            assert session.released.wait(5)
            assert session.state == 'completed'
            stats = log_manager.LogManager.get_async_stats()
            assert stats['dropped_full'] > 0
            assert stats['pending'] <= 3  # two queued records plus the blocked batch
        finally:
            release.set()
            service.shutdown()
            assert service.closed.wait(10)


def records(caplog, request_id=None):
    rows = [dict(part.split("=", 1) for part in row.message.split()[1:])
            for row in caplog.records if row.message.startswith("recording_startup ")]
    return rows if request_id is None else [r for r in rows if r["request_id"] == request_id]


def test_service_cold_and_reused_start_boundaries(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    service = RecordingService(backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
                               backend_options={"trace_dir": str(tmp_path)})
    seen = []
    try:
        for name in ("one", "two"):
            session = service.start(request(tmp_path, request_id=name), RecordingCallbacks(
                started=lambda s: seen.append(s.request.request_id),
                result_ready=lambda s, audio: s.accept_result()))
            assert session.released.wait(15)
            rows = records(caplog, name)
            events = [r["event"] for r in rows]
            expected = ["service_accept", "worker_selected", "command_enqueue", "command_dequeue", "parent_started"]
            assert all(event in events for event in expected)
            assert [events.index(event) for event in expected] == sorted(events.index(event) for event in expected)
            summary, = [r for r in rows if r["event"] == "summary"]
            assert summary["outcome"] == "started"
            assert summary["completion_boundary"] == "service_callback_return"
            assert summary["start_boundary"] == "service_accept"
            assert summary["generation"] == str(session.generation)
            assert summary["worker_pid"] == str(session.worker_pid)
            selected = next(r for r in rows if r["event"] == "worker_selected")
            assert selected["worker_mode"] == ("cold" if name == "one" else "reused")
            assert next(r for r in rows if r["event"] == "command_dequeue")["domain"] == "service_sender"
            assert next(r for r in rows if r["event"] == "service_submit")["csv_active"] == "unknown"
            assert next(r for r in rows if r["stage"] == "command_send" and r["event"] == "end")["outcome"] == "ok"
            assert summary["dropped_events"] == "0"
        assert seen == ["one", "two"]
    finally:
        service.shutdown()
        assert service.closed.wait(10)


def test_service_started_callback_failure_is_not_success(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    service = RecordingService(backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
                               backend_options={"trace_dir": str(tmp_path)})
    def broken(session):
        raise ValueError("callback failed")
    try:
        session = service.start(request(tmp_path), RecordingCallbacks(started=broken,
            result_ready=lambda s, audio: s.accept_result()))
        assert session.released.wait(15)
        summary, = [r for r in records(caplog, "one") if r["event"] == "summary"]
        assert summary["outcome"] == "failed"
        end, = [r for r in records(caplog, "one")
                if r["event"] == "end" and r["stage"] == "service_callback"]
        assert (end["outcome"], end["error_type"], end["reason"]) == (
            "failed", "ValueError", "callback_failed")
        assert session.state == "completed"
    finally:
        service.shutdown()
        assert service.closed.wait(10)


@pytest.mark.parametrize("intent", ["cancel", "shutdown"])
def test_cancel_intent_precedes_queued_started_and_stale_requests(tmp_path, monkeypatch, caplog, intent):
    from types import SimpleNamespace
    from base.recording_service import _Worker
    from base.recording_process_protocol import RecordingEvent
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    service = RecordingService()
    worker = _Worker(1, SimpleNamespace(pid=123), None, None, None)
    worker.ready = True
    service._worker = worker
    called = []
    session = service.start(request(tmp_path), RecordingCallbacks(started=lambda s: called.append(s)))
    service._dispatch(service._inbox.get_nowait())
    service.cancel("one") if intent == "cancel" else service.shutdown()
    service._event(worker, RecordingEvent(1, "one", "started"))
    summary, = [r for r in records(caplog, "one") if r["event"] == "summary"]
    assert summary["outcome"] == "cancelled"
    # Business callback behavior stays unchanged, but cannot revive the trace.
    service._event(worker, RecordingEvent(1, "old-request", "started"))
    service._event(worker, RecordingEvent(2, "one", "started"))
    assert len([r for r in records(caplog, "one") if r["event"] == "summary"]) == 1


def test_start_failure_closes_ready_interval(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    service = RecordingService()
    session = service.start(request(tmp_path))
    monkeypatch.setattr(service, "_spawn", lambda: (_ for _ in ()).throw(OSError("spawn denied")))
    with pytest.raises(OSError) as error:
        service._begin(session)
    service._handle_supervisor_exception(error.value)
    rows = records(caplog, "one")
    summary, = [r for r in rows if r["event"] == "summary"]
    assert summary["outcome"] == "failed"
    end, = [r for r in rows if r["event"] == "end" and r["stage"] == "worker_ready"]
    assert end["outcome"] == "failed"


def test_supplied_standalone_trace_with_mock_callback_finishes_locally(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from base.recording_startup_trace import RecordingStartupTrace
    from base.recording_service import _Worker
    from base.recording_process_protocol import RecordingEvent
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    service = RecordingService()
    worker = _Worker(1, SimpleNamespace(pid=123), None, None, None)
    worker.ready = True
    service._worker = worker
    trace = RecordingStartupTrace(service._logger, process="parent")
    started = Mock()
    session = service.start(request(tmp_path), RecordingCallbacks(started=started), startup_trace=trace)
    service._dispatch(service._inbox.get_nowait())
    service._event(worker, RecordingEvent(1, "one", "started"))
    started.assert_called_once_with(session)
    summary, = [r for r in records(caplog, "one") if r["event"] == "summary"]
    assert summary["completion_boundary"] == "service_callback_return"
    assert summary["outcome"] == "started"


@pytest.mark.parametrize("terminal,reason", [
    ("timeout", "ready_timeout"), ("failed", "worker"),
    ("cancelled", "cancel_requested"),
])
def test_worker_ready_terminal_records_actual_outcome(tmp_path, monkeypatch, caplog, terminal, reason):
    from types import SimpleNamespace
    from base.recording_service import _Worker
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    service = RecordingService(monotonic=lambda: 100.0)
    process = SimpleNamespace(pid=123, is_alive=lambda: True, terminate=lambda: None)
    worker = _Worker(1, process, None, None, 101.0)
    monkeypatch.setattr(service, "_spawn", lambda: setattr(service, "_worker", worker))
    session = service.start(request(tmp_path))
    service._dispatch(service._inbox.get_nowait())
    if terminal == "timeout":
        worker.deadline = 99.0
        service._tick()
    elif terminal == "failed":
        service._fail(session, "worker", "worker failed before ready")
    else:
        service.cancel("one")
        service._dispatch(service._inbox.get_nowait())
    end, = [r for r in records(caplog, "one")
            if r["event"] == "end" and r["stage"] == "worker_ready"]
    expected = "cancelled" if terminal == "cancelled" else "failed"
    assert (end["outcome"], end["error_type"], end["reason"]) == (expected, "unknown", reason)
    assert session.state == expected
    assert session._startup_ready_stage is None
