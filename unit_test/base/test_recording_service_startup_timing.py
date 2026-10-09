"""Direct parent logs at existing service boundaries."""
import logging
from types import SimpleNamespace
import pytest
from base.recording_service import RecordingCallbacks, RecordingService, _Worker
from base.recording_process_protocol import RecordingEvent
from unit_test.base.test_recording_service import request
from unit_test.base.test_recording_capture_startup_timing import records


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

    with isolated_project_logger(tmp_path, monkeypatch):
        monkeypatch.setattr(log_manager._BatchFileHandler, 'write_batch', blocked)
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


def test_readiness_enqueue_send_and_receipt_have_request_identity(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, '_start_thread', lambda *a, **k: None)
    clock = [10.0]
    monkeypatch.setattr('base.recording_service.perf_counter', lambda: clock[0])
    service = RecordingService()
    worker = _Worker(7, SimpleNamespace(pid=123), None, None, None)
    monkeypatch.setattr(service, '_spawn', lambda: setattr(service, '_worker', worker))
    called = []
    session = service.start(request(tmp_path), RecordingCallbacks(started=called.append))
    service._dispatch(service._inbox.get_nowait())
    assert any(r['stage'] == 'worker_ready' and r.get('event') == 'begin' for r in records(caplog))
    clock[0] += 2.2
    service._event(worker, RecordingEvent(7, '', 'ready'))
    ready, = [r for r in records(caplog) if r['stage'] == 'worker_ready' and r.get('event') == 'end']
    assert float(ready['seconds']) == pytest.approx(2.2)
    def send(event):
        assert event.kind == 'start'
        assert not [r for r in records(caplog) if r['stage'] == 'start_send']
        worker.stop.set()
    worker.control = SimpleNamespace(send=send)
    service._send(worker)
    service._event(worker, RecordingEvent(7, 'one', 'started'))
    assert called == [session]
    assert {'worker_ready', 'start_enqueue', 'start_send', 'started_receive'} <= {r['stage'] for r in records(caplog, 'one')}


@pytest.mark.parametrize('terminal', ['failed', 'cancelled'])
def test_readiness_failure_and_cancellation_preserve_terminal_state(tmp_path, monkeypatch, caplog, terminal):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, '_start_thread', lambda *a, **k: None)
    service = RecordingService()
    worker = _Worker(1, SimpleNamespace(pid=123), None, None, None)
    monkeypatch.setattr(service, '_spawn', lambda: setattr(service, '_worker', worker))
    session = service.start(request(tmp_path))
    service._dispatch(service._inbox.get_nowait())
    if terminal == 'failed':
        service._fail(session, 'worker', 'worker failed before ready')
    else:
        service.cancel('one')
        service._dispatch(service._inbox.get_nowait())
    assert session.state == terminal
    end, = [r for r in records(caplog, 'one') if r['stage'] == 'worker_ready' and r.get('event') == 'end']
    assert end['outcome'] == terminal
