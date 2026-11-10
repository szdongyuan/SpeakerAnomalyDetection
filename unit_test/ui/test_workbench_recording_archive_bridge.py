from datetime import date
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QSignalSpy, QTest

from base.workbench_recording_archive import WorkbenchRecordingArchiveError
from ui import workbench_recording_archive_bridge as module


def _wait_for_qt(predicate):
    deadline = monotonic() + 5
    while not predicate():
        assert monotonic() < deadline, "Timed out waiting for the archive bridge"
        QTest.qWait(10)


@pytest.mark.parametrize("error", [None, WorkbenchRecordingArchiveError("database locked"), ValueError("unexpected")])
def test_archive_runs_off_gui_thread_and_delivers_once_on_gui(qt_app, caplog, error):
    entered, release = Event(), Event()
    writer_calls, results, timer_calls = [], [], []
    gui_thread = get_ident()
    completion = SimpleNamespace(request_id="request")
    recorded_on = date(2026, 11, 9)

    def save(*args):
        writer_calls.append((get_ident(), args))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release the writer")
        if error is not None:
            raise error

    bridge = module.WorkbenchRecordingArchiveBridge(SimpleNamespace(save=save))
    idle = QSignalSpy(bridge.idle)

    def receive(request_id, failure):
        results.append((get_ident(), request_id, failure, bridge.pending))

    assert not bridge.pending
    bridge.submit(completion, "电机", recorded_on, receive)
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="already pending"):
            bridge.submit(completion, "电机", recorded_on, receive)
        QTimer.singleShot(0, lambda: timer_calls.append(get_ident()))
        _wait_for_qt(lambda: timer_calls)
        assert timer_calls == [gui_thread]
        assert bridge.pending and not results and not idle
    finally:
        release.set()
        _wait_for_qt(lambda: not bridge.pending)

    (writer_thread, args), = writer_calls
    assert writer_thread != gui_thread
    assert args == (completion, "电机", recorded_on)
    assert results == [(gui_thread, "request", error, False)]
    assert len(idle) == 1
    traces = [record.exc_info for record in caplog.records if record.exc_info]
    if isinstance(error, ValueError):
        trace, = traces
        assert trace[1] is error and trace[2] is not None
    else:
        assert not traces


def test_thread_start_error_leaves_archive_bridge_idle(monkeypatch):
    error = RuntimeError("thread start failed")

    def fail_start(_thread):
        raise error

    bridge = module.WorkbenchRecordingArchiveBridge(SimpleNamespace())
    completion = SimpleNamespace(request_id="request")
    results = []
    monkeypatch.setattr(module.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed") as raised:
        bridge.submit(completion, "电机", date(2026, 11, 9), lambda *result: results.append(result))
    assert raised.value is error
    assert not bridge.pending
    assert not results


def test_consumer_failure_is_logged_and_bridge_signals_idle(qt_app, caplog):
    bridge = module.WorkbenchRecordingArchiveBridge(SimpleNamespace(save=lambda *_: None))
    idle = QSignalSpy(bridge.idle)
    completion = SimpleNamespace(request_id="request")
    error = ValueError("result consumer failed")

    def fail_consumer(_request_id, _failure):
        raise error

    bridge.submit(completion, "电机", date(2026, 11, 9), fail_consumer)
    _wait_for_qt(lambda: idle)
    assert not bridge.pending
    assert len(idle) == 1
    assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)
