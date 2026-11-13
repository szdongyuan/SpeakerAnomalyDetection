"""Qt completion delivery, GUI responsiveness, and the shutdown barrier."""

from threading import Event, current_thread
from time import monotonic

import pytest
from PyQt5.QtCore import QThread, QTimer

from base import audio_label_sync as sync_module
from base.audio_label_repository import AudioLabelPersistenceError, AudioLabelUpdate
from base.audio_label_sync import AudioLabelJob, AudioLabelSubmissionError
from ui.audio_label_sync_bridge import AudioLabelSyncBridge


def spin_until(app, predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        app.processEvents()
        Event().wait(0.002)
    assert predicate()


def test_blocked_write_keeps_gui_alive_and_delivers_before_closed(ui_qapp):
    entered, release = Event(), Event()
    workers, signals, beats = [], [], []

    class Repository:
        def update(self, path, label):
            workers.append(current_thread())
            entered.set()
            assert release.wait(5)
            return AudioLabelUpdate(path, path, label)

    bridge = AudioLabelSyncBridge(Repository())
    bridge.changed.connect(lambda update: signals.append(("changed", update, QThread.currentThread(), bridge.pending)))
    bridge.closed.connect(lambda: signals.append(("closed", None, QThread.currentThread(), bridge.pending)))
    heartbeat = QTimer()
    heartbeat.setInterval(1)
    heartbeat.timeout.connect(lambda: beats.append(True))
    heartbeat.start()
    jobs = [AudioLabelJob(str(i), f"/{i}.wav", "OK") for i in range(2)]
    try:
        bridge.submit(jobs[0])
        assert entered.wait(5)
        bridge.submit(jobs[1])
        assert bridge.pending == 2
        started = monotonic()
        bridge.begin_shutdown()
        bridge.begin_shutdown()
        assert monotonic() - started < 0.5
        spin_until(ui_qapp, lambda: len(beats) >= 3)
        assert not bridge.is_closed
        assert not signals
        with pytest.raises(RuntimeError, match="clos"):
            bridge.submit(AudioLabelJob("late", "/late.wav", "NG"))
        assert bridge.pending == 2
    finally:
        release.set()
        bridge.begin_shutdown()
        spin_until(ui_qapp, lambda: bridge.is_closed)
        heartbeat.stop()
    assert [signal[0] for signal in signals] == ["changed", "changed", "closed"]
    assert [signal[1] for signal in signals[:2]] == [AudioLabelUpdate(job.wav_path, job.wav_path, job.label) for job in jobs]
    assert [signal[3] for signal in signals] == [1, 0, 0]
    assert all(signal[2] is ui_qapp.thread() for signal in signals)
    assert len(set(workers)) == 1 and not workers[0].is_alive()
    bridge.begin_shutdown()
    ui_qapp.processEvents()
    assert len(signals) == 3


@pytest.mark.parametrize("failure", [AudioLabelPersistenceError("locked"), ValueError("unexpected")])
def test_failures_deliver_on_gui_and_next_write_succeeds(ui_qapp, failure):
    received = []

    class Repository:
        def update(self, path, label):
            if path == "/bad.wav":
                raise failure
            return AudioLabelUpdate("good", path, label)

    bridge = AudioLabelSyncBridge(Repository())
    bridge.failed.connect(lambda job, error: received.append(("failed", job, error, QThread.currentThread())))
    bridge.changed.connect(lambda update: received.append(("changed", update, None, QThread.currentThread())))
    bridge.closed.connect(lambda: received.append(("closed", None, None, QThread.currentThread())))
    bad = AudioLabelJob("bad", "/bad.wav", "NG")
    try:
        bridge.submit(bad)
        bridge.submit(AudioLabelJob("good", "/good.wav", "OK"))
    finally:
        bridge.begin_shutdown()
        spin_until(ui_qapp, lambda: bridge.is_closed)
    assert [item[0] for item in received] == ["failed", "changed", "closed"]
    assert received[0][1] is bad and received[0][2] is failure
    assert received[1][1] == AudioLabelUpdate("good", "/good.wav", "OK")
    assert all(item[3] is ui_qapp.thread() for item in received)
    assert bridge.pending == 0


def test_worker_exit_alone_does_not_close_before_queued_completions(ui_qapp):
    written = Event()
    received = []

    class Repository:
        def update(self, path, label):
            written.set()
            return AudioLabelUpdate("id", path, label)

    bridge = AudioLabelSyncBridge(Repository())
    bridge.changed.connect(lambda update: received.append("changed"))
    bridge.closed.connect(lambda: received.append("closed"))
    try:
        bridge.submit(AudioLabelJob("job", "/job.wav", "OK"))
        bridge.begin_shutdown()
        assert written.wait(5)
        deadline = monotonic() + 5
        while not bridge._service.closed and monotonic() < deadline:
            Event().wait(0.002)
        assert bridge._service.closed
        assert bridge.pending == 1
        bridge._check_closed()
        assert not bridge.is_closed
        assert not received
    finally:
        bridge.begin_shutdown()
        spin_until(ui_qapp, lambda: bridge.is_closed)
    assert received == ["changed", "closed"]


def test_start_failure_is_queued_and_settles_pending(ui_qapp, monkeypatch):
    failure = RuntimeError("cannot start new thread")
    bridge = AudioLabelSyncBridge(object())
    received = []
    job = AudioLabelJob("start-failed", "/job.wav", "NG")
    bridge.failed.connect(lambda *args: received.append((args, QThread.currentThread(), bridge.pending)))
    bridge.changed.connect(lambda update: pytest.fail("failed start cannot change a label"))
    closed = []
    bridge.closed.connect(lambda: closed.append(True))

    def fail_start(thread):
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sync_module.Thread, "start", fail_start)
            bridge.submit(job)
        assert not received
        assert bridge.pending == 1
    finally:
        bridge.begin_shutdown()
        spin_until(ui_qapp, lambda: bridge.is_closed)
    (failed_job, error), handler_thread, pending = received[0]
    assert failed_job is job
    assert isinstance(error, AudioLabelSubmissionError)
    assert error.__cause__ is failure
    assert handler_thread is ui_qapp.thread() and pending == 0
    assert closed == [True]


def test_no_work_shutdown_is_idempotent(ui_qapp):
    bridge = AudioLabelSyncBridge(object())
    closed = []
    bridge.closed.connect(lambda: closed.append(True))
    assert bridge.pending == 0 and not bridge.is_closed
    bridge.begin_shutdown()
    bridge.begin_shutdown()
    spin_until(ui_qapp, lambda: bridge.is_closed)
    bridge.begin_shutdown()
    ui_qapp.processEvents()
    assert closed == [True]
    assert bridge._service._thread is None
