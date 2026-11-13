"""Serial label work and shutdown without any Qt event loop."""

from dataclasses import FrozenInstanceError
from threading import Event, current_thread
from time import monotonic

import pytest

from base import audio_label_sync as sync_module
from base.audio_label_repository import AudioLabelPersistenceError, AudioLabelUpdate
from base.audio_label_sync import AudioLabelJob, AudioLabelSubmissionError, AudioLabelSyncService


def wait_until(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        Event().wait(0.002)
    assert predicate()


def test_job_is_frozen():
    job = AudioLabelJob("one", "/one.wav", "NG")
    with pytest.raises(FrozenInstanceError):
        job.label = "OK"


def test_lazy_single_non_daemon_worker_returns_immediately_and_drains_in_order():
    entered, release = Event(), Event()
    calls, completed = [], []

    class Repository:
        def update(self, path, label):
            calls.append((path, label, current_thread()))
            if len(calls) == 1:
                entered.set()
                assert release.wait(5)
            return AudioLabelUpdate(path, path, label)

    service = AudioLabelSyncService(Repository(), lambda *args: completed.append(args))
    assert service._thread is None
    assert not service.closed
    jobs = [AudioLabelJob(str(i), f"/{i}.wav", "OK") for i in range(3)]
    try:
        started = monotonic()
        service.submit(jobs[0])
        assert monotonic() - started < 0.5
        assert entered.wait(5)
        service.submit(jobs[1])
        service.submit(jobs[2])
        service.begin_shutdown()
        service.begin_shutdown()
        assert not service.closed
        assert not completed
        assert [call[0] for call in calls] == [jobs[0].wav_path]
        with pytest.raises(RuntimeError, match="clos"):
            service.submit(AudioLabelJob("late", "/late.wav", "NG"))
    finally:
        release.set()
        service.begin_shutdown()
        wait_until(lambda: service.closed)
    assert [call[:2] for call in calls] == [(job.wav_path, job.label) for job in jobs]
    assert [result[0] for result in completed] == jobs
    assert all(result[1] == AudioLabelUpdate(job.wav_path, job.wav_path, job.label)
               and result[2] is None for result, job in zip(completed, jobs))
    workers = {call[2] for call in calls}
    assert len(workers) == 1
    worker = workers.pop()
    assert worker is not current_thread()
    assert not worker.daemon
    assert not worker.is_alive()


@pytest.mark.parametrize("failure", [AudioLabelPersistenceError("missing WAV"), ValueError("bad writer")])
def test_failed_job_preserves_error_and_next_job_runs(failure, caplog):
    completed = []

    class Repository:
        def update(self, path, label):
            if path == "/bad.wav":
                raise failure
            return AudioLabelUpdate("good", path, label)

    service = AudioLabelSyncService(Repository(), lambda *args: completed.append(args))
    bad, good = AudioLabelJob("bad", "/bad.wav", "NG"), AudioLabelJob("good", "/good.wav", "OK")
    try:
        service.submit(bad)
        service.submit(good)
    finally:
        service.begin_shutdown()
        wait_until(lambda: service.closed)
    assert completed == [(bad, None, failure), (good, AudioLabelUpdate("good", "/good.wav", "OK"), None)]
    if isinstance(failure, AudioLabelPersistenceError):
        assert not caplog.records
    else:
        assert any(record.name == "core" and record.exc_info and "bad" in record.message
                   for record in caplog.records)


def test_no_work_repeated_shutdown_is_closed_without_starting_a_thread():
    service = AudioLabelSyncService(object(), lambda *args: pytest.fail("no job accepted"))
    service.begin_shutdown()
    service.begin_shutdown()
    assert service.closed
    assert service._thread is None


def test_start_failure_is_diagnostic_accepts_no_job_and_can_retry(monkeypatch):
    failure = RuntimeError("cannot start new thread")
    completed, calls = [], []

    class Repository:
        def update(self, path, label):
            calls.append(path)
            return AudioLabelUpdate("id", path, label)

    service = AudioLabelSyncService(Repository(), lambda *args: completed.append(args))
    job = AudioLabelJob("start-failed", "/bad.wav", "NG")

    def fail_start(thread):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(sync_module.Thread, "start", fail_start)
        with pytest.raises(AudioLabelSubmissionError, match="start-failed") as captured:
            service.submit(job)
    assert captured.value.__cause__ is failure
    assert not completed
    good = AudioLabelJob("good", "/good.wav", "OK")
    try:
        service.submit(good)
    finally:
        service.begin_shutdown()
        wait_until(lambda: service.closed)
    assert calls == [good.wav_path]
    assert [result[0] for result in completed] == [good]
