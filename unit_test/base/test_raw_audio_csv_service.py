import multiprocessing
import os
from pathlib import Path
import threading
import time

import pytest

from base.raw_audio_csv_protocol import CsvFailure, CsvResult
from base.raw_audio_csv_service import RawAudioCsvService
from unit_test.base.raw_audio_csv_fakes import (
    blocked_worker, fault_then_healthy_worker, stale_messages_worker,
    zip_phase_worker, two_temporary_exit_worker,
    wrong_version_worker, never_ready_worker, disconnected_live_worker,
    ignore_shutdown_worker,
    result_then_exit_worker,
    eof_then_exit_worker,
)
from unit_test.base.test_raw_audio_csv_worker import make_command


def eventually(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "supervisor did not reach expected state"


def submit(service, directory, task_id):
    request = make_command(directory, task_id).request
    token = service.reserve(request.recording_id).reservation
    assert service.commit(token, request) == "accepted"
    return request


@pytest.fixture
def services():
    active = []

    def create(**kwargs):
        service = RawAudioCsvService(**kwargs)
        active.append(service)
        return service

    yield create
    for service in active:
        service.begin_shutdown()
        assert service.closed.wait(15)
        service._thread.join(2)
        assert not service._thread.is_alive()


def test_lazy_spawn_reuse_and_callbacks_on_supervisor(services, tmp_path, monkeypatch):
    caller = threading.get_ident()
    context = multiprocessing.get_context("spawn")
    creations = []

    class CheckedConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            value = getattr(self.connection, name)
            if callable(value):
                def checked(*args, **kwargs):
                    assert threading.get_ident() != caller, name
                    return value(*args, **kwargs)
                return checked
            return value

    class CheckedProcess:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            value = getattr(self.process, name)
            if callable(value):
                def checked(*args, **kwargs):
                    assert threading.get_ident() != caller, name
                    return value(*args, **kwargs)
                return checked
            return value

    class CheckedContext:
        def Pipe(self):
            assert threading.get_ident() != caller
            parent, child = context.Pipe()
            return CheckedConnection(parent), child

        def Process(self, **kwargs):
            assert threading.get_ident() != caller
            creations.append(threading.get_ident())
            return CheckedProcess(context.Process(**kwargs))

    service = services(context=CheckedContext())
    events = []
    callback_threads = []
    subscription = service.subscribe(lambda event: (events.append(event), callback_threads.append(threading.get_ident())))
    token = service.reserve("unused").reservation
    assert not creations
    assert service.release_reservation(token)
    for task_id in ("one", "two"):
        request = make_command(tmp_path, task_id).request
        with monkeypatch.context() as patch:
            for name in ("open", "stat", "lstat", "unlink", "mkdir"):
                original = getattr(Path, name)

                def checked_file(*args, _method=original, **kwargs):
                    assert threading.get_ident() != caller, _method.__name__
                    return _method(*args, **kwargs)

                patch.setattr(Path, name, checked_file)
            reservation = service.reserve(request.recording_id).reservation
            assert service.commit(reservation, request) == "accepted"
            service.snapshot()
            service.paths_busy((request.wav_path,))
            permit = service.try_acquire_mutation((str(tmp_path / "other"),))
            assert service.release_mutation(permit)
        eventually(lambda: service.snapshot().outstanding == 0)
    terminals = [event.result for event in events if event.kind == "terminal"]
    assert len(terminals) == 2
    assert all(isinstance(result, CsvResult) for result in terminals)
    assert terminals[0].worker_pid == terminals[1].worker_pid != os.getpid()
    assert len(creations) == 1
    assert set(callback_threads) == {service._thread.ident}
    subscription.unsubscribe()
    service.begin_shutdown()
    assert service.closed.wait(5)
    service._thread.join(2)
    assert not service._thread.is_alive()


def test_one_inflight_fifo_and_sixteen_total_slots(services, tmp_path):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    service = services(worker_target=blocked_worker, worker_args=(gate,))
    events = []
    service.subscribe(events.append)
    try:
        for index in range(15):
            submit(service, tmp_path, str(index))
        token = service.reserve("recording-last").reservation
        eventually(lambda: any(event.kind == "started" for event in events))
        snapshot = service.snapshot()
        assert (snapshot.active, snapshot.queued, snapshot.reserved) == (1, 14, 1)
        assert service.reserve("overflow").status == "full"
        assert len([event for event in events if event.kind == "started"]) == 1
        assert service.release_reservation(token)
    finally:
        gate.set()
    eventually(lambda: service.snapshot().outstanding == 0)
    assert [event.result.task_id for event in events if event.kind == "terminal"] == [str(i) for i in range(15)]


def test_terminal_precedes_release_and_bad_subscriber_isolated(services, tmp_path, caplog):
    service = services()
    observations = []

    def broken(event):
        if event.kind == "terminal":
            raise RuntimeError("consumer failure")

    def observe(event):
        if event.kind == "terminal":
            observations.append((service.snapshot().active, service.paths_busy((event.task.request.wav_path,))))

    service.subscribe(broken)
    service.subscribe(observe)
    submit(service, tmp_path, "one")
    eventually(lambda: service.snapshot().outstanding == 0)
    assert observations == [(1, True)]
    assert "consumer failure" in caplog.text


@pytest.mark.parametrize("mode", ["acknowledged", "unacknowledged", "substituted", "published"])
def test_crash_cleanup_requires_ownership_and_next_task_restarts(services, tmp_path, mode):
    service = services(worker_target=fault_then_healthy_worker, worker_args=(mode,))
    events = []
    service.subscribe(events.append)
    neighbor = tmp_path / ".other.tmp"
    neighbor.write_bytes(b"untouched")
    crashed = submit(service, tmp_path, "crash")
    Path(crashed.csv_path).write_bytes(b"old csv")
    submit(service, tmp_path, "next")
    eventually(lambda: service.snapshot().outstanding == 0, timeout=5)
    terminals = [event.result for event in events if event.kind == "terminal"]
    assert len(terminals) == 2
    assert isinstance(terminals[0], CsvFailure)
    assert terminals[0].exception_type == "WorkerExit"
    assert "unconfirmed" in terminals[0].message
    assert isinstance(terminals[1], CsvResult)
    assert terminals[1].generation == terminals[0].generation + 1
    assert neighbor.read_bytes() == b"untouched"
    remnants = list(tmp_path.glob(".raw-csv-*.tmp"))
    release = next(event.task for event in events if event.kind == "released" and event.task.request.task_id == "crash")
    if mode in ("unacknowledged", "substituted"):
        assert len(remnants) == 1
        assert release.cleanup_diagnostics
        assert remnants[0].read_bytes() == (b"partial" if mode == "unacknowledged" else b"neighbor replacement")
    else:
        assert not remnants
    if mode == "published":
        assert not Path(crashed.csv_path).exists()
        assert Path(crashed.csv_path + ".zip").exists()
    else:
        assert Path(crashed.csv_path).read_bytes() == b"old csv"


def test_normal_failure_reuses_worker_and_stale_events_never_release_twice(services, tmp_path):
    service = services()
    events = []
    service.subscribe(events.append)
    bad = make_command(tmp_path, "bad").request
    Path(bad.wav_path).unlink()
    assert service.commit(service.reserve(bad.recording_id).reservation, bad) == "accepted"
    submit(service, tmp_path, "good")
    eventually(lambda: service.snapshot().outstanding == 0)
    terminals = [event.result for event in events if event.kind == "terminal"]
    assert isinstance(terminals[0], CsvFailure)
    assert isinstance(terminals[1], CsvResult)
    assert terminals[0].generation == terminals[1].generation == 1
    service2 = services(worker_target=stale_messages_worker)
    events2 = []
    service2.subscribe(events2.append)
    submit(service2, tmp_path, "stale")
    eventually(lambda: service2.snapshot().outstanding == 0)
    service2.begin_shutdown()
    assert service2.closed.wait(5)
    assert len([event for event in events2 if event.kind == "terminal"]) == 1
    assert len([event for event in events2 if event.kind == "released"]) == 1


@pytest.mark.parametrize("target", [wrong_version_worker, never_ready_worker])
def test_ready_failure_unavailable_releases_queue_but_preserves_reserved(
    services, tmp_path, target, monkeypatch,
):
    service = services(worker_target=target, ready_timeout=0.7)
    events = []
    service.subscribe(events.append)
    setup_complete = threading.Event()
    startup_waiting = threading.Event()
    start = service._start

    def start_after_setup():
        startup_waiting.set()
        assert setup_complete.wait(10), "test setup did not release worker startup"
        start()

    # The ready failure may arrive immediately. Admit both queued tasks before
    # spawning, so neither worker speed nor WAV creation controls this test.
    monkeypatch.setattr(service, "_start", start_after_setup)
    token = service.reserve("recording-late").reservation
    try:
        submit(service, tmp_path, "one")
        assert startup_waiting.wait(10)
        submit(service, tmp_path, "two")
        assert (service.snapshot().queued, service.snapshot().reserved) == (2, 1)
        setup_complete.set()
        eventually(lambda: service.snapshot().phase == "unavailable")
        eventually(lambda: service.snapshot().outstanding == 1)
        assert service.snapshot().reserved == 1
        assert service.reserve("new").status == "unavailable"
        assert len([event for event in events if event.kind == "terminal"]) == 2
        late = make_command(tmp_path, "late").request
        assert service.commit(token, late) == "unavailable"
        assert service.snapshot().outstanding == 0
        assert len([event for event in events if event.kind == "unavailable"]) == 1
    finally:
        setup_complete.set()
        service.release_reservation(token)


@pytest.mark.parametrize("failure_stage", ["start", "construct"])
def test_start_error_is_unavailable_without_retry_and_closes_handles(services, tmp_path, failure_stage):
    context = multiprocessing.get_context("spawn")
    handles = []
    attempts = []

    class FailedProcess:
        pid = None

        def start(self):
            attempts.append(1)
            raise OSError("spawn denied")

        def close(self):
            attempts.append("closed")

    class FailedContext:
        def Pipe(self):
            pair = context.Pipe()
            handles.extend(pair)
            return pair

        def Process(self, **kwargs):
            if failure_stage == "construct":
                attempts.append("construct")
                raise OSError("process constructor denied")
            return FailedProcess()

    service = services(context=FailedContext())
    submit(service, tmp_path, "one")
    eventually(lambda: service.snapshot().phase == "unavailable")
    eventually(lambda: service.snapshot().outstanding == 0)
    assert attempts == ([1, "closed"] if failure_stage == "start" else ["construct"])
    assert all(handle.closed for handle in handles)


def test_preexisting_temp_collision_is_never_deleted(services, tmp_path):
    service = services()
    events = []
    collision = []

    def observe(event):
        events.append(event)
        if event.kind == "dispatch" and event.task.request.task_id == "collision":
            path = Path(next(iter(service._temporaries)))
            path.write_bytes(b"other owner")
            collision.append(path)

    service.subscribe(observe)
    submit(service, tmp_path, "collision")
    submit(service, tmp_path, "next")
    eventually(lambda: service.snapshot().outstanding == 0)
    results = [event.result for event in events if event.kind == "terminal"]
    assert isinstance(results[0], CsvFailure)
    assert results[0].exception_type == "FileExistsError"
    assert isinstance(results[1], CsvResult)
    assert results[0].generation == results[1].generation
    assert collision[0].read_bytes() == b"other owner"


def test_eof_live_worker_retains_paths_until_confirmed_dead(services, tmp_path):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    processes = []

    class NoTermination:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def terminate(self):
            pass  # Model an OS that has not confirmed the termination request.

    class Context:
        Pipe = context.Pipe

        def Process(self, **kwargs):
            process = context.Process(**kwargs)
            processes.append(process)
            return NoTermination(process)

    service = services(context=Context(), worker_target=disconnected_live_worker, worker_args=(gate,), shutdown_timeout=0.05)
    events = []
    service.subscribe(events.append)
    try:
        request = submit(service, tmp_path, "one")
        eventually(lambda: service.snapshot().phase == "unavailable")
        assert service.snapshot().active == 1
        assert all(service.paths_busy((path,)) for path in (
            request.wav_path, request.csv_path, request.csv_path + ".zip"))
        assert service.try_acquire_mutation((request.wav_path,)) is None
        service.begin_shutdown()
        assert not service.closed.wait(0.1)
        assert len(processes) == 1
    finally:
        gate.set()
    eventually(lambda: service.snapshot().outstanding == 0)


@pytest.mark.parametrize("role", ["csv", "zip"])
def test_unlink_denied_is_diagnostic_without_capacity_leak(services, tmp_path, monkeypatch, role):
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path.name.startswith(f".raw-{role}-"):
            raise PermissionError("cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    service = services(worker_target=two_temporary_exit_worker, worker_args=(False, "acknowledged"))
    events = []
    service.subscribe(events.append)
    submit(service, tmp_path, "crash")
    eventually(lambda: service.snapshot().outstanding == 0)
    released = next(event.task for event in events if event.kind == "released")
    assert "cleanup denied" in str(released.cleanup_diagnostics)
    assert list(tmp_path.glob(f".raw-{role}-*.tmp"))


def test_idle_shutdown_deadline_terminates_and_confirms_no_orphan(services, tmp_path):
    service = services(worker_target=ignore_shutdown_worker, shutdown_timeout=0.05)
    events = []
    service.subscribe(events.append)
    submit(service, tmp_path, "one")
    eventually(lambda: service.snapshot().outstanding == 0)
    pid = next(event.result.worker_pid for event in events if event.kind == "terminal")
    service.begin_shutdown()
    assert service.closed.wait(2)
    assert service.snapshot().phase == "closed"
    assert service._process is None and service._control is None
    assert pid not in [child.pid for child in multiprocessing.active_children()]
    assert len([event for event in events if event.kind == "closed"]) == 1
    service._thread.join(2)
    assert events[-1].kind == "closed"


def test_drain_waits_for_busy_work_and_late_commit_or_cancel(services, tmp_path):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    service = services(worker_target=blocked_worker, worker_args=(gate,), ready_timeout=2, shutdown_timeout=0.05)
    events = []
    service.subscribe(events.append)
    late = make_command(tmp_path, "late").request
    late_token = service.reserve(late.recording_id).reservation
    cancel_token = service.reserve("cancel").reservation
    try:
        submit(service, tmp_path, "busy")
        eventually(lambda: any(event.kind == "started" for event in events))
        service.begin_shutdown()
        service.begin_shutdown()
        assert service.reserve("new").status == "closing"
        # Export execution has no deadline, even in drain. Both injected
        # shutdown and ready deadlines are irrelevant once work has started.
        assert not service.closed.wait(0.15)
        assert service.snapshot().active == 1
        assert service.commit(late_token, late) == "accepted"
        gate.set()
        eventually(lambda: service.snapshot().outstanding == 1)
        assert service.snapshot().reserved == 1
        assert not service.closed.wait(0.1)
        assert service.release_reservation(cancel_token)
        assert not service.release_reservation(cancel_token)
    finally:
        gate.set()
        service.release_reservation(late_token)
        service.release_reservation(cancel_token)
    assert service.closed.wait(5)
    assert [event.result.task_id for event in events if event.kind == "terminal"] == ["busy", "late"]


def test_deferred_mutation_runs_on_supervisor_and_drain_waits_for_permit(services, tmp_path):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    service = services(worker_target=blocked_worker, worker_args=(gate,))
    callbacks = []
    request = submit(service, tmp_path, "one")
    permit = service.try_acquire_mutation((str(tmp_path / "other.wav"),))
    try:
        assert service.defer_mutation((request.wav_path,), lambda: callbacks.append(threading.get_ident()))
        assert service.try_acquire_mutation((request.wav_path,)) is None
        service.begin_shutdown()
        gate.set()
        eventually(lambda: callbacks)
        assert callbacks == [service._thread.ident]
        assert not service.closed.wait(0.1)
    finally:
        gate.set()
        service.release_mutation(permit)
    assert service.closed.wait(5)


def test_empty_service_shutdown_never_creates_process(services):
    class NoProcesses:
        def Pipe(self):
            pytest.fail("empty service created pipe")

    service = services(context=NoProcesses())
    service.begin_shutdown()
    assert service.closed.wait(2)
    assert service.reserve("late").status == "closing"


def test_buffered_terminal_wins_over_simultaneous_process_exit(services, tmp_path):
    service = services(worker_target=result_then_exit_worker)
    events = []
    service.subscribe(events.append)
    submit(service, tmp_path, "one")
    service.begin_shutdown()
    assert service.closed.wait(5)
    results = [event.result for event in events if event.kind == "terminal"]
    assert len(results) == 1
    assert isinstance(results[0], CsvResult)


@pytest.mark.parametrize("role", ["csv", "zip"])
def test_reparse_temporary_is_preserved_even_with_matching_identity(services, tmp_path, monkeypatch, role):
    from types import SimpleNamespace
    import stat

    real_lstat = Path.lstat

    def reparse(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if path.name.startswith(f".raw-{role}-"):
            return SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino, st_mode=info.st_mode,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info

    monkeypatch.setattr(Path, "lstat", reparse)
    service = services(worker_target=two_temporary_exit_worker, worker_args=(False, "acknowledged"))
    events = []
    service.subscribe(events.append)
    submit(service, tmp_path, "crash")
    eventually(lambda: service.snapshot().outstanding == 0)
    assert list(tmp_path.glob(f".raw-{role}-*.tmp"))
    released = next(event.task for event in events if event.kind == "released")
    assert "ownership unconfirmed" in str(released.cleanup_diagnostics)


def test_closed_notification_observes_supervisor_already_stopped(services):
    service = services()
    observed = []

    def observe(event):
        if event.kind == "closed":
            observed.append((service._thread.is_alive(), service._process, service._control))

    service.subscribe(observe)
    service.begin_shutdown()
    assert service.closed.wait(5)
    assert not service._thread.is_alive()
    assert observed == [(False, None, None)]


def test_eof_terminal_followed_by_confirmed_exit_diagnostic(services, tmp_path):
    context = multiprocessing.get_context("spawn")
    exit_gate = context.Event()
    service = services(worker_target=eof_then_exit_worker, worker_args=(exit_gate,))
    events = []
    service.subscribe(events.append)
    try:
        submit(service, tmp_path, "one")
        eventually(lambda: any(event.kind == "terminal" for event in events))
        assert service.snapshot().active == 1
        assert not any(event.kind == "released" for event in events)
    finally:
        exit_gate.set()
    eventually(lambda: service.snapshot().outstanding == 0)
    assert len([event for event in events if event.kind == "terminal"]) == 1
    exits = [event for event in events if event.kind == "worker_exit"]
    assert len(exits) == 1
    assert exits[0].task.request.task_id == "one"
    assert exits[0].task.generation == 1
    assert "exit 23" in exits[0].detail


def test_deferred_mutation_failure_does_not_disable_unrelated_exports(services, tmp_path):
    service = services()
    events = []
    callback_returned = threading.Event()
    cleanup_path = str(tmp_path / "unrelated.wav")
    service.subscribe(events.append)

    def fail_cleanup():
        callback_returned.set()
        raise PermissionError("simulated unrelated file deletion failure")

    assert service.defer_mutation((cleanup_path,), fail_cleanup)
    assert callback_returned.wait(5)
    eventually(lambda: any(event.kind in ("mutation_failed", "unavailable") for event in events))
    assert service.snapshot().phase == "open"
    diagnostics = [event for event in events if event.kind == "mutation_failed"]
    assert len(diagnostics) == 1
    assert cleanup_path in diagnostics[0].detail
    assert "PermissionError" in diagnostics[0].detail
    assert "simulated unrelated file deletion failure" in diagnostics[0].detail
    permit = service.try_acquire_mutation((cleanup_path,))
    assert permit is not None
    assert service.release_mutation(permit)
    submit(service, tmp_path, "unrelated-export")
    service.begin_shutdown()
    assert service.closed.wait(5)
    results = [event.result for event in events if event.kind == "terminal"]
    assert len(results) == 1 and isinstance(results[0], CsvResult)


def test_internal_supervisor_error_still_marks_service_unavailable(services, monkeypatch):
    service = services()
    original = service._ledger.run_deferred_mutations

    def internal_failure():
        monkeypatch.setattr(service._ledger, "run_deferred_mutations", original)
        raise RuntimeError("internal ledger invariant failure")

    monkeypatch.setattr(service._ledger, "run_deferred_mutations", internal_failure)
    eventually(lambda: service.snapshot().phase == "unavailable")
    assert service.reserve("new").status == "unavailable"


def test_terminal_messages_validate_pid_and_archive_and_forward_warnings(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from base.raw_audio_csv_tasks import CsvTaskLedger
    service = object.__new__(RawAudioCsvService)
    service._ledger = CsvTaskLedger()
    request = make_command(tmp_path, "identity").request
    service._ledger.commit(service._ledger.reserve(request.recording_id).reservation, request)
    service._active = service._ledger.dispatch_next(3)
    service._process = SimpleNamespace(pid=123)
    service._temporaries = {"csv.tmp": (1, 2), "zip.tmp": (3, 4)}
    emitted = []
    service._emit = lambda kind, **kwargs: emitted.append((kind, kwargs))
    result = CsvResult(task_id=request.task_id, generation=3,
        csv_path=service._active.request.csv_path, worker_pid=123,
        elapsed_seconds=1, frames=3, bytes_written=40,
        archive_path=service._active.request.csv_path + ".zip", archive_bytes=20,
        csv_retained=True, cleanup_diagnostics=("CSV locked",), csv_export_seconds=0.2,
        zip_write_seconds=0.3, zip_verify_seconds=0.2, zip_publish_seconds=0.1,
        csv_cleanup_seconds=0.1)
    for message in (replace(result, worker_pid=999), replace(result, archive_path="other.zip"),
                    replace(result, csv_path="other.csv"), replace(result, generation=2),
                    replace(result, task_id="other"),
                    CsvFailure(request.task_id, 3, "zip_write", "OSError", "bad", worker_pid=999),
                    CsvFailure(request.task_id, 3, "zip_write", "OSError", "bad")):
        service._control = SimpleNamespace(recv=lambda: message)
        service._receive()
        assert service.snapshot().active == 1
        assert emitted == []
    service._control = SimpleNamespace(recv=lambda: result)
    service._receive()
    assert service.snapshot().outstanding == 0
    assert emitted[-1][1]["task"].cleanup_diagnostics == ("CSV locked",)
    assert service._temporaries == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_both_temporary_acknowledgments_are_independent_and_exact(tmp_path, reverse):
    from types import SimpleNamespace
    from base.raw_audio_csv_tasks import CsvTaskLedger
    service = object.__new__(RawAudioCsvService)
    service._ledger = CsvTaskLedger()
    request = make_command(tmp_path, "temps").request
    service._ledger.commit(service._ledger.reserve(request.recording_id).reservation, request)
    service._active = service._ledger.dispatch_next(3)
    service._temporaries = {"csv.tmp": None, "zip.tmp": None}
    valid = [("temporary_owned", request.task_id, 3, "csv.tmp", (1, 2)),
             ("temporary_owned", request.task_id, 3, "zip.tmp", (3, 4))]
    invalid = [("temporary_owned", request.task_id, 2, "csv.tmp", (8, 9)),
               ("temporary_owned", "other", 3, "csv.tmp", (8, 9)),
               ("temporary_owned", request.task_id, 3, "other.tmp", (8, 9)),
               ("temporary_owned", request.task_id, 3, "csv.tmp", (True, 9))]
    for message in invalid + (list(reversed(valid)) if reverse else valid):
        service._control = SimpleNamespace(recv=lambda: message)
        service._receive()
    assert service._temporaries == {"csv.tmp": (1, 2), "zip.tmp": (3, 4)}
    # Conflicting later acknowledgments must not replace the first owned identity.
    message = ("temporary_owned", request.task_id, 3, "zip.tmp", (8, 9))
    service._receive()
    assert service._temporaries["zip.tmp"] == (3, 4)


@pytest.mark.parametrize("stage", ["zip_write", "zip_verify"])
def test_zip_phase_holds_sixteen_slots_paths_and_shutdown(services, tmp_path, stage):
    context = multiprocessing.get_context("spawn")
    entered, gate = context.Event(), context.Event()
    service = services(worker_target=zip_phase_worker, worker_args=(stage, entered, gate))
    events, callbacks = [], []
    service.subscribe(events.append)
    try:
        first = submit(service, tmp_path, "0")
        for index in range(1, 15):
            submit(service, tmp_path, str(index))
        token = service.reserve("last").reservation
        assert entered.wait(10)
        assert Path(first.csv_path).exists()
        assert not Path(first.csv_path + ".zip").exists()
        assert (service.snapshot().active, service.snapshot().queued, service.snapshot().reserved) == (1, 14, 1)
        assert service.reserve("overflow").status == "full"
        for path in (first.wav_path, first.csv_path, first.csv_path + ".zip"):
            assert service.paths_busy((path,))
            assert service.try_acquire_mutation((path,)) is None
        assert service.defer_mutation((first.csv_path + ".zip",), lambda: callbacks.append("done"))
        assert callbacks == []
        service.begin_shutdown()
        assert not service.closed.wait(0.1)
        assert not any(event.kind == "terminal" for event in events)
        assert service.release_reservation(token)
    finally:
        gate.set()
    assert service.closed.wait(15)
    results = [event.result for event in events if event.kind == "terminal"]
    assert [result.task_id for result in results] == [str(i) for i in range(15)]
    assert all(isinstance(result, CsvResult) for result in results)
    assert len({result.worker_pid for result in results}) == 1
    assert callbacks == ["done"]
    assert service._process is None and service._control is None
    assert service._temporaries == {}
    assert not service._thread.is_alive()


@pytest.mark.parametrize("mode", ["warning", "failure"])
def test_spawn_zip_warning_or_failure_releases_and_worker_is_reusable(services, tmp_path, mode):
    context = multiprocessing.get_context("spawn")
    entered, gate = context.Event(), context.Event()
    gate.set()
    service = services(worker_target=zip_phase_worker,
                       worker_args=("zip_verify", entered, gate, mode))
    events = []
    service.subscribe(events.append)
    first = submit(service, tmp_path, "first")
    submit(service, tmp_path, "next")
    service.begin_shutdown()
    assert service.closed.wait(15)
    results = [event.result for event in events if event.kind == "terminal"]
    assert len(results) == 2 and isinstance(results[1], CsvResult)
    assert results[0].worker_pid == results[1].worker_pid
    if mode == "warning":
        assert isinstance(results[0], CsvResult) and results[0].csv_retained
        assert Path(first.csv_path + ".zip").exists()
        released = [event.task for event in events if event.kind == "released"]
        assert released[0].cleanup_diagnostics == results[0].cleanup_diagnostics
        assert "locked in child" in str(released[0].cleanup_diagnostics)
    else:
        assert isinstance(results[0], CsvFailure) and results[0].stage == "zip_verify"
        assert not Path(first.csv_path + ".zip").exists()
    assert Path(first.csv_path).exists()
    assert service.snapshot().outstanding == 0


@pytest.mark.parametrize("stage", ["zip_write", "zip_verify", "zip_publish", "csv_cleanup"])
def test_real_zip_phase_death_preserves_finals_and_cleans_owned_temp(services, tmp_path, stage):
    context = multiprocessing.get_context("spawn")
    entered, gate = context.Event(), context.Event()
    service = services(worker_target=zip_phase_worker,
                       worker_args=(stage, entered, gate, "death"))
    events = []
    service.subscribe(events.append)
    try:
        request = submit(service, tmp_path, "death")
        assert entered.wait(10)
        assert service.snapshot().active == 1
    finally:
        gate.set()
    service.begin_shutdown()
    assert service.closed.wait(15)
    results = [event.result for event in events if event.kind == "terminal"]
    assert len(results) == 1 and isinstance(results[0], CsvFailure)
    assert Path(request.csv_path).exists()
    assert Path(request.csv_path + ".zip").exists() == (stage == "csv_cleanup")
    assert not list(tmp_path.glob(".raw-*.tmp"))
    assert service.snapshot().outstanding == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mode", ["acknowledged", "unacknowledged", "csv_published", "both_published", "zip_substituted"])
def test_two_temporary_death_cleanup_preserves_final_and_neighbor_files(services, tmp_path, reverse, mode):
    service = services(worker_target=two_temporary_exit_worker, worker_args=(reverse, mode))
    events = []
    service.subscribe(events.append)
    neighbor = tmp_path / ".neighbor.tmp"
    neighbor.write_bytes(b"neighbor")
    request = submit(service, tmp_path, "death")
    service.begin_shutdown()
    assert service.closed.wait(10)
    residuals = list(tmp_path.glob(".raw-*.tmp"))
    released = next(event.task for event in events if event.kind == "released")
    if mode == "unacknowledged":
        assert len(residuals) == 2 and len(released.cleanup_diagnostics) == 2
    elif mode == "zip_substituted":
        assert len(residuals) == 1 and residuals[0].read_bytes() == b"neighbor replacement"
        assert len(released.cleanup_diagnostics) == 1
    else:
        assert residuals == [] and released.cleanup_diagnostics == ()
    assert Path(request.csv_path).exists() == (mode in ("csv_published", "both_published"))
    assert Path(request.csv_path + ".zip").exists() == (mode == "both_published")
    assert neighbor.read_bytes() == b"neighbor"
    assert service._temporaries == {}


def test_long_csv_name_uses_short_distinct_controlled_temporary_names(services, tmp_path):
    from dataclasses import replace
    service = services()
    events, temporaries = [], []
    def observe(event):
        events.append(event)
        if event.kind == "dispatch":
            temporaries.extend(service._temporaries)
    service.subscribe(observe)
    request = make_command(tmp_path, "long").request
    request = replace(request, csv_path=str(tmp_path / ("x" * 220 + ".csv")))
    assert service.commit(service.reserve(request.recording_id).reservation, request) == "accepted"
    service.begin_shutdown()
    assert service.closed.wait(10)
    results = [event.result for event in events if event.kind == "terminal"]
    assert len(results) == 1 and isinstance(results[0], CsvResult)
    assert len(temporaries) == len(set(temporaries)) == 2
    assert all(len(Path(path).name) < 64 for path in temporaries)
    assert Path(request.csv_path + ".zip").exists()
