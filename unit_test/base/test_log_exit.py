"""Bounded forced-exit coordination, including actual spawn/terminate boundaries."""
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from unit_test.base.log_exit_process_fakes import tail


def reap(process):
    if process.pid is not None:
        if process.is_alive():
            process.kill()
        process.join(3)
    process.close()


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_real_self_exit_preserves_tail_metadata_and_code(tmp_path, count):
    process = multiprocessing.get_context("spawn").Process(target=tail, args=(str(tmp_path), count))
    process.start()
    try:
        process.join(8)
        assert process.exitcode == 23
        text = (tmp_path / "main.log").read_text()
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        for index, record in enumerate(metadata):
            assert f"tail={index}|log_exit_process_fakes.py:{record['line']}|" in text
            assert f"|{record['created']:.9f}" in text
    finally:
        reap(process)


def test_parent_drains_blocked_business_before_real_terminate(tmp_path):
    from base.log_exit import ProcessLogDrain, run_with_log_drain

    context = multiprocessing.get_context("spawn")
    drain = ProcessLogDrain.create(context)
    ready, blocked = context.Event(), context.Event()
    process = context.Process(target=run_with_log_drain,
                              args=(tail, (str(tmp_path), 3, ready, blocked), drain.child_endpoint))
    process.start()
    try:
        assert ready.wait(5)
        drain.begin("test retirement")
        result = drain.wait()
        assert result.status == "drained"
        assert result.stats["pending"] == 0
        assert process.is_alive()
        assert not (tmp_path / "returned").exists()
        process.terminate()
        process.join(3)
        assert process.exitcode != 0
        assert (tmp_path / "main.log").read_text().count("|INFO|tail=") == 3
    finally:
        reap(process)
        drain.close()


def test_begin_and_poll_are_nonblocking_and_one_shot_with_injected_clock():
    from base.log_exit import ProcessLogDrain
    now = [100.0]
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"), clock=lambda: now[0])
    drain.begin("first", timeout=0.2)
    now[0] += 0.1
    drain.begin("second", timeout=2)
    assert drain.poll() is None
    now[0] += 0.11
    result = drain.poll()
    assert result.status == "timeout" and result.stats is None
    assert drain.reason == "first"
    assert drain.poll() is result
    drain.close()


@pytest.mark.parametrize("failure", [RuntimeError, OSError])
def test_drain_thread_start_failure_preserves_hard_exit(monkeypatch, failure):
    from base import log_exit
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(failure("no thread")))
    exit_codes = []
    monkeypatch.setattr(log_exit.os, "_exit", exit_codes.append)
    log_exit.exit_with_log_drain(42, timeout=0.01)
    assert exit_codes == [42]
    assert log_exit._ProcessBinding.cycle.result.status == "timeout"


@pytest.mark.parametrize("payload", [
    {"status": "drained", "stats": {"pending": 1, "consumer_alive": True}},
    {"status": "drained", "stats": None},
    {"status": "drained", "stats": {"pending": 0, "consumer_alive": False, "write_errors": 1}},
])
def test_malformed_success_is_not_accepted(payload):
    from base.log_exit import ProcessLogDrain, _write
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    endpoint = drain.child_endpoint
    _write(endpoint.result, endpoint.token, **payload)
    drain.begin("invalid result", timeout=0.01)
    assert drain.wait().status == "timeout"
    drain.close()


@pytest.mark.parametrize("mode", ["stale", "partial", "invalid-json"])
def test_stale_or_partial_mailbox_never_waits_on_peer(mode):
    from base.log_exit import ProcessLogDrain, _write
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    endpoint = drain.child_endpoint
    if mode == "stale":
        _write(endpoint.result, "old-child-token", status="no-runtime")
    else:
        endpoint.result[1:5] = b"oops"
        endpoint.result[0] = int(mode == "invalid-json")
    drain.begin("bad peer", timeout=0.01)
    assert drain.wait().status == "timeout"
    drain.close()


def test_dead_child_and_unused_endpoint_cleanup():
    from base.log_exit import ProcessLogDrain, run_with_log_drain
    from unit_test.base.log_exit_process_fakes import abrupt
    context = multiprocessing.get_context("spawn")
    drain = ProcessLogDrain.create(context)
    child = context.Process(target=run_with_log_drain, args=(abrupt, (), drain.child_endpoint))
    child.start()
    try:
        child.join(4)
        assert child.exitcode == 31
        assert drain.poll(already_dead=True).status == "already-dead"
    finally:
        reap(child)
        drain.close()
    unused = ProcessLogDrain.create(context)
    unused.close()
    unused.close()
    assert unused.child_endpoint is None


def test_no_runtime_drain_creates_no_file(tmp_path):
    from base.log_exit import ProcessLogDrain, run_with_log_drain
    from unit_test.base.log_exit_process_fakes import idle
    context = multiprocessing.get_context("spawn")
    drain = ProcessLogDrain.create(context)
    ready, release = context.Event(), context.Event()
    child = context.Process(target=run_with_log_drain, args=(idle, (ready, release), drain.child_endpoint))
    child.start()
    try:
        assert ready.wait(4)
        drain.begin("empty")
        assert drain.wait().status == "no-runtime"
        assert not list(tmp_path.iterdir())
    finally:
        reap(child)
        drain.close()


@pytest.mark.parametrize("mode", ["parent", "self", "race"])
@pytest.mark.parametrize("budget", [0.08, 0.3])
def test_real_locked_sink_reaches_actual_exit_in_one_budget(tmp_path, mode, budget):
    result = subprocess.run([sys.executable, "-m", "unit_test.base.log_exit_process_fakes",
                             mode, str(tmp_path), str(budget)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads((tmp_path / "outcome.json").read_text())
    assert outcome["status"] == "timeout"
    assert outcome["elapsed"] < budget + 0.8


def test_normal_return_import_and_reacquisition_leave_no_control_threads(tmp_path):
    result = subprocess.run([sys.executable, "-m", "unit_test.base.log_exit_process_fakes",
                             "lifecycle", str(tmp_path)], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "main.log").read_text().count("cycle=") == 4


def wait_cycle(cycle):
    deadline = time.monotonic() + 3
    while cycle.poll() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cycle.result is not None
    if cycle.worker.ident is not None:
        cycle.worker.join(1)
    return cycle.result


@pytest.mark.parametrize("failure", ["format", "write", "unicode-write"])
def test_real_sink_errors_are_not_reported_as_written(tmp_path, monkeypatch, failure):
    from base import log_manager, log_exit
    from unit_test.logging_test_support import isolated_project_logger
    monkeypatch.setattr(log_manager.LogManager, "_forced_exit", False, raising=False)
    with isolated_project_logger(tmp_path, monkeypatch):
        if failure == "format":
            log_manager.DEFAULT_LOG["log_format"] = "%(nonexistent)s"
        else:
            def broken_write(sink, message):
                raise OSError(("🙂" if failure == "unicode-write" else "x") * 10000)
            monkeypatch.setattr(log_manager._BatchFileHandler, "do_write", broken_write)
        log_manager.LogManager.set_log_handler("core").info("error-probe")
        drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
        cycle = log_exit._ExitCycle(drain.child_endpoint)
        cycle.begin(1)
        result = wait_cycle(cycle)
        assert result.status == "drained-with-errors"
        assert result.stats["accepted"] == result.stats["write_errors"] == 1
        assert result.stats["written"] == result.stats["pending"] == 0
        assert len(result.stats["last_error"]) <= 512
        assert drain.poll() == result
        drain.close()


def test_closed_runtime_and_concurrent_begin_share_one_consumer(tmp_path, monkeypatch):
    from base import log_manager, log_exit
    from unit_test.logging_test_support import isolated_project_logger
    monkeypatch.setattr(log_manager.LogManager, "_forced_exit", False, raising=False)
    with isolated_project_logger(tmp_path, monkeypatch):
        manager = log_manager.LogManager
        manager.set_log_handler("core").info("closed")
        assert manager.shutdown_all(1)
        runtime = manager._runtime
        cycle = log_exit._ExitCycle()
        starts = []
        original = threading.Thread.start

        def start(thread):
            if thread.name == "project-log-drain":
                starts.append(thread)
            return original(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        threads = [threading.Thread(target=cycle.begin, args=(0.3,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(1)
        assert wait_cycle(cycle).status == "drained"
        first_deadline = cycle.deadline
        cycle.begin(2)
        assert cycle.deadline == first_deadline
        assert len(starts) == 1
        assert manager._runtime is runtime
        with pytest.raises(RuntimeError, match="forced"):
            manager.set_log_handler("core")


def test_self_deadline_precedes_later_parent_request():
    from base.log_exit import ProcessLogDrain, _write
    now = [100.0]
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"), clock=lambda: now[0])
    endpoint = drain.child_endpoint
    _write(endpoint.started, endpoint.token, deadline=100.1)
    drain.begin("later parent", timeout=2)
    now[0] = 100.11
    assert drain.poll().status == "timeout"
    drain.close()


def test_wrapper_start_failure_releases_binding(monkeypatch):
    from base import log_exit
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    def fail_start(thread):
        raise RuntimeError("thread creation failed")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    called = []
    with pytest.raises(RuntimeError, match="thread creation failed"):
        log_exit.run_with_log_drain(lambda: called.append(True), (), drain.child_endpoint)
    assert not called
    assert log_exit._ProcessBinding.cycle is None
    drain.close()


def test_forced_exit_without_runtime_does_not_create_one(monkeypatch, tmp_path):
    from base import log_exit, log_manager
    monkeypatch.setattr(log_manager.LogManager, "_runtime", None)
    monkeypatch.setattr(log_manager.LogManager, "_forced_exit", False, raising=False)
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    config = {"log_name": str(tmp_path / "unexpected.log")}
    monkeypatch.setattr(log_manager, "DEFAULT_LOG", config)
    monkeypatch.setattr(log_manager, "LOG_MAPPING", {"core": config})
    exit_codes = []
    monkeypatch.setattr(log_exit.os, "_exit", exit_codes.append)
    log_exit.exit_with_log_drain(7)
    assert exit_codes == [7]
    assert log_exit._ProcessBinding.cycle.result.status == "no-runtime"
    assert log_manager.LogManager._runtime is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="forced"):
        log_manager.LogManager.set_log_handler("core")


def test_forced_seal_includes_runtime_initialization_already_in_flight(tmp_path, monkeypatch):
    from base import log_exit, log_manager
    from unit_test.logging_test_support import isolated_project_logger
    monkeypatch.setattr(log_manager.LogManager, "_forced_exit", False, raising=False)
    entered, release = threading.Event(), threading.Event()
    register = log_manager._LogDispatcher.register

    def gated_register(runtime, name, info):
        entered.set()
        assert release.wait(2)
        return register(runtime, name, info)

    monkeypatch.setattr(log_manager._LogDispatcher, "register", gated_register)
    with isolated_project_logger(tmp_path, monkeypatch):
        producer = threading.Thread(target=log_manager.LogManager.set_log_handler, args=("core",))
        producer.start()
        try:
            assert entered.wait(1)
            cycle = log_exit._ExitCycle()
            cycle.begin(1)
            release.set()
            producer.join(1)
            assert wait_cycle(cycle).status == "drained"
            runtime = log_manager.LogManager._runtime
            assert runtime._closing and not runtime._thread.is_alive()
            with pytest.raises(RuntimeError, match="forced"):
                log_manager.LogManager.set_log_handler("core")
        finally:
            release.set()
            producer.join(2)


@pytest.mark.parametrize("corruption", ["status-list", "deep-json", "huge-deadline"])
def test_corrupt_committed_mailbox_cannot_break_deadline(corruption):
    from base.log_exit import ProcessLogDrain, _write
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    endpoint = drain.child_endpoint
    if corruption == "status-list":
        _write(endpoint.result, endpoint.token, status=[])
    elif corruption == "huge-deadline":
        _write(endpoint.started, endpoint.token, deadline=10 ** 1000)
    else:
        payload = b"[" * 1500 + b"]" * 1500
        endpoint.result[1:len(payload) + 1] = payload
        endpoint.result[0] = 1
    drain.begin("corrupt message", timeout=0.01)
    assert drain.wait().status == "timeout"
    drain.close()


def test_started_forced_cycle_survives_business_target_return(monkeypatch):
    from base import log_exit, log_manager
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    monkeypatch.setattr(log_manager.LogManager, "_runtime", None)
    monkeypatch.setattr(log_manager.LogManager, "_forced_exit", False)
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    cycles = []

    def target():
        cycle = log_exit._ProcessBinding.cycle
        cycles.append(cycle)
        cycle.begin(0.03)

    log_exit.run_with_log_drain(target, (), drain.child_endpoint)
    assert log_exit._ProcessBinding.cycle is cycles[0]
    first_deadline = cycles[0].deadline
    wait_cycle(cycles[0])
    exit_codes = []
    monkeypatch.setattr(log_exit.os, "_exit", exit_codes.append)
    log_exit.exit_with_log_drain(24)
    assert exit_codes == [24]
    assert cycles[0].deadline == first_deadline
    drain.close()


@pytest.mark.parametrize("parent_first", [False, True])
def test_parent_and_self_deadlines_share_minimum_in_both_orders(monkeypatch, parent_first):
    from base.log_exit import ProcessLogDrain, _ExitCycle, _deadline
    monkeypatch.setattr(_ExitCycle, "_drain", lambda self: None)
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    cycle = _ExitCycle(drain.child_endpoint)
    if parent_first:
        drain.begin("parent first", timeout=0.1)
        cycle.begin(2)
        assert cycle.deadline == drain._deadline
    else:
        cycle.begin(0.1)
        drain.begin("self first", timeout=2)
        assert drain._deadline == cycle.deadline
    assert _deadline(drain.child_endpoint.request, drain.child_endpoint.token) == cycle.deadline
    cycle.worker.join(1)
    drain.close()


def test_local_completion_is_visible_only_after_mailbox_commits(monkeypatch):
    from base import log_exit
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    cycle = log_exit._ExitCycle(drain.child_endpoint)
    entered, release = threading.Event(), threading.Event()
    write = log_exit._write

    def gated_write(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        write(*args, **kwargs)

    monkeypatch.setattr(log_exit, "_write", gated_write)
    finisher = threading.Thread(target=cycle.finish, args=(log_exit.DrainResult("no-runtime"),))
    finisher.start()
    try:
        assert entered.wait(1)
        assert cycle.poll() is None
    finally:
        release.set()
        finisher.join(2)
    assert drain.poll().status == "no-runtime"
    drain.close()


def test_success_requires_consistent_accepted_and_completed_counters():
    from base.log_exit import ProcessLogDrain, _write
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    endpoint = drain.child_endpoint
    stats = dict(accepted=1, written=0, pending=0, write_errors=0, snapshot_errors=0,
                 dropped_full=0, dropped_closed=0, consumer_alive=False, last_error=None)
    _write(endpoint.result, endpoint.token, status="drained", stats=stats)
    drain.begin("inconsistent result", timeout=0.01)
    assert drain.wait().status == "timeout"
    drain.close()


def test_timeout_diagnostics_fit_even_when_both_fields_are_astral_unicode():
    from base.log_exit import ProcessLogDrain, _ExitCycle, DrainResult
    drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    stats = dict(accepted=1, written=0, pending=1, write_errors=0, snapshot_errors=0,
                 dropped_full=0, dropped_closed=0, consumer_alive=True, last_error="🙂" * 512)
    cycle = _ExitCycle(drain.child_endpoint)
    result = cycle.finish(DrainResult("timeout", stats, "🙂" * 512))
    assert drain.poll() == result
    assert result.stats["pending"] == 1
    assert result.stats["last_error"] and result.detail
    drain.close()


def test_request_read_before_target_return_retains_cycle_for_later_self_exit(monkeypatch):
    from base import log_exit
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    monkeypatch.setattr(log_exit._ExitCycle, "_drain",
                        lambda cycle: cycle.finish(log_exit.DrainResult("no-runtime")))
    entered, release = threading.Event(), threading.Event()
    cycles, listeners = [], []
    original_begin = log_exit._ExitCycle.begin

    def gated_begin(cycle, timeout):
        if threading.current_thread().name == "project-log-control":
            cycles.append(cycle)
            listeners.append(threading.current_thread())
            entered.set()
            assert release.wait(2)
        return original_begin(cycle, timeout)

    monkeypatch.setattr(log_exit._ExitCycle, "begin", gated_begin)
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
    drain.begin("request preceding target return", timeout=0.3)

    def target():
        assert entered.wait(1)

    try:
        log_exit.run_with_log_drain(target, (), drain.child_endpoint)
        assert log_exit._ProcessBinding.cycle is cycles[0]
    finally:
        release.set()
        for listener in listeners:
            listener.join(2)
        for cycle in cycles:
            wait_cycle(cycle)
        drain.close()
    cycle = cycles[0]
    original_worker = cycle.worker
    original_deadline = cycle.deadline
    exit_codes = []
    monkeypatch.setattr(log_exit.os, "_exit", exit_codes.append)
    log_exit.exit_with_log_drain(24)
    assert exit_codes == [24]
    assert log_exit._ProcessBinding.cycle is cycle
    assert cycle.worker is original_worker
    assert cycle.deadline == original_deadline == drain._deadline


def test_detached_listener_cannot_begin_a_late_request(monkeypatch):
    from base import log_exit
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    monkeypatch.setattr(log_exit._ExitCycle, "_drain",
                        lambda cycle: cycle.finish(log_exit.DrainResult("no-runtime")))
    entered, release = threading.Event(), threading.Event()
    listeners, cycles = [], []
    original_deadline = log_exit._deadline

    def gated_read(mailbox, token):
        if threading.current_thread().name == "project-log-control":
            listeners.append(threading.current_thread())
            entered.set()
            assert release.wait(2)
        return original_deadline(mailbox, token)

    monkeypatch.setattr(log_exit, "_deadline", gated_read)
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))

    def target():
        cycles.append(log_exit._ProcessBinding.cycle)
        assert entered.wait(1)

    try:
        log_exit.run_with_log_drain(target, (), drain.child_endpoint)
        assert log_exit._ProcessBinding.cycle is None
        drain.begin("request after detach", timeout=0.3)
    finally:
        release.set()
        for listener in listeners[:]:
            listener.join(2)
        drain.close()
    assert cycles[0].deadline is None
    assert cycles[0].worker is None
    assert log_exit._ProcessBinding.cycle is None


def test_self_exit_selected_before_target_return_keeps_its_cycle(monkeypatch):
    from base import log_exit
    monkeypatch.setattr(log_exit._ProcessBinding, "cycle", None)
    monkeypatch.setattr(log_exit._ExitCycle, "_drain",
                        lambda cycle: cycle.finish(log_exit.DrainResult("no-runtime")))
    entered, release = threading.Event(), threading.Event()
    original_begin = log_exit._ExitCycle.begin
    cycles, errors, exit_codes = [], [], []
    monkeypatch.setattr(log_exit.os, "_exit", exit_codes.append)

    def gated_begin(cycle, timeout):
        if threading.current_thread().name == "self-exit-probe":
            cycles.append(cycle)
            entered.set()
            assert release.wait(2)
        return original_begin(cycle, timeout)

    monkeypatch.setattr(log_exit._ExitCycle, "begin", gated_begin)

    def self_exit():
        try:
            log_exit.exit_with_log_drain(24, timeout=0.3)
        except Exception as error:
            # Capture this thread's external helper outcome for the assertion.
            errors.append(error)

    helper = threading.Thread(target=self_exit, name="self-exit-probe")
    drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))

    def target():
        helper.start()
        assert entered.wait(1)

    try:
        log_exit.run_with_log_drain(target, (), drain.child_endpoint)
        assert log_exit._ProcessBinding.cycle is cycles[0]
    finally:
        release.set()
        helper.join(2)
        drain.close()
    assert not helper.is_alive()
    assert not errors
    assert exit_codes == [24]
    wait_cycle(cycles[0])
