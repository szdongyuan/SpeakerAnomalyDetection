"""Consumer-owned drop diagnostics with real sinks and controlled scheduling."""
import logging
import multiprocessing
import os
import threading
from types import SimpleNamespace
import weakref

import pytest

from base import log_manager
from base.log_manager import LogManager
from unit_test.logging_test_support import isolated_project_logger, managed_handlers


def wait_taken(runtime, count):
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._taken >= count, 2)


def assert_business_counts(runtime, accepted, written, errors=0):
    stats = runtime.stats()
    assert (stats["accepted"], stats["written"], stats["write_errors"]) == (
        accepted, written, errors)
    assert accepted == written + errors + stats["pending"]


def test_full_queue_reports_aggregate_in_consumer_without_blocking_producers(tmp_path, monkeypatch):
    monkeypatch.setattr(log_manager, "LOG_QUEUE_CAPACITY", 1)
    entered, release, reported, returned = [threading.Event() for _ in range(4)]
    original = log_manager._BatchFileHandler.write_batch
    threads = []

    def write(sink, entries):
        threads.append(threading.current_thread())
        if entries[0][0].msg == "blocked-business":
            entered.set()
            assert release.wait(5)
        result = original(sink, entries)
        if any(record.levelno == logging.WARNING for record, _ in entries):
            reported.set()
        return result

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger = LogManager.set_log_handler("core")
        runtime = LogManager._runtime
        logger.error("blocked-business")
        assert entered.wait(2)

        def produce():
            logger.info("partial-business")
            for _ in range(7):
                logger.info("rejected-payload")
            returned.set()

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            assert returned.wait(2), "producer waited on sink I/O"
            assert runtime.stats()["dropped_full"] == 7
        finally:
            release.set()
            producer.join(2)
        assert reported.wait(2), "no automatic drop WARNING reached the sink"
        wait_taken(runtime, 2)
        text = state.path.read_text()
        assert "WARNING" in text and f"pid={os.getpid()}" in text
        assert "queue_full delta=7 cumulative=7" in text
        assert "rejected-payload" not in text and "partial-business" not in text
        assert all(thread is runtime._thread for thread in threads)
        assert_business_counts(runtime, 2, 1)


def test_closed_routes_report_to_their_destinations_and_formatters(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        core = LogManager.set_log_handler("core")
        debug = LogManager.set_log_handler("debug")
        LogManager.set_log_handler("test")  # Keep the consumer alive.
        runtime = LogManager._runtime
        routes = [managed_handlers(logger)[0].route for logger in (core, debug)]
        for route, prefix in zip(routes, ("MAIN", "DEBUG")):
            route.formatter = logging.Formatter(prefix + " %(levelname)s %(message)s")
            assert runtime.close_route(route, 2)
        # Hold the condition so the consumer observes one aggregate per destination.
        with runtime._condition:
            for _ in range(3):
                core.info("closed-core")
            for _ in range(5):
                debug.error("closed-debug")
        assert LogManager.shutdown_all(2)
        main_text = state.path.read_text()
        debug_text = state.path.with_name("debug.log").read_text()
        assert "MAIN WARNING" in main_text and "closed delta=3 cumulative=3" in main_text
        assert "DEBUG WARNING" in debug_text and "closed delta=5 cumulative=5" in debug_text
        assert "DEBUG" not in main_text and "MAIN" not in debug_text
        assert runtime.stats()["dropped_closed"] == 8
        assert_business_counts(runtime, 0, 0)


def closed_logger():
    logger = LogManager.set_log_handler("core")
    other = LogManager.set_log_handler("debug")
    runtime = LogManager._runtime
    assert runtime.close_route(managed_handlers(logger)[0].route, 2)
    return logger, other, runtime


def test_rate_limit_and_idle_timer_preserve_partial_business_batch(tmp_path, monkeypatch):
    from consts.running_consts import LOG_DROP_REPORT_INTERVAL
    assert LOG_DROP_REPORT_INTERVAL == 1.0
    now = [100.0]
    monkeypatch.setattr(log_manager, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger, other, runtime = closed_logger()
        first_written, second_written, timer_seen = [threading.Event() for _ in range(3)]
        original = log_manager._BatchFileHandler.write_batch
        reports = []

        def write(sink, entries):
            result = original(sink, entries)
            reports.append(entries[0][0].getMessage())
            (first_written if len(reports) == 1 else second_written).set()
            return result

        monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
        logger.info("drop-one")
        assert first_written.wait(2)
        other.info("ordinary-partial")
        wait_taken(runtime, 1)
        wait = runtime._condition.wait
        waits = []

        def timed_wait(timeout=None):
            if threading.current_thread() is runtime._thread and timeout is not None:
                waits.append(timeout)
                # The consumer must request the remaining deadline, then an
                # idle timer wake alone must trigger the pending summary.
                assert len(reports) == 1
                now[0] += timeout
                timer_seen.set()
                return wait(0)
            return wait(timeout)

        monkeypatch.setattr(runtime._condition, "wait", timed_wait)
        with runtime._condition:
            now[0] = 100.25
            logger.info("drop-two")
            logger.info("drop-three")
        assert timer_seen.wait(2), "consumer did not schedule a report deadline"
        assert second_written.wait(2)
        assert waits == [0.75]
        assert "closed delta=1 cumulative=1" in reports[0]
        assert "closed delta=2 cumulative=3" in reports[1]
        assert not state.path.with_name("debug.log").exists()
        assert_business_counts(runtime, 1, 0)


def test_real_idle_wakeup_reports_without_new_business_or_explicit_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(log_manager, "LOG_DROP_REPORT_INTERVAL", 0.03)
    original = log_manager._BatchFileHandler.write_batch
    first, second = threading.Event(), threading.Event()
    reports = []

    def write(sink, entries):
        result = original(sink, entries)
        reports.extend(record.getMessage() for record, _ in entries)
        (first if len(reports) == 1 else second).set()
        return result

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
    with isolated_project_logger(tmp_path, monkeypatch):
        logger, _, runtime = closed_logger()
        logger.info("first-drop")
        assert first.wait(2)
        logger.info("idle-drop")
        assert second.wait(2)
        assert "closed delta=1 cumulative=2" in reports[1]
        assert_business_counts(runtime, 0, 0)


def test_drop_while_report_is_in_flight_survives_success_and_shutdown(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.write_batch
    reports = []

    def write(sink, entries):
        reports.append(entries[0][0].getMessage())
        if len(reports) == 1:
            entered.set()
            assert release.wait(5)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
    monkeypatch.setattr(log_manager, "LOG_DROP_REPORT_INTERVAL", 60)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger, _, runtime = closed_logger()
        logger.info("first-drop")
        try:
            assert entered.wait(2)
            for _ in range(3):
                logger.info("during-write")
            assert runtime.stats()["dropped_closed"] == 4
            assert not runtime.shutdown(0)
        finally:
            release.set()
        assert runtime.shutdown(2), "shutdown waited for aggregation deadline"
        assert len(reports) == 2
        assert "closed delta=1 cumulative=1" in reports[0]
        assert "closed delta=3 cumulative=4" in reports[1]
        assert state.path.read_text().count("WARNING") == 2
        assert_business_counts(runtime, 0, 0)


@pytest.mark.parametrize("failure", ["write", "format"])
def test_failed_report_waits_for_interval_and_retries_captured_delta(tmp_path, monkeypatch, failure):
    now = [100.0]
    monkeypatch.setattr(log_manager, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger, _, runtime = closed_logger()
        failed, recovered, timer_seen = [threading.Event() for _ in range(3)]
        original = log_manager._BatchFileHandler.write_batch
        reports = []

        class FailingFormatter(logging.Formatter):
            def format(self, record):
                assert threading.current_thread() is runtime._thread
                if len(reports) == 1:
                    raise ValueError("report-format-probe" + "x" * 600)
                return super().format(record)

        if failure == "format":
            managed_handlers(logger)[0].route.formatter = FailingFormatter("%(message)s")

        def write(sink, entries):
            reports.append(entries[0][0].getMessage())
            if len(reports) == 1 and failure == "write":
                failed.set()
                raise OSError("report-write-probe" + "x" * 600)
            result = original(sink, entries)
            (failed if len(reports) == 1 else recovered).set()
            return result

        monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
        wait = runtime._condition.wait

        def observe_wait(timeout=None):
            if threading.current_thread() is runtime._thread and timeout is not None:
                timer_seen.set()
            return wait(timeout)

        monkeypatch.setattr(runtime._condition, "wait", observe_wait)
        logger.info("failed-report-drop")
        assert failed.wait(2)
        assert timer_seen.wait(2), "failed report retried without an interval wait"
        with runtime._condition:
            assert len(reports) == 1
            assert 0 < len(runtime.stats()["last_error"]) <= 512
            assert failure in runtime.stats()["last_error"]
            now[0] = 101.0
            logger.info("new-drop-before-retry")
        assert recovered.wait(2)
        assert "closed delta=2 cumulative=2" in reports[1]
        assert "closed delta=2 cumulative=2" in state.path.read_text()
        assert_business_counts(runtime, 0, 0)


def test_final_snapshot_is_finite_and_post_stop_only_counts(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.write_batch
    reports = []

    def write(sink, entries):
        reports.append(entries[0][0].getMessage())
        entered.set()
        assert release.wait(5)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger, _, runtime = closed_logger()
        with runtime._condition:
            logger.info("before-final-snapshot")
            assert not runtime.shutdown(0)
        try:
            assert entered.wait(2)
            for _ in range(20):
                logger.info("after-final-snapshot")
        finally:
            release.set()
        assert runtime.shutdown(2)
        text = state.path.read_text()
        assert len(reports) == 1 and "closed delta=1 cumulative=1" in reports[0]
        for _ in range(4):
            logger.info("after-consumer-stopped")
        assert runtime.stats()["dropped_closed"] == 25
        assert LogManager._runtime is runtime and not runtime._thread.is_alive()
        assert state.path.read_text() == text
        assert_business_counts(runtime, 0, 0)


@pytest.mark.parametrize("failure", ["write", "format"])
def test_failed_final_summary_is_attempted_once_and_forced_exit_reports_errors(
        tmp_path, monkeypatch, failure):
    from base import log_exit

    monkeypatch.setattr(LogManager, "_forced_exit", False)
    monkeypatch.setattr(log_manager, "LOG_DROP_REPORT_INTERVAL", 60)
    with isolated_project_logger(tmp_path, monkeypatch):
        logger, other, runtime = closed_logger()
        other.info("accepted-tail")
        wait_taken(runtime, 1)
        attempts = []
        original = log_manager._BatchFileHandler.write_batch

        class FailingFormatter(logging.Formatter):
            def format(self, record):
                assert threading.current_thread() is runtime._thread
                raise ValueError("final-summary-format-probe" + "x" * 600)

        if failure == "format":
            managed_handlers(logger)[0].route.formatter = FailingFormatter()

        def write(sink, entries):
            if entries[0][0].levelno == logging.WARNING:
                attempts.append(entries[0][0].getMessage())
                if failure == "write":
                    raise OSError("final-summary-write-probe" + "x" * 600)
            return original(sink, entries)

        monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
        drain = log_exit.ProcessLogDrain.create(multiprocessing.get_context("spawn"))
        cycle = log_exit._ExitCycle(drain.child_endpoint)
        try:
            with runtime._condition:
                logger.info("final-drop")
                assert not runtime.shutdown(0)
                cycle.begin(1)
            original_deadline = cycle.deadline
            cycle.begin(2)
            cycle.worker.join(2)
            assert not cycle.worker.is_alive()
            result = cycle.poll()
            assert result.status == "drained-with-errors"
            assert drain.poll() == result
            assert cycle.deadline == original_deadline
            assert len(attempts) == 1
            assert "final-summary" in result.stats["last_error"]
            assert len(result.stats["last_error"]) <= 512
            assert result.stats["dropped_closed"] == 1
            assert_business_counts(runtime, 1, 1)
            assert LogManager._runtime is runtime
        finally:
            drain.close()


def test_shared_destination_keeps_bounded_counters_without_rejected_records(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger = LogManager.set_log_handler("core")
        alias = LogManager.set_log_handler("alias")
        LogManager.set_log_handler("keepalive")
        runtime = LogManager._runtime
        routes = [managed_handlers(item)[0].route for item in (logger, alias)]
        for route in routes:
            assert runtime.close_route(route, 2)
        with runtime._condition:
            for index in range(500):
                record = logging.LogRecord("core", logging.INFO, __file__, 0,
                                           "rejected-%s", (index,), None)
                reference = weakref.ref(record)
                runtime.admit(routes[index % 2], record)
                del record
                assert reference() is None
            assert len(runtime._drop_reports) == 1
        assert runtime.shutdown(2)
        assert "closed delta=500 cumulative=500" in state.path.read_text()
        assert_business_counts(runtime, 0, 0)
