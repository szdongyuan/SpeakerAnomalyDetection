"""Deterministic public async logging contracts with real temporary files."""
import logging
import inspect
import threading
import time

import pytest

from base import log_manager
from base.log_manager import LogManager
from unit_test.logging_test_support import isolated_project_logger, managed_handlers


@pytest.fixture
def isolated_async_logger(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        yield LogManager.set_log_handler("core"), state.path


def test_partial_batch_explicit_flush(isolated_async_logger):
    logger, path = isolated_async_logger
    logger.info("part=%s", 1)
    logger.warning("part=%s", 2)
    assert LogManager.flush(timeout=2.0)
    text = path.read_text()
    assert "part=1" in text and "part=2" in text
    stats = LogManager.get_async_stats()
    assert stats["accepted"] == stats["written"] == 2
    assert stats["pending"] == 0


def wait_taken(runtime, count):
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._taken >= count, timeout=2)


def wait_written(runtime, count):
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._written >= count, timeout=2)


def test_five_records_and_no_timer_flush(isolated_async_logger, monkeypatch):
    logger, path = isolated_async_logger
    calls = []
    original = log_manager._BatchFileHandler.write_batch

    def write(sink, entries):
        calls.append(len(entries))
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", write)
    runtime = LogManager._runtime
    waits = []
    wait = runtime._condition.wait

    def observe_wait(timeout=None):
        if threading.current_thread() is runtime._thread:
            waits.append(timeout)
        return wait(timeout)

    monkeypatch.setattr(runtime._condition, "wait", observe_wait)
    for count in range(1, 11):
        logger.warning("item=%d", count)
        wait_taken(runtime, count)
        if count % 5:
            assert calls == ([5] if count > 5 else [])
            assert LogManager.get_async_stats()["pending"] == count % 5
        else:
            wait_written(runtime, count)
            assert path.read_text().count("item=") == count
    assert calls == [5, 5]
    assert waits and all(timeout is None for timeout in waits)


@pytest.mark.parametrize("method", ["error", "critical", "exception"])
def test_errors_deliver_partial_batch(isolated_async_logger, method):
    logger, path = isolated_async_logger
    logger.info("before")
    try:
        raise ValueError("trace-probe")
    except ValueError:
        getattr(logger, method)("failure")
    wait_written(LogManager._runtime, 2)
    text = path.read_text()
    assert text.index("before") < text.index("failure")
    if method == "exception":
        assert "Traceback" in text and "ValueError: trace-probe" in text


def test_manual_shutdown_tail_reject_and_reacquire(isolated_async_logger, monkeypatch):
    logger, path = isolated_async_logger
    logger.propagate = False
    fallback = []
    monkeypatch.setattr(logging.lastResort, "emit", fallback.append)
    logger.info("tail")
    old = LogManager._runtime
    assert LogManager.shutdown_all(timeout=2)
    assert "tail" in path.read_text()
    for method in ("info", "warning", "error", "exception"):
        getattr(logger, method)("closed")
    assert LogManager.get_async_stats()["dropped_closed"] == 4
    assert not fallback
    assert LogManager.shutdown_all(timeout=2)
    assert LogManager.set_log_handler("core") is logger
    assert LogManager._runtime is not old
    assert len(logger.handlers) == 1
    logger.error("reopened")
    assert LogManager.flush(timeout=2)
    assert "reopened" in path.read_text()


def test_instance_close_preserves_other_routes(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        core = LogManager("core")
        core.logger.propagate = False
        fallback = []
        monkeypatch.setattr(logging.lastResort, "emit", fallback.append)
        other = LogManager.set_log_handler("other")
        core.info("core-tail")
        other.info("other-tail")
        core.shut_down()
        for method in ("info", "warning", "error", "exception"):
            getattr(core.logger, method)("rejected")
        other.error("still-active")
        assert LogManager.flush(timeout=2)
        assert LogManager.get_async_stats()["dropped_closed"] == 4
        assert not fallback and not core.logger.propagate
        text = state.path.read_text()
        assert "core-tail" in text and "still-active" in text
        assert "rejected" not in text


def test_stats_query_before_init_does_not_start_runtime():
    previous = getattr(LogManager, "_runtime", None)
    LogManager._runtime = None
    try:
        assert LogManager.get_async_stats() == dict(
            accepted=0, written=0, dropped_full=0, dropped_closed=0,
            write_errors=0, snapshot_errors=0, pending=0, last_error=None,
            consumer_alive=False)
        assert LogManager._runtime is None
    finally:
        LogManager._runtime = previous


def test_blocked_output_drops_full_without_blocking_producers(tmp_path, monkeypatch):
    monkeypatch.setattr(log_manager, "LOG_QUEUE_CAPACITY", 2)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.write_batch
    calls = []

    def gated(sink, entries):
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", gated)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        logger = LogManager.set_log_handler("core")
        logger.propagate = False
        fallback = []
        monkeypatch.setattr(logging.lastResort, "emit", fallback.append)
        logger.error("blocked-first")
        assert entered.wait(2)

        def produce():
            logger.info("queued-info")
            logger.warning("queued-warning")
            logger.error("dropped-error")
            try:
                raise ValueError("dropped-exception")
            except ValueError:
                logger.exception("dropped-exception")
            logger.info("dropped-info")
            logger.warning("dropped-warning")
            returned.set()

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            assert returned.wait(2), "producer waited on sink or queue space"
            stats = LogManager.get_async_stats()
            assert (stats["accepted"], stats["pending"], stats["written"]) == (3, 3, 0)
            assert stats["dropped_full"] == 4
            assert not fallback
        finally:
            release.set()
            producer.join(2)
        assert LogManager.flush(2)
        assert LogManager.get_async_stats()["written"] == 3
        assert all(thread is LogManager._runtime._thread for thread in calls)
        text = state.path.read_text()
        assert text.count("blocked-first") == 1
        assert "dropped-" not in text


def test_snapshot_metadata_and_original_record(isolated_async_logger, monkeypatch, caplog):
    logger, path = isolated_async_logger
    observed = []
    original = log_manager._BatchFileHandler.write_batch

    def capture(sink, entries):
        observed.extend(record for record, _ in entries)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", capture)
    argument = ["before"]
    before = time.time()
    line = inspect.currentframe().f_lineno + 1
    logger.info("snapshot=%s", argument, stack_info=True)
    after = time.time()
    argument.append("after")
    assert LogManager.flush(2)
    copied, = observed
    source, = [r for r in caplog.records if r.msg == "snapshot=%s"]
    assert copied is not source
    assert source.args == (argument,)
    assert copied.args is None and copied.msg == "snapshot=['before']"
    assert copied.created == source.created and before <= copied.created <= after
    assert copied.lineno == source.lineno == line
    assert copied.thread == threading.get_ident()
    assert copied.process == source.process
    assert copied.stack_info == source.stack_info
    text = path.read_text()
    assert "Stack (most recent call last)" in text and "after" not in text


def test_concurrent_initialization_destinations_and_grouping(tmp_path, monkeypatch):
    start = threading.Barrier(8)
    acquired, errors = [], []
    writes = []
    starts = []
    start_thread = threading.Thread.start

    def track_start(thread):
        if thread.name == "project-log-consumer":
            starts.append(thread)
        return start_thread(thread)

    monkeypatch.setattr(threading.Thread, "start", track_start)
    original = log_manager._BatchFileHandler.write_batch

    def capture(sink, entries):
        writes.append((sink.baseFilename, len(entries), threading.current_thread()))
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", capture)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        def acquire(index):
            try:
                start.wait(2)
                for name in ("core", "debug", "test", f"other{index}"):
                    for _ in range(3):
                        logger = LogManager.set_log_handler(name)
                        acquired.append((name, logger))
                    logger.info("unique-%d-%s", index, name)
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=acquire, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert not errors and all(not t.is_alive() for t in threads)
        assert LogManager.flush(2)
        runtime = LogManager._runtime
        assert starts == [runtime._thread]
        assert all(thread is runtime._thread for _, _, thread in writes)
        assert len([t for t in threading.enumerate()
                    if t.name == "project-log-consumer" and t is runtime._thread]) == 1
        for name, logger in acquired:
            assert logger is logging.getLogger(name)
            handler, = managed_handlers(logger)
            assert handler.runtime is runtime
        main = state.path.read_text()
        debug = state.path.with_name("debug.log").read_text()
        test = state.path.with_name("test.log").read_text()
        for i in range(8):
            assert main.count(f"unique-{i}-core") == 1
            assert main.count(f"unique-{i}-other{i}") == 1
            assert debug.count(f"unique-{i}-debug") == 1
            assert test.count(f"unique-{i}-test") == 1
        assert "-debug" not in main and "-test" not in main


@pytest.mark.parametrize("names", [("core", "core.video"), ("core.video", "core")])
def test_explicit_parent_child_acquisition_does_not_duplicate(tmp_path, monkeypatch, names):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        for name in names:
            LogManager.set_log_handler(name)
        logging.getLogger("core.video").info("child-once")
        assert LogManager.flush(2)
        assert state.path.read_text().count("child-once") == 1


def test_failed_thread_start_is_atomic(tmp_path, monkeypatch):
    original = threading.Thread.start
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        def fail(thread):
            raise RuntimeError("start-probe")

        monkeypatch.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError, match="start-probe"):
            LogManager.set_log_handler("core")
        assert LogManager._runtime is None
        assert not managed_handlers(state.logger)
        monkeypatch.setattr(threading.Thread, "start", original)
        logger = LogManager.set_log_handler("core")
        logger.error("started-once")
        assert LogManager.flush(2)
        assert state.path.read_text().count("started-once") == 1


def test_registration_failure_preserves_live_runtime(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        core = LogManager.set_log_handler("core")
        runtime = LogManager._runtime
        log_manager.LOG_MAPPING["broken"] = {"log_format": "%(invalid"}
        with pytest.raises(ValueError):
            LogManager.set_log_handler("broken")
        assert LogManager._runtime is runtime
        assert not managed_handlers(logging.getLogger("broken"))
        core.error("still-works")
        assert LogManager.flush(2)
        assert "still-works" in state.path.read_text()


@pytest.mark.parametrize("second_flush", [False, True])
def test_flush_segments_at_capture_before_later_admissions(tmp_path, monkeypatch, second_flush):
    first_entered, first_release = threading.Event(), threading.Event()
    later_entered, later_release = threading.Event(), threading.Event()
    requested, finished = threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.write_batch
    results = []

    def gated(sink, entries):
        if any(r.msg == "first" for r, _ in entries):
            first_entered.set()
            assert first_release.wait(5)
        if any(r.msg == "later" for r, _ in entries):
            later_entered.set()
            assert later_release.wait(5)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", gated)
    with isolated_project_logger(tmp_path, monkeypatch):
        logger = LogManager.set_log_handler("core")
        runtime = LogManager._runtime
        request = runtime._request_barrier

        def capture(target):
            request(target)
            requested.set()

        monkeypatch.setattr(runtime, "_request_barrier", capture)
        logger.error("first")
        assert first_entered.wait(2)
        logger.info("captured-tail")

        def flush():
            results.append(LogManager.flush(2))
            finished.set()

        flusher = threading.Thread(target=flush)
        flusher.start()
        later_flusher = None
        try:
            assert requested.wait(2)
            logger.error("later")
            if second_flush:
                later_flusher = threading.Thread(target=lambda: LogManager.flush(2))
                later_flusher.start()
                with runtime._condition:
                    assert runtime._condition.wait_for(lambda: 3 in runtime._barriers, 2)
            first_release.set()
            assert later_entered.wait(2)
            assert finished.wait(1), "later admission extended captured flush"
            assert results == [True]
        finally:
            first_release.set()
            later_release.set()
            flusher.join(3)
            if later_flusher is not None:
                later_flusher.join(3)


@pytest.mark.parametrize("operation", ["flush", "shutdown", "instance"])
def test_blocked_output_lifecycle_is_bounded(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(log_manager, "LOG_QUEUE_CAPACITY", 1)
    monkeypatch.setattr(log_manager, "LOG_SHUTDOWN_TIMEOUT", 0.03)
    entered, release = threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.write_batch

    def blocked(sink, entries):
        entered.set()
        assert release.wait(5)
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", blocked)
    with isolated_project_logger(tmp_path, monkeypatch):
        manager = LogManager("core")
        runtime = LogManager._runtime
        manager.error("blocking")
        assert entered.wait(2)
        manager.info("full-queue")
        try:
            started = time.monotonic()
            if operation == "flush":
                assert not LogManager.flush(0.03)
            elif operation == "shutdown":
                assert not LogManager.shutdown_all(0.03)
            else:
                manager.shut_down()
            assert time.monotonic() - started < 1
            stats = LogManager.get_async_stats()
            assert stats["pending"] == 2 and stats["consumer_alive"]
            if operation != "flush":
                with pytest.raises(RuntimeError, match="still shutting down"):
                    LogManager.set_log_handler("core")
                assert LogManager._runtime is runtime
                manager.error("rejected")
                assert LogManager.get_async_stats()["dropped_closed"] == 1
        finally:
            release.set()
        assert LogManager.shutdown_all(2)


@pytest.mark.parametrize("failure", ["format", "write", "handleError"])
def test_per_record_errors_and_recovery(isolated_async_logger, monkeypatch, failure):
    logger, path = isolated_async_logger
    original = log_manager._BatchFileHandler.do_write

    if failure == "format":
        route = managed_handlers(logger)[0].route
        route.formatter = logging.Formatter("%(probe)s %(message)s")
        logger.info("bad-format")
        logger.error("valid", extra={"probe": "ok"})
        expected = (1, 1)
    else:
        def fail(sink, text):
            if failure == "write":
                raise OSError("write-probe" + "x" * 600)
            try:
                raise OSError("library-probe")
            except OSError:
                sink.handleError(None)

        monkeypatch.setattr(log_manager._BatchFileHandler, "do_write", fail)
        logger.info("unconfirmed")
        logger.error("failure")
        expected = (0, 2)
    assert LogManager.flush(2)
    stats = LogManager.get_async_stats()
    assert (stats["written"], stats["write_errors"]) == expected
    assert stats["pending"] == 0 and 0 < len(stats["last_error"]) <= 512
    monkeypatch.setattr(log_manager._BatchFileHandler, "do_write", original)
    logger.error("recovered", extra={"probe": "ok"})
    assert LogManager.flush(2)
    assert path.read_text().count("recovered") == 1


def test_shared_destination_merges_one_process_batch(tmp_path, monkeypatch):
    writes = []
    original = log_manager._BatchFileHandler.write_batch

    def capture(sink, entries):
        writes.append((sink.baseFilename, [r.name for r, _ in entries]))
        return original(sink, entries)

    monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", capture)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        for name in ("core", "other", "debug", "test", "core"):
            LogManager.set_log_handler(name).info("group-%s", name)
        wait_written(LogManager._runtime, 5)
        groups = {log_manager.os.path.basename(path): names for path, names in writes}
        assert len(writes) == 3
        assert groups == {"main.log": ["core", "other", "core"],
                          "debug.log": ["debug"], "test.log": ["test"]}
        assert state.path.read_text().count("group-core") == 2


def test_concurrent_admission_sequence_matches_file_order(isolated_async_logger, monkeypatch):
    logger, path = isolated_async_logger
    runtime = LogManager._runtime
    admitted = []
    original = runtime._queue.put_nowait

    def capture(entry):
        original(entry)
        admitted.append((entry.sequence, entry.record.msg))

    monkeypatch.setattr(runtime._queue, "put_nowait", capture)
    start = threading.Barrier(4)

    def produce(index):
        start.wait(2)
        for n in range(15):
            logger.info("ordered-%d-%d", index, n)

    threads = [threading.Thread(target=produce, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert all(not thread.is_alive() for thread in threads)
    assert LogManager.flush(2)
    assert [seq for seq, _ in admitted] == list(range(1, 61))
    lines = path.read_text().splitlines()
    assert len(lines) == len(admitted) == 60
    assert [line.split(" INFO ")[1].split(" [")[0] for line in lines] == [
        message for _, message in admitted]


def test_fixture_drains_suspends_and_restores_existing_runtime(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path / "outer", monkeypatch) as outer:
        logger = LogManager.set_log_handler("core")
        original_runtime = LogManager._runtime
        user_handler = logging.NullHandler()
        closed = []
        monkeypatch.setattr(user_handler, "close", lambda: closed.append(True))
        logger.addHandler(user_handler)
        originals = (logger.handlers[:], logger.level, logger.propagate, logger.disabled)
        logger.info("outer-tail")
        with pytest.MonkeyPatch.context() as inner_patch:
            with isolated_project_logger(tmp_path / "inner", inner_patch) as inner:
                assert "outer-tail" in outer.path.read_text()
                assert original_runtime.stats()["pending"] == 0
                LogManager.set_log_handler("core").info("inner-tail")
                inner_runtime = LogManager._runtime
            assert "inner-tail" in inner.path.read_text()
        assert not inner_runtime._thread.is_alive()
        assert LogManager._runtime is original_runtime
        assert original_runtime._thread.is_alive()
        assert (logger.handlers, logger.level, logger.propagate, logger.disabled) == originals
        assert not closed
        logger.error("outer-again")
        assert LogManager.flush(2)
        assert "outer-again" in outer.path.read_text()
        assert "inner-tail" not in outer.path.read_text()


def test_shutdown_during_route_preparation_cannot_publish_dead_route(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch):
        LogManager.set_log_handler("core")
        entered, release = threading.Event(), threading.Event()
        original = log_manager.os.makedirs
        outcomes = []

        def gated(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(log_manager.os, "makedirs", gated)

        def acquire():
            try:
                outcomes.append(LogManager.set_log_handler("racing"))
            except RuntimeError as error:
                outcomes.append(error)

        thread = threading.Thread(target=acquire)
        thread.start()
        try:
            assert entered.wait(2)
            assert LogManager.shutdown_all(2)
        finally:
            release.set()
            thread.join(3)
        assert len(outcomes) == 1 and isinstance(outcomes[0], RuntimeError)
        assert not managed_handlers(logging.getLogger("racing"))


@pytest.mark.parametrize("failure", ["interpolation", "message", "unprintable_error"])
def test_snapshot_errors_are_contained_before_admission(
        isolated_async_logger, monkeypatch, capsys, failure):
    logger, path = isolated_async_logger
    originals = []

    class RawRecordHandler(logging.Handler):
        def emit(self, record):
            originals.append(record)

    class UnprintableError(ValueError):
        def __str__(self):
            raise RuntimeError("exception-string-probe")

    class BrokenMessage:
        def __str__(self):
            if failure == "unprintable_error":
                raise UnprintableError()
            raise ValueError("message-string-probe" + "x" * 600)

    monkeypatch.setattr(logging.root, "handlers", [RawRecordHandler()])
    message = "value=%d" if failure == "interpolation" else BrokenMessage()
    args = ("invalid",) if failure == "interpolation" else ()
    logger.info(message, *args)
    stats = LogManager.get_async_stats()
    assert stats["snapshot_errors"] == 1
    assert all(stats[key] == 0 for key in (
        "accepted", "written", "write_errors", "pending", "dropped_full", "dropped_closed"))
    assert 0 < len(stats["last_error"]) <= 512
    assert "snapshot" in stats["last_error"].lower()
    assert not path.exists()
    original, = originals
    assert original.msg is message and original.args == args
    assert not capsys.readouterr().err
    logger.error("successful-after-snapshot-failure")
    assert LogManager.flush(2)
    assert path.read_text().count("successful-after-snapshot-failure") == 1
    stats = LogManager.get_async_stats()
    assert stats["snapshot_errors"] == stats["accepted"] == stats["written"] == 1
    assert stats["pending"] == stats["write_errors"] == 0


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_snapshot_keeps_process_control_exceptions(isolated_async_logger, exception):
    logger, _ = isolated_async_logger
    logger.propagate = False

    class InterruptedMessage:
        def __str__(self):
            raise exception()

    with pytest.raises(exception):
        logger.info(InterruptedMessage())
    assert LogManager.get_async_stats()["snapshot_errors"] == 0
