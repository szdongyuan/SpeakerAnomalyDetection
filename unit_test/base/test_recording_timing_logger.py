"""Optional recording timing diagnostics never wait for a handler."""
import inspect
import logging
import threading
import time

import pytest

from base.recording_timing_logger import RecordingTimingLogger
from base.log_manager import LogManager


def test_source_timestamp_and_normal_close_drain():
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.Logger("original", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    assert writer.thread is None
    before = time.time()
    line = inspect.currentframe().f_lineno + 1
    assert writer.info(logger, "request=%s at=%.6f", "one", 123.25)
    after = time.time()
    writer.close()
    writer.thread.join(3)
    assert not writer.thread.is_alive()
    record, = records
    assert record.name == "original" and record.levelno == logging.INFO
    assert record.pathname == __file__ and record.lineno == line
    assert record.funcName == "test_source_timestamp_and_normal_close_drain"
    assert before <= record.created <= after
    assert record.args == ("one", 123.25)
    assert record.getMessage() == "request=one at=123.250000"
    assert record.thread == threading.get_ident()
    assert not writer.info(logger, "after close")
    assert writer.dropped == 1


def test_critical_flush_occurs_after_delivery_and_ordinary_default_does_not_flush(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    actions = []

    class Handler(logging.Handler):
        def emit(self, record):
            if record.msg == "critical":
                entered.set()
                assert release.wait(5)
            actions.append((record.msg, record.thread, record.levelno))

    monkeypatch.setattr(LogManager, "request_flush", lambda: actions.append("flush"), raising=False)
    logger = logging.Logger("forwarded", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    try:
        assert writer.info(logger, "critical", critical=True)
        assert entered.wait(2)
        assert actions == []
        assert writer.info(logger, "ordinary")
    finally:
        release.set()
        writer.close()
        if writer.thread is not None:
            writer.thread.join(3)
    assert actions == [("critical", threading.get_ident(), logging.INFO), "flush",
                       ("ordinary", threading.get_ident(), logging.INFO)]


def test_optional_warning_forwarding_and_flush_failure_continue_drain(monkeypatch):
    records = []

    def fail_flush():
        raise OSError("flush-probe")

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    monkeypatch.setattr(LogManager, "request_flush", fail_flush, raising=False)
    logger = logging.Logger("warning", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    assert writer.log(logger, logging.WARNING, "slow", critical=True)
    assert writer.info(logger, "after")
    writer.close()
    writer.thread.join(3)
    assert [(r.msg, r.levelno) for r in records] == [("slow", logging.WARNING), ("after", logging.INFO)]
    assert writer.errors == 1


@pytest.mark.parametrize("boundary", ["isEnabledFor", "findCaller", "makeRecord"])
def test_optional_record_preparation_failure_is_contained(monkeypatch, boundary):
    writer = RecordingTimingLogger()
    logger = logging.Logger("preparation-fault", logging.INFO)

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("broken exception string")

    def fail(*args, **kwargs):
        raise UnprintableError()

    monkeypatch.setattr(logger, boundary, fail)
    assert not writer.info(logger, "optional", critical=True)
    assert writer.errors == 1 and writer.thread is None
    assert "UnprintableError" in writer.last_error
    writer.close()


def test_unprintable_handler_error_does_not_kill_forwarder():
    writer = RecordingTimingLogger()
    logger = logging.Logger("unprintable", logging.INFO)
    records = []

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("broken exception string")

    class Handler(logging.Handler):
        def emit(self, record):
            if record.msg == "bad":
                raise UnprintableError()
            records.append(record)

    logger.addHandler(Handler())
    assert writer.info(logger, "bad")
    assert writer.info(logger, "good")
    writer.close()
    writer.thread.join(3)
    assert [r.msg for r in records] == ["good"]
    assert writer.errors == 1


def test_saturation_and_close_never_wait_for_blocked_handler():
    entered, release = threading.Event(), threading.Event()
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            entered.set()
            assert release.wait(5)
            records.append(record)

    logger = logging.Logger("gated", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    try:
        assert writer.info(logger, "first")
        assert entered.wait(2)
        consumer = writer.thread
        assert consumer.daemon
        for index in range(64):
            assert writer.info(logger, "queued=%s", index)
        for _ in range(200):
            assert not writer.info(logger, "overflow")
        writer.close()
        assert not writer.info(logger, "closed")
        assert writer.thread is consumer and consumer.is_alive()
        assert writer.dropped == 201
    finally:
        release.set()
        writer.close()
        writer.thread.join(3)
    assert not consumer.is_alive()
    assert len(records) == 65
    assert [r.args for r in records[1:]] == [(i,) for i in range(64)]


def test_handler_exception_is_bounded_and_does_not_stop_drain():
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            if record.msg == "bad":
                raise ValueError("x" * 2000)
            records.append(record)

    logger = logging.Logger("fault", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    assert writer.info(logger, "bad")
    assert writer.info(logger, "good")
    writer.close()
    writer.thread.join(3)
    assert not writer.thread.is_alive()
    assert [r.msg for r in records] == ["good"]
    assert writer.errors == 1
    assert writer.last_error.startswith("ValueError:")
    assert len(writer.last_error) <= 512


def test_start_failure_rejects_future_records_without_retry():
    calls = []

    def fail_start(thread):
        calls.append(thread)
        raise RuntimeError("thread-start-probe")

    writer = RecordingTimingLogger(start_thread=fail_start)
    logger = logging.Logger("start-fault", logging.INFO)
    assert not writer.info(logger, "first")
    assert not writer.info(logger, "second")
    writer.close()
    assert len(calls) == 1
    assert writer.errors == 1 and writer.dropped == 2
    assert writer.last_error == "RuntimeError: thread-start-probe"
    assert writer.thread.ident is None


def test_close_without_records_does_not_start_consumer():
    writer = RecordingTimingLogger()
    writer.close()
    assert not writer.info(logging.Logger("closed"), "ignored")
    assert writer.thread is None


def test_disabled_info_does_not_start_consumer():
    writer = RecordingTimingLogger()
    assert not writer.info(logging.Logger("disabled", logging.ERROR), "ignored")
    writer.close()
    assert writer.thread is None and writer.dropped == 0


def test_error_uses_existing_consumer_and_preserves_source_behind_blocked_handler():
    entered, release = threading.Event(), threading.Event()
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            if record.msg == "first":
                entered.set()
                assert release.wait(5)
            records.append(record)

    logger = logging.Logger("drain-diagnostic", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger()
    try:
        assert writer.info(logger, "first")
        assert entered.wait(2)
        consumer = writer.thread
        before = time.time()
        line = inspect.currentframe().f_lineno + 1
        assert writer.error(logger, "status=%s", "already-dead")
        after = time.time()
        writer.close()
        assert writer.thread is consumer and consumer.is_alive()
        assert records == []
    finally:
        release.set()
        writer.close()
        writer.thread.join(3)
    assert not consumer.is_alive()
    record = records[1]
    assert record.levelno == logging.ERROR
    assert record.pathname == __file__ and record.lineno == line
    assert record.thread == threading.get_ident()
    assert before <= record.created <= after
    assert record.getMessage() == "status=already-dead"


def test_starter_that_launches_then_raises_retains_and_drains_owned_consumer():
    records, threads = [], []

    def start_then_raise(thread):
        threads.append(thread)
        thread.start()
        raise RuntimeError("started-but-raised")

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.Logger("started", logging.INFO)
    logger.addHandler(Handler())
    writer = RecordingTimingLogger(start_thread=start_then_raise)
    try:
        assert writer.info(logger, "first")
        assert writer.info(logger, "second")
    finally:
        writer.close()
        writer.thread.join(3)
    assert threads == [writer.thread]
    assert not writer.thread.is_alive()
    assert [r.msg for r in records] == ["first", "second"]
    assert writer.errors == 1 and writer.dropped == 0
