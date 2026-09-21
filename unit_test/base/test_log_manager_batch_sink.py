"""Real-file tests of the single-consumer batch output boundary."""
import logging
import sys
import threading
from unittest.mock import Mock

import pytest

from base.log_manager import _BatchFileHandler


def record(message, level=logging.INFO, exc_info=None):
    return logging.LogRecord("core", level, "producer.py", 42, message, (), exc_info,
                             func="produce")


@pytest.fixture
def sink(tmp_path):
    handler = _BatchFileHandler(tmp_path / "batch.log", maxBytes=1024, backupCount=5,
                                encoding="utf-8")
    yield handler
    if not handler._closed:
        handler.close_owned()


def entries(*messages):
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s [%(filename)s:%(lineno)d]")
    return [(record(message), formatter) for message in messages]


@pytest.mark.parametrize("existing", [True, False])
def test_five_records_use_one_real_write_open_and_lock(sink, monkeypatch, existing):
    if existing:
        with open(sink.baseFilename, "w", encoding="utf-8"):
            pass
    calls = []
    original_open = sink.do_open

    class StreamSpy:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, text):
            assert sink.is_locked
            calls.append(text)
            return self.stream.write(text)

    opened = Mock(side_effect=lambda *a, **kw: StreamSpy(original_open(*a, **kw)))
    locked = Mock(wraps=sink._do_lock)
    unlocked = Mock(wraps=sink._do_unlock)
    monkeypatch.setattr(sink, "do_open", opened)
    monkeypatch.setattr(sink, "_do_lock", locked)
    monkeypatch.setattr(sink, "_do_unlock", unlocked)
    result = sink.write_batch(entries(*(f"entry-{i}" for i in range(5))))
    assert result.outcomes == (True,) * 5
    assert result.written == 5 and result.write_errors == 0
    expected = "".join(f"core INFO entry-{i} [producer.py:42]\n" for i in range(5))
    with open(sink.baseFilename, encoding="utf-8") as output:
        assert output.read() == expected
    assert calls == [expected]
    assert opened.call_count == locked.call_count == unlocked.call_count == 1
    assert sink.stream is None and not sink.is_locked


def test_single_error_preserves_unicode_traceback_and_metadata(sink):
    try:
        raise ValueError("异常细节")
    except ValueError:
        item = record("故障\n第二行", logging.ERROR, sys.exc_info())
    formatter = logging.Formatter("%(created)f %(funcName)s %(message)s")
    expected = formatter.format(item) + "\n"
    result = sink.write_batch([(item, formatter)])
    assert result.outcomes == (True,)
    with open(sink.baseFilename, encoding="utf-8") as output:
        assert output.read() == expected
    assert "Traceback" in expected and "ValueError: 异常细节" in expected


def test_formatter_failure_isolates_one_record_and_formats_before_lock(sink):
    class SelectiveFormatter(logging.Formatter):
        def format(self, item):
            assert not sink.is_locked
            if item.msg == "bad":
                raise ValueError("format-probe")
            return super().format(item)

    formatter = SelectiveFormatter("%(message)s")
    result = sink.write_batch([(record(text), formatter) for text in ("first", "bad", "last")])
    assert result.outcomes == (True, False, True)
    assert result.written == 2 and result.write_errors == 1
    assert "format-probe" in result.last_error
    with open(sink.baseFilename, encoding="utf-8") as output:
        assert output.read() == "first\nlast\n"


def test_shared_destination_preserves_each_logger_formatter_and_terminator(sink, monkeypatch):
    first = record("one")
    second = record("two")
    second.name = "soundcard_core"
    first_format = logging.Formatter("%(name)s: %(message)s")
    second_format = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    sink.terminator = " | "
    write = Mock(wraps=sink.do_write)
    monkeypatch.setattr(sink, "do_write", write)
    assert sink.write_batch([(first, first_format), (second, second_format)]).written == 2
    write.assert_called_once_with("core: one | INFO soundcard_core: two")
    with open(sink.baseFilename, encoding="utf-8") as output:
        assert output.read() == "core: one | INFO soundcard_core: two | "


@pytest.mark.parametrize("backup_count", [0, 5])
def test_rotation_occurs_only_at_next_batch_boundary(tmp_path, backup_count):
    path = tmp_path / "rotate.log"
    sink = _BatchFileHandler(path, maxBytes=8, backupCount=backup_count, encoding="utf-8")
    try:
        assert sink.write_batch(entries("first", "second")).written == 2
        first = path.read_text(encoding="utf-8")
        assert "first" in first and "second" in first
        assert path.stat().st_size > sink.maxBytes
        assert not path.with_suffix(".log.1").exists()
        assert sink.write_batch(entries("third", "fourth")).written == 2
        assert path.read_text(encoding="utf-8").splitlines() == [
            "core INFO third [producer.py:42]", "core INFO fourth [producer.py:42]"]
        assert sink.stream is None
        if backup_count:
            assert path.with_suffix(".log.1").read_text(encoding="utf-8") == first
        else:
            assert list(tmp_path.glob("rotate.log.*")) == []
    finally:
        sink.close_owned()


@pytest.mark.parametrize("failure", ["write", "flush", "close", "handle_error", "partial_write"])
def test_unconfirmed_group_is_failed_without_retry_and_next_batch_works(
        sink, monkeypatch, capsys, failure):
    original_write = sink.do_write
    original_open = sink.do_open
    writes = []

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def flush(self):
            if failure == "flush":
                raise OSError("flush-probe")
            return self.stream.flush()

        def close(self):
            self.stream.close()
            if failure == "close":
                raise OSError("close-probe")

    def fail_write(text):
        writes.append(text)
        if failure == "write":
            raise OSError("write-probe" + "!" * 1000)
        if failure == "handle_error":
            try:
                raise OSError("handle-error-probe")
            except OSError:
                sink.handleError(record("failed"))
            return
        original_write(text)
        if failure == "partial_write":
            raise OSError("partial-write-probe")

    with monkeypatch.context() as patch:
        patch.setattr(sink, "do_write", fail_write)
        patch.setattr(sink, "do_open", lambda *a, **kw: FailingStream(original_open(*a, **kw)))
        result = sink.write_batch(entries("unconfirmed-one", "unconfirmed-two"))
    assert result.outcomes == (False, False)
    assert result.written == 0 and result.write_errors == 2
    assert result.last_error and len(result.last_error) <= 512
    assert len(writes) == 1
    assert sink.stream is None and not sink.is_locked
    assert sink.write_batch(entries("recovered")).outcomes == (True,)
    with open(sink.baseFilename, encoding="utf-8") as output:
        content = output.read()
    assert content.count("unconfirmed-one") <= 1
    assert content.count("unconfirmed-two") <= 1
    assert content.count("recovered") == 1
    assert capsys.readouterr().err == ""


def test_foreign_shutdown_does_not_flush_close_or_wait_on_active_sink(sink, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    results = []
    original_write = sink.do_write

    def blocked_write(text):
        entered.set()
        assert release.wait(5)
        original_write(text)

    monkeypatch.setattr(sink, "do_write", blocked_write)

    def consume():
        try:
            results.append(sink.write_batch(entries("owned")))
        finally:
            sink.close_owned()

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert entered.wait(5)
        assert sink.lock is None and sink.is_locked
        # Exactly the handler operations used by logging.shutdown, without
        # shutting down unrelated test/root handlers.
        sink.acquire()
        try:
            sink.flush()
            sink.close()
        finally:
            sink.release()
        assert sink.is_locked and not sink._closed
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0].written == 1 and sink._closed
