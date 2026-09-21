"""Isolate real project file logging without acquiring handlers at import."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace

from base import log_manager


def managed_handlers(logger):
    """Project routes, independent of the consumer-owned file sink type."""
    return [handler for handler in logger.handlers
            if isinstance(handler, log_manager._ProjectQueueHandler)]


@contextmanager
def isolated_project_logger(tmp_path, monkeypatch):
    manager = log_manager.LogManager
    previous_runtime = manager._runtime
    if previous_runtime is not None:
        assert previous_runtime.flush(2), "pre-existing runtime did not drain"
    logger = logging.getLogger("core")
    originals = {
        name: (item, item.handlers[:], item.level, item.propagate, item.disabled)
        for name, item in logging.Logger.manager.loggerDict.copy().items()
        if isinstance(item, logging.Logger)
    }
    for item, handlers, *_ in originals.values():
        for handler in handlers:
            if item is logger or isinstance(handler, log_manager._ProjectQueueHandler):
                item.removeHandler(handler)
    manager._runtime = None
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger.disabled = False
    path = tmp_path / "logs" / "main.log"
    config = {"log_name": str(path),
              "log_format": "%(name)s %(levelname)s %(message)s [%(filename)s:%(lineno)d]"}
    monkeypatch.setattr(log_manager, "LOG_DIR", str(path.parent))
    monkeypatch.setattr(log_manager, "DEFAULT_LOG", config)
    monkeypatch.setattr(log_manager, "LOG_MAPPING", {
        "core": config,
        "debug": dict(config, log_name=str(path.with_name("debug.log"))),
        "test": dict(config, log_name=str(path.with_name("test.log"))),
    })
    try:
        yield SimpleNamespace(logger=logger, path=path)
    finally:
        assert manager.shutdown_all(2), "test-owned logging runtime did not stop"
        for name, item in logging.Logger.manager.loggerDict.copy().items():
            if not isinstance(item, logging.Logger):
                continue
            original = originals.get(name, (item, [], logging.NOTSET, True, False))
            item.handlers[:] = original[1]
            item.setLevel(original[2])
            item.propagate = original[3]
            item.disabled = original[4]
        manager._runtime = previous_runtime


def spawn_sender_fault(directory):
    """Configure logging in the child without importing the pytest test module."""
    from pathlib import Path
    import queue
    import threading

    import pytest

    from base.recording_worker import _send_loop

    class BrokenConnection:
        def send(self, value):
            raise ValueError("sender-serialization-probe")

    outgoing = queue.Queue()
    outgoing.put(object())
    broken = threading.Event()
    with pytest.MonkeyPatch.context() as monkeypatch:
        with isolated_project_logger(Path(directory), monkeypatch) as state:
            _send_loop(BrokenConnection(), outgoing, broken)
            assert broken.is_set()
            assert outgoing.unfinished_tasks == 0
            assert log_manager.LogManager.flush(timeout=2)
