"""Isolate real project file logging without acquiring handlers at import."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace

from base import log_manager


@contextmanager
def isolated_project_logger(tmp_path, monkeypatch):
    logger = logging.getLogger("core")
    original = (logger.handlers[:], logger.level, logger.propagate, logger.disabled)
    for handler in original[0]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger.disabled = False
    path = tmp_path / "logs" / "main.log"
    monkeypatch.setattr(log_manager, "LOG_DIR", str(path.parent))
    monkeypatch.setattr(log_manager, "LOG_MAPPING", {"core": {
        "log_name": str(path),
        "log_format": "%(name)s %(levelname)s %(message)s [%(filename)s:%(lineno)d]",
    }})
    try:
        yield SimpleNamespace(logger=logger, path=path)
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if handler not in original[0]:
                handler.close()
        for handler in original[0]:
            logger.addHandler(handler)
        logger.setLevel(original[1])
        logger.propagate = original[2]
        logger.disabled = original[3]


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
            for handler in state.logger.handlers:
                handler.flush()
