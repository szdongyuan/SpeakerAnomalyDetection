"""Bounded, instance-owned delivery of optional recording timing records."""
import logging
import queue
import threading


class RecordingTimingLogger:
    def __init__(self, *, start_thread=None):
        self._queue = queue.Queue(64)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._start_thread = start_thread
        self.thread = None
        self.dropped = 0
        self.errors = 0
        self.last_error = None

    def info(self, logger, message, *args):
        return self.log(logger, logging.INFO, message, *args, stacklevel=3)

    def error(self, logger, message, *args):
        return self.log(logger, logging.ERROR, message, *args, stacklevel=3)

    def log(self, logger, level, message, *args, stacklevel=2):
        if not logger.isEnabledFor(level):
            return False
        # Construct on the producer so wall time, thread and caller stay intact.
        # Neither findCaller nor makeRecord acquires a handler/file lock.
        filename, line, function, stack = logger.findCaller(stacklevel=stacklevel)
        record = logger.makeRecord(logger.name, level, filename, line,
                                   message, args, None, function, sinfo=stack)
        with self._lock:
            if self._closed:
                self.dropped += 1
                return False
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._run, name="recording-timing-log", daemon=True)
                try:
                    if self._start_thread is None:
                        self.thread.start()
                    else:
                        self._start_thread(self.thread)
                except Exception as exc:
                    # External thread-start boundary. A custom starter can launch
                    # then raise; retain that consumer and never start a second.
                    self._remember_error(exc)
                    if self.thread.ident is None:
                        self._closed = True
                        self.dropped += 1
                        return False
            try:
                self._queue.put_nowait((logger, record))
            except queue.Full:
                self.dropped += 1
                return False
            self._wake.set()
            return True

    def close(self):
        """Reject offers and request drain; never join potentially blocked I/O."""
        with self._lock:
            self._closed = True
            self._wake.set()

    def _remember_error(self, exc):
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:512]

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                try:
                    logger, record = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    logger.handle(record)
                except Exception as exc:
                    # External logger/filter/handler implementations can raise
                    # arbitrary exceptions. Drop only this optional diagnostic,
                    # keep bounded evidence and continue draining without logging.
                    with self._lock:
                        self._remember_error(exc)
                finally:
                    self._queue.task_done()
            with self._lock:
                if self._closed and self._queue.empty():
                    return
