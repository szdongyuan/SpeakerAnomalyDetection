import atexit
import copy
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.util import Finalize
from typing import Sequence
from concurrent_log_handler import ConcurrentRotatingFileHandler

from consts.running_consts import (
    LOG_DIR, LOG_MAPPING, DEFAULT_LOG, LOG_QUEUE_CAPACITY, LOG_BATCH_SIZE,
    LOG_SHUTDOWN_TIMEOUT, LOG_DROP_REPORT_INTERVAL,
)


@dataclass(frozen=True)
class _BatchWriteResult:
    """Outcomes in input order; failures are completed, never retried records."""

    outcomes: tuple[bool, ...]
    last_error: str | None = None

    @property
    def written(self) -> int:
        return sum(self.outcomes)

    @property
    def write_errors(self) -> int:
        return len(self.outcomes) - self.written


class _BatchFileHandler(ConcurrentRotatingFileHandler):
    """One destination, exclusively used and closed by one consumer thread.

    This handler is never attached to a Logger. Standard logging.shutdown may
    visit it, but must neither wait for its consumer nor touch an active stream.
    There is no redundant Handler lock; the library's cross-process lock still
    protects every rotation/write interval. Ownership starts at the first batch,
    allowing route construction before the consumer starts.
    """

    def __init__(self, *args, **kwargs):
        self._owner = None
        self._batch_error = None
        kwargs["keep_file_open"] = False
        super().__init__(*args, **kwargs)

    def createLock(self):
        self.lock = None

    def _claim_owner(self):
        current = threading.current_thread()
        if self._owner is None:
            self._owner = current
        elif self._owner is not current:
            raise RuntimeError("Batch sink may only be used by its consumer")

    def flush(self):
        if self._owner is threading.current_thread():
            super().flush()

    def close(self):
        if self._owner is threading.current_thread():
            super().close()

    def close_owned(self):
        """Consumer cleanup, including a sink that has not received records."""
        self._claim_owner()
        self.close()

    def _close(self):
        # The library's _close suppresses failures. Propagate them to the batch
        # boundary so an unconfirmed flush/close cannot be counted as written.
        try:
            if self.stream is not None:
                self.stream.close()
        finally:
            self.stream = None

    def handleError(self, record):
        # Some library paths report rather than raise. Never recurse into
        # logging or use Handler.handleError's synchronous stderr fallback.
        error = sys.exc_info()[1]
        self._batch_error = (f"{type(error).__name__}: {error}"[:512]
                             if error is not None else "File handler reported an error")

    def _shouldRollover(self):
        # Missing files have size zero. The library fallback opens them merely
        # to inspect size, causing a second open during the first actual write.
        if self.maxBytes <= 0:
            return False
        try:
            return os.path.getsize(self.baseFilename) >= self.maxBytes
        except FileNotFoundError:
            return False

    def write_batch(self, entries: Sequence[tuple[logging.LogRecord, logging.Formatter]]) -> _BatchWriteResult:
        """Write compatible-destination entries with their own formatters."""
        self._claim_owner()
        if self._closed:
            raise RuntimeError("Batch sink is closed")
        self._batch_error = None
        outcomes = [False] * len(entries)
        formatted = []
        last_error = None
        for index, (record, formatter) in enumerate(entries):
            try:
                formatted.append((index, formatter.format(record)))
            except Exception as error:
                # Formatters are external callbacks; one bad record must not
                # discard valid siblings or recursively log its own failure.
                last_error = f"{type(error).__name__}: {error}"[:512]
        if formatted:
            try:
                try:
                    self._do_lock()
                    if self.shouldRollover(entries[formatted[0][0]][0]):
                        self.doRollover()
                    # do_write supplies the last terminator itself.
                    self.do_write(self.terminator.join(text for _, text in formatted))
                finally:
                    try:
                        self._close()
                    finally:
                        self._do_unlock()
            except Exception as error:
                # External file/rotation/locking code can fail after writing
                # some bytes. Account the entire unconfirmed group; no retry.
                self._batch_error = f"{type(error).__name__}: {error}"[:512]
            if self._batch_error is None:
                for index, _ in formatted:
                    outcomes[index] = True
            else:
                last_error = self._batch_error
        return _BatchWriteResult(tuple(outcomes), last_error)


@dataclass
class _LogRoute:
    destination: tuple
    formatter: logging.Formatter
    active: bool = True
    last_sequence: int = 0


@dataclass(frozen=True)
class _QueuedRecord:
    sequence: int
    route: _LogRoute
    record: logging.LogRecord


@dataclass
class _DropReportState:
    route: _LogRoute
    full: int = 0
    closed: int = 0
    reported_full: int = 0
    reported_closed: int = 0
    next_report_at: float = 0.0


class _LogDispatcher:
    """Process-local FIFO; only its consumer ever constructs or uses sinks."""

    def __init__(self):
        self._condition = threading.Condition()
        self._queue = queue.Queue(maxsize=LOG_QUEUE_CAPACITY)
        self._routes = {}
        self._sinks = {}
        self._drop_reports = {}
        self._reports_finalized = False
        self._accepted = self._written = self._completed = self._taken = 0
        self._dropped_full = self._dropped_closed = self._write_errors = 0
        self._snapshot_errors = 0
        self._last_error = None
        self._closing = False
        self._exit_deadline = None
        # Keep each captured boundary until completed, not merely the largest:
        # a later flush must not extend an earlier caller's partial batch.
        self._barriers = set()
        self._thread = threading.Thread(
            target=self._consume, name="project-log-consumer", daemon=True)

    def register(self, name, info):
        destination = (os.path.normcase(os.path.abspath(info.get(
            "log_name", os.path.join(LOG_DIR, "main.log")))),
            info.get("max_size", 1 << 20), info.get("backup_count", 10))
        formatter = logging.Formatter(info.get("log_format"))
        os.makedirs(os.path.dirname(destination[0]), exist_ok=True)
        route = _LogRoute(destination, formatter)
        with self._condition:
            if self._closing:
                raise RuntimeError("Project logging consumer shut down during route registration")
            self._routes[name] = route
            self._drop_reports.setdefault(destination, _DropReportState(route))
        return route

    def _count_drop(self, route, *, full):
        # Called under the condition. Retain only counters and a route, never
        # the rejected record, its arguments, or a producer-created diagnostic.
        if not self._reports_finalized:
            state = self._drop_reports[route.destination]
            state.route = route
            if full:
                state.full += 1
            else:
                state.closed += 1
            self._condition.notify_all()

    def admit(self, route, record):
        with self._condition:
            if self._closing or not route.active:
                self._dropped_closed += 1
                self._count_drop(route, full=False)
                return
            entry = _QueuedRecord(self._accepted + 1, route, record)
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                self._dropped_full += 1
                self._count_drop(route, full=True)
                return
            self._accepted = route.last_sequence = entry.sequence
            self._condition.notify_all()

    def _request_barrier(self, target):
        if target > self._completed:
            self._barriers.add(target)
        self._condition.notify_all()

    def record_snapshot_error(self, error):
        # Do not stringify the exception: user conversion code may itself have
        # raised an exception whose __str__ fails. Keep a bounded type diagnostic.
        diagnostic = f"Message snapshot failed ({type(error).__name__})"[:512]
        with self._condition:
            self._snapshot_errors += 1
            self._last_error = diagnostic

    def request_flush(self):
        """Capture an accepted boundary; the existing consumer owns all I/O."""
        with self._condition:
            target = self._accepted
            self._request_barrier(target)
            return target

    def flush(self, timeout):
        with self._condition:
            target = self._accepted
            self._request_barrier(target)
            return self._condition.wait_for(
                lambda: self._completed >= target, timeout=max(0, timeout))

    def shutdown(self, timeout):
        with self._condition:
            self._closing = True
            self._request_barrier(self._accepted)
        self._thread.join(timeout=max(0, timeout))
        return not self._thread.is_alive()

    def shutdown_for_exit(self):
        # logging.shutdown visits every queue handler and can run again during
        # atexit, after multiprocessing finalizers. All these implicit closes
        # share one deadline, rather than spending a full budget per handler.
        with self._condition:
            if self._exit_deadline is None:
                self._exit_deadline = time.monotonic() + LOG_SHUTDOWN_TIMEOUT
            remaining = max(0, self._exit_deadline - time.monotonic())
        return self.shutdown(remaining)

    def close_route(self, route, timeout):
        deadline = time.monotonic() + max(0, timeout)
        with self._condition:
            route.active = False
            target = route.last_sequence
            last_route = not any(r.active for r in self._routes.values())
            if last_route:
                self._closing = True
            self._request_barrier(target)
            if not last_route:
                return self._condition.wait_for(
                    lambda: self._completed >= target,
                    timeout=max(0, deadline - time.monotonic()))
        self._thread.join(timeout=max(0, deadline - time.monotonic()))
        return not self._thread.is_alive()

    def stats(self):
        with self._condition:
            return dict(accepted=self._accepted, written=self._written,
                        dropped_full=self._dropped_full,
                        dropped_closed=self._dropped_closed,
                        write_errors=self._write_errors,
                        snapshot_errors=self._snapshot_errors,
                        pending=self._accepted - self._completed,
                        last_error=self._last_error,
                        consumer_alive=self._thread.is_alive())

    def _write_destination(self, destination, entries):
        # Only the consumer crosses this external construction/write boundary.
        # Both business records and diagnostics get the same failure handling;
        # their accounting and retry policies remain separate at the caller.
        try:
            sink = self._sinks.get(destination)
            if sink is None:
                filename, max_bytes, backups = destination
                sink = _BatchFileHandler(filename=filename, maxBytes=max_bytes,
                                         backupCount=backups)
                self._sinks[destination] = sink
            return sink.write_batch(entries)
        except Exception as error:
            return _BatchWriteResult((False,) * len(entries),
                                     f"{type(error).__name__}: {error}"[:512])

    def _write_batch(self, batch):
        groups = {}
        for entry in batch:
            groups.setdefault(entry.route.destination, []).append(
                (entry.record, entry.route.formatter))
        written = errors = 0
        last_error = None
        for destination, entries in groups.items():
            result = self._write_destination(destination, entries)
            written += result.written
            errors += result.write_errors
            if result.last_error is not None:
                last_error = result.last_error
        with self._condition:
            self._written += written
            self._write_errors += errors
            if last_error is not None:
                self._last_error = last_error
            self._completed = batch[-1].sequence
            self._barriers.difference_update(
                target for target in tuple(self._barriers) if target <= self._completed)
            self._condition.notify_all()

    def _snapshot_drop_reports(self, *, final=False):
        # Condition held: capture finite totals, so later drops cannot be
        # consumed by an in-flight write or keep the shutdown loop alive.
        now = time.monotonic()
        reports = []
        delay = None
        for destination, state in self._drop_reports.items():
            if state.full == state.reported_full and state.closed == state.reported_closed:
                continue
            remaining = state.next_report_at - now
            if final or remaining <= 0:
                reports.append((destination, state.route.formatter, state.full, state.closed,
                                state.full - state.reported_full,
                                state.closed - state.reported_closed))
            else:
                delay = remaining if delay is None else min(delay, remaining)
        return reports, delay

    def _write_drop_reports(self, reports):
        for destination, formatter, full, closed, delta_full, delta_closed in reports:
            record = logging.LogRecord(
                "project-log-consumer", logging.WARNING, __file__, 0,
                "Logging dropped records: pid=%d; queue_full delta=%d cumulative=%d; "
                "closed delta=%d cumulative=%d",
                (os.getpid(), delta_full, full, delta_closed, closed), None)
            result = self._write_destination(destination, [(record, formatter)])
            with self._condition:
                state = self._drop_reports[destination]
                if result.written == 1:
                    state.reported_full = full
                    state.reported_closed = closed
                if result.last_error is not None:
                    self._last_error = result.last_error
                state.next_report_at = time.monotonic() + LOG_DROP_REPORT_INTERVAL
                self._condition.notify_all()

    def _consume(self):
        batch = []
        while True:
            reports = []
            final = False
            with self._condition:
                while True:
                    boundary = min(self._barriers, default=0)
                    if batch and (len(batch) >= LOG_BATCH_SIZE
                                  or batch[-1].record.levelno >= logging.ERROR
                                  or (boundary and batch[-1].sequence >= boundary)
                                  or (self._closing and self._queue.empty())):
                        break
                    if not self._closing:
                        reports, report_delay = self._snapshot_drop_reports()
                        if reports:
                            break
                    if not self._queue.empty():
                        batch.append(self._queue.get_nowait())
                        self._taken += 1
                        self._condition.notify_all()
                    elif self._closing:
                        reports, _ = self._snapshot_drop_reports(final=True)
                        self._reports_finalized = final = True
                        break
                    else:
                        self._condition.wait(report_delay)
            if reports:
                self._write_drop_reports(reports)
            elif batch:
                self._write_batch(batch)
                batch = []
            if final:
                break
        for sink in self._sinks.values():
            try:
                sink.close_owned()
            except Exception as error:
                # Consumer retains ownership even on close failure. Preserve
                # diagnostics; no records are newly reported as written here.
                with self._condition:
                    self._last_error = f"{type(error).__name__}: {error}"[:512]


class _ProjectQueueHandler(logging.Handler):
    """Admission only; no formatter, file operation, or stderr fallback."""

    def __init__(self, runtime, route):
        super().__init__(logging.INFO)
        self.runtime = runtime
        self.route = route

    def createLock(self):
        # All shared state is guarded by the dispatcher's short condition.
        self.lock = None

    def emit(self, record):
        # Select the nearest project route in the propagation chain. This also
        # respects inactive child routes without mutating the shared record or
        # changing propagation to third-party/root handlers.
        source = logging.getLogger(record.name)
        while source is not None:
            nearest = next((handler for handler in source.handlers
                            if isinstance(handler, _ProjectQueueHandler)), None)
            if nearest is not None:
                if nearest is not self:
                    return
                break
            if not source.propagate:
                break
            source = source.parent
        try:
            snapshot = copy.copy(record)
            message = record.getMessage()
        except Exception as error:
            # Record copying and message conversion can execute user code.
            # Reject before admission; preserve the original for other handlers
            # and never fall back to file I/O or logging's stderr handleError.
            self.runtime.record_snapshot_error(error)
            return
        snapshot.msg = message
        snapshot.args = None
        self.runtime.admit(self.route, snapshot)

    def close(self):
        # During interpreter exit, logging.shutdown may precede multiprocessing
        # finalizers depending on import order. Defer only project draining so
        # those producers can still submit their tails. Explicit runtime calls
        # to logging.shutdown retain immediate, bounded shutdown semantics.
        if not (LogManager._interpreter_exiting
                and LogManager._process_finalizer.still_active()):
            self.runtime.shutdown_for_exit()
        super().close()


class LogManager(object):
    _initialization_lock = threading.Lock()
    _runtime = None
    _exit_hooks_registered = False
    _process_finalizer = None
    _interpreter_exiting = False
    _forced_exit = False

    @classmethod
    def seal_for_forced_exit(cls):
        """Atomically include initialization in flight and forbid reacquisition.

        Only the forced-exit coordinator calls this. Ordinary shutdown retains
        its existing explicit reacquisition behavior.
        """
        with cls._initialization_lock:
            cls._forced_exit = True
            return cls._runtime

    @classmethod
    def _shutdown_at_exit(cls):
        cls._interpreter_exiting = True
        # atexit is LIFO: this callback may precede multiprocessing's registered
        # exit function in the main interpreter. Let that function run ordinary
        # producer finalizers first; our negative-priority finalizer then drains.
        if cls._process_finalizer.still_active():
            return
        cls._shutdown_after_producers()

    @classmethod
    def _shutdown_after_producers(cls):
        runtime = cls._runtime
        if runtime is not None:
            runtime.shutdown_for_exit()

    @classmethod
    def _register_exit_hooks(cls):
        # Register lazily once for the process, resolving the current runtime
        # when invoked. Reacquisition cannot retain old consumers in callbacks.
        # Negative priority follows ordinary multiprocessing producer finalizers
        # and child joins; daemon consumers are still alive at this boundary.
        if not cls._exit_hooks_registered:
            cls._process_finalizer = Finalize(
                None, cls._shutdown_after_producers, exitpriority=-10)
            atexit.register(cls._shutdown_at_exit)
            cls._exit_hooks_registered = True

    def __init__(self, thread_holder="core"):
        self.logger = self.set_log_handler(thread_holder)

    @classmethod
    def set_log_handler(cls, thread_holder):
        with cls._initialization_lock:
            if cls._forced_exit:
                raise RuntimeError("Project logging is sealed for forced process exit")
            logger = logging.getLogger(thread_holder)
            runtime = cls._runtime
            if runtime is not None and runtime._closing:
                if runtime._thread.is_alive():
                    raise RuntimeError("Previous project logging consumer is still shutting down")
                runtime = None
            if runtime is not None:
                for handler in logger.handlers:
                    if (isinstance(handler, _ProjectQueueHandler)
                            and handler.runtime is runtime and handler.route.active):
                        return logger
            fresh = runtime is None
            if fresh:
                runtime = _LogDispatcher()
            route = runtime.register(thread_holder, LOG_MAPPING.get(thread_holder, DEFAULT_LOG))
            if fresh:
                try:
                    runtime._thread.start()
                except Exception:
                    # Thread.start is the initialization boundary: nothing is
                    # published or attached on failure, and no sink exists yet.
                    route.active = False
                    raise
                cls._runtime = runtime
                cls._register_exit_hooks()
            handler = _ProjectQueueHandler(runtime, route)
            for old in logger.handlers[:]:
                if isinstance(old, _ProjectQueueHandler):
                    logger.removeHandler(old)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            return logger

    @classmethod
    def request_flush(cls):
        """Request a partial batch without waiting or initializing a runtime.

        The returned watermark is an admission boundary, not a write guarantee.
        This remains safe after sealing and while the consumer is shutting down.
        """
        runtime = cls._runtime
        return 0 if runtime is None else runtime.request_flush()

    @classmethod
    def flush(cls, timeout=LOG_SHUTDOWN_TIMEOUT):
        runtime = cls._runtime
        return True if runtime is None else runtime.flush(timeout)

    @classmethod
    def shutdown_all(cls, timeout=LOG_SHUTDOWN_TIMEOUT):
        runtime = cls._runtime
        return True if runtime is None else runtime.shutdown(timeout)

    @classmethod
    def get_async_stats(cls):
        runtime = cls._runtime
        if runtime is not None:
            return runtime.stats()
        return dict(accepted=0, written=0, dropped_full=0, dropped_closed=0,
                    write_errors=0, snapshot_errors=0, pending=0, last_error=None,
                    consumer_alive=False)

    def info(self, info_str):
        return self.logger.info(info_str)

    def warning(self, warning_str):
        return self.logger.warning(warning_str)

    def error(self, error_str):
        return self.logger.error(error_str)

    def shut_down(self):
        for handler in self.logger.handlers:
            if isinstance(handler, _ProjectQueueHandler):
                handler.runtime.close_route(handler.route, LOG_SHUTDOWN_TIMEOUT)
