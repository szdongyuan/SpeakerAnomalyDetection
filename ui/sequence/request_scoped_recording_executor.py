"""Bounded background execution for request-owned result work."""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time


@dataclass(frozen=True)
class RequestScopedExecutionOutcome:
    request_id: str
    value: object = None
    error: BaseException | None = None


class RequestScopedDispatchError(RuntimeError):
    """The job finished, but its GUI terminal could not be marshalled."""


@dataclass(frozen=True)
class _RequestScopedJob:
    request_id: str
    work: object
    deliver: object
    generation: int
    implicit_reservation: bool = False


class RequestScopedRecordingExecutor:
    """Run bounded request-keyed jobs and marshal each terminal exactly once."""

    SUBMIT_ACCEPTED = "accepted"
    SUBMIT_FULL = "full"
    SUBMIT_CLOSED = "closed"
    SUBMIT_DUPLICATE = "duplicate"

    def __init__(self, *, dispatch, capacity=2, max_workers=1,
                 dispatch_failure_notify=None):
        if not callable(dispatch):
            raise TypeError("request-scoped executor dispatch must be callable")
        capacity = int(capacity)
        max_workers = int(max_workers)
        if capacity < 1 or max_workers < 1 or max_workers > capacity:
            raise ValueError("invalid request-scoped executor bounds")
        self._dispatch = dispatch
        if dispatch_failure_notify is not None and not callable(
                dispatch_failure_notify):
            raise TypeError("dispatch failure notifier must be callable")
        self._dispatch_failure_notify = dispatch_failure_notify
        self._capacity = capacity
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._reservations = set()
        self._pending = {}
        self._running = set()
        self._terminals = {}
        self._dispatch_failures = {}
        self._closed = False
        self._shutdown_retained = ()
        self._generation = 0
        self._workers = tuple(
            threading.Thread(
                target=self._run_worker, name=f"recording-result-{index}",
                daemon=True)
            for index in range(max_workers))
        for worker in self._workers:
            worker.start()

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

    def _occupied_ids_locked(self):
        return (set(self._reservations) | set(self._pending)
                | set(self._running) | set(self._terminals))

    @property
    def reserved_count(self):
        with self._lock:
            return len(self._reservations)

    @property
    def occupied_count(self):
        with self._lock:
            return len(self._occupied_ids_locked())

    @property
    def can_reserve(self):
        with self._lock:
            return (not self._closed
                    and len(self._occupied_ids_locked()) < self._capacity)

    def reserve(self, request_id):
        return self.reserve_with_status(request_id) == self.SUBMIT_ACCEPTED

    def reserve_with_status(self, request_id):
        key = str(request_id or "")
        if not key:
            raise TypeError("request-scoped reservation requires key")
        with self._lock:
            if self._closed:
                return self.SUBMIT_CLOSED
            if key in self._occupied_ids_locked():
                return self.SUBMIT_DUPLICATE
            if len(self._occupied_ids_locked()) >= self._capacity:
                return self.SUBMIT_FULL
            self._reservations.add(key)
        return self.SUBMIT_ACCEPTED

    def release_reservation(self, request_id):
        key = str(request_id or "")
        with self._lock:
            if key not in self._reservations:
                return False
            self._reservations.remove(key)
            return True

    def submit(self, request_id, work, deliver):
        return self.submit_with_status(request_id, work, deliver) == self.SUBMIT_ACCEPTED

    def submit_with_status(self, request_id, work, deliver):
        key = str(request_id or "")
        if not key or not callable(work) or not callable(deliver):
            raise TypeError("request-scoped job requires key, work and delivery")
        with self._lock:
            if self._closed:
                return self.SUBMIT_CLOSED
            if key in self._pending or key in self._running or key in self._terminals:
                return self.SUBMIT_DUPLICATE
            implicit_reservation = key not in self._reservations
            if implicit_reservation:
                if key in self._occupied_ids_locked():
                    return self.SUBMIT_DUPLICATE
                if len(self._occupied_ids_locked()) >= self._capacity:
                    return self.SUBMIT_FULL
                self._reservations.add(key)
            job = _RequestScopedJob(
                key, work, deliver, self._generation, implicit_reservation)
            self._pending[key] = job
            # Queue.put is non-blocking for this intentionally unbounded
            # internal queue. Keep registration and enqueue atomic with close,
            # so shutdown cannot insert its sentinel between them.
            self._queue.put(job)
        return self.SUBMIT_ACCEPTED

    def cancel(self, request_id):
        with self._lock:
            job = self._pending.pop(str(request_id or ""), None)
            if job is not None and job.implicit_reservation:
                self._reservations.discard(job.request_id)
            return job is not None

    def _run_worker(self):
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                with self._lock:
                    if self._pending.get(job.request_id) is not job:
                        continue
                    self._running.add(job.request_id)
                try:
                    outcome = RequestScopedExecutionOutcome(
                        job.request_id, value=job.work())
                except BaseException as error:
                    # This is the terminal boundary for arbitrary analysis and
                    # durable publishers. Preserve it for GUI diagnostics.
                    outcome = RequestScopedExecutionOutcome(
                        job.request_id, error=error)
                finally:
                    with self._lock:
                        self._running.discard(job.request_id)
                self._queue_terminal(job, outcome)
            finally:
                self._queue.task_done()

    def _queue_terminal(self, job, outcome):
        with self._lock:
            if self._pending.get(job.request_id) is not job:
                return
            self._pending.pop(job.request_id, None)
            self._terminals[job.request_id] = job
            closed = self._closed
        if closed:
            return
        callback = lambda: self._deliver_if_current(job, outcome)
        try:
            self._dispatch(callback)
        except Exception as dispatch_error:
            # A failed GUI marshaller is an external contract boundary.  Do
            # not let it terminate the sole worker or silently strand this
            # request.  Re-dispatch one diagnosable failure terminal; a
            # persistently broken dispatcher is retained for owner polling.
            failure = RequestScopedDispatchError(
                "request-scoped terminal dispatcher failed: "
                f"{dispatch_error}")
            failure.__cause__ = dispatch_error
            failed_outcome = RequestScopedExecutionOutcome(
                job.request_id, error=failure)
            failed_callback = lambda: self._deliver_if_current(
                job, failed_outcome)
            try:
                self._dispatch(failed_callback)
            except Exception as repeated_error:
                with self._lock:
                    self._dispatch_failures[job.request_id] = (
                        job, failed_outcome, repeated_error)
                notify = self._dispatch_failure_notify
                if callable(notify):
                    try:
                        notify(job.request_id)
                    except Exception as notify_error:
                        # Retention is the state-consistency fallback when the
                        # owner notification boundary itself is unavailable.
                        with self._lock:
                            retained = self._dispatch_failures.get(job.request_id)
                            if retained is not None:
                                self._dispatch_failures[job.request_id] = (
                                    retained[0], retained[1], notify_error)

    def _deliver_if_current(self, job, outcome):
        # The GUI callback may sit in the Qt queue after shutdown. Validate the
        # generation when it executes, not merely when it is queued.
        with self._lock:
            if (self._closed or job.generation != self._generation
                    or self._terminals.get(job.request_id) is not job):
                return
        try:
            job.deliver(outcome)
        finally:
            with self._lock:
                if self._terminals.get(job.request_id) is job:
                    self._terminals.pop(job.request_id, None)
                if job.implicit_reservation:
                    self._reservations.discard(job.request_id)

    @property
    def dispatch_failure_ids(self):
        with self._lock:
            return tuple(self._dispatch_failures)

    def reconcile_dispatch_failures(self, request_id=None):
        """Deliver retained failures when called from the owner/GUI thread."""
        with self._lock:
            if request_id is None:
                retained = tuple(self._dispatch_failures.values())
                self._dispatch_failures.clear()
            else:
                value = self._dispatch_failures.pop(str(request_id), None)
                retained = () if value is None else (value,)
        delivered = []
        for job, outcome, _dispatch_error in retained:
            self._deliver_if_current(job, outcome)
            delivered.append(job.request_id)
        return tuple(delivered)

    def shutdown(self, *, wait=False, timeout=None):
        with self._lock:
            if self._closed:
                return self._shutdown_retained
            self._closed = True
            self._generation += 1
            retained = self._occupied_ids_locked()
            self._reservations.clear()
            self._pending.clear()
            self._terminals.clear()
            self._dispatch_failures.clear()
        for _worker in self._workers:
            self._queue.put(None)
        if wait:
            deadline = None if timeout is None else time.monotonic() + max(
                0.0, float(timeout))
            for worker in self._workers:
                remaining = None if deadline is None else max(
                    0.0, deadline - time.monotonic())
                worker.join(remaining)
        with self._lock:
            retained.update(self._running)
            self._shutdown_retained = tuple(sorted(retained))
            return self._shutdown_retained
