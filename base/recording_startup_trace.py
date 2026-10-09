"""Bounded attempt-owned startup timing, delivered through a borrowed logger.

This helper does not acquire a logger, create threads, flush, or retry delivery.
It must only be called off real-time audio callbacks. A callback's saved first
block timestamp/thread can be passed to ``mark`` later by its existing owner.
"""
from contextlib import contextmanager
from itertools import islice
import math
import os
import threading
from time import perf_counter_ns
from uuid import uuid4

from consts.recording_startup_consts import (
    CHILD_EVENT_BUDGET, CONTEXT_FIELDS, CONTROL_EVENTS, EVENT_BEGIN, EVENT_END,
    EVENT_REQUEST_LINK, EVENT_SUMMARY, FIRST_BLOCK_EVENTS, MAX_COUNTER,
    MAX_FIELD_LENGTH, MAX_MARK_FIELDS, PARENT_EVENT_BUDGET, RESERVED_FIELDS,
)


def _small_value(value):
    """Never stringify user objects, arrays, containers, or arbitrary config."""
    if type(value) is str:
        return "".join("_" if c.isspace() or c == "=" else c
                       for c in value[:MAX_FIELD_LENGTH]) or "unknown"
    if type(value) is bool:
        return str(value)
    if type(value) is int and value.bit_length() <= 128:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return str(value)
    return "unknown"


class _StageObservation:
    """Bounded diagnostic result for failures handled inside a stage body."""
    __slots__ = ("outcome", "error_type", "reason")

    def __init__(self):
        self.outcome, self.error_type, self.reason = "ok", "unknown", "unknown"

    def observe(self, outcome, *, error_type="unknown", reason="unknown"):
        # Keep only small scalar strings, never exception/traceback ownership.
        self.outcome = _small_value(outcome)
        self.error_type = _small_value(error_type)
        self.reason = _small_value(reason)


class RecordingStartupTrace:
    """One parent or child attempt, with independent monotonic timing.

    Each stage name and mark event is accepted once. ``stage`` infers nesting
    within the calling thread, or accepts an explicit parent. Pass ``domain``
    at each execution boundary (GUI, service_sender, worker_control, etc.).
    ``finish`` freezes context/identity and emits once; only the two first-block
    marks may follow a successful finish. Missing first blocks stay unobserved.
    """

    def __init__(self, logger, *, process, clock_ns=None, trace_id=None, request_id=None):
        self._logger = logger
        self._clock = clock_ns or perf_counter_ns
        self._start_ns = self._clock()
        self.trace_id = _small_value(trace_id) if trace_id is not None else uuid4().hex
        self.request_id = _small_value(request_id) if request_id is not None else None
        self.process = _small_value(process)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._budget = CHILD_EVENT_BUDGET if process == "child" else PARENT_EVENT_BUDGET
        # Reserve summary, request link and both first blocks. Stages reserve
        # both endpoints atomically so saturation never drops only their end.
        self._used = 4
        self._stages = {}
        self._active = {}
        self._marks = set()
        self._first = {}
        self._context = dict.fromkeys(CONTEXT_FIELDS, "unknown")
        self._outcome = None
        self._last_stage = "unknown"
        self.dropped_events = 0
        self.dropped_fields = 0
        self.delivery_failures = 0

    def _drop_locked(self):
        self.dropped_events = min(MAX_COUNTER, self.dropped_events + 1)

    def _snapshot_locked(self, event, timestamp_ns, domain, *, thread_id=None, thread_name=None, **fields):
        record = dict(event=event, trace_id=self.trace_id,
                      request_id=self.request_id or "unknown", process=self.process,
                      pid=self._pid, timestamp_ns=timestamp_ns,
                      thread_id=threading.get_ident() if thread_id is None else thread_id,
                      thread_name=threading.current_thread().name if thread_name is None else thread_name,
                      domain=domain, stage="unknown")
        record.update(self._context)
        record.update(fields)
        return record

    def _emit(self, record, *, stacklevel=3):
        # The only optional telemetry boundary: logger overrides, filters,
        # factories and handlers can raise arbitrary errors. Never retry or
        # recursively log; event reservations remain consumed on failure.
        try:
            text = " ".join(f"{key}={value if key == 'stage_ms' else _small_value(value)}"
                            for key, value in record.items())
            accepted = self._logger.info("recording_startup %s", text, stacklevel=stacklevel)
            if accepted is not False:
                return
        except Exception:
            accepted = False
        # Standard logging does not return dispatcher admission status. Real
        # project queue drops remain visible in LogManager's existing counters;
        # this counter covers raised delivery faults or explicit adapter refusal.
        with self._lock:
            self.delivery_failures = min(MAX_COUNTER, self.delivery_failures + 1)

    def set_context(self, **small_fields):
        """Fill the fixed request metadata vocabulary from existing snapshots."""
        fields = {key: _small_value(value) for key, value in islice(small_fields.items(), MAX_MARK_FIELDS)
                  if key in CONTEXT_FIELDS}
        with self._lock:
            if self._outcome is None:
                self._context.update(fields)
                self.dropped_fields = min(MAX_COUNTER, self.dropped_fields + len(small_fields) - len(fields))

    @contextmanager
    def stage(self, name, parent=None, *, domain="unknown"):
        """Yield an optional observation for handled failures or cancellation.

        Unhandled exceptions still propagate unchanged and take precedence over
        an explicit observation. Ignoring the yielded value retains old behavior.
        """
        observation = _StageObservation()
        name, domain = _small_value(name), _small_value(domain)
        parent = _small_value(parent) if parent is not None else None
        thread = threading.get_ident()
        start = self._clock()
        with self._lock:
            accepted = self._outcome is None and name not in self._stages and self._used + 2 <= self._budget
            if accepted:
                self._used += 2
                stack = self._active.setdefault(thread, [])
                parent = parent or (stack[-1] if stack else "unknown")
                stack.append(name)
                self._stages[name] = None
                self._last_stage = name
                begin = self._snapshot_locked(EVENT_BEGIN, start, domain, stage=name, parent=parent)
            else:
                self._drop_locked()
        if not accepted:
            yield observation
            return
        self._emit(begin, stacklevel=4)
        error_type = None
        try:
            yield observation
        except BaseException as error:
            # Observe only exceptions raised through this stage's body; rethrow
            # the same object, including cancellation/interrupt exceptions.
            error_type = type(error).__name__
            raise
        finally:
            end_ns = self._clock()
            elapsed = max(0, end_ns - start) / 1_000_000
            with self._lock:
                self._stages[name] = elapsed
                self._active[thread].pop()
                if not self._active[thread]:
                    del self._active[thread]
                end = self._snapshot_locked(EVENT_END, end_ns, domain, stage=name, parent=parent,
                                            elapsed_ms=elapsed,
                                            outcome="failed" if error_type else observation.outcome,
                                            error_type=error_type or observation.error_type,
                                            reason=observation.reason)
            self._emit(end, stacklevel=4)

    def mark(self, event, *, domain="unknown", timestamp_ns=None,
             thread_id=None, thread_name=None, **small_fields):
        """Record a once-only event; saved first-block source data is optional.

        ``timestamp_ns`` must use this process's clock. Explicit source thread
        fields describe observation; the LogRecord caller is the emitting owner.
        """
        event, domain = _small_value(event), _small_value(domain)
        timestamp_ns = _small_value(self._clock() if timestamp_ns is None else timestamp_ns)
        fields = {_small_value(key): _small_value(value)
                  for key, value in islice(small_fields.items(), MAX_MARK_FIELDS)
                  if key not in RESERVED_FIELDS}
        thread_id = _small_value(thread_id) if thread_id is not None else None
        thread_name = _small_value(thread_name) if thread_name is not None else None
        with self._lock:
            first = event in FIRST_BLOCK_EVENTS
            if (event in CONTROL_EVENTS or event in self._marks or event in self._first
                    or (self._outcome is not None and not (first and self._outcome == "started"))
                    or (not first and self._used >= self._budget)):
                self._drop_locked()
                return
            if first:
                self._first[event] = timestamp_ns
            else:
                self._used += 1
                self._marks.add(event)
            self.dropped_fields = min(MAX_COUNTER, self.dropped_fields + len(small_fields) - len(fields))
            record = self._snapshot_locked(event, timestamp_ns, domain, thread_id=thread_id,
                                           thread_name=thread_name, **fields)
        self._emit(record)

    def link_request(self, request_id, *, domain="unknown"):
        request_id, domain = _small_value(request_id), _small_value(domain)
        now = self._clock()
        with self._lock:
            if self.request_id is not None or self._outcome is not None:
                return
            self.request_id = request_id
            record = self._snapshot_locked(EVENT_REQUEST_LINK, now, domain)
        self._emit(record)

    def finish(self, outcome, *, domain="unknown"):
        outcome, domain = _small_value(outcome), _small_value(domain)
        now = self._clock()
        with self._lock:
            if self._outcome is not None:
                return
            self._outcome = outcome
            stages = tuple(self._stages.items())
            first_blocks = {event: self._first.get(event, "not_observed") for event in FIRST_BLOCK_EVENTS}
            record = self._snapshot_locked(EVENT_SUMMARY, now, domain, outcome=outcome,
                                           last_stage=self._last_stage,
                                           total_ms=max(0, now - self._start_ns) / 1_000_000,
                                           dropped_events=self.dropped_events, dropped_fields=self.dropped_fields,
                                           delivery_failures=self.delivery_failures, **first_blocks)
        # Inclusive stage durations are reported separately, never summed.
        record["stage_ms"] = ",".join(f"{name}:{elapsed if elapsed is not None else 'incomplete'}"
                                      for name, elapsed in stages) or "unknown"
        self._emit(record)
