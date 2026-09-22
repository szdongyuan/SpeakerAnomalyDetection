"""Bounded recording evidence; callers retain ownership of all business state.

Register a small, fixed set of categories. begin/end use numeric tokens, never
hold the diagnostic lock across the operation, and do not log by default.
Use end(..., emit_slow=True) only AFTER releasing business locks. For a measured
interval without an active phase use observe. Only low-frequency milestones,
summaries and explicitly requested slow reports submit logs.

One helper can cover multiple threads/requests. The child lifecycle owner may
start its single sampler and close it with a bounded join outside release paths.
No thread starts on import/construction. A supplied timing_logger is borrowed;
its owner remains responsible for closing it. Direct delivery assumes the
project's asynchronous logger, never a synchronous file handler.
"""
from collections import deque
from dataclasses import dataclass
import json
import logging
import os
import threading
import time

from base.log_manager import LogManager


@dataclass(slots=True)
class _Phase:
    token: int
    stage: str
    thread: int
    request: str | int | None
    generation: str | int | None
    start_ns: int
    monotonic: float
    monotonic_source: str = "sampled"
    active_request: str | int | None = None


class _FastLane:
    """One producer, immutable phase publication; no lock on phase switches.

    CPython reference assignment publishes a complete tuple to the sampler.
    Registration/retirement uses the helper lock, never a business lock. Detailed
    statistics may be sampled; every >=100ms completed interval is retained.
    A hot-path owner may publish an immutable (stage, perf_ns) tuple directly to
    active after enter binds its thread. Those sub-stages do not increment the
    observed_count; count/sampled_count describe only recorded detailed samples.
    """
    def __init__(self, owner, request, generation):
        self.owner = owner
        self.request, self.generation = request, generation
        self.perf_ns = owner._perf_ns
        self.anchor_ns = self.perf_ns()
        self.anchor_monotonic = owner._monotonic()
        self.thread = None
        self.active = None
        self.counts = dict.fromkeys(owner._stats, 0)
        self.closed = False

    def enter(self, stage):
        if self.thread is None:
            self.thread = threading.get_ident()
        phase = (stage, self.perf_ns())
        self.counts[stage] += 1
        self.active = phase
        return phase

    def finish(self, phase, *, measured=True, restore=None, emit_slow=True, ended_ns=None):
        ended = self.perf_ns() if ended_ns is None else ended_ns
        self.active = restore
        elapsed = ended - phase[1]
        if measured or elapsed >= 100_000_000:
            self.owner.observe(
                phase[0], elapsed, started_ns=phase[1],
                monotonic=self.anchor_monotonic + (phase[1] - self.anchor_ns) / 1e9,
                request=self.request, generation=self.generation, emit_slow=emit_slow,
                monotonic_source="qpc_aligned")
        return elapsed

    def phase_snapshot(self):
        active = self.active
        if active is None or self.closed:
            return None
        stage, started = active
        return _Phase(0, stage, self.thread, self.request, self.generation, started,
                      self.anchor_monotonic + (started - self.anchor_ns) / 1e9, "qpc_aligned")

    def close(self):
        self.owner._close_lane(self)


def _primitive(value):
    # Never invoke payload repr/str, including subclasses with custom methods.
    if type(value) is str:
        return value[:128].replace("\n", " ").replace("\r", " ")
    if type(value) in (int, float, bool) or value is None:
        return value
    return None


def _new_stats():
    return dict(count=0, sampled_count=0, total_ns=0, max_ns=0, peak=0,
                max_start_ns=0, max_end_ns=0, max_monotonic=0.0, max_monotonic_source=None,
                max_request=None, max_generation=None, max_active_request=None, slow_count=0,
                last_slow_start_ns=0, last_slow_end_ns=0,
                suppressed=0, in_progress_count=0)


class RecordingDiagnostics:
    def __init__(self, logger, *, categories, request=None, generation=None,
                 timing_logger=None, perf_ns=time.perf_counter_ns,
                 monotonic=time.monotonic, category_limit=32):
        self.logger = logger
        self._timing_logger = timing_logger
        self._perf_ns = perf_ns
        self._monotonic = monotonic
        self.request = _primitive(request)
        self.generation = _primitive(generation)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._stats = {}
        for stage in categories:
            if type(stage) is str and len(stage) <= 48:
                self._stats.setdefault(stage, _new_stats())
            if len(self._stats) == min(64, max(1, category_limit)):
                break
        self._active = {}
        self._lanes = []
        self._lane_counts = dict.fromkeys(self._stats, 0)
        self._last_report = {}
        self._recent = deque(maxlen=32)
        self._recent_truncated = 0
        self._token = self._sequence = 0
        self._dropped = self._errors = self._invalid_fields = 0
        self._last_error = None
        self._sample_skipped = 0
        # Written only by the caller-owned sampler, independently of producer
        # bookkeeping. Snapshot reads these approximate diagnostic counters;
        # reporting failures must never send the sampler back to a producer lock.
        self._sample_errors = self._sample_dropped = 0
        self._sample_last_error = None
        self._closed = False
        self._stop = threading.Event()
        self.sampler_thread = None

    def new_lane(self, *, request=None, generation=None):
        """Register one single-producer slot, with a fixed process-local cap."""
        with self._lock:
            if self._closed or len(self._lanes) >= 16:
                self._dropped += 1
                return None
            lane = _FastLane(self, self.request if request is None else _primitive(request),
                             self.generation if generation is None else _primitive(generation))
            self._lanes.append(lane)
            return lane

    def _close_lane(self, lane):
        with self._lock:
            if lane.closed:
                return
            lane.closed = True
            lane.active = None
            for stage, count in lane.counts.items():
                self._lane_counts[stage] += count
            self._lanes.remove(lane)

    def begin(self, stage, *, request=None, generation=None, active_request=None):
        """Track the current thread's nested phase; 0 means untracked/dropped."""
        started = self._perf_ns()
        monotonic = self._monotonic()
        thread = threading.get_ident()
        with self._lock:
            if self._closed or stage not in self._stats:
                self._dropped += 1
                return 0
            stack = self._active.get(thread)
            if stack is None:
                if len(self._active) >= 16:
                    self._dropped += 1
                    return 0
                stack = self._active[thread] = []
            if len(stack) >= 8:
                self._dropped += 1
                return 0
            self._token += 1
            stack.append(_Phase(self._token, stage, thread,
                                self.request if request is None else _primitive(request),
                                self.generation if generation is None else _primitive(generation),
                                started, monotonic, active_request=_primitive(active_request)))
            return self._token

    def end(self, token, *, emit_slow=False, recent=False, peak=0):
        """Finish on the originating thread. Default is numeric updates only.

        emit_slow=True permits a rate-limited WARNING outside our lock; the
        caller must also have released any business lock before requesting it.
        recent=True retains a task summary in the last-32 ring.
        """
        if not token:
            return 0
        ended = self._perf_ns()
        thread = threading.get_ident()
        with self._lock:
            stack = self._active.get(thread)
            if not stack:
                return 0
            for index in range(len(stack) - 1, -1, -1):
                if stack[index].token == token:
                    phase = stack.pop(index)
                    break
            else:
                return 0
            if not stack:
                del self._active[thread]
            elapsed = max(0, ended - phase.start_ns)
            report = self._update(phase, ended, elapsed, peak, recent, emit_slow)
        if report:
            self._slow_event(phase, ended, elapsed, "completed")
        return elapsed

    def observe(self, stage, elapsed_ns, *, started_ns=None, monotonic=None,
                request=None, generation=None, peak=0, emit_slow=False, recent=False,
                monotonic_source="sampled"):
        """Aggregate an externally measured interval, without an active phase."""
        ended = self._perf_ns() if started_ns is None else started_ns + elapsed_ns
        started = ended - elapsed_ns
        if monotonic is None:
            monotonic = self._monotonic() - elapsed_ns / 1e9
        phase = _Phase(0, stage, threading.get_ident(),
                       self.request if request is None else _primitive(request),
                       self.generation if generation is None else _primitive(generation),
                       started, monotonic, monotonic_source)
        with self._lock:
            if self._closed or stage not in self._stats:
                self._dropped += 1
                return
            report = self._update(phase, ended, max(0, elapsed_ns), peak, recent, emit_slow)
        if report:
            self._slow_event(phase, ended, elapsed_ns, "completed")

    def _update(self, phase, ended, elapsed, peak, recent, emit_slow):
        stats = self._stats[phase.stage]
        stats["count"] += 1
        stats["sampled_count"] += 1
        stats["total_ns"] += elapsed
        if peak > stats["peak"]:
            stats["peak"] = peak
        if elapsed > stats["max_ns"]:
            stats.update(max_ns=elapsed, max_start_ns=phase.start_ns, max_end_ns=ended,
                         max_monotonic=phase.monotonic, max_request=phase.request,
                         max_monotonic_source=phase.monotonic_source,
                         max_generation=phase.generation, max_active_request=phase.active_request)
        if recent:
            if len(self._recent) == 32:
                self._recent_truncated += 1
            self._recent.append(dict(stage=phase.stage, request=phase.request,
                                     generation=phase.generation, thread=phase.thread,
                                     start_ns=phase.start_ns, end_ns=ended,
                                     monotonic=phase.monotonic, active_request=phase.active_request))
        if elapsed < 100_000_000:
            return False
        stats["slow_count"] += 1
        stats["last_slow_start_ns"] = phase.start_ns
        stats["last_slow_end_ns"] = ended
        if emit_slow:
            return self._reserve_report(phase.stage, ended)
        stats["suppressed"] += 1
        return False

    def _reserve_report(self, stage, now):
        last = self._last_report.get(stage)
        if last is not None and now - last < 5_000_000_000:
            self._stats[stage]["suppressed"] += 1
            return False
        self._last_report[stage] = now
        return True

    def _slow_event(self, phase, now, elapsed, status, *, sequence=None):
        return self._emit(phase.stage, logging.WARNING, phase.request, phase.generation,
                          phase.thread, now, self._monotonic(),
                          dict(status=status, start_ns=phase.start_ns,
                               start_monotonic=phase.monotonic, elapsed_ns=elapsed,
                               monotonic_source=phase.monotonic_source,
                               active_request=phase.active_request),
                          sequence=sequence)

    def milestone(self, stage, *, request=None, generation=None, **fields):
        """Submit a low-frequency INFO boundary and request a non-wait flush."""
        return self._emit(stage, logging.INFO,
                          self.request if request is None else _primitive(request),
                          self.generation if generation is None else _primitive(generation),
                          threading.get_ident(), self._perf_ns(), self._monotonic(), fields)

    def _emit(self, stage, level, request, generation, thread, now, monotonic, fields,
              summary=None, sequence=None, max_chars=8192):
        sampled = sequence is not None
        if not self._enabled(level, sampled=sampled):
            return False
        clean = {}
        invalid = 0
        for index, (key, value) in enumerate(fields.items()):
            if index >= 24:
                invalid += len(fields) - index
                break
            if type(key) is not str or type(value) not in (str, int, float, bool, type(None)):
                invalid += 1
                continue
            clean[key[:48]] = _primitive(value)
        if sampled:
            # Sampling has already reserved its sequence during the initial
            # nonblocking scan. Its fixed internal fields need no producer
            # bookkeeping, and delivery must never reacquire this lock.
            if self._closed:
                self._sample_dropped += 1
                return False
        else:
            with self._lock:
                self._invalid_fields += invalid
                if self._closed:
                    self._dropped += 1
                    return False
                self._sequence += 1
                sequence = self._sequence
        stage = _primitive(stage)
        message = (f"Recording diagnostic stage={stage} request={request if request is not None else 'unknown'}"
                   f" pid={self._pid} thread={thread} monotonic={monotonic}"
                   f" perf_ns={now} seq={sequence}")
        if generation is not None:
            message += f" generation={generation}"
        message += " details=" + json.dumps(clean, separators=(",", ":"), ensure_ascii=False)
        if summary is not None:
            message += " summary=" + self._summary_json(summary, max_chars - len(message) - 9)
        try:
            # Optional logger/filter/handler and forwarder implementations are
            # external boundaries and may raise arbitrary errors. Drop this
            # diagnostic only; bounded type evidence avoids recursive logging
            # and even an exception with a broken __str__ is safe.
            if self._timing_logger is not None:
                accepted = self._timing_logger.log(self.logger, level, message, critical=True)
                if not accepted:
                    if sampled:
                        self._sample_dropped += 1
                    else:
                        with self._lock:
                            self._dropped += 1
                return accepted
            self.logger.log(level, message)
            LogManager.request_flush()
            return True
        except Exception as error:
            self._record_error(error, sampled=sampled)
            return False

    def _record_error(self, error, *, sampled):
        if sampled:
            self._sample_errors += 1
            self._sample_last_error = type(error).__name__[:128]
        else:
            with self._lock:
                self._errors += 1
                self._last_error = type(error).__name__[:128]

    def _enabled(self, level, *, sampled=False):
        try:
            return self.logger.isEnabledFor(level)
        except Exception as error:
            # The optional logger's level hook can be overridden externally.
            self._record_error(error, sampled=sampled)
            return False

    @staticmethod
    def _summary_json(state, limit):
        # Bound the structure before encoding: never cut a JSON value in half.
        # Active phases and recent tasks have reserved space so cumulative
        # statistics cannot displace the evidence needed to diagnose a timeout.
        summary = {key: value for key, value in state.items()
                   if key not in ("active", "recent", "categories")}
        categories = {stage: {key: value for key, value in stats.items()
                              if value not in (0, None)}
                      for stage, stats in state["categories"].items()
                      if stats["count"] or stats["in_progress_count"] or stats["suppressed"]}
        encode = lambda item: json.dumps(item, separators=(",", ":"), ensure_ascii=False)
        summary.update(active=state["active"], recent=state["recent"], categories=categories,
                       omitted_active=0, omitted_recent=0, omitted_categories=0)
        complete = encode(summary)
        if len(complete) <= limit:
            return complete
        # Only oversized summaries need per-item allocation. Normal summaries
        # serialize once and retain all evidence, including at the exact limit.
        summary.update(active=[], recent=[], categories={},
                       omitted_active=len(state["active"]), omitted_recent=len(state["recent"]),
                       omitted_categories=len(categories))
        remaining = max(0, limit - len(encode(summary)) - 32)
        for section, items in (("active", state["active"]),
                               ("recent", list(reversed(state["recent"]))),
                               ("categories", list(categories.items()))):
            # Each item is encoded only once for budgeting. The final payload
            # is serialized once more, so this work stays linear and low-frequency.
            budget = remaining if section == "categories" else remaining // 2
            used = 0
            for item in items:
                size = len(encode(dict([item]) if section == "categories" else item)) + 1
                if used + size > budget:
                    continue
                if section == "categories":
                    summary[section][item[0]] = item[1]
                else:
                    summary[section].append(item)
                summary["omitted_" + section] -= 1
                used += size
            remaining -= used
        summary["recent"].reverse()
        return encode(summary)

    def sample_once(self):
        """Best-effort independent scan; never wait for diagnostic/business locks."""
        now = self._perf_ns()
        if not self._lock.acquire(blocking=False):
            # Approximate contention counter; no second lock/wait on the sampler.
            self._sample_skipped += 1
            return 0
        reports = []
        try:
            if self._closed:
                return 0
            phases = [stack[-1] for stack in self._active.values()]
            phases.extend(phase for lane in self._lanes
                          if (phase := lane.phase_snapshot()) is not None)
            for phase in phases:
                elapsed = max(0, now - phase.start_ns)
                if elapsed >= 100_000_000 and self._reserve_report(phase.stage, now):
                    self._stats[phase.stage]["in_progress_count"] += 1
                    self._sequence += 1
                    reports.append((phase, elapsed, self._sequence))
        finally:
            self._lock.release()
        for phase, elapsed, sequence in reports:
            self._slow_event(phase, now, elapsed, "in_progress", sequence=sequence)
        return len(reports)

    def start_sampler(self):
        """Start at most one sampler per helper; child owner calls this once."""
        with self._lock:
            if self._closed:
                return False
            if self.sampler_thread is not None:
                return True
            self.sampler_thread = threading.Thread(
                target=self._sample_loop, name="recording-diagnostic-sampler", daemon=True)
            try:
                self.sampler_thread.start()
            except Exception as error:
                # Thread.start is an external resource boundary. Keep the same
                # thread object even if a custom starter launched then raised.
                self._errors += 1
                self._last_error = type(error).__name__[:128]
                self._closed = True
                self._stop.set()
                return False
            return True

    def _sample_loop(self):
        while not self._stop.wait(0.1):
            self.sample_once()

    def snapshot(self):
        """Bounded copies, with cumulative intervals and explicitly recent tasks."""
        with self._lock:
            state = dict(categories={stage: dict(stats) for stage, stats in self._stats.items()},
                         recent=[dict(item) for item in self._recent],
                         recent_truncated=self._recent_truncated,
                         active=[dict(stage=phase.stage, request=phase.request,
                                      generation=phase.generation, thread=phase.thread,
                                      start_ns=phase.start_ns, monotonic=phase.monotonic,
                                      active_request=phase.active_request)
                                 for stack in self._active.values() for phase in stack],
                         dropped=self._dropped + self._sample_dropped,
                         errors=self._errors + self._sample_errors,
                         last_error=self._last_error or self._sample_last_error,
                         sample_errors=self._sample_errors, sample_dropped=self._sample_dropped,
                         sample_last_error=self._sample_last_error,
                         invalid_fields=self._invalid_fields, sample_skipped=self._sample_skipped)
            for stage, stats in state["categories"].items():
                observed = self._lane_counts[stage] + sum(
                    lane.counts[stage] for lane in self._lanes)
                if observed:
                    stats["observed_count"] = observed
            for lane in self._lanes:
                phase = lane.phase_snapshot()
                if phase is not None:
                    state["active"].append(dict(
                        stage=phase.stage, request=phase.request, generation=phase.generation,
                        thread=phase.thread, start_ns=phase.start_ns, monotonic=phase.monotonic,
                        monotonic_source=phase.monotonic_source))
        if self._timing_logger is not None:
            state["delivery"] = dict(dropped=self._timing_logger.dropped,
                                     errors=self._timing_logger.errors)
        return state

    def summary(self, stage, *, request=None, generation=None, compact=False,
                max_chars=8192, **fields):
        if not self._enabled(logging.INFO):
            return False
        state = self.snapshot()
        if compact:
            state["recent"] = []
            state["categories"] = {
                stage: {key: value for key, value in stats.items()
                        if key in ("count", "in_progress_count", "suppressed", "max_ns",
                                   "max_start_ns", "max_end_ns", "max_request", "max_active_request")}
                for stage, stats in state["categories"].items()}
        delivery = LogManager.get_async_stats()
        state["log_delivery"] = dict(scope="process_cumulative", **{
            key: delivery[key] for key in (
                "accepted", "written", "pending", "dropped_full", "dropped_closed",
                "write_errors", "snapshot_errors", "consumer_alive")})
        return self._emit(stage, logging.INFO,
                          self.request if request is None else _primitive(request),
                          self.generation if generation is None else _primitive(generation),
                          threading.get_ident(), self._perf_ns(), self._monotonic(), fields,
                          summary=state, max_chars=min(32768, max(2048, max_chars)))

    def close(self, timeout=0):
        """Stop sampling, release phase references, optionally join for <= timeout."""
        with self._lock:
            self._closed = True
            self._active.clear()
            for lane in self._lanes:
                lane.active = None
                lane.closed = True
            thread = self.sampler_thread
        self._stop.set()
        if (timeout > 0 and thread is not None and thread.ident is not None
                and thread is not threading.current_thread()):
            thread.join(timeout=timeout)
