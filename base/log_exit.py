"""One bounded logging grace period before project-controlled process death.

Each spawn owns three single-writer, write-once shared-memory mailboxes. Parent
writes request, child writes started/result; the final ready byte commits a
bounded JSON payload. No pipes, semaphore locks, receivers, or feeder threads
can survive a dead peer or block a supervisor. Partial/uncommitted payloads are
ignored. Endpoints belong to exactly one child and must never be reused.
"""
from dataclasses import dataclass
import json
import math
import os
import threading
import time
import uuid

from base.log_manager import LogManager
from consts.running_consts import LOG_SHUTDOWN_TIMEOUT


_MAILBOX_SIZE = 4096
_POLL_INTERVAL = 0.005
_DIAGNOSTIC_BYTES = 1024
_STAT_KEYS = ("accepted", "written", "pending", "write_errors", "snapshot_errors",
              "dropped_full", "dropped_closed", "consumer_alive", "last_error")


def _budget(timeout):
    return min(LOG_SHUTDOWN_TIMEOUT, max(0.0, float(timeout)))


def _write(mailbox, token, **data):
    payload = json.dumps(dict(token=token, **data), ensure_ascii=True).encode("ascii")
    if len(payload) >= len(mailbox) - 1:
        raise ValueError("log-drain status exceeds fixed mailbox")
    mailbox[1:1 + len(payload)] = payload
    mailbox[0] = 1


def _read(mailbox, token):
    if mailbox[0] != 1:
        return None
    payload = bytes(mailbox[1:]).split(b"\0", 1)[0]
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    return data if isinstance(data, dict) and data.get("token") == token else None


def _deadline(mailbox, token):
    data = _read(mailbox, token)
    value = None if data is None else data.get("deadline")
    # Writers publish monotonic floats. Reject giant integers without converting
    # them (float conversion itself could overflow on a corrupt peer payload).
    return value if type(value) is float and math.isfinite(value) else None


def _result(data):
    if data is None or not isinstance(data.get("status"), str) or data["status"] not in {
            "drained", "drained-with-errors", "timeout", "no-runtime"}:
        return None
    status, stats = data["status"], data.get("stats")
    if stats is not None:
        if not isinstance(stats, dict) or set(stats) != set(_STAT_KEYS):
            return None
        if any(type(stats[key]) is not int or stats[key] < 0 for key in _STAT_KEYS[:-2]):
            return None
        if type(stats["consumer_alive"]) is not bool:
            return None
        if stats["last_error"] is not None and not isinstance(stats["last_error"], str):
            return None
        if stats["accepted"] != stats["written"] + stats["write_errors"] + stats["pending"]:
            return None
    if status.startswith("drained"):
        if stats is None or stats["consumer_alive"] or stats["pending"]:
            return None
        errors = bool(stats["write_errors"] or stats["snapshot_errors"] or stats["last_error"])
        if errors != (status == "drained-with-errors"):
            return None
    detail = data.get("detail")
    if detail is not None and not isinstance(detail, str):
        return None
    return DrainResult(status, stats, detail)


@dataclass(frozen=True)
class DrainResult:
    status: str
    stats: dict | None = None
    detail: str | None = None


def _bounded_diagnostic(value):
    """Bound encoded bytes, including JSON surrogate pairs for astral text."""
    if value is None:
        return None
    low, high = 0, min(512, len(value))
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(value[:middle], ensure_ascii=True)) <= _DIAGNOSTIC_BYTES:
            low = middle
        else:
            high = middle - 1
    return value[:low]


@dataclass(frozen=True)
class _Endpoint:
    token: str
    request: object
    started: object
    result: object


class ProcessLogDrain:
    """Parent-owned, one-shot state; begin/poll never wait or inspect a process.

    Supervisor serializes calls and supplies ``already_dead`` only after its
    existing death confirmation. close() releases parent references after death
    or failed startup; a delayed child retains its own endpoint. No OS handles
    need a blocking handshake to close. wait() is only for background owners
    and can observe current liveness through their optional is_alive callback.
    """

    def __init__(self, endpoint, clock=time.monotonic):
        self.child_endpoint = endpoint
        self._deadline = None
        self._result = None
        self.reason = None
        self._clock = clock

    @classmethod
    def create(cls, context, *, clock=time.monotonic):
        # Injected clocks are for deterministic supervisor tests; a real child
        # requires the system monotonic epoch shared by all spawned processes.
        return cls(_Endpoint(uuid.uuid4().hex, *(context.RawArray("B", _MAILBOX_SIZE)
                                                for _ in range(3))), clock)

    def begin(self, reason, timeout=LOG_SHUTDOWN_TIMEOUT):
        if self.reason is None:
            self.reason = str(reason)[:256]
        if self._deadline is None and self._result is None:
            self._deadline = self._clock() + _budget(timeout)
            endpoint = self.child_endpoint
            if endpoint is not None:
                started = _deadline(endpoint.started, endpoint.token)
                if started is not None:
                    self._deadline = min(self._deadline, started)
                _write(endpoint.request, endpoint.token, deadline=self._deadline)

    def poll(self, *, already_dead=False):
        if self._result is not None:
            return self._result
        endpoint = self.child_endpoint
        if endpoint is not None:
            result = _result(_read(endpoint.result, endpoint.token))
            if result is not None:
                self._result = result
                return self._result
            started = _deadline(endpoint.started, endpoint.token)
            if started is not None:
                self._deadline = started if self._deadline is None else min(self._deadline, started)
        if already_dead:
            self._result = DrainResult(
                "already-dead", detail=("log drain unconfirmed; child exited; pending unknown"
                                        if self._deadline is not None else None))
        elif self._deadline is not None and self._clock() >= self._deadline:
            self._result = DrainResult("timeout", detail="log drain unconfirmed; pending unknown")
        return self._result

    def wait(self, *, already_dead=False, is_alive=None):
        if self._deadline is None and self._result is None and not already_dead:
            raise RuntimeError("begin log drain before waiting")
        while True:
            result = self.poll(already_dead=already_dead or (is_alive is not None and not is_alive()))
            if result is not None:
                return result
            time.sleep(min(_POLL_INTERVAL, max(0, self._deadline - self._clock())))

    def close(self):
        self.child_endpoint = None


class _ExitCycle:
    """Child process ownership; its short local lock never encloses I/O/waits."""

    def __init__(self, endpoint=None):
        self.endpoint = endpoint
        self.lock = threading.Lock()
        self.deadline = None
        self.result = None
        self.stats = None
        self.worker = None
        self.stop = threading.Event()
        self.detached = False
        self.self_exit_claimed = False

    def claim_for_self_exit(self):
        # Called while selecting the binding, before begin can race teardown.
        # Thread.start and all waits remain outside both ownership locks.
        with self.lock:
            self.self_exit_claimed = True

    def detach_if_idle(self):
        """Serialize unbinding with begin, including a request already read.

        The listener can outlive its bounded join. A committed request retains
        this cycle for that listener or a later self-exit; an idle cycle becomes
        permanently unable to start after its process binding is released.
        """
        with self.lock:
            requested = (None if self.endpoint is None else
                         _deadline(self.endpoint.request, self.endpoint.token))
            if self.deadline is not None or requested is not None or self.self_exit_claimed:
                return False
            self.detached = True
            return True

    def remaining(self):
        deadline = self.deadline
        if self.endpoint is not None:
            requested = _deadline(self.endpoint.request, self.endpoint.token)
            if requested is not None:
                deadline = min(deadline, requested)
        return max(0, deadline - time.monotonic())

    def begin(self, timeout):
        with self.lock:
            if self.deadline is not None or self.detached:
                return
            self.deadline = time.monotonic() + _budget(timeout)
            if self.endpoint is not None:
                requested = _deadline(self.endpoint.request, self.endpoint.token)
                if requested is not None:
                    self.deadline = min(self.deadline, requested)
                _write(self.endpoint.started, self.endpoint.token, deadline=self.deadline)
            self.worker = threading.Thread(target=self._drain, name="project-log-drain", daemon=True)
        try:
            self.worker.start()
        except (RuntimeError, OSError) as error:
            # Thread.start is an external runtime boundary. Preserve forced exit
            # even when no thread can be allocated; do not touch the sink here.
            self.finish(DrainResult("timeout", detail=f"drain thread start failed: {error}"[:512]))

    def finish(self, result):
        stats = result.stats
        if stats is not None:
            stats = dict(stats, last_error=_bounded_diagnostic(stats["last_error"]))
        result = DrainResult(result.status, stats, _bounded_diagnostic(result.detail))
        with self.lock:
            if self.result is None:
                if self.endpoint is not None:
                    _write(self.endpoint.result, self.endpoint.token, status=result.status,
                           stats=result.stats, detail=result.detail)
                # A concurrent self-exit may act as soon as result is visible.
                # Publish the parent's diagnostic before permitting that exit.
                self.result = result
        return self.result

    def _drain(self):
        runtime = LogManager.seal_for_forced_exit()
        if runtime is None:
            self.finish(DrainResult("no-runtime"))
            return
        # Capture the existing runtime only. No set_log_handler/reacquisition.
        self.stats = runtime.stats()
        stopped = runtime.shutdown(self.remaining())
        self.stats = runtime.stats()
        status = "timeout"
        if stopped and not self.stats["consumer_alive"] and self.stats["pending"] == 0:
            status = "drained-with-errors" if (self.stats["write_errors"]
                                               or self.stats["snapshot_errors"]
                                               or self.stats["last_error"]) else "drained"
        self.finish(DrainResult(status, self.stats))

    def poll(self):
        if self.result is not None:
            return self.result
        if self.deadline is not None and self.remaining() <= 0:
            return self.finish(DrainResult("timeout", self.stats,
                                           "logging deadline expired; stats are last available snapshot"))
        return None

    def listen(self):
        while not self.stop.is_set():
            if _deadline(self.endpoint.request, self.endpoint.token) is not None:
                self.begin(LOG_SHUTDOWN_TIMEOUT)
            if self.poll() is not None:
                return
            self.stop.wait(_POLL_INTERVAL)


class _ProcessBinding:
    # Class ownership is process-scoped; import allocates no runtime or thread.
    cycle = None
    lock = threading.Lock()


def run_with_log_drain(target, args, endpoint, kwargs=None):
    """Spawn target wrapper; binds before business code and unbinds on return."""
    cycle = _ExitCycle(endpoint)
    with _ProcessBinding.lock:
        if _ProcessBinding.cycle is not None:
            raise RuntimeError("process already bound to a log drain cycle")
        _ProcessBinding.cycle = cycle
    listener = threading.Thread(target=cycle.listen, name="project-log-control", daemon=True)
    try:
        listener.start()
        return target(*args, **(kwargs or {}))
    finally:
        cycle.stop.set()
        if listener.ident is not None:
            listener.join(_POLL_INTERVAL * 2)
        with _ProcessBinding.lock:
            # The timed join does not prove the listener has stopped. Serialize
            # detachment with begin so it cannot start an abandoned cycle, and
            # retain committed requests even before the listener adopts them.
            if cycle.detach_if_idle():
                _ProcessBinding.cycle = None


def exit_with_log_drain(code, timeout=LOG_SHUTDOWN_TIMEOUT):
    """Retain the original hard exit, sharing any earlier parent/self deadline."""
    with _ProcessBinding.lock:
        if _ProcessBinding.cycle is None:
            _ProcessBinding.cycle = _ExitCycle()
        cycle = _ProcessBinding.cycle
        cycle.claim_for_self_exit()
    cycle.begin(timeout)
    while cycle.poll() is None:
        time.sleep(min(_POLL_INTERVAL, cycle.remaining()))
    os._exit(code)
