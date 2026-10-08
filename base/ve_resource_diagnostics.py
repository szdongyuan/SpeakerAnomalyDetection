"""Bounded, instance-owned evidence; never owns SDK or logging resources."""
import json
import logging
import math
import os
import sys
import threading
import time


class VeResourceDiagnostics:
    """Short diagnostic locks are independent of the native resource owner."""

    def __init__(self, logger=None, *, generation=None):
        self.logger = logger
        self._lock = threading.Lock()
        self._context = {"generation": generation}
        self._phases = []
        self._status = None
        self._timed_out = False
        self.delivery_failures = 0
        self.last_delivery_error = None

    def context(self, *, reset=False, **fields):
        with self._lock:
            if reset:
                self._status = None
                self._timed_out = False
            self._context.update(fields)

    def startup_attempt(self, number, task, budget):
        """Reset timeout evidence for a new owner while retaining its budget."""
        self.context(reset=True, attempt=number, task=task, startup_budget={
            name: getattr(budget, name) for name in (
                "started_at", "first_attempt_deadline", "cleanup_deadline", "deadline")})
        self.emit("BEGIN", "startup_attempt")

    def begin(self, phase, **fields):
        snapshot = dict(phase=phase, at=time.perf_counter(), **fields)
        with self._lock:
            self._phases.append(snapshot)
        self.emit("BEGIN", phase, **fields)
        return snapshot

    def end(self, snapshot, *, error=None, **fields):
        with self._lock:
            self._phases = [item for item in self._phases if item is not snapshot]
        if error is not None:
            fields.update(self._error_fields(error))
        self.emit("ERROR" if error is not None else "END", snapshot["phase"],
                  elapsed_ms=(time.perf_counter() - snapshot["at"]) * 1000,
                  **{key: value for key, value in snapshot.items() if key not in ("phase", "at")},
                  **fields)

    def call(self, phase, call):
        snapshot = self.begin(phase)
        try:
            result = call()
        except Exception as exc:
            # The SDK boundary may raise arbitrary native/loader exceptions.
            # Observe and re-raise the same exception; the controller owns recovery.
            self.end(snapshot, error=exc)
            raise
        self.end(snapshot)
        return result

    @staticmethod
    def _error_fields(error):
        return dict(error_type=type(error).__name__,
                    error_code=getattr(error, "code", None),
                    error_detail=getattr(error, "detail", str(error)))

    def observe(self, *, event, phase, at, **fields):
        """Called only by discovery's isolated optional-observer boundary."""
        error = fields.pop("error", None)
        if fields.get("attribute") == "DeviceStatus" and event != "BEGIN":
            status = dict(source="discovery", observed_at=at,
                          device_name=fields.get("device_name"),
                          machine_id=fields.get("machine_id"),
                          value=self._bounded(fields.get("result"), 256))
            if error is not None:
                status.update(self._bounded(self._error_fields(error), 256))
            with self._lock:
                if fields.get("machine_id") == self._context.get("machine_id"):
                    self._status = status
        if event == "BEGIN":
            self.begin(phase, **fields)
        else:
            with self._lock:
                snapshot = self._phases[-1]
            self.end(snapshot, error=error,
                     **({"result": fields["result"]} if "result" in fields else {}))

    def timeout(self, phase, owner, **fields):
        now = time.perf_counter()
        with self._lock:
            self._timed_out = True
            current = dict(self._phases[-1]) if self._phases else None
        if current is not None:
            current["elapsed_ms"] = (now - current["at"]) * 1000
        stack = []
        stack_error = None
        try:
            frame = sys._current_frames().get(owner.ident) if owner is not None else None
            while frame is not None and len(stack) < 12:
                stack.append(dict(file=frame.f_code.co_filename[-256:],
                                  function=frame.f_code.co_name[:128], line=frame.f_lineno))
                frame = frame.f_back
        except Exception as exc:
            # Interpreter frame inspection is optional and can be unavailable.
            # Keep timeout evidence and explain why its stack is absent.
            stack_error = str(exc)[:256]
        finally:
            frame = None  # Never retain frame references or their locals.
        self.emit("TIMEOUT", phase, current_phase=current,
                  owner_thread_id=owner.ident if owner is not None else None,
                  owner_alive=owner.is_alive() if owner is not None else False,
                  owner_stack=stack, stack_error=stack_error, **fields)

    @staticmethod
    def _bounded(value, width, depth=0):
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else str(value)
        if type(value) is str:
            return value[:width]
        if depth >= 4:
            return "<nested>"
        if type(value) is dict:
            return {str(key)[:64]: VeResourceDiagnostics._bounded(item, width, depth + 1)
                    for key, item in list(value.items())[:32]}
        if type(value) in (tuple, list):
            return [VeResourceDiagnostics._bounded(item, width, depth + 1) for item in value[:12]]
        return f"<{type(value).__name__[:64]}>"

    def emit(self, event, phase, **fields):
        if self.logger is None:
            return
        try:
            now = time.perf_counter()
            with self._lock:
                context = dict(self._context)
                status = dict(self._status) if self._status is not None else None
                after_timeout = self._timed_out
            if status is not None:
                status["age_ms"] = max(0, (now - status["observed_at"]) * 1000)
            payload = dict(event=event, phase=phase, **context, pid=os.getpid(),
                           thread_id=threading.get_ident(), at=now,
                           after_timeout=after_timeout, device_status=status)
            payload.update(fields)
            safe = self._bounded(payload, 256)
            safe["truncated"] = safe != payload
            encoded = json.dumps(safe, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            for width in (128, 64, 32, 16):
                if len(encoded) <= 8100:
                    break
                safe = self._bounded(payload, width)
                safe["truncated"] = True
                encoded = json.dumps(safe, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            while len(encoded) > 8100:
                del safe[next(key for key in reversed(safe) if key != "truncated")]
                encoded = json.dumps(safe, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            self.logger.log(logging.WARNING if event in ("ERROR", "TIMEOUT") else logging.INFO,
                            "VE resource diagnostic %s", encoded)
        except Exception as exc:
            # Serialization and borrowed logger delivery are noncritical external
            # boundaries. Retain bounded failure evidence without recursive logging.
            self.delivery_failures += 1
            self.last_delivery_error = type(exc).__name__[:64]
