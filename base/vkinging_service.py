"""One instance, one daemon SDK owner; callers never wait for native calls.

Capture handlers have signature ``handler(sdk, request, control)``. They own
task/file cleanup, NOT SDK construction/close, and return CaptureResult.
``resources_released=True`` proves their tasks/files are closed. On exceptions,
``control.confirm_released()`` supplies the same proof; absence is uncertainty.

Optional ``finalize(control) -> CaptureResult`` runs after SDK close (metadata,
file validation). Optional ``commit(control) -> JSON-like value`` publishes the
file after an atomic validity/commit claim, outside the service lock. Cancel
returns False after that claim; close still suppresses all business results.
Commit must not call the SDK. Finalize/commit start with fresh
cleanup uncertainty: exceptional exits need explicit confirm_released(). A
successful callback must leave no handles open. Only JSON-like descriptors,
never audio buffers or callbacks, are published to consumers.
"""
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable

from base.vkinging_discovery import discover_vkinging_devices
from base.vkinging_input import VkingingCaptureRequest
from base.vkinging_sdk import VkDaqClient


def _exception_text(error):
    """Exception formatting is external code; never retry its str/repr hooks."""
    try:
        return str(error)
    except Exception:
        return f"{type(error).__name__}: exception text unavailable"


def _freeze_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("result descriptor keys must be strings")
        return MappingProxyType({key: _freeze_value(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(child) for child in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("result must be a JSON-like descriptor, not an audio buffer or handle")


@dataclass(frozen=True, slots=True)
class CaptureResult:
    value: object = None
    resources_released: bool = False
    cleanup_error: str | None = None
    finalize: Callable | None = None
    commit: Callable | None = None

    def __post_init__(self):
        object.__setattr__(self, "value", _freeze_value(self.value))


@dataclass(frozen=True, slots=True)
class VkingingEvent:
    request_id: str
    kind: str
    state: str
    phase: str
    raw_frames: int = 0
    value: object = None
    reason: str | None = None
    cleanup_error: str | None = None
    resources_released: bool = False
    diagnostic_error: str | None = None


@dataclass(slots=True)
class _Operation:
    request_id: str
    kind: str
    request: VkingingCaptureRequest | None
    phase: str
    phase_started: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    valid: bool = True
    terminal_sent: bool = False
    commit_claimed: bool = False
    raw_frames: int = 0
    capture_started: float | None = None
    last_progress: float | None = None
    release_started: float | None = None
    resources_released: bool = False
    cleanup_error: str | None = None


class CaptureControl:
    """Worker-only phase/progress/release reporting; cancelled is thread safe.

    Phases are starting, capturing, stopping, releasing, finalizing, committing.
    report_progress receives cumulative RAW frames, including discarded frames.
    Call confirm_released only after all owned task/file handles are closed.
    No SDK operation is ever performed by this control object.
    """

    def __init__(self, service, operation):
        self._service, self._operation = service, operation

    @property
    def cancelled(self):
        return self._operation.cancel_event.is_set()

    def _assert_owner(self):
        if threading.current_thread() is not self._service._worker:
            raise RuntimeError("capture control reports belong to the SDK owner")

    def set_phase(self, phase):
        self._assert_owner()
        if phase not in ("starting", "capturing", "stopping", "releasing", "finalizing", "committing"):
            raise ValueError("unknown capture phase")
        with self._service._condition:
            self._service._set_phase(self._operation, phase)

    def report_progress(self, raw_frames):
        self._assert_owner()
        op = self._operation
        with self._service._condition:
            if type(raw_frames) is not int or raw_frames < op.raw_frames:
                raise ValueError("raw_frames must be a monotonic nonnegative integer")
            if raw_frames > op.raw_frames:
                op.raw_frames = raw_frames
                op.last_progress = self._service._clock()
                self._service._event(op, "progress")

    def confirm_released(self):
        self._assert_owner()
        with self._service._condition:
            self._operation.resources_released = True

    def report_cleanup_failure(self, reason):
        """Record a failed release without replacing a primary exception."""
        self._assert_owner()
        if not isinstance(reason, str) or not reason:
            raise ValueError("cleanup failure requires a diagnostic string")
        self._service._record_cleanup_failure(self._operation, reason)


class VkingingService:
    """Single-slot work admission with bounded, lossless terminal storage.

    submit_discovery/start_capture/cancel return acceptance booleans. Request
    IDs must be unique (the caller owns ID generation). poll_events drains an
    ordered list: up to terminal_capacity terminal events, one reliable release
    acknowledgement, and one coalesced state/progress event. A pending release
    acknowledgement blocks admission until polled. Each operation reserves a terminal
    slot; callers must drain events before more work can be admitted when full.

    tick(now) uses the injected monotonic clock's time domain. close() returns
    immediately; tick drives the three-second close deadline. A stuck native
    call remains on the original daemon owner, with no rescue thread or kill.
    """

    def __init__(self, *, sdk_factory=VkDaqClient, capture_handler=None,
                 clock=time.monotonic, terminal_capacity=16, logger=None):
        if type(terminal_capacity) is not int or terminal_capacity < 1:
            raise ValueError("terminal_capacity must be a positive integer")
        self._sdk_factory = sdk_factory
        self._capture_handler = capture_handler
        self._clock = clock
        self._capacity = terminal_capacity
        self._condition = threading.Condition(threading.RLock())
        self._terminals = deque()
        self._release_ack = None
        self._transient = None
        self._sequence = 0
        self._active = None
        self._pending = None
        self._closing_at = None
        self._state = "idle"
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._worker = threading.Thread(target=self._work, name="vkinging-sdk-owner", daemon=True)
        self._worker.start()

    @property
    def state(self):
        with self._condition:
            return self._state

    def submit_discovery(self, request_id):
        return self._submit(request_id, "discovery", None)

    def start_capture(self, request):
        if not isinstance(request, VkingingCaptureRequest):
            raise TypeError("capture requires a frozen VkingingCaptureRequest")
        return self._submit(request.request_id, "capture", request)

    def _submit(self, request_id, kind, request):
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be nonempty text")
        with self._condition:
            if (self._closing_at is not None or self._state != "idle"
                    or self._active is not None or self._release_ack is not None
                    or len(self._terminals) >= self._capacity):
                return False
            phase = "discovering" if kind == "discovery" else "starting"
            op = _Operation(request_id, kind, request, phase, self._clock(),
                            resources_released=kind == "discovery")
            self._active = self._pending = op
            self._state = phase
            self._event(op, "state")
            self._condition.notify_all()
            return True

    def cancel(self, request_id):
        with self._condition:
            op = self._active
            if (op is None or op.request_id != request_id or not op.valid
                    or op.commit_claimed or self._closing_at is not None):
                return False
            self._revoke(op, "cancelled", "cancelled", unavailable=False)
            return True

    def poll_events(self):
        with self._condition:
            queued = list(self._terminals)
            if self._release_ack is not None:
                queued.append(self._release_ack)
            if self._transient is not None:
                queued.append(self._transient)
            self._terminals.clear()
            self._release_ack = None
            self._transient = None
            return [event for _, event in sorted(queued, key=lambda item: item[0])]

    def tick(self, now):
        with self._condition:
            if self._closing_at is not None:
                if self._active is None or now - self._closing_at >= 3:
                    self._state = "closed"
                    self._condition.notify_all()
                return
            op = self._active
            if op is None or op.commit_claimed:
                return
            reason = None
            if op.release_started is not None and now - op.release_started >= 3:
                reason = "release_timeout"
            elif op.valid and op.phase == "discovering" and now - op.phase_started >= 10:
                reason = "discovery_timeout"
            elif op.valid and op.phase == "starting" and now - op.phase_started >= 10:
                reason = "start_timeout"
            elif op.valid and op.phase == "capturing":
                budget = (op.request.target_samples + op.request.discard_initial_samples) / op.request.sample_rate + 5
                if now - op.capture_started >= budget:
                    reason = "capture_timeout"
                elif now - op.last_progress >= 3:
                    reason = "read_stall"
            if reason:
                self._revoke(op, "failed", reason, unavailable=True)

    def close(self):
        with self._condition:
            if self._closing_at is not None:
                return
            self._closing_at = self._clock()
            if self._active is not None:
                self._active.valid = False
                self._active.cancel_event.set()
            self._terminals.clear()
            self._release_ack = None
            self._transient = None
            self._state = "closing" if self._active is not None else "closed"
            self._condition.notify_all()

    def _set_phase(self, op, phase):
        now = self._clock()
        if op.phase != phase:
            op.phase, op.phase_started = phase, now
        if phase == "capturing" and op.capture_started is None:
            op.capture_started = op.last_progress = now
        if phase in ("stopping", "releasing") and op.release_started is None:
            op.release_started = now
        # Finalization is not a native release or sampling wait.
        if op.valid and phase in ("finalizing", "committing"):
            op.release_started = None
        if op.valid and self._closing_at is None:
            self._state = phase
            self._event(op, "state")

    def _event(self, op, kind, *, value=None, reason=None, cleanup_error=None,
               resources_released=False, diagnostic_error=None):
        if self._closing_at is not None:
            return
        terminal = kind in ("succeeded", "failed", "cancelled")
        if terminal and op.terminal_sent:
            return
        event = VkingingEvent(op.request_id, kind, self._state, op.phase, op.raw_frames,
                              value, reason, cleanup_error, resources_released, diagnostic_error)
        self._sequence += 1
        item = (self._sequence, event)
        if terminal:
            op.terminal_sent = True
            self._terminals.append(item)
        elif kind == "released":
            self._release_ack = item
        else:
            self._transient = item
        self._condition.notify_all()

    def _revoke(self, op, kind, reason, *, unavailable):
        op.valid = False
        op.cancel_event.set()
        if op.release_started is None:
            op.release_started = self._clock()
        self._state = "unavailable" if unavailable else "stopping"
        self._event(op, kind, reason=reason)
        self._condition.notify_all()

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closing_at is not None)
                if self._pending is None:
                    return
                op, self._pending = self._pending, None
            self._execute(op)

    def _accept_result(self, op, result):
        if not isinstance(result, CaptureResult):
            raise TypeError("capture/finalize handler must return CaptureResult")
        with self._condition:
            op.resources_released = result.resources_released
            if result.cleanup_error:
                self._record_cleanup_failure(op, result.cleanup_error)
        return result

    def _record_cleanup_failure(self, op, reason):
        with self._condition:
            op.resources_released = False
            op.cleanup_error = f"{op.cleanup_error}; {reason}" if op.cleanup_error else reason

    def _execute(self, op):
        sdk, result, value, error = None, None, None, None
        error_text, diagnostic_error = None, None
        sdk_closed, close_attempted, body_error = True, False, None
        body_phase = op.phase
        error_phase, exception_info = op.phase, None
        control = CaptureControl(self, op)
        try:
            # This is the one external SDK/handler/I/O exception boundary. The
            # finally closes the SDK on this owner even when a handler raises;
            # Python exception chaining preserves a simultaneous close failure.
            try:
                if control.cancelled:
                    op.resources_released = True  # No native/file work began.
                else:
                    sdk = self._sdk_factory()
                    sdk_closed = False
                    if op.kind == "discovery":
                        value = discover_vkinging_devices(sdk)
                    else:
                        if self._capture_handler is None:
                            # Load production capture only on the SDK owner;
                            # imports/construction never touch native hardware.
                            if op.request.purpose == "calibration" and op.request.calibration_target_spl is not None:
                                from base.vkinging_calibration import capture_calibration
                                captured = capture_calibration(sdk, op.request, control, clock=self._clock)
                            else:
                                from base.vkinging_capture import capture_vkinging
                                captured = capture_vkinging(sdk, op.request, control, clock=self._clock)
                        else:
                            captured = self._capture_handler(sdk, op.request, control)
                        result = self._accept_result(op, captured)
                        value = result.value
            finally:
                body_error = sys.exception()
                body_phase = op.phase
                if sdk is not None:
                    with self._condition:
                        self._set_phase(op, "releasing")
                    close_attempted = True
                    sdk.close()
                    sdk_closed = True
            if result is not None and op.resources_released and not op.cleanup_error:
                if result.finalize is not None:
                    with self._condition:
                        self._set_phase(op, "finalizing")
                        op.resources_released = False
                    result = self._accept_result(op, result.finalize(control))
                    value = result.value
                    if result.finalize is not None:
                        raise ValueError("finalize must not return another finalizer")
                if result.commit is not None and op.resources_released and not op.cleanup_error:
                    # Claim atomically against cancel/tick/close after release
                    # proof. External file I/O must never hold the caller lock.
                    commit = None
                    with self._condition:
                        if op.valid and self._closing_at is None:
                            self._set_phase(op, "committing")
                            op.commit_claimed = True
                            op.resources_released = False
                            commit = result.commit
                    if commit is not None:
                        value = _freeze_value(commit(control))
                        with self._condition:
                            op.resources_released = True
        except Exception as exc:
            # An SDK close error may mask a primary exception from the body.
            error = body_error if close_attempted and not sdk_closed and body_error is not None else exc
            error_text = _exception_text(error)
            error_phase = body_phase if body_error is not None else op.phase
            exception_info = sys.exc_info()
            if close_attempted and not sdk_closed:
                self._record_cleanup_failure(op, error_text if error is exc else _exception_text(exc))
        with self._condition:
            released = sdk_closed and op.resources_released and not op.cleanup_error
            if not released and not op.cleanup_error:
                op.cleanup_error = "resource release was not confirmed"
            device = op.request.device if op.request is not None else {}
            diagnostic = (
                op.request_id, error_phase, op.phase, device.get("hardware_id"),
                op.request.sample_rate if op.request else None,
                op.request.range_index if op.request else None,
                op.request.channels if op.request else None, op.raw_frames,
                error_text, op.cleanup_error, op.resources_released, sdk_closed)
        if error is not None or op.cleanup_error:
            try:
                # Handler/formatter hooks and log destinations are an external
                # I/O boundary. Failure must not strand released resources or
                # bypass terminal/ack publication. The event diagnoses a broken
                # log sink independently of actual native/file cleanup status.
                self._logger.error(
                    "vkinging request=%s phase=%s cleanup_phase=%s device=%s rate=%s range=%s "
                    "channels=%s raw_frames=%s primary=%s cleanup=%s resources_released=%s sdk_closed=%s",
                    *diagnostic, exc_info=exception_info)
            except Exception as exc:
                diagnostic_error = _exception_text(exc)
        with self._condition:
            self._state = ("closed" if self._closing_at is not None else
                           "idle" if released else "unavailable")
            if not op.terminal_sent:
                self._event(op, "succeeded" if op.valid and released and error is None else "failed",
                            value=value if op.valid and released and error is None else None,
                            reason=error_text if error is not None else op.cleanup_error,
                            cleanup_error=op.cleanup_error, resources_released=released,
                            diagnostic_error=diagnostic_error)
            elif released:
                self._event(op, "released", resources_released=True, diagnostic_error=diagnostic_error)
            elif op.cleanup_error:
                self._event(op, "state", reason=op.cleanup_error, cleanup_error=op.cleanup_error,
                            diagnostic_error=diagnostic_error)
            self._active = None
            self._condition.notify_all()
