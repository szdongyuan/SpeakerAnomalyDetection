"""Persistent, instance-owned VE3668N native resource and request adapters."""
from dataclasses import dataclass, field
import ctypes
import math
import queue
import threading
import time
import uuid

from base.recording_process_protocol import VeLifecycleCounts, VeResourceRetryProof
from base.ve_startup_policy import VeStartupBudget
from base.ve3668n_capture import to_physical_slots
from base.ve3668n_capture_timing import VeCaptureDeadline, VeCaptureProgress
from base.ve3668n_discovery import resolve_device
from base.ve3668n_input import ve_acquisition_signature
from base.ve_resource_diagnostics import VeResourceDiagnostics
from base.vkinging_sdk import VkDaqClient, VkDaqError


class VeResourceConfigurationError(ValueError):
    """The requested acquisition cannot reuse the retained native resource."""


@dataclass(frozen=True)
class VeResourceFault:
    """First resource failure with native code preserved when one exists."""

    stage: str
    code: int | None
    detail: str

    def __post_init__(self):
        if type(self.stage) is not str or not self.stage.strip():
            raise ValueError("stage is required")
        if self.code is not None and type(self.code) is not int:
            raise ValueError("code must be an integer or None")
        if type(self.detail) is not str or not self.detail.strip():
            raise ValueError("detail is required")


@dataclass(frozen=True)
class VeResourceReleaseOutcome:
    """Bounded controller cleanup result, independent from worker generation."""

    success: bool
    released_signature: tuple | None
    diagnostics: tuple[str, ...]
    lifecycle_counts: VeLifecycleCounts


@dataclass(frozen=True)
class _Command:
    kind: str
    adapter: object = None


class _StartupRevoked(RuntimeError):
    """Control flow back to owner cleanup; never an SDK failure."""


@dataclass
class _StartupAttempt:
    adapter: object
    number: int
    task: str
    deadline: float
    revoked: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    reason: str | None = None
    committed: bool = False
    cleanup_success: bool = False
    cleanup_errors: tuple = ()
    native_failure: tuple | None = None
    owner: threading.Thread | None = None


class VeRecordingAdapter:
    """One request's synchronized binding to a persistent controller."""

    def __init__(self, controller, request, callback, fail, stop_event, signature,
                 startup_budget=None, on_retry=None):
        self._controller = controller
        self.request = request
        self.callback = callback
        self.fail = fail
        self.stop_event = stop_event
        self.signature = signature
        self.startup_budget = startup_budget
        self.on_retry = on_retry
        self.started = threading.Event()
        self._bound = threading.Event()
        self._detached = threading.Event()
        self._lock = threading.RLock()
        self._start_attempted = False
        self._start_succeeded = False
        self._uncertain = False
        self._startup_release_confirmed = False
        self._failure_reported = False
        self._failure = None
        self._diagnostics = []
        self._deadline = None
        self._pending_started_at = None
        self._progress = VeCaptureProgress(None, 0, None)

    def start(self):
        with self._lock:
            if self._start_attempted:
                return self._start_succeeded
            self._start_attempted = True
        succeeded = self._controller._bind(self)
        with self._lock:
            self._start_succeeded = succeeded
        return succeeded

    def stop(self):
        self.stop_event.set()
        with self._lock:
            if not self._start_attempted:
                self._detached.set()
                return True
        return self._controller._detach(self)

    def close(self):
        return self.stop()

    @property
    def started_at(self):
        return self.progress_snapshot.started_at

    @property
    def handles_released(self):
        startup_released = self._controller._startup_handles_released(self)
        with self._lock:
            # An active owner is an observation, not terminal uncertainty.
            # In particular, polling must never invalidate a pending bind.
            released = not self._uncertain if startup_released is None else startup_released
            return self._detached.is_set() and (self._startup_release_confirmed or released)

    @property
    def completed(self):
        """Signal that this request no longer owns the controller binding."""
        return self._detached

    @property
    def failure_snapshot(self):
        with self._lock:
            return self._failure

    @property
    def progress_snapshot(self):
        with self._lock:
            return self._progress

    @property
    def diagnostics(self):
        with self._lock:
            return tuple(self._diagnostics)

    def _latch_native_start(self, started_at):
        with self._lock:
            if self._pending_started_at is not None or self._deadline is not None:
                raise RuntimeError("VE adapter native start time was already recorded")
            self._pending_started_at = started_at

    def _accept_bind(self, clock, checkpoint=None):
        with self._lock:
            if self._uncertain or self.stop_event.is_set():
                return False
            if checkpoint is not None:
                checkpoint()
            started_at = self._pending_started_at
            self._pending_started_at = None
            if started_at is None:
                started_at = clock()
            self._deadline = VeCaptureDeadline(
                self.request.sample_rate, self.request.target_samples,
                started_at, clock=clock,
            )
            self._progress = self._deadline.snapshot()
            if checkpoint is not None:
                checkpoint()
            self.started.set()
            self._bound.set()
            return True

    def _reject_bind(self, stage, message):
        with self._lock:
            self._pending_started_at = None
        self._record_failure(stage, message)
        self._detached.set()
        self._bound.set()

    def _bind_timeout(self, message):
        with self._lock:
            self._pending_started_at = None
            self._detached.set()
            self._bound.set()
        self._record_failure("bind", message)

    def _detach_timeout(self, message):
        with self._lock:
            self._uncertain = True

    def _confirm_detached(self):
        self._detached.set()

    def _record_failure(self, stage, message, fault=None):
        with self._lock:
            self._pending_started_at = None
            self._diagnostics.append(f"{stage}: {message}")
            first = not self._failure_reported
            self._failure_reported = True
            if first:
                self._failure = (
                    VeResourceFault(stage, None, message)
                    if fault is None else fault
                )
        if first:
            try:
                self.fail(stage, message)
            except Exception as exc:
                # Request failure delivery is an external callback boundary.
                # Its failure remains diagnostic and cannot interrupt cleanup.
                with self._lock:
                    self._diagnostics.append(f"failure_callback: {exc}")

    def _check_deadline(self):
        with self._lock:
            deadline = self._deadline
        deadline.check()

    def _deliver(self, physical, returned, clock):
        with self._lock:
            deadline = self._deadline
            remaining = self.request.target_samples - self._progress.frames
            accepted = min(returned, remaining)
            cumulative = self._progress.frames + accepted
            deadline.observe(cumulative, at=clock())
            self._progress = deadline.snapshot()
        if accepted:
            self.callback(physical[:accepted], accepted, None, None)
        return cumulative == self.request.target_samples


class VeResourceController:
    """Own one reusable VE SDK/task and serialize every native call on its thread."""

    _UNINITIALIZED = "UNINITIALIZED"
    _STARTING = "STARTING"
    _RECORDING = "RECORDING"
    _IDLE = "IDLE"
    _RELEASING = "RELEASING"
    _FAILED = "FAILED"

    def __init__(self, sdk_factory=VkDaqClient, fatal=None, clock=time.monotonic,
                 *, bind_timeout=3.0, detach_timeout=.5, logger=None, generation=None):
        for name, value in (("bind_timeout", bind_timeout),
                            ("detach_timeout", detach_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._sdk_factory = sdk_factory
        self._fatal_callback = fatal if fatal is not None else lambda stage, message: None
        self._clock = clock
        self._bind_timeout = float(bind_timeout)
        self._detach_timeout = float(detach_timeout)
        self._commands = queue.Queue()
        self._command_wakeup = threading.Event()
        self._lock = threading.RLock()
        self._state = self._UNINITIALIZED
        self._signature = None
        self._owner = None
        self._active = None
        self._sdk = None
        self._task = None
        self._fresh = None
        self._task_created = False
        self._diagnostics = []
        self._fatal_reported = False
        self._failure = None
        self._counts = dict(sdk_open=0, task_create=0, task_start=0,
                            task_stop=0, task_clear=0, sdk_close=0)
        self._release_outcome = None
        self._resource_diagnostics = VeResourceDiagnostics(logger, generation=generation)
        self._has_bound_request = False
        self._startup_attempt = None
        self._closing = False
        self._generation = generation if generation is not None else 1

    def stream(self, *, request, callback, fail, stop_event,
               startup_budget=None, on_retry=None):
        return self._adapter(
            request=request, callback=callback, fail=fail,
            stop_event=stop_event, startup_budget=startup_budget, on_retry=on_retry)

    def prewarm(self, *, request, fail, startup_budget=None, on_retry=None):
        """Bind a no-file adapter that validates and discards target frames."""
        return self._adapter(
            request=request,
            callback=lambda _physical, _frames, _first, _second: None,
            fail=fail,
            stop_event=threading.Event(),
            startup_budget=startup_budget, on_retry=on_retry,
        )

    def _adapter(self, *, request, callback, fail, stop_event,
                 startup_budget=None, on_retry=None):
        if startup_budget is not None:
            if not isinstance(startup_budget, VeStartupBudget):
                raise ValueError("startup_budget must be a VeStartupBudget")
            VeStartupBudget.__post_init__(startup_budget)
        if on_retry is not None and not callable(on_retry):
            raise ValueError("on_retry must be callable")
        signature = ve_acquisition_signature(
            request.device, request.channels, request.sample_rate)
        with self._lock:
            self._retire_cancelled_startup_locked()
            if self._state == self._FAILED:
                raise RuntimeError("VE resource controller has permanently failed")
            if self._state == self._RELEASING:
                raise RuntimeError("VE resource controller is releasing")
            if self._signature is not None and self._signature != signature:
                raise VeResourceConfigurationError(
                    f"retained VE signature {self._signature!r} is incompatible with {signature!r}")
        return VeRecordingAdapter(self, request, callback, fail, stop_event, signature,
                                  startup_budget, on_retry)

    def release(self, timeout):
        timeout = self._validated_timeout(timeout)
        started = time.perf_counter()
        association = "last_bound" if self._has_bound_request else "initial_request"
        self._resource_diagnostics.emit("REQUEST", "release", request_association=association)
        outcome = self._release(timeout)
        self._resource_diagnostics.emit(
            "END" if outcome.success else "ERROR", "release",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            request_association=association, success=outcome.success)
        return outcome

    def _release(self, timeout):
        with self._lock:
            self._retire_cancelled_startup_locked()
            if self._state == self._FAILED:
                return self._failed_outcome_locked()
            if self._state == self._STARTING:
                return VeResourceReleaseOutcome(
                    False, self._signature,
                    ("release: initialization in progress",),
                    self._counts_snapshot_locked())
            if self._owner is None:
                return VeResourceReleaseOutcome(
                    True, None, (), self._counts_snapshot_locked())
            if self._active is not None:
                return VeResourceReleaseOutcome(
                    False, self._signature, ("release: active adapter",),
                    self._counts_snapshot_locked())
            released_signature = self._signature
            if self._state != self._RELEASING:
                self._state = self._RELEASING
                self._release_outcome = None
                self._submit(_Command("release"))
            owner = self._owner
        owner.join(timeout)
        if owner.is_alive():
            self._resource_diagnostics.timeout(
                "release", owner, resource_state=self._state, request_association=(
                    "last_bound" if self._has_bound_request else "initial_request"))
            message = "VE native owner did not exit before release timeout"
            self._mark_failed("release", message)
            with self._lock:
                self._release_outcome = VeResourceReleaseOutcome(
                    False, released_signature, tuple(self._diagnostics),
                    self._counts_snapshot_locked())
                return self._release_outcome
        with self._lock:
            if self._release_outcome is None:
                return self._failed_outcome_locked()
            return self._release_outcome

    def close(self, timeout):
        timeout = self._validated_timeout(timeout)
        started = time.monotonic()
        with self._lock:
            self._retire_cancelled_startup_locked()
            self._closing = True
            active = self._active
            starting = self._state == self._STARTING
            owner = self._owner
        if starting:
            self._mark_failed("close", "VE controller closed during initialization")
            if owner is not None:
                owner.join(timeout)
            with self._lock:
                return self._failed_outcome_locked()
        if active is not None and not self._detach(active, min(timeout, self._detach_timeout)):
            with self._lock:
                return self._failed_outcome_locked()
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            self._mark_failed("close", "VE controller close deadline exceeded")
            with self._lock:
                return self._failed_outcome_locked()
        return self.release(remaining)

    @property
    def signature(self):
        with self._lock:
            return self._signature

    @property
    def failed(self):
        with self._lock:
            return self._state == self._FAILED

    @property
    def lifecycle_counts(self):
        with self._lock:
            return self._counts_snapshot_locked()

    @property
    def failure_snapshot(self):
        with self._lock:
            return self._failure

    @staticmethod
    def _validated_timeout(timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        return float(timeout)

    def _counts_snapshot_locked(self):
        return VeLifecycleCounts(**self._counts)

    def _failed_outcome_locked(self):
        if self._release_outcome is not None and not self._release_outcome.success:
            return self._release_outcome
        return VeResourceReleaseOutcome(
            False, self._signature, tuple(self._diagnostics),
            self._counts_snapshot_locked())

    def _submit(self, command):
        self._commands.put(command)
        self._command_wakeup.set()

    def _bind(self, adapter):
        adapter._diagnostic_bind_at = time.perf_counter()
        with self._lock:
            self._retire_cancelled_startup_locked()
            if self._startup_attempt is not None:
                raise RuntimeError("VE resource initialization already in progress")
            budgeted = self._owner is None and self._startup_attempt is None
            if budgeted and adapter.startup_budget is None:
                adapter.startup_budget = VeStartupBudget.create(self._clock())
        if budgeted:
            return self._bind_startup(adapter)
        with self._lock:
            if self._startup_attempt is not None:
                raise RuntimeError("VE resource initialization already in progress")
            if self._state == self._FAILED:
                raise RuntimeError("VE resource controller has permanently failed")
            if self._signature is not None and self._signature != adapter.signature:
                raise VeResourceConfigurationError("VE resource signature changed before bind")
            if self._owner is None:
                self._signature = adapter.signature
                self._state = self._STARTING
                self._task = "VE_" + uuid.uuid4().hex[:24]
                self._owner = threading.Thread(
                    target=self._run, args=(adapter,), name=self._task, daemon=True)
                self._owner.start()
            self._submit(_Command("bind", adapter))
        if not adapter._bound.wait(self._bind_timeout):
            self._resource_diagnostics.timeout(
                "bind", self._owner, resource_state=self._state,
                request_id=getattr(adapter.request, "request_id", None),
                warmup_id=getattr(adapter.request, "warmup_id", None))
            message = "VE adapter bind confirmation timed out"
            adapter._bind_timeout(message)
            self._mark_failed("bind", message)
            return False
        return adapter.started.is_set()

    def _checkpoint_locked(self, attempt):
        if not attempt.revoked.is_set():
            if (self._closing or attempt.adapter.stop_event.is_set()
                    or self._state == self._FAILED):
                attempt.reason = "cancelled"
            elif self._clock() >= attempt.deadline:
                attempt.reason = "timeout"
            if attempt.reason is not None:
                attempt.revoked.set()
        if attempt.revoked.is_set():
            raise _StartupRevoked(attempt.reason)

    def _checkpoint(self, attempt):
        if attempt is not None:
            with self._lock:
                self._checkpoint_locked(attempt)

    def _start_attempt_locked(self, attempt):
        self._startup_attempt = attempt
        self._signature = attempt.adapter.signature
        self._state = self._STARTING
        self._task = attempt.task
        self._release_outcome = None
        attempt.owner = threading.Thread(target=self._run_startup, args=(attempt,),
                                         name=attempt.task, daemon=True)
        self._owner = attempt.owner
        attempt.owner.start()

    def _startup_failure(self, adapter, stage, message):
        self._resource_diagnostics.emit("ERROR", stage, detail=message)
        self._mark_failed(stage, message, adapter)
        adapter._bound.set()
        released = self._startup_handles_released(adapter)
        with adapter._lock:
            adapter._uncertain = not (adapter._startup_release_confirmed or released)
        adapter._detached.set()
        return False

    def _failed_startup(self, adapter):
        fault = self.failure_snapshot
        if fault is not None:
            adapter._record_failure(fault.stage, fault.detail, fault)
        released = self._startup_handles_released(adapter)
        with adapter._lock:
            adapter._uncertain = not (adapter._startup_release_confirmed or released)
        adapter._bound.set()
        adapter._detached.set()
        return False

    def _startup_handles_released(self, adapter):
        with self._lock:
            attempt = self._startup_attempt
            if attempt is None or attempt.adapter is not adapter or attempt.committed:
                return None
            released = (attempt.cleanup_success and attempt.finished.is_set()
                        and (attempt.owner is None or not attempt.owner.is_alive()))
            if released:
                if attempt.reason == "cancelled" or self._state == self._FAILED:
                    # Terminal proof belongs to the adapter, surviving removal
                    # of its attempt and any earlier in-flight False snapshots.
                    # Timeout cleanup before retry is deliberately not terminal.
                    with adapter._lock:
                        adapter._startup_release_confirmed = True
                        adapter._uncertain = False
                self._retire_cancelled_startup_locked()
            return released

    def _retire_cancelled_startup_locked(self):
        attempt = self._startup_attempt
        if (attempt is not None and attempt.reason == "cancelled"
                and attempt.cleanup_success and attempt.finished.is_set()
                and (attempt.owner is None or not attempt.owner.is_alive())
                and self._state != self._FAILED and not self._closing):
            with attempt.adapter._lock:
                attempt.adapter._startup_release_confirmed = True
                attempt.adapter._uncertain = False
            self._startup_attempt = None
            self._owner = None
            self._signature = None
            self._state = self._UNINITIALIZED

    def _publish_startup_native_failure(self, attempt):
        stage, message, fault = attempt.native_failure
        self._mark_failed(stage, message, attempt.adapter, fault)
        # Preserve the original native fault immediately, then allow the owner
        # its bounded normal cleanup before taking a terminal ownership snapshot.
        # No SDK work runs on this waiter and no new startup budget is created.
        if attempt.owner is not None:
            attempt.owner.join(min(self._detach_timeout,
                                   attempt.adapter.startup_budget.remaining(self._clock())))
        return self._failed_startup(attempt.adapter)

    def _wait_startup(self, attempt):
        """Waiter only revokes. The owner remains the sole native caller."""
        while True:
            with self._lock:
                if attempt.committed:
                    return True
                if attempt.native_failure is not None:
                    return False
                if self._state == self._FAILED:
                    return False
                try:
                    self._checkpoint_locked(attempt)
                except _StartupRevoked:
                    return False
                if attempt.finished.is_set():
                    return False
            attempt.adapter._bound.wait(min(.01, max(0, attempt.deadline - self._clock())))

    def _bind_startup(self, adapter):
        budget = adapter.startup_budget
        first = _StartupAttempt(adapter, 1, "VE_" + uuid.uuid4().hex[:24],
                                budget.first_attempt_deadline)
        with self._lock:
            if self._state == self._FAILED:
                raise RuntimeError("VE resource controller has permanently failed")
            if self._owner is not None or self._startup_attempt is not None:
                raise RuntimeError("VE resource initialization already in progress")
            before = self._counts_snapshot_locked()
            self._startup_attempt = first
            self._state = self._STARTING
            self._signature = adapter.signature
            try:
                self._checkpoint_locked(first)
            except _StartupRevoked:
                # A queued request may have no owner and no native calls.
                first.cleanup_success = True
                first.finished.set()
            else:
                self._start_attempt_locked(first)
        if self._wait_startup(first):
            return True
        if first.native_failure is not None:
            return self._publish_startup_native_failure(first)
        if self.failed:
            return self._failed_startup(adapter)
        if first.reason == "cancelled":
            return self._cancel_startup(first)
        if first.reason != "timeout":
            return self._startup_failure(adapter, "bind", "VE startup ended without bind confirmation")
        self._resource_diagnostics.timeout("startup_revoke", first.owner,
                                           attempt=1, task=first.task,
                                           startup_budget=vars(budget))
        # The absolute cleanup window includes the blocked call and owner exit.
        while not first.finished.is_set() and budget.cleanup_remaining(self._clock()) > 0:
            if self.failed:
                return self._failed_startup(adapter)
            if adapter.stop_event.is_set() or self._closing:
                return self._cancel_startup(first)
            first.finished.wait(min(.01, budget.cleanup_remaining(self._clock())))
        while (first.owner is not None and first.owner.is_alive()
               and budget.cleanup_remaining(self._clock()) > 0):
            if self.failed:
                return self._failed_startup(adapter)
            if adapter.stop_event.is_set() or self._closing:
                return self._cancel_startup(first)
            first.owner.join(min(.01, budget.cleanup_remaining(self._clock())))
        now = self._clock()
        if self.failed:
            return self._failed_startup(adapter)
        if adapter.stop_event.is_set() or self._closing:
            return self._cancel_startup(first)
        if (now >= budget.cleanup_deadline or now >= budget.deadline
                or not first.finished.is_set()
                or (first.owner is not None and first.owner.is_alive())):
            return self._startup_failure(adapter, "startup_cleanup_timeout",
                                         "first initialization timed out; cleanup/owner exit unconfirmed")
        if not first.cleanup_success:
            return self._startup_failure(adapter, "startup_cleanup_failed",
                                         "first initialization timed out; " + "; ".join(first.cleanup_errors))
        if adapter.stop_event.is_set() or self._closing:
            return self._cancel_startup(first)
        second = _StartupAttempt(adapter, 2, "VE_" + uuid.uuid4().hex[:24], budget.deadline)
        with self._lock:
            self._owner = None
            after = self._counts_snapshot_locked()
        proof = VeResourceRetryProof(
            getattr(adapter.request, "request_id", None) or adapter.request.warmup_id,
            self._generation, adapter.signature, 2, first.task, second.task,
            budget, now, before, after)
        self._resource_diagnostics.emit("END", "startup_cleanup", attempt=1,
                                       old_task_id=first.task, new_task_id=second.task,
                                       cleanup_confirmed_at=now,
                                       before_counts=vars(before), after_cleanup_counts=vars(after),
                                       startup_budget=vars(budget),
                                       remaining_budget=budget.remaining(now))
        if adapter.on_retry is not None:
            try:
                adapter.on_retry(proof)
            except Exception as exc:
                # Reliable publication is an external boundary. Never open a
                # replacement whose lifecycle proof the parent did not receive.
                return self._startup_failure(adapter, "startup_retry_failed",
                                             f"retry proof delivery failed: {exc}")
        with adapter._lock:
            adapter._pending_started_at = None
            adapter._deadline = None
            adapter._progress = VeCaptureProgress(None, 0, None)
        with self._lock:
            try:
                self._checkpoint_locked(second)
            except _StartupRevoked:
                pass  # Handled outside the lock, without native work/callbacks.
            else:
                self._start_attempt_locked(second)
        if second.reason == "cancelled":
            # A cancellation during proof delivery can precede installation of
            # the replacement. Retire the still-registered, confirmed-clean
            # first attempt; there is no second owner to cancel in that case.
            return self._cancel_startup(first if second.owner is None else second)
        if self._wait_startup(second):
            return True
        if second.native_failure is not None:
            return self._publish_startup_native_failure(second)
        if self.failed:
            return self._failed_startup(adapter)
        if second.reason == "cancelled":
            return self._cancel_startup(second)
        return self._startup_failure(adapter, "startup_deadline",
                                     "first initialization timed out; replacement exceeded startup deadline")

    def _cancel_startup(self, attempt):
        with self._lock:
            attempt.reason = "cancelled"
            attempt.revoked.set()
        adapter = attempt.adapter
        released = self._startup_handles_released(adapter)
        with adapter._lock:
            adapter._pending_started_at = None
            adapter._uncertain = not (adapter._startup_release_confirmed or released)
        adapter._bound.set()
        adapter._detached.set()
        return False

    def _run_startup(self, attempt):
        adapter = attempt.adapter
        try:
            if not self._initialize(adapter, attempt):
                return
            with self._lock:
                self._checkpoint_locked(attempt)
                if not adapter._accept_bind(self._clock, lambda: self._checkpoint_locked(attempt)):
                    attempt.reason = "cancelled"
                    attempt.revoked.set()
                    raise _StartupRevoked("cancelled")
                attempt.committed = True
                self._startup_attempt = None
                self._active = adapter
                self._state = self._RECORDING
            self._has_bound_request = True
            self._set_diagnostic_request(adapter)
            self._resource_diagnostics.emit(
                "END", "bind", success=True,
                elapsed_ms=(time.perf_counter() - adapter._diagnostic_bind_at) * 1000)
            self._owner_loop()
        except _StartupRevoked:
            self._resource_diagnostics.emit("DISCARD", "startup_late_return", attempt=attempt.number)
        finally:
            self._cleanup(None if attempt.committed else attempt)
            with self._lock:
                active = self._active
                self._active = None
            if active is not None:
                active._confirm_detached()
            if not attempt.committed and (self.failed or attempt.reason == "cancelled"):
                with adapter._lock:
                    adapter._uncertain = True  # This owner has not exited yet.
                adapter._confirm_detached()
                adapter._bound.set()
            self._reject_pending_binds()
            attempt.finished.set()

    def _detach(self, adapter, timeout=None):
        if adapter._detached.is_set():
            return adapter.handles_released
        timeout = self._detach_timeout if timeout is None else timeout
        self._submit(_Command("detach", adapter))
        if not adapter._detached.wait(timeout):
            message = "VE adapter detach confirmation timed out"
            adapter._detach_timeout(message)
            self._mark_failed("detach", message, adapter)
            return False
        return adapter.handles_released

    def _mark_failed(self, stage, message, adapter=None, fault=None):
        callback = None
        with self._lock:
            self._diagnostics.append(f"{stage}: {message}")
            self._state = self._FAILED
            if self._failure is None:
                self._failure = (
                    VeResourceFault(stage, None, message)
                    if fault is None else fault
                )
            if not self._fatal_reported:
                self._fatal_reported = True
                callback = self._fatal_callback
        if adapter is not None:
            adapter._record_failure(stage, message, fault)
        if callback is not None:
            try:
                callback(stage, message)
            except Exception as exc:
                # Worker fatal publication is an external boundary. Cleanup
                # still owns all native resources even if publication fails.
                with self._lock:
                    self._diagnostics.append(f"fatal_callback: {exc}")
        self._command_wakeup.set()

    def _run(self, initial_adapter):
        try:
            if self._initialize(initial_adapter):
                self._owner_loop()
        finally:
            self._cleanup()
            with self._lock:
                active = self._active
                self._active = None
            if active is not None:
                active._confirm_detached()
            self._reject_pending_binds()

    def _owner_loop(self):
        while True:
            try:
                self._process_commands()
            except Exception as exc:
                self._mark_failed("owner_loop", str(exc))
                return
            with self._lock:
                if self._state in (self._RELEASING, self._FAILED):
                    return
                active = self._active
            if active is not None and active.stop_event.is_set():
                self._confirm_detach(active)
                continue
            read_adapter = active
            try:
                requested = self._requested_frames(read_adapter)
                if read_adapter is not None:
                    read_adapter._check_deadline()
                buffer, returned = self._sdk.read_task_data(
                    self._task, channel_count=len(self._signature[2]),
                    samples_per_channel=requested, timeout_seconds=.2)
                if read_adapter is not None:
                    read_adapter._check_deadline()
                self._validate_read(buffer, returned, requested)
                self._process_commands()
                with self._lock:
                    if self._state in (self._RELEASING, self._FAILED):
                        return
                    still_active = self._active
                if read_adapter is None:
                    if returned:
                        # Validate the native data before immediately dropping
                        # this one bounded idle block.
                        to_physical_slots(buffer, returned, self._signature[2])
                    else:
                        self._command_wakeup.wait(.01)
                        self._command_wakeup.clear()
                    continue
                if (still_active is not read_adapter
                        or read_adapter.stop_event.is_set()):
                    if still_active is read_adapter and read_adapter is not None:
                        self._confirm_detach(read_adapter)
                    continue
                if returned == 0:
                    self._command_wakeup.wait(.01)
                    self._command_wakeup.clear()
                    continue
                physical = to_physical_slots(buffer, returned, self._signature[2])
                complete = read_adapter._deliver(physical, returned, self._clock)
                if complete or read_adapter.stop_event.is_set():
                    self._confirm_detach(read_adapter)
            except Exception as exc:
                # ``read_adapter`` is the binding associated with this exact
                # read/callback operation; controller state may change while a
                # native call is blocked and cannot identify the fault owner.
                message, fault = self._failure_from_exception(
                    "read_task_data", exc)
                self._mark_failed(
                    "read_task_data", message, read_adapter, fault,
                )
                return

    def _latch_startup_native_failure(self, attempt, stage, exc):
        if attempt is not None:
            with self._lock:
                if (attempt.native_failure is None and not attempt.revoked.is_set()
                        and self._clock() < attempt.deadline
                        and not self._closing and not attempt.adapter.stop_event.is_set()):
                    message, fault = self._failure_from_exception(stage, exc)
                    if attempt.number == 2:
                        stage = "startup_retry_failed"
                        message = f"first initialization timed out; replacement {message}"
                        fault = VeResourceFault(stage, fault.code, message)
                    attempt.native_failure = stage, message, fault

    def _startup_native_call(self, attempt, stage, call, counter=None):
        # Diagnostic delivery may yield. Recheck immediately at the native
        # boundary and only count operations that will actually be invoked.
        self._checkpoint(attempt)
        if counter is not None:
            with self._lock:
                self._counts[counter] += 1
        try:
            return call()
        except Exception as exc:
            # Capture an SDK/loader failure before diagnostic delivery can
            # yield past the deadline. The waiter may publish it, but never
            # perform the owner's cleanup. Late failures remain diagnostics.
            self._latch_startup_native_failure(attempt, stage, exc)
            raise

    def _resolve_startup_device(self, adapter, attempt):
        try:
            return resolve_device(
                self._sdk, adapter.request.device["machine_id"], adapter.request.channels,
                observer=self._resource_diagnostics.observe,
                checkpoint=None if attempt is None else lambda: self._checkpoint(attempt),
                on_failure=None if attempt is None else lambda exc: self._latch_startup_native_failure(
                    attempt, "device", exc))
        except ValueError as exc:
            # Resolution has finished filtering candidates. Its final identity
            # or channel rejection is decisive before the outer span is logged.
            self._latch_startup_native_failure(attempt, "device", exc)
            raise

    def _initialize(self, adapter, attempt=None):
        self._has_bound_request = False
        self._set_diagnostic_request(adapter, reset=True, association="initial_request")
        if attempt is not None:
            self._resource_diagnostics.startup_attempt(
                attempt.number, attempt.task, adapter.startup_budget)
        initialization = self._resource_diagnostics.begin("initialize")
        stage = "sdk_open"
        try:
            self._checkpoint(attempt)
            self._sdk = self._resource_diagnostics.call(stage, lambda: self._startup_native_call(
                attempt, stage, self._sdk_factory, "sdk_open"))
            self._checkpoint(attempt)
            stage = "device"
            fresh = self._resource_diagnostics.call(
                stage, lambda: self._resolve_startup_device(adapter, attempt))
            self._checkpoint(attempt)
            stage = "create_task"
            self._resource_diagnostics.call(stage, lambda: self._startup_native_call(
                attempt, stage, lambda: self._sdk.create_task(self._task), "task_create"))
            self._task_created = True
            self._checkpoint(attempt)
            routes = ",".join(
                f"{fresh['name']}/AIN{channel + 1}" for channel in adapter.request.channels)
            stage = "create_iepe_voltage_channel"
            config = adapter.request.device["input_config"]
            self._resource_diagnostics.call(stage, lambda: self._startup_native_call(
                attempt, stage, lambda: self._sdk.create_iepe_voltage_channel(
                    self._task, routes, range_min=config["range_min"], range_max=config["range_max"])))
            self._checkpoint(attempt)
            stage = "configure_sample_clock"
            self._resource_diagnostics.call(stage, lambda: self._startup_native_call(
                attempt, stage, lambda: self._sdk.configure_sample_clock(self._task, adapter.request.sample_rate)))
            self._checkpoint(attempt)
            stage = "start_task"
            native_start = self._resource_diagnostics.begin(stage)
            self._startup_native_call(attempt, stage, lambda: self._sdk.start_task(self._task), "task_start")
            started_at = self._clock()
            self._resource_diagnostics.end(native_start)
            native_start = None
            self._checkpoint(attempt)
            adapter._latch_native_start(started_at)
            stage = "verify_actual_sample_rate"
            verification = self._resource_diagnostics.begin(stage)
            actual = self._startup_native_call(attempt, stage, lambda: self._sdk.verify_actual_sample_rate(
                fresh["name"], adapter.request.sample_rate))
            self._checkpoint(attempt)
            if type(actual) is not int or actual != adapter.request.sample_rate:
                raise ValueError(
                    f"actual sample rate {actual!r} differs from {adapter.request.sample_rate}")
            self._resource_diagnostics.end(verification, actual_sample_rate=actual)
            self._fresh = fresh
            self._resource_diagnostics.end(initialization)
            return True
        except _StartupRevoked:
            if stage == "start_task" and native_start is not None:
                self._resource_diagnostics.end(native_start, discarded=True)
            if stage == "verify_actual_sample_rate":
                self._resource_diagnostics.end(verification, discarded=True)
            self._resource_diagnostics.end(initialization, discarded=True)
            self._resource_diagnostics.emit("DISCARD", "startup_late_return",
                                           native_stage=stage, attempt=attempt.number)
            return False
        except Exception as exc:
            # SDK initialization is a single external boundary with explicit
            # operation provenance. It always belongs to the initial request.
            # Native-output validation also fails decisively here: latch it
            # before ending spans, whose diagnostic delivery may yield.
            self._latch_startup_native_failure(attempt, stage, exc)
            if stage == "start_task" and native_start is not None:
                self._resource_diagnostics.end(native_start, error=exc)
            if stage == "verify_actual_sample_rate":
                self._resource_diagnostics.end(verification, error=exc)
            self._resource_diagnostics.end(initialization, error=exc)
            message, fault = self._failure_from_exception(stage, exc)
            if attempt is not None and attempt.native_failure is not None:
                stage, message, fault = attempt.native_failure
                self._mark_failed(stage, message, adapter, fault)
                return False
            if attempt is not None:
                try:
                    self._checkpoint(attempt)
                except _StartupRevoked:
                    with self._lock:
                        self._diagnostics.append(f"attempt {attempt.number} late {stage}: {message}")
                    return False
                if attempt.number == 2:
                    stage = "startup_retry_failed"
                    message = f"first initialization timed out; replacement {message}"
                    fault = VeResourceFault(stage, fault.code, message)
            self._mark_failed(
                stage, message, adapter, fault,
            )
            return False

    def _set_diagnostic_request(self, adapter, *, reset=False, association="active"):
        request = adapter.request
        config = request.device["input_config"]
        self._resource_diagnostics.context(
            reset=reset, task=self._task,
            request_id=getattr(request, "request_id", None),
            warmup_id=getattr(request, "warmup_id", None),
            request_association=association,
            machine_id=request.device["machine_id"], channels=request.channels,
            sample_rate=request.sample_rate,
            range_min=config["range_min"], range_max=config["range_max"])

    @staticmethod
    def _failure_from_exception(stage, exc):
        if isinstance(exc, VkDaqError):
            code = exc.code if type(exc.code) is int else None
            detail = (exc.detail if type(exc.detail) is str
                      and exc.detail.strip() else type(exc).__name__)
            operation = (exc.operation if type(exc.operation) is str
                         and exc.operation.strip() else type(exc).__name__)
            message = f"{operation} (code={code}): {detail}"
            return message, VeResourceFault(stage, code, detail)
        message = str(exc)
        if not message.strip():
            message = type(exc).__name__
        return message, VeResourceFault(stage, None, message)

    def _process_commands(self):
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                self._command_wakeup.clear()
                return
            if command.kind == "release":
                with self._lock:
                    if self._state != self._FAILED:
                        self._state = self._RELEASING
                continue
            if command.kind == "detach":
                with self._lock:
                    active = self._active
                if active is command.adapter:
                    self._confirm_detach(active)
                elif command.adapter._bound.is_set():
                    command.adapter._confirm_detached()
                continue
            if command.kind != "bind":
                raise RuntimeError(f"unknown VE owner command {command.kind!r}")
            adapter = command.adapter
            with self._lock:
                state = self._state
                active = self._active
                compatible = adapter.signature == self._signature
            if (state in (self._STARTING, self._IDLE)
                    and active is None and compatible
                    and adapter._accept_bind(self._clock)):
                with self._lock:
                    self._active = adapter
                    self._state = self._RECORDING
                self._has_bound_request = True
                self._set_diagnostic_request(adapter)
                self._resource_diagnostics.emit(
                    "END", "bind", success=True,
                    elapsed_ms=(time.perf_counter() - adapter._diagnostic_bind_at) * 1000)
            else:
                with self._lock:
                    if self._state == self._STARTING and active is None:
                        self._state = self._IDLE
                adapter._reject_bind(
                    "bind", "VE resource is not idle and compatible")

    def _confirm_detach(self, adapter):
        with self._lock:
            if self._active is adapter:
                self._active = None
                if self._state != self._FAILED:
                    self._state = self._IDLE
        adapter._confirm_detached()

    def _requested_frames(self, adapter):
        if adapter is not None:
            with adapter._lock:
                return adapter._deadline.block_frames
        return max(1, min(2048, math.ceil(self._signature[3] * .05)))

    def _validate_read(self, buffer, returned, requested):
        if type(returned) is not int:
            raise ValueError("returned frame count must be an integer")
        if not 0 <= returned <= requested:
            raise VkDaqError(
                "read_task_data", returned,
                f"invalid sample count; requested {requested}")
        channels = len(self._signature[2])
        if (not isinstance(buffer, ctypes.Array) or buffer._type_ is not ctypes.c_double
                or len(buffer) != requested * channels):
            raise ValueError("VkDaq buffer capacity/type does not match the read request")

    def _cleanup(self, attempt=None):
        self._resource_diagnostics.context(request_association=(
            "last_bound" if self._has_bound_request else "initial_request"))
        with self._lock:
            sdk = self._sdk
            task = self._task
            released_signature = self._signature
            task_created = self._task_created
        cleanup_errors = []
        if attempt is not None:
            self._resource_diagnostics.emit("BEGIN", "startup_cleanup", attempt=attempt.number)
        operations = []
        if sdk is not None and task_created:
            operations.extend((
                ("stop_task", "task_stop", lambda: sdk.stop_task(task)),
                ("clear_task", "task_clear", lambda: sdk.clear_task(task)),
            ))
        if sdk is not None:
            operations.append(("close", "sdk_close", sdk.close))
        for operation, counter, call in operations:
            with self._lock:
                self._counts[counter] += 1
            cleanup_span = self._resource_diagnostics.begin(operation)
            try:
                call()
            except Exception as exc:
                detail = f"{operation}: {exc}"
                cleanup_errors.append(detail)
                with self._lock:
                    self._diagnostics.append(detail)
                if attempt is not None:
                    # Cancellation may already have returned the bind waiter.
                    # The owner must retire an unsafe controller immediately,
                    # before error logging or subsequent cleanup can block.
                    message, fault = self._failure_from_exception(operation, exc)
                    message = f"startup attempt {attempt.number} cleanup failed: {message}"
                    stage = "startup_cleanup_failed"
                    self._mark_failed(stage, message, attempt.adapter,
                                      VeResourceFault(stage, fault.code, message))
                    with self._lock:
                        attempt.cleanup_errors = tuple(cleanup_errors)
                self._resource_diagnostics.end(cleanup_span, error=exc)
            else:
                self._resource_diagnostics.end(cleanup_span)
        callback = None
        with self._lock:
            failed_before_cleanup = self._state == self._FAILED
            if attempt is not None:
                # Revocation is not a release failure. The waiter separately
                # verifies this result AND the actual owner thread exit.
                attempt.cleanup_success = not cleanup_errors
                attempt.cleanup_errors = tuple(cleanup_errors)
                if not cleanup_errors:
                    self._sdk = None
                    self._task = None
                    self._fresh = None
                    self._task_created = False
                self._release_outcome = VeResourceReleaseOutcome(
                    not cleanup_errors, released_signature, tuple(self._diagnostics),
                    self._counts_snapshot_locked())
                return
            if cleanup_errors:
                self._state = self._FAILED
                if not self._fatal_reported:
                    self._fatal_reported = True
                    callback = self._fatal_callback
            success = not failed_before_cleanup and not cleanup_errors
            if success:
                self._state = self._UNINITIALIZED
                self._signature = None
                self._owner = None
                self._sdk = None
                self._task = None
                self._fresh = None
                self._task_created = False
            else:
                self._state = self._FAILED
            self._release_outcome = VeResourceReleaseOutcome(
                success, released_signature, tuple(self._diagnostics),
                self._counts_snapshot_locked())
        if callback is not None:
            try:
                callback("cleanup", cleanup_errors[0])
            except Exception as exc:
                with self._lock:
                    self._diagnostics.append(f"fatal_callback: {exc}")

    def _reject_pending_binds(self):
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command.kind == "bind":
                command.adapter._reject_bind(
                    "bind", "VE resource owner exited before binding")
