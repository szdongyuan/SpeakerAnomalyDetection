"""Persistent, instance-owned VE3668N native resource and request adapters."""
from dataclasses import dataclass
import ctypes
import math
import queue
import threading
import time
import uuid

from base.recording_process_protocol import VeLifecycleCounts
from base.ve3668n_capture import to_physical_slots
from base.ve3668n_capture_timing import VeCaptureDeadline, VeCaptureProgress
from base.ve3668n_discovery import resolve_device
from base.ve3668n_input import ve_acquisition_signature
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


class VeRecordingAdapter:
    """One request's synchronized binding to a persistent controller."""

    def __init__(self, controller, request, callback, fail, stop_event, signature):
        self._controller = controller
        self.request = request
        self.callback = callback
        self.fail = fail
        self.stop_event = stop_event
        self.signature = signature
        self.started = threading.Event()
        self._bound = threading.Event()
        self._detached = threading.Event()
        self._lock = threading.RLock()
        self._start_attempted = False
        self._start_succeeded = False
        self._uncertain = False
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
        with self._lock:
            return self._detached.is_set() and not self._uncertain

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

    def _accept_bind(self, clock):
        with self._lock:
            if self._uncertain or self.stop_event.is_set():
                return False
            started_at = self._pending_started_at
            self._pending_started_at = None
            if started_at is None:
                started_at = clock()
            self._deadline = VeCaptureDeadline(
                self.request.sample_rate, self.request.target_samples,
                started_at, clock=clock,
            )
            self._progress = self._deadline.snapshot()
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
                 *, bind_timeout=3.0, detach_timeout=.5):
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

    def stream(self, *, request, callback, fail, stop_event):
        return self._adapter(
            request=request, callback=callback, fail=fail,
            stop_event=stop_event)

    def prewarm(self, *, request, fail):
        """Bind a no-file adapter that validates and discards target frames."""
        return self._adapter(
            request=request,
            callback=lambda _physical, _frames, _first, _second: None,
            fail=fail,
            stop_event=threading.Event(),
        )

    def _adapter(self, *, request, callback, fail, stop_event):
        signature = ve_acquisition_signature(
            request.device, request.channels, request.sample_rate)
        with self._lock:
            if self._state == self._FAILED:
                raise RuntimeError("VE resource controller has permanently failed")
            if self._state == self._RELEASING:
                raise RuntimeError("VE resource controller is releasing")
            if self._signature is not None and self._signature != signature:
                raise VeResourceConfigurationError(
                    f"retained VE signature {self._signature!r} is incompatible with {signature!r}")
        return VeRecordingAdapter(self, request, callback, fail, stop_event, signature)

    def release(self, timeout):
        timeout = self._validated_timeout(timeout)
        with self._lock:
            if self._state == self._FAILED:
                return self._failed_outcome_locked()
            if self._owner is None:
                return VeResourceReleaseOutcome(
                    True, None, (), self._counts_snapshot_locked())
            if self._state == self._STARTING:
                return VeResourceReleaseOutcome(
                    False, self._signature,
                    ("release: initialization in progress",),
                    self._counts_snapshot_locked())
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
            active = self._active
            starting = self._state == self._STARTING
            owner = self._owner
        if starting:
            self._mark_failed("close", "VE controller closed during initialization")
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
        with self._lock:
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
            message = "VE adapter bind confirmation timed out"
            adapter._bind_timeout(message)
            self._mark_failed("bind", message)
            return False
        return adapter.started.is_set()

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

    def _initialize(self, adapter):
        stage = "sdk_open"
        try:
            with self._lock:
                self._counts["sdk_open"] += 1
            self._sdk = self._sdk_factory()
            stage = "device"
            fresh = resolve_device(
                self._sdk, adapter.request.device["machine_id"], adapter.request.channels)
            stage = "create_task"
            with self._lock:
                self._counts["task_create"] += 1
            self._sdk.create_task(self._task)
            self._task_created = True
            routes = ",".join(
                f"{fresh['name']}/AIN{channel + 1}" for channel in adapter.request.channels)
            stage = "create_iepe_voltage_channel"
            config = adapter.request.device["input_config"]
            self._sdk.create_iepe_voltage_channel(
                self._task, routes, range_min=config["range_min"], range_max=config["range_max"])
            stage = "configure_sample_clock"
            self._sdk.configure_sample_clock(self._task, adapter.request.sample_rate)
            stage = "start_task"
            with self._lock:
                self._counts["task_start"] += 1
            self._sdk.start_task(self._task)
            adapter._latch_native_start(self._clock())
            stage = "verify_actual_sample_rate"
            actual = self._sdk.verify_actual_sample_rate(
                fresh["name"], adapter.request.sample_rate)
            if type(actual) is not int or actual != adapter.request.sample_rate:
                raise ValueError(
                    f"actual sample rate {actual!r} differs from {adapter.request.sample_rate}")
            self._fresh = fresh
            return True
        except Exception as exc:
            # SDK initialization is a single external boundary with explicit
            # operation provenance. It always belongs to the initial request.
            message, fault = self._failure_from_exception(stage, exc)
            self._mark_failed(
                stage, message, adapter, fault,
            )
            return False

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

    def _cleanup(self):
        with self._lock:
            sdk = self._sdk
            task = self._task
            released_signature = self._signature
            task_created = self._task_created
        cleanup_errors = []
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
            try:
                call()
            except Exception as exc:
                detail = f"{operation}: {exc}"
                cleanup_errors.append(detail)
                with self._lock:
                    self._diagnostics.append(detail)
        callback = None
        with self._lock:
            failed_before_cleanup = self._state == self._FAILED
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
