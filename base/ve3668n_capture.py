"""Single-thread native VE3668N owner, injected for hardware-free tests."""
import ctypes
import logging
import math
import threading
import time
import uuid

import numpy as np

from base.ve3668n_capture_timing import VeCaptureDeadline, VeCaptureProgress
from base.ve3668n_discovery import resolve_device
from base.vkinging_sdk import VkDaqClient, VkDaqError


def to_physical_slots(buffer, returned, channels):
    """Convert the compact GROUP_BY_CHANNEL=1 prefix, never unused capacity.

    The vendor CHM does not define the short-read stride. This implements the
    chosen compact contract; real short-read layout still needs acceptance on
    hardware. NaN prefill makes unwritten gaps fail instead of becoming audio.
    """
    compact = np.ctypeslib.as_array(buffer)[:returned * len(channels)].reshape(len(channels), returned).T
    if not np.isfinite(compact).all():
        raise ValueError("VkDaq returned non-finite voltage samples")
    physical = np.zeros((returned, max(channels) + 1), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        physical[:, channels] = compact
    if not np.isfinite(physical).all():
        raise ValueError("VkDaq returned non-finite float32 voltage samples")
    return physical


class Ve3668nInputStream:
    """Thread-control facade; SDK construction/use/cleanup belong only to owner.

    In production this lives inside RecordingWorker's child, never in the GUI.
    Bounded waits cannot interrupt a hung DLL: an uncertain owner must cause
    process retirement, not a second owner or a retry on the same task.
    """

    def __init__(self, request, callback, fail, stop_event, sdk_factory=VkDaqClient,
                 clock=time.monotonic, *, ready_timeout=3.0, join_timeout=.5):
        for name, value in (("ready_timeout", ready_timeout), ("join_timeout", join_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.request = request
        self.callback = callback
        self.fail = fail
        self.stop_event = stop_event
        self._sdk_factory = sdk_factory
        self._clock = clock
        self._ready_timeout = ready_timeout
        self._join_timeout = join_timeout
        self._logger = logging.getLogger(__name__)
        self.started = threading.Event()
        self.done = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._owner = None
        self._sdk = None
        self._task = "VE_" + uuid.uuid4().hex[:24]
        self._uncertain = False
        self._failure = None
        self._diagnostics = []
        self._cleanup_stage = None
        self._progress = VeCaptureProgress(None, 0, None)

    @property
    def started_at(self):
        return self.progress_snapshot().started_at

    def progress_snapshot(self):
        """Immutable native progress, independent of preview and owner polling."""
        with self._lock:
            return self._progress

    @property
    def diagnostics(self):
        with self._lock:
            return tuple(self._diagnostics)

    @property
    def handles_released(self):
        with self._lock:
            return not self._uncertain and (self._owner is None or not self._owner.is_alive())

    def start(self):
        if self._owner is not None:
            raise RuntimeError("a VE input stream can only be started once")
        if self.stop_event.is_set():
            self._ready.set()
            self.done.set()
            return False
        self._owner = threading.Thread(target=self._run, name=self._task, daemon=True)
        self._owner.start()
        if not self._ready.wait(self._ready_timeout):
            self._report_failure("starting", "VE native startup timed out", uncertain=True)
        return self.started.is_set()

    def stop(self):
        self.stop_event.set()
        if self._owner is not None:
            self._owner.join(self._join_timeout)
            if self._owner.is_alive():
                cleanup_stage = self._cleanup_stage_snapshot() or "none"
                self._report_failure(
                    "close_stream",
                    f"VE native owner did not exit (cleanup_stage={cleanup_stage})",
                    uncertain=True,
                )

    def close(self):
        self.stop()

    def _cleanup_stage_snapshot(self):
        with self._lock:
            return self._cleanup_stage

    def _format_cleanup_diagnostic(self, event, operation, elapsed, error):
        line = (f"[VE cleanup] task={self._task} event={event} operation={operation} "
                f"elapsed_s={elapsed:.6f}")
        if error is not None:
            detail = str(error).replace("\r", " ").replace("\n", " ")
            line += f" error_type={type(error).__name__} error={detail}"
        return line

    def _record_diagnostic_output_failure(self, sink, error):
        with self._lock:
            self._diagnostics.append(f"diagnostic_output: {sink} failed: {error}")

    def _emit_cleanup_diagnostic(self, event, operation, elapsed, error=None):
        try:
            line = self._format_cleanup_diagnostic(event, operation, elapsed, error)
        except Exception as exc:
            # Best-effort diagnostic formatter boundary: malformed exception
            # text must not replace or interrupt native cleanup.
            self._record_diagnostic_output_failure("formatter", exc)
            return
        try:
            print(line, flush=True)
        except Exception as exc:
            # Console diagnostics are additive and cannot become capture state.
            self._record_diagnostic_output_failure("print", exc)
        try:
            if error is None:
                self._logger.info(line)
            else:
                self._logger.error(
                    line,
                    exc_info=(type(error), error, error.__traceback__),
                )
        except Exception as exc:
            # Application logging is another best-effort output boundary.
            self._record_diagnostic_output_failure("logger", exc)

    def _report_failure(self, stage, message, *, uncertain=False):
        with self._lock:
            self._uncertain |= uncertain
            self._diagnostics.append(f"{stage}: {message}")
            first = self._failure is None
            if first:
                self._failure = (stage, message)
        self.stop_event.set()
        self._ready.set()
        if first:
            self.fail(stage, message)

    def _run(self):
        created = False
        stage = "device"
        try:
            if self.stop_event.is_set():
                return
            self._sdk = self._sdk_factory()
            if self.stop_event.is_set():
                return
            fresh = resolve_device(self._sdk, self.request.device["machine_id"], self.request.channels)
            if self.stop_event.is_set():
                return
            stage = "create_task"
            self._sdk.create_task(self._task)
            created = True
            if self.stop_event.is_set():
                return
            stage = "create_iepe_voltage_channel"
            routes = ",".join(f"{fresh['name']}/AIN{channel + 1}" for channel in self.request.channels)
            self._sdk.create_iepe_voltage_channel(self._task, routes)
            if self.stop_event.is_set():
                return
            stage = "configure_sample_clock"
            self._sdk.configure_sample_clock(self._task, self.request.sample_rate)
            if self.stop_event.is_set():
                return
            stage = "start_task"
            self._sdk.start_task(self._task)
            started_at = self._clock()
            deadline = VeCaptureDeadline(self.request.sample_rate, self.request.target_samples,
                                         started_at, clock=self._clock)
            with self._lock:
                self._progress = deadline.snapshot()
            if self.stop_event.is_set():
                return
            stage = "verify_actual_sample_rate"
            actual = self._sdk.verify_actual_sample_rate(fresh["name"], self.request.sample_rate)
            if type(actual) is not int or actual != self.request.sample_rate:
                raise ValueError(f"actual sample rate {actual!r} differs from {self.request.sample_rate}")
            if self.stop_event.is_set():
                return
            deadline.check()
            self.started.set()
            self._ready.set()
            stage = "capture"
            self._read(deadline)
        except Exception as exc:
            # Owner thread's external SDK/callback boundary: SDK adapters can
            # raise arbitrary native errors. Publish once, then clean only our
            # successfully created task; do not let cleanup replace the cause.
            self._report_failure(stage, str(exc))
        finally:
            self._ready.set()
            if self._sdk is not None:
                operations = [("stop_task", lambda: self._sdk.stop_task(self._task)),
                              ("clear_task", lambda: self._sdk.clear_task(self._task))] if created else []
                operations.append(("close_sdk", self._sdk.close))
                for operation, close in operations:
                    with self._lock:
                        self._cleanup_stage = operation
                    started = self._clock()
                    self._emit_cleanup_diagnostic("start", operation, 0.0)
                    try:
                        close()
                    except Exception as exc:
                        # Real external cleanup boundary. Continue best-effort
                        # cleanup, retain every diagnostic and forbid reuse even
                        # if a subsequent Clear/close appears to succeed.
                        self._emit_cleanup_diagnostic(
                            "failed", operation, self._clock() - started, exc)
                        self._report_failure(operation, str(exc), uncertain=True)
                    else:
                        self._emit_cleanup_diagnostic(
                            "completed", operation, self._clock() - started)
                    finally:
                        with self._lock:
                            self._cleanup_stage = None
            self.done.set()

    def _read(self, deadline):
        received = 0
        channels = self.request.channels
        while not self.stop_event.is_set() and received < self.request.target_samples:
            deadline.check()
            requested = deadline.requested_frames
            buffer, returned = self._sdk.read_task_data(
                self._task, channel_count=len(channels), samples_per_channel=requested,
                timeout_seconds=.2,
            )
            if self.stop_event.is_set():
                return
            deadline.check()
            if type(returned) is not int or not 0 <= returned <= requested:
                raise VkDaqError("read_task_data", returned, f"invalid sample count; requested {requested}")
            if returned == 0:
                self.stop_event.wait(.01)
                continue
            if (not isinstance(buffer, ctypes.Array) or buffer._type_ is not ctypes.c_double
                    or len(buffer) != requested * len(channels)):
                raise ValueError("VkDaq buffer capacity/type does not match the read request")
            physical = to_physical_slots(buffer, returned, channels)
            received += returned
            deadline.observe(received)
            with self._lock:
                self._progress = deadline.snapshot()
            self.callback(physical, returned, None, None)
