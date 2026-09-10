"""Instance-owned asynchronous recording process and file-lease lifecycle.

All callbacks run on the supervisor thread; Qt callers must queue them to Qt.
``result_ready`` offers arrays for semantic validation. Only ``accepted`` seals
success. Public methods only reserve state/enqueue work: no pipe/file/process I/O.
``released`` is the continuation for same-path reuse, moves and deletion.
``accepted`` precedes ``released``; starting/moving inside accepted is too early.
``release_failed`` reports a retained lease without changing the recording's
terminal result. It is not permission to reuse, move or delete that path.
Failure/cancellation may also precede release. Always use that exact session's
released continuation, never a mutable current-recording path, for file actions.
"""
from contextlib import ExitStack
from dataclasses import dataclass, replace
import logging
import math
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import threading
import time

from base.recording_process_protocol import (
    CaptureSlotReleased, FrozenConfig, RecordingCancelled, RecordingEvent,
    RecordingFailure, RecordingProgress, RecordingRequest, RecordingResult,
    VE_PREWARM_COMMAND, VE_PREWARM_DETACHING, VE_PREWARM_PROGRESS,
    VE_PREWARM_STARTED, VE_PREWARM_TERMINAL, VeLifecycleCounts,
    VePrewarmProgress, VePrewarmRequest, VePrewarmResult, VePrewarmStarted,
    VeReleaseOutcome, WorkerFatal,
)
from base.recording_result_reader import ResultReader
from base.recording_worker import recording_worker
from base.ve3668n_capture_timing import VeCaptureDeadline
from base.ve3668n_input import ve_acquisition_signature
from consts.ve3668n_consts import VE_BACKEND


@dataclass(frozen=True)
class RecordingCallbacks:
    started: object = None
    preview: object = None
    result_ready: object = None
    accepted: object = None
    failed: object = None
    cancelled: object = None
    released: object = None
    release_failed: object = None


class RecordingSession:
    def __init__(self, service, request, callbacks):
        self.service, self.request, self.callbacks = service, request, callbacks
        self.state = "starting"
        self.generation = self.worker_pid = None
        self.released = threading.Event()
        self.reader = None
        self.audio = self.descriptor = None
        self.failure = None
        self.acknowledged = False
        self.cancel_requested = False
        self._accept_requested = self._reject_requested = False
        self._terminal = False
        self._trusted_terminal = False
        # Reserving a path is not giving a child ownership. Only dispatching
        # start can let the worker open it; setup failures need no child release.
        self._child_released = True
        self._reader_released = True
        self._sent = False
        self._deadline = None
        self._capture_requested_at = None
        self._capture_deadline = None
        self._slot_release_deadline = None
        self._target_reached_at = None
        self._slot_released_at = None
        self._slot_lifecycle_counts = None
        self._admission_reason = None
        self._last_sequence = self._last_sample_stop = 0
        self._preview_pending = None
        self._preview_ack_requested = None
        self._preview_disabled = False
        self._temporary_dir = None
        self._cleanup_paths = ()
        self._cleanup_failed = False
        self.release_error = None
        self._release_actions = {}
        self._lease_key = None
        self._lease_keys = set()

    @property
    def lifecycle_diagnostics(self):
        """Immutable parent-observed admission evidence for production runs."""
        return {
            "request_id": self.request.request_id,
            "generation": self.generation,
            "target_reached_at": self._target_reached_at,
            "capture_slot_released_at": self._slot_released_at,
            "lifecycle_counts": self._slot_lifecycle_counts,
            "admission_reason": self._admission_reason,
        }

    def cancel(self):
        self.service.cancel(self.request.request_id)

    def accept_result(self):
        self.service.accept_result(self.request.request_id)

    def reject_result(self, reason="result rejected by caller"):
        self.service.reject_result(self.request.request_id, reason)

    def release_preview(self, sequence):
        self.service.release_preview(self.request.request_id, sequence)


class _Worker:
    def __init__(self, generation, process, control, preview, ready_deadline):
        self.generation, self.process = generation, process
        self.control, self.preview = control, preview
        self.outgoing = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.threads = []
        self.ready = False
        self.deadline = ready_deadline
        self.retiring = False
        self.kill_deadline = None
        self.kill_reported = False


@dataclass
class _PendingVeRelease:
    required_signature: tuple | None
    callback: object = None
    preparing: object = None
    generation: int | None = None
    deadline: float | None = None
    sent: bool = False


@dataclass(frozen=True)
class VePrewarmCompletion:
    """Parent-owned terminal delivered once for an admitted prewarm cycle."""
    warmup_id: str
    signature: tuple
    success: bool
    stage: str
    code: int | None
    detail: str
    diagnostics: tuple[str, ...]
    lifecycle_counts: VeLifecycleCounts
    ownership_safe: bool
    attempts: tuple[VePrewarmResult, ...] = ()

    @property
    def failed_signature(self):
        return None if self.success else self.signature


@dataclass
class _PendingVePrewarm:
    base_request: VePrewarmRequest
    callback: object
    current_request: VePrewarmRequest | None = None
    attempt: int = 1
    generation: int | None = None
    phase: str = "admitted"
    first_fault: VePrewarmResult | None = None
    results: tuple[VePrewarmResult, ...] = ()
    start_sent_at: float | None = None
    start_deadline: float | None = None
    capture_deadline: VeCaptureDeadline | None = None
    detach_deadline: float | None = None
    retry_deadline: float | None = None
    completed: bool = False
    ownership_safe: bool = True
    prerequisite_release: bool = False
    validation_error: str | None = None


class RecordingService:
    def __init__(self, *, backend_factory=None, backend_options=None,
                 ready_timeout=10.0, start_timeout=10.0, cancel_timeout=5.0,
                 shutdown_timeout=5.0, terminate_timeout=2.0, preview_interval=.05,
                 reader_factory=ResultReader, release_timeout=None,
                 pipeline_capacity=2, monotonic=None, retry_delay=.75):
        if backend_factory is not None and (not isinstance(backend_factory, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", backend_factory)):
            raise ValueError("backend_factory must be an importable module:function identifier")
        self._backend_factory = backend_factory
        self._backend_options = FrozenConfig.snapshot(backend_options or {}).to_dict()
        for name, value in (("ready", ready_timeout), ("start", start_timeout),
                            ("cancel", cancel_timeout), ("shutdown", shutdown_timeout),
                            ("terminate", terminate_timeout), ("preview", preview_interval)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} timeout/interval must be positive and finite")
        self._ready_timeout, self._start_timeout = ready_timeout, start_timeout
        self._cancel_timeout, self._shutdown_timeout = cancel_timeout, shutdown_timeout
        self._terminate_timeout, self._preview_interval = terminate_timeout, preview_interval
        self._release_timeout = shutdown_timeout if release_timeout is None else release_timeout
        if (not isinstance(self._release_timeout, (int, float))
                or not math.isfinite(self._release_timeout) or self._release_timeout <= 0):
            raise ValueError("release timeout must be positive and finite")
        if type(pipeline_capacity) is not int or pipeline_capacity not in (1, 2):
            raise ValueError("pipeline_capacity must be a strict integer 1 or 2")
        self._reader_factory = reader_factory
        self._clock = time.monotonic if monotonic is None else monotonic
        if not callable(self._clock):
            raise TypeError("monotonic must be callable")
        if (type(retry_delay) not in (int, float)
                or not math.isfinite(retry_delay) or retry_delay <= 0):
            raise ValueError("prewarm retry delay must be positive and finite")
        self._retry_delay = float(retry_delay)
        self._lock = threading.RLock()
        self._inbox = queue.Queue(maxsize=64)
        self._capture_session = None
        self._sessions = {}
        self._pipeline_capacity = pipeline_capacity
        self._pending_ve_release = None
        self._pending_ve_prewarm = None
        self._terminal_ve_prewarm_ids = set()
        self._retained_ve_signature = None
        self._retained_lifecycle_counts = None
        self._expected_next_lifecycle_counts = None
        self._ownership_uncertain = False
        self._leases = {}
        self._request_ids = set()
        self._worker = None
        self._generation = 0
        self._closing = False
        self._shutdown_deadline = None
        self._shutdown_callbacks = []
        self._shutdown_reported = False
        self.closed = threading.Event()
        self.diagnostics = []
        self.threads = []
        self._logger = logging.getLogger(__name__)
        self._supervisor = threading.Thread(target=self._run, name="recording-supervisor", daemon=True)
        self._start_thread(self._supervisor)

    @property
    def worker_pid(self):
        worker = self._worker
        return worker.process.pid if worker is not None else None

    @property
    def generation(self):
        return self._generation

    @property
    def busy(self):
        with self._lock:
            return bool(self._sessions or self._pending_ve_release or self._pending_ve_prewarm
                        or self._ownership_uncertain
                        or (self._worker is not None and self._worker.retiring))

    @property
    def can_start_recording(self):
        with self._lock:
            worker = self._worker
            return (not self._closing and not self.closed.is_set()
                    and not self._ownership_uncertain
                    and self._pending_ve_release is None
                    and self._pending_ve_prewarm is None
                    and self._capture_session is None
                    and len(self._sessions) < self._pipeline_capacity
                    and (worker is None or (worker.ready and not worker.retiring)))

    @property
    def retained_ve_signature(self):
        with self._lock:
            return self._retained_ve_signature

    def session_diagnostics(self, request_id):
        with self._lock:
            session = self._sessions.get(request_id)
            return None if session is None else dict(session.lifecycle_diagnostics)

    def is_path_leased(self, path):
        with self._lock:
            return self._path_key(path) in self._leases

    def defer_path_cleanup(self, path, cleanup):
        """Reserve exact-path cleanup before its current lease can be released.

        The parent-only callback runs on the supervisor after child/reader handles
        close, while the same session still owns this path. It must not touch Qt.
        False means no lease exists; no delayed authority is retained in that case.
        """
        key = self._path_key(path)
        with self._lock:
            session = self._leases.get(key)
            if session is None:
                return False
            session._release_actions.setdefault(key, (os.path.abspath(path), cleanup))
            return True

    @staticmethod
    def _path_key(path):
        return os.path.normcase(os.path.abspath(path))

    def start(self, request, callbacks=None):
        if not isinstance(request, RecordingRequest):
            raise TypeError("start requires a RecordingRequest")
        with self._lock:
            if self._closing:
                raise RuntimeError("recording service is shutting down")
            key = self._path_key(request.path) if request.purpose == "main" else None
            if key is not None and key in self._leases:
                raise RuntimeError("recording path is still leased; service is busy for this path")
            if not self.can_start_recording:
                raise RuntimeError("recording service is busy or pipeline capacity is exhausted")
            if request.request_id in self._request_ids:
                raise ValueError("request_id must be unique for the lifetime of this service")
            session = RecordingSession(self, request, callbacks or RecordingCallbacks())
            session._lease_key = key
            if key is not None:
                self._leases[key] = session
                session._lease_keys.add(key)
            self._capture_session = session
            self._sessions[request.request_id] = session
            self._request_ids.add(request.request_id)
            required = self._request_signature(request)
            if self._retained_ve_signature is not None and required != self._retained_ve_signature:
                session.state = "preparing"
                self._pending_ve_release = _PendingVeRelease(
                    required_signature=required, preparing=session)
            self._inbox.put_nowait(("start", session))
            return session

    def _session(self, request_id):
        return self._sessions.get(request_id)

    @staticmethod
    def _request_signature(request):
        if request.device.get("backend") != VE_BACKEND:
            return None
        return ve_acquisition_signature(request.device, request.channels, request.sample_rate)

    def release_ve(self, required_signature=None, callback=None):
        """Asynchronously release an incompatible retained VE native resource.

        The immediate status is one of ``released``, ``unchanged``, ``pending``,
        ``busy`` or ``closing``. A callback, when supplied, is invoked by the
        supervisor as ``callback(status, diagnostics)``.
        """
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            worker = self._worker
            if self._ownership_uncertain or (worker is not None and worker.retiring):
                return "busy"
            if self._capture_session is not None:
                return "busy"
            if self._pending_ve_prewarm is not None:
                return "busy"
            if self._pending_ve_release is not None:
                return "pending"
            if worker is None or self._retained_ve_signature is None:
                self._inbox.put_nowait(("release_callback", callback, "released", ()))
                return "released"
            if required_signature == self._retained_ve_signature:
                self._inbox.put_nowait(("release_callback", callback, "unchanged", ()))
                return "unchanged"
            self._pending_ve_release = _PendingVeRelease(required_signature, callback)
            self._inbox.put_nowait(("release_ve",))
            return "pending"

    def prewarm_ve(self, request, callback):
        """Atomically reserve the sole hardware slot for a no-file VE prewarm."""
        if not isinstance(request, VePrewarmRequest):
            raise TypeError("prewarm_ve requires a VePrewarmRequest")
        if callback is not None and not callable(callback):
            raise TypeError("prewarm callback must be callable or None")
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            worker = self._worker
            if (self._ownership_uncertain
                    or self._capture_session is not None
                    or self._pending_ve_release is not None
                    or self._pending_ve_prewarm is not None
                    or (worker is not None and (worker.retiring or not worker.ready))):
                return "busy"
            pending = _PendingVePrewarm(request, callback)
            try:
                VePrewarmRequest.__post_init__(request)
            except (TypeError, ValueError, AttributeError, OverflowError) as exc:
                pending.validation_error = str(exc)
            self._pending_ve_prewarm = pending
            self._inbox.put_nowait(("prewarm_ve", pending))
            return "accepted"

    @staticmethod
    def _empty_ve_lifecycle_counts():
        return VeLifecycleCounts(0, 0, 0, 0, 0, 0)

    def _begin_ve_prewarm(self, pending):
        if pending is not self._pending_ve_prewarm or pending.completed:
            return
        if pending.validation_error is not None:
            self._finish_ve_prewarm(
                pending, success=False, stage="validation", code=None,
                detail=pending.validation_error, ownership_safe=True)
            return
        retained = self._retained_ve_signature
        if retained is not None and retained != pending.base_request.signature:
            pending.phase = "releasing"
            pending.prerequisite_release = True
            pending.generation = None if self._worker is None else self._worker.generation
            release = _PendingVeRelease(
                pending.base_request.signature, preparing=pending)
            self._pending_ve_release = release
            if self._worker is None:
                self._remember_prewarm_fault(
                    pending, "release_ve",
                    "retained VE signature has no live resource owner")
                self._pending_ve_release = None
                self._finish_ve_prewarm(
                    pending, success=False, ownership_safe=True)
            elif self._worker.ready:
                self._send_ve_release(release)
            return
        self._dispatch_ve_prewarm_attempt(pending)

    def _dispatch_ve_prewarm_attempt(self, pending):
        if (pending is None or pending is not self._pending_ve_prewarm or pending.completed
                or self._closing or self._ownership_uncertain):
            return
        pending.current_request = replace(pending.base_request, attempt=pending.attempt)
        if self._worker is None:
            try:
                self._spawn()
            except Exception as exc:
                worker = self._worker
                pending.generation = None if worker is None else worker.generation
                self._remember_prewarm_fault(pending, "service", str(exc))
                if worker is not None:
                    pending.phase = (
                        "retiring_retry" if pending.attempt == 1
                        else "retiring_final")
                    self._retire_generation(worker, "service", str(exc))
                elif pending.attempt == 1:
                    pending.phase = "retry_wait"
                    pending.retry_deadline = self._clock() + self._retry_delay
                else:
                    self._finish_ve_prewarm(
                        pending, success=False, ownership_safe=True)
                return
        worker = self._worker
        pending.generation = worker.generation
        pending.start_sent_at = None
        pending.start_deadline = None
        pending.capture_deadline = None
        pending.detach_deadline = None
        pending.retry_deadline = None
        pending.phase = "waiting_ready"
        if worker.ready:
            self._send_ve_prewarm(pending)

    def _send_ve_prewarm(self, pending):
        worker = self._worker
        if (pending is not self._pending_ve_prewarm or pending.completed
                or worker is None or worker.retiring or not worker.ready
                or pending.generation != worker.generation
                or pending.phase not in ("waiting_ready", "admitted")):
            return
        request = pending.current_request
        pending.phase = "starting"
        pending.start_sent_at = self._clock()
        pending.start_deadline = pending.start_sent_at + self._start_timeout
        worker.outgoing.put_nowait(RecordingEvent(
            worker.generation, request.warmup_id, VE_PREWARM_COMMAND, request))

    def _remember_prewarm_fault(self, pending, stage, detail, *, code=None,
                                frames=0, handles_released=False,
                                diagnostics=(), lifecycle_counts=None):
        request = pending.current_request or replace(
            pending.base_request, attempt=pending.attempt)
        fault = VePrewarmResult(
            request.warmup_id, pending.generation or 1, request.attempt,
            request.signature, False, stage, code, str(detail), frames,
            handles_released, tuple(diagnostics),
            lifecycle_counts or self._empty_ve_lifecycle_counts())
        if pending.first_fault is None:
            pending.first_fault = fault
        pending.results += (fault,)
        return fault

    def _finish_ve_prewarm(self, pending, *, success=False, result=None,
                           stage=None, code=None, detail=None,
                           ownership_safe=True):
        if pending is not self._pending_ve_prewarm or pending.completed:
            return
        pending.completed = True
        pending.ownership_safe = ownership_safe
        if result is not None and result not in pending.results:
            pending.results += (result,)
        first = pending.first_fault
        final = result or (pending.results[-1] if pending.results else None)
        if success:
            cause_stage, cause_code, cause_detail = "completed", None, ""
        elif first is not None:
            cause_stage, cause_code, cause_detail = first.stage, first.code, first.detail
        else:
            cause_stage = stage or "prewarm"
            cause_code = code
            cause_detail = detail or "VE prewarm failed"
        diagnostics = []
        for attempt in pending.results:
            diagnostics.extend(attempt.diagnostics)
            if not attempt.success:
                diagnostics.append(
                    f"attempt {attempt.attempt}: {attempt.stage}"
                    + (f" (code={attempt.code})" if attempt.code is not None else "")
                    + f": {attempt.detail}")
        counts = (final.lifecycle_counts if final is not None
                  else self._empty_ve_lifecycle_counts())
        completion = VePrewarmCompletion(
            pending.base_request.warmup_id, pending.base_request.signature,
            success, cause_stage, cause_code, cause_detail,
            tuple(diagnostics), counts, ownership_safe, pending.results)
        callback = pending.callback
        with self._lock:
            if self._pending_ve_prewarm is pending:
                self._pending_ve_prewarm = None
                self._terminal_ve_prewarm_ids.add(pending.base_request.warmup_id)
        if callback is not None:
            try:
                callback(completion)
            except Exception:
                self._logger.exception("VE prewarm callback failed")

    def _retire_ve_prewarm(self, pending, stage, detail, *, result=None,
                           retryable=True, release_failure=False):
        if pending.phase == "releasing":
            self._retire_prerequisite_release(
                self._worker, detail,
                diagnostics=() if result is None else result.diagnostics)
            return
        if result is not None:
            if result not in pending.results:
                pending.results += (result,)
            if pending.first_fault is None:
                pending.first_fault = result
        else:
            self._remember_prewarm_fault(pending, stage, detail)
        if release_failure:
            pending.phase = "retiring_release"
        elif retryable and pending.attempt == 1:
            pending.phase = "retiring_retry"
        else:
            pending.phase = "retiring_final"
        worker = self._worker
        if worker is None:
            self._finish_ve_prewarm(pending, success=False, ownership_safe=True)
        else:
            self._retire_generation(worker, stage, detail)

    def _transition_prerequisite_release_failure(self, pending, detail, *, diagnostics=()):
        """Latch every prerequisite-release error into the same non-retry path."""
        if (pending is None or pending is not self._pending_ve_prewarm
                or pending.completed
                or pending.phase not in ("releasing", "retiring_release")):
            return False
        if pending.phase == "releasing":
            self._remember_prewarm_fault(
                pending, "release_ve", detail, diagnostics=diagnostics,
                lifecycle_counts=self._retained_lifecycle_counts)
        pending.phase = "retiring_release"
        return True

    def _retire_prerequisite_release(self, worker, detail, *, diagnostics=()):
        pending = self._pending_ve_prewarm
        if not self._transition_prerequisite_release_failure(
                pending, detail, diagnostics=diagnostics):
            return False
        if worker is None:
            self._finish_ve_prewarm(
                pending, success=False, ownership_safe=True)
        elif not worker.retiring:
            self._retire_generation(
                worker, "release_ve", detail,
                pending_diagnostics=diagnostics or None)
        return True

    def cancel(self, request_id):
        with self._lock:
            session = self._session(request_id)
            if session is not None and not session._terminal and not session.cancel_requested:
                session.cancel_requested = True
                self._clear_unsent_preparing_release(session)
                self._inbox.put_nowait(("cancel", session))

    def accept_result(self, request_id):
        with self._lock:
            session = self._session(request_id)
            if (session is not None and not session._terminal and not session.cancel_requested
                    and not session._accept_requested and not session._reject_requested):
                session._accept_requested = True
                self._inbox.put_nowait(("accept", session))

    def reject_result(self, request_id, reason="result rejected by caller"):
        with self._lock:
            session = self._session(request_id)
            if session is not None and not session._terminal and not session._reject_requested:
                session._reject_requested = True
                self._inbox.put_nowait(("reject", session, reason))

    def release_preview(self, request_id, sequence):
        with self._lock:
            session = self._session(request_id)
            if (session is not None and session._preview_pending == sequence
                    and session._preview_ack_requested != sequence):
                session._preview_ack_requested = sequence
                self._inbox.put_nowait(("preview_ack", session, sequence))

    def shutdown(self, callback=None):
        """Stop asynchronously. Callback reports bounded shutdown, possibly with
        diagnosed pending leases; ``closed`` additionally requires all leases gone.
        A stuck reader remains supervised and cannot grant same-path reuse.
        """
        already_reported = False
        with self._lock:
            already_reported = self._shutdown_reported
            if callback is not None and not already_reported and callback not in self._shutdown_callbacks:
                self._shutdown_callbacks.append(callback)
            if not self._closing:
                self._closing = True
                self._inbox.put_nowait(("shutdown",))
        if callback is not None and already_reported:
            # Preserve asynchronous notification even after the supervisor exits.
            thread = threading.Thread(target=self._invoke_shutdown, args=(callback,),
                                      name="recording-shutdown-notification", daemon=True)
            self._start_thread(thread)

    def _start_thread(self, thread, *, start=None, worker=None):
        try:
            (start or thread.start)()
        finally:
            # ident remains set even after a fast thread exits. An injected start
            # may launch the thread and then raise: it still requires ownership.
            # Never register an unstarted thread that cannot safely be joined.
            if thread.ident is not None:
                self.threads.append(thread)
                if worker is not None:
                    worker.threads.append(thread)

    def _diagnose(self, message):
        self.diagnostics.append(message)
        self._logger.error(message)

    def _notify(self, session, kind, payload=None):
        callback = getattr(session.callbacks, kind)
        if callback is None:
            return
        try:
            if kind in ("started", "released"):
                callback(session)
            else:
                callback(session, payload)
        except Exception as exc:
            # User callback boundary: diagnose rather than kill the supervisor or
            # silently accept a result. Presentation failure doesn't invalidate WAV.
            self._logger.exception("Recording %s callback failed", kind)
            self._diagnose(f"{session.request.request_id}: {kind} callback failed: {exc}")
            if kind == "result_ready":
                self._fail(session, "delivery", str(exc))
            elif kind == "preview":
                session._preview_disabled = True
                self.release_preview(session.request.request_id, payload.sequence)

    def _spawn(self):
        context = multiprocessing.get_context("spawn")
        # Until a child actually exists, every acquired endpoint/wrapper belongs
        # to this setup transaction. The child-side endpoints always close here;
        # parent-side resources transfer to _dead only after successful spawn.
        with ExitStack() as rollback, ExitStack() as child_endpoints:
            control, child_control = context.Pipe(duplex=True)
            rollback.callback(control.close)
            child_endpoints.callback(child_control.close)
            preview, child_preview = context.Pipe(duplex=False)
            rollback.callback(preview.close)
            child_endpoints.callback(child_preview.close)
            self._generation += 1
            process = context.Process(target=recording_worker,
                args=(child_control, child_preview, self._generation, self._backend_factory,
                      self._backend_options, self._cancel_timeout, self._preview_interval),
                name=f"recording-worker-{self._generation}")
            rollback.callback(process.close)
            worker = _Worker(self._generation, process, control, preview,
                             self._clock() + self._ready_timeout)
            try:
                process.start()
            finally:
                if process.pid is not None:
                    # Also covers a custom start that launches and then raises.
                    # Preserve supervision before any IPC-thread startup failure.
                    self._worker = worker
                    if self._capture_session is not None:
                        self._capture_session.generation = worker.generation
                        self._capture_session.worker_pid = process.pid
                    rollback.pop_all()
        for target, args, name in ((self._send, (worker,), "sender"),
                                   (self._receive, (worker, control), "control"),
                                   (self._receive, (worker, preview), "preview")):
            thread = threading.Thread(target=target, args=args,
                                      name=f"recording-parent-{name}-{worker.generation}", daemon=True)
            self._start_thread(thread, worker=worker)

    def _receive(self, worker, connection):
        try:
            while not worker.stop.is_set():
                if connection.poll(.05):
                    event = connection.recv()
                    self._inbox.put(("event", worker, event))
        except (EOFError, OSError) as exc:
            if not worker.stop.is_set():
                self._inbox.put(("broken", worker, str(exc)))

    def _send(self, worker):
        try:
            while not worker.stop.is_set():
                try:
                    event = worker.outgoing.get(timeout=.05)
                except queue.Empty:
                    continue
                worker.control.send(event)
        except (EOFError, OSError) as exc:
            if not worker.stop.is_set():
                self._inbox.put(("broken", worker, str(exc)))

    def _command(self, kind, session=None, payload=None):
        worker = self._worker
        if (worker is not None and not worker.retiring
                and (session is None or session.generation == worker.generation)):
            worker.outgoing.put_nowait(RecordingEvent(worker.generation,
                session.request.request_id if session else "", kind, payload))

    def _begin(self, session):
        if session.cancel_requested:
            self._clear_unsent_preparing_release(session)
            session._child_released = True
            self._cancelled(session)
            return
        if session.request.purpose == "calibration":
            session._temporary_dir = tempfile.mkdtemp(prefix="recording-calibration-")
            session.request = replace(session.request, path=os.path.join(session._temporary_dir, "capture.wav"))
            session._lease_key = self._path_key(session.request.path)
            with self._lock:
                self._leases[session._lease_key] = session
                session._lease_keys.add(session._lease_key)
        if self._worker is None:
            self._spawn()
        session.generation = self._worker.generation
        session.worker_pid = self._worker.process.pid
        pending = self._pending_ve_release
        if pending is not None and pending.preparing is session:
            pending.generation = self._worker.generation
            if self._worker.ready:
                self._send_ve_release(pending)
        elif self._worker.ready:
            self._start_capture(session)

    def _send_ve_release(self, pending):
        if pending.sent or self._worker is None or self._worker.retiring:
            return
        pending.sent = True
        pending.generation = self._worker.generation
        pending.deadline = self._clock() + self._release_timeout
        try:
            self._command("release_ve")
        except Exception as exc:
            preparing = pending.preparing
            if (isinstance(preparing, _PendingVePrewarm)
                    and preparing is self._pending_ve_prewarm):
                self._retire_prerequisite_release(
                    self._worker, f"VE release command failed: {exc}")
                return
            raise

    def _finish_ve_release(self, status, diagnostics=()):
        pending = self._pending_ve_release
        if pending is None:
            return
        self._pending_ve_release = None
        if pending.callback is not None:
            self._invoke_release_callback(pending.callback, status, diagnostics)

    def _clear_unsent_preparing_release(self, session):
        """Cancel only an internal preparation that has not reached the child."""
        with self._lock:
            pending = self._pending_ve_release
            if (pending is None or pending.preparing is not session or pending.sent):
                return False
            self._pending_ve_release = None
            return True

    def _invoke_release_callback(self, callback, status, diagnostics):
        if callback is None:
            return
        try:
            callback(status, tuple(diagnostics))
        except Exception:
            self._logger.exception("VE release callback failed")

    def _start_capture(self, session):
        cancelled_before_send = False
        with self._lock:
            if session._sent or session._terminal:
                return
            if session.cancel_requested:
                self._clear_unsent_preparing_release(session)
                session._deadline = None
                session._child_released = True
                cancelled_before_send = True
            else:
                # Set ownership before enqueueing: the sender may dispatch immediately,
                # and a send failure cannot prove that the worker never opened the file.
                session._child_released = False
                session._sent = True
                session.state = "starting"
                now = time.monotonic()
                session._deadline = now + self._start_timeout
                if session.request.device.get("backend") == VE_BACKEND:
                    session._capture_requested_at = now
                self._command("start", session, session.request)
        if cancelled_before_send:
            self._cancelled(session)

    def _request_cancel(self, session):
        if session._terminal:
            return
        session.cancel_requested = True
        if self._clear_unsent_preparing_release(session):
            session._deadline = None
            session._child_released = True
            self._cancelled(session)
            return
        if session._slot_released_at is not None and session.descriptor is None:
            # Child finalizers have no cancellation command. Preserve the
            # request-scoped intent and apply it when its terminal arrives.
            return
        session._deadline = time.monotonic() + self._cancel_timeout
        if session.reader is not None:
            session.reader.cancel()
        if session._child_released:
            self._cancelled(session)
        elif session._sent:
            self._command("cancel", session)
        else:
            session._child_released = True
            self._cancelled(session)

    def _cancelled(self, session):
        if not session._terminal:
            session._terminal = True
            session.state = "cancelled"
            descriptor = session.descriptor or RecordingCancelled(session.request.request_id,
                                                                   session.request.path, 0, 0)
            self._notify(session, "cancelled", descriptor)
        self._release(session)

    def _fail(self, session, stage, message, failure=None):
        if not session._terminal:
            session._terminal = True
            session.state = "failed"
            session.failure = failure or RecordingFailure(session.request.request_id, stage,
                                                           session.request.path, message)
            if session.reader is not None:
                session.reader.cancel()
            session._deadline = time.monotonic() + self._cancel_timeout
            self._notify(session, "failed", session.failure)
        self._release(session)

    @staticmethod
    def _has_trusted_child_release(session):
        descriptor = None if session is None else session.descriptor
        return bool(session is not None and session._trusted_terminal
                    and descriptor is not None and descriptor.handles_released is True)

    def _retire_terminal_fault(self, worker, session, message, *, stage="protocol",
                               failure=None):
        """Retire a terminal contradiction without undoing prior trusted ownership."""
        trusted_release = self._has_trusted_child_release(session)
        self._retire_generation(worker, stage, message, session, failure)
        if session is not None and not trusted_release:
            session._child_released = False

    def _event(self, worker, event):
        if worker is not self._worker or worker.retiring:
            return
        if not isinstance(event, RecordingEvent):
            pending = self._pending_ve_prewarm
            if pending is not None and pending.generation == worker.generation:
                if pending.phase == "releasing":
                    self._retire_prerequisite_release(
                        worker, "worker sent a non-event payload during VE release")
                else:
                    self._retire_ve_prewarm(
                        pending, "protocol", "worker sent a non-event payload",
                        retryable=False)
            else:
                self._retire_generation(worker, "protocol", "worker sent a non-event payload")
            return
        # Pickle can bypass frozen-dataclass construction. Re-run every typed
        # validation boundary before a current generation can mutate admission.
        payload = getattr(event, "payload", None)
        lifecycle_event = (getattr(event, "kind", None) in (
            "capture_slot_released", "ve_released", "ve_release_failed", "worker_fatal",
            VE_PREWARM_STARTED, VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING,
            VE_PREWARM_TERMINAL)
            or isinstance(payload, (CaptureSlotReleased, VeReleaseOutcome, WorkerFatal,
                                    VePrewarmRequest, VePrewarmStarted,
                                    VePrewarmProgress, VePrewarmResult)))
        terminal_event = (getattr(event, "kind", None) in ("completed", "failed", "cancelled")
                          or isinstance(payload, (RecordingResult, RecordingFailure,
                                                  RecordingCancelled)))
        protocol_boundary_event = lifecycle_event or terminal_event
        prewarm_boundary_event = (getattr(event, "kind", None) in (
            VE_PREWARM_STARTED, VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING,
            VE_PREWARM_TERMINAL)
            or isinstance(payload, (VePrewarmRequest, VePrewarmStarted,
                                    VePrewarmProgress, VePrewarmResult)))
        if any(type(getattr(event, name, None)) is not expected for name, expected in (
                ("version", int), ("generation", int), ("request_id", str), ("kind", str))):
            if protocol_boundary_event:
                request_id = getattr(event, "request_id", None)
                cause = self._session(request_id) if type(request_id) is str else None
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        "worker event scalar envelope is invalid", retryable=False)
                elif terminal_event:
                    self._retire_terminal_fault(
                        worker, cause, "worker event scalar envelope is invalid")
                else:
                    self._retire_generation(worker, "protocol",
                                            "worker event scalar envelope is invalid", cause)
            return
        if event.generation != worker.generation:
            session = self._session(event.request_id)
            if session is not None and session.request.device.get("backend") != VE_BACKEND:
                return  # Preserve legacy soundcard stale-event isolation.
        try:
            RecordingEvent.__post_init__(event)
            payload = event.payload
            if isinstance(payload, (CaptureSlotReleased, VeReleaseOutcome, WorkerFatal,
                                    VePrewarmRequest, VePrewarmStarted,
                                    VePrewarmProgress, VePrewarmResult)):
                payload.__post_init__()
            if event.kind == "progress":
                RecordingProgress.__post_init__(payload)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            if protocol_boundary_event:
                cause = self._session(event.request_id)
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        f"invalid worker event: {exc}", retryable=False)
                elif terminal_event:
                    self._retire_terminal_fault(worker, cause, f"invalid worker event: {exc}")
                else:
                    self._retire_generation(worker, "protocol",
                                            f"invalid worker event: {exc}", cause)
            return
        if event.generation != worker.generation or event.version != 1:
            if protocol_boundary_event and event.generation > worker.generation:
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        "worker event generation is from the future", retryable=False)
                else:
                    self._retire_generation(worker, "protocol",
                                            "worker event generation is from the future")
            return
        capture = self._capture_session
        pending_prewarm = self._pending_ve_prewarm
        if event.kind in (VE_PREWARM_STARTED, VE_PREWARM_PROGRESS,
                          VE_PREWARM_DETACHING, VE_PREWARM_TERMINAL):
            if self._closing:
                return
            if (pending_prewarm is None
                    and event.request_id in self._terminal_ve_prewarm_ids):
                return
            if (pending_prewarm is None
                    or pending_prewarm.generation != worker.generation
                    or event.request_id != pending_prewarm.base_request.warmup_id):
                if pending_prewarm is not None:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "unexpected VE prewarm event identity", retryable=False)
                else:
                    self._retire_generation(worker, "protocol",
                                            "unexpected VE prewarm event identity")
                return
            current = pending_prewarm.current_request
            if event.kind == VE_PREWARM_STARTED:
                started = event.payload
                now = self._clock()
                if (pending_prewarm.phase != "starting" or current is None
                        or started.warmup_id != current.warmup_id
                        or started.generation != worker.generation
                        or started.attempt != pending_prewarm.attempt
                        or started.signature != pending_prewarm.base_request.signature
                        or pending_prewarm.start_sent_at is None
                        or not pending_prewarm.start_sent_at <= started.started_at <= now):
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm started identity mismatch", retryable=False)
                    return
                pending_prewarm.start_deadline = None
                pending_prewarm.phase = "capturing"
                pending_prewarm.capture_deadline = VeCaptureDeadline(
                    current.sample_rate, current.frames_per_channel,
                    started.started_at, clock=self._clock)
                return
            if event.kind in (VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING):
                progress = event.payload
                deadline = pending_prewarm.capture_deadline
                now = self._clock()
                valid_identity = (
                    current is not None
                    and progress.warmup_id == current.warmup_id
                    and progress.generation == worker.generation
                    and progress.attempt == pending_prewarm.attempt
                    and progress.signature == current.signature
                    and pending_prewarm.start_sent_at is not None
                    and pending_prewarm.start_sent_at <= progress.started_at
                    and progress.started_at <= progress.observed_at <= now)
                valid_phase = (pending_prewarm.phase == "capturing")
                if not valid_identity or not valid_phase or deadline is None:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress identity or ordering mismatch",
                        retryable=False)
                    return
                previous = deadline.snapshot()
                if previous.started_at != progress.started_at:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm native start timestamp changed",
                        retryable=False)
                    return
                if progress.observed_at < previous.last_frame_at:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress observation time regressed",
                        retryable=False)
                    return
                if (event.kind == VE_PREWARM_DETACHING
                        and not deadline.complete):
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm detach preceded authenticated target progress",
                        retryable=False)
                    return
                if not previous.frames <= progress.frames_per_channel:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress regressed", retryable=False)
                    return
                if (not deadline.complete
                        and (progress.observed_at - previous.last_frame_at >= 5.0
                             or progress.observed_at >= deadline.capture_deadline)):
                    message = (
                        "VE capture made no progress for 5 seconds"
                        if progress.observed_at - previous.last_frame_at >= 5.0
                        else "VE capture total deadline exceeded before target frames")
                    self._retire_ve_prewarm(
                        pending_prewarm, "capture_timeout", message,
                        retryable=True)
                    return
                deadline.observe(
                    progress.frames_per_channel, at=progress.observed_at)
                if event.kind == VE_PREWARM_DETACHING:
                    pending_prewarm.phase = "detaching"
                    pending_prewarm.detach_deadline = now + .5
                return
            result = event.payload
            if (current is None or result.warmup_id != current.warmup_id
                    or result.generation != worker.generation
                    or result.attempt != pending_prewarm.attempt
                    or result.signature != current.signature
                    or pending_prewarm.phase not in ("starting", "capturing", "detaching")
                    or (result.success and pending_prewarm.phase != "detaching")):
                self._retire_ve_prewarm(
                    pending_prewarm, "protocol",
                    "VE prewarm terminal identity mismatch", retryable=False)
                return
            pending_prewarm.start_deadline = None
            pending_prewarm.capture_deadline = None
            pending_prewarm.detach_deadline = None
            pending_prewarm.results += (result,)
            if result.success:
                self._retained_ve_signature = result.signature
                self._retained_lifecycle_counts = result.lifecycle_counts
                self._expected_next_lifecycle_counts = None
                self._finish_ve_prewarm(
                    pending_prewarm, success=True, result=result,
                    ownership_safe=True)
            else:
                if pending_prewarm.first_fault is None:
                    pending_prewarm.first_fault = result
                retryable = result.stage not in (
                    "validation", "protocol", "cancelled", "shutdown")
                self._retire_ve_prewarm(
                    pending_prewarm, result.stage, result.detail,
                    result=result, retryable=retryable)
            return
        if event.kind == "worker_fatal":
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation):
                if pending_prewarm.phase == "releasing":
                    self._retire_prerequisite_release(
                        worker, event.payload.message)
                else:
                    self._retire_ve_prewarm(
                        pending_prewarm, event.payload.stage,
                        event.payload.message, retryable=True)
                return
            self._retire_generation(worker, event.payload.stage, event.payload.message)
            return
        if event.kind in ("ve_released", "ve_release_failed"):
            pending = self._pending_ve_release
            if pending is None or pending.generation != worker.generation or not pending.sent:
                self._retire_generation(worker, "protocol", "unexpected VE release terminal")
                return
            outcome = event.payload
            if event.kind == "ve_release_failed":
                preparing = pending.preparing
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    detail = "; ".join(outcome.diagnostics) or "VE release failed"
                    self._retire_prerequisite_release(
                        worker, detail, diagnostics=outcome.diagnostics)
                    return
                self._retire_generation(worker, "release_ve",
                                        "; ".join(outcome.diagnostics) or "VE release failed",
                                        preparing, pending_diagnostics=outcome.diagnostics)
                return
            if outcome.released_signature != self._retained_ve_signature:
                if (isinstance(pending.preparing, _PendingVePrewarm)
                        and pending.preparing is self._pending_ve_prewarm):
                    self._retire_prerequisite_release(
                        worker, "VE release signature contradiction")
                    return
                self._retire_generation(worker, "protocol", "VE release signature contradiction")
                return
            preparing = pending.preparing
            self._retained_ve_signature = None
            counts = self._expected_next_lifecycle_counts or self._retained_lifecycle_counts
            if counts is not None:
                self._expected_next_lifecycle_counts = replace(
                    counts, sdk_open=counts.sdk_open + 1,
                    task_create=counts.task_create + 1,
                    task_start=counts.task_start + 1,
                    task_stop=counts.task_stop + 1,
                    task_clear=counts.task_clear + 1,
                    sdk_close=counts.sdk_close + 1)
            self._finish_ve_release("released", outcome.diagnostics)
            if (isinstance(preparing, _PendingVePrewarm)
                    and preparing is self._pending_ve_prewarm
                    and not preparing.completed):
                preparing.prerequisite_release = False
                self._dispatch_ve_prewarm_attempt(preparing)
                return
            if (preparing is not None and not preparing._terminal
                    and not preparing.cancel_requested
                    and preparing is self._capture_session
                    and self._session(preparing.request.request_id) is preparing):
                self._start_capture(preparing)
            return
        if (event.kind == "failed" and not event.request_id and not worker.ready
                and (capture is not None or pending_prewarm is not None)):
            if pending_prewarm is not None:
                self._retire_ve_prewarm(
                    pending_prewarm, event.payload.stage,
                    event.payload.message, retryable=True)
            else:
                self._retire_generation(worker, event.payload.stage, event.payload.message, capture)
            return
        if event.kind == "ready":
            if worker.ready:
                self._retire_generation(worker, "protocol", "duplicate worker ready event")
                return
            worker.ready = True
            worker.deadline = None
            pending = self._pending_ve_release
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation
                    and pending_prewarm.phase == "waiting_ready"):
                self._send_ve_prewarm(pending_prewarm)
            elif capture is not None and capture.cancel_requested and not capture._sent:
                self._clear_unsent_preparing_release(capture)
                capture._child_released = True
                self._cancelled(capture)
            elif pending is not None and pending.preparing is capture:
                if capture.cancel_requested and self._clear_unsent_preparing_release(capture):
                    capture._child_released = True
                    self._cancelled(capture)
                else:
                    self._send_ve_release(pending)
            elif pending is not None and not pending.sent:
                # An explicit release can be queued while worker readiness is
                # pending. Never leave it without either a command or deadline.
                self._send_ve_release(pending)
            elif capture is not None:
                self._start_capture(capture)
            return
        session = self._session(event.request_id)
        if session is None or session.generation != worker.generation:
            if event.kind == "capture_slot_released":
                self._retire_generation(worker, "protocol", "capture slot event has no matching session")
            return
        is_ve = session.request.device.get("backend") == VE_BACKEND
        if event.kind == "progress":
            deadline = session._capture_deadline
            if not is_ve or deadline is None or session._terminal or session.cancel_requested:
                return
            progress = event.payload
            previous = deadline.snapshot()
            if (not previous.frames <= progress.frames <= session.request.target_samples
                    or not previous.last_frame_at <= progress.last_frame_at <= time.monotonic()):
                return
            # Equal counts are deliberately a no-op, even with a newer time.
            deadline.observe(progress.frames, at=progress.last_frame_at)
            if (progress.frames == session.request.target_samples
                    and previous.frames < session.request.target_samples
                    and session._target_reached_at is None):
                session._target_reached_at = progress.last_frame_at
                session._slot_release_deadline = progress.last_frame_at + .5
        elif event.kind == "capture_slot_released":
            slot = event.payload
            counts = slot.lifecycle_counts
            if self._expected_next_lifecycle_counts is not None:
                expected_counts = self._expected_next_lifecycle_counts
            elif self._retained_lifecycle_counts is None:
                expected_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
            else:
                expected_counts = self._retained_lifecycle_counts
            valid = (is_ve and session is self._capture_session
                     and session._slot_released_at is None
                     and session._target_reached_at is not None
                     and slot.target_reached_at == session._target_reached_at
                     and slot.raw_frames == session.request.target_samples
                     and slot.adapter_released is True and slot.writer_released is True
                     and counts == expected_counts
                     and session._slot_release_deadline is not None)
            if not valid:
                self._retire_generation(worker, "protocol", "invalid or out-of-order capture slot release", session)
                return
            late = False
            with self._lock:
                # Admission cannot observe this tentative transition while the
                # lock is held. The timestamp after it is the authoritative t1.
                self._capture_session = None
                admitted_at = time.monotonic()
                if admitted_at > session._slot_release_deadline:
                    self._capture_session = session
                    late = True
                else:
                    session._slot_release_deadline = None
                    session._slot_lifecycle_counts = counts
                    self._retained_lifecycle_counts = counts
                    self._expected_next_lifecycle_counts = None
                    self._retained_ve_signature = self._request_signature(session.request)
                    session._slot_released_at = admitted_at
                    session._admission_reason = (
                        "CAPACITY_BACKPRESSURE"
                        if len(self._sessions) >= self._pipeline_capacity else None)
            if late:
                self._retire_generation(worker, "capture_release_timeout",
                                        "capture slot release exceeded target + 0.5 seconds", session)
        elif event.kind == "preview":
            snapshot = event.payload
            if snapshot.sequence == session._preview_pending:
                return  # Duplicate of a still-owned snapshot is not consumption.
            if (session._terminal or session.cancel_requested or session._preview_disabled or session.state == "delivering"
                    or snapshot.generation != worker.generation
                    or snapshot.channels != session.request.channels
                    or snapshot.sequence <= session._last_sequence
                    or snapshot.sample_stop < session._last_sample_stop):
                self._command("preview_ack", session, snapshot.sequence)
                return
            if session._preview_pending is not None:
                return
            session._last_sequence, session._last_sample_stop = snapshot.sequence, snapshot.sample_stop
            session._preview_pending = snapshot.sequence
            self._notify(session, "preview", snapshot)
            if session.callbacks.preview is None:
                self.release_preview(session.request.request_id, snapshot.sequence)
        elif event.kind == "finalizing" and not session._terminal:
            if session.state == "recording":
                session.state = "finalizing"
        elif event.kind == "started" and not session._terminal:
            if session.state == "starting":
                if is_ve:
                    if (event.payload is None or session._capture_requested_at is None
                            or not session._capture_requested_at <= event.payload <= time.monotonic()):
                        return
                    session._capture_deadline = VeCaptureDeadline(
                        session.request.sample_rate, session.request.target_samples, event.payload)
                session.state = "recording"
                if is_ve:
                    self._retained_ve_signature = self._request_signature(session.request)
                    if (self._retained_lifecycle_counts is None
                            and self._expected_next_lifecycle_counts is None):
                        self._expected_next_lifecycle_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
                if not session.cancel_requested:
                    session._deadline = None
                self._notify(session, "started")
        elif event.kind in ("completed", "failed", "cancelled"):
            if session.descriptor is not None:
                if event.payload != session.descriptor:
                    self._retire_terminal_fault(worker, session, "contradictory worker terminal")
                return
            descriptor = event.payload
            invalid = self._terminal_validation_error(session, event.kind, descriptor)
            if invalid is not None:
                self._retire_terminal_fault(worker, session, invalid)
                return
            if (descriptor.handles_released and is_ve and session._slot_released_at is None
                    and (session._target_reached_at is not None or event.kind == "completed")):
                self._retire_terminal_fault(
                    worker, session, "worker terminal preceded capture slot release")
                return
            cleanup_keys = {self._path_key(path) for path in descriptor.cleanup_paths}
            with self._lock:
                if any(self._leases.get(key) not in (None, session) for key in cleanup_keys):
                    conflict = True
                else:
                    conflict = False
                    for key in cleanup_keys:
                        self._leases[key] = session
                    session._lease_keys.update(cleanup_keys)
            if conflict:
                self._retire_terminal_fault(
                    worker, session, "worker cleanup paths conflict with another session lease")
                return
            session._cleanup_paths = descriptor.cleanup_paths
            if not descriptor.handles_released:
                failure = descriptor if event.kind == "failed" else None
                self._retire_terminal_fault(
                    worker, session, "worker retained file/device handles",
                    stage="close", failure=failure)
                return
            # Only a fully validated, handle-released descriptor crosses the
            # trust boundary. From here reader/UI flow may safely continue even
            # if another request later retires this generation.
            session.descriptor = descriptor
            session._child_released = True
            if is_ve and session._slot_released_at is None and self._capture_session is session:
                self._capture_session = None
            if event.kind == "failed":
                session._trusted_terminal = is_ve
                self._fail(session, descriptor.stage, descriptor.message, descriptor)
            elif session.cancel_requested or event.kind == "cancelled":
                session._trusted_terminal = is_ve
                self._cancelled(session)
            elif event.kind == "completed" and not session._terminal:
                session._trusted_terminal = is_ve
                session.state = "delivering"
                session._deadline = None
                try:
                    session.reader = self._reader_factory(descriptor,
                        lambda outcome: self._inbox.put(("read", session, outcome)))
                    session._reader_released = False
                    self._start_thread(session.reader.thread, start=session.reader.start)
                except Exception as exc:
                    reader_thread = (None if session.reader is None
                                     else getattr(session.reader, "thread", None))
                    no_reader_ownership = (session.reader is None
                                           or (reader_thread is not None
                                               and reader_thread.ident is None))
                    if no_reader_ownership:
                        # Reader construction allocates no file handles; only its
                        # thread may open the result. A failed pre-start attempt
                        # never acquired read ownership. Otherwise wait for _read.
                        session._reader_released = True
                        session.reader = None
                    self._diagnose(str(exc))
                    if is_ve and no_reader_ownership:
                        self._fail(session, "service", str(exc))
                    else:
                        self._retire_generation(worker, "service", str(exc), session)
                        self._fail(session, "service", str(exc))
            else:
                self._release(session)

    @staticmethod
    def _terminal_validation_error(session, kind, descriptor):
        """Return a protocol error before any terminal ownership is trusted."""
        expected_type = {"completed": RecordingResult,
                         "failed": RecordingFailure,
                         "cancelled": RecordingCancelled}[kind]
        request = session.request
        if type(descriptor) is not expected_type:
            return "worker terminal descriptor type mismatch"
        if (type(descriptor.request_id) is not str
                or descriptor.request_id != request.request_id):
            return "worker terminal request differs from active request"
        if type(descriptor.path) is not str or descriptor.path != request.path:
            return "worker result path differs from leased path"
        if type(descriptor.handles_released) is not bool:
            return "worker terminal handle ownership is invalid"
        if (type(descriptor.cleanup_paths) is not tuple
                or any(type(path) is not str or not path for path in descriptor.cleanup_paths)
                or len(set(descriptor.cleanup_paths)) != len(descriptor.cleanup_paths)):
            return "worker terminal cleanup paths are invalid"

        def valid_count(value):
            return type(value) is int and 0 <= value <= request.target_samples

        if kind == "completed":
            trim = (request.trim_samples
                    if request.purpose == "main" and request.trim_samples < request.target_samples
                    else 0)
            if (type(descriptor.purpose) is not str or descriptor.purpose != request.purpose
                    or type(descriptor.sample_rate) is not int
                    or descriptor.sample_rate != request.sample_rate
                    or type(descriptor.channels) is not tuple
                    or descriptor.channels != request.channels
                    or not valid_count(descriptor.raw_frames)
                    or not valid_count(descriptor.final_frames)
                    or descriptor.raw_frames != request.target_samples
                    or descriptor.final_frames != request.target_samples - trim
                    or type(descriptor.metadata_appended) is not bool
                    or type(descriptor.warnings) is not tuple
                    or any(type(warning) is not str for warning in descriptor.warnings)):
                return "worker final count/channel/rate contract mismatch"
        elif kind == "failed":
            if (type(descriptor.stage) is not str or not descriptor.stage
                    or type(descriptor.message) is not str or not descriptor.message
                    or not valid_count(descriptor.raw_frames)
                    or not valid_count(descriptor.written_frames)
                    or descriptor.written_frames > descriptor.raw_frames
                    or (session._slot_released_at is not None
                        and descriptor.raw_frames != request.target_samples)):
                return "worker failure descriptor violates request contract"
        elif (not valid_count(descriptor.raw_frames)
                or not valid_count(descriptor.final_frames)
                or descriptor.final_frames > descriptor.raw_frames
                or (session._slot_released_at is not None
                    and descriptor.raw_frames != request.target_samples)):
            return "worker cancellation descriptor violates request contract"
        return None

    def _read(self, session, outcome):
        session._reader_released = outcome.handles_released
        if outcome.error:
            self._fail(session, "read", outcome.error)
        elif not session._terminal and not session.cancel_requested and not session._reject_requested:
            session.audio = outcome.audio
            self._notify(session, "result_ready", outcome.audio)
        self._release(session)

    def _accept(self, session):
        if (self._session(session.request.request_id) is not session
                or session._terminal or session.cancel_requested
                or session._reject_requested):
            return
        if session.request.device.get("backend") != VE_BACKEND:
            worker = self._worker
            if worker is None or worker.retiring or not worker.process.is_alive():
                self._fail(session, "worker", "recording worker died before result acceptance")
                return
        if session.audio is None or not session._reader_released:
            session._accept_requested = False
            return
        session._terminal = True
        session.state = "completed"
        session._deadline = None
        self._notify(session, "accepted", session.audio)
        self._release(session)

    def _release(self, session):
        if (not session._terminal or not session._child_released or not session._reader_released
                or session.released.is_set() or session._cleanup_failed):
            return
        worker = self._worker
        if worker is not None and worker.retiring and worker.generation == session.generation:
            return  # Confirm OS death before releasing a forcibly retired writer.
        if worker is not None and worker.generation == session.generation and session.descriptor is not None:
            self._command("result_ack", session, "accepted" if session.state == "completed" else "rejected")
            session.acknowledged = True
        try:
            for path in session._cleanup_paths:
                if self._leases.get(self._path_key(path)) is not session:
                    raise OSError(f"cleanup path no longer owned by this session: {path}")
                if os.path.exists(path):
                    os.unlink(path)
            if session._temporary_dir is not None:
                shutil.rmtree(session._temporary_dir)
            elif session.state == "failed" and session._sent and os.path.exists(session.request.path):
                os.unlink(session.request.path)
        except OSError as exc:
            session._cleanup_failed = True
            session.release_error = str(exc)
            self._diagnose(f"{session.request.request_id}: path remains leased after cleanup failure: {exc}")
        while not session._cleanup_failed:
            with self._lock:
                actions = list(session._release_actions.values())
                session._release_actions.clear()
                if not actions:
                    for key in session._lease_keys:
                        if self._leases.get(key) is session:
                            del self._leases[key]
                    self._sessions.pop(session.request.request_id, None)
                    if self._capture_session is session:
                        self._capture_session = None
                    break
            try:
                for path, cleanup in actions:
                    cleanup(path)
            except Exception as exc:
                # User-supplied file/database cleanup is a real external boundary.
                # Retain this lease on any failure, diagnose, never permit reuse.
                session._cleanup_failed = True
                session.release_error = str(exc)
                self._diagnose(f"{session.request.request_id}: leased cleanup failed: {exc}")
        if session._cleanup_failed:
            with self._lock:
                self._sessions.pop(session.request.request_id, None)
                if self._capture_session is session:
                    self._capture_session = None
            self._notify(session, "release_failed", session.release_error)
        if not session._cleanup_failed:
            session.released.set()
            self._notify(session, "released")

    def _retire_generation(self, worker, stage="worker",
                           message="recording worker retired", cause_session=None,
                           cause_failure=None, pending_diagnostics=None):
        pending_prewarm = self._pending_ve_prewarm
        if (pending_prewarm is not None
                and pending_prewarm.generation == worker.generation
                and pending_prewarm.phase == "releasing"):
            self._transition_prerequisite_release_failure(
                pending_prewarm, message,
                diagnostics=() if pending_diagnostics is None else pending_diagnostics)
        if worker.retiring:
            return
        with self._lock:
            worker.retiring = True
            self._ownership_uncertain = True
            worker.kill_deadline = self._clock() + self._terminate_timeout
            pending = self._pending_ve_release
            self._pending_ve_release = None
            sessions = [session for session in self._sessions.values()
                        if session.generation == worker.generation]
        if pending is not None and pending.callback is not None:
            diagnostics = ((message,) if pending_diagnostics is None
                           else tuple(pending_diagnostics))
            self._invoke_release_callback(pending.callback, "failed", diagnostics)
        for session in sessions:
            if session._trusted_terminal:
                continue
            if session._terminal:
                continue
            if session is cause_session:
                failure_stage, failure_message = stage, message
            elif session._slot_released_at is not None:
                failure_stage = "worker_retired_during_finalization"
                failure_message = "worker retired before finalization completed"
            elif session.state == "preparing":
                failure_stage, failure_message = "release_ve", message
            else:
                failure_stage, failure_message = stage, message
            self._fail(session, failure_stage, failure_message,
                       cause_failure if session is cause_session else None)
        if worker.process.is_alive():
            worker.process.terminate()

    def _retire(self, worker):
        self._retire_generation(worker)

    def _dead(self, worker):
        pending_prewarm = self._pending_ve_prewarm
        sessions = [session for session in self._sessions.values()
                    if session.generation == worker.generation]
        if (not worker.retiring and pending_prewarm is not None
                and pending_prewarm.generation == worker.generation):
            if pending_prewarm.phase == "releasing":
                self._transition_prerequisite_release_failure(
                    pending_prewarm,
                    "VE resource owner exited during prerequisite release")
            else:
                self._remember_prewarm_fault(
                    pending_prewarm, "worker",
                    "recording worker exited during VE prewarm")
                pending_prewarm.phase = (
                    "retiring_retry" if pending_prewarm.attempt == 1
                    else "retiring_final")
        if not worker.retiring:
            self._retire_generation(worker, "worker", "recording worker exited before result acceptance")
        with self._lock:
            pending = self._pending_ve_release
            self._pending_ve_release = None
        if pending is not None and pending.callback is not None:
            self._invoke_release_callback(
                pending.callback, "failed", ("recording worker retirement completed before VE release",))
        for session in sessions:
            if session.descriptor is None or not session.descriptor.handles_released:
                session._child_released = True
        worker.stop.set()
        worker.process.join(timeout=0)
        for thread in worker.threads:
            thread.join(.2)
        worker.control.close()
        worker.preview.close()
        worker.process.close()
        self._worker = None
        self._ownership_uncertain = False
        self._retained_ve_signature = None
        self._retained_lifecycle_counts = None
        self._expected_next_lifecycle_counts = None
        for session in sessions:
            self._release(session)
            if (session.request.device.get("backend") != VE_BACKEND
                    and session._terminal and not session._reader_released):
                # Confirmed PID death isolates the stalled reader to its leased path.
                with self._lock:
                    if self._capture_session is session:
                        self._capture_session = None
        if (pending_prewarm is self._pending_ve_prewarm
                and pending_prewarm is not None and not pending_prewarm.completed
                and pending_prewarm.generation == worker.generation):
            if pending_prewarm.phase == "retiring_retry":
                pending_prewarm.phase = "retry_wait"
                pending_prewarm.retry_deadline = self._clock() + self._retry_delay
                pending_prewarm.generation = None
            elif pending_prewarm.phase in ("retiring_final", "retiring_release"):
                self._finish_ve_prewarm(
                    pending_prewarm, success=False, ownership_safe=True)

    def _tick(self):
        retained = []
        for thread in self.threads:
            if thread is self._supervisor or thread.is_alive():
                retained.append(thread)
            else:
                thread.join(timeout=0)
        self.threads = retained
        now = self._clock()
        worker = self._worker
        if worker is not None:
            if worker.process.pid is not None and not worker.process.is_alive():
                self._dead(worker)
                worker = None
            elif worker.retiring:
                if now >= worker.kill_deadline and not worker.kill_reported:
                    worker.process.kill()
                    worker.kill_reported = True
                    self._diagnose("Worker exit not yet confirmed; restart remains disabled")
                    pending = self._pending_ve_prewarm
                    if (pending is not None and not pending.completed
                            and pending.generation == worker.generation
                            and pending.phase in (
                                "retiring_retry", "retiring_final", "retiring_release")):
                        self._finish_ve_prewarm(
                            pending, success=False, ownership_safe=False)
            elif worker.deadline is not None and now >= worker.deadline:
                pending = self._pending_ve_prewarm
                if (pending is not None and pending.generation == worker.generation):
                    self._retire_ve_prewarm(
                        pending, "ready_timeout",
                        "recording worker ready deadline exceeded", retryable=True)
                else:
                    self._retire_generation(worker, "ready_timeout",
                                            "recording worker ready deadline exceeded",
                                            self._capture_session)
        pending_prewarm = self._pending_ve_prewarm
        if pending_prewarm is not None and not pending_prewarm.completed:
            if (pending_prewarm.phase == "starting"
                    and pending_prewarm.start_deadline is not None
                    and now >= pending_prewarm.start_deadline):
                pending_prewarm.start_deadline = None
                self._retire_ve_prewarm(
                    pending_prewarm, "start_timeout",
                    "VE prewarm start deadline exceeded", retryable=True)
            elif (pending_prewarm.phase == "capturing"
                    and pending_prewarm.capture_deadline is not None):
                try:
                    pending_prewarm.capture_deadline.check(now=now)
                except TimeoutError as exc:
                    pending_prewarm.capture_deadline = None
                    self._retire_ve_prewarm(
                        pending_prewarm, "capture_timeout", str(exc), retryable=True)
            elif (pending_prewarm.phase == "detaching"
                    and pending_prewarm.detach_deadline is not None
                    and now >= pending_prewarm.detach_deadline):
                pending_prewarm.detach_deadline = None
                self._retire_ve_prewarm(
                    pending_prewarm, "detach",
                    "VE prewarm detach deadline exceeded", retryable=True)
            elif (pending_prewarm.phase == "retry_wait"
                    and pending_prewarm.retry_deadline is not None
                    and now >= pending_prewarm.retry_deadline):
                pending_prewarm.retry_deadline = None
                pending_prewarm.attempt = 2
                self._dispatch_ve_prewarm_attempt(pending_prewarm)
        session = self._capture_session
        if (session is not None and session._capture_deadline is not None
                and not session._terminal and not session.cancel_requested):
            try:
                session._capture_deadline.check(now=now)
            except TimeoutError as exc:
                progress = session._capture_deadline.snapshot()
                self._fail(session, "capture_timeout", str(exc), RecordingFailure(
                    session.request.request_id, "capture_timeout", session.request.path,
                    str(exc), raw_frames=progress.frames, handles_released=session._child_released))
                # _fail marks terminal, so _request_cancel would return without
                # sending anything. Preserve the first semantic failure and its
                # existing cancel deadline, but actually ask the child to stop.
                if session._sent and not session._child_released:
                    self._command("cancel", session)
        if (session is not None and session._slot_release_deadline is not None
                and now >= session._slot_release_deadline and not session._terminal):
            self._retire_generation(worker, "capture_release_timeout",
                                    "capture slot release exceeded target + 0.5 seconds", session)
        pending = self._pending_ve_release
        if pending is not None and pending.deadline is not None and now >= pending.deadline:
            preparing = pending.preparing
            if worker is not None:
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    self._retire_prerequisite_release(
                        worker, "VE release deadline exceeded",
                        diagnostics=("VE release deadline exceeded",))
                else:
                    self._retire_generation(
                        worker, "release_ve", "VE release deadline exceeded", preparing,
                        pending_diagnostics=("VE release deadline exceeded",))
            else:
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    self._pending_ve_release = None
                    self._retire_prerequisite_release(
                        None, "VE release deadline exceeded",
                        diagnostics=("VE release deadline exceeded",))
                else:
                    self._finish_ve_release("failed", ("VE release deadline exceeded",))
        for timed in list(self._sessions.values()):
            if timed._deadline is None or now < timed._deadline:
                continue
            timed._deadline = None
            stage = "cancel_timeout" if timed.cancel_requested else "start_timeout"
            if not timed._terminal:
                self._fail(timed, stage, f"recording {stage} deadline exceeded")
            else:
                self._diagnose(f"{timed.request.request_id}: reader/path release deadline exceeded; path remains leased")
            if worker is not None:
                self._retire_generation(worker, stage, f"recording {stage} deadline exceeded", timed)
        if (self._shutdown_deadline is not None and now >= self._shutdown_deadline
                and worker is not None and not worker.retiring):
            self._retire(worker)
        if (self._closing and self._worker is None and self._shutdown_deadline is not None
                and now >= self._shutdown_deadline and not self._shutdown_reported):
            if self._leases:
                self._diagnose("Recording stopped; pending reader paths remain leased during shutdown")
            self._report_shutdown()

    def _report_shutdown(self):
        with self._lock:
            self._shutdown_reported = True
            callbacks, self._shutdown_callbacks = self._shutdown_callbacks, []
        for callback in callbacks:
            self._invoke_shutdown(callback)

    def _invoke_shutdown(self, callback):
        try:
            callback()
        except Exception:
            self._logger.exception("Recording shutdown callback failed")

    def _dispatch(self, item):
        kind = item[0]
        if kind == "start":
            self._begin(item[1])
        elif kind == "event":
            self._event(item[1], item[2])
        elif kind == "read":
            self._read(item[1], item[2])
        elif kind == "cancel":
            self._request_cancel(item[1])
        elif kind == "accept":
            self._accept(item[1])
        elif kind == "reject":
            session = item[1]
            if not session._terminal:
                self._fail(session, "semantic", item[2])
                if not session._child_released:
                    self._command("cancel", session)
        elif kind == "preview_ack":
            session, sequence = item[1:]
            worker = self._worker
            current_generation = (worker is not None and not worker.retiring
                                  and session.generation == worker.generation)
            if not current_generation:
                if session._preview_pending == sequence:
                    session._preview_pending = None
                if session._preview_ack_requested == sequence:
                    session._preview_ack_requested = None
            elif (self._session(session.request.request_id) is session
                  and session._preview_pending == sequence):
                session._preview_pending = None
                session._preview_ack_requested = None
                self._command("preview_ack", session, sequence)
        elif kind == "broken":
            worker = item[1]
            if worker is self._worker and not worker.retiring:
                pending_prewarm = self._pending_ve_prewarm
                if (pending_prewarm is not None
                        and pending_prewarm.generation == worker.generation):
                    if pending_prewarm.phase == "releasing":
                        self._retire_prerequisite_release(
                            worker,
                            f"recording IPC closed during VE release: {item[2]}")
                    else:
                        self._retire_ve_prewarm(
                            pending_prewarm, "worker",
                            f"recording IPC closed: {item[2]}", retryable=True)
                else:
                    self._retire_generation(worker, "worker", f"recording IPC closed: {item[2]}",
                                            self._capture_session)
        elif kind == "release_ve":
            pending = self._pending_ve_release
            if pending is not None:
                self._send_ve_release(pending)
        elif kind == "release_callback":
            self._invoke_release_callback(item[1], item[2], item[3])
        elif kind == "prewarm_ve":
            self._begin_ve_prewarm(item[1])
        elif kind == "shutdown":
            self._shutdown_deadline = self._clock() + self._shutdown_timeout
            pending_prewarm = self._pending_ve_prewarm
            if pending_prewarm is not None:
                if (self._pending_ve_release is not None
                        and self._pending_ve_release.preparing is pending_prewarm):
                    self._pending_ve_release = None
                if (self._worker is not None and not self._worker.retiring
                        and pending_prewarm.generation == self._worker.generation
                        and pending_prewarm.current_request is not None):
                    self._worker.outgoing.put_nowait(RecordingEvent(
                        self._worker.generation,
                        pending_prewarm.current_request.warmup_id,
                        "cancel"))
                pending_prewarm.completed = True
                self._pending_ve_prewarm = None
            for session in list(self._sessions.values()):
                if not session._terminal:
                    self._request_cancel(session)
            self._command("shutdown")

    def _run(self):
        while True:
            try:
                item = self._inbox.get(timeout=.02)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    self._dispatch(item)
                self._tick()
            except Exception as exc:
                # Supervisor contract boundary for process creation, IPC queues,
                # filesystem setup and custom reader construction. Fail once and
                # retire the worker; keep observing leases rather than abandon them.
                self._handle_supervisor_exception(exc)
            if self._closing and self._worker is None and not self._leases:
                self._report_shutdown()
                self.closed.set()
                return

    def _handle_supervisor_exception(self, exc):
        self._logger.exception("Recording service operation failed", exc_info=exc)
        self._diagnose(str(exc))
        worker = self._worker
        pending_prewarm = self._pending_ve_prewarm
        if worker is not None:
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation):
                if pending_prewarm.phase == "releasing":
                    self._retire_prerequisite_release(worker, str(exc))
                else:
                    self._retire_ve_prewarm(
                        pending_prewarm, "service", str(exc), retryable=True)
            else:
                self._retire_generation(worker, "service", str(exc),
                                        self._capture_session)
        elif self._capture_session is not None:
            self._fail(self._capture_session, "service", str(exc))
        elif pending_prewarm is not None:
            if pending_prewarm.phase == "releasing":
                self._transition_prerequisite_release_failure(
                    pending_prewarm, str(exc))
                self._finish_ve_prewarm(
                    pending_prewarm, success=False, ownership_safe=True)
            else:
                self._remember_prewarm_fault(
                    pending_prewarm, "service", str(exc))
                if pending_prewarm.attempt == 1 and not self._closing:
                    pending_prewarm.phase = "retry_wait"
                    pending_prewarm.retry_deadline = self._clock() + self._retry_delay
                else:
                    self._finish_ve_prewarm(
                        pending_prewarm, success=False, ownership_safe=True)
