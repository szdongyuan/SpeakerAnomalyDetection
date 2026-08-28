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
    FrozenConfig, RecordingCancelled, RecordingEvent, RecordingFailure, RecordingRequest,
)
from base.recording_result_reader import ResultReader
from base.recording_worker import recording_worker


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
        # Reserving a path is not giving a child ownership. Only dispatching
        # start can let the worker open it; setup failures need no child release.
        self._child_released = True
        self._reader_released = True
        self._sent = False
        self._deadline = None
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


class RecordingService:
    def __init__(self, *, backend_factory=None, backend_options=None,
                 ready_timeout=10.0, start_timeout=10.0, cancel_timeout=5.0,
                 shutdown_timeout=5.0, terminate_timeout=2.0, preview_interval=.05,
                 reader_factory=ResultReader):
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
        self._reader_factory = reader_factory
        self._lock = threading.RLock()
        self._inbox = queue.Queue(maxsize=64)
        self._active = None
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
        return self._active is not None or (self._worker is not None and self._worker.retiring)

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
            if self.busy:
                raise RuntimeError("recording service is busy")
            if request.request_id in self._request_ids:
                raise ValueError("request_id must be unique for the lifetime of this service")
            session = RecordingSession(self, request, callbacks or RecordingCallbacks())
            session._lease_key = key
            if key is not None:
                self._leases[key] = session
                session._lease_keys.add(key)
            self._active = session  # Before even the async start command is queued.
            self._request_ids.add(request.request_id)
            self._inbox.put_nowait(("start", session))
            return session

    def _session(self, request_id):
        session = self._active
        return session if session is not None and session.request.request_id == request_id else None

    def cancel(self, request_id):
        with self._lock:
            session = self._session(request_id)
            if session is not None and not session._terminal and not session.cancel_requested:
                session.cancel_requested = True
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
                             time.monotonic() + self._ready_timeout)
            try:
                process.start()
            finally:
                if process.pid is not None:
                    # Also covers a custom start that launches and then raises.
                    # Preserve supervision before any IPC-thread startup failure.
                    self._worker = worker
                    if self._active is not None:
                        self._active.generation = worker.generation
                        self._active.worker_pid = process.pid
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
        if worker is not None and not worker.retiring:
            worker.outgoing.put_nowait(RecordingEvent(worker.generation,
                session.request.request_id if session else "", kind, payload))

    def _begin(self, session):
        if session.cancel_requested:
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
        if self._worker.ready:
            self._start_capture(session)

    def _start_capture(self, session):
        if session._sent or session._terminal:
            return
        # Set ownership before enqueueing: the sender may dispatch immediately,
        # and a send failure cannot prove that the worker never opened the file.
        session._child_released = False
        session._sent = True
        session._deadline = time.monotonic() + self._start_timeout
        self._command("start", session, session.request)
        if session.cancel_requested:
            self._request_cancel(session)

    def _request_cancel(self, session):
        if session._terminal:
            return
        session.cancel_requested = True
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

    def _event(self, worker, event):
        if (worker is not self._worker or worker.retiring or not isinstance(event, RecordingEvent)
                or event.generation != worker.generation or event.version != 1):
            return
        session = self._active
        if (event.kind == "failed" and not event.request_id and not worker.ready
                and session is not None):
            self._retire(worker)
            self._fail(session, event.payload.stage, event.payload.message)
            return
        if event.kind == "ready":
            worker.ready = True
            worker.deadline = None
            if session is not None:
                self._start_capture(session)
            return
        if session is None or event.request_id != session.request.request_id:
            return
        if event.kind == "preview":
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
                session.state = "recording"
                if not session.cancel_requested:
                    session._deadline = None
                self._notify(session, "started")
        elif event.kind in ("completed", "failed", "cancelled"):
            if session.descriptor is not None:
                return
            descriptor = event.payload
            if descriptor.path != session.request.path:
                self._fail(session, "protocol", "worker result path differs from leased path")
                self._retire(worker)
                return
            session.descriptor = descriptor
            session._child_released = descriptor.handles_released
            cleanup_keys = {self._path_key(path) for path in descriptor.cleanup_paths}
            with self._lock:
                if any(self._leases.get(key) not in (None, session) for key in cleanup_keys):
                    raise ValueError("worker cleanup paths conflict with another session lease")
                for key in cleanup_keys:
                    self._leases[key] = session
                session._lease_keys.update(cleanup_keys)
            session._cleanup_paths = descriptor.cleanup_paths
            if not descriptor.handles_released:
                self._fail(session, "close", "worker retained file/device handles",
                           descriptor if event.kind == "failed" else None)
                self._retire(worker)
            elif event.kind == "failed":
                self._fail(session, descriptor.stage, descriptor.message, descriptor)
            elif session.cancel_requested or event.kind == "cancelled":
                self._cancelled(session)
            elif event.kind == "completed" and not session._terminal:
                req = session.request
                trim = req.trim_samples if req.purpose == "main" and req.trim_samples < req.target_samples else 0
                if (descriptor.sample_rate != req.sample_rate or descriptor.channels != req.channels
                        or descriptor.purpose != req.purpose or descriptor.raw_frames != req.target_samples
                        or descriptor.final_frames != req.target_samples - trim):
                    self._fail(session, "protocol", "worker final count/channel/rate contract mismatch")
                    return
                session.state = "delivering"
                session._deadline = None
                session.reader = self._reader_factory(descriptor,
                    lambda outcome: self._inbox.put(("read", session, outcome)))
                session._reader_released = False
                try:
                    self._start_thread(session.reader.thread, start=session.reader.start)
                finally:
                    if session.reader.thread.ident is None:
                        # Reader construction allocates no file handles; only its
                        # thread may open the result. A failed pre-start attempt
                        # never acquired read ownership. Otherwise wait for _read.
                        session._reader_released = True
                        session.reader = None
            else:
                self._release(session)

    def _read(self, session, outcome):
        session._reader_released = outcome.handles_released
        if outcome.error:
            self._fail(session, "read", outcome.error)
        elif not session._terminal and not session.cancel_requested and not session._reject_requested:
            session.audio = outcome.audio
            self._notify(session, "result_ready", outcome.audio)
        self._release(session)

    def _accept(self, session):
        if (session is not self._active or session._terminal or session.cancel_requested
                or session._reject_requested):
            return
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
                    if self._active is session:
                        self._active = None
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
                if self._active is session:
                    self._active = None
            self._notify(session, "release_failed", session.release_error)
        if not session._cleanup_failed:
            session.released.set()
            self._notify(session, "released")

    def _retire(self, worker):
        if worker.retiring:
            return
        worker.retiring = True
        worker.kill_deadline = time.monotonic() + self._terminate_timeout
        if worker.process.is_alive():
            worker.process.terminate()

    def _dead(self, worker):
        session = self._active
        if session is not None and session.generation == worker.generation:
            if not session._terminal:
                self._fail(session, "worker", "recording worker exited before result acceptance")
            session._child_released = True
        worker.stop.set()
        worker.process.join(timeout=0)
        for thread in worker.threads:
            thread.join(.2)
        worker.control.close()
        worker.preview.close()
        worker.process.close()
        self._worker = None
        if session is not None:
            self._release(session)
            # A stalled reader isolates only its old path after PID death.
            if session._terminal and not session._reader_released:
                with self._lock:
                    if self._active is session:
                        self._active = None

    def _tick(self):
        retained = []
        for thread in self.threads:
            if thread is self._supervisor or thread.is_alive():
                retained.append(thread)
            else:
                thread.join(timeout=0)
        self.threads = retained
        now = time.monotonic()
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
            elif worker.deadline is not None and now >= worker.deadline:
                self._retire(worker)
                if self._active is not None:
                    self._fail(self._active, "ready_timeout", "recording worker ready deadline exceeded")
        session = self._active
        if session is not None and session._deadline is not None and now >= session._deadline:
            session._deadline = None
            if not session._terminal:
                stage = "cancel_timeout" if session.cancel_requested else "start_timeout"
                self._fail(session, stage, f"recording {stage} deadline exceeded")
            else:
                self._diagnose(f"{session.request.request_id}: reader/path release deadline exceeded; path remains leased")
            if worker is not None:
                self._retire(worker)
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
            if session is self._active and session._preview_pending == sequence:
                session._preview_pending = None
                self._command("preview_ack", session, sequence)
        elif kind == "broken":
            worker = item[1]
            if worker is self._worker and not worker.retiring:
                self._retire(worker)
                if self._active is not None:
                    self._fail(self._active, "worker", f"recording IPC closed: {item[2]}")
        elif kind == "shutdown":
            self._shutdown_deadline = time.monotonic() + self._shutdown_timeout
            if self._active is not None:
                self._request_cancel(self._active)
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
                self._logger.exception("Recording service operation failed")
                self._diagnose(str(exc))
                if self._worker is not None:
                    self._retire(self._worker)
                if self._active is not None:
                    self._fail(self._active, "service", str(exc))
            if self._closing and self._worker is None and not self._leases:
                self._report_shutdown()
                self.closed.set()
                return
