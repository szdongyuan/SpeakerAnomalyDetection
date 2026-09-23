"""Instance-owned CSV supervisor. Public operations only change memory and wake it."""

import logging
from collections import deque
import multiprocessing
from multiprocessing.connection import wait
from pathlib import Path
import stat
import threading
import time
from uuid import uuid4

from base.raw_audio_csv_protocol import (
    CsvExportCommand, CsvExportRequest, CsvFailure, CsvResult, CsvServiceEvent, CsvTiming,
)
from base.raw_audio_csv_zip import raw_csv_zip_path
from base.raw_audio_csv_tasks import CsvTaskLedger
from base.raw_audio_csv_worker import raw_audio_csv_worker
from consts.raw_audio_csv_consts import (
    RAW_AUDIO_CSV_PROTOCOL_VERSION, RAW_AUDIO_CSV_READY_TIMEOUT_SECONDS,
    RAW_AUDIO_CSV_SHUTDOWN_TIMEOUT_SECONDS,
)


class CsvSubscription:
    def __init__(self, service, key):
        self._service = service
        self._key = key

    def unsubscribe(self):
        with self._service._subscriber_lock:
            self._service._subscribers.pop(self._key, None)


class RawAudioCsvService:
    """A single spawn worker supervised without GUI-thread I/O.

    ``commit`` returns accepted/invalid/path_busy/unavailable; rejection consumes
    a valid reservation and is reported synchronously by that status. Accepted
    tasks receive terminal and released events with their original task snapshot.
    Subscribers execute briefly on the supervisor (a Qt bridge should enqueue).
    The final closed notification runs on a completion thread after supervisor exit.
    ``closed`` is an observation event, never a GUI-thread wait instruction.
    """

    def __init__(self, *, context=None, worker_target=raw_audio_csv_worker,
                 worker_args=(), ready_timeout=RAW_AUDIO_CSV_READY_TIMEOUT_SECONDS,
                 shutdown_timeout=RAW_AUDIO_CSV_SHUTDOWN_TIMEOUT_SECONDS,
                 clock=time.monotonic):
        self._ledger = CsvTaskLedger()
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._worker_args = worker_args
        self._ready_timeout = ready_timeout
        self._shutdown_timeout = shutdown_timeout
        self._clock = clock
        self._wake = threading.Event()
        self._shutdown_requested = threading.Event()
        self.closed = threading.Event()
        self._subscriber_lock = threading.Lock()
        self._subscribers = {}
        # Diagnostics are best effort and bounded independently of correctness.
        # Public methods only append small immutable observations; consumers
        # (including log handlers) always run on the supervisor.
        self._observation_lock = threading.RLock()
        self._observations = deque(maxlen=256)
        self._process = None
        self._control = None
        self._generation = 0
        self._ready = False
        self._ready_observed_at = None
        self._last_worker_pid = None
        self._active = None
        self._temporaries = {}
        self._deadline = None
        self._shutdown_deadline = None
        self._stopping = False
        self._termination_sent = False
        self._pipe_eof = False
        self._eof_since = None
        self._last_snapshot = None
        self._supervision_done = False
        self._completion_thread = None
        self._thread = threading.Thread(target=self._run, name="raw-csv-supervisor", daemon=True)
        self._thread.start()

    def reserve(self, owner_id):
        with self._observation_lock:
            result = self._ledger.reserve(owner_id)
            if result.reservation is not None:
                self._observations.append(CsvServiceEvent("reserve", self.snapshot(), timing=CsvTiming(
                    "reserve", time.perf_counter(), owner_id, token_id=result.reservation.token_id)))
        self._wake.set()
        return result

    def release_reservation(self, token):
        result = self._ledger.release_reservation(token)
        self._wake.set()
        return result

    def commit(self, token, request, *, recording_permit=None):
        with self._observation_lock:
            # The supervisor can dispatch as soon as the ledger publishes the
            # task, even before this call returns or its diagnostic is delivered.
            submitted_at = time.perf_counter()
            result = self._ledger.commit(token, request, recording_permit=recording_permit)
            recording_id = request.recording_id if isinstance(request, CsvExportRequest) else token.owner_id
            task_id = request.task_id if isinstance(request, CsvExportRequest) else ""
            self._observations.append(CsvServiceEvent("submit", self.snapshot(), detail=result,
                timing=CsvTiming("submit", submitted_at, recording_id, task_id, token.token_id)))
        self._wake.set()
        return result

    def snapshot(self):
        return self._ledger.snapshot()

    def begin_shutdown(self):
        self._ledger.begin_shutdown()
        self._shutdown_requested.set()
        self._wake.set()

    def subscribe(self, callback):
        key = uuid4().hex
        with self._subscriber_lock:
            self._subscribers[key] = callback
        return CsvSubscription(self, key)

    def paths_busy(self, paths):
        return self._ledger.paths_busy(paths)

    def try_acquire_mutation(self, paths):
        return self._ledger.try_acquire_mutation(paths)

    def release_mutation(self, permit, *, ready=None):
        result = self._ledger.release_mutation(permit, ready=ready)
        self._wake.set()
        return result

    def defer_mutation(self, paths, callback, *, ready=None):
        paths = tuple(paths)

        def run_mutation():
            try:
                callback()
            except OSError as error:
                # User-requested file operations can fail independently of CSV.
                # Report the exact scope; the ledger's finally releases its
                # permit. Internal supervisor errors still reach the fatal gate.
                detail = f"{'; '.join(paths)}: {type(error).__name__}: {str(error)[:1000]}"
                logging.getLogger(__name__).warning("CSV deferred mutation failed: %s", detail)
                self._emit("mutation_failed", detail=detail)

        result = self._ledger.defer_mutation(paths, run_mutation, ready=ready)
        self._wake.set()
        return result

    def _emit(self, kind, *, task=None, result=None, detail=""):
        stage = "terminal_received" if kind == "terminal" else kind
        timing = CsvTiming(stage, time.perf_counter(),
            task.request.recording_id if task else "", task.request.task_id if task else "",
            generation=task.generation if task and task.generation is not None else self._generation,
            worker_pid=self._process.pid if self._process else self._last_worker_pid,
            worker_ready_seconds=self._ready_observed_at)
        event = CsvServiceEvent(kind, self.snapshot(), task, result, detail, timing)
        self._deliver_event(event)

    def _deliver_event(self, event):
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers.values())
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # External consumer boundary: any subscriber can fail. Ledger
                # transitions are owned here, never delegated to a consumer.
                logging.getLogger(__name__).exception("CSV event consumer failed kind=%s", event.kind)

    def _drain_observations(self):
        with self._observation_lock:
            observations = tuple(self._observations)
            self._observations.clear()
        for event in observations:
            self._deliver_event(event)

    def _run(self):
        while not self._supervision_done:
            self._wake.wait(0.01)
            self._wake.clear()
            try:
                self._step()
            except Exception as error:
                # Outermost supervisor boundary covers process creation, IPC,
                # filesystem and deferred user callbacks. Preserve live leases
                # and stop dispatch; only proven death can release active work.
                logging.getLogger(__name__).exception("CSV supervisor failed generation=%s", self._generation)
                self._unavailable(f"{type(error).__name__}: {error}"[:1000])
                self._terminate()

    def _notify_closed(self):
        self._thread.join()
        self._emit("closed")
        self.closed.set()

    def _step(self):
        self._drain_observations()
        self._ledger.run_deferred_mutations()
        if self._process is not None:
            handles = [self._process.sentinel]
            if not self._pipe_eof:
                handles.append(self._control)
            available = wait(handles, timeout=0)
            # Drain buffered terminal messages before interpreting the sentinel.
            if self._control in available:
                for _ in range(64):
                    if self._pipe_eof or not self._poll():
                        break
                    self._receive()
            if self._process.sentinel in available:
                self._reap()
            elif self._pipe_eof and not self._stopping:
                if self._eof_since is None:
                    self._eof_since = self._clock()
                    if self._active is not None:
                        self._fail(self._active, "WorkerExit", "Worker pipe EOF; result unconfirmed")
                if self._clock() >= self._eof_since + self._shutdown_timeout:
                    self._unavailable("Worker pipe EOF; resources not released")
                    self._terminate()
            elif not self._ready and not self._stopping and self._clock() >= self._deadline:
                self._unavailable("Worker ready deadline exceeded")
                self._terminate()
            elif (self._stopping and not self._termination_sent and self._shutdown_deadline is not None
                  and self._clock() >= self._shutdown_deadline):
                self._terminate()
        snapshot = self.snapshot()
        if snapshot.phase != "unavailable" and snapshot.queued and self._process is None:
            self._start()
        if self._ready and self._active is None and not self._stopping and not self._pipe_eof:
            task = self._ledger.dispatch_next(self._generation)
            if task is not None:
                self._active = task
                target = Path(task.request.csv_path)
                # Keep controlled components short even for maximum-length CSV names.
                temporary_path = str(target.with_name(f".raw-csv-{uuid4().hex}.tmp"))
                zip_temporary_path = str(target.with_name(f".raw-zip-{uuid4().hex}.tmp"))
                self._temporaries = {temporary_path: None, zip_temporary_path: None}
                self._emit("dispatch", task=task)
                self._control.send(CsvExportCommand(
                    request=task.request, generation=self._generation,
                    temporary_path=temporary_path, zip_temporary_path=zip_temporary_path,
                ))
        if self._shutdown_requested.is_set() and self.snapshot().outstanding == 0:
            if self._process is not None and not self._stopping and not self._pipe_eof:
                self._control.send("shutdown")
                self._stopping = True
                self._shutdown_deadline = self._clock() + self._shutdown_timeout
            elif self._process is None and self._ledger.mark_closed():
                self._supervision_done = True
                self._completion_thread = threading.Thread(
                    target=self._notify_closed, name="raw-csv-completion", daemon=True,
                )
                self._completion_thread.start()
                return
        snapshot = self.snapshot()
        if snapshot != self._last_snapshot:
            self._last_snapshot = snapshot
            self._emit("state")

    def _start(self):
        self._generation += 1
        self._ready_observed_at = None
        self._last_worker_pid = None
        self._control, child = self._context.Pipe()
        try:
            self._process = self._context.Process(target=self._worker_target, args=(child, *self._worker_args))
            self._process.start()
            self._last_worker_pid = self._process.pid
        finally:
            child.close()
            if self._process is None or self._process.pid is None:
                if self._process is not None:
                    self._process.close()
                self._process = None
                self._control.close()
                self._control = None
        self._deadline = self._clock() + self._ready_timeout
        self._stopping = False
        self._termination_sent = False
        self._pipe_eof = False
        self._eof_since = None

    def _poll(self):
        try:
            return self._control.poll()
        except OSError:
            # Windows PeekNamedPipe reports a closed peer as BrokenPipeError.
            self._pipe_eof = True
            return False

    def _receive(self):
        try:
            message = self._control.recv()
        except (EOFError, OSError):
            self._pipe_eof = True
            return
        if isinstance(message, tuple) and len(message) == 3 and message[0] == "ready":
            if self._stopping or self._ready:
                return
            if message[1] == RAW_AUDIO_CSV_PROTOCOL_VERSION and message[2] == self._process.pid:
                self._ready = True
                self._ready_observed_at = time.perf_counter()
                tasks = self._ledger.task_snapshots()
                self._emit("ready", task=tasks[0] if tasks else None)
            else:
                self._unavailable("Worker ready protocol/PID mismatch")
                self._terminate()
        elif isinstance(message, tuple) and len(message) == 4 and message[0] == "started":
            if message[3] == self._process.pid and self._ledger.mark_running(message[1], message[2]):
                self._emit("started", task=self._ledger.task_snapshot(message[1]))
        elif isinstance(message, tuple) and len(message) == 5 and message[0] == "temporary_owned":
            if (self._matches_active(message[1], message[2]) and message[3] in self._temporaries
                    and self._temporaries[message[3]] is None
                    and isinstance(message[4], tuple) and len(message[4]) == 2
                    and all(type(value) is int for value in message[4])):
                self._temporaries[message[3]] = message[4]
        elif isinstance(message, (CsvResult, CsvFailure)):
            if (not self._matches_active(message.task_id, message.generation)
                    or message.worker_pid != self._process.pid):
                return
            if isinstance(message, CsvResult) and (
                    message.csv_path != self._active.request.csv_path
                    or message.archive_path != str(raw_csv_zip_path(self._active.request.csv_path))):
                return
            status = "succeeded" if isinstance(message, CsvResult) else "failed"
            if self._ledger.mark_terminal(message.task_id, message.generation, status):
                self._emit("terminal", task=self._ledger.task_snapshot(message.task_id), result=message)
                diagnostics = message.cleanup_diagnostics
                self._release_active(diagnostics)

    def _matches_active(self, task_id, generation):
        return (self._active is not None and self._active.request.task_id == task_id
                and self._active.generation == generation)

    def _fail(self, task, exception_type, message):
        if self._ledger.mark_terminal(task.request.task_id, task.generation, "failed"):
            failure = CsvFailure(task.request.task_id, task.generation or self._generation,
                                 "supervisor", exception_type, message[:1000])
            self._emit("terminal", task=self._ledger.task_snapshot(task.request.task_id), result=failure)

    def _unavailable(self, reason):
        first = self.snapshot().phase not in ("unavailable", "closed")
        self._ledger.mark_unavailable()
        if self._active is not None:
            self._fail(self._active, "WorkerExit", reason)
        for task in self._ledger.task_snapshots():
            if task.state == "queued":
                self._fail(task, "ServiceUnavailable", reason)
                released = self._ledger.release_task(task.request.task_id, task.generation)
                self._emit("released", task=released)
        if first:
            self._emit("unavailable", detail=reason)

    def _terminate(self):
        if self._process is None or self._termination_sent:
            return
        self._stopping = True
        self._termination_sent = True
        try:
            self._process.terminate()
        except OSError as error:
            # Failed termination is not death. Keep handles and every active
            # lease until the sentinel proves exit; never spawn a replacement.
            self._emit("resource_pending", detail=f"Termination unconfirmed: {error}"[:1000])

    def _reap(self):
        self._process.join(0)
        if self._process.exitcode is None:
            return
        exitcode = self._process.exitcode
        task = self._ledger.task_snapshot(self._active.request.task_id) if self._active else None
        self._emit("worker_exit", task=task, detail=f"Worker exit {exitcode}; death confirmed")
        if not self._ready and not self._stopping:
            self._unavailable(f"Worker exited before ready: {exitcode}")
        if self._active is not None:
            self._fail(self._active, "WorkerExit", f"Worker exit {exitcode}; result unconfirmed")
            diagnostics = self._cleanup_temporary()
        else:
            diagnostics = ()
        self._control.close()
        self._process.close()
        self._process = self._control = None
        self._ready = False
        self._stopping = False
        if self._active is not None:
            self._release_active(diagnostics)

    def _cleanup_temporary(self):
        diagnostics = []
        for temporary_path, identity in self._temporaries.items():
            diagnostics.extend(self._cleanup_owned_temporary(Path(temporary_path), identity))
        return tuple(diagnostics)

    @staticmethod
    def _cleanup_owned_temporary(path, identity):
        try:
            info = path.lstat()
        except FileNotFoundError:
            return ()
        except OSError as error:
            return (f"{path}: cleanup inspection failed: {error}"[:1000],)
        if (identity is None or (info.st_dev, info.st_ino) != identity
                or not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            return (f"{path}: temporary ownership unconfirmed; residual preserved"[:1000],)
        try:
            path.unlink()
        except OSError as error:
            return (f"{path}: {type(error).__name__}: {error}"[:1000],)
        return ()

    def _release_active(self, diagnostics):
        task = self._active
        released = self._ledger.release_task(task.request.task_id, task.generation,
                                             cleanup_diagnostics=diagnostics)
        self._active = None
        self._temporaries = {}
        self._emit("released", task=released)
