"""Manage one resident station worker, without business queues or UI objects.

The future AnalysisService owns one manager per station and dispatches only one
task at a time. Public methods belong to one host thread. Pipe I/O runs in a
separate thread, so receiving a large/partial matrix never blocks poll_events.
"""

from collections import deque
import math
import multiprocessing
import pickle
import sys
from queue import Empty, Queue, SimpleQueue
from threading import Thread
from time import monotonic, sleep

from .contracts import AnalysisTask
from .runtime_protocol import RunTask, StopWorker, WorkerEvent, failed_task, runtime_error
from .worker import analysis_worker_main


def _restore_worker_pipe(handle, readable, writable):
    from multiprocessing.connection import PipeConnection

    return PipeConnection(handle, readable=readable, writable=writable)


class _SpawnOwnedPipe:
    """Duplicate into the spawned child, not a handle it must steal from us.

    CPython's normal Windows PipeConnection reducer leaves a duplicate in the
    parent until the child unpickles it. An early startup kill never steals it.
    A child-owned duplicate is closed by Windows even before unpickling starts.
    """

    def __init__(self, connection):
        self.connection = connection

    def __reduce__(self):
        from multiprocessing.context import get_spawning_popen

        popen = get_spawning_popen()
        handle = popen.duplicate_for_child(self.connection.fileno())
        return _restore_worker_pipe, (
            handle, self.connection.readable, self.connection.writable,
        )


def _exchange(connection, outbound, inbound):
    """Own the parent pipe; peer exit breaks an incomplete receive or send."""
    try:
        while True:
            try:
                serialized = outbound.get_nowait()
            except Empty:
                serialized = None
            if serialized is not None:
                connection.send_bytes(serialized)
            del serialized
            if connection.poll(0.02):
                message = connection.recv()
                inbound.put(("message", message))
                del message
    except Exception as error:
        if isinstance(error, EOFError):
            return
        inbound.put(("transport_error", runtime_error(
            "communication", type(error).__name__, error,
        )))
    finally:
        connection.close()
        inbound.put(("transport_closed", None))


class AnalysisProcessManager:
    """One station's Process/pipe owner; no automatic recovery or task backlog."""

    def __init__(self, station_id, *, startup_timeout=60.0,
                 worker_target=analysis_worker_main):
        if not isinstance(station_id, str) or not station_id.strip():
            raise ValueError("station_id must be non-empty")
        self._check_timeout(startup_timeout)
        self.station_id = station_id
        self._startup_timeout = startup_timeout
        self._worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._thread = None
        self._generation = 0
        self._state = "closed"
        self._pending = deque()
        self._task = None
        self._result_pending = False
        self._failure = None
        self._stop_requested = False
        self._stopped_seen = False
        self._transport_closed_at = None
        self._transport_error = None
        self._abort_at = None
        self._cleanup_reported = False

    @property
    def state(self):
        return self._state

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def closed(self):
        return self._process is None and (self._thread is None or not self._thread.is_alive())

    def start(self):
        if self._result_pending:
            raise RuntimeError("Consume the previous task result before starting again")
        if not self.closed:
            if self.state in {"starting", "ready", "busy"}:
                return self.pid
            raise RuntimeError("Previous worker is still closing")
        self._generation += 1
        self._task = None
        self._result_pending = False
        self._failure = None
        self._stop_requested = False
        self._stopped_seen = False
        self._transport_closed_at = None
        self._transport_error = None
        self._abort_at = None
        self._cleanup_reported = False
        # One run command plus one stop command; there is no business backlog.
        self._outbound = Queue(maxsize=2)
        # A trusted worker emits at most ready/start/result/stopped per dispatch.
        # No subsequent dispatch is allowed until the prior result is consumed.
        self._inbound = SimpleQueue()
        parent = child = None
        self._started_at = monotonic()
        self._status("starting")
        try:
            parent, child = self._context.Pipe(duplex=True)
            child_argument = _SpawnOwnedPipe(child) if sys.platform == "win32" else child
            self._process = self._context.Process(
                target=self._worker_target,
                args=(child_argument, self.station_id, self._generation),
                name=f"analysis-station-{self.station_id[:24]}",
            )
            self._process.start()
            self._thread = Thread(
                target=_exchange, args=(parent, self._outbound, self._inbound),
                name=f"analysis-io-{self.pid}", daemon=True,
            )
            self._thread.start()
        except Exception as error:
            if parent is not None:
                parent.close()
            self._fail(runtime_error("startup", type(error).__name__, error))
            if self._process is not None:
                if self._process.pid is not None:
                    self.abort(self._failure)
                else:
                    self._process.close()
                    self._process = None
            if self.closed:
                self._status("closed", self._failure)
            raise
        finally:
            # Keeping the peer handle in the parent prevents EOF after a crash.
            if child is not None:
                child.close()
        return self.pid

    def submit(self, task: AnalysisTask):
        if self.state != "ready" or self._task is not None or self._result_pending or self._stop_requested:
            raise RuntimeError("Worker is not ready for another task")
        if not isinstance(task, AnalysisTask) or task.station_id != self.station_id:
            raise ValueError("Task station does not match this manager")
        # Serialize before acceptance: caller changes cannot alter the IPC snapshot.
        serialized = pickle.dumps(RunTask(task), protocol=pickle.HIGHEST_PROTOCOL)
        self._outbound.put_nowait(serialized)
        self._task = task
        self._status("busy")

    def request_stop(self):
        """Stop after the current accepted task; keep polling until closed."""
        if self.closed or self._stop_requested:
            return
        self._outbound.put_nowait(pickle.dumps(StopWorker()))
        self._stop_requested = True
        if self._failure is None:
            self._status("stopping")

    def poll_events(self, max_events=32):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if self._process is not None:
            drained = self._drain_transport(max_events)
            self._maintain(drained, max_events)
        events = []
        while self._pending and len(events) < max_events:
            event = self._pending.popleft()
            events.append(event)
            if event["event_type"] == "task_result":
                self._result_pending = False
                if self.state == "busy" and not self._stop_requested:
                    self._status("ready")
        return tuple(events)

    def abort(self, error, *, kill_after=1.0, cleanup_timeout=3.0):
        """Fail the in-flight task and request termination without joining.

        Continue polling until closed. Repeated requests do not extend deadlines.
        A cleanup timeout is reported once; unreleased resources stay owned here.
        """
        self._check_timeout(kill_after)
        self._check_timeout(cleanup_timeout)
        if self.closed or self._abort_at is not None:
            return
        self._fail(error)
        self._abort_at = monotonic()
        self._kill_at = self._abort_at + kill_after
        self._cleanup_at = self._abort_at + cleanup_timeout
        self._process.terminate()

    def close(self, timeout=5.0, *, terminate_timeout=2.0):
        """Headless convenience: drain, then bounded terminate/kill if needed.

        This method waits; a GUI should request_stop and continue polling instead.
        Returns all events collected during closing, including the final result.
        """
        self._check_timeout(timeout)
        self._check_timeout(terminate_timeout)
        self.request_stop()
        events = list(self.poll_events())
        deadline = monotonic() + timeout
        while not self.closed and monotonic() < deadline:
            sleep(0.01)
            events.extend(self.poll_events())
        if not self.closed:
            self._fail(runtime_error("shutdown", "ShutdownTimeout", "Worker did not finish before shutdown deadline"))
            self._process.terminate()
            deadline = monotonic() + terminate_timeout
            kill_at = monotonic() + terminate_timeout / 2
            killed = False
            while not self.closed and monotonic() < deadline:
                if not killed and monotonic() >= kill_at and self._process.is_alive():
                    self._process.kill()
                    killed = True
                sleep(0.01)
                events.extend(self.poll_events())
        events.extend(self.poll_events())
        if not self.closed:
            # Preserve collected task results for a subsequent poll by the caller.
            self._pending.extendleft(reversed(events))
            raise TimeoutError("Analysis process or IPC resources have not been released")
        return tuple(events)

    def _drain_transport(self, limit):
        for _ in range(limit):
            try:
                kind, message = self._inbound.get_nowait()
            except Empty:
                return True
            if kind == "transport_closed":
                self._transport_closed_at = monotonic()
            elif kind == "transport_error":
                self._transport_error = message
            else:
                self._handle(message)
        return False

    def _handle(self, event):
        if not isinstance(event, WorkerEvent) or not isinstance(event.payload, dict):
            self.abort(runtime_error("communication", "InvalidWorkerMessage", "Unexpected IPC event"))
            return
        if event.generation != self._generation:
            return
        if event.station_id != self.station_id:
            self.abort(runtime_error("communication", "StationMismatch", "Worker event station does not match"))
        elif event.kind == "ready":
            if self.state == "starting":
                self._status("ready")
        elif event.kind in {"task_started", "task_result"}:
            if self._task is None:
                return  # Already failed/closed; never emit a second terminal result.
            if event.task_id != self._task.task_id:
                self.abort(runtime_error("communication", "TaskMismatch", "Worker event task does not match"))
                return
            if event.kind == "task_result":
                expected = (self._task.task_id, self.station_id, self._task.condition_id)
                actual = tuple(event.payload.get(k) for k in ("task_id", "station_id", "condition_id"))
                if actual != expected:
                    self.abort(runtime_error("communication", "ResultMismatch", "Worker result identity does not match"))
                    return
            self._emit(event.kind, event.task_id, event.payload)
            if event.kind == "task_result":
                self._task = None
        elif event.kind == "worker_error":
            self._fail(event.payload)
        elif event.kind == "stopped":
            self._stopped_seen = True
        else:
            self.abort(runtime_error("communication", "InvalidWorkerMessage", f"Unknown event: {event.kind}"))

    def _maintain(self, drained, budget):
        process = self._process
        if self._abort_at is not None:
            now = monotonic()
            if now >= self._kill_at and process.is_alive():
                process.kill()
                self._kill_at = float("inf")
            resources_live = process.is_alive() or (
                self._thread is not None and self._thread.is_alive()
            )
            if now >= self._cleanup_at and resources_live and not self._cleanup_reported:
                self._cleanup_reported = True
                self._status("failed", runtime_error(
                    "shutdown", "CleanupTimeout", "Process or IPC resources are still owned",
                ))
        if process.is_alive():
            if self.state == "starting" and monotonic() - self._started_at > self._startup_timeout:
                self.abort(runtime_error("startup", "StartupTimeout", "Worker did not report ready"))
            elif self._transport_closed_at is not None and monotonic() - self._transport_closed_at > 0.2:
                self.abort(self._transport_error or runtime_error(
                    "communication", "ConnectionClosed", "Worker closed IPC without exiting",
                ))
            return
        # A child may have exited while a large last result is still being decoded.
        # Reap only after the I/O thread has delivered everything and closed its end.
        if self._thread is not None and self._thread.is_alive():
            return
        if not drained or not self._drain_transport(budget):
            return
        process.join(timeout=0)
        if (
            self._transport_error is not None
            or self._task is not None
            or not self._stopped_seen
            or process.exitcode != 0
        ):
            self._fail(self._failure or self._transport_error or runtime_error(
                "worker", "WorkerExit", f"Worker exited with code {process.exitcode}",
            ))
        process.close()
        self._process = None
        self._thread = None
        self._status("closed", self._failure)

    def _fail(self, error):
        if self._failure is not None:
            return
        self._failure = error
        self._status("failed", error)
        if self._task is not None:
            self._emit("task_result", self._task.task_id, failed_task(self._task, error))
            self._task = None

    def _status(self, state, error=None):
        self._state = state
        self._emit("station_status", None, {"state": state, "pid": self.pid, "error": error})

    def _emit(self, event_type, task_id, payload):
        if event_type == "task_result":
            self._result_pending = True
        self._pending.append({
            "event_type": event_type, "station_id": self.station_id,
            "task_id": task_id, "payload": payload,
        })

    @staticmethod
    def _check_timeout(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Timeout must be a finite non-negative number")
