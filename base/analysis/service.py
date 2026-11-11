"""Controller-facing resident analysis scheduling; no Qt or configuration files.

All public methods belong to one host thread. Call poll_events throughout startup,
work, removal and shutdown. This service never waits for a task or joins a process.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
import logging
import math
from time import monotonic

from .contracts import AnalysisInputError, AnalysisTask
from .input_adapter import build_analysis_batch
from .process_manager import AnalysisProcessManager
from .runtime_protocol import failed_task, runtime_error


@dataclass
class _Station:
    manager: AnalysisProcessManager
    waiting: deque = field(default_factory=deque)
    active: AnalysisTask | None = None
    dispatched_at: float = 0.0
    result_count: int = 0
    drain_deadline: float | None = None
    failed: bool = False
    exhausted: bool = False
    retries: deque = field(default_factory=deque)
    retry_at: float = 0.0


class AnalysisService:
    """One service owns all station managers and bounded business queues.

    waiting_capacity counts tasks not dispatched yet. Completed but undelivered
    results also reserve capacity, so a slow consumer cannot accumulate matrices.
    Recent IDs are bounded; callers must still provide globally unique task IDs.
    """

    def __init__(self, *, waiting_capacity=8, recent_task_capacity=4096,
                 startup_timeout=60.0, task_timeout=120.0, drain_timeout=30.0,
                 shutdown_timeout=30.0, max_restarts=2, restart_window=60.0,
                 restart_delay=0.2, manager_factory=AnalysisProcessManager):
        for name, value, minimum in (
            ("waiting_capacity", waiting_capacity, 1),
            ("recent_task_capacity", recent_task_capacity, 1),
            ("max_restarts", max_restarts, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("startup_timeout", startup_timeout), ("task_timeout", task_timeout),
            ("drain_timeout", drain_timeout), ("shutdown_timeout", shutdown_timeout),
            ("restart_window", restart_window), ("restart_delay", restart_delay),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if restart_window == 0:
            raise ValueError("restart_window must be positive to bound recovery attempts")
        self._capacity = waiting_capacity
        self._recent_capacity = recent_task_capacity
        self._startup_timeout = startup_timeout
        self._task_timeout = task_timeout
        self._drain_timeout = drain_timeout
        self._shutdown_timeout = shutdown_timeout
        self._max_restarts = max_restarts
        self._restart_window = restart_window
        self._restart_delay = restart_delay
        self._factory = manager_factory
        self._stations = {}
        self._targets = set()
        self._accepted = set()
        self._terminal = set()
        self._recent = OrderedDict()
        self._events = deque()
        self._shutdown_deadline = None
        self._closed = False
        self._cursor = 0
        self._logger = logging.getLogger(__name__)

    @property
    def closed(self):
        return self._closed

    @property
    def pending_task_count(self):
        """Includes accepted tasks whose terminal result has not been delivered."""
        return len(self._accepted)

    @property
    def station_states(self):
        return {
            sid: {"state": self._state(slot), "pid": slot.manager.pid,
                  "waiting_count": len(slot.waiting)}
            for sid, slot in self._stations.items()
        }

    def _state(self, slot):
        if slot.exhausted:
            return "failed"
        if slot.drain_deadline is not None and not slot.manager.closed:
            return "draining"
        return slot.manager.state

    def sync_stations(self, station_ids):
        """Accept a complete desired set; process startup happens while polling."""
        valid = isinstance(station_ids, (list, tuple)) and all(
            isinstance(sid, str) and sid.strip() == sid and sid for sid in station_ids
        )
        if not valid or len(set(station_ids)) != len(station_ids):
            return self._sync_receipt(False, error=runtime_error(
                "input", "InvalidStations", "Provide distinct non-empty station IDs",
            ))
        if self._shutdown_deadline is not None:
            return self._sync_receipt(False, error=runtime_error(
                "service", "ServiceClosing", "Service is shutting down",
            ))
        targets = set(station_ids)
        added, removed = targets - self._targets, self._targets - targets
        retained = targets & self._targets
        for sid in sorted(added):
            if sid not in self._stations:
                self._stations[sid] = _Station(self._factory(
                    sid, startup_timeout=self._startup_timeout,
                ))
        now = monotonic()
        for sid in removed:
            slot = self._stations[sid]
            if slot.drain_deadline is None:
                slot.drain_deadline = now + self._drain_timeout
                self._emit("station_status", sid, payload={
                    "state": "draining", "pid": slot.manager.pid, "error": None,
                })
        self._targets = targets
        return self._sync_receipt(True, added, retained, removed)

    def _sync_receipt(self, accepted, added=(), retained=(), removed=(), error=None):
        return {
            "accepted": accepted, "target_station_ids": sorted(self._targets),
            "added_station_ids": sorted(added), "retained_station_ids": sorted(retained),
            "removed_station_ids": sorted(removed), "error": error,
        }

    def submit_batch(self, payload):
        """Construct a snapshot and validate the whole batch before any admission."""
        def reject(kind, message, task_id=None, field_path="$input"):
            return {"accepted": False, "accepted_task_ids": [], "issues": [{
                "task_id": task_id, "field_path": field_path,
                "error_type": kind, "message": message,
            }]}

        if self._shutdown_deadline is not None:
            return reject("ServiceClosing", "Service is shutting down")
        try:
            batch = build_analysis_batch(payload)
        except AnalysisInputError as error:
            return {"accepted": False, "accepted_task_ids": [], "issues": [asdict(error.issue)]}
        counts = Counter()
        for task in batch.tasks:
            if task.task_id in self._accepted or task.task_id in self._recent:
                return reject("DuplicateTask", "Task ID has already been accepted", task.task_id)
            slot = self._stations.get(task.station_id)
            if (task.station_id not in self._targets or slot is None
                    or self._state(slot) not in {"ready", "busy"}):
                return reject("StationNotReady", "Station is not accepting tasks", task.task_id)
            counts[task.station_id] += 1
        for sid, count in counts.items():
            slot = self._stations[sid]
            if len(slot.waiting) + slot.result_count + count > self._capacity:
                return reject("QueueFull", f"Station {sid} waiting capacity exceeded")
        for task in batch.tasks:
            self._stations[task.station_id].waiting.append(task)
            self._accepted.add(task.task_id)
        return {"accepted": True, "accepted_task_ids": [t.task_id for t in batch.tasks],
                "issues": [asdict(issue) for issue in batch.preparation_issues]}

    def shutdown(self):
        """Drain accepted work up to the original deadline; never restart workers."""
        if self._shutdown_deadline is not None:
            return
        self._shutdown_deadline = monotonic() + self._shutdown_timeout
        self._targets.clear()
        for slot in self._stations.values():
            if slot.drain_deadline is None:
                slot.drain_deadline = self._shutdown_deadline
            else:
                slot.drain_deadline = min(slot.drain_deadline, self._shutdown_deadline)

    def poll_events(self, max_events=32):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        # Rotate the first station; each has a fixed per-poll message budget.
        ids = list(self._stations)
        if ids:
            offset = self._cursor % len(ids)
            ids = ids[offset:] + ids[:offset]
            self._cursor += 1
        for sid in ids:
            self._advance(sid, self._stations[sid])
        output = []
        while self._events and len(output) < max_events:
            event = self._events.popleft()
            output.append(event)
            if event["event_type"] == "task_result":
                task_id = event["task_id"]
                self._accepted.remove(task_id)
                self._terminal.remove(task_id)
                self._stations[event["station_id"]].result_count -= 1
                self._recent[task_id] = None
                if len(self._recent) > self._recent_capacity:
                    self._recent.popitem(last=False)
        if (self._shutdown_deadline is not None and not self._stations
                and not self._accepted and not self._events and not self._closed):
            self._closed = True
            event = {"event_type": "service_closed", "station_id": None,
                     "task_id": None, "payload": {}}
            if len(output) < max_events:
                output.append(event)
            else:
                self._events.append(event)
        return tuple(output)

    def _advance(self, sid, slot):
        manager = slot.manager
        # Drain available valid results before evaluating deadlines.
        self._receive(slot)
        now = monotonic()
        if slot.drain_deadline is not None and now >= slot.drain_deadline:
            error = runtime_error("shutdown", "DrainTimeout", "Accepted work exceeded drain deadline")
            self._fail_waiting(slot, error)
            manager.abort(error)
        elif slot.active is not None and now - slot.dispatched_at >= self._task_timeout:
            manager.abort(runtime_error("execution", "TaskTimeout", "Task exceeded execution deadline"))

        if manager.closed:
            self._advance_closed(sid, slot, now)
        else:
            self._dispatch(slot)
            if slot.drain_deadline is not None and not slot.waiting and slot.active is None:
                manager.request_stop()

    def _receive(self, slot):
        manager = slot.manager
        for event in manager.poll_events(max_events=16):
            kind = event["event_type"]
            if kind == "task_result":
                if slot.active is not None and event["task_id"] == slot.active.task_id:
                    self._complete(slot, slot.active, event["payload"])
                    slot.active = None
            elif kind == "station_status":
                if event["payload"]["error"] is not None:
                    slot.failed = True
                if (slot.drain_deadline is not None
                        and event["payload"]["state"] in {"ready", "busy", "starting"}):
                    event["payload"]["state"] = "draining"
                self._events.append(event)
            elif slot.active is not None and event["task_id"] == slot.active.task_id:
                self._events.append(event)

    def _advance_closed(self, sid, slot, now):
        if slot.active is not None or slot.result_count:
            # Preserve the slot until manager and caller have received all terminals.
            return
        if not slot.waiting and (sid not in self._targets or self._shutdown_deadline is not None):
            del self._stations[sid]
            return
        if self._shutdown_deadline is not None:
            self._fail_waiting(slot, runtime_error(
                "worker", "WorkerUnavailable", "Worker closed during service shutdown",
            ))
            return
        if slot.drain_deadline is not None and not slot.waiting:
            # Re-added during removal: finish closing the old generation first.
            slot.drain_deadline = None
        if slot.exhausted or now < slot.retry_at:
            return
        if slot.failed and not self._allow_restart(slot, now):
            slot.exhausted = True
            error = runtime_error("worker", "WorkerUnavailable", "Restart budget exhausted")
            self._fail_waiting(slot, error)
            self._emit("station_status", sid, payload={"state": "failed", "pid": None, "error": error})
            return
        slot.failed = False
        try:
            slot.manager.start()
        except Exception:
            # The manager emits startup diagnostics and retains any live resources.
            self._logger.exception("Cannot start analysis station %s", sid)
            slot.failed = True
        slot.retry_at = now + self._restart_delay

    def _dispatch(self, slot):
        manager = slot.manager
        if (manager.state == "ready" and slot.active is None
                and not slot.result_count and slot.waiting):
            task = slot.waiting.popleft()
            try:
                manager.submit(task)
            except Exception as error:
                self._logger.exception("Cannot dispatch analysis task %s", task.task_id)
                self._complete(slot, task, failed_task(task, runtime_error(
                    "dispatch", type(error).__name__, error,
                )))
            else:
                slot.active = task
                slot.dispatched_at = monotonic()

    def _allow_restart(self, slot, now):
        while slot.retries and slot.retries[0] < now - self._restart_window:
            slot.retries.popleft()
        if len(slot.retries) >= self._max_restarts:
            return False
        slot.retries.append(now)
        return True

    def _fail_waiting(self, slot, error):
        while slot.waiting:
            task = slot.waiting.popleft()
            self._complete(slot, task, failed_task(task, error))

    def _complete(self, slot, task, result):
        if task.task_id in self._terminal:
            return
        self._terminal.add(task.task_id)
        slot.result_count += 1
        self._emit("task_result", task.station_id, task.task_id, result)

    def _emit(self, kind, station_id=None, task_id=None, payload=None):
        self._events.append({"event_type": kind, "station_id": station_id,
                             "task_id": task_id, "payload": payload or {}})
