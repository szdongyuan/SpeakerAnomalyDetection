"""In-memory task admission. No process, filesystem or GUI dependencies."""

from collections import deque
from dataclasses import replace
import os
from threading import RLock
from typing import Callable, Iterable
from uuid import uuid4

from base.raw_audio_csv_protocol import (
    CsvAdmission, CsvExportRequest, CsvLedgerSnapshot, CsvReservation,
    CsvMutationPermit, CsvTaskSnapshot,
)
from base.raw_audio_csv_zip import raw_csv_zip_path
from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_CAPACITY


class CsvTaskLedger:
    def __init__(self):
        self._lock = RLock()
        self._phase = "open"
        self._reservations = {}
        self._tasks = {}
        self._seen_task_ids = set()
        self._queue = deque()
        self._active_task_id = None
        self._task_paths = {}
        self._permits = {}
        self._pending_mutations = {}
        self._pending_permit_releases = {}

    def reserve(self, owner_id: str) -> CsvAdmission:
        with self._lock:
            if self._phase != "open":
                return CsvAdmission("unavailable" if self._phase == "unavailable" else "closing")
            if len(self._reservations) + len(self._tasks) >= RAW_AUDIO_CSV_CAPACITY:
                return CsvAdmission("full")
            token = CsvReservation(uuid4().hex, owner_id)
            self._reservations[token.token_id] = token
            return CsvAdmission("accepted", token)

    def release_reservation(self, token: CsvReservation) -> bool:
        with self._lock:
            if self._reservations.get(token.token_id) != token:
                return False
            del self._reservations[token.token_id]
            return True

    def commit(self, token: CsvReservation, request: CsvExportRequest, *,
               recording_permit: CsvMutationPermit | None = None) -> str:
        # Resolve relative paths before taking the lock (abspath can call getcwd).
        paths = ()
        if (isinstance(request, CsvExportRequest)
                and isinstance(request.wav_path, str) and request.wav_path
                and isinstance(request.csv_path, str) and request.csv_path):
            paths = self._normalize_paths((request.wav_path, request.csv_path))
            if len(paths) == 2:
                # Derive the ZIP from the exact CSV path passed to the worker.
                paths = self._normalize_paths((*paths, str(raw_csv_zip_path(paths[1]))))
        with self._lock:
            if not self.release_reservation(token):
                return "invalid"
            if self._phase in ("unavailable", "closed"):
                return "unavailable"
            if not self._valid_request(token, request):
                return "invalid"
            if len(paths) != 3:
                return "invalid"
            if recording_permit is not None and (
                    self._permits.get(recording_permit.owner_id) is not recording_permit
                    or paths[0] not in recording_permit.paths):
                return "invalid"
            # A recording already owns its WAV. Keep that claim live while adding
            # the task, so even a waiting cleanup cannot enter between owners.
            if self._paths_busy(paths, ignore_permit=recording_permit):
                return "path_busy"
            request = replace(request, wav_path=paths[0], csv_path=paths[1])
            self._seen_task_ids.add(request.task_id)
            self._task_paths[request.task_id] = paths
            self._tasks[request.task_id] = CsvTaskSnapshot(request, "queued")
            self._queue.append(request.task_id)
            return "accepted"

    def _valid_request(self, token, request):
        return (
            isinstance(request, CsvExportRequest)
            and isinstance(request.task_id, str) and bool(request.task_id)
            and request.task_id not in self._seen_task_ids
            and isinstance(request.recording_id, str)
            and request.recording_id == token.owner_id
            and isinstance(request.owner_group, str)
            and isinstance(request.owner_record, str)
            and isinstance(request.wav_path, str) and bool(request.wav_path)
            and isinstance(request.csv_path, str) and bool(request.csv_path)
            and "\0" not in request.wav_path and "\0" not in request.csv_path
            and isinstance(request.raw_channels, tuple) and bool(request.raw_channels)
            and all(type(channel) is int and channel >= 0 for channel in request.raw_channels)
            and len(set(request.raw_channels)) == len(request.raw_channels)
        )

    def snapshot(self) -> CsvLedgerSnapshot:
        with self._lock:
            return CsvLedgerSnapshot(
                self._phase, RAW_AUDIO_CSV_CAPACITY, len(self._reservations),
                len(self._queue), len(self._tasks) - len(self._queue),
            )

    def task_snapshot(self, task_id: str) -> CsvTaskSnapshot | None:
        with self._lock:
            return self._tasks.get(task_id)

    def task_snapshots(self) -> tuple[CsvTaskSnapshot, ...]:
        """Bounded immutable view used to fail queued work on service loss."""
        with self._lock:
            return tuple(self._tasks.values())

    def dispatch_next(self, generation: int) -> CsvTaskSnapshot | None:
        with self._lock:
            if self._phase in ("closed", "unavailable") or self._active_task_id or not self._queue:
                return None
            task_id = self._queue.popleft()
            task = replace(self._tasks[task_id], state="dispatched", generation=generation)
            self._tasks[task_id] = task
            self._active_task_id = task_id
            return task

    def mark_running(self, task_id: str, generation: int) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.generation != generation or task.state != "dispatched":
                return False
            self._tasks[task_id] = replace(task, state="running")
            return True

    def mark_terminal(self, task_id: str, generation: int | None, status: str) -> bool:
        """Record a result without claiming that worker resources are released."""
        with self._lock:
            task = self._tasks.get(task_id)
            if (task is None or task.generation != generation or task.terminal_status
                    or status not in ("succeeded", "failed")
                    or (task.state == "queued" and status != "failed")):
                return False
            if task.state == "queued":
                self._queue.remove(task_id)
            self._tasks[task_id] = replace(task, state=status, terminal_status=status)
            return True

    def release_task(self, task_id: str, generation: int | None, *,
                     cleanup_diagnostics: tuple[str, ...] = ()) -> CsvTaskSnapshot | None:
        """Supervisor calls only after handles close or worker death is confirmed.

        Return the release diagnostic for logging; retain only the task ID tombstone.
        Failed unlink is diagnostic once handles are closed, not a capacity leak.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.generation != generation or not task.terminal_status:
                return None
            del self._tasks[task_id]
            del self._task_paths[task_id]
            if self._active_task_id == task_id:
                self._active_task_id = None
            return replace(task, state="released", cleanup_diagnostics=tuple(cleanup_diagnostics))

    def begin_shutdown(self) -> None:
        with self._lock:
            if self._phase == "open":
                self._phase = "draining"

    def mark_unavailable(self) -> None:
        with self._lock:
            if self._phase != "closed":
                self._phase = "unavailable"

    def mark_closed(self) -> bool:
        """Supervisor calls after worker and IPC teardown, once ownership is empty."""
        with self._lock:
            if (self._phase == "open" or self._reservations or self._tasks
                    or self._permits or self._pending_mutations):
                return False
            self._phase = "closed"
            return True

    @staticmethod
    def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(os.path.normcase(os.path.abspath(path)) for path in paths))

    def _paths_busy(self, paths, *, include_pending=True, ignore_permit=None):
        requested = set(paths)
        pending_requested = requested.difference(ignore_permit.paths) if ignore_permit else requested
        return (
            any(requested.intersection(owned) for owned in self._task_paths.values())
            or any(requested.intersection(permit.paths)
                   for permit in self._permits.values() if permit is not ignore_permit)
            or (include_pending and any(pending_requested.intersection(permit.paths)
                                        for permit, _, _ in self._pending_mutations.values()))
        )

    def paths_busy(self, paths: Iterable[str]) -> bool:
        normalized = self._normalize_paths(paths)
        with self._lock:
            return self._paths_busy(normalized)

    def try_acquire_mutation(self, paths: Iterable[str]) -> CsvMutationPermit | None:
        normalized = self._normalize_paths(paths)
        with self._lock:
            if self._phase == "closed" or not normalized or self._paths_busy(normalized):
                return None
            permit = CsvMutationPermit(uuid4().hex, normalized)
            self._permits[permit.owner_id] = permit
            return permit

    def release_mutation(self, permit: CsvMutationPermit, *,
                         ready: Callable[[], bool] | None = None) -> bool:
        """Release now, or retain ownership until supervisor readiness is true."""
        with self._lock:
            if self._permits.get(permit.owner_id) is not permit:
                return False
            if ready is not None:
                self._pending_permit_releases.setdefault(permit.owner_id, (permit, ready))
                return True
            self._pending_permit_releases.pop(permit.owner_id, None)
            del self._permits[permit.owner_id]
            return True

    def defer_mutation(self, paths: Iterable[str], callback: Callable[[], None], *,
                       ready: Callable[[], bool] | None = None) -> bool:
        """Claim future mutation rights now; callbacks run only on explicit drain."""
        normalized = self._normalize_paths(paths)
        with self._lock:
            if self._phase == "closed" or not normalized:
                return False
            if any(set(normalized).intersection(permit.paths)
                   for permit, _, _ in self._pending_mutations.values()):
                return False
            permit = CsvMutationPermit(uuid4().hex, normalized)
            self._pending_mutations[permit.owner_id] = (permit, callback, ready)
            return True

    def run_deferred_mutations(self) -> int:
        """Supervisor-only callback drain. Failures propagate with permits released."""
        with self._lock:
            releases = tuple(self._pending_permit_releases.values())
        for permit, ready in releases:
            if ready():
                self.release_mutation(permit)
        completed = 0
        while True:
            with self._lock:
                candidates = tuple(self._pending_mutations.values())
            selected = None
            for permit, callback, ready in candidates:
                # Read another service's lease only outside our lock. The pending
                # claim prevents new path owners during this readiness check.
                if ready is not None and not ready():
                    continue
                with self._lock:
                    if (permit.owner_id in self._pending_mutations
                            and not self._paths_busy(permit.paths, include_pending=False)):
                        del self._pending_mutations[permit.owner_id]
                        self._permits[permit.owner_id] = permit
                        selected = (permit, callback)
                        break
            if selected is None:
                return completed
            permit, callback = selected
            try:
                callback()
                completed += 1
            finally:
                self.release_mutation(permit)
