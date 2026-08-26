"""Main-thread state and immutable values for sequence result export."""

from __future__ import annotations

import os
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from ui.sequence.sequence_messages import ExportRequested


def immutable_export_value(value: Any) -> Any:
    """Detach mutable values before they enter an export job."""
    if isinstance(value, np.ndarray):
        detached = np.array(value, copy=True)
        detached.setflags(write=False)
        return detached
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                immutable_export_value(key): immutable_export_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(immutable_export_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(immutable_export_value(item) for item in value)
    if isinstance(value, PurePath):
        return type(value)(value)
    return value


def mutable_export_value(value: Any) -> Any:
    """Create a private mutable copy for legacy file exporters."""
    if isinstance(value, Mapping):
        return {
            mutable_export_value(key): mutable_export_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [mutable_export_value(item) for item in value]
    if isinstance(value, frozenset):
        return {mutable_export_value(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


class ExportJobKind(Enum):
    RECORD = auto()
    REBUILD = auto()


class ExportAttemptState(Enum):
    ACTIVE = auto()
    FAILED_AWAITING_DECISION = auto()
    COMPLETED = auto()


@dataclass(frozen=True, slots=True)
class ExportJob:
    kind: ExportJobKind
    job_id: str
    record_id: str
    result_snapshot: Any
    target_configurations: tuple[Any, ...]

    @classmethod
    def from_request(cls, request: ExportRequested) -> "ExportJob":
        if type(request) is not ExportRequested:
            raise TypeError("request must be ExportRequested")
        return cls(
            ExportJobKind.RECORD,
            request.job_id,
            request.record_id,
            immutable_export_value(request.result_snapshot),
            tuple(immutable_export_value(request.target_configurations)),
        )


@dataclass(frozen=True, slots=True)
class RecordExportWork:
    """One immutable attempt view over an immutable logical record job."""

    logical_job: ExportJob
    target_indices: tuple[int, ...]

    @property
    def kind(self) -> ExportJobKind:
        return self.logical_job.kind

    @property
    def job_id(self) -> str:
        return self.logical_job.job_id

    @property
    def record_id(self) -> str:
        return self.logical_job.record_id

    @property
    def result_snapshot(self) -> Any:
        return self.logical_job.result_snapshot

    @property
    def target_configurations(self) -> tuple[Any, ...]:
        targets = self.logical_job.target_configurations
        return tuple(targets[index] for index in self.target_indices)


@dataclass(frozen=True, slots=True)
class ExportAttempt:
    job_id: str
    attempt_id: str
    attempt_number: int


@dataclass(frozen=True, slots=True)
class SpoolTarget:
    key: tuple[str, str]
    config_name: str
    configuration: Any
    file_path: str
    spool_dir: str

    @classmethod
    def create(
        cls,
        config_name: str,
        configuration: Any,
        file_path: str,
        spool_dir: str,
    ) -> "SpoolTarget":
        name = str(config_name or "")
        output = str(file_path or "")
        spool = str(spool_dir or "")
        if not name or not output or not spool:
            raise ValueError("spool target requires a name and resolved paths")
        key = (name, os.path.normcase(os.path.abspath(output)))
        return cls(
            key,
            name,
            immutable_export_value(configuration),
            output,
            spool,
        )


@dataclass(frozen=True, slots=True)
class SpoolRebuildJob:
    kind: ExportJobKind
    job_id: str
    attempt_id: str
    target: SpoolTarget
    generation: int
    attempt_number: int = 1


@dataclass(slots=True)
class _TargetState:
    target: SpoolTarget
    dirty_generation: int = 0
    completed_generation: int = 0
    active: SpoolRebuildJob | None = None
    failed: SpoolRebuildJob | None = None
    failure: Any = None


class SequenceExportModel:
    """Own FIFO record jobs and target-keyed rebuild generations."""

    def __init__(self, *, history_limit: int = 128) -> None:
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        self.history_limit = history_limit
        self._record_queue: deque[ExportJob] = deque()
        self.active_record_job: ExportJob | None = None
        self.active_record_attempt: ExportAttempt | None = None
        self.record_attempt_state: ExportAttemptState | None = None
        self.record_attempt_number = 0
        self.record_failure: Any = None
        self._record_completed_target_indices: set[int] = set()
        self._record_target_results: dict[int, Any] = {}
        self._record_target_failures: dict[int, Any] = {}
        self._record_dirty_targets: OrderedDict[
            tuple[str, str], SpoolTarget
        ] = OrderedDict()
        self._record_job_nonce = ""
        self.record_cancel_pending = False
        self.record_cancel_reason = ""
        self._retired_attempts: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._completed_jobs: OrderedDict[str, None] = OrderedDict()
        self._retained_result_snapshots: OrderedDict[str, Any] = OrderedDict()
        self._export_preparations: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._cancelled_export_preparations: OrderedDict[str, int] = OrderedDict()
        self._targets: OrderedDict[tuple[str, str], _TargetState] = OrderedDict()
        self.shutdown_flush_pending = False
        self.shutdown_flush_generation: int | None = None
        self.shutdown_flush_failures: tuple[Any, ...] = ()
        self.shutdown_flush_final_started = False
        self.shutdown_flush_terminal = False
        self.shutdown_flush_failure_identity: tuple[str, str] | None = None
        self.shutdown_flush_completion_identity: tuple[int, str, str] | None = None
        self._rebuild_number = 0

    def retain_result_snapshot(self, record_id: str, snapshot: Any) -> bool:
        """Keep the exact frozen analysis result associated with one record."""
        if type(record_id) is not str or not record_id:
            return False
        if not isinstance(snapshot, Mapping):
            return False
        handoff = snapshot.get("export_handoff")
        canonical_record_id = (
            handoff.get("record_id")
            if isinstance(handoff, Mapping)
            else snapshot.get("record_id")
        )
        if canonical_record_id is None:
            canonical_record_id = snapshot.get("record_id")
        if canonical_record_id is not None and canonical_record_id != record_id:
            return False
        frozen = immutable_export_value(snapshot)
        self._retained_result_snapshots[record_id] = frozen
        self._retained_result_snapshots.move_to_end(record_id)
        if len(self._retained_result_snapshots) > self.history_limit:
            self._retained_result_snapshots.popitem(last=False)
        return True

    def retained_result_snapshot(self, record_id: str) -> Any:
        if type(record_id) is not str or not record_id:
            return None
        return self._retained_result_snapshots.get(record_id)

    def prepared_export_response(self, request: Any) -> Any:
        """Return the terminal cached for an exact retried preparation request."""
        request_id = getattr(request, "request_id", None)
        cached = self._export_preparations.get(request_id)
        if cached is None:
            return None
        cached_request, response = cached
        if cached_request is not request:
            return False
        self._export_preparations.move_to_end(request_id)
        return response

    def remember_export_preparation(self, request: Any, response: Any) -> bool:
        request_id = getattr(request, "request_id", None)
        if type(request_id) is not str or not request_id:
            return False
        cached = self._export_preparations.get(request_id)
        if cached is not None:
            return cached[0] is request and cached[1] is response
        self._export_preparations[request_id] = (request, response)
        if len(self._export_preparations) > self.history_limit:
            self._export_preparations.popitem(last=False)
        return True

    def replace_export_preparation_response(
        self, request: Any, expected_response: Any, replacement: Any
    ) -> bool:
        """Replace one exact cached response without changing history order."""
        request_id = getattr(request, "request_id", None)
        if type(request_id) is not str or not request_id:
            return False
        cached = self._export_preparations.get(request_id)
        if (
            cached is None
            or cached[0] is not request
            or cached[1] is not expected_response
        ):
            return False
        self._export_preparations[request_id] = (request, replacement)
        return True

    def cancel_export_preparation(self, request_id: str, generation: int) -> bool:
        if type(request_id) is not str or not request_id or type(generation) is not int:
            return False
        current = self._cancelled_export_preparations.get(request_id)
        if current is not None:
            return current == generation
        self._cancelled_export_preparations[request_id] = generation
        if len(self._cancelled_export_preparations) > self.history_limit:
            self._cancelled_export_preparations.popitem(last=False)
        return True

    def is_export_preparation_cancelled(self, request: Any) -> bool:
        return self._cancelled_export_preparations.get(
            getattr(request, "request_id", None)
        ) == getattr(request, "workflow_generation", None)

    @property
    def queued_record_count(self) -> int:
        return len(self._record_queue)

    @property
    def target_state_count(self) -> int:
        return len(self._targets)

    def enqueue_record(self, request: ExportRequested) -> ExportJob:
        job = ExportJob.from_request(request)
        if (
            self.active_record_job is not None
            and self.active_record_job.job_id == job.job_id
        ) or any(queued.job_id == job.job_id for queued in self._record_queue):
            raise ValueError("export job identifier is already active or queued")
        if job.job_id in self._completed_jobs:
            raise ValueError("export job identifier is retired")
        self._record_queue.append(job)
        return job

    def begin_next_record_job(self) -> ExportJob | None:
        if self.active_record_job is not None or not self._record_queue:
            return None
        self.active_record_job = self._record_queue.popleft()
        self.active_record_attempt = None
        self.record_attempt_state = None
        self.record_attempt_number = 0
        self.record_failure = None
        self._record_completed_target_indices.clear()
        self._record_target_results.clear()
        self._record_target_failures.clear()
        self._record_dirty_targets.clear()
        self._record_job_nonce = uuid4().hex
        self.record_cancel_pending = False
        self.record_cancel_reason = ""
        return self.active_record_job

    def begin_record_attempt(
        self, attempt_id_factory: Callable[[str, int], str]
    ) -> ExportAttempt:
        job = self.active_record_job
        if job is None:
            raise RuntimeError("no active record export job")
        if self.active_record_attempt is not None:
            raise RuntimeError("record export attempt is already active")
        next_number = self.record_attempt_number + 1
        hint = attempt_id_factory(job.job_id, next_number)
        attempt_id = self._record_attempt_identity(hint, next_number)
        attempt = ExportAttempt(job.job_id, attempt_id, next_number)
        self.record_attempt_number = next_number
        self.active_record_attempt = attempt
        self.record_attempt_state = ExportAttemptState.ACTIVE
        self.record_failure = None
        return attempt

    def begin_record_recovery_attempt(self) -> ExportAttempt:
        """Create an instance-owned identity after an untrusted ID factory fails."""
        job = self.active_record_job
        if job is None or self.active_record_attempt is not None:
            raise RuntimeError("record recovery attempt is unavailable")
        next_number = self.record_attempt_number + 1
        attempt_id = self._record_attempt_identity("recovery", next_number)
        attempt = ExportAttempt(job.job_id, attempt_id, next_number)
        self.record_attempt_number = next_number
        self.active_record_attempt = attempt
        self.record_attempt_state = ExportAttemptState.ACTIVE
        self.record_failure = None
        return attempt

    def _record_attempt_identity(self, hint: Any, number: int) -> str:
        """Create an instance/job-owned opaque identity from an untrusted hint."""
        if type(hint) is not str or not hint:
            raise ValueError("attempt identifier factory must return non-empty text")
        if not self._record_job_nonce:
            raise RuntimeError("record attempt identity has no active job nonce")
        # Every attempt is structurally unique from the immutable job nonce and
        # monotonic number; the external factory contributes a readable hint
        # but never controls uniqueness.
        return f"{hint}::{self._record_job_nonce}:{number}"

    def active_record_work(self) -> RecordExportWork:
        job = self.active_record_job
        attempt = self.active_record_attempt
        if (
            job is None
            or attempt is None
            or self.record_attempt_state is not ExportAttemptState.ACTIVE
        ):
            raise RuntimeError("no active record export attempt")
        pending = tuple(
            index
            for index in range(len(job.target_configurations))
            if index not in self._record_completed_target_indices
        )
        return RecordExportWork(job, pending)

    def accept_worker_terminal(self, job_id: str, attempt_id: str) -> bool:
        attempt = self.active_record_attempt
        return bool(
            attempt is not None
            and attempt.job_id == job_id
            and attempt.attempt_id == attempt_id
            and self.record_attempt_state is ExportAttemptState.ACTIVE
            and (job_id, attempt_id) not in self._retired_attempts
        )

    def _remember_record_progress(
        self,
        completed_target_indices: Any,
        target_results: Any = (),
        failed_target_indices: Any = (),
        failures: Any = (),
        dirty_targets: Any = (),
    ) -> None:
        job = self.active_record_job
        if job is None:
            return
        completed = tuple(completed_target_indices or ())
        results = tuple(target_results or ())
        failed = tuple(failed_target_indices or ())
        failure_values = tuple(failures or ())
        for index, result in zip(completed, results):
            if type(index) is int and 0 <= index < len(job.target_configurations):
                self._record_completed_target_indices.add(index)
                self._record_target_results[index] = immutable_export_value(result)
                self._record_target_failures.pop(index, None)
        for index, failure in zip(failed, failure_values):
            if type(index) is int and 0 <= index < len(job.target_configurations):
                self._record_target_failures[index] = immutable_export_value(
                    failure
                )
        for target in tuple(dirty_targets or ()):
            if type(target) is SpoolTarget:
                self._record_dirty_targets[target.key] = target

    @staticmethod
    def _target_identity(target: Any) -> tuple[str, str]:
        if isinstance(target, Mapping):
            target_type = str(target.get("type") or "unknown").lower()
            config_name = str(
                target.get("config_name") or target_type or "unknown"
            )
            return target_type, config_name
        return "unknown", "unknown"

    def record_terminal_results(self, *, ignored: bool = False) -> tuple[Any, ...]:
        job = self.active_record_job
        if job is None:
            return ()
        results: list[Any] = []
        for index, target in enumerate(job.target_configurations):
            if index in self._record_target_results:
                results.append(self._record_target_results[index])
                continue
            target_type, config_name = self._target_identity(target)
            failure = self._record_target_failures.get(index)
            if failure is not None:
                results.append(
                    immutable_export_value(
                        {
                            "target": target_type,
                            "config_name": config_name,
                            "message": str(
                                getattr(failure, "message", "export failed")
                            ),
                            "ignored": ignored,
                            "attempted": True,
                        }
                    )
                )
            elif ignored:
                results.append(
                    immutable_export_value(
                        {
                            "target": target_type,
                            "config_name": config_name,
                            "message": "not attempted because a prerequisite failed",
                            "ignored": True,
                            "attempted": False,
                        }
                    )
                )
        return tuple(results)

    def record_dirty_targets(self) -> tuple[SpoolTarget, ...]:
        return tuple(self._record_dirty_targets.values())

    def complete_record_attempt(
        self,
        job_id: str,
        attempt_id: str,
        completed_target_indices: Any = (),
        target_results: Any = (),
        dirty_targets: Any = (),
    ) -> bool:
        if not self.accept_worker_terminal(job_id, attempt_id):
            return False
        self._remember_record_progress(
            completed_target_indices,
            target_results,
            dirty_targets=dirty_targets,
        )
        self.record_attempt_state = ExportAttemptState.COMPLETED
        return True

    def fail_record_attempt(
        self,
        job_id: str,
        attempt_id: str,
        failures: Any,
        completed_target_indices: Any = (),
        target_results: Any = (),
        failed_target_indices: Any = (),
        dirty_targets: Any = (),
    ) -> bool:
        if not self.accept_worker_terminal(job_id, attempt_id):
            return False
        self._remember_record_progress(
            completed_target_indices,
            target_results,
            failed_target_indices,
            failures,
            dirty_targets,
        )
        self.record_failure = immutable_export_value(failures)
        self.record_attempt_state = ExportAttemptState.FAILED_AWAITING_DECISION
        return True

    def retry_record_attempt(
        self,
        job_id: str,
        attempt_id: str,
        attempt_id_factory: Callable[[str, int], str],
    ) -> ExportAttempt | None:
        if (
            self.record_failure is None
            or self.record_attempt_state
            is not ExportAttemptState.FAILED_AWAITING_DECISION
            or self.active_record_attempt is None
            or self.active_record_attempt.job_id != job_id
            or self.active_record_attempt.attempt_id != attempt_id
        ):
            return None
        next_number = self.record_attempt_number + 1
        next_hint = attempt_id_factory(job_id, next_number)
        next_attempt_id = self._record_attempt_identity(next_hint, next_number)
        self._retire_attempt(job_id, attempt_id)
        attempt = ExportAttempt(job_id, next_attempt_id, next_number)
        self.active_record_attempt = attempt
        self.record_attempt_state = ExportAttemptState.ACTIVE
        self.record_attempt_number = next_number
        self.record_failure = None
        return attempt

    def rollback_record_retry(
        self,
        previous_attempt: ExportAttempt,
        new_attempt: ExportAttempt,
        previous_failure: Any,
    ) -> bool:
        """Restore the exact failed decision if Workflow rejects retry ack."""
        active = self.active_record_attempt
        if (
            active != new_attempt
            or self.record_attempt_state is not ExportAttemptState.ACTIVE
            or previous_attempt.job_id != new_attempt.job_id
        ):
            return False
        self._retire_attempt(new_attempt.job_id, new_attempt.attempt_id)
        self._retired_attempts.pop(
            (previous_attempt.job_id, previous_attempt.attempt_id), None
        )
        self.active_record_attempt = previous_attempt
        self.record_attempt_state = ExportAttemptState.FAILED_AWAITING_DECISION
        self.record_failure = immutable_export_value(previous_failure)
        return True

    def ignore_record_failure(
        self, job_id: str, attempt_id: str
    ) -> ExportJob | None:
        if (
            self.record_failure is None
            or self.record_attempt_state
            is not ExportAttemptState.FAILED_AWAITING_DECISION
            or self.active_record_attempt is None
            or self.active_record_attempt.job_id != job_id
            or self.active_record_attempt.attempt_id != attempt_id
        ):
            return None
        return self.active_record_job

    def complete_record_job(self, job_id: str, attempt_id: str) -> ExportJob | None:
        job = self.active_record_job
        attempt = self.active_record_attempt
        if job is None or job.job_id != job_id:
            return None
        if attempt is not None and attempt.attempt_id != attempt_id:
            return None
        if attempt is not None:
            self._retire_attempt(job_id, attempt_id)
        self._remember_completed(job_id)
        self.active_record_job = None
        self.active_record_attempt = None
        self.record_attempt_state = None
        self.record_attempt_number = 0
        self.record_failure = None
        self._record_completed_target_indices.clear()
        self._record_target_results.clear()
        self._record_target_failures.clear()
        self._record_dirty_targets.clear()
        self._record_job_nonce = ""
        self.record_cancel_pending = False
        self.record_cancel_reason = ""
        return job

    def request_record_cancel(self, job_id: str, reason: str) -> bool:
        if self.active_record_job is None or self.active_record_job.job_id != job_id:
            return False
        self.record_cancel_pending = True
        self.record_cancel_reason = str(reason or "export cancelled")
        return True

    def _retire_attempt(self, job_id: str, attempt_id: str) -> None:
        key = (job_id, attempt_id)
        self._retired_attempts[key] = None
        self._retired_attempts.move_to_end(key)
        if len(self._retired_attempts) > self.history_limit:
            self._retired_attempts.popitem(last=False)

    def _remember_completed(self, job_id: str) -> None:
        self._completed_jobs[job_id] = None
        self._completed_jobs.move_to_end(job_id)
        if len(self._completed_jobs) > self.history_limit:
            self._completed_jobs.popitem(last=False)

    def mark_target_dirty(self, target: SpoolTarget) -> int:
        if type(target) is not SpoolTarget:
            raise TypeError("target must be SpoolTarget")
        state = self._targets.get(target.key)
        if state is None:
            state = _TargetState(target)
            self._targets[target.key] = state
        else:
            state.target = target
        state.dirty_generation += 1
        return state.dirty_generation

    def dirty_target_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            key
            for key, state in self._targets.items()
            if state.active is None
            and state.failure is None
            and state.dirty_generation > state.completed_generation
        )

    def begin_rebuild(self, key: tuple[str, str]) -> SpoolRebuildJob | None:
        state = self._targets.get(key)
        if (
            state is None
            or state.active is not None
            or state.failure is not None
            or state.dirty_generation <= state.completed_generation
        ):
            return None
        self._rebuild_number += 1
        generation = state.dirty_generation
        job_id = f"spool-rebuild-{self._rebuild_number}"
        attempt_id = f"{job_id}-attempt-1"
        job = SpoolRebuildJob(
            ExportJobKind.REBUILD,
            job_id,
            attempt_id,
            state.target,
            generation,
            1,
        )
        state.active = job
        return job

    def active_rebuild(self, key: tuple[str, str]) -> SpoolRebuildJob | None:
        state = self._targets.get(key)
        return None if state is None else state.active

    def complete_rebuild(
        self,
        job_id: str,
        attempt_id: str,
        *,
        succeeded: bool,
        failure: Any = None,
    ) -> SpoolRebuildJob | None:
        for key, state in tuple(self._targets.items()):
            active = state.active
            if (
                active is None
                or active.job_id != job_id
                or active.attempt_id != attempt_id
            ):
                continue
            state.active = None
            if succeeded:
                state.completed_generation = max(
                    state.completed_generation, active.generation
                )
                state.failure = None
                state.failed = None
            else:
                state.failure = immutable_export_value(failure)
                state.failed = active
                return None
            if state.dirty_generation > state.completed_generation:
                return self.begin_rebuild(state.target.key)
            self._targets.pop(key, None)
            return None
        return None

    def fail_rebuild_boundary(
        self, job_id: str, attempt_id: str, failure: Any
    ) -> SpoolRebuildJob | None:
        """Seal an exact rebuild after controller/worker boundary failure."""
        for state in self._targets.values():
            active = state.active
            if (
                active is not None
                and active.job_id == job_id
                and active.attempt_id == attempt_id
            ):
                state.active = None
                state.failed = active
                state.failure = immutable_export_value(failure)
                return active
            failed = state.failed
            if (
                failed is not None
                and failed.job_id == job_id
                and failed.attempt_id == attempt_id
            ):
                state.failure = immutable_export_value(failure)
                return failed
        return None

    def retry_rebuild(
        self, job_id: str, attempt_id: str
    ) -> SpoolRebuildJob | None:
        for key, state in tuple(self._targets.items()):
            failed = state.failed
            if (
                failed is None
                or failed.job_id != job_id
                or failed.attempt_id != attempt_id
                or state.active is not None
            ):
                continue
            next_number = failed.attempt_number + 1
            retried = SpoolRebuildJob(
                ExportJobKind.REBUILD,
                failed.job_id,
                f"{failed.job_id}-attempt-{next_number}",
                failed.target,
                failed.generation,
                next_number,
            )
            state.failed = None
            state.failure = None
            state.active = retried
            return retried
        return None

    def ignore_rebuild_failure(self, job_id: str, attempt_id: str) -> bool:
        for key, state in tuple(self._targets.items()):
            failed = state.failed
            if (
                failed is None
                or failed.job_id != job_id
                or failed.attempt_id != attempt_id
            ):
                continue
            state.completed_generation = max(
                state.completed_generation, failed.generation
            )
            state.failed = None
            state.failure = None
            if state.dirty_generation <= state.completed_generation:
                self._targets.pop(key, None)
            return True
        return False

    def clear_rebuild_failure(self, key: tuple[str, str]) -> bool:
        state = self._targets.get(key)
        if state is None or state.failure is None:
            return False
        state.failure = None
        state.failed = None
        if state.dirty_generation <= state.completed_generation:
            self._targets.pop(key, None)
        return True

    def tracked_spool_targets(self) -> tuple[SpoolTarget, ...]:
        return tuple(state.target for state in self._targets.values())

    def has_active_work(self) -> bool:
        return bool(
            self.active_record_job is not None
            or self._record_queue
            or any(
                state.active is not None
                or state.failed is not None
                or state.dirty_generation > state.completed_generation
                for state in self._targets.values()
            )
        )


__all__ = [
    "ExportAttempt",
    "ExportAttemptState",
    "ExportJob",
    "ExportJobKind",
    "RecordExportWork",
    "SequenceExportModel",
    "SpoolRebuildJob",
    "SpoolTarget",
    "immutable_export_value",
    "mutable_export_value",
]
