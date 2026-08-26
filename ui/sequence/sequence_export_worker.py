"""One-shot worker for immutable per-record and spool-rebuild export jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ui.sequence.sequence_export_model import (
    ExportJob,
    RecordExportWork,
    SpoolRebuildJob,
    SpoolTarget,
)
from ui.sequence.sequence_export_service import (
    ExportExecutionOutcome,
    ExportTargetFailure,
    ExportTargetResult,
    SequenceExportService,
)


def _safe_error_text(error: BaseException) -> str:
    try:
        text = str(error)
    except BaseException:
        return "export worker failed"
    return text[:1024] if text else "export worker failed"


class SequenceExportWorker(QObject):
    """Run exactly one blocking export attempt and emit exactly one terminal."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        job: ExportJob | RecordExportWork | SpoolRebuildJob,
        attempt_id: str,
        *,
        execute: Callable[[Any, str], Any],
        validate_dirty_checkpoint: Callable[[Any, Any], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(job, (ExportJob, RecordExportWork, SpoolRebuildJob)):
            raise TypeError("export worker requires an immutable export job")
        if type(attempt_id) is not str or not attempt_id:
            raise ValueError("attempt_id must be non-empty text")
        self.job = job
        self.attempt_id = attempt_id
        self.execute = execute
        self.validate_dirty_checkpoint = validate_dirty_checkpoint
        self._terminal_emitted = False

    def _record_id(self) -> str:
        try:
            value = getattr(self.job, "record_id", None)
            if value is None:
                value = getattr(self.job.target, "file_path", "unknown")
            return str(value or "unknown")[:1024]
        except BaseException:
            return "unknown"

    @staticmethod
    def _freeze_target_result(item: Any, *, failure: bool) -> Any:
        target_type = str(getattr(item, "target_type", "unknown"))[:128]
        config_name = str(getattr(item, "config_name", "unknown"))[:256]
        message = str(getattr(item, "message", ""))[:1024]
        result_type = ExportTargetFailure if failure else ExportTargetResult
        return result_type(target_type, config_name, message)

    @staticmethod
    def _target_identity(target: Any) -> tuple[str, str]:
        if not isinstance(target, Mapping):
            raise ValueError("record checkpoint target is not a mapping")
        target_type = str(target.get("type") or "").lower()
        config_name = str(
            target.get("config_name") or target_type or "unknown"
        )
        return target_type, config_name

    def _freeze_record_outcome(
        self, outcome: Any
    ) -> ExportExecutionOutcome:
        logical_job = (
            self.job.logical_job
            if type(self.job) is RecordExportWork
            else self.job
        )
        requested = (
            self.job.target_indices
            if type(self.job) is RecordExportWork
            else tuple(range(len(logical_job.target_configurations)))
        )
        results = tuple(
            self._freeze_target_result(item, failure=False)
            for item in tuple(getattr(outcome, "target_results", ()))
        )
        failures = tuple(
            self._freeze_target_result(item, failure=True)
            for item in tuple(getattr(outcome, "failures", ()))
        )
        completed = tuple(getattr(outcome, "completed_target_indices", ()))
        failed = tuple(getattr(outcome, "failed_target_indices", ()))
        ok = getattr(outcome, "ok")
        general_failure = bool(
            ok is not True
            and not completed
            and not failed
            and len(failures) == 1
            and failures[0].target_type == "snapshot"
        )
        if (
            any(type(index) is not int for index in completed + failed)
            or len(set(completed + failed)) != len(completed) + len(failed)
            or len(results) != len(completed)
            or (len(failures) != len(failed) and not general_failure)
        ):
            raise ValueError("record export checkpoint shape is invalid")
        requested_set = set(requested)
        if any(index not in requested_set for index in completed + failed):
            raise ValueError("record export checkpoint is outside requested work")
        positions = {index: position for position, index in enumerate(requested)}
        if any(
            sequence != tuple(sorted(sequence, key=positions.__getitem__))
            for sequence in (completed, failed)
        ):
            raise ValueError("record export checkpoint indices are out of order")
        attempted = tuple(
            sorted(completed + failed, key=positions.__getitem__)
        )
        if attempted != requested[: len(attempted)]:
            raise ValueError("record export checkpoint is not a requested prefix")
        for index, item in zip(completed, results):
            if (item.target_type, item.config_name) != self._target_identity(
                logical_job.target_configurations[index]
            ):
                raise ValueError("record export result identity is invalid")
        for index, item in zip(failed, failures):
            if (item.target_type, item.config_name) != self._target_identity(
                logical_job.target_configurations[index]
            ):
                raise ValueError("record export failure identity is invalid")
        if ok is True:
            if failures or failed or attempted != requested:
                raise ValueError("successful record checkpoint is incomplete")
        elif not failures:
            raise ValueError("failed record checkpoint has no target failure")
        elif not general_failure:
            mes_failed = any(item.target_type == "mes" for item in failures)
            if mes_failed:
                if not attempted or attempted[-1] not in failed:
                    raise ValueError("MES failure checkpoint is inconsistent")
            elif attempted != requested:
                raise ValueError("Excel failure checkpoint is incomplete")
        dirty_targets = tuple(getattr(outcome, "dirty_targets", ()))
        dirty_indices = tuple(
            getattr(outcome, "dirty_target_indices", ())
        )
        if any(
            type(index) is not int for index in dirty_indices
        ) or (
            len(dirty_targets) != len(dirty_indices)
            or len(set(dirty_indices)) != len(dirty_indices)
        ):
            raise ValueError("record export dirty target checkpoint is invalid")
        if dirty_targets:
            for index, target in zip(dirty_indices, dirty_targets):
                if (
                    index not in completed
                    or type(target) is not SpoolTarget
                    or self._target_identity(
                        logical_job.target_configurations[index]
                    )[0]
                    != "excel"
                ):
                    raise ValueError(
                        "record export dirty target checkpoint is invalid"
                    )
            validator = self.validate_dirty_checkpoint
            if (
                getattr(validator, "__self__", None).__class__
                is not SequenceExportService
                or getattr(validator, "__func__", None)
                is not SequenceExportService.validate_dirty_checkpoint
                or validator(self.job, outcome) is not True
            ):
                raise ValueError(
                    "record export dirty target provenance is untrusted"
                )
        return ExportExecutionOutcome(
            ok is True,
            self.job.job_id,
            self.attempt_id,
            str(getattr(outcome, "record_id", self._record_id()))[:1024],
            results,
            failures,
            dirty_targets,
            completed,
            failed,
            dirty_indices,
        )

    def _freeze_outcome(self, outcome: Any) -> ExportExecutionOutcome:
        job_id = getattr(outcome, "job_id")
        attempt_id = getattr(outcome, "attempt_id")
        if job_id != self.job.job_id or attempt_id != self.attempt_id:
            raise RuntimeError("export worker returned a mismatched identity")
        if isinstance(self.job, (ExportJob, RecordExportWork)):
            return self._freeze_record_outcome(outcome)
        ok = getattr(outcome, "ok")
        record_id = str(getattr(outcome, "record_id", self._record_id()))[:1024]
        target_results = tuple(
            self._freeze_target_result(item, failure=False)
            for item in tuple(getattr(outcome, "target_results", ()))
        )
        failures = tuple(
            self._freeze_target_result(item, failure=True)
            for item in tuple(getattr(outcome, "failures", ()))
        )
        dirty_targets = tuple(getattr(outcome, "dirty_targets", ()))
        completed_target_indices = tuple(
            index
            for index in tuple(
                getattr(outcome, "completed_target_indices", ())
            )
            if type(index) is int and index >= 0
        )
        if ok is not True and not failures:
            failures = (
                ExportTargetFailure(
                    "worker", "worker", "export worker reported failure"
                ),
            )
        return ExportExecutionOutcome(
            ok is True,
            self.job.job_id,
            self.attempt_id,
            record_id,
            target_results,
            failures,
            dirty_targets,
            completed_target_indices,
            (),
            (),
        )

    def _failure_outcome(self, error: BaseException) -> ExportExecutionOutcome:
        return ExportExecutionOutcome(
            False,
            self.job.job_id,
            self.attempt_id,
            self._record_id(),
            (),
            (
                ExportTargetFailure(
                    "worker", "worker", _safe_error_text(error)
                ),
            ),
            (),
            (),
            (),
            (),
        )

    @pyqtSlot()
    def run(self) -> None:
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        try:
            outcome = self._freeze_outcome(
                self.execute(self.job, self.attempt_id)
            )
        except BaseException as error:
            outcome = self._failure_outcome(error)
        try:
            if outcome.ok is True:
                self.completed.emit(outcome)
            else:
                self.failed.emit(outcome)
        except BaseException:
            # Terminal identity is sealed before emission. A hostile receiver
            # must not unwind the worker entry point or cause a second terminal.
            pass
        finally:
            try:
                self.finished.emit()
            except BaseException:
                pass


__all__ = ["SequenceExportWorker"]
