"""Durable file/application-state commit boundary for sequence recordings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from ui.sequence.sequence_recording_model import StagedRecording


@dataclass(frozen=True, slots=True)
class RecordingWarning:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class RecordingCommitResult:
    audio_committed: bool
    completed: bool
    recovery_path: Path | None
    analysis_snapshot: Mapping[str, Any] | None
    warnings: tuple[RecordingWarning, ...]
    rollback_outcome: Mapping[str, Any]
    reason: str = ""
    sample_count: int = 0
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class RecordingDurableResult:
    """Worker-owned result through the durable WAV promotion boundary."""

    staged: StagedRecording
    fields: Mapping[str, Any]
    audio_committed: bool
    recovery_path: Path | None
    warnings: tuple[RecordingWarning, ...]
    rollback_outcome: Mapping[str, Any]
    reason: str = ""
    cancelled: bool = False


class RecordingCancellationRequested(Exception):
    """Cooperative cancellation observed before the durable commit barrier."""


def _noop_staged(_staged: StagedRecording) -> None:
    return None


def _no_alignment(_staged: StagedRecording) -> Mapping[str, Any]:
    return {}


def _noop_database(_info: Mapping[str, Any], _stimulus: Any) -> None:
    return None


def _noop_count(_count: int | None) -> None:
    return None


def _noop_commit_barrier() -> None:
    return None


class RecordingTransaction:
    """Apply the spec's single durable promotion point and ordered commits."""

    _MISSING = object()

    def __init__(
        self,
        *,
        data_struct: Any,
        finalize_output: Callable[[StagedRecording], None] = _noop_staged,
        alignment_handoff: Callable[
            [StagedRecording], Mapping[str, Any]
        ] = _no_alignment,
        finalize_metadata: Callable[[StagedRecording], None] = _noop_staged,
        promote_output: Callable[[StagedRecording], None] | None = None,
        save_database: Callable[[Mapping[str, Any], Any], Any] = _noop_database,
        commit_count: Callable[[int | None], None] = _noop_count,
        persist_count: Callable[[int | None], None] = _noop_count,
        cleanup: Callable[[StagedRecording], None] | None = None,
        cancellation_checkpoint: Callable[[], None] = _noop_commit_barrier,
        begin_durable_commit: Callable[[], None] = _noop_commit_barrier,
        promotion_succeeded: Callable[[], None] = _noop_commit_barrier,
        logger: Any = None,
    ) -> None:
        self.data_struct = data_struct
        self.finalize_output = finalize_output
        self.alignment_handoff = alignment_handoff
        self.finalize_metadata = finalize_metadata
        self.promote_output = promote_output or self._promote_output
        self.save_database = save_database
        self.commit_count = commit_count
        self.persist_count = persist_count
        self.cleanup = cleanup or self._cleanup_success
        self.cancellation_checkpoint = cancellation_checkpoint
        self.begin_durable_commit = begin_durable_commit
        self.promotion_succeeded = promotion_succeeded
        self.logger = logger

    def bind_commit_barrier(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("recording commit barrier must be callable")
        self.begin_durable_commit = callback

    def bind_cancellation_checkpoint(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("recording cancellation checkpoint must be callable")
        self.cancellation_checkpoint = callback

    def bind_promotion_succeeded(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("recording promotion callback must be callable")
        self.promotion_succeeded = callback

    def _cancelled_result(
        self,
        staged: StagedRecording,
        warnings: list[RecordingWarning],
        error: RecordingCancellationRequested,
    ) -> RecordingCommitResult:
        rollback = self._rollback_precommit(staged)
        return RecordingCommitResult(
            audio_committed=False,
            completed=False,
            recovery_path=None,
            analysis_snapshot=None,
            warnings=tuple(warnings),
            rollback_outcome=rollback,
            reason=str(error) or "recording cancelled",
            sample_count=staged.sample_count,
            cancelled=True,
        )

    def _cancelled_durable_result(
        self,
        staged: StagedRecording,
        fields: Mapping[str, Any],
        warnings: list[RecordingWarning],
        error: RecordingCancellationRequested,
    ) -> RecordingDurableResult:
        rollback = self._rollback_precommit(staged)
        return RecordingDurableResult(
            staged=staged,
            fields=MappingProxyType(dict(fields)),
            audio_committed=False,
            recovery_path=None,
            warnings=tuple(warnings),
            rollback_outcome=rollback,
            reason=str(error) or "recording cancelled",
            cancelled=True,
        )

    def _warning(
        self, warnings: list[RecordingWarning], stage: str, error: Exception
    ) -> None:
        warning = RecordingWarning(stage, str(error) or type(error).__name__)
        warnings.append(warning)
        callback = getattr(self.logger, "warning", None)
        if callable(callback):
            callback(f"recording transaction {stage} warning: {warning.message}")

    @staticmethod
    def _remove_if_exists(path: Path | None) -> None:
        if path is not None and path.exists():
            path.unlink()

    def _promote_output(self, staged: StagedRecording) -> None:
        snapshot = staged.snapshot
        output_path = snapshot.output_path
        temp_path = snapshot.temp_path
        backup_path = snapshot.backup_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.replay and output_path.exists():
            if backup_path is None:
                raise RuntimeError("replay promotion requires a backup path")
            self._remove_if_exists(backup_path)
            os.replace(output_path, backup_path)
        try:
            os.replace(temp_path, output_path)
        except BaseException as error:
            if (
                snapshot.replay
                and backup_path is not None
                and backup_path.exists()
                and not output_path.exists()
            ):
                try:
                    os.replace(backup_path, output_path)
                except BaseException as restore_error:
                    error.add_note(f"replay restore failed: {restore_error}")
            raise

    def _cleanup_success(self, staged: StagedRecording) -> None:
        snapshot = staged.snapshot
        self._remove_if_exists(snapshot.backup_path)
        self._remove_if_exists(snapshot.temp_path)

    def _rollback_precommit(self, staged: StagedRecording) -> Mapping[str, Any]:
        snapshot = staged.snapshot
        errors: list[str] = []
        try:
            self._remove_if_exists(snapshot.temp_path)
        except OSError as error:
            errors.append(f"temp cleanup: {error}")
        backup = snapshot.backup_path
        try:
            if (
                snapshot.replay
                and backup is not None
                and backup.exists()
                and not snapshot.output_path.exists()
            ):
                os.replace(backup, snapshot.output_path)
        except OSError as error:
            errors.append(f"replay restore: {error}")
        return MappingProxyType(
            {"restored": not errors, "errors": tuple(errors)}
        )

    def _apply_data_struct(
        self, fields: Mapping[str, Any]
    ) -> tuple[bool, str, Mapping[str, Any]]:
        prior: dict[str, Any] = {}
        attempted: list[str] = []
        try:
            for name, value in fields.items():
                prior[name] = getattr(self.data_struct, name, self._MISSING)
                attempted.append(name)
                setattr(self.data_struct, name, value)
            return True, "", MappingProxyType(
                {"data_struct_restored": None, "data_struct_restore_errors": ()}
            )
        except BaseException as error:
            restore_errors: list[str] = []
            for name in reversed(attempted):
                old_value = prior[name]
                try:
                    if old_value is self._MISSING:
                        if hasattr(self.data_struct, name):
                            delattr(self.data_struct, name)
                    else:
                        setattr(self.data_struct, name, old_value)
                except BaseException as restore_error:
                    restore_errors.append(f"{name}: {restore_error}")
            if not isinstance(error, Exception):
                for restore_error in restore_errors:
                    error.add_note(f"DataDealStruct restore failed: {restore_error}")
                raise
            reason = str(error) or type(error).__name__
            if restore_errors:
                reason += "; restore failed: " + "; ".join(restore_errors)
            return False, reason, MappingProxyType(
                {
                    "data_struct_restored": not restore_errors,
                    "data_struct_restore_errors": tuple(restore_errors),
                }
            )

    @staticmethod
    def _numeric_array(
        value: Any, field_name: str, *, dimensions: int
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{field_name} must be a detached NumPy array")
        if value.ndim != dimensions:
            raise ValueError(
                f"{field_name} dimension mismatch: expected {dimensions}D, "
                f"got {value.ndim}D"
            )
        if not (
            np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.signedinteger)
            or np.issubdtype(value.dtype, np.unsignedinteger)
        ):
            raise ValueError(f"{field_name} must contain real numeric samples")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{field_name} contains non-finite samples")
        return value

    @classmethod
    def _validate_staged(
        cls,
        staged: StagedRecording,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        target = staged.snapshot.target_samples
        if staged.sample_count != target:
            raise ValueError(
                f"sample count mismatch: expected {target}, got {staged.sample_count}"
            )
        stored_fields = staged.data_struct_fields if fields is None else fields
        for field_name in ("store_wave_data", "store_wave_data_multi"):
            if field_name not in stored_fields:
                raise ValueError(f"sample field is missing: {field_name}")
        mono = cls._numeric_array(
            stored_fields["store_wave_data"],
            "store_wave_data",
            dimensions=1,
        )
        multi = cls._numeric_array(
            stored_fields["store_wave_data_multi"],
            "store_wave_data_multi",
            dimensions=2,
        )
        for field_name, array in (
            ("store_wave_data", mono),
            ("store_wave_data_multi", multi),
        ):
            if array.shape[0] != staged.sample_count:
                raise ValueError(
                    f"sample length mismatch for {field_name}: "
                    f"expected {staged.sample_count}, got {array.shape[0]}"
                )
        expected_channels = (
            1
            if staged.snapshot.mode == "PLAY_AND_RECORD"
            else len(staged.snapshot.input_channels)
        )
        if multi.shape[1] != expected_channels:
            raise ValueError(
                "channel layout mismatch for store_wave_data_multi: "
                f"expected {expected_channels}, got {multi.shape[1]}"
            )
        audio_length = stored_fields.get("audio_lenth")
        if audio_length is not None and (
            type(audio_length) is not int or audio_length != staged.sample_count
        ):
            raise ValueError(
                f"sample length mismatch for audio_lenth: "
                f"expected {staged.sample_count}, got {audio_length}"
            )

    @staticmethod
    def _analysis_snapshot(
        staged: StagedRecording, fields: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        snapshot = staged.snapshot
        return MappingProxyType(
            {
                "record_id": snapshot.record_id,
                "recorded_path": str(snapshot.output_path),
                "sample_rate": snapshot.sample_rate,
                "sample_count": staged.sample_count,
                "session": snapshot.as_message_payload(),
                "recorded_signal_info": staged.recorded_signal_info,
                "data_struct_fields": MappingProxyType(dict(fields)),
            }
        )

    def prepare_durable(self, staged: StagedRecording) -> RecordingDurableResult:
        """Run worker-safe validation, transformation, metadata and promotion."""
        warnings = [
            warning
            for warning in staged.warnings
            if type(warning) is RecordingWarning
        ]
        fields = dict(staged.data_struct_fields)
        try:
            self.cancellation_checkpoint()
            self.finalize_output(staged)
            self.cancellation_checkpoint()
            alignment_fields = self.alignment_handoff(staged)
            if alignment_fields:
                fields.update(dict(alignment_fields))
            self.cancellation_checkpoint()
            self._validate_staged(staged, fields)
            try:
                self.finalize_metadata(staged)
            except Exception as error:
                self._warning(warnings, "metadata", error)
            self.cancellation_checkpoint()
        except RecordingCancellationRequested as error:
            return self._cancelled_durable_result(
                staged, fields, warnings, error
            )
        except Exception as error:
            # Cancellation is allowed to win only while the transaction is still
            # precommit. This checkpoint is deliberately never called after the
            # promotion barrier has been entered.
            try:
                self.cancellation_checkpoint()
            except RecordingCancellationRequested as cancellation:
                return self._cancelled_durable_result(
                    staged, fields, warnings, cancellation
                )
            rollback = self._rollback_precommit(staged)
            return RecordingDurableResult(
                staged=staged,
                fields=MappingProxyType(dict(fields)),
                audio_committed=False,
                recovery_path=None,
                warnings=tuple(warnings),
                rollback_outcome=rollback,
                reason=str(error) or type(error).__name__,
            )
        except BaseException as error:
            try:
                self._rollback_precommit(staged)
            except BaseException as rollback_error:
                error.add_note(f"precommit rollback failed: {rollback_error}")
            raise

        try:
            self.begin_durable_commit()
        except RecordingCancellationRequested as error:
            return self._cancelled_durable_result(
                staged, fields, warnings, error
            )
        except Exception as error:
            rollback = self._rollback_precommit(staged)
            return RecordingDurableResult(
                staged=staged,
                fields=MappingProxyType(dict(fields)),
                audio_committed=False,
                recovery_path=None,
                warnings=tuple(warnings),
                rollback_outcome=rollback,
                reason=str(error) or type(error).__name__,
            )
        except BaseException as error:
            try:
                self._rollback_precommit(staged)
            except BaseException as rollback_error:
                error.add_note(f"precommit rollback failed: {rollback_error}")
            raise

        try:
            self.promote_output(staged)
        except Exception as error:
            rollback = self._rollback_precommit(staged)
            return RecordingDurableResult(
                staged=staged,
                fields=MappingProxyType(dict(fields)),
                audio_committed=False,
                recovery_path=None,
                warnings=tuple(warnings),
                rollback_outcome=rollback,
                reason=str(error) or type(error).__name__,
            )
        except BaseException as error:
            try:
                self._rollback_precommit(staged)
            except BaseException as rollback_error:
                error.add_note(f"promotion rollback failed: {rollback_error}")
            raise

        self.promotion_succeeded()
        return RecordingDurableResult(
            staged=staged,
            fields=MappingProxyType(dict(fields)),
            audio_committed=True,
            recovery_path=staged.snapshot.output_path,
            warnings=tuple(warnings),
            rollback_outcome=MappingProxyType({"audio_committed": True}),
        )

    def apply_data_struct_projection(
        self, durable: RecordingDurableResult
    ) -> RecordingCommitResult | None:
        if not durable.audio_committed:
            return RecordingCommitResult(
                audio_committed=False,
                completed=False,
                recovery_path=None,
                analysis_snapshot=None,
                warnings=durable.warnings,
                rollback_outcome=durable.rollback_outcome,
                reason=durable.reason,
                sample_count=durable.staged.sample_count,
                cancelled=durable.cancelled,
            )
        applied, apply_reason, data_struct_rollback = self._apply_data_struct(
            durable.fields
        )
        if not applied:
            return RecordingCommitResult(
                audio_committed=True,
                completed=False,
                recovery_path=durable.recovery_path,
                analysis_snapshot=None,
                warnings=durable.warnings,
                rollback_outcome=data_struct_rollback,
                reason=apply_reason,
                sample_count=durable.staged.sample_count,
            )
        return None

    def precommit_launch_failure(
        self, staged: StagedRecording, error: Exception
    ) -> RecordingCommitResult:
        """Rollback an attempt whose pre-promotion worker never started."""
        rollback = self._rollback_precommit(staged)
        return RecordingCommitResult(
            audio_committed=False,
            completed=False,
            recovery_path=None,
            analysis_snapshot=None,
            warnings=(),
            rollback_outcome=rollback,
            reason=str(error) or type(error).__name__,
            sample_count=staged.sample_count,
        )

    def with_worker_warning(
        self,
        durable: RecordingDurableResult,
        stage: str,
        error: Exception,
    ) -> RecordingDurableResult:
        warnings = list(durable.warnings)
        self._warning(warnings, stage, error)
        return RecordingDurableResult(
            staged=durable.staged,
            fields=durable.fields,
            audio_committed=durable.audio_committed,
            recovery_path=durable.recovery_path,
            warnings=tuple(warnings),
            rollback_outcome=durable.rollback_outcome,
            reason=durable.reason,
            cancelled=durable.cancelled,
        )

    def save_database_worker(
        self, durable: RecordingDurableResult
    ) -> RecordingDurableResult:
        warnings = list(durable.warnings)
        staged = durable.staged
        try:
            self.save_database(
                staged.recorded_signal_info, staged.stimulus_info
            )
        except Exception as error:
            self._warning(warnings, "database", error)
        return RecordingDurableResult(
            staged=staged,
            fields=durable.fields,
            audio_committed=durable.audio_committed,
            recovery_path=durable.recovery_path,
            warnings=tuple(warnings),
            rollback_outcome=durable.rollback_outcome,
            reason=durable.reason,
            cancelled=durable.cancelled,
        )

    def apply_count_projection(
        self, durable: RecordingDurableResult
    ) -> RecordingCommitResult | None:
        staged = durable.staged
        try:
            self.commit_count(staged.snapshot.pending_count)
        except Exception as error:
            return RecordingCommitResult(
                audio_committed=True,
                completed=False,
                recovery_path=staged.snapshot.output_path,
                analysis_snapshot=None,
                warnings=durable.warnings,
                rollback_outcome=MappingProxyType(
                    {"data_struct_committed": True, "count_committed": False}
                ),
                reason=str(error) or type(error).__name__,
                sample_count=staged.sample_count,
            )
        return None

    def persist_and_cleanup_worker(
        self, durable: RecordingDurableResult
    ) -> RecordingCommitResult:
        warnings = list(durable.warnings)
        staged = durable.staged
        try:
            self.persist_count(staged.snapshot.pending_count)
        except Exception as error:
            self._warning(warnings, "count-persistence", error)

        try:
            self.cleanup(staged)
        except Exception as error:
            self._warning(warnings, "cleanup", error)

        return self._completed_result(durable, warnings)

    def persistence_launch_failure(
        self, durable: RecordingDurableResult, error: Exception
    ) -> RecordingCommitResult:
        """Complete truthfully when the optional persistence worker cannot run."""
        warnings = list(durable.warnings)
        if durable.staged.snapshot.pending_count is not None:
            self._warning(warnings, "count-persistence", error)
        self._warning(warnings, "cleanup", error)
        return self._completed_result(durable, warnings)

    def _completed_result(
        self,
        durable: RecordingDurableResult,
        warnings: list[RecordingWarning],
    ) -> RecordingCommitResult:
        staged = durable.staged
        analysis_snapshot = dict(
            self._analysis_snapshot(staged, durable.fields)
        )
        analysis_snapshot["warnings"] = tuple(
            MappingProxyType(
                {"stage": warning.stage, "message": warning.message}
            )
            for warning in warnings
        )
        immutable_snapshot = MappingProxyType(analysis_snapshot)
        return RecordingCommitResult(
            audio_committed=True,
            completed=True,
            recovery_path=staged.snapshot.output_path,
            analysis_snapshot=immutable_snapshot,
            warnings=tuple(warnings),
            rollback_outcome=MappingProxyType(
                {"data_struct_committed": True, "count_committed": True}
            ),
            sample_count=staged.sample_count,
        )

    def commit(self, staged: StagedRecording) -> RecordingCommitResult:
        """Synchronous compatibility pipeline used by direct transaction callers."""
        durable = self.prepare_durable(staged)
        projection_failure = self.apply_data_struct_projection(durable)
        if projection_failure is not None:
            return projection_failure
        durable = self.save_database_worker(durable)
        count_failure = self.apply_count_projection(durable)
        if count_failure is not None:
            return count_failure
        return self.persist_and_cleanup_worker(durable)
