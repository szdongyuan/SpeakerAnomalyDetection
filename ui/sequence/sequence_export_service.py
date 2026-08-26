"""Pure synchronous export operations used by the sequence export worker."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from base.excel_result_exporter import (
    build_excel_from_csv_spool,
    export_analysis_to_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
    resolve_excel_spool_dir,
)
from base.mes_result_exporter import _validate_mes_runtime_config, write_mes_result
from ui.sequence.sequence_export_model import (
    ExportJob,
    ExportJobKind,
    RecordExportWork,
    SpoolRebuildJob,
    SpoolTarget,
    immutable_export_value,
    mutable_export_value,
)


_MISSING_ANALYSIS_CONFIGURATION = object()


def _bounded_error_text(error: BaseException, fallback: str) -> str:
    try:
        text = str(error)
    except BaseException:
        return fallback
    return text[:1024] if text else fallback


@dataclass(frozen=True, slots=True)
class ExportTargetResult:
    target_type: str
    config_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ExportTargetFailure:
    target_type: str
    config_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ExportExecutionOutcome:
    ok: bool
    job_id: str
    attempt_id: str
    record_id: str
    target_results: tuple[ExportTargetResult, ...]
    failures: tuple[ExportTargetFailure, ...]
    dirty_targets: tuple[SpoolTarget, ...]
    completed_target_indices: tuple[int, ...] = ()
    failed_target_indices: tuple[int, ...] = ()
    dirty_target_indices: tuple[int, ...] = ()
    dirty_provenance: Any = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RebuildExecutionOutcome:
    ok: bool
    job_id: str
    attempt_id: str
    record_id: str
    target_results: tuple[ExportTargetResult, ...]
    failures: tuple[ExportTargetFailure, ...]
    dirty_targets: tuple[SpoolTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class _FrozenDirtyTarget:
    index: int
    raw_target: Any
    target_type: str
    config_name: str
    fast_mode: bool
    target: SpoolTarget


@dataclass(frozen=True, slots=True)
class _DirtyCheckpointProvenance:
    capability: Any = field(repr=False, compare=False)
    token: Any = field(repr=False, compare=False)
    job_id: str
    attempt_id: str
    requested_indices: tuple[int, ...]
    entries: tuple[_FrozenDirtyTarget, ...]


@dataclass(frozen=True, slots=True)
class _TrustedDirtyCheckpoint:
    work: ExportJob | RecordExportWork = field(repr=False, compare=False)
    provenance: _DirtyCheckpointProvenance
    completed_indices: tuple[int, ...]


class SequenceExportService:
    """Execute one immutable job without reading Qt or live widget state."""

    def __init__(
        self,
        *,
        mes_validator: Callable[[dict[str, Any] | None], tuple[bool, str]] = (
            _validate_mes_runtime_config
        ),
        mes_writer: Callable[..., Any] = write_mes_result,
        output_path_resolver: Callable[..., str] = resolve_excel_output_path,
        spool_dir_resolver: Callable[..., str] = resolve_excel_spool_dir,
        csv_exporter: Callable[..., Any] = export_analysis_to_csv_spool,
        excel_exporter: Callable[..., Any] = export_analysis_to_excel,
        spool_builder: Callable[..., Any] = build_excel_from_csv_spool,
        logger: Any = None,
    ) -> None:
        self.mes_validator = mes_validator
        self.mes_writer = mes_writer
        self.output_path_resolver = output_path_resolver
        self.spool_dir_resolver = spool_dir_resolver
        self.csv_exporter = csv_exporter
        self.excel_exporter = excel_exporter
        self.spool_builder = spool_builder
        self.logger = logger
        self._dirty_capability = object()
        self._dirty_checkpoints: OrderedDict[
            Any, _TrustedDirtyCheckpoint
        ] = OrderedDict()
        self._dirty_checkpoint_limit = 128

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    @staticmethod
    def _handoff(snapshot: Any) -> dict[str, Any]:
        if not isinstance(snapshot, Mapping):
            raise ValueError("export result snapshot is unavailable")
        handoff = snapshot.get("export_handoff", snapshot)
        if not isinstance(handoff, Mapping):
            raise ValueError("export handoff is unavailable")
        detached = mutable_export_value(handoff)
        if not isinstance(detached, dict):
            raise ValueError("export handoff is unavailable")
        return detached

    @staticmethod
    def _target(target: Any) -> tuple[str, str, dict[str, Any]]:
        if not isinstance(target, Mapping):
            raise ValueError("export target is not a mapping")
        target_type = str(target.get("type") or "").lower()
        config_name = str(target.get("config_name") or target_type or "unknown")
        configuration = mutable_export_value(target.get("configuration"))
        if not isinstance(configuration, dict):
            raise ValueError(f"export target {config_name!r} has no configuration")
        if target_type not in {"mes", "excel"}:
            raise ValueError(f"unsupported export target type: {target_type or 'missing'}")
        return target_type, config_name, configuration

    @staticmethod
    def resolve_analysis_configuration(
        result_snapshot: Any,
        legacy_request_configuration: Any = _MISSING_ANALYSIS_CONFIGURATION,
    ) -> Mapping[str, Any] | None:
        """Resolve one frozen analysis configuration without merging sources."""
        if not isinstance(result_snapshot, Mapping):
            raise ValueError("analysis result snapshot must be a mapping")

        handoff = result_snapshot.get(
            "export_handoff", _MISSING_ANALYSIS_CONFIGURATION
        )
        if (
            handoff is not _MISSING_ANALYSIS_CONFIGURATION
            and handoff is not None
            and not isinstance(handoff, Mapping)
        ):
            raise ValueError("export handoff is malformed")
        if isinstance(handoff, Mapping):
            configuration = handoff.get(
                "analysis_config", _MISSING_ANALYSIS_CONFIGURATION
            )
            if (
                configuration is not _MISSING_ANALYSIS_CONFIGURATION
                and configuration is not None
            ):
                if not isinstance(configuration, Mapping):
                    raise ValueError("analysis configuration is malformed")
                return configuration

        if (
            legacy_request_configuration is not _MISSING_ANALYSIS_CONFIGURATION
            and legacy_request_configuration is not None
        ):
            if not isinstance(legacy_request_configuration, Mapping):
                raise ValueError("analysis configuration is malformed")
            return legacy_request_configuration

        configuration = result_snapshot.get(
            "analysis_configuration", _MISSING_ANALYSIS_CONFIGURATION
        )
        if (
            configuration is _MISSING_ANALYSIS_CONFIGURATION
            or configuration is None
        ):
            return None
        if not isinstance(configuration, Mapping):
            raise ValueError("analysis configuration is malformed")
        return configuration

    @staticmethod
    def resolve_target_configurations(
        analysis_config: Any,
    ) -> tuple[dict[str, Any], ...]:
        """Resolve the current MES-before-Excel target order from frozen config."""
        if not isinstance(analysis_config, Mapping):
            return ()
        display_sequence = analysis_config.get("display_sequence")
        displayed = (
            tuple(display_sequence)
            if isinstance(display_sequence, (tuple, list))
            else ()
        )
        excel_items = tuple(
            (str(name), mutable_export_value(configuration))
            for name, configuration in analysis_config.items()
            if isinstance(configuration, Mapping)
            and configuration.get("type") == "Excel"
        )
        targets: list[dict[str, Any]] = []
        for name, configuration in excel_items:
            if (
                name in displayed
                and configuration.get("enabled", True) is not False
                and configuration.get("save_mes_enabled") is True
            ):
                targets.append(
                    {
                        "type": "mes",
                        "config_name": name,
                        "configuration": configuration,
                    }
                )
        for name, configuration in excel_items:
            targets.append(
                {
                    "type": "excel",
                    "config_name": name,
                    "configuration": configuration,
                }
            )
        return tuple(targets)

    def prepare_analysis_export(
        self,
        record_id: str,
        result_snapshot: Any,
        analysis_configuration: Any,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Freeze one admitted analysis result and resolve its export targets."""
        if type(record_id) is not str or not record_id:
            raise ValueError("record_id must be non-empty text")
        if not isinstance(result_snapshot, Mapping):
            raise ValueError("analysis result snapshot must be a mapping")
        snapshot = mutable_export_value(result_snapshot)
        canonical_record_id = snapshot.get("record_id")
        if canonical_record_id is not None and canonical_record_id != record_id:
            raise ValueError("analysis result record does not match request")
        snapshot["record_id"] = record_id
        resolved_configuration = self.resolve_analysis_configuration(
            result_snapshot, analysis_configuration
        )
        targets = self.resolve_target_configurations(resolved_configuration)
        snapshot["export_targets"] = targets
        return snapshot, targets

    def prepare_manual_label_export(
        self, record_id: str, label: str, source_snapshot: Any
    ) -> tuple[Any, tuple[Any, ...]]:
        """Build and freeze a manual-label result for the exact retained record."""
        snapshot = self.build_labeled_result(record_id, label, source_snapshot)
        targets = snapshot.get("export_targets", ())
        return snapshot, tuple(targets)

    def build_labeled_result(
        self, record_id: str, label: str, source_snapshot: Any
    ) -> dict[str, Any]:
        """Build a detached manual-label result for the exact retained record."""
        if type(record_id) is not str or not record_id:
            raise ValueError("record_id must be non-empty text")
        if label not in {"OK", "NG"}:
            raise ValueError("label must be OK or NG")
        snapshot = mutable_export_value(source_snapshot)
        if not isinstance(snapshot, dict):
            snapshot = {}
        handoff = mutable_export_value(snapshot.get("export_handoff"))
        if not isinstance(handoff, dict):
            handoff = {}
        analysis_config = handoff.get("analysis_config")
        if not isinstance(analysis_config, Mapping):
            analysis_config = {}
        summary = (label == "OK", label)
        handoff["record_id"] = record_id
        handoff["ok_ng_summary"] = summary
        handoff["can_output_ok_ng"] = True
        handoff["analysis_config"] = mutable_export_value(analysis_config)
        snapshot["record_id"] = record_id
        snapshot["manual_label"] = label
        snapshot["ok_ng_summary"] = summary
        snapshot["export_handoff"] = handoff
        snapshot["export_targets"] = tuple(
            target
            for target in self.resolve_target_configurations(analysis_config)
            if isinstance(target, Mapping)
            and str(target.get("type") or "").lower() == "excel"
        )
        return snapshot

    @staticmethod
    def _validate_mes_handoff(
        handoff: Mapping[str, Any], config_name: str
    ) -> tuple[str, str]:
        result_dict = handoff.get("analysis_result_dict")
        if not isinstance(result_dict, Mapping) or not result_dict:
            raise ValueError("mes_write_skip_missing_analysis_result")
        if handoff.get("can_output_ok_ng") is not True:
            raise ValueError("mes_write_skip_output_not_ready")
        sn = str(handoff.get("sn") or "").strip()
        if not sn:
            raise ValueError(f"mes_write_skip_missing_sn[{config_name}]")
        if any(character in sn for character in (",", "\r", "\n")):
            raise ValueError(f"mes_write_skip_bad_sn[{config_name}]: {sn!r}")
        for item_name, item_result in result_dict.items():
            if (
                not isinstance(item_result, (tuple, list))
                or not item_result
                or not isinstance(item_result[0], (bool, np.bool_))
            ):
                raise ValueError(
                    f"mes_write_skip_invalid_summary_input[{config_name}]: {item_name!r}"
                )
        summary = handoff.get("ok_ng_summary")
        if not isinstance(summary, (tuple, list)) or len(summary) != 2:
            raise ValueError(f"mes_write_skip_bad_summary[{config_name}]: {summary!r}")
        label = summary[1]
        if label not in {"OK", "NG"}:
            raise ValueError(f"mes_write_skip_bad_label[{config_name}]: {label!r}")
        return sn, label

    def _execute_mes(
        self,
        handoff: Mapping[str, Any],
        config_name: str,
        configuration: dict[str, Any],
    ) -> ExportTargetResult:
        sn, label = self._validate_mes_handoff(handoff, config_name)
        valid, message = self.mes_validator(configuration)
        if not valid:
            raise RuntimeError(f"mes_write_skip_bad_config[{config_name}]: {message}")
        result = self.mes_writer(
            configuration,
            sn=sn,
            label=label,
            logger=self.logger,
        )
        if getattr(result, "ok", None) is not True:
            raise RuntimeError(
                f"mes_write_fail[{config_name}]: "
                f"{getattr(result, 'message', 'unknown MES failure')}"
            )
        message = str(getattr(result, "message", "MES export completed"))
        self._log("info", f"mes_write_ok[{config_name}]: {message}")
        return ExportTargetResult("mes", config_name, message)

    def _execute_excel(
        self,
        handoff: Mapping[str, Any],
        config_name: str,
        configuration: dict[str, Any],
    ) -> tuple[ExportTargetResult, SpoolTarget | None]:
        product_model = str(handoff.get("product_model") or "")
        file_path = str(
            self.output_path_resolver(
                configuration,
                product_model=product_model,
            )
        )
        arguments = {
            "sn": str(handoff.get("sn") or ""),
            "date_text": str(handoff.get("date_text") or ""),
            "analysis_items_data": mutable_export_value(
                handoff.get("analysis_items_data") or {}
            ),
            "analysis_config": mutable_export_value(
                handoff.get("analysis_config") or {}
            ),
            "analysis_result_dict": mutable_export_value(
                handoff.get("analysis_result_dict") or {}
            ),
            "product_model": product_model,
            "file_path": file_path,
        }
        if bool(configuration.get("fast_mode", True)):
            spool_dir = str(
                self.spool_dir_resolver(configuration, file_path=file_path)
            )
            result = self.csv_exporter(
                configuration,
                **arguments,
                spool_dir=spool_dir,
            )
            target = SpoolTarget.create(
                config_name,
                configuration,
                file_path,
                spool_dir,
            )
            log_tag = "excel_spool"
        else:
            result = self.excel_exporter(configuration, **arguments)
            target = None
            log_tag = "excel_export"
        message = str(getattr(result, "message", "unknown Excel result"))
        if getattr(result, "ok", None) is not True:
            raise RuntimeError(message)
        self._log("info", f"{log_tag}_ok[{config_name}]: {message}")
        return ExportTargetResult("excel", config_name, message), target

    def execute_record_job(
        self, job: ExportJob | RecordExportWork, attempt_id: str
    ) -> ExportExecutionOutcome:
        if not isinstance(job, (ExportJob, RecordExportWork)) or job.kind is not ExportJobKind.RECORD:
            raise TypeError("record export requires an ExportJob")
        if type(attempt_id) is not str or not attempt_id:
            raise ValueError("attempt_id must be non-empty text")
        results: list[ExportTargetResult] = []
        failures: list[ExportTargetFailure] = []
        dirty: list[SpoolTarget] = []
        dirty_entries: list[_FrozenDirtyTarget] = []
        dirty_indices: list[int] = []
        completed_indices: list[int] = []
        failed_indices: list[int] = []
        if type(job) is RecordExportWork:
            logical_job = job.logical_job
            target_indices = job.target_indices
        else:
            logical_job = job
            target_indices = tuple(range(len(job.target_configurations)))
        try:
            handoff = self._handoff(job.result_snapshot)
        except BaseException as error:
            failures.append(
                ExportTargetFailure(
                    "snapshot", "snapshot", _bounded_error_text(error, "invalid export snapshot")
                )
            )
        else:
            for target_index in target_indices:
                raw_target = logical_job.target_configurations[target_index]
                target_type = "unknown"
                config_name = "unknown"
                try:
                    target_type, config_name, configuration = self._target(raw_target)
                    if target_type == "mes":
                        result = self._execute_mes(
                            handoff, config_name, configuration
                        )
                        spool_target = None
                    else:
                        result, spool_target = self._execute_excel(
                            handoff, config_name, configuration
                        )
                    results.append(result)
                    completed_indices.append(target_index)
                    if spool_target is not None:
                        dirty.append(spool_target)
                        dirty_indices.append(target_index)
                        dirty_entries.append(
                            _FrozenDirtyTarget(
                                target_index,
                                raw_target,
                                target_type,
                                config_name,
                                bool(configuration.get("fast_mode", True)),
                                spool_target,
                            )
                        )
                except BaseException as error:
                    message = _bounded_error_text(error, "export target failed")
                    failures.append(
                        ExportTargetFailure(target_type, config_name, message)
                    )
                    failed_indices.append(target_index)
                    self._log(
                        "error",
                        f"{target_type}_export_fail[{config_name}]: {message}",
                    )
                    # MES is the admission gate for all Excel output. Once MES
                    # succeeds, independent Excel targets are all attempted so
                    # one locked workbook cannot hide failures in later files.
                    if target_type == "mes":
                        break
        provenance = None
        if dirty_entries:
            token = object()
            provenance = _DirtyCheckpointProvenance(
                self._dirty_capability,
                token,
                job.job_id,
                attempt_id,
                tuple(target_indices),
                tuple(dirty_entries),
            )
            self._dirty_checkpoints[token] = _TrustedDirtyCheckpoint(
                job,
                provenance,
                tuple(completed_indices),
            )
            if len(self._dirty_checkpoints) > self._dirty_checkpoint_limit:
                self._dirty_checkpoints.popitem(last=False)
        return ExportExecutionOutcome(
            not failures,
            job.job_id,
            attempt_id,
            job.record_id,
            tuple(immutable_export_value(results)),
            tuple(immutable_export_value(failures)),
            tuple(dirty),
            tuple(completed_indices),
            tuple(failed_indices),
            tuple(dirty_indices),
            provenance,
        )

    def validate_dirty_checkpoint(
        self, job: ExportJob | RecordExportWork, outcome: Any
    ) -> bool:
        """Validate dirty rebuild metadata against frozen target provenance."""
        try:
            dirty_targets = tuple(getattr(outcome, "dirty_targets", ()))
            dirty_indices = tuple(
                getattr(outcome, "dirty_target_indices", ())
            )
            if not dirty_targets:
                return not dirty_indices
            provenance = getattr(outcome, "dirty_provenance", None)
            if (
                type(provenance) is not _DirtyCheckpointProvenance
                or provenance.capability is not self._dirty_capability
            ):
                return False
            trusted = self._dirty_checkpoints.get(provenance.token)
            if (
                trusted is None
                or trusted.work is not job
                or trusted.provenance is not provenance
                or provenance.job_id != job.job_id
                or provenance.attempt_id != getattr(outcome, "attempt_id", None)
            ):
                return False
            if (
                len(dirty_targets) != len(dirty_indices)
                or len(set(dirty_indices)) != len(dirty_indices)
            ):
                return False
            logical_job = job.logical_job if type(job) is RecordExportWork else job
            requested = (
                job.target_indices
                if type(job) is RecordExportWork
                else tuple(range(len(logical_job.target_configurations)))
            )
            completed = tuple(getattr(outcome, "completed_target_indices", ()))
            if (
                provenance.requested_indices != requested
                or trusted.completed_indices != completed
                or dirty_indices
                != tuple(entry.index for entry in provenance.entries)
                or dirty_targets
                != tuple(entry.target for entry in provenance.entries)
            ):
                return False
            for entry, index, target in zip(
                provenance.entries, dirty_indices, dirty_targets
            ):
                if (
                    type(index) is not int
                    or index not in requested
                    or index not in completed
                    or type(target) is not SpoolTarget
                    or entry.index != index
                    or entry.target is not target
                ):
                    return False
                if (
                    logical_job.target_configurations[index]
                    is not entry.raw_target
                    or entry.target_type != "excel"
                    or entry.config_name != target.config_name
                    or entry.fast_mode is not True
                ):
                    return False
            self._dirty_checkpoints.pop(provenance.token, None)
            return True
        except BaseException:
            return False

    def execute_rebuild_job(
        self, job: SpoolRebuildJob
    ) -> RebuildExecutionOutcome:
        if type(job) is not SpoolRebuildJob:
            raise TypeError("rebuild export requires a SpoolRebuildJob")
        target = job.target
        try:
            result = self.spool_builder(
                mutable_export_value(target.configuration),
                file_path=target.file_path,
                spool_dir=target.spool_dir,
            )
            message = str(getattr(result, "message", "unknown rebuild result"))
            if getattr(result, "ok", None) is not True:
                raise RuntimeError(message)
        except BaseException as error:
            failure = ExportTargetFailure(
                "spool-rebuild",
                target.config_name,
                _bounded_error_text(error, "Excel spool rebuild failed"),
            )
            return RebuildExecutionOutcome(
                False,
                job.job_id,
                job.attempt_id,
                target.file_path,
                (),
                (failure,),
            )
        success = ExportTargetResult(
            "spool-rebuild", target.config_name, message
        )
        return RebuildExecutionOutcome(
            True,
            job.job_id,
            job.attempt_id,
            target.file_path,
            (success,),
            (),
        )

    def _resolve_flush_entries(
        self,
        tracked_targets: Any,
        *,
        analysis_config: Any,
        product_model: str,
    ) -> tuple[tuple[str, SpoolTarget | None, str | None], ...]:
        targets: OrderedDict[tuple[str, str], SpoolTarget] = OrderedDict()
        for target in tuple(tracked_targets or ()):
            if type(target) is SpoolTarget:
                targets[target.key] = target
        if targets or not isinstance(analysis_config, Mapping):
            return tuple(
                (target.config_name, target, None)
                for target in targets.values()
            )
        entries: list[tuple[str, SpoolTarget | None, str | None]] = []
        for name, raw_configuration in analysis_config.items():
            if not isinstance(raw_configuration, Mapping):
                continue
            configuration = mutable_export_value(raw_configuration)
            if (
                configuration.get("type") != "Excel"
                or configuration.get("enabled", True) is False
                or not bool(configuration.get("fast_mode", True))
            ):
                continue
            try:
                file_path = str(
                    self.output_path_resolver(
                        configuration,
                        product_model=product_model,
                    )
                )
                spool_dir = str(
                    self.spool_dir_resolver(configuration, file_path=file_path)
                )
                target = SpoolTarget.create(
                    str(name), configuration, file_path, spool_dir
                )
            except BaseException as error:
                entries.append(
                    (
                        str(name),
                        None,
                        _bounded_error_text(error, "path resolution failed"),
                    )
                )
                continue
            targets[target.key] = target
            entries.append((target.config_name, target, None))
        return tuple(entries)

    def resolve_flush_targets(
        self,
        tracked_targets: Any,
        *,
        analysis_config: Any,
        product_model: str,
    ) -> tuple[SpoolTarget, ...]:
        """Resolve valid targets; synchronous flush additionally reports errors."""
        entries = self._resolve_flush_entries(
            tracked_targets,
            analysis_config=analysis_config,
            product_model=product_model,
        )
        for config_name, _target, error in entries:
            if error is not None:
                self._log(
                    "error",
                    f"excel_spool_build_path_error[{config_name}]: {error}",
                )
        return tuple(
            target
            for _config_name, target, error in entries
            if target is not None and error is None
        )

    def flush_spool_targets(
        self,
        tracked_targets: Any,
        *,
        analysis_config: Any,
        product_model: str,
        on_close: bool = False,
    ) -> list[tuple[str, str]]:
        """Synchronously rebuild every resolved target and return all failures."""
        failures: list[tuple[str, str]] = []
        entries = self._resolve_flush_entries(
            tracked_targets,
            analysis_config=analysis_config,
            product_model=product_model,
        )
        for config_name, target, resolution_error in entries:
            if resolution_error is not None or target is None:
                message = resolution_error or "path resolution failed"
                failures.append((config_name, message))
                self._log(
                    "error",
                    f"excel_spool_build_path_error[{config_name}]: {message}",
                )
                continue
            try:
                result = self.spool_builder(
                    mutable_export_value(target.configuration),
                    file_path=target.file_path,
                    spool_dir=target.spool_dir,
                )
                message = str(getattr(result, "message", "unknown rebuild result"))
                if getattr(result, "ok", None) is not True:
                    failures.append((target.config_name, message))
                    self._log(
                        "warning",
                        f"excel_spool_build_{'on_close' if on_close else 'on_exit'}_fail"
                        f"[{target.config_name}]: {message}",
                    )
                else:
                    self._log(
                        "info",
                        f"excel_spool_build_{'on_close' if on_close else 'on_exit'}_ok"
                        f"[{target.config_name}]: {message}",
                    )
            except BaseException as error:
                message = _bounded_error_text(error, "Excel spool rebuild failed")
                failures.append((target.config_name, message))
                self._log(
                    "error",
                    f"excel_spool_build_{'on_close' if on_close else 'on_exit'}_error"
                    f"[{target.config_name}]: {message}",
                )
        return failures


__all__ = [
    "ExportExecutionOutcome",
    "ExportTargetFailure",
    "ExportTargetResult",
    "RebuildExecutionOutcome",
    "SequenceExportService",
]
