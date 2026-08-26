"""Main-thread state and immutable values for one sequence recording session."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePath
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ui.sequence.sequence_messages import (
    ConfigurationSnapshot,
    _FrozenMapping,
    _freeze_numpy_array,
    _freeze_payload,
)


_RECORDING_CANONICAL_MARKER = object()


def freeze_recording_session_value(value: Any) -> Any:
    """Detach and deeply freeze one worker-visible admission value.

    The sequence message freezer deliberately accepts only exact, data-only host
    types and validates NumPy allocation provenance.  Reusing that boundary here
    prevents mapping/list/ndarray subclasses from smuggling mutable aliases into a
    session which may outlive the Qt admission turn.
    """
    # A recording source can be live while admission runs.  Two bounded,
    # detached reads prevent publishing a torn array/container graph.  The
    # per-read memo also preserves repeated ndarray identity within related
    # fields, so one source cannot be copied at two different instants in the
    # same candidate snapshot.
    first_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    first_occurrences: list[int] = []
    first = _freeze_payload(
        value,
        array_memo=first_arrays,
        array_occurrences=first_occurrences,
        detach_configuration_snapshots=True,
        detach_immutable_arrays=True,
    )
    second_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    second_occurrences: list[int] = []
    second = _freeze_payload(
        value,
        array_memo=second_arrays,
        array_occurrences=second_occurrences,
        detach_configuration_snapshots=True,
        detach_immutable_arrays=True,
    )
    if (
        tuple(first_occurrences) != tuple(second_occurrences)
        or not _recording_frozen_values_equal(first, second)
    ):
        raise ValueError("recording source changed during stable snapshot capture")
    # Keep the message freezer's sealed mapping representation.  Unlike a
    # public MappingProxyType, its exact tuple storage can be revalidated at a
    # later worker boundary without invoking an arbitrary backing Mapping.
    return second


def _recording_frozen_values_equal(left: Any, right: Any) -> bool:
    """Compare two validated frozen reads without behavioral callbacks."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            tuple(np.ndarray.shape.__get__(left))
            == tuple(np.ndarray.shape.__get__(right))
            and np.ndarray.dtype.__get__(left) == np.ndarray.dtype.__get__(right)
            and np.ndarray.tobytes(left, order="C")
            == np.ndarray.tobytes(right, order="C")
        )
    if type(left) is _FrozenMapping:
        left_items = object.__getattribute__(left, "_items")
        right_items = object.__getattribute__(right, "_items")
        if tuple.__len__(left_items) != tuple.__len__(right_items):
            return False
        return all(
            _recording_frozen_values_equal(left_key, right_key)
            and _recording_frozen_values_equal(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                tuple.__iter__(left_items), tuple.__iter__(right_items)
            )
        )
    if type(left) is ConfigurationSnapshot:
        return all(
            _recording_frozen_values_equal(
                object.__getattribute__(left, field_name),
                object.__getattribute__(right, field_name),
            )
            for field_name in (
                "sequence_config",
                "analysis_config",
                "mic",
                "speaker",
                "mic_channels",
                "using_config_path",
                "streaming_stimulus_data",
            )
        )
    if type(left) is tuple:
        return tuple.__len__(left) == tuple.__len__(right) and all(
            _recording_frozen_values_equal(left_item, right_item)
            for left_item, right_item in zip(
                tuple.__iter__(left), tuple.__iter__(right)
            )
        )
    if type(left) is float:
        return left == right or (math.isnan(left) and math.isnan(right))
    if type(left) is complex:
        return _recording_frozen_values_equal(left.real, right.real) and (
            _recording_frozen_values_equal(left.imag, right.imag)
        )
    return bool(left == right)


def thaw_recording_session_value(value: Any, *, _hashable: bool = False) -> Any:
    """Create a fresh task-local mutable value for an immediate external API call."""
    if type(value) is ConfigurationSnapshot:
        if _hashable:
            raise TypeError(
                "configuration snapshots cannot be thawed as hashable values"
            )
        return freeze_recording_session_value(value)
    if isinstance(value, Mapping):
        if _hashable:
            raise TypeError("recording mappings cannot be thawed as hashable values")
        return {
            thaw_recording_session_value(key, _hashable=True): (
                thaw_recording_session_value(item)
            )
            for key, item in value.items()
        }
    if type(value) is tuple:
        values = [
            thaw_recording_session_value(item, _hashable=_hashable)
            for item in value
        ]
        return tuple(values) if _hashable else values
    if type(value) is frozenset:
        values = {
            thaw_recording_session_value(item, _hashable=True) for item in value
        }
        return frozenset(values) if _hashable else values
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Path):
        return type(value)(value)
    return value


def _recording_session_path(name: str, value: Any) -> Path:
    frozen = freeze_recording_session_value(value)
    if type(frozen) is not str and not isinstance(frozen, PurePath):
        raise TypeError(f"{name} must be exact text or an exact path")
    return Path(frozen)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_freeze_value(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_retained_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_numpy_array(value, reuse_immutable=True)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _freeze_retained_value(key): _freeze_retained_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_retained_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_retained_value(item) for item in value)
    return value


def plain_message_value(value: Any) -> Any:
    """Return the exact data-only container types accepted by sequence messages."""
    if isinstance(value, Mapping):
        return {
            plain_message_value(key): plain_message_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(plain_message_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(plain_message_value(item) for item in value)
    return value


def immutable_recording_value(value: Any) -> Any:
    """Detach and freeze a main-thread-visible recording result."""
    return _freeze_value(value)


class RecordingState(Enum):
    IDLE = auto()
    PREPARING = auto()
    RECORDING = auto()
    FINALIZING = auto()


def _sealed_recording_value(value_type: type, values: Mapping[str, Any]) -> Any:
    instance = object.__new__(value_type)
    for name in value_type.__dataclass_fields__:
        object.__setattr__(
            instance,
            name,
            _RECORDING_CANONICAL_MARKER
            if name == "_canonical_marker"
            else values[name],
        )
    return instance


@dataclass(frozen=True, slots=True, init=False)
class RecordingSessionSnapshot:
    session_id: str
    workflow_generation: int
    configuration_generation: int
    mode: str
    sample_rate: int | float
    bit_depth: int
    input_channels: tuple[int, ...]
    input_device: Any
    output_device: Any
    stimulus_snapshot: Any
    target_samples: int
    acquisition_sample_count: int
    output_path: Path
    temp_path: Path
    backup_path: Path | None
    record_id: str
    label: str
    analysis_config: Mapping[str, Any]
    configuration_snapshot: Any
    product_model: str
    serial_number: str
    scanner_enabled: bool
    count_persistence: Mapping[str, Any]
    pending_count: int | None
    replay: bool
    _canonical_marker: Any = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("use RecordingSessionSnapshot.create()")

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("RecordingSessionSnapshot is sealed")

    @classmethod
    def create(cls, **values: Any) -> "RecordingSessionSnapshot":
        session_id = values.get("session_id")
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty string")
        workflow_generation = values.get("workflow_generation")
        if type(workflow_generation) is not int or workflow_generation < 0:
            raise ValueError("workflow_generation must be a non-negative integer")
        configuration_generation = values.get("configuration_generation", 0)
        if type(configuration_generation) is not int or configuration_generation < 0:
            raise ValueError("configuration_generation must be a non-negative integer")
        mode = values.get("mode")
        if type(mode) is not str or not mode:
            raise ValueError("mode must be a non-empty string")
        sample_rate = values.get("sample_rate")
        if (
            type(sample_rate) not in (int, float)
            or not math.isfinite(float(sample_rate))
            or float(sample_rate) <= 0
        ):
            raise ValueError("sample_rate must be positive and finite")
        bit_depth = values.get("bit_depth")
        if type(bit_depth) is not int or bit_depth <= 0:
            raise ValueError("bit_depth must be a positive integer")
        supplied_channels = values.get("input_channels")
        if type(supplied_channels) is list:
            channels = tuple(list.__iter__(supplied_channels))
        elif type(supplied_channels) is tuple:
            channels = tuple(tuple.__iter__(supplied_channels))
        else:
            raise TypeError("input_channels must be an exact list or tuple")
        if not channels or any(type(channel) is not int or channel < 0 for channel in channels):
            raise ValueError("input_channels must contain non-negative integers")
        target_samples = values.get("target_samples")
        if type(target_samples) is not int or target_samples <= 0:
            raise ValueError("target_samples must be a positive integer")
        acquisition_sample_count = values.get(
            "acquisition_sample_count", target_samples
        )
        if (
            type(acquisition_sample_count) is not int
            or acquisition_sample_count <= 0
        ):
            raise ValueError(
                "acquisition_sample_count must be a positive integer"
            )
        output_path = _recording_session_path("output_path", values.get("output_path"))
        temp_path = _recording_session_path("temp_path", values.get("temp_path"))
        backup_value = values.get("backup_path")
        backup_path = (
            None
            if backup_value is None
            else _recording_session_path("backup_path", backup_value)
        )
        record_id = values.get("record_id")
        if type(record_id) is not str or not record_id:
            raise ValueError("record_id must be a non-empty string")
        label = values.get("label", "not_labeled")
        if type(label) is not str or not label:
            raise ValueError("label must be a non-empty string")
        pending_count = values.get("pending_count")
        if pending_count is not None and (
            type(pending_count) is not int or pending_count < 0
        ):
            raise ValueError("pending_count must be a non-negative integer or None")
        replay = values.get("replay", False)
        if type(replay) is not bool:
            raise ValueError("replay must be a boolean")
        product_model = values.get("product_model", "")
        serial_number = values.get("serial_number", "")
        scanner_enabled = values.get("scanner_enabled", False)
        if type(product_model) is not str:
            raise TypeError("product_model must be exact text")
        if type(serial_number) is not str:
            raise TypeError("serial_number must be exact text")
        if type(scanner_enabled) is not bool:
            raise TypeError("scanner_enabled must be an exact boolean")
        expected_count_persistence = {
            "product_model": product_model,
            "serial_number": serial_number,
            "scanner_enabled": scanner_enabled,
        }
        count_persistence_value = values.get("count_persistence")
        analysis_config_value = values.get("analysis_config")
        if analysis_config_value is None:
            analysis_config_value = {}
        frozen_session_values = freeze_recording_session_value(
            {
                "analysis_config": analysis_config_value,
                "input_device": values.get("input_device"),
                "output_device": values.get("output_device"),
                "stimulus_snapshot": values.get("stimulus_snapshot"),
                "configuration_snapshot": values.get(
                    "configuration_snapshot"
                ),
            }
        )
        analysis_config = frozen_session_values["analysis_config"]
        if not isinstance(analysis_config, Mapping):
            raise ValueError("analysis_config must be a mapping")
        if count_persistence_value is not None:
            supplied_count_persistence = freeze_recording_session_value(
                count_persistence_value
            )
            if not isinstance(supplied_count_persistence, Mapping):
                raise TypeError(
                    "count_persistence must be an exact data-only mapping"
                )
            if (
                set(supplied_count_persistence) != set(expected_count_persistence)
                or any(
                    supplied_count_persistence[name]
                    != expected_count_persistence[name]
                    for name in expected_count_persistence
                )
            ):
                raise ValueError(
                    "count_persistence must exactly match the frozen recording identity"
                )
        count_persistence = freeze_recording_session_value(
            expected_count_persistence
        )
        return _sealed_recording_value(
            cls,
            {
                "session_id": session_id,
                "workflow_generation": workflow_generation,
                "configuration_generation": configuration_generation,
                "mode": mode,
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "input_channels": channels,
                "input_device": frozen_session_values["input_device"],
                "output_device": frozen_session_values["output_device"],
                "stimulus_snapshot": frozen_session_values["stimulus_snapshot"],
                "target_samples": target_samples,
                "acquisition_sample_count": acquisition_sample_count,
                "output_path": output_path,
                "temp_path": temp_path,
                "backup_path": backup_path,
                "record_id": record_id,
                "label": label,
                "analysis_config": analysis_config,
                "configuration_snapshot": frozen_session_values[
                    "configuration_snapshot"
                ],
                "product_model": product_model,
                "serial_number": serial_number,
                "scanner_enabled": scanner_enabled,
                "count_persistence": count_persistence,
                "pending_count": pending_count,
                "replay": replay,
            },
        )

    @classmethod
    def canonicalize(cls, value: Any) -> "RecordingSessionSnapshot":
        if type(value) is not cls:
            raise TypeError("snapshot must be RecordingSessionSnapshot")
        return cls.create(
            **{
                name: object.__getattribute__(value, name)
                for name in cls.__dataclass_fields__
                if name != "_canonical_marker"
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            field: thaw_recording_session_value(getattr(self, field))
            for field in self.__dataclass_fields__
            if field != "_canonical_marker"
        }

    def as_message_payload(self) -> dict[str, Any]:
        return plain_message_value(self.as_dict())


@dataclass(frozen=True, slots=True, init=False)
class StagedRecording:
    snapshot: RecordingSessionSnapshot
    sample_count: int
    data_struct_fields: Mapping[str, Any]
    recorded_signal_info: Mapping[str, Any]
    stimulus_info: Any = None
    warnings: tuple[Any, ...] = ()
    _canonical_marker: Any = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("use StagedRecording.create()")

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("StagedRecording is sealed")

    @classmethod
    def create(
        cls,
        *,
        snapshot: RecordingSessionSnapshot,
        sample_count: int,
        data_struct_fields: Mapping[str, Any],
        recorded_signal_info: Mapping[str, Any],
        stimulus_info: Any = None,
        warnings: tuple[Any, ...] = (),
    ) -> "StagedRecording":
        if type(snapshot) is not RecordingSessionSnapshot:
            raise TypeError("snapshot must be RecordingSessionSnapshot")
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        canonical_snapshot = RecordingSessionSnapshot.canonicalize(snapshot)
        frozen_fields = freeze_recording_session_value(data_struct_fields)
        frozen_info = freeze_recording_session_value(recorded_signal_info)
        if not isinstance(frozen_fields, Mapping) or not isinstance(
            frozen_info, Mapping
        ):
            raise TypeError("recording snapshots must be mappings")
        frozen_warnings = freeze_recording_session_value(warnings)
        if type(frozen_warnings) is not tuple:
            raise TypeError("recording warnings must be an exact tuple")
        return _sealed_recording_value(
            cls,
            {
                "snapshot": canonical_snapshot,
                "sample_count": sample_count,
                "data_struct_fields": frozen_fields,
                "recorded_signal_info": frozen_info,
                "stimulus_info": freeze_recording_session_value(stimulus_info),
                "warnings": frozen_warnings,
            },
        )

    @classmethod
    def canonicalize(cls, value: Any) -> "StagedRecording":
        if type(value) is not cls:
            raise TypeError("worker result must be StagedRecording")
        return cls.create(
            snapshot=object.__getattribute__(value, "snapshot"),
            sample_count=object.__getattribute__(value, "sample_count"),
            data_struct_fields=object.__getattribute__(
                value, "data_struct_fields"
            ),
            recorded_signal_info=object.__getattribute__(
                value, "recorded_signal_info"
            ),
            stimulus_info=object.__getattribute__(value, "stimulus_info"),
            warnings=object.__getattribute__(value, "warnings"),
        )


@dataclass(frozen=True, slots=True)
class RecordingCounterStatistics:
    """One persisted count-board section owned by Recording state."""

    total: int
    ok: int
    ng: int
    datatime: str
    ok_percent: str | None = None
    not_labels: int | None = None

    def __post_init__(self) -> None:
        for name in ("total", "ok", "ng"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.datatime) is not str or not self.datatime:
            raise ValueError("datatime must be non-empty text")
        if self.ok_percent is not None and type(self.ok_percent) is not str:
            raise TypeError("ok_percent must be text or None")
        if self.not_labels is not None and (
            type(self.not_labels) is not int or self.not_labels < 0
        ):
            raise ValueError("not_labels must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class RecordingStatisticsSnapshot:
    """Canonical test and mark counts for one statistics projection."""

    test: RecordingCounterStatistics
    mark: RecordingCounterStatistics

    def __post_init__(self) -> None:
        if type(self.test) is not RecordingCounterStatistics:
            raise TypeError("test statistics must be RecordingCounterStatistics")
        if type(self.mark) is not RecordingCounterStatistics:
            raise TypeError("mark statistics must be RecordingCounterStatistics")
        if self.test.ok_percent is None:
            raise ValueError("test statistics must include ok_percent")
        if self.mark.not_labels is None:
            raise ValueError("mark statistics must include not_labels")


@dataclass(frozen=True, slots=True)
class RetainedRecordingSnapshot:
    """Exact immutable recording/configuration pair retained for later analysis."""

    record_id: str
    source_id: str
    workflow_generation: int
    recording_snapshot: Any
    configuration_snapshot: Any

    @classmethod
    def create(
        cls,
        record_id: str,
        recording_snapshot: Any,
        configuration_snapshot: Any,
        *,
        source_id: str,
        workflow_generation: int,
    ) -> "RetainedRecordingSnapshot":
        if type(record_id) is not str or not record_id:
            raise ValueError("retained record_id must be non-empty text")
        if type(source_id) is not str or not source_id:
            raise ValueError("retained source_id must be non-empty text")
        if type(workflow_generation) is not int or workflow_generation < 0:
            raise ValueError(
                "retained workflow_generation must be a non-negative integer"
            )
        return cls(
            record_id,
            source_id,
            workflow_generation,
            _freeze_retained_value(recording_snapshot),
            _freeze_retained_value(configuration_snapshot),
        )


class RecordingModel:
    """Own the active session and its final main-thread-visible result."""

    def __init__(self) -> None:
        self.state = RecordingState.IDLE
        self.active_session_id: str | None = None
        self.active_snapshot: RecordingSessionSnapshot | None = None
        self.final_result: Any = None
        self.recorded_path: str | None = None
        self.recorded_signal_info: dict[str, Any] | None = {}
        self.current_recorded_count: int | None = None
        self.last_play_count: int | None = None
        self.statistics: RecordingStatisticsSnapshot | None = None
        self.streaming_stimulus_data: Any = None
        self._retained_recording: RetainedRecordingSnapshot | None = None
        self._last_cleared_recording_identity: tuple[str, int] | None = None
        self._retention_active = False
        self._retention_lock = RLock()
        self.active_import_id: str | None = None
        self.active_import_workflow_generation: int | None = None
        self.import_cancel_pending = False
        self.import_cancellation_reason = ""
        self.import_runtime_consistent = True
        self.import_consistency_failure = ""
        self._retired_import_ids: deque[str] = deque(maxlen=128)
        self._retired_import_id_set: set[str] = set()

    def begin_import(self, import_id: str, workflow_generation: int = 0) -> None:
        if type(import_id) is not str or not import_id:
            raise ValueError("import_id must be a non-empty string")
        if self.active_import_id is not None:
            raise RuntimeError("an imported-audio load is already active")
        self.active_import_id = import_id
        self.active_import_workflow_generation = workflow_generation
        self.import_cancel_pending = False
        self.import_cancellation_reason = ""

    def request_import_cancel(self, reason: str) -> None:
        if self.active_import_id is None:
            raise RuntimeError("no imported-audio load is active")
        if self.import_cancel_pending:
            raise RuntimeError("import cancellation is already pending")
        self.import_cancel_pending = True
        self.import_cancellation_reason = str(
            reason or "audio import was cancelled"
        )

    def set_import_consistency(self, consistent: bool, reason: str = "") -> None:
        self.import_runtime_consistent = bool(consistent)
        self.import_consistency_failure = "" if consistent else str(reason)

    def retire_import(self, import_id: str) -> None:
        if import_id in self._retired_import_id_set:
            return
        if len(self._retired_import_ids) == self._retired_import_ids.maxlen:
            evicted = self._retired_import_ids[0]
            self._retired_import_id_set.discard(evicted)
        self._retired_import_ids.append(import_id)
        self._retired_import_id_set.add(import_id)
        if self.active_import_id == import_id:
            self.active_import_id = None
            self.active_import_workflow_generation = None
            self.import_cancel_pending = False
            self.import_cancellation_reason = ""

    def is_retired_import(self, import_id: str) -> bool:
        return import_id in self._retired_import_id_set

    def retain_recording_snapshot(
        self,
        record_id: str,
        recording_snapshot: Any,
        configuration_snapshot: Any,
        *,
        source_id: str,
        workflow_generation: int,
    ) -> bool:
        """Atomically retain one exact generation without duplicate replacement."""
        if type(record_id) is not str or not record_id:
            raise ValueError("retained record_id must be non-empty text")
        if type(source_id) is not str or not source_id:
            raise ValueError("retained source_id must be non-empty text")
        if type(workflow_generation) is not int or workflow_generation < 0:
            raise ValueError(
                "retained workflow_generation must be a non-negative integer"
            )
        with self._retention_lock:
            if self._retention_active:
                return False
            current = self._retained_recording
            if current is not None:
                if workflow_generation < current.workflow_generation:
                    return False
                if workflow_generation == current.workflow_generation:
                    return bool(
                        record_id == current.record_id
                        and source_id == current.source_id
                    )
            elif (
                self._last_cleared_recording_identity is not None
                and workflow_generation
                <= self._last_cleared_recording_identity[1]
            ):
                return False
            self._retention_active = True
        try:
            candidate = RetainedRecordingSnapshot.create(
                record_id,
                recording_snapshot,
                configuration_snapshot,
                source_id=source_id,
                workflow_generation=workflow_generation,
            )
            with self._retention_lock:
                current = self._retained_recording
                if current is not None:
                    if candidate.workflow_generation < current.workflow_generation:
                        return False
                    if candidate.workflow_generation == current.workflow_generation:
                        return bool(
                            candidate.record_id == current.record_id
                            and candidate.source_id == current.source_id
                        )
                self._retained_recording = candidate
            return True
        finally:
            with self._retention_lock:
                self._retention_active = False

    def retained_recording_snapshot(
        self, record_id: str
    ) -> RetainedRecordingSnapshot | None:
        if type(record_id) is not str or not record_id:
            return None
        with self._retention_lock:
            retained = self._retained_recording
            if retained is None or retained.record_id != record_id:
                return None
            return retained

    def retained_analysis_inputs(self, record_id: str) -> Any:
        retained = self.retained_recording_snapshot(record_id)
        if retained is None:
            return None
        return (
            plain_message_value(retained.recording_snapshot),
            plain_message_value(retained.configuration_snapshot),
        )

    def clear_retained_recording_snapshot(
        self, record_id: str, *, workflow_generation: int
    ) -> bool:
        if (
            type(record_id) is not str
            or not record_id
            or type(workflow_generation) is not int
            or workflow_generation < 0
        ):
            return False
        with self._retention_lock:
            if self._retention_active:
                return False
            retained = self._retained_recording
            if retained is None:
                return self._last_cleared_recording_identity == (
                    record_id,
                    workflow_generation,
                )
            if (
                retained.record_id != record_id
                or workflow_generation < retained.workflow_generation
            ):
                return False
            self._retained_recording = None
            self._last_cleared_recording_identity = (
                record_id,
                workflow_generation,
            )
            return True

    def commit_identity(self, staged: StagedRecording) -> None:
        if type(staged) is not StagedRecording:
            raise TypeError("recording identity requires a staged recording")
        self.recorded_path = str(staged.snapshot.output_path)
        self.recorded_signal_info = dict(staged.recorded_signal_info)

    def begin(self, snapshot: RecordingSessionSnapshot) -> None:
        if self.state is not RecordingState.IDLE:
            raise RuntimeError("a recording session is already active")
        if type(snapshot) is not RecordingSessionSnapshot:
            raise TypeError("snapshot must be RecordingSessionSnapshot")
        self.active_session_id = snapshot.session_id
        self.active_snapshot = snapshot
        self.final_result = None
        self.state = RecordingState.PREPARING

    def mark_recording(self) -> None:
        if self.state is not RecordingState.PREPARING:
            raise RuntimeError("recording can start only after preparation")
        self.state = RecordingState.RECORDING

    def mark_finalizing(self) -> None:
        if self.state is not RecordingState.RECORDING:
            raise RuntimeError("only an active recording can finalize")
        self.state = RecordingState.FINALIZING

    def finish(self, result: Any = None) -> None:
        self.final_result = result
        self.active_session_id = None
        self.active_snapshot = None
        self.state = RecordingState.IDLE

    def fail(self) -> None:
        self.finish(None)


class RecordingStreamingStimulusPort:
    """Explicit compatibility port for Recording-owned streaming stimulus state."""

    def __init__(self, model: RecordingModel) -> None:
        if type(model) is not RecordingModel:
            raise TypeError("streaming stimulus port requires RecordingModel")
        self._model = model

    def read(self) -> Any:
        return self._model.streaming_stimulus_data

    def write(self, value: Any) -> None:
        self._model.streaming_stimulus_data = value
