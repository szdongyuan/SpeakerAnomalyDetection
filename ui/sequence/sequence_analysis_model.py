"""Main-thread state and immutable values for sequence analysis."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Any

import numpy as np

from ui.sequence.sequence_messages import AnalysisRequested


DEFAULT_ANALYSIS_CALIBRATION_TYPES = frozenset(
    {
        "SPL",
        "SPLF",
        "FFT",
        "HD",
        "RB",
        "PRB",
        "LP",
        "PD",
        "ED",
        "FBA",
        "LOUD",
    }
)
MAX_ANALYSIS_CALIBRATION_TYPES = 128
MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH = 64


@dataclass(frozen=True, slots=True)
class AnalysisCalibrationPolicySnapshot:
    """Immutable, generation-tagged calibration preparation policy."""

    generation: int
    analysis_types: frozenset[str]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("calibration policy generation must be a non-negative integer")
        if type(self.analysis_types) is not frozenset:
            raise TypeError("calibration policy analysis_types must be a frozenset")
        if len(self.analysis_types) > MAX_ANALYSIS_CALIBRATION_TYPES:
            raise ValueError("calibration policy contains too many analysis types")
        for item in frozenset.__iter__(self.analysis_types):
            if type(item) is not str:
                raise TypeError("calibration analysis types must be exact strings")
            if not item or len(item) > MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH:
                raise ValueError("calibration analysis type has invalid length")


class SequenceAnalysisCalibrationPolicyService:
    """Build bounded policy snapshots without invoking hostile containers."""

    def __init__(
        self,
        *,
        max_types: int = MAX_ANALYSIS_CALIBRATION_TYPES,
        max_type_length: int = MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH,
    ) -> None:
        if type(max_types) is not int:
            raise TypeError("max_types must be an exact integer")
        if not 1 <= max_types <= MAX_ANALYSIS_CALIBRATION_TYPES:
            raise ValueError("max_types is outside the canonical bounds")
        if type(max_type_length) is not int:
            raise TypeError("max_type_length must be an exact integer")
        if not 1 <= max_type_length <= MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH:
            raise ValueError("max_type_length is outside the canonical bounds")
        self.max_types = max_types
        self.max_type_length = max_type_length

    def _plain_items(self, analysis_types: Any) -> tuple[Any, ...]:
        value_type = type(analysis_types)
        if value_type is tuple:
            size = tuple.__len__(analysis_types)
            iterator = tuple.__iter__
        elif value_type is list:
            size = list.__len__(analysis_types)
            iterator = list.__iter__
        elif value_type is set:
            size = set.__len__(analysis_types)
            iterator = set.__iter__
        elif value_type is frozenset:
            size = frozenset.__len__(analysis_types)
            iterator = frozenset.__iter__
        else:
            raise TypeError("analysis types must be a plain sequence or set")
        if size > self.max_types:
            raise ValueError("calibration policy contains too many analysis types")
        if value_type is tuple:
            return analysis_types
        return tuple(iterator(analysis_types))

    def snapshot(
        self, analysis_types: Any, *, generation: int
    ) -> AnalysisCalibrationPolicySnapshot:
        if type(generation) is not int or generation < 0:
            raise ValueError(
                "calibration policy generation must be a non-negative integer"
            )
        items = self._plain_items(analysis_types)
        normalized: set[str] = set()
        for item in items:
            if type(item) is not str:
                raise TypeError("calibration analysis types must be exact strings")
            if not item or len(item) > self.max_type_length:
                raise ValueError("calibration analysis type has invalid length")
            normalized.add(item)
        return AnalysisCalibrationPolicySnapshot(generation, frozenset(normalized))


def normalize_analysis_geometry(value: Any) -> dict[str, int] | None:
    """Normalize one persisted window rectangle without trusting JSON shape."""
    if not isinstance(value, Mapping):
        return None
    try:
        normalized = {key: int(value.get(key)) for key in ("x", "y", "w", "h")}
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if normalized["w"] < 200 or normalized["h"] < 150:
        return None
    return normalized


def immutable_analysis_value(value: Any) -> Any:
    """Detach mutable analysis data before it crosses a domain boundary."""
    if isinstance(value, np.ndarray):
        detached = np.array(value, copy=True)
        detached.setflags(write=False)
        return detached
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                immutable_analysis_value(key): immutable_analysis_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(immutable_analysis_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(immutable_analysis_value(item) for item in value)
    return value


def mutable_analysis_value(value: Any, *, _hashable: bool = False) -> Any:
    """Create a detached mutable value for immediate legacy analysis execution."""
    if isinstance(value, Mapping):
        if _hashable:
            raise TypeError("analysis mappings cannot be thawed as hashable values")
        return {
            mutable_analysis_value(key, _hashable=True): mutable_analysis_value(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        items = [
            mutable_analysis_value(item, _hashable=_hashable) for item in value
        ]
        return tuple(items) if _hashable else items
    if type(value) is frozenset:
        items = {
            mutable_analysis_value(item, _hashable=True) for item in value
        }
        return frozenset(items) if _hashable else items
    if isinstance(value, np.ndarray):
        if _hashable:
            raise TypeError("analysis arrays cannot be thawed as hashable values")
        return np.array(value, copy=True)
    return value


def plain_analysis_value(value: Any) -> Any:
    """Convert model snapshots to data-only containers accepted by messages."""
    if isinstance(value, Mapping):
        return {
            plain_analysis_value(key): plain_analysis_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(plain_analysis_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(plain_analysis_value(item) for item in value)
    return value


class AnalysisState(Enum):
    IDLE = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class SequenceAnalysisModel:
    """Own one admitted task, instantiated analyses, results and geometry."""

    def __init__(self) -> None:
        self.state = AnalysisState.IDLE
        self.active_analysis_id: str | None = None
        self.active_source_id: str | None = None
        self.active_workflow_generation: int | None = None
        self.recording_snapshot: Any = None
        self.configuration_snapshot: Any = None
        self.automatic = False
        self.analysis_instances: list[Any] = []
        self.analysis_registry: dict[str, Any] = {}
        self.result_snapshot: Mapping[str, Any] = MappingProxyType({})
        self.calibration_results: tuple[Any, ...] = ()
        self.calibration_policy_snapshot = AnalysisCalibrationPolicySnapshot(
            generation=0,
            analysis_types=DEFAULT_ANALYSIS_CALIBRATION_TYPES,
        )
        self.geometry: dict[str, dict[str, int]] = {}
        self.geometry_dirty = False
        self.cancel_pending = False
        self.cancellation_reason = ""
        self._retired_identities: deque[tuple[str, str, int]] = deque(maxlen=128)
        self._retired_identity_set: set[tuple[str, str, int]] = set()

    def begin(self, command: AnalysisRequested) -> None:
        if type(command) is not AnalysisRequested:
            raise TypeError("command must be AnalysisRequested")
        if self.state is AnalysisState.RUNNING:
            raise RuntimeError("an analysis task is already running")
        self.state = AnalysisState.RUNNING
        self.active_analysis_id = command.analysis_id
        self.active_source_id = command.source_id
        self.active_workflow_generation = command.workflow_generation
        self.recording_snapshot = immutable_analysis_value(command.recording_snapshot)
        self.configuration_snapshot = immutable_analysis_value(
            command.configuration_snapshot
        )
        self.automatic = command.automatic
        self.calibration_results = ()
        self.result_snapshot = MappingProxyType({})
        self.cancel_pending = False
        self.cancellation_reason = ""

    def request_cancel(self, reason: str) -> None:
        if self.state is not AnalysisState.RUNNING:
            raise RuntimeError("no analysis task is running")
        self.cancel_pending = True
        self.cancellation_reason = str(reason or "analysis cancelled")

    def retire(self, analysis_id: str, source_id: str, workflow_generation: int) -> None:
        identity = (analysis_id, source_id, workflow_generation)
        if identity in self._retired_identity_set:
            return
        if len(self._retired_identities) == self._retired_identities.maxlen:
            evicted = self._retired_identities[0]
            self._retired_identity_set.discard(evicted)
        self._retired_identities.append(identity)
        self._retired_identity_set.add(identity)

    def is_retired(
        self, analysis_id: str, source_id: str, workflow_generation: int
    ) -> bool:
        return (analysis_id, source_id, workflow_generation) in self._retired_identity_set

    def is_retired_analysis_id(
        self, analysis_id: str, workflow_generation: int
    ) -> bool:
        return any(
            retired_id == analysis_id and retired_generation == workflow_generation
            for retired_id, _source_id, retired_generation in self._retired_identities
        )

    def reset_output(self) -> None:
        self.analysis_instances.clear()
        self.analysis_registry.clear()
        self.result_snapshot = MappingProxyType({})

    def register_instance(self, key: str, instance: Any) -> None:
        if type(key) is not str or not key:
            raise ValueError("analysis instance key must be a non-empty string")
        self.analysis_registry[key] = instance
        self.analysis_instances.append(instance)

    def complete(self, result_snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.state is not AnalysisState.RUNNING:
            raise RuntimeError("no analysis task is running")
        frozen = immutable_analysis_value(result_snapshot)
        if not isinstance(frozen, Mapping):
            raise TypeError("analysis result snapshot must be a mapping")
        self.result_snapshot = frozen
        self.state = AnalysisState.COMPLETED
        self.active_analysis_id = None
        self.active_source_id = None
        self.active_workflow_generation = None
        self.cancel_pending = False
        self.cancellation_reason = ""
        return frozen

    def fail(self) -> None:
        self.state = AnalysisState.FAILED
        self.result_snapshot = MappingProxyType({})
        self.active_analysis_id = None
        self.active_source_id = None
        self.active_workflow_generation = None
        self.cancel_pending = False
        self.cancellation_reason = ""

    def set_calibration_results(self, values: Any) -> None:
        self.calibration_results = tuple(immutable_analysis_value(values or ()))

    def apply_calibration_policy(
        self, snapshot: AnalysisCalibrationPolicySnapshot
    ) -> bool:
        if type(snapshot) is not AnalysisCalibrationPolicySnapshot:
            raise TypeError("snapshot must be an AnalysisCalibrationPolicySnapshot")
        if snapshot.generation <= self.calibration_policy_snapshot.generation:
            return False
        self.calibration_policy_snapshot = snapshot
        return True

    def set_geometry(self, key: str, geometry: Mapping[str, int]) -> None:
        normalized = normalize_analysis_geometry(geometry)
        if normalized is None:
            return
        self.geometry[key] = normalized
        self.geometry_dirty = True

    def replace_geometry(self, geometry: Mapping[str, Mapping[str, int]]) -> None:
        normalized: dict[str, dict[str, int]] = {}
        if isinstance(geometry, Mapping):
            try:
                items = geometry.items()
                for key, value in items:
                    item = normalize_analysis_geometry(value)
                    if item is not None:
                        normalized[str(key)] = item
            except (AttributeError, TypeError, ValueError, RuntimeError):
                normalized = {}
        self.geometry = normalized
        self.geometry_dirty = False
