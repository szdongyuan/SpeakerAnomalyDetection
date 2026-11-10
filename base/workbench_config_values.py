"""Shared workbench identity values and JSON snapshot operations, without I/O."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_config(value, field: str):
    """Detach JSON containers and diagnose unsupported nesting at construction."""
    try:
        return _freeze_json(value)
    except RecursionError as exc:
        raise ValueError(f"{field} is nested too deeply") from exc


def _thaw_json(data: Mapping) -> dict:
    """Copy a snapshot into plain containers without recursive Python calls."""
    result = {}
    pending = [(data.items(), result)]
    while pending:
        entries, destination = pending.pop()
        for key, value in entries:
            if isinstance(value, Mapping):
                item = {}
                pending.append((value.items(), item))
            elif isinstance(value, (tuple, list)):
                item = [None] * len(value)
                pending.append((enumerate(value), item))
            else:
                item = value
            destination[key] = item
    return result


def _same_snapshot_value(first, second) -> bool:
    """Compare snapshots strictly, including bool/int and nested value objects."""
    pending = [(first, second)]
    while pending:
        left, right = pending.pop()
        if left is right:
            continue
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or left.keys() != right.keys():
                return False
            pending.extend((value, right[key]) for key, value in left.items())
        elif isinstance(left, tuple):
            if not isinstance(right, tuple) or len(left) != len(right):
                return False
            pending.extend(zip(left, right))
        elif is_dataclass(left) and not isinstance(left, type):
            if type(left) is not type(right):
                return False
            pending.extend((getattr(left, field.name), getattr(right, field.name)) for field in fields(left))
        elif type(left) is not type(right) or left != right:
            return False
    return True


class _SnapshotValue:
    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return _same_snapshot_value(self, other)


@dataclass(frozen=True, eq=False)
class WorkbenchHardwareChannel(_SnapshotValue):
    device_ref: str
    physical_channel_index: int


@dataclass(frozen=True, eq=False)
class WorkbenchChannel(_SnapshotValue):
    channel_id: str
    channel_name: str
    signal_type: str
    hardware_channel: WorkbenchHardwareChannel


@dataclass(frozen=True, eq=False)
class WorkbenchStation(_SnapshotValue):
    station_id: str
    station_name: str
    channels: tuple[WorkbenchChannel, ...]

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))


@dataclass(frozen=True, eq=False)
class WorkbenchCondition(_SnapshotValue):
    condition_id: str
    condition_name: str
    test_queue: str
    applicable_station_ids: tuple[str, ...]
    trigger_state: str
    segmented_analysis: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "applicable_station_ids", tuple(self.applicable_station_ids))
        object.__setattr__(self, "segmented_analysis", _freeze_config(
            self.segmented_analysis, "segmented_analysis",
        ))


@dataclass(frozen=True, eq=False)
class WorkbenchContext(_SnapshotValue):
    project_id: str
    project_name: str
    result_root_directory: str
    stations: tuple[WorkbenchStation, ...]
    conditions: tuple[WorkbenchCondition, ...]

    def __post_init__(self):
        object.__setattr__(self, "stations", tuple(self.stations))
        object.__setattr__(self, "conditions", tuple(self.conditions))

    def get_condition(self, condition_id: str) -> WorkbenchCondition | None:
        return next((item for item in self.conditions if item.condition_id == condition_id), None)

    def get_station(self, station_id: str) -> WorkbenchStation | None:
        return next((item for item in self.stations if item.station_id == station_id), None)
