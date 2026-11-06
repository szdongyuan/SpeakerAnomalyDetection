"""Read-only configuration snapshots for the motor workbench."""

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

from base.recording_preview_config import resolve_recording_preview_time_mode
from consts.product_test_project_consts import (
    CONDITION_NAME_KEY,
    PROJECT_NAME_KEY,
    REGISTRY_ACTIVE_FILE_KEY,
    REGISTRY_CONFIGS_KEY,
    REGISTRY_FILE_KEY,
    RESULT_ROOT_DIRECTORY_KEY,
    TEST_QUEUE_KEY,
    TRIGGER_STATE_KEY,
)
from consts.recording_preview_consts import RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
from consts.running_consts import (
    DEFAULT_DIR,
    PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
    SEQUENCE_CONFIG_REGISTRY_PATH,
)


class WorkbenchConfigurationError(ValueError):
    """A saved workbench configuration could not be applied."""


@dataclass(frozen=True)
class WorkbenchPlotCell:
    title: str
    analysis_item: str
    analysis_type: str


@dataclass(frozen=True)
class WorkbenchPlotLayout:
    rows: int
    columns: int
    cells: tuple[tuple[WorkbenchPlotCell, ...], ...]


@dataclass(frozen=True)
class WorkbenchHardwareChannel:
    device_ref: str
    physical_channel_index: int


@dataclass(frozen=True)
class WorkbenchChannel:
    channel_id: str
    channel_name: str
    signal_type: str
    hardware_channel: WorkbenchHardwareChannel


@dataclass(frozen=True)
class WorkbenchStation:
    station_id: str
    station_name: str
    channels: tuple[WorkbenchChannel, ...]


@dataclass(frozen=True, eq=False)
class WorkbenchQueueConfig:
    sequence_config: tuple[object, ...]
    acquisition: Mapping[str, object]
    acquisition_mode: str  # Normalized mode while acquisition preserves the saved fields.
    analysis: Mapping[str, object]
    metric_names: tuple[str, ...]

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return _same_snapshot_value(vars(self), vars(other))


@dataclass(frozen=True, eq=False)
class WorkbenchCondition:
    condition_id: str
    condition_name: str
    test_queue: str
    applicable_station_ids: tuple[str, ...]
    trigger_state: str
    segmented_analysis: Mapping[str, object]
    queue: WorkbenchQueueConfig
    project_id: str
    result_root_directory: str
    condition_config: Mapping[str, object]

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return _same_snapshot_value(vars(self), vars(other))

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self.queue.metric_names

    def export_process_config(self) -> dict:
        """Return independent plain containers for a recording or analysis request."""
        return _thaw_json({
            "acq": self.queue.acquisition,
            "condition_config": self.condition_config,
            "sequence_config": self.queue.sequence_config,
            "analysis_config": self.queue.analysis,
        })


@dataclass(frozen=True)
class WorkbenchConfig:
    project_id: str
    project_name: str
    result_root_directory: str
    stations: tuple[WorkbenchStation, ...]
    conditions: tuple[WorkbenchCondition, ...]
    active_file: str
    plot_layout: WorkbenchPlotLayout
    conditions_by_id: Mapping[str, WorkbenchCondition] = field(init=False, repr=False, compare=False)
    station_channels_by_id: Mapping[str, tuple[WorkbenchChannel, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        object.__setattr__(self, "conditions_by_id", MappingProxyType({
            condition.condition_id: condition for condition in self.conditions
        }))
        object.__setattr__(self, "station_channels_by_id", MappingProxyType({
            station.station_id: station.channels for station in self.stations
        }))


def _read_json(path: Path, description: str):
    def reject_constant(token):
        raise json.JSONDecodeError(f"invalid JSON constant {token}", token, 0)

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkbenchConfigurationError(f"{path}: could not read {description}: {exc}") from exc
    try:
        return json.loads(content, parse_constant=reject_constant)
    except (ValueError, RecursionError) as exc:
        # The decoder can hit Python's integer conversion or recursion limits.
        raise WorkbenchConfigurationError(f"{path}: could not decode {description}: {exc}") from exc


def _object(value, path: Path, field: str) -> dict:
    if not isinstance(value, dict):
        raise WorkbenchConfigurationError(f"{path}: {field} must be an object")
    return value


def _array(value, path: Path, field: str) -> list:
    if not isinstance(value, list):
        raise WorkbenchConfigurationError(f"{path}: {field} must be an array")
    return value


def _nonempty_string(value, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkbenchConfigurationError(f"{path}: {field} must be a nonempty string")
    return value


def _unique(values, path: Path, field: str):
    seen = set()
    for value in values:
        if value in seen:
            raise WorkbenchConfigurationError(f"{path}: {field} contains duplicate {value!r}")
        seen.add(value)


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_config(value, path: Path, field: str):
    try:
        return _freeze_json(value)
    except RecursionError as exc:
        raise WorkbenchConfigurationError(f"{path}: {field} is nested too deeply") from exc


def _thaw_json(data: Mapping[str, object]) -> dict:
    result = {}
    pending = [(data.items(), result)]
    while pending:
        entries, destination = pending.pop()
        for key, value in entries:
            if isinstance(value, Mapping):
                item = {}
                pending.append((value.items(), item))
            elif isinstance(value, tuple):
                item = [None] * len(value)
                pending.append((enumerate(value), item))
            else:
                item = value
            destination[key] = item
    return result


def _same_snapshot_value(first, second) -> bool:
    """Compare nested values without coercing numbers or relying on recursion depth."""
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
        elif type(left) is not type(right) or left != right:
            return False
    return True


def _active_product_filename(path: Path) -> str | None:
    registry = _object(_read_json(path, "product registry"), path, "product registry")
    entries = _array(registry.get(REGISTRY_CONFIGS_KEY), path, REGISTRY_CONFIGS_KEY)
    filenames = []
    for index, entry in enumerate(entries):
        field = f"{REGISTRY_CONFIGS_KEY}[{index}]"
        entry = _object(entry, path, field)
        file_field = f"{field}.{REGISTRY_FILE_KEY}"
        name = _nonempty_string(entry.get(REGISTRY_FILE_KEY), path, file_field)
        filename = PureWindowsPath(name)
        if (
            filename.name != name or filename.suffix.lower() != ".json"
            or filename.is_reserved() or any(char in '<>:"|?*' or ord(char) < 32 for char in name)
        ):
            raise WorkbenchConfigurationError(f"{path}: {file_field} must be a plain JSON filename")
        _nonempty_string(entry.get(PROJECT_NAME_KEY), path, f"{field}.{PROJECT_NAME_KEY}")
        filenames.append(name)
    _unique(map(os.path.normcase, filenames), path, f"{REGISTRY_CONFIGS_KEY}.{REGISTRY_FILE_KEY}")

    if REGISTRY_ACTIVE_FILE_KEY not in registry:
        raise WorkbenchConfigurationError(f"{path}: missing {REGISTRY_ACTIVE_FILE_KEY}")
    active_file = registry[REGISTRY_ACTIVE_FILE_KEY]
    if active_file is not None:
        active_file = _nonempty_string(active_file, path, REGISTRY_ACTIVE_FILE_KEY)
        if active_file not in filenames:
            raise WorkbenchConfigurationError(
                f"{path}: {REGISTRY_ACTIVE_FILE_KEY} {active_file!r} is not registered"
            )
    return active_file


def _layout_dimension(data: dict, path: Path, field: str, maximum: int) -> int:
    value = data.get(field)
    if type(value) is not int or not 1 <= value <= maximum:
        raise WorkbenchConfigurationError(f"{path}: {field} must be an integer from 1 to {maximum}")
    return value


def _load_plot_cell(data, path: Path, field: str) -> WorkbenchPlotCell:
    data = _object(data, path, field)
    title = _nonempty_string(data.get("title"), path, f"{field}.title")
    analysis_item = _nonempty_string(data.get("analysis_item"), path, f"{field}.analysis_item")
    analysis_type = _nonempty_string(data.get("analysis_type"), path, f"{field}.analysis_type")
    if analysis_type not in ("SPL", "FFT"):
        raise WorkbenchConfigurationError(
            f"{path}: {field}.analysis_type {analysis_type!r} must be SPL or FFT"
        )
    return WorkbenchPlotCell(title, analysis_item, analysis_type)


def _load_plot_layout(path: Path) -> WorkbenchPlotLayout:
    data = _object(_read_json(path, "plot layout"), path, "plot layout")
    rows = _layout_dimension(data, path, "rows", 3)
    columns = _layout_dimension(data, path, "columns", 4)
    cell_data = _array(data.get("cells"), path, "cells")
    if len(cell_data) != rows:
        raise WorkbenchConfigurationError(f"{path}: cells must contain {rows} rows")

    cells = []
    for index, row_data in enumerate(cell_data):
        field = f"cells[{index}]"
        row_data = _array(row_data, path, field)
        if len(row_data) != columns:
            raise WorkbenchConfigurationError(f"{path}: {field} must contain {columns} columns")
        cells.append(tuple(
            _load_plot_cell(cell, path, f"{field}[{column}]") for column, cell in enumerate(row_data)
        ))
    return WorkbenchPlotLayout(rows, columns, tuple(cells))


def _load_channel(data, path: Path, field: str) -> WorkbenchChannel:
    data = _object(data, path, field)
    hardware_field = f"{field}.hardware_channel"
    hardware = _object(data.get("hardware_channel"), path, hardware_field)
    index = hardware.get("physical_channel_index")
    if type(index) is not int or index < 0:
        raise WorkbenchConfigurationError(
            f"{path}: {hardware_field}.physical_channel_index must be a nonnegative integer"
        )
    return WorkbenchChannel(
        channel_id=_nonempty_string(data.get("channel_id"), path, f"{field}.channel_id"),
        channel_name=_nonempty_string(data.get("channel_name"), path, f"{field}.channel_name"),
        signal_type=_nonempty_string(data.get("signal_type"), path, f"{field}.signal_type"),
        hardware_channel=WorkbenchHardwareChannel(
            device_ref=_nonempty_string(hardware.get("device_ref"), path, f"{hardware_field}.device_ref"),
            physical_channel_index=index,
        ),
    )


def _load_station(data, path: Path, index: int) -> WorkbenchStation:
    field = f"stations[{index}]"
    data = _object(data, path, field)
    channel_data = _array(data.get("channels"), path, f"{field}.channels")
    channels = tuple(
        _load_channel(channel, path, f"{field}.channels[{channel_index}]")
        for channel_index, channel in enumerate(channel_data)
    )
    _unique((channel.channel_id for channel in channels), path, f"{field}.channels.channel_id")
    return WorkbenchStation(
        station_id=_nonempty_string(data.get("station_id"), path, f"{field}.station_id"),
        station_name=_nonempty_string(data.get("station_name"), path, f"{field}.station_name"),
        channels=channels,
    )


def _string_array(value, path: Path, field: str) -> tuple[str, ...]:
    values = _array(value, path, field)
    if any(not isinstance(item, str) for item in values):
        raise WorkbenchConfigurationError(f"{path}: {field} must contain only strings")
    _unique(values, path, field)
    return tuple(values)


def _validate_acquisition(data, path: Path) -> str:
    field = "[0].seq1.acq"
    acquisition = _object(data, path, field)
    raw_mode = acquisition.get("mode")
    if raw_mode is not None and not isinstance(raw_mode, str):
        raise WorkbenchConfigurationError(f"{path}: {field}.mode must be RECORD_ONLY or empty")
    mode = (raw_mode or "RECORD_ONLY").strip().upper()
    if mode != "RECORD_ONLY":
        raise WorkbenchConfigurationError(f"{path}: {field}.mode {raw_mode!r} must be RECORD_ONLY")

    detail_field = f"{field}.detail"
    detail = _object(acquisition.get("detail"), path, detail_field)
    for key in ("total_time", "sample_rate"):
        value = detail.get(key)
        if (
            type(value) not in (int, float)
            or value <= 0
            or (type(value) is float and not math.isfinite(value))
        ):
            raise WorkbenchConfigurationError(f"{path}: {detail_field}.{key} must be a finite positive number")
    try:
        resolve_recording_preview_time_mode(detail)
    except ValueError as exc:
        raise WorkbenchConfigurationError(
            f"{path}: {detail_field}.{RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY}: {exc}"
        ) from exc
    return mode


def _load_queue(queue_registry, registry_path: Path, queue: str) -> WorkbenchQueueConfig:
    saved_path = _nonempty_string(queue_registry.get(queue), registry_path, f"queue {queue!r} path")
    if "\0" in saved_path:
        raise WorkbenchConfigurationError(f"{registry_path}: invalid path for queue {queue!r}")
    configured_path = Path(saved_path)
    windows_path = PureWindowsPath(saved_path)
    if not configured_path.is_absolute() and windows_path.anchor and not windows_path.is_absolute():
        raise WorkbenchConfigurationError(
            f"{registry_path}: path for queue {queue!r} must be fully qualified "
            f"or relative to the registry directory: {saved_path!r}"
        )
    path = registry_path.parent / configured_path
    sequence = _read_json(path, f"queue {queue!r}")
    if not isinstance(sequence, list) or not sequence:
        raise WorkbenchConfigurationError(f"{path}: queue {queue!r} must be a nonempty array with seq1")

    first = _object(sequence[0], path, "[0]")
    seq1 = _object(first.get("seq1"), path, "[0].seq1")
    mode = _validate_acquisition(seq1.get("acq"), path)
    analysis = _object(seq1.get("analysis_list"), path, "[0].seq1.analysis_list")
    names = _string_array(
        analysis.get("display_sequence"), path, "[0].seq1.analysis_list.display_sequence"
    )
    for name in names:
        _object(analysis.get(name), path, f"[0].seq1.analysis_list[{name!r}]")
    sequence_config = _freeze_config(sequence, path, f"queue {queue!r}")
    frozen_seq1 = sequence_config[0]["seq1"]
    return WorkbenchQueueConfig(
        sequence_config=sequence_config,
        acquisition=frozen_seq1["acq"],
        acquisition_mode=mode,
        analysis=frozen_seq1["analysis_list"],
        metric_names=names,
    )


def _load_conditions(
    data, product_path: Path, station_ids: set[str], queue_registry_path: Path,
    *, project_id: str, result_root_directory: str,
) -> tuple[WorkbenchCondition, ...]:
    condition_data = _array(data, product_path, "conditions")
    if not condition_data:
        return ()

    queue_registry = _object(
        _read_json(queue_registry_path, "queue registry"), queue_registry_path, "queue registry"
    )
    queues_by_name = {}
    conditions = []
    for index, item in enumerate(condition_data):
        field = f"conditions[{index}]"
        item = _object(item, product_path, field)
        condition_id = _nonempty_string(item.get("condition_id"), product_path, f"{field}.condition_id")
        name = _nonempty_string(
            item.get(CONDITION_NAME_KEY), product_path, f"{field}.{CONDITION_NAME_KEY}"
        )
        applicable = _string_array(
            item.get("applicable_station_ids"), product_path, f"{field}.applicable_station_ids"
        )
        for station_id in applicable:
            if station_id not in station_ids:
                raise WorkbenchConfigurationError(
                    f"{product_path}: {field}.applicable_station_ids "
                    f"references unknown station {station_id!r}"
                )
        trigger_state = item.get(TRIGGER_STATE_KEY)
        if not isinstance(trigger_state, str):
            raise WorkbenchConfigurationError(
                f"{product_path}: {field}.{TRIGGER_STATE_KEY} must be a string"
            )
        segmented_field = f"{field}.segmented_analysis"
        segmented = _object(item.get("segmented_analysis"), product_path, segmented_field)
        _nonempty_string(segmented.get("mode"), product_path, f"{segmented_field}.mode")
        condition_config = MappingProxyType({
            key: _freeze_config(value, product_path, f"{field}.{key}") for key, value in item.items()
        })

        queue = _nonempty_string(item.get(TEST_QUEUE_KEY), product_path, f"{field}.{TEST_QUEUE_KEY}")
        if queue == "using_config_path" or queue not in queue_registry:
            raise WorkbenchConfigurationError(
                f"{product_path}: {field}.{TEST_QUEUE_KEY} "
                f"references unknown queue {queue!r} in {queue_registry_path}"
            )
        if queue not in queues_by_name:
            queues_by_name[queue] = _load_queue(queue_registry, queue_registry_path, queue)
        conditions.append(WorkbenchCondition(
            condition_id=condition_id,
            condition_name=name,
            test_queue=queue,
            applicable_station_ids=applicable,
            trigger_state=trigger_state,
            segmented_analysis=condition_config["segmented_analysis"],
            queue=queues_by_name[queue],
            project_id=project_id,
            result_root_directory=result_root_directory,
            condition_config=condition_config,
        ))
    _unique((condition.condition_id for condition in conditions), product_path, "conditions.condition_id")
    return tuple(conditions)


class MotorWorkbenchConfigRepository:
    """Load the plot layout, active product and its referenced queues from saved files."""

    def __init__(
        self,
        program_registry_path=PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
        queue_registry_path=SEQUENCE_CONFIG_REGISTRY_PATH,
        plot_layout_path=Path(DEFAULT_DIR) / "ui" / "sequence" / "motor_workbench_plot_layout.json",
    ):
        self._program_registry_path = Path(program_registry_path)
        self._queue_registry_path = Path(queue_registry_path)
        self._plot_layout_path = Path(plot_layout_path)

    def load_active(self) -> WorkbenchConfig | None:
        active_file = _active_product_filename(self._program_registry_path)
        if active_file is None:
            return None

        plot_layout = _load_plot_layout(self._plot_layout_path)
        path = self._program_registry_path.parent / active_file
        product = _object(_read_json(path, "active product"), path, "product")
        project_id = _nonempty_string(product.get("project_id"), path, "project_id")
        project_name = _nonempty_string(product.get(PROJECT_NAME_KEY), path, PROJECT_NAME_KEY)
        result_directory = _nonempty_string(
            product.get(RESULT_ROOT_DIRECTORY_KEY), path, RESULT_ROOT_DIRECTORY_KEY
        )
        station_data = _array(product.get("stations"), path, "stations")
        if not 1 <= len(station_data) <= 4:
            raise WorkbenchConfigurationError(f"{path}: stations must contain between 1 and 4 entries")
        if len(station_data) > plot_layout.columns:
            raise WorkbenchConfigurationError(
                f"{path}: {len(station_data)} stations "
                f"exceed the {plot_layout.columns} columns in {self._plot_layout_path}"
            )
        stations = tuple(_load_station(station, path, index) for index, station in enumerate(station_data))
        station_ids = tuple(station.station_id for station in stations)
        _unique(station_ids, path, "stations.station_id")
        conditions = _load_conditions(
            product.get("conditions"), path, set(station_ids), self._queue_registry_path,
            project_id=project_id, result_root_directory=result_directory,
        )
        return WorkbenchConfig(
            project_id=project_id,
            project_name=project_name,
            result_root_directory=result_directory,
            stations=stations,
            conditions=conditions,
            active_file=active_file,
            plot_layout=plot_layout,
        )
