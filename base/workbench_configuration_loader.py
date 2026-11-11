"""Read saved workbench inputs once and assemble responsibility-specific snapshots."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from base.recording_preview_config import resolve_recording_preview_time_mode
from base.workbench_analysis_config import AnalysisTestQueue, WorkbenchAnalysisConfig
from base.workbench_config_values import (
    WorkbenchChannel, WorkbenchCondition, WorkbenchContext, WorkbenchHardwareChannel,
    WorkbenchStation, _freeze_config,
)
from base.workbench_recording_config import WorkbenchRecordingConfig
from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchPlotLayout, WorkbenchUiConfig
from consts.product_test_project_consts import (
    CONDITION_NAME_KEY, PROJECT_NAME_KEY, REGISTRY_ACTIVE_FILE_KEY, REGISTRY_CONFIGS_KEY,
    REGISTRY_FILE_KEY, RESULT_ROOT_DIRECTORY_KEY, TEST_QUEUE_KEY, TRIGGER_STATE_KEY,
)
from consts.recording_preview_consts import RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY


class WorkbenchConfigurationError(ValueError):
    """A saved workbench configuration could not be applied."""


@dataclass(frozen=True)
class WorkbenchConfiguration:
    recording: WorkbenchRecordingConfig
    analysis: WorkbenchAnalysisConfig
    ui: WorkbenchUiConfig


def _config_value(path, field, factory, *args, **kwargs):
    """Add input location to value constructors' snapshot-freezing diagnostics."""
    try:
        return factory(*args, **kwargs)
    except ValueError as exc:
        raise WorkbenchConfigurationError(f"{path}: {field}: {exc}") from (exc.__cause__ or exc)


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


def _read_product_registry(path: Path) -> dict:
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
    return registry


def _active_product_filename(path: Path) -> str | None:
    registry = _read_product_registry(path)
    if REGISTRY_ACTIVE_FILE_KEY not in registry:
        raise WorkbenchConfigurationError(f"{path}: missing {REGISTRY_ACTIVE_FILE_KEY}")
    active_file = registry[REGISTRY_ACTIVE_FILE_KEY]
    if active_file is not None:
        active_file = _nonempty_string(active_file, path, REGISTRY_ACTIVE_FILE_KEY)
        if not any(entry[REGISTRY_FILE_KEY] == active_file for entry in registry[REGISTRY_CONFIGS_KEY]):
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


def _load_plot_layout(path: Path) -> WorkbenchPlotLayout | None:
    try:
        data = _read_json(path, "plot layout")
    except WorkbenchConfigurationError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise
    data = _object(data, path, "plot layout")
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
    if len({channel.hardware_channel.device_ref for channel in channels}) > 1:
        raise WorkbenchConfigurationError(
            f"{path}: {field}.channels.hardware_channel.device_ref must be consistent "
            "within a station"
        )
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


def _validate_acquisition(data, path: Path) -> dict:
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
    value = detail.get("total_time")
    if (
        type(value) not in (int, float)
        or value <= 0
        or (type(value) is float and not math.isfinite(value))
    ):
        raise WorkbenchConfigurationError(f"{path}: {detail_field}.total_time must be a finite positive number")
    try:
        resolve_recording_preview_time_mode(detail)
    except ValueError as exc:
        raise WorkbenchConfigurationError(
            f"{path}: {detail_field}.{RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY}: {exc}"
        ) from exc
    return detail


def _load_queue(queue_registry, registry_path: Path, queue: str):
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
    channel_basis = seq1.get("analysis_channel_basis")
    if channel_basis is not None and not isinstance(channel_basis, str):
        raise WorkbenchConfigurationError(
            f"{path}: [0].seq1.analysis_channel_basis must be a string or null"
        )
    detail = _validate_acquisition(seq1.get("acq"), path)
    analysis = _object(seq1.get("analysis_list"), path, "[0].seq1.analysis_list")
    names = _string_array(
        analysis.get("display_sequence"), path, "[0].seq1.analysis_list.display_sequence"
    )
    for name in names:
        _object(analysis.get(name), path, f"[0].seq1.analysis_list[{name!r}]")
    parameters = {key: detail[key] for key in ("total_time", "startup_trim_ms", "audio_validation")
                  if key in detail}
    parameters = _config_value(
        path, "[0].seq1.acq.detail", _freeze_config, parameters, "recording parameters",
    )
    test_queue = _config_value(
        path, "[0].seq1.analysis_list", AnalysisTestQueue,
        analysis_channel_basis=channel_basis, analysis_config=analysis,
    )
    return parameters, test_queue, resolve_recording_preview_time_mode(detail)


def _load_conditions(
    data, product_path: Path, station_ids: set[str], queue_registry_path: Path,
):
    condition_data = _array(data, product_path, "conditions")
    if not condition_data:
        return (), {}, {}, {}

    queue_registry = _object(
        _read_json(queue_registry_path, "queue registry"), queue_registry_path, "queue registry"
    )
    parameters_by_queue = {}
    test_queues = {}
    preview_time_modes = {}
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
        queue = _nonempty_string(item.get(TEST_QUEUE_KEY), product_path, f"{field}.{TEST_QUEUE_KEY}")
        if queue == "using_config_path" or queue not in queue_registry:
            raise WorkbenchConfigurationError(
                f"{product_path}: {field}.{TEST_QUEUE_KEY} "
                f"references unknown queue {queue!r} in {queue_registry_path}"
            )
        if queue not in test_queues:
            parameters, test_queue, preview = _load_queue(queue_registry, queue_registry_path, queue)
            parameters_by_queue[queue] = parameters
            test_queues[queue] = test_queue
            preview_time_modes[queue] = preview
        conditions.append(_config_value(
            product_path, segmented_field, WorkbenchCondition,
            condition_id=condition_id,
            condition_name=name,
            test_queue=queue,
            applicable_station_ids=applicable,
            trigger_state=trigger_state,
            segmented_analysis=segmented,
        ))
    _unique((condition.condition_id for condition in conditions), product_path, "conditions.condition_id")
    return tuple(conditions), parameters_by_queue, test_queues, preview_time_modes


def load_workbench_configuration(
    program_registry_path, queue_registry_path, plot_layout_path,
) -> WorkbenchConfiguration | None:
    """Load one immutable snapshot without writing inputs or result directories."""
    program_registry_path = Path(program_registry_path)
    queue_registry_path = Path(queue_registry_path)
    plot_layout_path = Path(plot_layout_path)
    active_file = _active_product_filename(program_registry_path)
    return _load_active_configuration(program_registry_path, queue_registry_path, plot_layout_path, active_file)


def _load_active_configuration(program_registry_path, queue_registry_path, plot_layout_path, active_file):
    if active_file is None:
        return None

    plot_layout = _load_plot_layout(plot_layout_path)
    path = program_registry_path.parent / active_file
    product = _object(_read_json(path, "active product"), path, "product")
    project_id = _nonempty_string(product.get("project_id"), path, "project_id")
    project_name = _nonempty_string(product.get(PROJECT_NAME_KEY), path, PROJECT_NAME_KEY)
    result_directory = _nonempty_string(
        product.get(RESULT_ROOT_DIRECTORY_KEY), path, RESULT_ROOT_DIRECTORY_KEY
    )
    station_data = _array(product.get("stations"), path, "stations")
    if not 1 <= len(station_data) <= 4:
        raise WorkbenchConfigurationError(f"{path}: stations must contain between 1 and 4 entries")
    if plot_layout is not None and len(station_data) > plot_layout.columns:
        raise WorkbenchConfigurationError(
            f"{path}: {len(station_data)} stations "
            f"exceed the {plot_layout.columns} columns in {plot_layout_path}"
        )
    stations = tuple(_load_station(station, path, index) for index, station in enumerate(station_data))
    station_ids = tuple(station.station_id for station in stations)
    _unique(station_ids, path, "stations.station_id")
    conditions, parameters, test_queues, previews = _load_conditions(
        product.get("conditions"), path, set(station_ids), queue_registry_path,
    )
    context = WorkbenchContext(
        project_id=project_id,
        project_name=project_name,
        result_root_directory=result_directory,
        stations=stations,
        conditions=conditions,
    )
    return WorkbenchConfiguration(
        recording=WorkbenchRecordingConfig(context=context, parameters_by_queue=parameters),
        analysis=WorkbenchAnalysisConfig(context=context, test_queues=test_queues),
        ui=WorkbenchUiConfig(layout=plot_layout, preview_time_modes=previews),
    )
