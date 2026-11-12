"""Immutable workbench display data derived without retaining analysis test queues."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from base.workbench_analysis_config import WorkbenchAnalysisConfig
from base.workbench_config_values import WorkbenchCondition
from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchUiConfig
from ui.sequence.station_plot_grid import WorkbenchPlotCellDisplay
from ui.sequence.workbench_units import format_workbench_spl_unit


@dataclass(frozen=True)
class WorkbenchPlotBinding:
    station_id: str
    condition_id: str
    analysis_item: str
    analysis_type: str


_PlotBindings = tuple[WorkbenchPlotBinding | None, WorkbenchPlotBinding | None]


@dataclass(frozen=True)
class WorkbenchUiProjection:
    stations: tuple[tuple[str, tuple[str, ...]], ...]
    metric_keys: Mapping[str, tuple[str, ...]]  # station ID -> config keys in row order
    column_names: tuple[str, ...]
    cells: tuple[tuple[WorkbenchPlotCellDisplay, ...], ...]
    bindings: tuple[tuple[_PlotBindings, ...], ...]
    preview_time_modes: Mapping[str, str]  # condition ID -> display mode


def _station_display(
    analysis_config: WorkbenchAnalysisConfig, metric_keys: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (station.station_name, metric_keys[station.station_id])
        for station in analysis_config.context.stations
    )


def _plot_unit(analysis: Mapping[str, object]) -> str:
    return format_workbench_spl_unit(analysis.get("weighting"))


def _plot_cell(
    cell: WorkbenchPlotCell, station_id: str, conditions: tuple[WorkbenchCondition, ...],
    analysis_config: WorkbenchAnalysisConfig,
) -> tuple[WorkbenchPlotCellDisplay, _PlotBindings]:
    names = ["", ""]
    units = ["", ""]
    bindings = [None, None]
    issues = []
    for slot, condition in enumerate(conditions):
        names[slot] = condition.condition_name
        test_queue = analysis_config.test_queues[condition.test_queue].analysis_config
        if cell.analysis_item not in test_queue["display_sequence"]:
            issues.append(f"{condition.condition_name}：缺少分析项「{cell.analysis_item}」")
            continue
        analysis = test_queue[cell.analysis_item]
        if str(analysis.get("type") or "").strip() != cell.analysis_type:
            issues.append(f"{condition.condition_name}：分析项「{cell.analysis_item}」类型不符")
            continue
        units[slot] = _plot_unit(analysis)
        bindings[slot] = WorkbenchPlotBinding(
            station_id, condition.condition_id, cell.analysis_item, cell.analysis_type,
        )

    warning = "分析项不可用" if issues else ""
    if units[0] and units[1] and units[0] != units[1]:
        warning = "单位不一致"
        issues = [f"{name} {unit}" for name, unit in zip(names, units)]
        names[1] = ""
        bindings[1] = None

    display = WorkbenchPlotCellDisplay(
        title=cell.title, unit=units[0] or units[1],
        first_condition=names[0], second_condition=names[1],
        warning=warning, warning_detail="\n".join(issues),
    )
    return display, (bindings[0], bindings[1])


def build_ui_projection(
    ui_config: WorkbenchUiConfig, analysis_config: WorkbenchAnalysisConfig,
) -> WorkbenchUiProjection:
    """Project validated settings into display values in product and queue order."""
    context = analysis_config.context
    layout = ui_config.layout
    keys_by_station = {station.station_id: {} for station in context.stations}
    for condition in context.conditions:
        sequence = analysis_config.test_queues[condition.test_queue].analysis_config["display_sequence"]
        for station_id in condition.applicable_station_ids:
            keys_by_station[station_id].update(dict.fromkeys(sequence))
    metric_keys = MappingProxyType({
        station_id: tuple(keys) for station_id, keys in keys_by_station.items()
    })
    preview_time_modes = MappingProxyType({
        condition.condition_id: ui_config.preview_time_modes[condition.test_queue]
        for condition in context.conditions
    })
    if layout is None:
        return WorkbenchUiProjection(
            stations=_station_display(analysis_config, metric_keys), metric_keys=metric_keys,
            column_names=(), cells=(), bindings=(),
            preview_time_modes=preview_time_modes,
        )
    padding = layout.columns - len(context.stations)
    names = tuple(station.station_name for station in context.stations) + ("",) * padding
    station_conditions = tuple(
        tuple(condition for condition in context.conditions
              if station.station_id in condition.applicable_station_ids)[:2]
        for station in context.stations
    )
    display_rows = []
    binding_rows = []
    empty_cell = WorkbenchPlotCellDisplay()
    for row in layout.cells:
        displays = [empty_cell] * layout.columns
        bindings: list[_PlotBindings] = [(None, None)] * layout.columns
        for column, (station, conditions) in enumerate(zip(context.stations, station_conditions)):
            displays[column], bindings[column] = _plot_cell(
                row[column], station.station_id, conditions, analysis_config,
            )
        display_rows.append(tuple(displays))
        binding_rows.append(tuple(bindings))
    return WorkbenchUiProjection(
        stations=_station_display(analysis_config, metric_keys),
        metric_keys=metric_keys,
        column_names=names,
        cells=tuple(display_rows),
        bindings=tuple(binding_rows),
        preview_time_modes=preview_time_modes,
    )
