"""Coordinate product configuration and user intents across workbench views.

Send cross-widget business updates through the controller. Views render data
and emit intents. The repository reads committed configuration, and the
configuration editors own writes and send commit notifications to the caller.
"""

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from base.motor_workbench_config import (
    WorkbenchChannel,
    WorkbenchCondition,
    WorkbenchConfig,
    WorkbenchConfigurationError,
    WorkbenchPlotCell,
)
from ui.sequence.station_plot_grid import WorkbenchPlotCellDisplay


@dataclass(frozen=True)
class WorkbenchPlotBinding:
    station_id: str
    condition_id: str
    analysis_item: str
    analysis_type: str


_PlotBindings = tuple[WorkbenchPlotBinding | None, WorkbenchPlotBinding | None]


def _station_display(config: WorkbenchConfig) -> tuple[tuple[str, tuple[str, ...]], ...]:
    metrics_by_station = {station.station_id: [] for station in config.stations}
    for condition in config.conditions:
        for station_id in condition.applicable_station_ids:
            metrics_by_station[station_id].extend(
                (metric, condition.condition_name) for metric in condition.metric_names
            )

    display = []
    for station in config.stations:
        metrics = metrics_by_station[station.station_id]
        counts = Counter(metric for metric, _ in metrics)
        labels = tuple(
            f"{metric}（{condition}）" if counts[metric] > 1 else metric
            for metric, condition in metrics
        )
        display.append((station.station_name, labels))
    return tuple(display)


def _plot_unit(analysis: Mapping[str, object]) -> str:
    weighting = str(analysis.get("weighting") or "Z").upper()
    if weighting in ("NONE", "Z（NONE）"):
        weighting = "Z"
    return f"dB({weighting})"


def _plot_cell(
    cell: WorkbenchPlotCell, station_id: str, conditions: tuple[WorkbenchCondition, ...],
) -> tuple[WorkbenchPlotCellDisplay, _PlotBindings]:
    names = ["", ""]
    units = ["", ""]
    bindings = [None, None]
    issues = []
    for slot, condition in enumerate(conditions):
        names[slot] = condition.condition_name
        queue = condition.queue
        if cell.analysis_item not in queue.metric_names:
            issues.append(f"{condition.condition_name}：缺少分析项「{cell.analysis_item}」")
            continue
        analysis = queue.analysis[cell.analysis_item]
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


def _plot_projection(config: WorkbenchConfig) -> tuple[
    tuple[str, ...],
    tuple[tuple[WorkbenchPlotCellDisplay, ...], ...],
    tuple[tuple[_PlotBindings, ...], ...],
]:
    layout = config.plot_layout
    padding = layout.columns - len(config.stations)
    names = tuple(station.station_name for station in config.stations) + ("",) * padding
    station_conditions = tuple(
        tuple(condition for condition in config.conditions
              if station.station_id in condition.applicable_station_ids)[:2]
        for station in config.stations
    )
    display_rows = []
    binding_rows = []
    empty_cell = WorkbenchPlotCellDisplay()
    for row in layout.cells:
        displays = [empty_cell] * layout.columns
        bindings: list[_PlotBindings] = [(None, None)] * layout.columns
        for column, (station, conditions) in enumerate(zip(config.stations, station_conditions)):
            displays[column], bindings[column] = _plot_cell(row[column], station.station_id, conditions)
        display_rows.append(tuple(displays))
        binding_rows.append(tuple(bindings))
    return names, tuple(display_rows), tuple(binding_rows)


class MotorWorkbenchController:
    """Apply validated snapshots and route intents on the caller's GUI thread.

    The caller owns Qt signal bindings. The product callback receives an empty
    string when no product is applied.
    """

    def __init__(
        self,
        repository,
        station_panel,
        plot_grid,
        set_active_product: Callable[[str], None],
        show_video_settings: Callable[[], None],
    ):
        self._repository = repository
        self._station_panel = station_panel
        self._plot_grid = plot_grid
        self._set_active_product = set_active_product
        self._show_video_settings = show_video_settings
        self._initialized = False
        self._applied_config: WorkbenchConfig | None = None
        self._plot_bindings: tuple[tuple[_PlotBindings, ...], ...] = ()

    def refresh_configuration(self) -> None:
        """Read the active configuration and replace both displays if it changed."""
        if not self._initialized:
            self._apply_configuration(None)
            self._initialized = True

        try:
            config = self._repository.load_active()
        except WorkbenchConfigurationError as error:
            logging.getLogger("core").warning("Motor workbench configuration: %s", error)
            return

        if config != self._applied_config:
            self._apply_configuration(config)

    def _apply_configuration(self, config: WorkbenchConfig | None) -> None:
        if config is None:
            stations, names, cells, bindings, product_name = (), (), (), (), ""
        else:
            stations = _station_display(config)
            names, cells, bindings = _plot_projection(config)
            product_name = config.project_name

        self._station_panel.show_workbench_stations(stations)
        self._plot_grid.show_workbench_layout(names, cells)
        self._set_active_product(product_name)
        self._applied_config = config
        self._plot_bindings = bindings

    def get_plot_bindings(self, row: int, column: int) -> _PlotBindings:
        """Return the two condition bindings at a zero-based plot position."""
        if 0 <= row < len(self._plot_bindings):
            bindings = self._plot_bindings[row]
            if 0 <= column < len(bindings):
                return bindings[column]
        return None, None

    def get_condition_config(self, condition_id: str) -> WorkbenchCondition | None:
        """Return the saved runtime configuration for a condition."""
        if self._applied_config is None:
            return None
        return self._applied_config.conditions_by_id.get(condition_id)

    def get_station_channels(self, station_id: str) -> tuple[WorkbenchChannel, ...] | None:
        """Return the ordered, shared channel mapping for a station."""
        if self._applied_config is None:
            return None
        return self._applied_config.station_channels_by_id.get(station_id)

    def open_video_settings(self) -> None:
        """Forward the video settings intent to the existing settings entry point."""
        self._show_video_settings()
