"""Immutable workbench display data derived without retaining analysis test queues."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from base.workbench_analysis_config import WorkbenchAnalysisConfig
from base.workbench_config_values import WorkbenchCondition
from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchUiConfig
from ui.sequence.workbench_units import format_workbench_spl_unit
from ui.sequence.workbench_plot_display import (
    ConditionSegmentDisplay, ConditionPlotDisplay, WorkbenchPlotCellDisplay,
)


@dataclass(frozen=True)
class WorkbenchPlotBinding:
    station_id: str
    condition_id: str
    analysis_item: str
    analysis_type: str
    wav_channels: tuple[int, ...] = ()


_PlotBindings = tuple[WorkbenchPlotBinding | None, ...]


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


def _configured_plot_cell(cell, station_id, analysis_config, durations):
    settings = cell.display_settings
    if settings.get('data_source') == 'simulated':
        # Explicit presentation-only data; no queue or physical channel binding.
        panels = tuple(ConditionSegmentDisplay(
            condition.condition_id, condition.condition_name, durations.get(condition.condition_id),
            tuple((ch, f'模拟通道{ch}') for ch in settings['channel_display']['display_wav_channels']),
            's', settings['value_unit'], '',
        ) for condition in analysis_config.context.conditions if station_id in condition.applicable_station_ids)
        return WorkbenchPlotCellDisplay(
            title=f'{cell.title}（模拟数据）', warning_detail='仅用于展示，不参与测试判定或结果保存',
            condition_plot=ConditionPlotDisplay(settings, panels, station_id),
        ), (None,) * len(panels)
    station = next(item for item in analysis_config.context.stations if item.station_id == station_id)
    panels, bindings, issues = [], [], []
    for condition in analysis_config.context.conditions:
        if station_id not in condition.applicable_station_ids:
            continue
        condition_id, name = condition.condition_id, condition.condition_name
        warning, unit, selected = '', '', ()
        queue = analysis_config.test_queues[condition.test_queue].analysis_config
        analysis = queue.get(cell.analysis_item)
        if cell.analysis_item not in queue['display_sequence'] or analysis is None:
            warning = f'缺少分析项「{cell.analysis_item}」'
        elif str(analysis.get('type') or '').strip() != cell.analysis_type:
            warning = f'分析项「{cell.analysis_item}」类型不符'
        else:
            if cell.analysis_type == 'Spec':
                unit = 'Hz'
            elif cell.analysis_type == 'VIBRATION_RMS':
                unit = settings['value_unit']
            else:
                unit = 'dB' if cell.analysis_type == 'FFT' and analysis.get('baseline_display_mode') == 'delta' else _plot_unit(analysis)
            analyzed = analysis.get('analysis_channels', (analysis.get('analysis_channel', 0),))
            if not isinstance(analyzed, (list, tuple)) or any(type(ch) is not int or ch < 0 for ch in analyzed):
                warning = '分析项的通道配置无效'
            else:
                selection = settings['channel_display']
                wanted = tuple(ch + 1 for ch in analyzed) if selection['show_all_analyzed_channels'] else selection['display_wav_channels']
                invalid = [ch for ch in wanted if ch > len(station.channels) or ch - 1 not in analyzed]
                if invalid:
                    warning = f'WAV 通道 {list(invalid)} 不存在或未配置该分析项'
                elif not wanted:
                    warning = '分析项没有可显示的通道'
                elif cell.analysis_type == 'Spec' and len(wanted) != 1:
                    warning = '每个时频图只显示一个 WAV 通道，请在绘图配置中指定通道'
                elif cell.analysis_type == 'VIBRATION_RMS' and any(
                        station.channels[ch - 1].signal_type != 'acceleration' for ch in wanted):
                    warning = '振动 RMS 只能显示加速度通道'
                else:
                    selected = tuple((ch, station.channels[ch - 1].channel_name) for ch in wanted)
        if warning:
            issues.append(f'{name}：{warning}')
        bindings.append(None if warning else WorkbenchPlotBinding(
            station_id, condition_id, cell.analysis_item, cell.analysis_type,
            tuple(ch for ch, _ in selected),
        ))
        panels.append(ConditionSegmentDisplay(
            condition_id, name, durations.get(condition_id), selected,
            'Hz' if cell.analysis_type == 'FFT' else 's', unit, warning,
        ))
    units = {panel.y_unit for panel in panels if not panel.warning}
    if len(units) > 1:
        warning = '工况单位不一致，不能共用纵轴'
        issues.append(warning)
        panels = [replace(panel, warning=panel.warning or warning) for panel in panels]
        bindings = [None] * len(panels)
    if not panels:
        issues.append('当前工位没有适用工况')
    return WorkbenchPlotCellDisplay(
        title=cell.title, warning='配置不匹配' if issues else '', warning_detail='\n'.join(issues),
        condition_plot=ConditionPlotDisplay(
            settings, tuple(panels),
            summary_rows=max((len(panel.channels) for panel in panels), default=0)
            if cell.analysis_type == 'SPL' else 0,
        ),
    ), tuple(bindings)


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
            cell = row[column]
            if cell.display_settings is not None:
                displays[column], bindings[column] = _configured_plot_cell(
                    cell, station.station_id, analysis_config, ui_config.condition_durations,
                )
            else:
                displays[column], bindings[column] = _plot_cell(
                    cell, station.station_id, conditions, analysis_config,
                )
        display_rows.append(tuple(displays))
        binding_rows.append(tuple(bindings))
    # Align line-chart headings using configured summary space, never live results.
    summary_rows = max((
        cell.condition_plot.summary_rows for row in display_rows for cell in row
        if cell.condition_plot is not None
    ), default=0)
    display_rows = [tuple(
        replace(cell, condition_plot=replace(cell.condition_plot, summary_rows=summary_rows))
        if cell.condition_plot is not None and cell.condition_plot.settings['chart_type'] == 'line'
        else cell
        for cell in row
    ) for row in display_rows]
    return WorkbenchUiProjection(
        stations=_station_display(analysis_config, metric_keys),
        metric_keys=metric_keys,
        column_names=names,
        cells=tuple(display_rows),
        bindings=tuple(binding_rows),
        preview_time_modes=preview_time_modes,
    )
