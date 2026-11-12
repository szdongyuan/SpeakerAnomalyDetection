"""Resolve and validate plot templates without Qt, file I/O or analysis execution."""

from copy import deepcopy
import math
import re

from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchPlotLayout


LINE_STYLES = ("solid", "dashed", "dotted", "dash_dot")
DEFAULT_STYLE = {
    "background_color": "#FFFFFF", "curve_line_width": 1.5,
    "curve_line_style": "solid", "channel_curve_styles": [],
    "grid": {"visible": False, "color": "#E5EAF0", "line_style": "solid", "line_width": 1.0},
    "legend": {"visible": True, "position": "top_right"},
    "exceedance_highlight": {"visible": True, "color": "#E5484D", "line_style": "solid", "line_width": 2.0, "marker_size": 5},
    "limit_lines": {
        "visible": True,
        "upper": {"color": "#D9534F", "line_style": "dashed", "line_width": 1.0},
        "lower": {"color": "#800080", "line_style": "dashed", "line_width": 1.0},
    },
}
CELL_FIELDS = {
    "title", "analysis_item_name", "analysis_type", "chart_type", "channel_display",
    "plot_style", "condition_display", "x_axis", "y_axis", "display_downsampling", "spectrogram", "value_unit", "data_source",
}


def _object(value, field, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{field}: 必须是对象")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown:
        raise ValueError(f"{field}: 不支持的字段 {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{field}: 缺少字段 {', '.join(sorted(missing))}")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: 必须是非空文字")
    return value


def _choice(value, choices, field):
    if value not in choices:
        raise ValueError(f"{field}: 必须为 {', '.join(choices)}")


def _number(value, field, minimum=0, *, integer=False, strict=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"{field}: 数值无效")
    return value


def _bool(value, field):
    if type(value) is not bool:
        raise ValueError(f"{field}: 必须是 true 或 false")


def _array(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field}: 必须是非空数组")
    return value


def _color(value, field):
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError(f"{field}: 必须是 #RRGGBB 颜色")


def _pen(data, field):
    _object(data, field, ("color", "line_style", "line_width"), ("color", "line_style", "line_width"))
    _color(data["color"], f"{field}.color")
    _choice(data["line_style"], LINE_STYLES, f"{field}.line_style")
    _number(data["line_width"], f"{field}.line_width", strict=True)


def _style(data, field):
    _object(data, field, DEFAULT_STYLE, DEFAULT_STYLE)
    _color(data["background_color"], f"{field}.background_color")
    _choice(data["curve_line_style"], LINE_STYLES, f"{field}.curve_line_style")
    _number(data["curve_line_width"], f"{field}.curve_line_width", strict=True)
    entries = data["channel_curve_styles"]
    if not isinstance(entries, list):
        raise ValueError(f"{field}.channel_curve_styles: 必须是数组")
    seen = set()
    for index, entry in enumerate(entries):
        key = f"{field}.channel_curve_styles[{index}]"
        _object(entry, key, ("wav_channel", "curve_color", "curve_line_style", "curve_line_width"),
                ("wav_channel", "curve_color"))
        channel = _number(entry["wav_channel"], f"{key}.wav_channel", 1, integer=True)
        if channel in seen:
            raise ValueError(f"{key}.wav_channel: 通道重复")
        seen.add(channel)
        _color(entry["curve_color"], f"{key}.curve_color")
        if "curve_line_style" in entry:
            _choice(entry["curve_line_style"], LINE_STYLES, f"{key}.curve_line_style")
        if "curve_line_width" in entry:
            _number(entry["curve_line_width"], f"{key}.curve_line_width", strict=True)
    grid = data["grid"]
    _object(grid, f"{field}.grid", ("visible", "color", "line_style", "line_width"),
            ("visible", "color", "line_style", "line_width"))
    _bool(grid["visible"], f"{field}.grid.visible")
    _pen({k: v for k, v in grid.items() if k != "visible"}, f"{field}.grid")
    legend = data["legend"]
    _object(legend, f"{field}.legend", ("visible", "position"), ("visible", "position"))
    _bool(legend["visible"], f"{field}.legend.visible")
    _choice(legend["position"], ("top_left", "top_right", "bottom_left", "bottom_right"), f"{field}.legend.position")
    limits = data["limit_lines"]
    _object(limits, f"{field}.limit_lines", ("visible", "upper", "lower"), ("visible", "upper", "lower"))
    _bool(limits["visible"], f"{field}.limit_lines.visible")
    for side in ("upper", "lower"):
        _pen(limits[side], f"{field}.limit_lines.{side}")
    highlight = data['exceedance_highlight']
    _object(highlight, f'{field}.exceedance_highlight',
            ('visible', 'color', 'line_style', 'line_width', 'marker_size'),
            ('visible', 'color', 'line_style', 'line_width', 'marker_size'))
    _bool(highlight['visible'], f'{field}.exceedance_highlight.visible')
    _pen({k: highlight[k] for k in ('color', 'line_style', 'line_width')}, f'{field}.exceedance_highlight')
    _number(highlight['marker_size'], f'{field}.exceedance_highlight.marker_size', strict=True)


def _axis(data, field):
    empty_fields = ("empty_data_range_min", "empty_data_range_max") if field.endswith('.y_axis') else ()
    _object(data, field, ("label", "scale", "auto_range", "range_min", "range_max", *empty_fields),
            ("label", "scale", "auto_range"))
    _text(data["label"], f"{field}.label")
    _choice(data["scale"], ("linear", "log"), f"{field}.scale")
    _bool(data["auto_range"], f"{field}.auto_range")
    if any(key in data for key in empty_fields):
        if not data['auto_range']:
            raise ValueError(f'{field}: 无数据时的范围仅用于自动纵轴')
        low = _number(data.get('empty_data_range_min'), f'{field}.empty_data_range_min', -math.inf)
        high = _number(data.get('empty_data_range_max'), f'{field}.empty_data_range_max', -math.inf)
        if low >= high or (data['scale'] == 'log' and low <= 0):
            raise ValueError(f'{field}: 无数据时的范围必须递增，对数轴范围必须大于零')
    if data["auto_range"]:
        if "range_min" in data or "range_max" in data:
            raise ValueError(f"{field}: 自动范围不能同时指定 range_min / range_max")
    else:
        low = _number(data.get("range_min"), f"{field}.range_min", -math.inf)
        high = _number(data.get("range_max"), f"{field}.range_max", -math.inf)
        if low >= high or (data["scale"] == "log" and low <= 0):
            raise ValueError(f"{field}: 范围必须递增，对数轴范围必须大于零")


def _conditions(data, field):
    fields = {"layout", "condition_source", "x_axis_mode", "show_condition_names",
              "show_condition_separator"}
    _object(data, field, fields | {"condition_gap_px"}, fields)
    _number(data.get("condition_gap_px", 16), f"{field}.condition_gap_px", 0, integer=True)
    _choice(data["layout"], ("single_canvas",), f"{field}.layout")
    _choice(data["condition_source"], ("current_configuration",), f"{field}.condition_source")
    _choice(data["x_axis_mode"], ("cumulative_time", "condition_local"), f"{field}.x_axis_mode")
    for key in ("show_condition_names", "show_condition_separator"):
        _bool(data[key], f"{field}.{key}")


def _downsampling(data, field):
    _object(data, field, ("method", "pixel_bucket_width", "keep_one_point_every_n_points"), ("method",))
    method = data["method"]
    _choice(method, ("none", "preserve_min_max", "keep_every_nth_point"), f"{field}.method")
    parameter = {"preserve_min_max": "pixel_bucket_width",
                 "keep_every_nth_point": "keep_one_point_every_n_points"}.get(method)
    fields = ("method", parameter) if parameter else ("method",)
    _object(data, field, fields, fields)
    if parameter:
        _number(data[parameter], f"{field}.{parameter}", 1, integer=True)
    return dict(data)


def _cell(data, style, conditions, field):
    _object(data, field, CELL_FIELDS,
            CELL_FIELDS - {"plot_style", "condition_display", "display_downsampling", "spectrogram", "value_unit", "data_source"})
    for key in ("title", "analysis_item_name"):
        _text(data[key], f"{field}.{key}")
    _choice(data["analysis_type"], ("SPL", "FFT", "Spec", "VIBRATION_RMS"), f"{field}.analysis_type")
    _choice(data["chart_type"], ("heatmap",) if data['analysis_type'] == 'Spec' else ("line",), f"{field}.chart_type")
    if data['analysis_type'] == 'VIBRATION_RMS':
        _choice(data.get('value_unit'), ('m/s²', 'g'), f'{field}.value_unit')
    elif 'value_unit' in data:
        raise ValueError(f'{field}.value_unit: 仅用于振动 RMS，声学单位由分析结果决定')
    channels = data["channel_display"]
    _object(channels, f"{field}.channel_display", ("show_all_analyzed_channels", "display_wav_channels"),
            ("show_all_analyzed_channels",))
    _bool(channels["show_all_analyzed_channels"], f"{field}.channel_display.show_all_analyzed_channels")
    if channels["show_all_analyzed_channels"]:
        if "display_wav_channels" in channels:
            raise ValueError(f"{field}.channel_display: 显示全部通道时不能再指定通道列表")
    else:
        selected = _array(channels.get("display_wav_channels"), f"{field}.channel_display.display_wav_channels")
        for ch in selected:
            _number(ch, f"{field}.channel_display.display_wav_channels", 1, integer=True)
        if len(set(selected)) != len(selected):
            raise ValueError(f"{field}.channel_display: 通道重复")
    local_style = _object(data.get("plot_style", {}), f"{field}.plot_style", DEFAULT_STYLE)
    resolved_style = {**style, **local_style}
    _style(resolved_style, f"{field}.plot_style")
    resolved_conditions = data.get("condition_display", conditions)
    _conditions(resolved_conditions, f"{field}.condition_display")
    for axis in ("x_axis", "y_axis"):
        _axis(data[axis], f"{field}.{axis}")
    if resolved_conditions["x_axis_mode"] == "cumulative_time" and (
            data["analysis_type"] == "FFT" or data["x_axis"]["scale"] != "linear"):
        raise ValueError(f"{field}.condition_display.x_axis_mode: 累计时间仅用于线性时间轴")
    source = data.get('data_source', 'analysis_result')
    _choice(source, ('analysis_result', 'simulated'), f'{field}.data_source')
    if source == 'simulated' and (data['analysis_type'] != 'VIBRATION_RMS'
                                 or channels['show_all_analyzed_channels']):
        raise ValueError(f'{field}.data_source: 模拟数据仅用于振动 RMS，且必须指定展示通道')
    settings = {key: data[key] for key in ("chart_type", "channel_display", "x_axis", "y_axis")}
    settings['data_source'] = source
    if 'value_unit' in data:
        settings['value_unit'] = data['value_unit']
    if data['analysis_type'] == 'Spec':
        if data['x_axis']['scale'] != 'linear':
            raise ValueError(f'{field}.x_axis.scale: 时频图时间轴必须为 linear')
        heatmap = data.get('spectrogram', {})
        _object(heatmap, f'{field}.spectrogram', ('color_map', 'show_color_bar', 'color_range'))
        heatmap = {'color_map': 'viridis', 'show_color_bar': True,
                   'color_range': {'mode': 'auto'}, **heatmap}
        _choice(heatmap['color_map'], ('viridis', 'plasma', 'inferno', 'magma'), f'{field}.spectrogram.color_map')
        _bool(heatmap['show_color_bar'], f'{field}.spectrogram.show_color_bar')
        levels = heatmap['color_range']
        _object(levels, f'{field}.spectrogram.color_range', ('mode', 'min_db', 'max_db'), ('mode',))
        _choice(levels['mode'], ('auto', 'fixed'), f'{field}.spectrogram.color_range.mode')
        if levels['mode'] == 'fixed':
            low = _number(levels.get('min_db'), f'{field}.spectrogram.color_range.min_db', -math.inf)
            high = _number(levels.get('max_db'), f'{field}.spectrogram.color_range.max_db', -math.inf)
            if low >= high:
                raise ValueError(f'{field}.spectrogram.color_range: 颜色范围必须递增')
        elif len(levels) != 1:
            raise ValueError(f'{field}.spectrogram.color_range: auto 不能指定 min_db / max_db')
        settings['spectrogram'] = heatmap
    elif 'spectrogram' in data:
        raise ValueError(f'{field}.spectrogram: 仅用于 Spec 时频图')
    settings.update(plot_style=resolved_style, condition_display={"condition_gap_px": 16, **resolved_conditions})
    settings['display_downsampling'] = _downsampling(
        data.get('display_downsampling', {"method": "preserve_min_max", "pixel_bucket_width": 1}),
        f'{field}.display_downsampling')
    return WorkbenchPlotCell(data["title"], data["analysis_item_name"], data["analysis_type"], settings)


def parse_plot_templates(data):
    """Expand defaults -> named template -> position override into immutable cells."""
    _object(data, "$", ("rows", "columns", "cells", "plot_templates", "default_plot_style", "default_condition_display"),
            ("rows", "columns", "cells", "plot_templates"))
    rows = _number(data["rows"], "rows", 1, integer=True)
    columns = _number(data["columns"], "columns", 1, integer=True)
    if rows > 3 or columns > 4:
        raise ValueError("rows / columns: 当前工作台最多支持 3 行、4 个工位列")
    defaults = _object(data.get("default_plot_style", {}), "default_plot_style", DEFAULT_STYLE)
    style = {**deepcopy(DEFAULT_STYLE), **defaults}
    _style(style, "default_plot_style")
    conditions = data.get("default_condition_display")
    if conditions is not None:
        _conditions(conditions, "default_condition_display")
    templates = data["plot_templates"]
    if not isinstance(templates, dict) or not templates:
        raise ValueError("plot_templates: 必须包含至少一个模板")
    for name, template in templates.items():
        _text(name, "plot_templates key")
        _cell(template, style, conditions, f"plot_templates.{name}")
    cell_rows = _array(data["cells"], "cells")
    if len(cell_rows) != rows:
        raise ValueError("cells: 行数与 rows 不一致")
    resolved = []
    for row, values in enumerate(cell_rows):
        if len(_array(values, f"cells[{row}]")) != columns:
            raise ValueError(f"cells[{row}]: 列数与 columns 不一致")
        resolved_row = []
        for column, reference in enumerate(values):
            field = f"cells[{row}][{column}]"
            _object(reference, field, ("plot_template", "overrides"), ("plot_template",))
            name = _text(reference["plot_template"], f"{field}.plot_template")
            if name not in templates:
                raise ValueError(f"{field}.plot_template: 模板 {name!r} 不存在")
            overrides = _object(reference.get("overrides", {}), f"{field}.overrides", CELL_FIELDS)
            if "plot_style" in overrides:
                _object(overrides["plot_style"], f"{field}.overrides.plot_style", DEFAULT_STYLE)
            template = {**templates[name], **overrides}
            template["plot_style"] = {
                **templates[name].get("plot_style", {}), **overrides.get("plot_style", {}),
            }
            resolved_row.append(_cell(template, style, conditions, field))
        resolved.append(tuple(resolved_row))
    return WorkbenchPlotLayout(rows, columns, tuple(resolved))
