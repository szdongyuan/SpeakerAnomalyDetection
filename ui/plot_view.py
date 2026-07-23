"""Shared analysis-plot view range configuration and runtime application."""

from __future__ import annotations

import math

from consts.acoustic_analysis.curve_style_consts import (
    DEFAULT_PLOT_VIEW_CONFIG,
    PLOT_VIEW_KEY,
)


def _normalize_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_plot_view_config(plot_view_config):
    """Return a stable plot-view dictionary from an untrusted config payload."""
    source = plot_view_config if isinstance(plot_view_config, dict) else {}
    normalized = dict(DEFAULT_PLOT_VIEW_CONFIG)
    normalized["x_enabled"] = _normalize_enabled(source.get("x_enabled", False))
    normalized["y_enabled"] = _normalize_enabled(source.get("y_enabled", False))
    valid_bounds = {}
    for key in ("x_min", "x_max", "y_min", "y_max"):
        number = _finite_number(source.get(key)) if key in source else None
        valid_bounds[key] = number is not None
        normalized[key] = number
    for axis_name in ("x", "y"):
        enabled_key = f"{axis_name}_enabled"
        lower_key = f"{axis_name}_min"
        upper_key = f"{axis_name}_max"
        if not normalized[enabled_key] or not (
            valid_bounds[lower_key] and valid_bounds[upper_key]
        ):
            normalized[enabled_key] = False
            normalized[lower_key] = None
            normalized[upper_key] = None
    return normalized


def resolve_plot_view_config(config):
    """Resolve ``display.plot_view`` or return None when it is not configured."""
    if not isinstance(config, dict):
        return None
    display_config = config.get("display")
    if not isinstance(display_config, dict):
        return None
    plot_view_config = display_config.get(PLOT_VIEW_KEY)
    if not isinstance(plot_view_config, dict):
        return None
    return normalize_plot_view_config(plot_view_config)


def build_plot_view_config(config, plot_view_config):
    """Build nested display config while preserving colors and future fields."""
    source_display = config.get("display", {}) if isinstance(config, dict) else {}
    display_config = dict(source_display) if isinstance(source_display, dict) else {}
    source_plot_view = display_config.get(PLOT_VIEW_KEY, {})
    merged_plot_view = dict(source_plot_view) if isinstance(source_plot_view, dict) else {}
    incoming = plot_view_config if isinstance(plot_view_config, dict) else {}
    for axis_name in ("x", "y"):
        enabled_key = f"{axis_name}_enabled"
        lower_key = f"{axis_name}_min"
        upper_key = f"{axis_name}_max"
        enabled = _normalize_enabled(
            incoming.get(enabled_key, merged_plot_view.get(enabled_key, False))
        )
        lower = _finite_number(incoming.get(lower_key, merged_plot_view.get(lower_key)))
        upper = _finite_number(incoming.get(upper_key, merged_plot_view.get(upper_key)))
        if not enabled or lower is None or upper is None:
            merged_plot_view[enabled_key] = False
            merged_plot_view.pop(lower_key, None)
            merged_plot_view.pop(upper_key, None)
            continue
        merged_plot_view[enabled_key] = True
        merged_plot_view[lower_key] = lower
        merged_plot_view[upper_key] = upper
    display_config[PLOT_VIEW_KEY] = merged_plot_view
    return {"display": display_config}


def _axis_view_range(plot_view_config, axis_name, log_mode):
    if not plot_view_config.get(f"{axis_name}_enabled", False):
        return None
    lower = float(plot_view_config[f"{axis_name}_min"])
    upper = float(plot_view_config[f"{axis_name}_max"])
    if lower >= upper:
        return None
    if log_mode:
        if lower <= 0.0 or upper <= 0.0:
            return None
        lower = math.log10(lower)
        upper = math.log10(upper)
    return lower, upper


def apply_plot_view_range(plot_widget, config, allow_x=True, allow_y=True):
    """Apply configured physical ranges to a pyqtgraph PlotWidget once."""
    plot_view_config = resolve_plot_view_config(config)
    if plot_view_config is None:
        return False

    bottom_axis = plot_widget.getAxis("bottom")
    left_axis = plot_widget.getAxis("left")
    x_range = None
    y_range = None
    if allow_x:
        x_range = _axis_view_range(
            plot_view_config,
            "x",
            bool(getattr(bottom_axis, "logMode", False)),
        )
    if allow_y:
        y_range = _axis_view_range(
            plot_view_config,
            "y",
            bool(getattr(left_axis, "logMode", False)),
        )

    view_box = plot_widget.getViewBox()
    applied = False
    if x_range is not None:
        view_box.setXRange(x_range[0], x_range[1], padding=0.0)
        applied = True
    if y_range is not None:
        view_box.setYRange(y_range[0], y_range[1], padding=0.0)
        applied = True
    return applied
