"""Pure curve-color configuration helpers for analysis runtimes and UI."""

import re

from consts.acoustic_analysis.curve_style_consts import DEFAULT_CURVE_COLORS


def normalize_curve_color(value, fallback):
    """Return an uppercase #RRGGBB color or the supplied fallback."""
    if not isinstance(value, str):
        return fallback
    color = value.strip().upper()
    if re.fullmatch(r"#[0-9A-F]{6}", color):
        return color
    return fallback


def resolve_curve_colors(config):
    """Resolve curve colors from an analysis configuration dictionary."""
    display_config = config.get("display", {}) if isinstance(config, dict) else {}
    if not isinstance(display_config, dict):
        display_config = {}
    return {
        key: normalize_curve_color(display_config.get(key), default)
        for key, default in DEFAULT_CURVE_COLORS.items()
    }


def build_curve_color_config(config, colors):
    """Build the nested display config while preserving unrelated fields."""
    source_display = config.get("display", {}) if isinstance(config, dict) else {}
    display_config = dict(source_display) if isinstance(source_display, dict) else {}
    for key, default in DEFAULT_CURVE_COLORS.items():
        display_config[key] = normalize_curve_color(colors.get(key), default)
    return {"display": display_config}


__all__ = [
    "build_curve_color_config",
    "normalize_curve_color",
    "resolve_curve_colors",
]
