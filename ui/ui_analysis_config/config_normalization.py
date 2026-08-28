"""Compatibility helpers for analysis configuration dialogs.

The analysis configuration UI keeps legacy persisted keys stable while newer
dialogs and shared widgets use clearer internal concept names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import numbers
from typing import Any

from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
    GOLDEN_SAMPLE_RESULT_PATH_KEY,
)
from consts.harmonic_detection_consts import HARMONIC_DETECTION_METHOD_KEY


WEIGHTING_OPTIONS = ("Z", "A", "B", "C", "D")
WEIGHTING_DISPLAY_LABELS = {
    "Z": "Z（None）",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}

OCTAVE_SMOOTHING_OPTIONS = (0, 1, 3, 6, 12, 24, 48)

CONFIG_CONCEPTS = {
    "analysis_channel": {
        "meaning": "Input channel selected for this analysis item.",
        "legacy_keys": ("analysis_channel",),
    },
    "weighting": {
        "meaning": "Acoustic weighting curve.",
        "legacy_keys": ("weighting",),
    },
    "frequency_smoothing": {
        "meaning": "Frequency-domain octave smoothing.",
        "legacy_keys": ("octave_smoothing", "smooth_checked"),
    },
    "time_smoothing": {
        "meaning": "Time-domain or point-domain smoothing for event detection.",
        "legacy_keys": (
            "smooth_checked",
            "smooth_enabled",
            "smooth_unit",
            "smooth_time_sec",
            "smooth_points",
            "smooth_algo",
        ),
    },
    "threshold_curve": {
        "meaning": "CSV upper/lower curve threshold.",
        "legacy_keys": ("limit_checked", "limit_data"),
    },
    "reference_threshold": {
        "meaning": "Reference-spectrum dB offset threshold.",
        "legacy_keys": ("enable_threshold_judgment", "lower_offset_db", "upper_offset_db"),
    },
    "golden_sample": {
        "meaning": "Comparison against a golden baseline result.",
        "legacy_keys": (
            GOLDEN_SAMPLE_CHECKED_KEY,
            GOLDEN_SAMPLE_RESULT_PATH_KEY,
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
        ),
    },
    "harmonic_selection": {
        "meaning": "Selected harmonic orders.",
        "legacy_keys": ("selected_labels", "all_checked"),
    },
    "harmonic_detection_method": {
        "meaning": "Standard HD/THD and RB/high-order harmonic detection algorithm.",
        "legacy_keys": (HARMONIC_DETECTION_METHOD_KEY,),
    },
}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_weighting(value: Any, default: str = "Z") -> str:
    """Return a canonical weighting value: Z, A, B, C, or D."""
    normalized_default = str(default or "Z").strip().upper()
    if normalized_default in ("NONE", "Z（NONE）"):
        normalized_default = "Z"
    if normalized_default not in WEIGHTING_OPTIONS:
        normalized_default = "Z"

    if value in (None, ""):
        return normalized_default

    normalized = str(value).strip().upper()
    if normalized in ("NONE", "Z（NONE）"):
        return "Z"
    return normalized if normalized in WEIGHTING_OPTIONS else normalized_default


def weighting_to_display_label(value: Any, default: str = "Z") -> str:
    """Return the dialog label for a weighting value."""
    return WEIGHTING_DISPLAY_LABELS[normalize_weighting(value, default=default)]


def normalize_octave_smoothing(
    cfg: Mapping[str, Any] | None,
    default: int = 0,
    legacy_true_default: int = 6,
) -> int:
    """Return a supported octave smoothing fraction from legacy config keys."""
    config = cfg or {}
    normalized_default = _to_int(default, 0)
    if normalized_default not in OCTAVE_SMOOTHING_OPTIONS:
        normalized_default = 0

    if "octave_smoothing" in config:
        value = _to_int(config.get("octave_smoothing"), normalized_default)
        return value if value in OCTAVE_SMOOTHING_OPTIONS else normalized_default

    if bool(config.get("smooth_checked", False)):
        value = _to_int(legacy_true_default, 6)
        return value if value in OCTAVE_SMOOTHING_OPTIONS else 6

    return normalized_default


def normalize_time_smoothing(
    cfg: Mapping[str, Any] | None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable internal shape for time or point smoothing settings."""
    config = cfg or {}
    fallback = {
        "enabled": False,
        "unit": "time",
        "time_sec": 0.02,
        "points": 0,
        "algo": 1,
    }
    fallback.update(defaults or {})

    unit = str(config.get("smooth_unit", fallback["unit"]) or fallback["unit"]).strip().lower()
    if unit not in ("time", "points"):
        unit = str(fallback["unit"] or "time").strip().lower()
    if unit not in ("time", "points"):
        unit = "time"

    return {
        "enabled": bool(config.get("smooth_enabled", config.get("smooth_checked", fallback["enabled"]))),
        "unit": unit,
        "time_sec": _to_float(config.get("smooth_time_sec", fallback["time_sec"]), float(fallback["time_sec"])),
        "points": _to_int(config.get("smooth_points", fallback["points"]), int(fallback["points"])),
        "algo": _to_int(config.get("smooth_algo", fallback["algo"]), int(fallback["algo"])),
    }


def normalize_analysis_channel(
    cfg: Mapping[str, Any] | None,
    available_channels: Iterable[Any] | None = None,
) -> int:
    """Return a canonical zero-based channel index in the range 0 through 127."""
    _ = available_channels  # Compatibility only; hardware availability is checked at runtime.
    channel = _parse_analysis_channel((cfg or {}).get("analysis_channel", 0))
    return channel if channel is not None else 0


def normalize_analysis_channels(cfg: Mapping[str, Any] | None) -> list[int]:
    """Normalize recorded selections without replacing unavailable channels."""
    config = cfg or {}
    values = config.get("analysis_channels")
    if not isinstance(values, (list, tuple)):
        return [normalize_analysis_channel(config)]
    channels = {_parse_analysis_channel(value) for value in values}
    channels.discard(None)
    return sorted(channels) or [normalize_analysis_channel(config)]


def _parse_analysis_channel(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, numbers.Real):
        try:
            real_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(real_value) or not real_value.is_integer():
            return None
        normalized = int(real_value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdecimal():
            return None
        significant_text = text.lstrip("0") or "0"
        if len(significant_text) > 3:
            return None
        try:
            normalized = int(significant_text)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None

    return normalized if 0 <= normalized <= 127 else None


def normalize_legacy_available_analysis_channel(
    cfg: Mapping[str, Any] | None,
    available_channels: Iterable[Any] | None,
) -> int:
    """Preserve the availability-based coercion used by legacy combo boxes."""
    channels = []
    for channel in available_channels or []:
        try:
            channels.append(int(channel))
        except (TypeError, ValueError):
            continue
    channels = sorted(set(channels))
    if not channels:
        channels = [0]

    selected = _to_int(
        (cfg or {}).get("analysis_channel", channels[0]),
        channels[0],
    )
    return selected if selected in channels else channels[0]
