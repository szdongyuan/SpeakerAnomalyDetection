"""Helpers for SPL runtime configuration compatibility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def resolve_spl_window_size(config: Mapping[str, Any] | None, sample_rate: float) -> int:
    cfg = config or {}
    unit = str(cfg.get("spl_window_unit", "points") or "points").lower()
    if unit == "time":
        seconds = float(cfg.get("spl_window_time_sec", 0.0272) or 0.0272)
        return positive_int(seconds * float(sample_rate), 1201)
    return positive_int(cfg.get("spl_window_points", 1201), 1201)


def resolve_spl_smoothing(
    config: Mapping[str, Any] | None,
    sample_rate: float,
    series_len: int,
) -> tuple[int, str] | None:
    cfg = config or {}
    enabled = bool(cfg.get("smooth_enabled", cfg.get("smooth_checked", False)))
    if not enabled:
        return None

    unit = str(cfg.get("smooth_unit", "points") or "points").lower()
    if unit == "time":
        seconds = float(cfg.get("smooth_time_sec", 0.025) or 0.025)
        window_size = positive_int(seconds * float(sample_rate), 1102)
    else:
        window_size = positive_int(cfg.get("smooth_points", 1102), 1102)
    if series_len > 0:
        window_size = min(window_size, int(series_len))

    algo = int(cfg.get("smooth_algo", 2) or 2)
    method = {1: "mean", 2: "savgol", 3: "gaussian"}.get(algo, "savgol")
    return window_size, method


def apply_spl_analysis_time_range(
    recorded_signal,
    sample_rate: float,
    config: Mapping[str, Any] | None,
):
    cfg = config or {}
    if not cfg.get("analysis_time_range_enabled", False):
        return recorded_signal, 0

    signal = np.asarray(recorded_signal)
    start_sec = max(0.0, float(cfg.get("analysis_start_time_sec", 0.0) or 0.0))
    end_sec = max(0.0, float(cfg.get("analysis_end_time_sec", 0.0) or 0.0))
    start_sample = min(int(np.floor(start_sec * float(sample_rate))), len(signal))
    end_sample = min(int(np.ceil(end_sec * float(sample_rate))), len(signal))
    if end_sample <= start_sample:
        return recorded_signal, 0
    return signal[start_sample:end_sample], start_sample
