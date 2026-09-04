"""Helpers for SPL runtime calculation and display compatibility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ui.ui_analysis_config.analysis_compat import (
    interpolate_spl_limit_curves,
    interpolate_spl_limit_side,
)
from ui.ui_analysis_config.manual_limit_segments import (
    limits_from_constant_values,
    limits_from_manual_config,
)
from ui.ui_analysis_config.threshold_csv_manual import (
    validate_limit_data_values,
)


def calculate_overall_spl(
    recorded_signal,
    reference_pressure: float = 20e-6,
    v2pa_factor: float | None = None,
) -> float:
    """Compute the RMS sound pressure level over the full input signal."""
    signal_float = np.asarray(recorded_signal, dtype=float)
    if signal_float.size == 0:
        return float("nan")

    factor = 1.0 if v2pa_factor is None else float(v2pa_factor)
    pressure_pa = signal_float * factor
    pressure_rms = float(np.sqrt(np.mean(pressure_pa**2)))
    pressure_rms = max(pressure_rms, 1.0e-10)
    return float(20 * np.log10(pressure_rms / float(reference_pressure)))


def resolve_spl_unit(weighting: Any) -> str:
    """Return the SPL display unit for a configured frequency weighting."""
    normalized = str(weighting or "Z").strip().upper()
    return {
        "A": "dBA",
        "B": "dBB",
        "C": "dBC",
        "D": "dBD",
    }.get(normalized, "dB")


def _validated_csv_curve_limit_data(limit_data):
    try:
        x_values, upper_values, lower_values = limit_data
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CSV SPL limit data must contain X, upper, and lower arrays"
        ) from exc

    fields = (x_values, upper_values, lower_values)
    if any(isinstance(values, (str, bytes)) for values in fields):
        raise ValueError("CSV SPL limit arrays must be one-dimensional sequences")
    try:
        field_arrays = tuple(np.asarray(values, dtype=object) for values in fields)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV SPL limit arrays must be one-dimensional sequences") from exc
    if any(values.ndim != 1 for values in field_arrays):
        raise ValueError("CSV SPL limit arrays must be one-dimensional sequences")
    x_list, upper_list, lower_list = (values.tolist() for values in field_arrays)
    if not x_list or not (len(x_list) == len(upper_list) == len(lower_list)):
        raise ValueError("CSV SPL limit arrays must have equal nonzero lengths")

    normalized_x = []
    normalized_upper = []
    normalized_lower = []
    for row, (raw_x, raw_upper, raw_lower) in enumerate(
        zip(x_list, upper_list, lower_list),
        start=1,
    ):
        if isinstance(raw_x, (bool, np.bool_)):
            raise ValueError(f"CSV SPL limit X at row {row} must be finite")
        try:
            x_value = float(raw_x)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"CSV SPL limit X at row {row} must be finite") from exc
        if not np.isfinite(x_value):
            raise ValueError(f"CSV SPL limit X at row {row} must be finite")

        normalized_bounds = []
        for raw_bound in (raw_upper, raw_lower):
            if raw_bound is None or (
                isinstance(raw_bound, str) and not raw_bound.strip()
            ):
                bound = float("nan")
            else:
                if isinstance(raw_bound, (bool, np.bool_)):
                    raise ValueError(
                        f"CSV SPL limit bound at row {row} must be finite or absent"
                    )
                try:
                    bound = float(raw_bound)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"CSV SPL limit bound at row {row} must be finite or absent"
                    ) from exc
                if not np.isfinite(bound) and not np.isnan(bound):
                    raise ValueError(
                        f"CSV SPL limit bound at row {row} must be finite or absent"
                    )
            normalized_bounds.append(bound)
        upper_value, lower_value = normalized_bounds
        if not (np.isfinite(upper_value) or np.isfinite(lower_value)):
            raise ValueError(
                f"CSV SPL limit row {row} must have at least one finite upper or lower bound"
            )
        normalized_x.append(x_value)
        normalized_upper.append(upper_value)
        normalized_lower.append(lower_value)

    normalized = (normalized_x, normalized_upper, normalized_lower)
    validate_limit_data_values(normalized)
    return normalized


def resolve_spl_limit_data(config: Mapping[str, Any], target_x):
    """Resolve validated curve limits using the existing CSV/manual contract."""
    limit_mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if limit_mode == "manual":
        return limits_from_manual_config(dict(config), target_x)
    if limit_mode == "csv":
        limit_data = config.get("limit_data")
        if limit_data is None:
            raise ValueError("已启用阈值，但未加载 CSV 配置文件")
        return _validated_csv_curve_limit_data(limit_data)
    raise ValueError(f"不支持的阈值模式: {limit_mode}")


def resolve_spl_overall_limit_values(config: Mapping[str, Any]):
    """Resolve validated constant limits for overall SPL judgment."""
    scalar_config = {
        "constant_upper_enabled": bool(config.get("scalar_upper_enabled", True)),
        "constant_upper_value": config.get("scalar_upper_value", 100.0),
        "constant_lower_enabled": bool(config.get("scalar_lower_enabled", False)),
        "constant_lower_value": config.get("scalar_lower_value", 0.0),
    }
    _, upper_limits, lower_limits = limits_from_constant_values(
        scalar_config,
        [0.0],
    )
    return np.asarray(upper_limits), np.asarray(lower_limits)


def compare_spl_with_limits(
    plot_y,
    upper_limits,
    lower_limits,
    valid_mask=None,
):
    """Apply the established SPL limit deviation and OK/NG rules."""
    plot_y = np.asarray(plot_y, dtype=float)
    upper_limits = np.asarray(upper_limits, dtype=float)
    lower_limits = np.asarray(lower_limits, dtype=float)
    if plot_y.ndim != 1:
        raise ValueError("SPL comparison values must be one-dimensional")
    if upper_limits.shape != plot_y.shape or lower_limits.shape != plot_y.shape:
        raise ValueError("SPL limit arrays must match the SPL curve")
    if valid_mask is None:
        valid_mask = np.ones(plot_y.size, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != plot_y.shape:
            raise ValueError("SPL limit validity mask must match the SPL curve")

    upper_finite = np.isfinite(upper_limits)
    lower_finite = np.isfinite(lower_limits)
    out_mask = valid_mask & (
        (upper_finite & (plot_y > upper_limits))
        | (lower_finite & (plot_y < lower_limits))
    )

    deviation = 0.0
    is_ok = True
    if np.any(out_mask):
        is_ok = False
        upper_deviation = np.where(
            out_mask & upper_finite,
            plot_y - upper_limits,
            0.0,
        )
        lower_deviation = np.where(
            out_mask & lower_finite,
            lower_limits - plot_y,
            0.0,
        )
        deviation = float(
            np.nanmax(np.maximum(upper_deviation, lower_deviation))
        )
    else:
        in_range = valid_mask & np.isfinite(plot_y)
        if np.any(in_range):
            upper_margin = np.where(
                upper_finite[in_range],
                upper_limits[in_range] - plot_y[in_range],
                np.inf,
            )
            lower_margin = np.where(
                lower_finite[in_range],
                plot_y[in_range] - lower_limits[in_range],
                np.inf,
            )
            margins = np.minimum(upper_margin, lower_margin)
            margins = margins[np.isfinite(margins)]
            if margins.size:
                deviation = float(np.min(margins))

    return out_mask, round(deviation, 2), is_ok


def evaluate_spl_limits(
    config: Mapping[str, Any],
    time_seconds,
    spl_db,
    overall_spl: float | None,
) -> tuple[bool | None, float | None]:
    """Evaluate enabled SPL limits without plotting or GUI side effects."""
    if not bool(config.get("limit_checked", False)):
        return None, None

    metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()
    if metric == "overall_spl":
        if overall_spl is None or not np.isfinite(overall_spl):
            raise ValueError("总体声压级不是有限数值")
        upper_limits, lower_limits = resolve_spl_overall_limit_values(config)
        _, deviation, is_ok = compare_spl_with_limits(
            np.asarray([overall_spl], dtype=float),
            upper_limits,
            lower_limits,
        )
        return is_ok, deviation
    if metric != "curve_y":
        raise ValueError(f"不支持的 SPL 判定依据: {metric}")

    time_values = np.asarray(time_seconds, dtype=float)
    spl_values = np.asarray(spl_db, dtype=float)
    limit_x, upper_limits, lower_limits = resolve_spl_limit_data(
        config,
        time_values,
    )
    upper_at, lower_at = interpolate_spl_limit_curves(
        time_values,
        limit_x,
        upper_limits,
        lower_limits,
    )
    valid_mask = (
        np.isfinite(time_values)
        & np.isfinite(spl_values)
        & (np.isfinite(upper_at) | np.isfinite(lower_at))
    )
    if not np.any(valid_mask):
        raise ValueError(
            "SPL curve limits do not overlap any finite analysis result samples"
        )
    _, deviation, is_ok = compare_spl_with_limits(
        spl_values,
        upper_at,
        lower_at,
        valid_mask,
    )
    return is_ok, deviation


def resolve_free_field_distance_correction_db(
    config: Mapping[str, Any] | None,
) -> float:
    """Return the free-field spherical-spreading correction in decibels."""
    cfg = config or {}
    if not cfg.get("free_field_distance_enabled", False):
        return 0.0
    distances = (
        ("measurement_distance_m", "测量距离"),
        ("target_distance_m", "目标距离"),
    )
    resolved = {}
    for key, label in distances:
        try:
            distance = float(cfg.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}必须为有限正数。") from exc
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"{label}必须为有限正数。")
        resolved[key] = distance

    return float(
        -20.0
        * np.log10(
            resolved["target_distance_m"]
            / resolved["measurement_distance_m"]
        )
    )


def resolve_directional_additional_correction_db(
    config: Mapping[str, Any] | None,
) -> float:
    """Return the configured manual directional correction in decibels."""
    cfg = config or {}
    if not cfg.get("directional_correction_enabled", False):
        return 0.0

    try:
        correction_db = float(
            cfg.get("directional_additional_correction_db", 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("方向修正必须为有限数值。") from exc
    if not np.isfinite(correction_db):
        raise ValueError("方向修正必须为有限数值。")
    return correction_db


def apply_spl_analysis_time_range(
    recorded_signal,
    sample_rate: float,
    config: Mapping[str, Any] | None,
):
    """Slice SPL input to the configured time range and return its source offset."""
    cfg = config or {}
    if not cfg.get("analysis_time_range_enabled", False):
        return recorded_signal, 0

    signal = np.asarray(recorded_signal)
    start_sec = max(
        0.0,
        float(cfg.get("analysis_start_time_sec", 0.0) or 0.0),
    )
    end_sec = max(
        0.0,
        float(cfg.get("analysis_end_time_sec", 0.0) or 0.0),
    )
    start_sample = min(
        int(np.floor(start_sec * float(sample_rate))),
        len(signal),
    )
    end_sample = (
        len(signal)
        if end_sec == 0.0
        else min(
            int(np.ceil(end_sec * float(sample_rate))),
            len(signal),
        )
    )
    if end_sample <= start_sample:
        return recorded_signal, 0
    return signal[start_sample:end_sample], start_sample
