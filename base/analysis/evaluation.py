"""Threshold judgement on complete calculation data, separate from execution."""

import numpy as np

from base.analysis_limit_evaluation import (
    compare_with_limits,
    interpolate_spl_limit_curves,
    resolve_limit_source,
    resolve_limits,
    resolve_spl_overall_limits,
)


def evaluate_instance(instance, calculation):
    config = dict(instance.parameters)
    if instance.analysis_type == "Spec":
        raise ValueError("时频图尚不支持阈值判定；颜色范围不是判定阈值")
    scalar = instance.analysis_type == "SPL" and str(
        config.get("limit_metric", "curve_y") or "curve_y"
    ).lower() == "overall_spl"
    if scalar:
        x = np.asarray([0.0])
        y = np.asarray([calculation["metrics"]["overall_spl"]])
        upper, lower = resolve_spl_overall_limits(config)
    else:
        csv_limits = str(config.get("limit_mode", "csv") or "csv").lower() == "csv"
        if csv_limits and config.get("limit_data") is not None:
            # JSON uses null for a missing bound; the numerical helpers use NaN.
            # Only adapt active CSV limits, keeping the input snapshot intact.
            raw_x, raw_upper, raw_lower = config["limit_data"]
            config["limit_data"] = [
                raw_x, [np.nan if v is None else v for v in raw_upper],
                [np.nan if v is None else v for v in raw_lower],
            ]
        x = np.asarray(calculation["curve"]["x"], dtype=float)
        y = np.asarray(calculation["curve"]["y"], dtype=float)
        if instance.analysis_type == "SPL":
            upper, lower = interpolate_spl_limit_curves(
                x, *resolve_limit_source(config, x),
            )
        else:
            upper, lower = resolve_limits(config, x)
    valid = np.isfinite(x) & np.isfinite(y) & (np.isfinite(upper) | np.isfinite(lower))
    if not np.any(valid):
        raise ValueError("阈值与有效分析数据没有重叠，无法判定")
    if np.any((lower > upper) & np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("下限不能大于上限")
    _, _, passed = compare_with_limits(y, upper, lower, valid_mask=valid)
    violations = _violations(
        x, y, upper, lower, valid, scalar=scalar,
        discrete=instance.analysis_type == "FBA",
        x_unit="s" if instance.analysis_type == "SPL" else "Hz",
    )
    return {
        "judgement": "OK" if passed else "NG", "violations": violations,
        "upper": upper, "lower": lower, "scalar": scalar,
    }


def _violations(x, y, upper, lower, valid, *, scalar, discrete, x_unit):
    records = []
    for side, limits, deviations in (
        ("above_upper", upper, y - upper),
        ("below_lower", lower, lower - y),
    ):
        indices = np.flatnonzero(valid & np.isfinite(limits) & (deviations > 0))
        # Missing limits/data split runs. Bands remain individual measurements.
        groups = ([np.asarray([i]) for i in indices] if discrete else
                  np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1))
        for group in groups:
            if not group.size:
                continue
            peak = group[np.argmax(deviations[group])]
            records.append({
                "side": side,
                "x_start": None if scalar else float(x[group[0]]),
                "x_end": None if scalar else float(x[group[-1]]),
                "x_unit": None if scalar else x_unit,
                "max_deviation_at": None if scalar else float(x[peak]),
                "measured_value": float(y[peak]), "limit_value": float(limits[peak]),
                "max_deviation": float(deviations[peak]), "deviation_unit": "dB",
            })
    return records
