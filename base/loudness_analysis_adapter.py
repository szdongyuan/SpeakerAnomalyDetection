"""Headless integration of the develop loudness engine and queue configuration."""

from copy import deepcopy

import numpy as np

from base.analysis_limit_evaluation import (
    compare_with_limits,
    interpolate_spl_limit_curves,
    limits_from_constant_values,
    resolve_limit_source,
)
from base.core_algorithm.sound_quality.loudness import LoudnessMethod
from base.core_algorithm.sound_quality.service import run_sound_quality


_SUMMARY_METRICS = (
    "steady_state_average_sone", "steady_state_average_phon",
    "max_transient_sone", "max_transient_phon",
)


def loudness_unit(config, *, limits=False):
    display = str((config.get("advanced") or {}).get("curve_y_unit", "sone") or "sone").lower()
    unit = str(config.get("curve_limit_unit") or display).lower() if limits else display
    if unit not in {"sone", "phon"}:
        raise ValueError(f"不支持的响度单位：{unit}")
    return unit


def calculate_loudness(signal, sample_rate, config, v2pa_factor, **context):
    """Use the existing service; preserve source summary and preprocessing rules."""
    settings = deepcopy(config)
    settings.setdefault("enabled", True)
    settings.setdefault("display", {
        "curves": ["loudness_time"],
        "summary_metrics": ["steady_state_average_loudness", "max_transient_loudness"],
    })
    method = settings.get("method", LoudnessMethod.TIME_VARYING_ISO532_1.value)
    LoudnessMethod(method)  # Reject unknown methods instead of silently substituting one.
    settings["required_summary_metrics"] = list(_SUMMARY_METRICS)
    if settings.get("limit_metric") == "specific_loudness_summed_exceedance":
        settings["required_summary_metrics"].append("specific_loudness_summed_exceedance")
    advanced = settings.setdefault("advanced", {})
    window = context.get("sequence_snapshot", {}).get("segment_window_samples")
    if window is not None:
        # The adapter dispatcher already selected these samples.
        advanced["analysis_time_range_enabled"] = False
    output = run_sound_quality(
        signal, sample_rate, v2pa_factor,
        {"enabled": True, "items": {"LOUD": settings}},
    ).loudness
    if output is None or not output.enabled or output.raw_result is None:
        raise ValueError(output.skipped_reason if output is not None else "响度未产生计算结果")
    raw = output.raw_result
    offset = (window[0] / sample_rate if window is not None
              else float(raw.metadata.get("analysis_source_start_s", 0.0)))
    time_s = np.asarray(raw.time_s, dtype=float) + offset
    curves = {"sone": np.asarray(raw.loudness_sone), "phon": np.asarray(raw.loudness_level_phon)}
    if not time_s.size or any(not np.all(np.isfinite(v)) for v in curves.values()):
        raise ValueError("响度计算未产生有限曲线")
    if any(not np.isfinite(output.summary[key]) for key in _SUMMARY_METRICS):
        raise ValueError("响度计算未产生有限汇总指标")
    unit = loudness_unit(config)
    display = output.display_payload
    for curve in display["curves"]:
        if curve["key"] == "loudness_time":
            curve.update(x=time_s, y=curves[unit], y_label=f"Loudness ({unit})",
                         zero_based=bool(advanced.get("curve_y_axis_zero_based", True)))
    for heatmap in display["heatmaps"]:
        heatmap["x"] = time_s
        heatmap["color_map"] = advanced.get("specific_loudness_colormap", "viridis")
    result = {
        "judgement": None,
        "metrics": {**output.summary, "unit": unit, "method": method},
        "curve": {"x": time_s, "y": curves[unit], "upper": [], "lower": []},
        "loudness_curves": curves,
        "plot": {"kind": "loudness", **display},
    }
    if config.get("limit_checked", False):
        _apply_legacy_judgement(config, result)
    return result


def _apply_legacy_judgement(config, result):
    x, y, upper, lower, scalar, unit = loudness_limit_data(config, result)
    valid = np.isfinite(x) & np.isfinite(y) & (np.isfinite(upper) | np.isfinite(lower))
    if not np.any(valid):
        raise ValueError("阈值与有效响度数据没有重叠，无法判定")
    _, _, passed = compare_with_limits(y, upper, lower, valid_mask=valid)
    result["judgement"] = "OK" if passed else "NG"
    result["metrics"].update(limit_metric=config.get("limit_metric", "curve_y"), limit_unit=unit)
    if scalar:
        result["metrics"].update(
            overall_upper_limit=float(upper[0]) if np.isfinite(upper[0]) else None,
            overall_lower_limit=float(lower[0]) if np.isfinite(lower[0]) else None,
        )
    elif result["metrics"]["unit"] == unit:
        result["curve"].update(upper=upper, lower=lower)
        for curve in result["plot"]["curves"]:
            if curve["key"] == "loudness_time":
                curve.update(upper=upper, lower=lower)


def loudness_limit_data(config, calculation):
    """Resolve judgement values in the configured limit unit, not the display unit."""
    unit = loudness_unit(config, limits=True)
    metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()
    scalar = metric != "curve_y"
    if scalar:
        if metric not in {"steady_state_average", "max_transient", "specific_loudness_summed_exceedance"}:
            raise ValueError(f"不支持的响度判定指标：{metric}")
        key = metric if metric == "specific_loudness_summed_exceedance" else f"{metric}_{unit}"
        if metric == "specific_loudness_summed_exceedance":
            unit = "sone"
        x = np.asarray([0.0])
        y = np.asarray([calculation["metrics"][key]])
        limits = {}
        for side in ("upper", "lower"):
            limits[f"constant_{side}_enabled"] = config.get(
                f"scalar_{side}_enabled", config.get(f"curve_{side}_enabled", False))
            limits[f"constant_{side}_value"] = config.get(
                f"scalar_{side}_value", config.get(f"curve_{side}_value", 0.0))
        _, upper, lower = limits_from_constant_values(limits, x)
    else:
        x = np.asarray(calculation["curve"]["x"])
        y = calculation["loudness_curves"][unit]
        limits = dict(config)
        if "limit_mode" not in limits:
            limits.update(limit_mode="manual", manual_input_mode="constant")
            for side in ("upper", "lower"):
                limits[f"constant_{side}_enabled"] = config.get(f"curve_{side}_enabled", False)
                limits[f"constant_{side}_value"] = config.get(f"curve_{side}_value", 0.0)
        if limits.get("limit_mode") == "csv" and limits.get("limit_data") is not None:
            # JSON snapshots encode an absent bound as null.
            raw_x, raw_upper, raw_lower = limits["limit_data"]
            limits["limit_data"] = [raw_x,
                [np.nan if value is None else value for value in raw_upper],
                [np.nan if value is None else value for value in raw_lower]]
        upper, lower = interpolate_spl_limit_curves(x, *resolve_limit_source(limits, x))
    upper, lower = np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    if np.any((lower > upper) & np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("下限不能大于上限")
    return x, y, upper, lower, scalar, unit
