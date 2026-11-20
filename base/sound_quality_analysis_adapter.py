"""SHRP/ROUGH configuration and result adaptation; numerical cores stay independent."""

from copy import deepcopy

import numpy as np

from base.analysis_config_validation import sharpness_execution_issue
from consts.acoustic_analysis.common_consts import SHARPNESS_METHOD, SHARPNESS_OUTPUT_TIME_RESOLUTION_MS
from base.analysis_limit_evaluation import (
    compare_with_limits, interpolate_spl_limit_curves, limits_from_constant_values,
    resolve_limit_source,
)
from base.core_algorithm.sound_quality.roughness import compute_roughness_daniel_weber
from base.core_algorithm.sound_quality.service import (
    compute_loudness_raw, loudness_compute_options, sound_quality_time_window,
)
from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


from consts.acoustic_analysis.common_consts import SOUND_QUALITY_ANALYSES

_UNITS = {kind: spec["unit"] for kind, spec in SOUND_QUALITY_ANALYSES.items()}
_SUMMARY_METRICS = {kind: tuple(key for key, _ in spec["metrics"]) for kind, spec in SOUND_QUALITY_ANALYSES.items()}


def _prepare_config(config, v2pa_factor, context):
    factor = float(v2pa_factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("声压标定系数必须为有限正数")
    settings = deepcopy(config)
    advanced = settings.setdefault("advanced", {})
    window = context.get("sequence_snapshot", {}).get("segment_window_samples")
    if window is not None:
        advanced["analysis_time_range_enabled"] = False
    return settings, factor, window


def calculate_sharpness(signal, sample_rate, config, v2pa_factor, **context):
    issue = sharpness_execution_issue(config)
    if issue is not None:
        raise ValueError(issue[1])
    settings, factor, window = _prepare_config(config, v2pa_factor, context)
    # Fixed time-varying defaults. Ignore obsolete, inactive segment settings
    # so they cannot prevent reuse of an otherwise identical LOUD frontend.
    advanced = settings["advanced"]
    _, options = loudness_compute_options({
        "method": SHARPNESS_METHOD,
        "advanced": {"output_time_resolution_ms": SHARPNESS_OUTPUT_TIME_RESOLUTION_MS},
    })
    raw, source_start, _ = compute_loudness_raw(
        signal, sample_rate, factor, "free", advanced, options,
        compute_context=context.get("compute_context"), input_identity=context.get("input_identity"),
    )
    # Consume only the raw frontend, never LOUD's summaries, judgement or export.
    sharp = compute_sharpness_din45692(
        raw.specific_loudness, raw.bark_axis, raw.time_s,
        method=options["method"], **(settings.get("sharpness") or {}),
    )
    offset = window[0] / sample_rate if window is not None else source_start
    return _quality_result("SHRP", sharp.time_s + offset, sharp.sharpness_acum, {
        "s_max_acum": sharp.summary.s_max, "s_mean_acum": sharp.summary.s_mean,
        "s_stationary_acum": sharp.summary.s_stationary,
    }, sharp.metadata, config)


def calculate_roughness(signal, sample_rate, config, v2pa_factor, **context):
    settings, factor, window = _prepare_config(config, v2pa_factor, context)
    if settings.get("method", "daniel_weber") != "daniel_weber":
        raise ValueError("ROUGH 仅支持 daniel_weber 方法")
    advanced = settings["advanced"]
    if advanced.get("background_noise_processing_enabled", False):
        raise ValueError("ROUGH 尚未接入背景参考降噪")
    first, stop, source_start, _ = sound_quality_time_window(signal.size, sample_rate, advanced)
    # Core accepts Pa. Calibration is applied here once, after selecting samples.
    signal_pa = signal[first:stop].astype(np.float64, copy=False) * factor
    options = {"overlap": settings["overlap"]} if "overlap" in settings else {}
    rough = compute_roughness_daniel_weber(signal_pa, sample_rate, **options)
    offset = window[0] / sample_rate if window is not None else source_start
    return _quality_result("ROUGH", rough.time_s + offset, rough.roughness_asper, {
        "r_mean_asper": rough.summary.r_mean_asper, "r_max_asper": rough.summary.r_max_asper,
    }, rough.metadata, config)


def _quality_result(kind, time_s, values, summary, metadata, config):
    valid = np.isfinite(values)
    curve = {"x": time_s, "y": values, "valid_mask": valid, "upper": [], "lower": []}
    result = {
        "analysis_type": kind, "judgement": None,
        "metrics": {
            **summary, "unit": _UNITS[kind], "method": metadata.get("method", "daniel_weber"),
            "valid_frame_count": int(np.count_nonzero(valid)), "frame_count": int(valid.size),
            "valid_summary_metrics": [key for key, value in summary.items() if np.isfinite(value)],
        },
        "curve": curve,
        "plot": {"kind": "curve", **curve, "x_unit": "s", "y_unit": _UNITS[kind]},
        "metadata": metadata,
    }
    if config.get("limit_checked", False):
        x, y, upper, lower, scalar, unit = sound_quality_limit_data(config, result)
        valid_limits = np.isfinite(x) & np.isfinite(y) & (np.isfinite(upper) | np.isfinite(lower))
        if not np.any(valid_limits):
            raise ValueError("阈值与有效分析数据没有重叠，无法判定")
        _, _, passed = compare_with_limits(y, upper, lower, valid_mask=valid_limits)
        result["judgement"] = "OK" if passed else "NG"
        result["metrics"].update(limit_metric=config.get("limit_metric", "curve_y"), limit_unit=unit)
        if scalar:
            result["metrics"].update(overall_upper_limit=float(upper[0]) if np.isfinite(upper[0]) else None,
                                     overall_lower_limit=float(lower[0]) if np.isfinite(lower[0]) else None)
        else:
            curve.update(upper=upper, lower=lower)
            result["plot"].update(upper=upper, lower=lower)
    return result


def sound_quality_limit_data(config, calculation):
    """Resolve quality metrics in their physical units using existing limit helpers."""
    kind = calculation["analysis_type"]
    metric = config.get("limit_metric", "curve_y")
    scalar = metric != "curve_y"
    if scalar:
        if metric not in _SUMMARY_METRICS[kind]:
            raise ValueError(f"不支持的 {kind} 判定指标：{metric}")
        x = np.asarray([0.0])
        y = np.asarray([calculation["metrics"][metric]])
        limits = {
            f"constant_{side}_{field}": config.get(f"scalar_{side}_{field}", default)
            for side in ("upper", "lower") for field, default in (("enabled", False), ("value", 0.0))
        }
        _, upper, lower = limits_from_constant_values(limits, x)
    else:
        x, y = calculation["curve"]["x"], calculation["curve"]["y"]
        limits = dict(config)
        if limits.get("limit_mode", "csv") == "csv" and limits.get("limit_data") is not None:
            axis, upper, lower = limits["limit_data"]
            limits["limit_data"] = [axis, [np.nan if v is None else v for v in upper],
                                   [np.nan if v is None else v for v in lower]]
        upper, lower = interpolate_spl_limit_curves(x, *resolve_limit_source(limits, x))
    upper, lower = np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    if np.any((lower > upper) & np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("下限不能大于上限")
    return x, y, upper, lower, scalar, _UNITS[kind]
