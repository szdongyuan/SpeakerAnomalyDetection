"""Pure request-owned analysis used when recording results overlap.

This module intentionally imports no Qt/UI objects and receives every input as
an immutable request snapshot.  It is the safe completion boundary for an old
recording while a newer recording owns the live SequenceWidget presentation.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from base.core_algorithm.response import FrequencyBandAnalyzer
from ui.ui_analysis_config.analysis_compat import (
    interpolate_spl_limit_curves,
    parse_fba_custom_bands_text,
)
from ui.ui_analysis_config.config_normalization import normalize_analysis_channels
from ui.ui_analysis_config.manual_limit_segments import limits_from_manual_config


_MAX_RESULT_SERIES_POINTS = 20_000
_MAX_SPECTROGRAM_TIME_BINS = 1024
_MAX_SPECTROGRAM_FREQUENCY_BINS = 512


@dataclass(frozen=True)
class RequestScopedAnalysisOutcome:
    analysis_result_dict: dict
    label: str
    diagnostics: tuple = ()
    analysis_items_data: dict = None


def _recording_column(request, config):
    raw_channel = int(config.get("analysis_channel", 0) or 0)
    channels = tuple(int(channel) for channel in request.channels)
    if raw_channel not in channels:
        raise ValueError(
            f"analysis channel In{raw_channel + 1} is absent from request "
            f"channels {channels}")
    return raw_channel, channels.index(raw_channel)


def _calibration_factor(request, column):
    metadata = request.calibration_metadata
    if not isinstance(metadata, Mapping):
        return 1.0
    recorded_channels = metadata.get("recorded_channels")
    if not isinstance(recorded_channels, (list, tuple)) or column >= len(recorded_channels):
        return 1.0
    item = recorded_channels[column]
    if not isinstance(item, Mapping):
        return 1.0
    factor = item.get("v2pa_factor")
    if factor is None:
        if request.device.get("backend") == "vkinging":
            raise ValueError(
                f"VE request channel In{request.channels[column] + 1} has no calibration")
        return 1.0
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("recording calibration factor must be finite and positive")
    return factor


def _finite_limit(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _compare_scalar(value, config):
    upper = None
    lower = None
    if bool(config.get("scalar_upper_enabled", True)):
        upper = _finite_limit(config.get("scalar_upper_value", 100.0), "SPL upper limit")
    if bool(config.get("scalar_lower_enabled", False)):
        lower = _finite_limit(config.get("scalar_lower_value", 0.0), "SPL lower limit")
    if upper is None and lower is None:
        raise ValueError("SPL overall limit enables neither upper nor lower bound")
    exceedances = []
    if upper is not None:
        exceedances.append(float(value) - upper)
    if lower is not None:
        exceedances.append(lower - float(value))
    deviation = max(0.0, *exceedances)
    return deviation == 0.0, deviation


def _curve_limits(config, target_time):
    mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if mode == "csv":
        data = config.get("limit_data")
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError("SPL curve limit_data must contain time/upper/lower arrays")
        limit_time, upper, lower = data
    elif mode == "manual":
        limit_time, upper, lower = limits_from_manual_config(config, target_time)
    else:
        raise ValueError(f"unsupported SPL limit mode: {mode}")
    limit_time = np.asarray(limit_time, dtype=float)
    if limit_time.ndim != 1 or limit_time.size < 1 or not np.all(np.isfinite(limit_time)):
        raise ValueError("SPL limit time values are invalid")

    return interpolate_spl_limit_curves(
        target_time,
        limit_time,
        upper,
        lower,
    )


def _compare_curve(values, x_axis, config):
    if not bool(config.get("limit_checked", False)):
        return (None, 0.0), None, None
    upper, lower = _curve_limits(config, np.asarray(x_axis, dtype=float))
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (np.isfinite(upper) | np.isfinite(lower))
    if not np.any(valid):
        raise ValueError("analysis limits do not overlap the result")
    excess = np.maximum(
        np.where(np.isfinite(upper), values - upper, 0.0),
        np.where(np.isfinite(lower), lower - values, 0.0),
    )
    deviation = float(max(0.0, np.max(excess[valid])))
    return (deviation == 0.0, deviation), upper, lower


def _spectrogram_retention_plan(
        *, frame_count, sample_rate, n_fft, hop_length, channel_count=1):
    from base.spectrogram_analysis_service import spectrogram_retention_plan
    return spectrogram_retention_plan(
        frame_count=frame_count, sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, channel_count=channel_count)


def _bounded_indices(length, limit):
    length = max(0, int(length))
    limit = max(1, int(limit))
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, limit, dtype=np.int64)


def _bounded_analysis_detail(value):
    """Bound persisted/display payloads without changing computed verdicts."""
    if isinstance(value, dict):
        return {key: _bounded_analysis_detail(item)
                for key, item in value.items()}
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return type(value)(_bounded_analysis_detail(item) for item in value)
    else:
        return value
    if array.ndim == 1 and array.size > _MAX_RESULT_SERIES_POINTS:
        array = array[_bounded_indices(array.size, _MAX_RESULT_SERIES_POINTS)]
    elif array.ndim == 2:
        row_index = _bounded_indices(
            array.shape[0], _MAX_SPECTROGRAM_FREQUENCY_BINS)
        column_index = _bounded_indices(
            array.shape[1], _MAX_SPECTROGRAM_TIME_BINS)
        array = array[np.ix_(row_index, column_index)]
    return array.tolist()


def _analyze_spec(request, multi, sample_rate, config):
    from base.spectrogram_analysis_service import compute_spectrogram
    _raw, column = _recording_column(request, config)
    analysis = compute_spectrogram(
        multi[:, column], sample_rate, config,
        v2pa_factor=_calibration_factor(request, column),
        channel_count=multi.shape[1])
    return (None, 0.0), analysis.as_dict()


def _parse_custom_bands(text):
    return parse_fba_custom_bands_text(text)


def _analyze_fba(request, multi, sample_rate, config):
    _raw, column = _recording_column(request, config)
    labels = {"1/1 倍频程": ("octave", 1), "1/3 倍频程": ("octave", 3),
              "1/6 倍频程": ("octave", 6), "1/12 倍频程": ("octave", 12),
              "Bark": ("bark", 3), "等宽": ("equal_width", 3),
              "自定义": ("custom", 3)}
    strategy, fraction = labels.get(config.get("band_strategy"), ("octave", 3))
    custom_edges = None
    if strategy == "custom":
        custom_edges = _parse_custom_bands(config.get("custom_bands_text"))
        if not custom_edges:
            raise ValueError("请至少输入一个频段")
    analyzer = FrequencyBandAnalyzer(
        strategy=strategy, weighting=config.get("weighting", "A"),
        f_min=float(config.get("f_min", 20)), f_max=float(config.get("f_max", 20000)),
        fraction=fraction, n_bands=int(config.get("n_bands", 40)),
        bandwidth=float(config.get("bandwidth", 100)),
        custom_edges=custom_edges)
    result = analyzer.analyze(np.asarray(multi[:, column]), int(sample_rate),
                              v2pa_factor=_calibration_factor(request, column))
    centers = np.asarray([band.f_center for band in result.bands], dtype=float)
    levels = np.asarray(result.band_levels_weighted_db, dtype=float)
    judgement, upper, lower = _compare_curve(levels, centers, config)
    return judgement, {
        "bands": [band.label for band in result.bands],
        "band_centers": centers.tolist(), "band_levels_db": result.band_levels_db.tolist(),
        "band_levels_weighted_db": levels.tolist(), "overall_db": result.overall_db,
        "overall_weighted_db": result.overall_weighted_db, "weighting": result.weighting,
        "upper_limits": [] if upper is None else upper.tolist(),
        "lower_limits": [] if lower is None else lower.tolist(),
    }


def _analyze_spl(request, recorded_multi, sample_rate, key, config):
    from base.spl_analysis_service import compute_spl_trace
    _raw_channel, column = _recording_column(request, config)
    signal = np.asarray(recorded_multi[:, column], dtype=np.float64)
    factor = _calibration_factor(request, column)
    trace = compute_spl_trace(
        signal, int(sample_rate), config, v2pa_factor=factor)
    overall = trace.overall_spl
    detail = {
        "overall_spl": float(overall),
        "signal_duration": trace.signal_duration,
        "recorded_signal": trace.analysis_signal,
        "signal_spl": trace.signal_spl,
    }
    if not bool(config.get("limit_checked", False)):
        return (None, 0.0), detail
    metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()
    if metric == "overall_spl":
        judgement = _compare_scalar(overall, config)
        return judgement, detail

    signal_spl = trace.signal_spl
    time_axis = trace.signal_duration
    upper, lower = _curve_limits(config, time_axis)
    values = np.asarray(signal_spl, dtype=float)
    valid = np.isfinite(values) & (np.isfinite(upper) | np.isfinite(lower))
    if not np.any(valid):
        raise ValueError("SPL curve limits do not overlap the analyzed recording")
    excess = np.zeros(values.shape, dtype=float)
    excess = np.maximum(excess, np.where(np.isfinite(upper), values - upper, 0.0))
    excess = np.maximum(excess, np.where(np.isfinite(lower), lower - values, 0.0))
    deviation = float(np.max(excess[valid]))
    return (deviation == 0.0, deviation), detail


def analyze_recording_request(
        *, request, recorded_mono, recorded_multi, sample_rate,
        config_snapshot, recorded_signal_info):
    """Analyze frozen audio without consulting or mutating a SequenceWidget."""
    multi = np.asarray(recorded_multi, dtype=np.float32)
    if multi.ndim == 1:
        multi = multi.reshape(-1, 1)
    if multi.ndim != 2 or multi.shape[1] != len(tuple(request.channels)):
        raise ValueError("request-scoped analysis audio does not match request channels")
    analysis_config = (
        config_snapshot.get("analysis_config", {})
        if isinstance(config_snapshot, dict) else {})
    configured = analysis_config.get("display_sequence", [])
    if not isinstance(configured, (list, tuple)):
        raise ValueError("request-scoped analysis display_sequence is invalid")
    if not configured:
        return RequestScopedAnalysisOutcome({}, "not_labeled", (), {})
    results = {}
    items = {}
    diagnostics = []
    for key in configured:
        config = analysis_config.get(key)
        if not isinstance(config, dict):
            raise ValueError(f"request-scoped analysis config is missing: {key}")
        analysis_type = str(config.get("type") or "")
        normalized_type = analysis_type.upper()
        if normalized_type == "EXCEL":
            # Excel is an A-owned publisher, never an analysis computation.
            continue
        multi_channel_types = {"SPL", "SPEC", "FBA"}
        raw_channels = (
            normalize_analysis_channels(config)
            if normalized_type in multi_channel_types and "analysis_channels" in config
            else [int(config.get("analysis_channel", 0) or 0)]
        )
        for raw_channel in raw_channels:
            runtime_config = dict(config)
            runtime_config["analysis_channel"] = int(raw_channel)
            golden_path = analysis_config.get("golden_sample_result_path")
            if golden_path:
                runtime_config["golden_sample_result_path"] = golden_path
            runtime_key = (
                f"{key}--通道{int(raw_channel) + 1}"
                if len(raw_channels) > 1 else str(key))
            if int(raw_channel) not in tuple(request.channels):
                message = (
                    f"请求分析通道 In{int(raw_channel) + 1} 不存在；"
                    f"录音请求通道为 {tuple(request.channels)}")
                diagnostics.append(f"{runtime_key}: {message}")
                continue
            if normalized_type == "SPL":
                judgement, detail = _analyze_spl(
                    request, multi, int(sample_rate), runtime_key, runtime_config)
            elif normalized_type == "SPEC":
                judgement, detail = _analyze_spec(
                    request, multi, int(sample_rate), runtime_config)
            elif normalized_type == "FBA":
                judgement, detail = _analyze_fba(
                    request, multi, int(sample_rate), runtime_config)
            else:
                raise ValueError(
                    f"request-scoped background analysis is limited to SPL/FBA/SPEC: "
                    f"{analysis_type or key}")
            results[runtime_key] = judgement
            items[runtime_key] = {
                "type": analysis_type,
                "result": _bounded_analysis_detail(detail),
                "config_key": str(key),
                "result_key": runtime_key,
                "raw_channel": int(raw_channel),
                "multi_channel_expansion": len(raw_channels) > 1,
            }
    if not results:
        # Publication-only entries (currently Excel) have no computational
        # result, matching the established active-window no-op semantics.
        return RequestScopedAnalysisOutcome({}, "not_labeled", tuple(diagnostics), {})
    judged = [value[0] for value in results.values() if value[0] is not None]
    label = "NG" if any(value is False for value in judged) else (
        "OK" if judged and all(value is True for value in judged) else "not_labeled")
    return RequestScopedAnalysisOutcome(
        results, label, tuple(diagnostics), items)
