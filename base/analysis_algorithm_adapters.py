"""Headless adapters around the existing analysis calculation modules.

This module is the stable boundary used by the asynchronous worker. It keeps
configuration interpretation, threshold evaluation, and plain-data result
assembly together while delegating signal calculations to lower-level analyzers.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from base.analysis_curve_style import resolve_curve_colors
from base.analysis_limit_evaluation import (
    compare_with_limits as _compare_with_limits,
    interpolate_limit_side as _interpolate_limit_side,
    interpolate_spl_limit_curves as _interpolate_spl_limit_curves,
    resolve_limit_source as _resolve_limit_source,
    resolve_limits as _resolve_limits,
    resolve_spl_overall_limits as _resolve_spl_overall_limits,
)
from base.core_algorithm.response import (
    DEFAULT_MAX_SPEC_TIME_BINS,
    SpectrogramAnalyzer,
    load_fft_baseline,
    parse_custom_bands,
    smooth_fft_baseline,
)


_MAX_SPEC_TIME_BINS = DEFAULT_MAX_SPEC_TIME_BINS

_FBA_STRATEGIES = {
    "1/1 倍频程": ("octave", {"fraction": 1}),
    "1/3 倍频程": ("octave", {"fraction": 3}),
    "1/6 倍频程": ("octave", {"fraction": 6}),
    "1/12 倍频程": ("octave", {"fraction": 12}),
    "Bark": ("bark", {}),
    "等宽": ("equal_width", {}),
    "自定义": ("custom", {}),
}


def _first_finite_or_none(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite[0]) if finite.size else None


def _finite_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _calculate_spl(signal, sample_rate, config, v2pa_factor, **_context):
    from base.analysis_display_payload import min_max_envelope
    from base.core_algorithm.harmonic_distortion.weighted import (
        apply_weighting_filter,
    )
    from base.pre_processing.audio_thd_frequency_response_analysis import (
        AudioThdFrequencyResponseAnalysis,
    )
    from base.pre_processing.spl_runtime_config import (
        apply_spl_analysis_time_range,
        calculate_overall_spl,
        resolve_spl_unit,
    )
    from base.utils.smooth import smooth

    reference_pressure = 20e-6
    window_size = 1201
    weighting = str(config.get("weighting", "Z") or "Z")
    weighted_signal = signal
    if weighting.upper() not in {"NONE", "Z"}:
        weighted_signal = apply_weighting_filter(
            signal,
            sample_rate,
            weighting=weighting,
            zero_phase=False,
        )
    analysis_signal, analysis_start_sample = apply_spl_analysis_time_range(
        weighted_signal,
        sample_rate,
        config,
    )
    overall_spl = calculate_overall_spl(
        analysis_signal,
        reference_pressure,
        v2pa_factor=v2pa_factor,
    )
    signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
        analysis_signal,
        reference_pressure,
        window_size=window_size,
        v2pa_factor=v2pa_factor,
        trim_edges=True,
    )
    signal_spl = np.asarray(signal_spl, dtype=np.float64)
    start_index = (
        0 if len(signal_spl) == len(analysis_signal) else window_size // 2
    )
    time_axis = (
        np.arange(len(signal_spl), dtype=np.float64) + float(start_index)
    ) / float(sample_rate)
    time_axis += float(analysis_start_sample) / float(sample_rate)
    if config.get("smooth_checked"):
        signal_spl = np.asarray(
            smooth(signal_spl, window_size=1102, method="savgol"),
            dtype=np.float64,
        )

    upper_limits = np.array([], dtype=np.float64)
    lower_limits = np.array([], dtype=np.float64)
    overall_upper = None
    overall_lower = None
    judgement = None
    limit_checked = bool(config.get("limit_checked", False))
    limit_metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()
    if limit_checked and limit_metric == "overall_spl":
        if not np.isfinite(overall_spl):
            raise ValueError("总体声压级不是有限数值")
        upper_values, lower_values = _resolve_spl_overall_limits(config)
        _, _, is_ok = _compare_with_limits(
            np.asarray([overall_spl], dtype=np.float64),
            upper_values,
            lower_values,
        )
        overall_upper = _first_finite_or_none(upper_values)
        overall_lower = _first_finite_or_none(lower_values)
        judgement = "OK" if is_ok else "NG"
    elif limit_checked:
        raw_x, raw_upper, raw_lower = _resolve_limit_source(config, time_axis)
        upper_limits, lower_limits = _interpolate_spl_limit_curves(
            time_axis,
            raw_x,
            raw_upper,
            raw_lower,
        )
        valid = (
            np.isfinite(time_axis)
            & np.isfinite(signal_spl)
            & (np.isfinite(upper_limits) | np.isfinite(lower_limits))
        )
        _, _, is_ok = _compare_with_limits(
            signal_spl,
            upper_limits,
            lower_limits,
            valid_mask=valid,
        )
        judgement = "OK" if is_ok else "NG"

    unit = resolve_spl_unit(weighting)
    plot_x_list, plot_y_list = min_max_envelope(time_axis, signal_spl)
    plot_x = np.asarray(plot_x_list, dtype=np.float64)
    plot_y = np.asarray(plot_y_list, dtype=np.float64)
    plot_upper = (
        _interpolate_limit_side(plot_x, time_axis, upper_limits)
        if upper_limits.size == time_axis.size
        else np.array([], dtype=np.float64)
    )
    plot_lower = (
        _interpolate_limit_side(plot_x, time_axis, lower_limits)
        if lower_limits.size == time_axis.size
        else np.array([], dtype=np.float64)
    )
    return {
        "judgement": judgement,
        "metrics": {
            "overall_spl": overall_spl,
            "overall_lower_limit": overall_lower,
            "overall_upper_limit": overall_upper,
            "unit": unit,
        },
        "curve": {
            "x": time_axis,
            "y": signal_spl,
            "lower": lower_limits,
            "upper": upper_limits,
        },
        "plot": {
            "kind": "curve",
            "x": plot_x,
            "y": plot_y,
            "lower": plot_lower,
            "upper": plot_upper,
            "x_label": "Time (s)",
            "y_label": f"SPL ({unit})",
            "title": (
                f"总体声压级：{overall_spl:.2f} {unit}"
                if config.get("show_overall_spl") and np.isfinite(overall_spl)
                else ""
            ),
            "colors": resolve_curve_colors(config),
        },
    }


def _calculate_fba(signal, sample_rate, config, v2pa_factor, **_context):
    from base.core_algorithm.response import FrequencyBandAnalyzer

    strategy_label = config.get("band_strategy", "1/3 倍频程")
    strategy_name, strategy_kwargs = _FBA_STRATEGIES.get(
        strategy_label,
        ("octave", {"fraction": 3}),
    )
    weighting = str(config.get("weighting", "A") or "A")
    if weighting in {"None", "Z（None）"}:
        weighting = "Z"
    f_min = float(config.get("f_min", 20))
    f_max = float(config.get("f_max", 20000))
    if (
        not np.isfinite(f_min)
        or not np.isfinite(f_max)
        or f_min <= 0.0
        or f_max <= f_min
    ):
        raise ValueError("FBA 分析频率范围配置无效")
    custom_edges = None
    if strategy_label == "自定义":
        custom_edges = parse_custom_bands(config.get("custom_bands_text", ""))
    analyzer = FrequencyBandAnalyzer(
        strategy=strategy_name,
        weighting=weighting,
        f_min=f_min,
        f_max=f_max,
        fraction=strategy_kwargs.get("fraction", 3),
        n_bands=int(config.get("n_bands", 40)),
        bandwidth=float(config.get("bandwidth", 100)),
        custom_edges=custom_edges,
    )
    result = analyzer.analyze(signal, fs=sample_rate, v2pa_factor=v2pa_factor)
    centers = np.asarray(
        [band.f_center for band in result.bands],
        dtype=np.float64,
    )
    levels = np.asarray(result.band_levels_weighted_db, dtype=np.float64)
    upper_limits = np.array([], dtype=np.float64)
    lower_limits = np.array([], dtype=np.float64)
    out_mask = np.zeros(levels.shape, dtype=bool)
    judgement = None
    if config.get("limit_checked"):
        if not np.any(np.isfinite(levels)):
            raise ValueError("当前 FBA 结果没有有效频段，无法执行阈值判定")
        upper_limits, lower_limits = _resolve_limits(config, centers)
        valid = np.isfinite(levels) & (
            np.isfinite(upper_limits) | np.isfinite(lower_limits)
        )
        out_mask, _, is_ok = _compare_with_limits(
            levels,
            upper_limits,
            lower_limits,
            valid_mask=valid,
        )
        judgement = "OK" if is_ok else "NG"
    return {
        "judgement": judgement,
        "metrics": {
            "overall_db": result.overall_db,
            "overall_weighted_db": result.overall_weighted_db,
            "weighting": result.weighting,
        },
        "curve": {
            "x": centers,
            "y": levels,
            "lower": lower_limits,
            "upper": upper_limits,
        },
        "plot": {
            "kind": "bar",
            "x": centers,
            "y": levels,
            "labels": [band.label for band in result.bands],
            "lower": lower_limits,
            "upper": upper_limits,
            "out_mask": out_mask,
            "x_label": "Frequency Band",
            "y_label": f"Band SPL [dB({weighting})]",
            "colors": resolve_curve_colors(config),
        },
    }


def _calculate_fft(signal, sample_rate, config, v2pa_factor, **_context):
    from base.core_algorithm.response import FftAnalyzer

    n_fft = int(config.get("n_fft", 4096))
    window = str(config.get("window", "hann") or "hann")
    overlap_ratio = float(config.get("overlap_ratio", 0.5))
    weighting = str(config.get("weighting", "Z") or "Z")
    analyzer = FftAnalyzer()
    result = analyzer.analyze(
        signal,
        fs=sample_rate,
        n_fft=n_fft,
        window=window,
        overlap_ratio=overlap_ratio,
        weighting=weighting,
        v2pa_factor=v2pa_factor,
    )
    frequency = np.asarray(result.frequencies_hz, dtype=np.float64)
    fft_db = np.asarray(result.spectrum_db, dtype=np.float64)
    baseline_db = load_fft_baseline(
        config,
        analyzer,
        frequency,
        sample_rate=sample_rate,
        n_fft=n_fft,
        window=window,
        overlap_ratio=overlap_ratio,
        weighting=result.weighting,
        v2pa_factor=v2pa_factor,
    )
    display_mode = str(
        config.get("baseline_display_mode", "overlay") or "overlay"
    )
    if display_mode == "delta" and baseline_db is None:
        raise ValueError("FFT 差值显示需要可用的背景音频基线")
    if display_mode != "delta":
        display_mode = "overlay"
    delta_db = fft_db - baseline_db if baseline_db is not None else None
    plot_y = delta_db if display_mode == "delta" else fft_db

    x_axis_scale = str(config.get("x_axis_scale", "log") or "log").lower()
    if x_axis_scale not in {"linear", "log"}:
        x_axis_scale = "log"
    focus_enabled = bool(config.get("focus_range_enabled", True))
    focus_min = float(config.get("focus_min_hz", 100))
    focus_max = float(config.get("focus_max_hz", 20000))
    if focus_enabled and (
        not np.isfinite(focus_min)
        or not np.isfinite(focus_max)
        or focus_min < 0.0
        or focus_max <= focus_min
    ):
        raise ValueError("FFT 频率聚焦范围配置无效")
    mask = np.isfinite(frequency)
    if x_axis_scale == "log":
        mask &= frequency > 0.0
    if focus_enabled:
        mask &= (frequency >= focus_min) & (frequency <= focus_max)
    plot_x = frequency[mask]
    display_y = np.asarray(plot_y, dtype=np.float64)[mask]
    if plot_x.size == 0:
        raise ValueError("当前频率聚焦范围内没有可显示的 FFT 数据")
    display_baseline = baseline_db[mask] if baseline_db is not None else None

    upper_limits = np.array([], dtype=np.float64)
    lower_limits = np.array([], dtype=np.float64)
    out_mask = np.zeros(display_y.shape, dtype=bool)
    judgement = None
    if config.get("limit_checked"):
        if not np.any(np.isfinite(display_y)):
            raise ValueError("当前 FFT 结果没有有效频点，无法执行阈值判定")
        upper_limits, lower_limits = _resolve_limits(config, plot_x)
        valid = np.isfinite(display_y) & (
            np.isfinite(upper_limits) | np.isfinite(lower_limits)
        )
        out_mask, _, is_ok = _compare_with_limits(
            display_y,
            upper_limits,
            lower_limits,
            valid_mask=valid,
        )
        judgement = "OK" if is_ok else "NG"

    y_label = (
        "FFT - Baseline [dB]"
        if display_mode == "delta"
        else (
            f"FFT Spectrum [dB({result.weighting}) SPL]"
            if result.weighting != "Z"
            else "FFT Spectrum [dB SPL]"
        )
    )
    return {
        "judgement": judgement,
        "metrics": {
            "weighting": result.weighting,
            "display_mode": display_mode,
            "peak_value": (
                float(np.nanmax(display_y))
                if np.any(np.isfinite(display_y))
                else None
            ),
        },
        "curve": {
            "x": plot_x,
            "y": display_y,
            "lower": lower_limits,
            "upper": upper_limits,
        },
        "plot": {
            "kind": "curve",
            "x": plot_x,
            "y": display_y,
            "baseline": display_baseline,
            "lower": lower_limits,
            "upper": upper_limits,
            "out_mask": out_mask,
            "x_label": "Frequency (Hz)",
            "y_label": y_label,
            "log_x": x_axis_scale == "log",
            "colors": resolve_curve_colors(config),
        },
    }


def _calculate_spec(signal, sample_rate, config, _v2pa_factor, **_context):
    result = SpectrogramAnalyzer().analyze(
        signal,
        fs=sample_rate,
        n_fft=int(config.get("n_fft", 2048)),
        hop_length=int(config.get("hop_length", 256)),
        window=str(config.get("window_func", "hann") or "hann"),
        scale=str(config.get("freq_scale_type", "linear") or "linear"),
        max_time_bins=_MAX_SPEC_TIME_BINS,
    )
    levels = None
    if config.get("custom_limit"):
        top = float(config.get("top_limit", 70))
        bottom = float(config.get("bottom_limit", 50))
        middle = (top - bottom) / 2.0
        levels = (bottom - middle, top + middle)
    title = (
        "Spectrogram (Log Scale)"
        if result.scale == "log"
        else "Spectrogram (Linear Scale)"
    )
    return {
        "judgement": None,
        "metrics": {},
        "curve": {},
        "plot": {
            "kind": "spectrogram",
            "x": result.times_s,
            "y": result.frequencies_hz,
            "z": result.values_db,
            "color_map": str(config.get("color_map", "viridis") or "viridis"),
            "levels": levels,
            "x_label": "Time (s)",
            "y_label": "Frequency (Hz)",
            "title": title,
        },
    }


def _calculate_ai(
    signal,
    sample_rate,
    config,
    _v2pa_factor,
    *,
    source,
    sequence_snapshot,
):
    from base.model_runtime_validation import validate_model_duration
    from base.predict_model import predict_from_audio
    from base.training_model_management import TrainingModelManagement
    from consts import error_code
    from consts.running_consts import DEFAULT_DIR

    model_name = str(config.get("analyse_model_name") or "").strip()
    if not model_name:
        raise ValueError("未配置 AI 分析模型")
    manager = TrainingModelManagement()
    code, query_result = manager.get_model_path_from_db(model_name)
    if code != error_code.OK or not query_result:
        raise ValueError("模型不存在，请重新选择")
    model_path, config_path = query_result[0]
    model_path = os.path.abspath(os.path.join(DEFAULT_DIR, model_path))
    config_path = os.path.abspath(os.path.join(DEFAULT_DIR, config_path))
    if not os.path.isfile(model_path):
        raise ValueError("模型不存在，请重新选择")

    sequence_configs = sequence_snapshot.get("sequence_config") or []
    acq_mode = None
    if sequence_configs and isinstance(sequence_configs[0], dict):
        acq_mode = sequence_configs[0].get("seq1", {}).get("acq", {}).get("mode")
    mode = "test" if source == "自动分析" else "view"
    matched, message = validate_model_duration(
        model_name,
        len(signal),
        sample_rate=sample_rate,
        config_path=config_path,
        model_manager=manager,
    )
    if not matched:
        raise ValueError(message or f"{mode}/{acq_mode or ''} 模型时长不匹配")

    return_text, prediction_config = predict_from_audio(
        signals=[np.asarray(signal, dtype=np.float32)],
        file_names=["modelpredict.wav"],
        fs=[sample_rate],
        load_model_path=model_path,
        config_path=config_path,
    )
    response = json.loads(return_text)
    if response.get("ret_code") != error_code.OK:
        raise RuntimeError(str(response.get("ret_msg") or "AI 分析失败"))
    prediction = response.get("result") or []
    if not prediction or len(prediction[0]) < 3:
        raise RuntimeError("AI 未产生分析结果")
    label = str(prediction[0][1] or "").upper()
    if label not in {"OK", "NG"}:
        raise RuntimeError("AI 未产生 OK/NG 判定")
    output_value = float(prediction[0][2])
    threshold = _finite_or_none((prediction_config or {}).get("acc_req"))
    metrics = {
        "model_output_value": output_value,
        "decision_threshold": threshold,
        "ok_score": round(output_value * 100.0, 2),
        "ng_score": round((1.0 - output_value) * 100.0, 2),
        "model_name": model_name,
    }
    return {
        "judgement": label,
        "metrics": metrics,
        "curve": {},
        "plot": {
            "kind": "values",
            "title": "AI 分析结果",
            "values": {
                "最终判定": label,
                "评分模型": model_name,
                "模型输出值": output_value,
                "判定阈值": threshold,
            },
        },
    }


_HANDLERS = {
    "SPL": _calculate_spl,
    "FBA": _calculate_fba,
    "FFT": _calculate_fft,
    "Spec": _calculate_spec,
    "AI": _calculate_ai,
}


def calculate_analysis_instance(
    analysis_type,
    signal,
    sample_rate,
    config,
    v2pa_factor,
    *,
    source,
    sequence_snapshot,
):
    """Run one configured analysis channel and return plain calculation data."""
    try:
        handler = _HANDLERS[analysis_type]
    except KeyError as exc:
        raise ValueError(f"不支持的分析类型：{analysis_type}") from exc
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("分析通道没有音频数据")
    if int(sample_rate) <= 0:
        raise ValueError("采样率必须为正整数")
    return handler(
        values,
        int(sample_rate),
        dict(config or {}),
        float(v2pa_factor),
        source=source,
        sequence_snapshot=sequence_snapshot or {},
    )


__all__ = [
    "_MAX_SPEC_TIME_BINS",
    "calculate_analysis_instance",
    "load_fft_baseline",
    "parse_custom_bands",
    "smooth_fft_baseline",
]
