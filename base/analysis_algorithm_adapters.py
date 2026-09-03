"""Headless adapters around the existing analysis calculation modules.

The worker must not import ``ui.signal_analysis_window``: that module eagerly
loads every UI and model dependency.  These adapters keep the established
calculation formulas and configuration semantics, but return plain data that a
short-lived child process can serialize or render without constructing Qt
widgets.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from ui.curve_style import resolve_curve_colors
from ui.ui_analysis_config.manual_limit_segments import (
    limits_from_constant_values,
    limits_from_manual_config,
)
from ui.ui_analysis_config.threshold_csv_manual import (
    validate_limit_data_values,
)


_FBA_STRATEGIES = {
    "1/1 倍频程": ("octave", {"fraction": 1}),
    "1/3 倍频程": ("octave", {"fraction": 3}),
    "1/6 倍频程": ("octave", {"fraction": 6}),
    "1/12 倍频程": ("octave", {"fraction": 12}),
    "Bark": ("bark", {}),
    "等宽": ("equal_width", {}),
    "自定义": ("custom", {}),
}
_MAX_SPEC_TIME_BINS = 2_000
_SPEC_BATCH_FRAMES = 256


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
    handlers = {
        "SPL": _calculate_spl,
        "FBA": _calculate_fba,
        "FFT": _calculate_fft,
        "Spec": _calculate_spec,
        "AI": _calculate_ai,
    }
    try:
        handler = handlers[analysis_type]
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


def _calculate_spl(
    signal,
    sample_rate,
    config,
    v2pa_factor,
    **_context,
):
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
    from base.analysis_display_payload import min_max_envelope

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


def _calculate_fba(
    signal,
    sample_rate,
    config,
    v2pa_factor,
    **_context,
):
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
        custom_edges = _parse_custom_bands(config.get("custom_bands_text", ""))
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
    result = analyzer.analyze(
        signal,
        fs=sample_rate,
        v2pa_factor=v2pa_factor,
    )
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


def _calculate_fft(
    signal,
    sample_rate,
    config,
    v2pa_factor,
    **_context,
):
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
    baseline_db = _load_fft_baseline(
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


def _calculate_spec(
    signal,
    sample_rate,
    config,
    _v2pa_factor,
    **_context,
):
    n_fft = int(config.get("n_fft", 2048))
    hop_length = int(config.get("hop_length", 256))
    color_map = str(config.get("color_map", "viridis") or "viridis")
    window = str(config.get("window_func", "hann") or "hann")
    scale = str(config.get("freq_scale_type", "linear") or "linear")
    if scale == "log":
        import librosa

        from base.pre_processing.audio_thd_frequency_response_analysis import (
            AudioThdFrequencyResponseAnalysis,
        )

        complex_values, frequencies, times = (
            AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=signal,
                sr=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                fmin=librosa.note_to_hz("C1"),
            )
        )
        magnitude, times = _pool_spectrogram_time(
            np.abs(complex_values),
            times,
            max_time_bins=_MAX_SPEC_TIME_BINS,
        )
        values_db = librosa.amplitude_to_db(
            magnitude,
            ref=20e-6,
        )
        title = "Spectrogram (Log Scale)"
    else:
        if n_fft <= 0 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("Spec 的 FFT 点数或帧移配置无效")
        frequencies, times, magnitude = _chunked_linear_spectrogram(
            signal,
            sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            max_time_bins=_MAX_SPEC_TIME_BINS,
        )
        values_db = _amplitude_to_db(
            magnitude,
            reference=20e-6,
            top_db=80.0,
        )
        title = "Spectrogram (Linear Scale)"
    levels = None
    if config.get("custom_limit"):
        top = float(config.get("top_limit", 70))
        bottom = float(config.get("bottom_limit", 50))
        middle = (top - bottom) / 2.0
        levels = (bottom - middle, top + middle)
    return {
        "judgement": None,
        "metrics": {},
        "curve": {},
        "plot": {
            "kind": "spectrogram",
            "x": np.asarray(times, dtype=np.float64),
            "y": np.asarray(frequencies, dtype=np.float64),
            "z": np.asarray(values_db, dtype=np.float32),
            "color_map": color_map,
            "levels": levels,
            "x_label": "Time (s)",
            "y_label": "Frequency (Hz)",
            "title": title,
        },
    }


def _chunked_linear_spectrogram(
    signal,
    sample_rate,
    *,
    n_fft,
    hop_length,
    window,
    max_time_bins,
):
    """Calculate a bounded full-duration STFT without a full complex matrix."""
    from scipy.fft import rfft
    from scipy.signal import get_window

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Spec 分析没有音频数据")
    if type(max_time_bins) is not int or max_time_bins < 1:
        raise ValueError("Spec 显示时间列上限必须是正整数")

    pad = n_fft // 2
    pad_mode = "reflect" if values.size > 1 else "edge"
    centered = np.pad(values, (pad, pad), mode=pad_mode)
    if centered.size < n_fft:
        centered = np.pad(centered, (0, n_fft - centered.size))
    remainder = (centered.size - n_fft) % hop_length
    if remainder:
        centered = np.pad(centered, (0, hop_length - remainder))

    frame_count = 1 + (centered.size - n_fft) // hop_length
    time_bin_count = min(frame_count, max_time_bins)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    pooled = np.full(
        (frequencies.size, time_bin_count),
        -np.inf,
        dtype=np.float32,
    )
    window_values = np.asarray(
        get_window(window, n_fft, fftbins=True),
        dtype=np.float32,
    )
    sliding = np.lib.stride_tricks.sliding_window_view(centered, n_fft)
    for first_frame in range(0, frame_count, _SPEC_BATCH_FRAMES):
        frame_indices = np.arange(
            first_frame,
            min(first_frame + _SPEC_BATCH_FRAMES, frame_count),
            dtype=np.int64,
        )
        starts = frame_indices * hop_length
        frames = np.asarray(sliding[starts], dtype=np.float32)
        magnitude = np.abs(
            rfft(frames * window_values, n=n_fft, axis=1, workers=1)
        ).astype(np.float32, copy=False)
        bucket_indices = frame_indices * time_bin_count // frame_count
        for bucket in np.unique(bucket_indices):
            bucket_values = magnitude[bucket_indices == bucket]
            pooled[:, int(bucket)] = np.maximum(
                pooled[:, int(bucket)],
                np.max(bucket_values, axis=0),
            )

    pooled[~np.isfinite(pooled)] = 0.0
    boundaries = np.linspace(0, frame_count, time_bin_count + 1)
    center_frames = (boundaries[:-1] + boundaries[1:] - 1.0) * 0.5
    times = np.maximum(center_frames, 0.0) * hop_length / float(sample_rate)
    return frequencies, times, pooled


def _pool_spectrogram_time(values, times, *, max_time_bins):
    magnitude = np.asarray(values, dtype=np.float32)
    time_values = np.asarray(times, dtype=np.float64).reshape(-1)
    if magnitude.ndim != 2 or magnitude.shape[1] != time_values.size:
        raise ValueError("Spec 频谱矩阵与时间轴长度不一致")
    if time_values.size <= max_time_bins:
        return magnitude, time_values

    pooled = np.empty(
        (magnitude.shape[0], max_time_bins),
        dtype=np.float32,
    )
    pooled_times = np.empty(max_time_bins, dtype=np.float64)
    edges = np.linspace(0, time_values.size, max_time_bins + 1, dtype=np.int64)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        pooled[:, index] = np.max(magnitude[:, start:stop], axis=1)
        pooled_times[index] = (time_values[start] + time_values[stop - 1]) * 0.5
    return pooled, pooled_times


def _amplitude_to_db(values, *, reference, top_db):
    magnitude = np.asarray(values, dtype=np.float64)
    minimum_amplitude = 1e-5
    log_values = 20.0 * np.log10(np.maximum(minimum_amplitude, magnitude))
    log_values -= 20.0 * np.log10(max(minimum_amplitude, abs(float(reference))))
    if top_db is not None and log_values.size:
        log_values = np.maximum(log_values, np.nanmax(log_values) - float(top_db))
    return log_values


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


def _resolve_spl_overall_limits(config):
    scalar = {
        "constant_upper_enabled": bool(config.get("scalar_upper_enabled", True)),
        "constant_upper_value": config.get("scalar_upper_value", 100.0),
        "constant_lower_enabled": bool(config.get("scalar_lower_enabled", False)),
        "constant_lower_value": config.get("scalar_lower_value", 0.0),
    }
    _, upper, lower = limits_from_constant_values(scalar, [0.0])
    return np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)


def _resolve_limit_source(config, target_x):
    mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if mode == "manual":
        return limits_from_manual_config(config, target_x)
    if mode != "csv":
        raise ValueError(f"不支持的阈值模式: {mode}")
    limit_data = config.get("limit_data")
    if not limit_data:
        raise ValueError("已启用阈值，但未加载 CSV 配置文件")
    try:
        x_values, upper, lower = limit_data
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV 阈值数据格式不正确") from exc
    validate_limit_data_values(limit_data)
    return x_values, upper, lower


def _resolve_limits(config, target_x):
    mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if mode == "manual":
        _, upper, lower = limits_from_manual_config(config, target_x)
        upper_values = np.asarray(upper, dtype=np.float64)
        lower_values = np.asarray(lower, dtype=np.float64)
    elif mode == "csv":
        limit_data = config.get("limit_data")
        if not limit_data:
            raise ValueError("已启用阈值，但未加载 CSV 配置文件")
        try:
            csv_x, csv_upper, csv_lower = limit_data
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV 阈值数据格式不正确") from exc
        validate_limit_data_values(limit_data)
        upper_values = _interpolate_limit_side(target_x, csv_x, csv_upper)
        lower_values = _interpolate_limit_side(target_x, csv_x, csv_lower)
    else:
        raise ValueError(f"不支持的阈值模式: {mode}")
    overlap = np.isfinite(upper_values) & np.isfinite(lower_values)
    if np.any(lower_values[overlap] > upper_values[overlap]):
        raise ValueError("下限不能大于上限")
    return upper_values, lower_values


def _interpolate_limit_side(target_x, raw_x, raw_values):
    x_values = np.asarray(list(raw_x), dtype=np.float64)
    side_values = np.asarray(list(raw_values), dtype=np.float64)
    if x_values.size != side_values.size:
        raise ValueError("CSV 阈值数据长度不一致")
    finite = np.isfinite(x_values) & np.isfinite(side_values)
    target = np.asarray(target_x, dtype=np.float64)
    output = np.full(target.shape, np.nan, dtype=np.float64)
    if not np.any(finite):
        return output
    points = {
        float(x_value): float(side_value)
        for x_value, side_value in zip(x_values[finite], side_values[finite])
    }
    sorted_x = np.asarray(sorted(points), dtype=np.float64)
    sorted_y = np.asarray([points[value] for value in sorted_x], dtype=np.float64)
    inside = (target >= sorted_x[0]) & (target <= sorted_x[-1])
    if np.any(inside):
        output[inside] = np.interp(target[inside], sorted_x, sorted_y)
    return output


def _interpolate_spl_limit_curves(target_x, raw_x, raw_upper, raw_lower):
    target = np.asarray(target_x, dtype=np.float64)
    x_values = np.asarray(raw_x, dtype=np.float64)
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    return (
        _interpolate_spl_limit_side(
            target,
            x_values,
            np.asarray(raw_upper, dtype=np.float64)[order],
        ),
        _interpolate_spl_limit_side(
            target,
            x_values,
            np.asarray(raw_lower, dtype=np.float64)[order],
        ),
    )


def _interpolate_spl_limit_side(target, x_values, y_values):
    output = np.full(target.shape, np.nan, dtype=np.float64)
    if x_values.size == 0 or y_values.size != x_values.size:
        return output
    finite_target = np.isfinite(target)
    right = np.searchsorted(x_values, target, side="right")
    left = right - 1
    between = finite_target & (left >= 0) & (right < x_values.size)
    safe_left = np.clip(left, 0, x_values.size - 1)
    safe_right = np.clip(right, 0, x_values.size - 1)
    left_x = x_values[safe_left]
    right_x = x_values[safe_right]
    left_y = y_values[safe_left]
    right_y = y_values[safe_right]
    segments = (
        between
        & (right_x > left_x)
        & np.isfinite(left_y)
        & np.isfinite(right_y)
    )
    ratio = (target[segments] - left_x[segments]) / (
        right_x[segments] - left_x[segments]
    )
    output[segments] = left_y[segments] + ratio * (
        right_y[segments] - left_y[segments]
    )
    finite_rows = np.isfinite(y_values)
    if np.any(finite_rows):
        exact_x, first = np.unique(x_values[finite_rows], return_index=True)
        exact_y = y_values[finite_rows][first]
        positions = np.searchsorted(exact_x, target, side="left")
        safe = np.clip(positions, 0, exact_x.size - 1)
        exact = finite_target & np.isclose(
            target,
            exact_x[safe],
            rtol=1e-12,
            atol=1e-12,
        )
        output[exact] = exact_y[safe[exact]]
    return output


def _compare_with_limits(plot_y, upper_limits, lower_limits, valid_mask=None):
    values = np.asarray(plot_y, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    if values.shape != upper.shape or values.shape != lower.shape:
        raise ValueError("分析曲线与上下限长度不一致")
    valid = (
        np.ones(values.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    upper_ok = np.isfinite(upper)
    lower_ok = np.isfinite(lower)
    out_mask = valid & (
        (upper_ok & (values > upper)) | (lower_ok & (values < lower))
    )
    deviation = 0.0
    is_ok = not np.any(out_mask)
    if not is_ok:
        above = np.where(out_mask & upper_ok, values - upper, 0.0)
        below = np.where(out_mask & lower_ok, lower - values, 0.0)
        deviation = float(np.nanmax(np.maximum(above, below)))
    else:
        inside = valid & np.isfinite(values)
        if np.any(inside):
            margin_upper = np.where(
                upper_ok[inside],
                upper[inside] - values[inside],
                np.inf,
            )
            margin_lower = np.where(
                lower_ok[inside],
                values[inside] - lower[inside],
                np.inf,
            )
            margins = np.minimum(margin_upper, margin_lower)
            margins = margins[np.isfinite(margins)]
            if margins.size:
                deviation = float(np.min(margins))
    return out_mask, round(deviation, 2), bool(is_ok)


def _load_fft_baseline(
    config,
    analyzer,
    frequency,
    *,
    sample_rate,
    n_fft,
    window,
    overlap_ratio,
    weighting,
    v2pa_factor,
):
    path = str(config.get("baseline_file_path", "") or "").strip()
    if not path:
        return None
    import librosa

    baseline_signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    baseline_result = analyzer.analyze(
        baseline_signal,
        fs=sample_rate,
        n_fft=n_fft,
        window=window,
        overlap_ratio=overlap_ratio,
        weighting=weighting,
        v2pa_factor=v2pa_factor,
    )
    baseline_db = np.interp(
        frequency,
        np.asarray(baseline_result.frequencies_hz, dtype=np.float64),
        np.asarray(baseline_result.spectrum_db, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    if config.get("baseline_smooth_third_octave"):
        baseline_db = _smooth_fft_baseline(frequency, baseline_db)
    return baseline_db


def _smooth_fft_baseline(frequency, baseline_db):
    frequency = np.asarray(frequency, dtype=np.float64)
    baseline = np.asarray(baseline_db, dtype=np.float64)
    smoothed = np.full_like(baseline, np.nan, dtype=np.float64)
    factor = 2.0 ** (1.0 / 6.0)
    valid = np.isfinite(frequency) & np.isfinite(baseline)
    if not np.any(valid):
        return smoothed
    order = np.argsort(frequency[valid])
    sorted_frequency = frequency[valid][order]
    sorted_power = np.power(10.0, baseline[valid][order] / 10.0)
    prefix = np.concatenate(([0.0], np.cumsum(sorted_power)))
    centers = np.isfinite(frequency) & (frequency > 0.0)
    left = np.searchsorted(
        sorted_frequency,
        frequency[centers] / factor,
        side="left",
    )
    right = np.searchsorted(
        sorted_frequency,
        frequency[centers] * factor,
        side="right",
    )
    counts = right - left
    sums = prefix[right] - prefix[left]
    values = np.full(counts.shape, np.nan, dtype=np.float64)
    non_empty = counts > 0
    values[non_empty] = 10.0 * np.log10(
        np.maximum(sums[non_empty] / counts[non_empty], 1e-30)
    )
    smoothed[centers] = values
    return smoothed


def _parse_custom_bands(text):
    edges = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = (
            [part.strip() for part in line.split(",") if part.strip()]
            if "," in line
            else [part for part in line.replace("\t", " ").split(" ") if part]
        )
        label = None
        try:
            if len(parts) == 1 and "-" in parts[0]:
                lower, upper = parts[0].split("-", 1)
                low, high = float(lower.strip()), float(upper.strip())
            elif len(parts) >= 2:
                low, high = float(parts[0]), float(parts[1])
                if len(parts) >= 3:
                    label = " ".join(parts[2:]).strip() or None
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"格式错误: {raw!r}") from exc
        if low <= 0 or high <= 0:
            raise ValueError(f"频率必须为正数: {raw!r}")
        if high <= low:
            raise ValueError(f"频段上限必须大于下限: {raw!r}")
        edges.append((low, high, label))
    edges.sort(key=lambda item: item[0])
    if not edges:
        raise ValueError("请至少输入一个频段")
    for index in range(1, len(edges)):
        if edges[index][0] < edges[index - 1][1]:
            raise ValueError("自定义频段不允许重叠，请检查相邻频段边界")
    return edges


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
