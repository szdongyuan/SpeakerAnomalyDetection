"""GUI-free SPL analysis for standalone recorder channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from base.core_algorithm.harmonic_distortion.weighted import (
    apply_weighting_filter,
)
from base.pre_processing.audio_thd_frequency_response_analysis import (
    AudioThdFrequencyResponseAnalysis,
)
from base.pre_processing.spl_runtime_config import (
    calculate_overall_spl,
    evaluate_spl_limits,
    resolve_spl_unit,
)
from base.utils.smooth import smooth


@dataclass(frozen=True)
class StandaloneSplResult:
    time_seconds: np.ndarray
    spl_db: np.ndarray
    overall_spl: float | None
    judged_ok: bool | None
    deviation_db: float | None
    unit: str


def _positive_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _positive_integral_sample_rate(value: Any) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
    ):
        raise ValueError("sample rate must be a positive integer")
    rate = int(value)
    if rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    return rate


def _finite_nonnegative_time(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _validated_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    resolved = dict(config)
    weighting = str(resolved.get("weighting", "Z") or "Z").strip().upper()
    if weighting not in {"NONE", "Z", "A", "B", "C", "D"}:
        raise ValueError(f"unsupported SPL weighting: {weighting}")
    resolved["weighting"] = weighting

    if bool(resolved.get("analysis_time_range_enabled", False)):
        _finite_nonnegative_time(
            resolved.get("analysis_start_time_sec", 0.0),
            label="analysis start time",
        )
        _finite_nonnegative_time(
            resolved.get("analysis_end_time_sec", 0.0),
            label="analysis end time",
        )

    if bool(resolved.get("limit_checked", False)):
        metric = str(
            resolved.get("limit_metric", "curve_y") or "curve_y"
        ).lower()
        if metric not in {"curve_y", "overall_spl"}:
            raise ValueError(f"unsupported SPL limit metric: {metric}")
        resolved["limit_metric"] = metric
    return resolved


def _resolve_analysis_range(
    signal: np.ndarray,
    sample_rate: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    if not bool(config.get("analysis_time_range_enabled", False)):
        return signal, 0

    start_seconds = _finite_nonnegative_time(
        config.get("analysis_start_time_sec", 0.0),
        label="analysis start time",
    )
    end_seconds = _finite_nonnegative_time(
        config.get("analysis_end_time_sec", 0.0),
        label="analysis end time",
    )
    if end_seconds != 0.0 and end_seconds <= start_seconds:
        raise ValueError("analysis end must be greater than analysis start")
    source_duration = signal.size / sample_rate
    if start_seconds >= source_duration:
        raise ValueError(
            "analysis start is at or after the end of the source signal"
        )
    try:
        start_sample = int(np.floor(start_seconds * sample_rate))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("analysis start time cannot be converted to a sample index") from exc
    if start_sample >= signal.size:
        raise ValueError(
            "analysis start is at or after the end of the source signal"
        )
    if end_seconds == 0.0 or end_seconds >= source_duration:
        end_sample = signal.size
    else:
        try:
            end_sample = int(np.ceil(end_seconds * sample_rate))
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "analysis end time cannot be converted to a sample index"
            ) from exc
    if end_sample <= start_sample:
        raise ValueError("analysis end must be greater than analysis start")
    return signal[start_sample:end_sample], start_sample


def _extrema_reduction_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    size = int(values.size)
    if size <= max_points:
        return np.arange(size, dtype=np.intp)
    if max_points == 2:
        return np.asarray([0, size - 1], dtype=np.intp)
    if max_points == 3:
        interior = values[1:-1]
        most_extreme = int(np.argmax(np.abs(interior - np.median(interior)))) + 1
        return np.asarray([0, most_extreme, size - 1], dtype=np.intp)

    interior = np.arange(1, size - 1, dtype=np.intp)
    bucket_count = min(interior.size, (max_points - 2) // 2)
    buckets = np.array_split(interior, bucket_count)
    selected = [0]
    for bucket in buckets:
        if bucket.size == 0:
            continue
        bucket_values = values[bucket]
        minimum = int(bucket[int(np.argmin(bucket_values))])
        maximum = int(bucket[int(np.argmax(bucket_values))])
        selected.extend(sorted({minimum, maximum}))
    selected.append(size - 1)
    return np.asarray(selected, dtype=np.intp)


def _read_only_owned(values) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64)


def analyze_standalone_spl(
    voltage,
    *,
    sample_rate: int,
    v2pa_factor: float,
    config: Mapping,
    max_plot_points: int,
    source_start_sample: int = 0,
    preweighted: bool = False,
) -> StandaloneSplResult:
    """Analyze one voltage channel and return a bounded immutable SPL result."""
    rate = _positive_integral_sample_rate(sample_rate)
    factor = _positive_finite_number(v2pa_factor, label="Pa/V factor")
    if (
        isinstance(source_start_sample, (bool, np.bool_))
        or not isinstance(source_start_sample, Integral)
        or int(source_start_sample) < 0
    ):
        raise ValueError("source_start_sample must be a nonnegative integer")
    source_offset = int(source_start_sample)
    try:
        source_offset_float = float(source_offset)
    except OverflowError as exc:
        raise ValueError(
            "source_start_sample must be a nonnegative integer with a finite time offset"
        ) from exc
    if not np.isfinite(source_offset_float):
        raise ValueError(
            "source_start_sample must be a nonnegative integer with a finite time offset"
        )
    if not isinstance(preweighted, (bool, np.bool_)):
        raise ValueError("preweighted must be a boolean")
    input_is_preweighted = bool(preweighted)
    if (
        isinstance(max_plot_points, (bool, np.bool_))
        or not isinstance(max_plot_points, Integral)
        or int(max_plot_points) < 2
    ):
        raise ValueError("max_plot_points must be an integer of at least 2")
    point_limit = int(max_plot_points)
    resolved_config = _validated_config(config)

    try:
        source_signal = np.asarray(voltage, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("voltage samples must be numeric") from exc
    if source_signal.ndim != 1 or source_signal.size == 0:
        raise ValueError("voltage samples must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(source_signal)):
        raise ValueError("voltage samples must all be finite")

    weighting = resolved_config["weighting"]
    if weighting not in {"NONE", "Z"} and not input_is_preweighted:
        weighted_signal = np.asarray(
            apply_weighting_filter(
                source_signal,
                rate,
                weighting=weighting,
                zero_phase=False,
            ),
            dtype=np.float64,
        )
    else:
        weighted_signal = source_signal
    analysis_signal, analysis_start_sample = _resolve_analysis_range(
        weighted_signal,
        rate,
        resolved_config,
    )

    show_overall = bool(resolved_config.get("show_overall_spl", False))
    judge_overall = (
        bool(resolved_config.get("limit_checked", False))
        and resolved_config.get("limit_metric", "curve_y") == "overall_spl"
    )
    overall_spl = None
    if show_overall or judge_overall:
        overall_spl = calculate_overall_spl(
            analysis_signal,
            20e-6,
            v2pa_factor=factor,
        )

    full_spl = np.asarray(
        AudioThdFrequencyResponseAnalysis().spl_calculation(
            analysis_signal,
            20e-6,
            window_size=1201,
            v2pa_factor=factor,
            trim_edges=True,
        ),
        dtype=np.float64,
    )
    if full_spl.ndim != 1 or full_spl.size == 0:
        raise ValueError("SPL calculation produced no display values")
    if bool(resolved_config.get("smooth_checked", False)):
        full_spl = np.asarray(
            smooth(full_spl, window_size=1102, method="savgol"),
            dtype=np.float64,
        )
    if not np.all(np.isfinite(full_spl)):
        raise ValueError("SPL calculation produced non-finite values")

    start_index = 0 if full_spl.size == analysis_signal.size else 1201 // 2
    full_time = (
        np.arange(full_spl.size, dtype=np.float64)
        + float(start_index)
        + float(analysis_start_sample)
        + source_offset_float
    ) / rate
    judged_ok, deviation_db = evaluate_spl_limits(
        resolved_config,
        full_time,
        full_spl,
        overall_spl,
    )

    display_indices = _extrema_reduction_indices(full_spl, point_limit)
    display_time = _read_only_owned(full_time[display_indices])
    display_spl = _read_only_owned(full_spl[display_indices])

    del display_indices, full_time, full_spl
    del analysis_signal, weighted_signal, source_signal

    return StandaloneSplResult(
        time_seconds=display_time,
        spl_db=display_spl,
        overall_spl=(None if overall_spl is None else float(overall_spl)),
        judged_ok=judged_ok,
        deviation_db=(None if deviation_db is None else float(deviation_db)),
        unit=resolve_spl_unit(weighting),
    )
