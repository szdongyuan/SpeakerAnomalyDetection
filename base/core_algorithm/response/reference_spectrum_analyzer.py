"""
Reference spectrum analysis helpers for reference-vs-current comparison.

This module provides a reusable backend-only analyzer for:
- per-channel Welch spectrum calculation
- optional octave smoothing
- threshold-band comparison against a reference spectrum
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from base.utils.octave_smoothing import smooth_to_octave_grid


SUPPORTED_WINDOWS = {"hann", "hamming", "blackman"}
SUPPORTED_SMOOTHING = {0, 1, 3, 6, 12, 24, 48}
MIN_SIGNAL_SAMPLES = 16


@dataclass(frozen=True)
class ReferenceSpectrumParams:
    window: str = "hann"
    nperseg: int = 4096
    overlap_ratio: float = 0.5
    smoothing: int = 0


@dataclass(frozen=True)
class ReferenceSpectrumChannelResult:
    channel_index: int
    frequencies_hz: np.ndarray
    spectrum_db: np.ndarray


@dataclass(frozen=True)
class ReferenceSpectrumCompareResult:
    channel_index: int
    frequencies_hz: np.ndarray
    reference_db: np.ndarray
    current_db: np.ndarray
    lower_limit_db: np.ndarray
    upper_limit_db: np.ndarray
    band_mask: np.ndarray
    out_of_range_mask: np.ndarray
    max_over_upper_db: float
    max_under_lower_db: float
    max_exceed_db: float
    out_of_range_point_count: int
    out_of_range_ratio: float
    channel_ok: bool | None
    threshold_enabled: bool


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _remove_dc_offset(x: np.ndarray) -> np.ndarray:
    arr = _as_1d_float_array(x)
    if arr.size == 0:
        return arr
    return arr - float(np.mean(arr))


def _validate_params(params: ReferenceSpectrumParams) -> None:
    if not isinstance(params, ReferenceSpectrumParams):
        raise ValueError("params must be ReferenceSpectrumParams")
    if str(params.window).strip().lower() not in SUPPORTED_WINDOWS:
        raise ValueError(f"Unsupported Welch window: {params.window!r}")
    if int(params.nperseg) <= 0:
        raise ValueError(f"nperseg must be positive, got {params.nperseg}")
    overlap_ratio = float(params.overlap_ratio)
    if not np.isfinite(overlap_ratio) or overlap_ratio < 0.0 or overlap_ratio >= 1.0:
        raise ValueError(f"overlap_ratio must satisfy 0 <= value < 1, got {params.overlap_ratio}")
    smoothing = int(params.smoothing)
    if smoothing not in SUPPORTED_SMOOTHING:
        raise ValueError(
            f"smoothing must be one of {sorted(SUPPORTED_SMOOTHING)}, got {params.smoothing}"
        )


def _normalize_multi_channel_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected mono or (frames, channels) audio, got shape {arr.shape}")


def _compute_welch_power(
    x: np.ndarray,
    sample_rate: int,
    params: ReferenceSpectrumParams,
) -> tuple[np.ndarray, np.ndarray]:
    signal_arr = _as_1d_float_array(x)
    if signal_arr.size < MIN_SIGNAL_SAMPLES:
        raise ValueError(
            f"Signal is too short for spectrum analysis: need at least {MIN_SIGNAL_SAMPLES} samples"
        )

    nperseg = min(int(params.nperseg), int(signal_arr.size))
    if nperseg < MIN_SIGNAL_SAMPLES:
        raise ValueError(
            f"Effective nperseg is too small for stable analysis: got {nperseg}, "
            f"need at least {MIN_SIGNAL_SAMPLES}"
        )
    noverlap = min(int(round(nperseg * float(params.overlap_ratio))), nperseg - 1)
    freqs_hz, power = sp_signal.welch(
        signal_arr,
        fs=int(sample_rate),
        window=str(params.window).strip().lower(),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="spectrum",
    )
    return np.asarray(freqs_hz, dtype=np.float64), np.asarray(power, dtype=np.float64)


def _power_to_db(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    power_arr = np.asarray(power, dtype=np.float64)
    safe_power = np.maximum(power_arr, float(eps))
    return 10.0 * np.log10(safe_power)


def _apply_optional_smoothing(
    freq_hz: np.ndarray,
    spectrum_db: np.ndarray,
    smoothing: int,
) -> tuple[np.ndarray, np.ndarray]:
    smoothing_value = int(smoothing)
    if smoothing_value <= 0:
        return np.asarray(freq_hz, dtype=np.float64), np.asarray(spectrum_db, dtype=np.float64)

    if smoothing_value not in SUPPORTED_SMOOTHING:
        raise ValueError(f"Unsupported smoothing value: {smoothing}")

    freq_arr = np.asarray(freq_hz, dtype=np.float64)
    value_arr = np.asarray(spectrum_db, dtype=np.float64)
    mask = np.isfinite(freq_arr) & np.isfinite(value_arr) & (freq_arr > 0.0)
    freq_valid = freq_arr[mask]
    value_valid = value_arr[mask]
    if freq_valid.size < 2:
        return freq_arr, value_arr

    smoothed_freq_hz, smoothed_db = smooth_to_octave_grid(
        freq_valid,
        value_valid,
        fraction=smoothing_value,
        method="log",
    )
    return np.asarray(smoothed_freq_hz, dtype=np.float64), np.asarray(smoothed_db, dtype=np.float64)


def _align_curve_to_reference_axis(
    current_freq_hz: np.ndarray,
    current_db: np.ndarray,
    reference_freq_hz: np.ndarray,
) -> np.ndarray:
    x_current = np.asarray(current_freq_hz, dtype=np.float64)
    y_current = np.asarray(current_db, dtype=np.float64)
    x_reference = np.asarray(reference_freq_hz, dtype=np.float64)

    mask = np.isfinite(x_current) & np.isfinite(y_current) & (x_current >= 0.0)
    x_current = x_current[mask]
    y_current = y_current[mask]
    if x_current.size < 2:
        raise ValueError("Current spectrum does not have enough valid points for interpolation")

    sort_idx = np.argsort(x_current)
    x_current = x_current[sort_idx]
    y_current = y_current[sort_idx]
    x_current, unique_idx = np.unique(x_current, return_index=True)
    y_current = y_current[unique_idx]
    if x_current.size < 2:
        raise ValueError("Current spectrum frequency axis must contain at least two unique points")

    aligned = np.interp(x_reference, x_current, y_current)
    in_range = (x_reference >= float(np.min(x_current))) & (x_reference <= float(np.max(x_current)))
    return np.where(in_range, aligned, np.nan)


def _build_threshold_band(
    reference_db: np.ndarray,
    lower_offset_db: float,
    upper_offset_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_offset = float(lower_offset_db)
    upper_offset = float(upper_offset_db)
    if lower_offset > upper_offset:
        raise ValueError(
            f"lower_offset_db must be <= upper_offset_db, got {lower_offset_db} and {upper_offset_db}"
        )
    reference_arr = np.asarray(reference_db, dtype=np.float64)
    return reference_arr + lower_offset, reference_arr + upper_offset


def _build_band_mask(freq_hz: np.ndarray, start_freq_hz: float, end_freq_hz: float) -> np.ndarray:
    start_value = float(start_freq_hz)
    end_value = float(end_freq_hz)
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        raise ValueError("Analysis band frequencies must be finite values")
    if start_value >= end_value:
        raise ValueError(
            f"Invalid analysis band: start_freq_hz ({start_freq_hz}) must be smaller than end_freq_hz ({end_freq_hz})"
        )
    freq_arr = np.asarray(freq_hz, dtype=np.float64)
    return (freq_arr >= start_value) & (freq_arr <= end_value)


def validate_same_frequency_axis(channel_results: list[ReferenceSpectrumChannelResult]) -> np.ndarray:
    if not channel_results:
        raise ValueError("channel_results cannot be empty")
    base_axis = np.asarray(channel_results[0].frequencies_hz, dtype=np.float64)
    for result in channel_results[1:]:
        current_axis = np.asarray(result.frequencies_hz, dtype=np.float64)
        if base_axis.shape != current_axis.shape or not np.allclose(base_axis, current_axis, equal_nan=True):
            raise ValueError("All channel results must share the same frequency axis")
    return base_axis


class ReferenceSpectrumAnalyzer:
    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    def build_channel_spectrum(
        self,
        signal_channel: np.ndarray,
        channel_index: int,
        params: ReferenceSpectrumParams,
    ) -> ReferenceSpectrumChannelResult:
        _validate_params(params)
        signal_arr = _as_1d_float_array(signal_channel)
        if signal_arr.size == 0:
            raise ValueError("signal_channel cannot be empty")
        if not np.all(np.isfinite(signal_arr)):
            raise ValueError("signal_channel contains non-finite values")

        signal_arr = _remove_dc_offset(signal_arr)
        freq_hz, power = _compute_welch_power(signal_arr, self.sample_rate, params)
        spectrum_db = _power_to_db(power)
        freq_hz, spectrum_db = _apply_optional_smoothing(freq_hz, spectrum_db, int(params.smoothing))
        return ReferenceSpectrumChannelResult(
            channel_index=int(channel_index),
            frequencies_hz=np.asarray(freq_hz, dtype=np.float64),
            spectrum_db=np.asarray(spectrum_db, dtype=np.float64),
        )

    def build_multi_channel_spectrum(
        self,
        signal_multi: np.ndarray,
        params: ReferenceSpectrumParams,
    ) -> list[ReferenceSpectrumChannelResult]:
        signal_arr = _normalize_multi_channel_audio(signal_multi)
        results: list[ReferenceSpectrumChannelResult] = []
        for channel_index in range(int(signal_arr.shape[1])):
            results.append(
                self.build_channel_spectrum(signal_arr[:, channel_index], channel_index=channel_index, params=params)
            )
        validate_same_frequency_axis(results)
        return results

    def compare_channel_to_reference(
        self,
        current_channel: np.ndarray,
        reference_channel: ReferenceSpectrumChannelResult,
        params: ReferenceSpectrumParams,
        start_freq_hz: float | None,
        end_freq_hz: float | None,
        lower_offset_db: float | None,
        upper_offset_db: float | None,
    ) -> ReferenceSpectrumCompareResult:
        current_result = self.build_channel_spectrum(
            current_channel,
            channel_index=int(reference_channel.channel_index),
            params=params,
        )

        reference_freq_hz = np.asarray(reference_channel.frequencies_hz, dtype=np.float64)
        reference_db = np.asarray(reference_channel.spectrum_db, dtype=np.float64)
        current_db = _align_curve_to_reference_axis(
            current_result.frequencies_hz,
            current_result.spectrum_db,
            reference_freq_hz,
        )
        threshold_enabled = lower_offset_db is not None and upper_offset_db is not None
        if threshold_enabled:
            lower_limit_db, upper_limit_db = _build_threshold_band(reference_db, lower_offset_db, upper_offset_db)
        elif lower_offset_db is None and upper_offset_db is None:
            lower_limit_db = np.full_like(reference_db, np.nan, dtype=np.float64)
            upper_limit_db = np.full_like(reference_db, np.nan, dtype=np.float64)
        else:
            raise ValueError("lower_offset_db and upper_offset_db must both be set or both be None")

        if start_freq_hz is None and end_freq_hz is None:
            band_mask = np.isfinite(reference_freq_hz) & (reference_freq_hz >= 0.0)
        elif start_freq_hz is not None and end_freq_hz is not None:
            band_mask = _build_band_mask(reference_freq_hz, start_freq_hz, end_freq_hz)
        else:
            raise ValueError("start_freq_hz and end_freq_hz must both be set or both be None")

        valid_mask = band_mask & np.isfinite(reference_freq_hz) & np.isfinite(reference_db) & np.isfinite(current_db)
        if threshold_enabled:
            valid_mask = valid_mask & np.isfinite(lower_limit_db) & np.isfinite(upper_limit_db)
        if not np.any(valid_mask):
            raise ValueError("No valid comparison points fall inside the analysis band")

        if threshold_enabled:
            is_above = valid_mask & (current_db > upper_limit_db)
            is_below = valid_mask & (current_db < lower_limit_db)
            out_of_range_mask = is_above | is_below

            over_upper_db = np.maximum(current_db - upper_limit_db, 0.0)
            under_lower_db = np.maximum(lower_limit_db - current_db, 0.0)
            max_over_upper_db = float(np.max(over_upper_db[valid_mask])) if np.any(valid_mask) else 0.0
            max_under_lower_db = float(np.max(under_lower_db[valid_mask])) if np.any(valid_mask) else 0.0
            out_of_range_point_count = int(np.count_nonzero(out_of_range_mask))
            valid_point_count = int(np.count_nonzero(valid_mask))
            out_of_range_ratio = float(out_of_range_point_count / valid_point_count)
            max_exceed_db = float(max(max_over_upper_db, max_under_lower_db))
            channel_ok: bool | None = out_of_range_point_count == 0
        else:
            out_of_range_mask = np.zeros_like(reference_freq_hz, dtype=bool)
            deviation_db = np.abs(current_db - reference_db)
            max_exceed_db = float(np.max(deviation_db[valid_mask])) if np.any(valid_mask) else 0.0
            max_over_upper_db = 0.0
            max_under_lower_db = 0.0
            out_of_range_point_count = 0
            out_of_range_ratio = 0.0
            channel_ok = None

        return ReferenceSpectrumCompareResult(
            channel_index=int(reference_channel.channel_index),
            frequencies_hz=reference_freq_hz,
            reference_db=reference_db,
            current_db=current_db,
            lower_limit_db=lower_limit_db,
            upper_limit_db=upper_limit_db,
            band_mask=band_mask,
            out_of_range_mask=out_of_range_mask,
            max_over_upper_db=max_over_upper_db,
            max_under_lower_db=max_under_lower_db,
            max_exceed_db=max_exceed_db,
            out_of_range_point_count=out_of_range_point_count,
            out_of_range_ratio=out_of_range_ratio,
            channel_ok=channel_ok,
            threshold_enabled=threshold_enabled,
        )
