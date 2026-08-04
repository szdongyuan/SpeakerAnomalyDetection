"""Noise-reduction helpers for sound-quality analysis.

Provides conservative power spectral subtraction for removing stationary
background noise from a recorded signal before loudness computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

DEFAULT_SPECTRAL_SUBTRACTION_N_FFT = 4096
DEFAULT_SPECTRAL_SUBTRACTION_HOP_SIZE = 1024
DEFAULT_SPECTRAL_SUBTRACTION_ALPHA = 1.0
DEFAULT_SPECTRAL_SUBTRACTION_FLOOR = 0.02
DEFAULT_SPECTRAL_SUBTRACTION_MIN_GAIN_DB = -20.0
DEFAULT_SPECTRAL_SUBTRACTION_FREQ_SMOOTHING_BINS = 3
DEFAULT_SPECTRAL_SUBTRACTION_GAIN_SMOOTHING = 0.6


@dataclass(frozen=True)
class SpectralSubtractionResult:
    signal: np.ndarray
    metadata: Dict[str, object]


def spectral_subtract_audio(
    signal: np.ndarray,
    noise_reference: np.ndarray,
    *,
    n_fft: int = DEFAULT_SPECTRAL_SUBTRACTION_N_FFT,
    hop_size: int = DEFAULT_SPECTRAL_SUBTRACTION_HOP_SIZE,
    alpha: float = DEFAULT_SPECTRAL_SUBTRACTION_ALPHA,
    spectral_floor: float = DEFAULT_SPECTRAL_SUBTRACTION_FLOOR,
    min_gain_db: float = DEFAULT_SPECTRAL_SUBTRACTION_MIN_GAIN_DB,
    frequency_smoothing_bins: int = DEFAULT_SPECTRAL_SUBTRACTION_FREQ_SMOOTHING_BINS,
    gain_time_smoothing: float = DEFAULT_SPECTRAL_SUBTRACTION_GAIN_SMOOTHING,
) -> SpectralSubtractionResult:
    """Apply conservative power spectral subtraction and preserve input length."""

    x = _as_1d_float(signal, "signal")
    noise = _as_1d_float(noise_reference, "noise_reference")
    n_fft = _sanitize_n_fft(n_fft)
    hop_size = _sanitize_hop_size(hop_size, n_fft)
    alpha = max(float(alpha), 0.0)
    spectral_floor = max(float(spectral_floor), 0.0)
    min_gain = _min_gain_from_db(min_gain_db)
    smoothing = min(max(float(gain_time_smoothing), 0.0), 0.99)
    freq_smoothing = max(int(frequency_smoothing_bins), 1)

    window = np.hanning(n_fft).astype(np.float64)
    if not np.any(window > 0.0):
        window = np.ones(n_fft, dtype=np.float64)

    noise_power = _estimate_noise_power(noise, window, n_fft, hop_size)
    if freq_smoothing > 1:
        noise_power = _smooth_frequency(noise_power, freq_smoothing)
    noise_power = np.maximum(noise_power, np.finfo(np.float64).tiny)

    padded, pad = _pad_for_overlap_add(x, n_fft)
    starts = _frame_starts(padded.size, n_fft, hop_size)
    output = np.zeros_like(padded, dtype=np.float64)
    norm = np.zeros_like(padded, dtype=np.float64)
    previous_gain = None
    gain_sum = 0.0
    gain_count = 0
    floor_count = 0
    limited_count = 0
    bin_count = 0

    for start in starts:
        frame = padded[start : start + n_fft]
        spectrum = np.fft.rfft(frame * window)
        magnitude = np.abs(spectrum)
        power = magnitude * magnitude

        clean_power_raw = power - alpha * noise_power
        floor_power = spectral_floor * noise_power
        clean_power = np.maximum(clean_power_raw, floor_power)
        floor_count += int(np.count_nonzero(clean_power_raw < floor_power))

        raw_gain = np.sqrt(clean_power / np.maximum(power, np.finfo(np.float64).tiny))
        limited_count += int(np.count_nonzero(raw_gain < min_gain))
        gain = np.maximum(raw_gain, min_gain)
        gain = np.minimum(gain, 1.0)
        if previous_gain is not None and smoothing > 0.0:
            gain = smoothing * previous_gain + (1.0 - smoothing) * gain
        previous_gain = gain

        clean_spectrum = spectrum * gain
        clean_frame = np.fft.irfft(clean_spectrum, n=n_fft)
        output[start : start + n_fft] += clean_frame * window
        norm[start : start + n_fft] += window * window

        gain_sum += float(np.sum(gain))
        gain_count += int(gain.size)
        bin_count += int(gain.size)

    valid = norm > np.finfo(np.float64).eps
    reconstructed = np.array(padded, dtype=np.float64, copy=True)
    reconstructed[valid] = output[valid] / norm[valid]
    if pad > 0:
        reconstructed = reconstructed[pad : pad + x.size]
    else:
        reconstructed = reconstructed[: x.size]

    metadata = {
        "algorithm": "spectral_subtraction_audio",
        "n_fft": int(n_fft),
        "hop_size": int(hop_size),
        "window": "hann",
        "alpha": float(alpha),
        "spectral_floor": float(spectral_floor),
        "min_gain_db": float(min_gain_db),
        "frequency_smoothing_bins": int(freq_smoothing),
        "gain_time_smoothing": float(smoothing),
        "input_sample_count": int(x.size),
        "noise_sample_count": int(noise.size),
        "output_sample_count": int(reconstructed.size),
        "mean_gain_db": float(
            20.0 * np.log10(max(gain_sum / max(gain_count, 1), np.finfo(np.float64).tiny))
        ),
        "floor_limited_bin_ratio": float(floor_count / max(bin_count, 1)),
        "max_attenuation_limited_bin_ratio": float(limited_count / max(bin_count, 1)),
    }
    return SpectralSubtractionResult(
        signal=reconstructed.astype(np.float64, copy=False),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _sanitize_n_fft(value: int) -> int:
    n_fft = int(value)
    if n_fft < 32:
        raise ValueError("spectral subtraction n_fft must be at least 32")
    return n_fft


def _sanitize_hop_size(value: int, n_fft: int) -> int:
    hop = int(value)
    if hop <= 0:
        raise ValueError("spectral subtraction hop_size must be positive")
    if hop >= n_fft:
        hop = max(1, n_fft // 4)
    return hop


def _min_gain_from_db(min_gain_db: float) -> float:
    gain = 10.0 ** (float(min_gain_db) / 20.0)
    return min(max(gain, 0.0), 1.0)


def _frame_starts(length: int, n_fft: int, hop_size: int) -> np.ndarray:
    if length <= n_fft:
        return np.asarray([0], dtype=np.int64)
    starts = list(range(0, length - n_fft + 1, hop_size))
    last = length - n_fft
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def _pad_to_frame_length(values: np.ndarray, n_fft: int) -> np.ndarray:
    if values.size >= n_fft:
        return values
    return np.pad(values, (0, n_fft - values.size), mode="constant")


def _pad_for_overlap_add(values: np.ndarray, n_fft: int) -> tuple[np.ndarray, int]:
    pad = n_fft // 2
    if pad <= 0:
        return values.astype(np.float64, copy=False), 0
    return np.pad(values, (pad, pad), mode="constant").astype(np.float64, copy=False), pad


def _estimate_noise_power(
    noise: np.ndarray,
    window: np.ndarray,
    n_fft: int,
    hop_size: int,
) -> np.ndarray:
    prepared = _pad_to_frame_length(noise, n_fft)
    starts = _frame_starts(prepared.size, n_fft, hop_size)
    power_accum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for start in starts:
        frame = prepared[start : start + n_fft]
        spectrum = np.fft.rfft(frame * window)
        power_accum += np.abs(spectrum) ** 2
    return power_accum / max(int(starts.size), 1)


def _smooth_frequency(values: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    if width <= 1 or values.size <= 1:
        return values
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(padded, kernel, mode="valid")
