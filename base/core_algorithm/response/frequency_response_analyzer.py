"""
Frequency response estimation algorithms.

This module centralizes FR computation so callers (UI / pipelines) can select an
algorithm appropriate for the stimulus type:
  - chirp sweep ("chirps"): sweep deconvolution (Wiener)
  - wideband stationary excitation: Welch/CSD H1 estimate (kept for future use)

It also supports computing an output-level curve (dB SPL vs frequency) for a
given drive level when a microphone calibration (V -> Pa) is available; that
functionality lives in `spl_frequency_analyzer.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np


class FrequencyResponseMethod(str, Enum):
    """Supported frequency response estimation methods."""

    AUTO = "auto"
    SWEEP_WIENER = "sweep_wiener"
    WELCH_H1 = "welch_h1"


@dataclass(frozen=True)
class FrequencyResponseResult:
    """Frequency response result (magnitude in dB)."""

    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _slice_to_band(freqs_hz: np.ndarray, values: np.ndarray, f_min: Optional[float], f_max: Optional[float]):
    if f_min is None and f_max is None:
        return freqs_hz, values
    f_min_value = float(np.min(freqs_hz) if f_min is None else f_min)
    f_max_value = float(np.max(freqs_hz) if f_max is None else f_max)
    if f_min_value > f_max_value:
        f_min_value, f_max_value = f_max_value, f_min_value
    mask = (freqs_hz >= f_min_value) & (freqs_hz <= f_max_value)
    return freqs_hz[mask], values[mask]


def _stimulus_band_from_metadata(stimulus_metadata: Optional[Dict]) -> tuple[Optional[float], Optional[float]]:
    if not stimulus_metadata:
        return None, None
    start_freq = stimulus_metadata.get("start_freq")
    stop_freq = stimulus_metadata.get("stop_freq")
    if start_freq is None or stop_freq is None:
        return None, None
    return float(min(start_freq, stop_freq)), float(max(start_freq, stop_freq))


class FrequencyResponseAnalyzer:
    """Compute frequency response for different stimulus types."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)

    def compute(
        self,
        reference_signal: np.ndarray,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Optional[Dict] = None,
        method: str | FrequencyResponseMethod = FrequencyResponseMethod.SWEEP_WIENER,
        **kwargs,
    ) -> FrequencyResponseResult:
        if isinstance(method, FrequencyResponseMethod):
            method_value = method
        else:
            method_value = FrequencyResponseMethod(str(method))

        if method_value == FrequencyResponseMethod.AUTO:
            # FR is currently standardized to sweep deconvolution for consistency.
            method_value = FrequencyResponseMethod.SWEEP_WIENER

        if method_value == FrequencyResponseMethod.WELCH_H1:
            return self._welch_h1(reference_signal, recorded_signal, stimulus_metadata=stimulus_metadata, **kwargs)
        if method_value == FrequencyResponseMethod.SWEEP_WIENER:
            return self._sweep_wiener(reference_signal, recorded_signal, stimulus_metadata=stimulus_metadata, **kwargs)

        raise ValueError(f"Unknown frequency response method: {method_value}")

    def _welch_h1(
        self,
        reference_signal: np.ndarray,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Optional[Dict],
        window_seconds: float = 1.0,
        eps_db: float = 1e-12,
    ) -> FrequencyResponseResult:
        """
        Welch/CSD H1 estimate (magnitude), suitable for wideband stationary excitation.

        Returns amplitude magnitude in dB: 20*log10(|H1|).
        """
        try:
            from scipy import signal as sp_signal
        except Exception as e:
            raise ImportError("scipy is required for Welch/CSD frequency response estimation") from e

        x = _as_1d_float_array(reference_signal)
        y = _as_1d_float_array(recorded_signal)
        n = int(min(x.size, y.size))
        if n <= 1:
            return FrequencyResponseResult(np.array([]), np.array([]))
        x = x[:n]
        y = y[:n]

        nperseg_target = int(round(float(window_seconds) * self.sample_rate))
        nperseg = max(16, min(nperseg_target, n))
        noverlap = nperseg // 2
        window = np.hanning(nperseg)

        freqs_hz, pxy = sp_signal.csd(
            x, y, window=window, nperseg=nperseg, nfft=nperseg, noverlap=noverlap, fs=self.sample_rate
        )
        _, pxx = sp_signal.welch(x, window=window, nperseg=nperseg, nfft=nperseg, noverlap=noverlap, fs=self.sample_rate)

        h1 = pxy / (pxx + np.finfo(np.float64).tiny)
        magnitude_db = 20.0 * np.log10(np.abs(h1) + float(eps_db))

        f_min, f_max = _stimulus_band_from_metadata(stimulus_metadata)
        freqs_hz, magnitude_db = _slice_to_band(freqs_hz, magnitude_db, f_min, f_max)
        return FrequencyResponseResult(freqs_hz, magnitude_db)

    def _sweep_wiener(
        self,
        reference_signal: np.ndarray,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Optional[Dict],
        n_fft: Optional[int] = None,
        regularization: float = 1e-8,
        apply_window: bool = False,
        eps_db: float = 1e-12,
    ) -> FrequencyResponseResult:
        """
        Deterministic sweep deconvolution in frequency domain (Wiener-style inversion).

        Suitable for chirps and other deterministic wideband excitations.
        """
        x = _as_1d_float_array(reference_signal)
        y = _as_1d_float_array(recorded_signal)
        n = int(min(x.size, y.size))
        if n <= 1:
            return FrequencyResponseResult(np.array([]), np.array([]))
        x = x[:n]
        y = y[:n]

        if apply_window:
            window = np.hanning(n)
            x = x * window
            y = y * window

        n_fft_value = int(_next_pow2(n) if n_fft is None else int(n_fft))
        n_fft_value = max(n_fft_value, n)

        X = np.fft.rfft(x, n=n_fft_value)
        Y = np.fft.rfft(y, n=n_fft_value)
        power = np.abs(X) ** 2
        eps = float(regularization) * float(np.max(power) if power.size else 1.0)
        H = Y * np.conj(X) / (power + eps)

        freqs_hz = np.fft.rfftfreq(n_fft_value, d=1.0 / self.sample_rate)
        magnitude_db = 20.0 * np.log10(np.abs(H) + float(eps_db))

        f_min, f_max = _stimulus_band_from_metadata(stimulus_metadata)
        freqs_hz, magnitude_db = _slice_to_band(freqs_hz, magnitude_db, f_min, f_max)
        return FrequencyResponseResult(freqs_hz, magnitude_db)
