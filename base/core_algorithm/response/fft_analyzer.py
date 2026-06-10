"""
Welch FFT spectrum analyzer.

This module provides a small algorithm layer for the FFT analysis UI. It keeps
FFT scaling, overlap validation, and frequency-domain weighting out of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, welch

from base.core_algorithm.response.frequency_band_analyzer import get_weighting_fn


REFERENCE_PRESSURE_PA = 20e-6
MIN_FFT_SIZE = 512
MAX_FFT_SIZE = 65535
MAX_OVERLAP_RATIO = 0.95


@dataclass
class FftAnalysisResult:
    frequencies_hz: np.ndarray
    spectrum_db: np.ndarray
    raw_spectrum_db: np.ndarray
    weighting: str
    n_fft: int
    window: str
    overlap_ratio: float


class FftAnalyzer:
    """Compute a calibrated, averaged FFT spectrum using Welch averaging."""

    def analyze(
        self,
        signal: np.ndarray,
        *,
        fs: int,
        n_fft: int = 4096,
        window: str = "hann",
        overlap_ratio: float = 0.5,
        weighting: str = "Z",
        v2pa_factor: float = 1.0,
    ) -> FftAnalysisResult:
        n_fft = self._validate_n_fft(n_fft)
        overlap_ratio = self._validate_overlap_ratio(overlap_ratio)
        fs = int(fs)
        if fs <= 0:
            raise ValueError("采样率必须为正数")

        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        if x.size < n_fft:
            raise ValueError("FFT 点数不能大于信号长度")

        weighting = self._normalize_weighting(weighting)
        noverlap = int(n_fft * overlap_ratio)
        if noverlap >= n_fft:
            raise ValueError("重叠率配置无效")

        x_pa = x * float(v2pa_factor or 1.0)
        window_values = get_window(window, n_fft)
        freqs, power = welch(
            x_pa,
            fs=fs,
            window=window_values,
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            detrend="constant",
            scaling="spectrum",
        )

        power = np.asarray(power, dtype=np.float64)
        raw_db = 10.0 * np.log10(np.maximum(power, 1e-30) / (REFERENCE_PRESSURE_PA ** 2))

        weighting_fn = get_weighting_fn(weighting)
        weighted_db = raw_db + weighting_fn(freqs)
        return FftAnalysisResult(
            frequencies_hz=np.asarray(freqs, dtype=np.float64),
            spectrum_db=np.asarray(weighted_db, dtype=np.float64),
            raw_spectrum_db=np.asarray(raw_db, dtype=np.float64),
            weighting=weighting,
            n_fft=n_fft,
            window=str(window),
            overlap_ratio=overlap_ratio,
        )

    @staticmethod
    def _validate_n_fft(n_fft: int) -> int:
        try:
            n_fft = int(n_fft)
        except Exception as exc:
            raise ValueError("FFT 点数必须为整数") from exc
        if not (MIN_FFT_SIZE <= n_fft <= MAX_FFT_SIZE):
            raise ValueError(f"FFT 点数必须在 {MIN_FFT_SIZE} ~ {MAX_FFT_SIZE} 范围内")
        return n_fft

    @staticmethod
    def _validate_overlap_ratio(overlap_ratio: float) -> float:
        try:
            overlap_ratio = float(overlap_ratio)
        except Exception as exc:
            raise ValueError("重叠率必须为数字") from exc
        if not (0.0 <= overlap_ratio <= MAX_OVERLAP_RATIO):
            raise ValueError("重叠率必须在 0% ~ 95% 范围内")
        return overlap_ratio

    @staticmethod
    def _normalize_weighting(weighting: str) -> str:
        weighting = str(weighting or "Z").upper()
        if weighting in ("NONE", "Z（NONE）"):
            return "Z"
        if weighting not in {"Z", "A", "C"}:
            raise ValueError("计权方式必须为 Z、A 或 C")
        return weighting
