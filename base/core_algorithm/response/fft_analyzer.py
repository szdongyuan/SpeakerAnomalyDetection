"""Welch-based FFT spectrum analysis."""

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
        fs: int,
        n_fft: int = 4096,
        window: str = "hann",
        overlap_ratio: float = 0.5,
        weighting: str = "Z",
        v2pa_factor: float = 1.0,
    ) -> FftAnalysisResult:
        n_fft = self._validate_n_fft(n_fft)
        overlap_ratio = self._validate_overlap_ratio(overlap_ratio)
        try:
            fs = int(fs)
        except (TypeError, ValueError) as exc:
            raise ValueError("采样率必须为正整数") from exc
        if fs <= 0:
            raise ValueError("采样率必须为正整数")

        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        if values.size < n_fft:
            raise ValueError("FFT 点数不能大于信号长度")

        weighting = self._normalize_weighting(weighting)
        noverlap = int(n_fft * overlap_ratio)
        if noverlap >= n_fft:
            raise ValueError("重叠率配置无效")

        calibrated_values = values * float(v2pa_factor or 1.0)
        window_values = get_window(window, n_fft)
        frequencies, power = welch(
            calibrated_values,
            fs=fs,
            window=window_values,
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            detrend="constant",
            scaling="spectrum",
        )

        power = np.asarray(power, dtype=np.float64)
        raw_spectrum_db = 10.0 * np.log10(
            np.maximum(power, 1e-30) / (REFERENCE_PRESSURE_PA**2)
        )
        weighted_spectrum_db = raw_spectrum_db + get_weighting_fn(weighting)(frequencies)
        return FftAnalysisResult(
            frequencies_hz=np.asarray(frequencies, dtype=np.float64),
            spectrum_db=np.asarray(weighted_spectrum_db, dtype=np.float64),
            raw_spectrum_db=np.asarray(raw_spectrum_db, dtype=np.float64),
            weighting=weighting,
            n_fft=n_fft,
            window=str(window),
            overlap_ratio=overlap_ratio,
        )

    @staticmethod
    def _validate_n_fft(n_fft: int) -> int:
        try:
            n_fft = int(n_fft)
        except (TypeError, ValueError) as exc:
            raise ValueError("FFT 点数必须为整数") from exc
        if not (MIN_FFT_SIZE <= n_fft <= MAX_FFT_SIZE):
            raise ValueError(f"FFT 点数必须在 {MIN_FFT_SIZE} ~ {MAX_FFT_SIZE} 范围内")
        return n_fft

    @staticmethod
    def _validate_overlap_ratio(overlap_ratio: float) -> float:
        try:
            overlap_ratio = float(overlap_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("重叠率必须为数字") from exc
        if not (0.0 <= overlap_ratio <= MAX_OVERLAP_RATIO):
            max_overlap_percent = MAX_OVERLAP_RATIO * 100.0
            raise ValueError(f"重叠率必须在 0% ~ {max_overlap_percent:g}% 范围内")
        return overlap_ratio

    @staticmethod
    def _normalize_weighting(weighting: str) -> str:
        weighting = str(weighting or "Z").upper()
        if weighting in ("NONE", "Z（NONE）"):
            return "Z"
        if weighting not in {"Z", "A", "C"}:
            raise ValueError("计权方式必须为 Z、A 或 C")
        return weighting
