"""Welch-based FFT spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, welch

from base.core_algorithm.response.frequency_weighting import get_weighting_fn
from consts.acoustic_analysis.common_consts import REFERENCE_PRESSURE_PA
from consts.acoustic_analysis.specific_consts.fft_consts import (
    MAX_FFT_SIZE,
    MAX_OVERLAP_RATIO,
    MIN_FFT_SIZE,
)


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


def load_fft_baseline(
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
    audio_loader=None,
):
    """Load and calculate an FFT baseline on the target frequency axis."""
    path = str(config.get("baseline_file_path", "") or "").strip()
    if not path:
        return None
    if audio_loader is None:
        import librosa

        audio_loader = librosa.load
    baseline_signal, _ = audio_loader(path, sr=sample_rate, mono=True)
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
        baseline_db = smooth_fft_baseline(frequency, baseline_db)
    return baseline_db


def smooth_fft_baseline(frequency, baseline_db):
    """Smooth an FFT baseline using one-third-octave neighborhoods."""
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
