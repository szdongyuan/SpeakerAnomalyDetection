"""Spectrogram calculation without UI or artifact-rendering dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_MAX_TIME_BINS = 2_000
_BATCH_FRAMES = 256


@dataclass(frozen=True)
class SpectrogramAnalysisResult:
    frequencies_hz: np.ndarray
    times_s: np.ndarray
    values_db: np.ndarray
    scale: str


class SpectrogramAnalyzer:
    """Calculate linear STFT or logarithmic CQT spectrogram data."""

    def analyze(
        self,
        signal,
        *,
        fs,
        n_fft=2048,
        hop_length=256,
        window="hann",
        scale="linear",
        max_time_bins=DEFAULT_MAX_TIME_BINS,
    ) -> SpectrogramAnalysisResult:
        sample_rate = int(fs)
        if sample_rate <= 0:
            raise ValueError("采样率必须为正整数")
        n_fft = int(n_fft)
        hop_length = int(hop_length)
        if n_fft <= 0 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("Spec 的 FFT 点数或帧移配置无效")
        if type(max_time_bins) is not int or max_time_bins < 1:
            raise ValueError("Spec 显示时间列上限必须是正整数")

        normalized_scale = str(scale or "linear").lower()
        if normalized_scale == "log":
            frequencies, times, values_db = self._calculate_log(
                signal,
                sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                max_time_bins=max_time_bins,
            )
        else:
            normalized_scale = "linear"
            frequencies, times, magnitude = _chunked_linear_spectrogram(
                signal,
                sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                window=str(window or "hann"),
                max_time_bins=max_time_bins,
            )
            values_db = _amplitude_to_db(
                magnitude,
                reference=20e-6,
                top_db=80.0,
            )

        return SpectrogramAnalysisResult(
            frequencies_hz=np.asarray(frequencies, dtype=np.float64),
            times_s=np.asarray(times, dtype=np.float64),
            values_db=np.asarray(values_db, dtype=np.float32),
            scale=normalized_scale,
        )

    @staticmethod
    def _calculate_log(signal, sample_rate, *, n_fft, hop_length, max_time_bins):
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
            max_time_bins=max_time_bins,
        )
        return (
            frequencies,
            times,
            librosa.amplitude_to_db(magnitude, ref=20e-6),
        )


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
    for first_frame in range(0, frame_count, _BATCH_FRAMES):
        frame_indices = np.arange(
            first_frame,
            min(first_frame + _BATCH_FRAMES, frame_count),
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


__all__ = [
    "DEFAULT_MAX_TIME_BINS",
    "SpectrogramAnalysisResult",
    "SpectrogramAnalyzer",
]
