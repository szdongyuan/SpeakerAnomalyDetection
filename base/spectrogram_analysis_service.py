"""Bounded headless spectrogram computation shared by UI and result workers."""
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import get_window

from base.pre_processing.audio_thd_frequency_response_analysis import (
    AudioThdFrequencyResponseAnalysis,
)


MAX_SPECTROGRAM_TIME_BINS = 1024
MAX_SPECTROGRAM_FREQUENCY_BINS = 512
_CQT_SELECTED_FRAMES_PER_CHUNK = 32


@dataclass(frozen=True)
class SpectrogramAnalysis:
    mode: str
    spectrogram_db: np.ndarray
    frequency_bins: np.ndarray
    time_s: np.ndarray
    retention: dict

    def as_dict(self):
        return {
            "mode": self.mode,
            "spectrogram_db": self.spectrogram_db,
            "frequency_bins": self.frequency_bins,
            "time_s": self.time_s,
            "retention": dict(self.retention),
        }


def _bounded_indices(length, limit):
    length = max(0, int(length))
    limit = max(1, int(limit))
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, limit, dtype=np.int64)


def spectrogram_retention_plan(
        *, frame_count, sample_rate, n_fft, hop_length, channel_count=1,
        source_frequency_bins=None):
    frame_count = max(0, int(frame_count))
    sample_rate = int(sample_rate)
    n_fft = max(2, int(n_fft))
    hop_length = max(1, int(hop_length))
    channel_count = max(1, int(channel_count))
    source_time_bins = 1 + frame_count // hop_length
    source_frequency_bins = int(
        source_frequency_bins
        if source_frequency_bins is not None else n_fft // 2 + 1)
    retained_time_bins = min(
        source_time_bins, MAX_SPECTROGRAM_TIME_BINS)
    retained_frequency_bins = min(
        source_frequency_bins, MAX_SPECTROGRAM_FREQUENCY_BINS)
    return {
        "source_samples": frame_count,
        "source_channels": channel_count,
        "source_numeric_bytes": frame_count * channel_count * 4,
        "sample_rate": sample_rate,
        "source_time_bins": source_time_bins,
        "source_frequency_bins": source_frequency_bins,
        "retained_time_bins": retained_time_bins,
        "retained_frequency_bins": retained_frequency_bins,
        "estimated_numeric_bytes": (
            retained_time_bins * retained_frequency_bins * 4),
        "frame_indices": tuple(_bounded_indices(
            source_time_bins, retained_time_bins).tolist()),
    }


def _cqt_parameters(sample_rate, n_fft):
    fmin = float(librosa.note_to_hz("C1"))
    fmax = float(librosa.note_to_hz("C9"))
    bins_per_octave = max(12, int(12 * np.log2(n_fft / 1024) + 24))
    n_bins = int(np.ceil(np.log2(fmax / fmin) * bins_per_octave))
    frequencies = librosa.cqt_frequencies(
        n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    if frequencies[-1] >= sample_rate / 2.0:
        raise ValueError(
            "CQT maximum frequency must be below the Nyquist frequency")
    return fmin, bins_per_octave, n_bins, frequencies


def _cqt_chunk_ranges(frame_count, frame_indices, hop_length, context_samples):
    ranges = []
    for first_output in range(
            0, len(frame_indices), _CQT_SELECTED_FRAMES_PER_CHUNK):
        last_output = min(
            first_output + _CQT_SELECTED_FRAMES_PER_CHUNK,
            len(frame_indices))
        selected = frame_indices[first_output:last_output]
        source_start = max(
            0, int(selected[0]) * hop_length - context_samples)
        source_start = source_start // hop_length * hop_length
        source_stop = min(
            int(frame_count), int(selected[-1]) * hop_length
            + context_samples + hop_length)
        ranges.append((first_output, last_output, source_start, source_stop))
    return tuple(ranges)


def cqt_resource_plan(
        *, frame_count, sample_rate, n_fft, hop_length, channel_count=1):
    """Plan long CQT chunks without allocating the represented recording."""
    _fmin, _bpo, n_bins, frequencies = _cqt_parameters(
        int(sample_rate), int(n_fft))
    plan = spectrogram_retention_plan(
        frame_count=frame_count, sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, channel_count=channel_count,
        source_frequency_bins=n_bins)
    frame_indices = np.asarray(plan["frame_indices"], dtype=np.int64)
    lengths, _cutoff = librosa.filters.wavelet_lengths(
        freqs=frequencies, sr=sample_rate, window="hann")
    context_samples = int(np.ceil(np.max(lengths))) * 2
    ranges = _cqt_chunk_ranges(
        frame_count, frame_indices, int(hop_length), context_samples)
    max_chunk_samples = max(
        (source_stop - source_start
         for _first, _last, source_start, source_stop in ranges),
        default=0)
    max_chunk_time_bins = 1 + max_chunk_samples // int(hop_length)
    # float32 input + complex64 CQT chunk + retained float32 dB matrix.
    peak_bytes = (
        max_chunk_samples * 4
        + max_chunk_time_bins * n_bins * 8
        + plan["estimated_numeric_bytes"])
    return {
        **{key: value for key, value in plan.items()
           if key != "frame_indices"},
        "context_samples": context_samples,
        "chunk_count": len(ranges),
        "max_chunk_samples": max_chunk_samples,
        "max_chunk_time_bins": max_chunk_time_bins,
        "estimated_peak_numeric_bytes": peak_bytes,
        "chunk_ranges": ranges,
        "frame_indices": plan["frame_indices"],
    }


def _bounded_cqt(signal, sample_rate, n_fft, hop_length, plan):
    fmin, bins_per_octave, n_bins, frequencies = _cqt_parameters(
        sample_rate, n_fft)
    frame_indices = np.asarray(plan["frame_indices"], dtype=np.int64)
    if plan["source_time_bins"] <= MAX_SPECTROGRAM_TIME_BINS:
        cqt, computed_frequency, computed_time = (
            AudioThdFrequencyResponseAnalysis.compute_cqt(
                y=signal, sr=sample_rate, hop_length=hop_length,
                n_fft=n_fft, fmin=fmin,
                bins_per_octave=bins_per_octave, n_bins=n_bins))
        frequency_index = _bounded_indices(
            len(computed_frequency), plan["retained_frequency_bins"])
        return (np.abs(cqt)[frequency_index][:, frame_indices],
                np.asarray(computed_frequency)[frequency_index],
                np.asarray(computed_time)[frame_indices])

    frequency_index = _bounded_indices(
        len(frequencies), plan["retained_frequency_bins"])
    retained = np.empty(
        (len(frequency_index), len(frame_indices)), dtype=np.float32)
    resource_plan = cqt_resource_plan(
        frame_count=len(signal), sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length)
    for first_output, last_output, source_start, source_stop in (
            resource_plan["chunk_ranges"]):
        output_slice = slice(first_output, last_output)
        selected = frame_indices[output_slice]
        segment = signal[source_start:source_stop]
        chunk, _chunk_frequency, _chunk_time = (
            AudioThdFrequencyResponseAnalysis.compute_cqt(
                y=segment, sr=sample_rate, hop_length=hop_length,
                n_fft=n_fft, fmin=fmin,
                bins_per_octave=bins_per_octave, n_bins=n_bins))
        local_indices = selected - source_start // hop_length
        local_indices = np.clip(local_indices, 0, chunk.shape[1] - 1)
        retained[:, output_slice] = np.abs(
            chunk[frequency_index][:, local_indices])
    return (retained, frequencies[frequency_index],
            frame_indices.astype(np.float64) * hop_length / sample_rate)


def _bounded_stft(signal, sample_rate, n_fft, hop_length, window_name, plan):
    frame_indices = np.asarray(plan["frame_indices"], dtype=np.int64)
    frequency_index = _bounded_indices(
        n_fft // 2 + 1, plan["retained_frequency_bins"])
    if plan["source_time_bins"] <= MAX_SPECTROGRAM_TIME_BINS:
        magnitude = np.abs(librosa.stft(
            y=signal, n_fft=n_fft, hop_length=hop_length,
            window=window_name))
        magnitude = magnitude[frequency_index][:, frame_indices]
    else:
        window = get_window(window_name, n_fft)
        magnitude = np.empty(
            (len(frequency_index), len(frame_indices)), dtype=np.float32)
        half = n_fft // 2
        for output_column, frame_index in enumerate(frame_indices):
            center = int(frame_index) * hop_length
            start, stop = center - half, center - half + n_fft
            frame = np.zeros(n_fft, dtype=np.float64)
            source_start, source_stop = max(0, start), min(len(signal), stop)
            if source_stop > source_start:
                frame[source_start - start:source_stop - start] = signal[
                    source_start:source_stop]
            magnitude[:, output_column] = np.abs(
                np.fft.rfft(frame * window))[frequency_index]
    frequencies = librosa.fft_frequencies(
        sr=sample_rate, n_fft=n_fft)[frequency_index]
    times = frame_indices.astype(np.float64) * hop_length / sample_rate
    return magnitude, frequencies, times


def compute_spectrogram(
        signal, sample_rate, config, *, v2pa_factor=1.0, channel_count=1):
    sample_rate = int(sample_rate)
    config = dict(config or {})
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    factor = 1.0 if v2pa_factor is None else float(v2pa_factor)
    if (sample_rate <= 0 or values.size == 0 or not np.isfinite(factor)
            or factor <= 0.0):
        raise ValueError("invalid spectrogram signal, sample rate, or calibration")
    values = values * factor
    n_fft = int(config.get("n_fft", 2048))
    hop_length = int(config.get("hop_length", 256))
    mode = str(config.get("freq_scale_type", "linear") or "linear").lower()
    if mode == "log":
        _fmin, _bpo, n_bins, _frequencies = _cqt_parameters(
            sample_rate, n_fft)
        plan = spectrogram_retention_plan(
            frame_count=len(values), sample_rate=sample_rate,
            n_fft=n_fft, hop_length=hop_length,
            channel_count=channel_count, source_frequency_bins=n_bins)
        magnitude, frequencies, times = _bounded_cqt(
            values, sample_rate, n_fft, hop_length, plan)
    else:
        mode = "linear"
        plan = spectrogram_retention_plan(
            frame_count=len(values), sample_rate=sample_rate,
            n_fft=n_fft, hop_length=hop_length,
            channel_count=channel_count)
        magnitude, frequencies, times = _bounded_stft(
            values, sample_rate, n_fft, hop_length,
            config.get("window_func", "hann"), plan)
    db = librosa.amplitude_to_db(magnitude, ref=20e-6)
    retention = {key: value for key, value in plan.items()
                 if key != "frame_indices"}
    return SpectrogramAnalysis(
        mode=mode, spectrogram_db=np.asarray(db),
        frequency_bins=np.asarray(frequencies), time_s=np.asarray(times),
        retention=retention)
