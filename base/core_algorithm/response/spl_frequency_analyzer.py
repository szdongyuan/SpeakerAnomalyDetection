"""
SPL (Sound Pressure Level) vs Frequency estimation.

This module computes an output-level curve (dB SPL vs frequency) for the current
drive level. Unlike a transfer-function (Y/X), it does not normalize by the
stimulus. It relies on microphone calibration (V -> Pa) via `v2pa_factor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np

from base.stimulus_signal.frequency_stepped import resolve_frequency_stepped_schedule


class SplFrequencyMethod(str, Enum):
    """Supported SPL (dB SPL) vs frequency computation methods."""

    OUTPUT_SPL = "output_spl"
    OUTPUT_SPL_STEP = "output_spl_step"
    OUTPUT_SPL_CHIRP = "output_spl_chirp"


@dataclass(frozen=True)
class SplFrequencyResult:
    frequencies_hz: np.ndarray
    spl_db: np.ndarray


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _spl_db_from_pa_rms(pa_rms: np.ndarray, reference_pressure_pa: float) -> np.ndarray:
    pa_rms = np.asarray(pa_rms, dtype=np.float64)
    p_ref = float(reference_pressure_pa)
    if p_ref <= 0.0 or not np.isfinite(p_ref):
        raise ValueError(f"reference_pressure_pa must be finite positive, got {reference_pressure_pa}")
    pa_safe = np.maximum(pa_rms, np.finfo(np.float64).tiny)
    return 20.0 * np.log10(pa_safe / p_ref)


def _instantaneous_frequency(time_s: np.ndarray, stimulus_metadata: Dict, single_rep_duration_s: float) -> np.ndarray:
    stimulus_type = str(stimulus_metadata.get("stimulus_type", "linear"))
    start_freq = float(stimulus_metadata["start_freq"])
    stop_freq = float(stimulus_metadata["stop_freq"])

    if stimulus_type == "linear":
        return start_freq + (stop_freq - start_freq) * time_s / single_rep_duration_s
    if stimulus_type == "log":
        return start_freq * np.exp(np.log(stop_freq / start_freq) * time_s / single_rep_duration_s)
    if stimulus_type == "mirror_linear":
        half = single_rep_duration_s / 2.0
        return np.where(
            time_s < half,
            stop_freq - (stop_freq - start_freq) * time_s / half,
            start_freq + (stop_freq - start_freq) * (time_s - half) / half,
        )
    if stimulus_type == "mirror_log":
        half = single_rep_duration_s / 2.0
        return np.where(
            time_s < half,
            stop_freq * np.exp(-np.log(stop_freq / start_freq) * time_s / half),
            start_freq * np.exp(np.log(stop_freq / start_freq) * (time_s - half) / half),
        )
    raise ValueError(f"Unsupported stimulus_type: {stimulus_type}")


def _cached_segment_bounds(segment_payload):
    try:
        start = int(segment_payload["start_sample"])
        end = int(segment_payload["end_sample"])
        sample_count = int(segment_payload["sample_count"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("frequency_stepped segment metadata is invalid") from None
    if start < 0 or end < start or sample_count != end - start:
        raise ValueError("frequency_stepped segment metadata is invalid")
    return start, end


def _validate_cached_frequency_stepped_segments(stimulus_metadata: Dict, signal_length: int, sample_rate: int) -> None:
    segments = stimulus_metadata.get("segments")
    if segments is None:
        return
    schedule_sample_rate = stimulus_metadata.get("schedule_sample_rate")
    if schedule_sample_rate is not None:
        try:
            if int(schedule_sample_rate) != int(sample_rate):
                return
        except (TypeError, ValueError):
            pass
    if not isinstance(segments, (list, tuple)):
        raise ValueError("frequency_stepped segments must be a list")

    bounds = [_cached_segment_bounds(segment) for segment in segments]
    previous_end = -1
    for index, (start, end) in enumerate(bounds):
        if start < previous_end:
            raise ValueError("frequency_stepped segments overlap")
        if end > signal_length:
            has_later_recorded_segment = any(later_start < signal_length for later_start, _ in bounds[index + 1:])
            if has_later_recorded_segment:
                raise ValueError("frequency_stepped segment is non-trailing out of range")
            return
        previous_end = end


def _frequency_stepped_analysis_segments(recorded_signal: np.ndarray, stimulus_metadata: Dict, sample_rate: int):
    recorded = _as_1d_float_array(recorded_signal)
    _validate_cached_frequency_stepped_segments(stimulus_metadata, int(recorded.size), sample_rate)
    schedule = resolve_frequency_stepped_schedule(stimulus_metadata, sample_rate)

    segments = []
    all_segments = list(schedule.segments)
    for index, segment in enumerate(all_segments):
        start = int(segment.start_sample)
        end = int(segment.end_sample)
        if start < 0 or end < start:
            raise ValueError("frequency_stepped resolved segment metadata is invalid")
        if end > recorded.size:
            has_later_recorded_segment = any(int(later.start_sample) < recorded.size for later in all_segments[index + 1:])
            if has_later_recorded_segment:
                raise ValueError("frequency_stepped segment is non-trailing out of range")
            break
        if start >= recorded.size:
            break
        segment_signal = recorded[start:end]
        if segment_signal.size != end - start:
            raise ValueError("frequency_stepped segment slicing failed")
        segments.append((segment, segment_signal))
    return schedule, segments


class SplFrequencyAnalyzer:
    """Compute output SPL (dB SPL) vs frequency for step/chirp stimuli."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)

    def compute(
        self,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Dict,
        method: str | SplFrequencyMethod = SplFrequencyMethod.OUTPUT_SPL,
        v2pa_factor: Optional[float] = None,
        reference_pressure_pa: float = 20.0e-6,
        splf_calc_mode: str = "fundamental",
        frame_size: int = 2048,
        hop_length: int = 1024,
        eps: float = 1e-12,
    ) -> SplFrequencyResult:
        if isinstance(method, SplFrequencyMethod):
            method_value = method
        else:
            method_value = SplFrequencyMethod(str(method))

        calc_mode = str(splf_calc_mode or "fundamental").strip().lower()
        if calc_mode not in {"fundamental", "total"}:
            raise ValueError(f"Unsupported splf_calc_mode: {splf_calc_mode!r} (expected 'fundamental' or 'total')")

        if method_value == SplFrequencyMethod.OUTPUT_SPL:
            stimulus_method = stimulus_metadata.get("stimulus_method")
            if stimulus_method == "steps":
                return self._step_output_spl(
                    recorded_signal,
                    stimulus_metadata=stimulus_metadata,
                    v2pa_factor=v2pa_factor,
                    reference_pressure_pa=reference_pressure_pa,
                    splf_calc_mode=calc_mode,
                    eps=eps,
                )
            if stimulus_method == "chirps":
                return self._chirp_output_spl(
                    recorded_signal,
                    stimulus_metadata=stimulus_metadata,
                    v2pa_factor=v2pa_factor,
                    reference_pressure_pa=reference_pressure_pa,
                    splf_calc_mode=calc_mode,
                    frame_size=frame_size,
                    hop_length=hop_length,
                    eps=eps,
                )
            if stimulus_method == "frequency_stepped":
                return self._frequency_stepped_output_spl(
                    recorded_signal,
                    stimulus_metadata=stimulus_metadata,
                    v2pa_factor=v2pa_factor,
                    reference_pressure_pa=reference_pressure_pa,
                    splf_calc_mode=calc_mode,
                    eps=eps,
                )
            raise ValueError(f"Unsupported stimulus_method: {stimulus_method!r}")

        stimulus_method = stimulus_metadata.get("stimulus_method")
        if method_value == SplFrequencyMethod.OUTPUT_SPL_STEP:
            if stimulus_method is not None and stimulus_method != "steps":
                raise ValueError("output_spl_step requires stimulus_method == 'steps'")
            return self._step_output_spl(
                recorded_signal,
                stimulus_metadata=stimulus_metadata,
                v2pa_factor=v2pa_factor,
                reference_pressure_pa=reference_pressure_pa,
                splf_calc_mode=calc_mode,
                eps=eps,
            )
        if method_value == SplFrequencyMethod.OUTPUT_SPL_CHIRP:
            if stimulus_method is not None and stimulus_method != "chirps":
                raise ValueError("output_spl_chirp requires stimulus_method == 'chirps'")
            return self._chirp_output_spl(
                recorded_signal,
                stimulus_metadata=stimulus_metadata,
                v2pa_factor=v2pa_factor,
                reference_pressure_pa=reference_pressure_pa,
                splf_calc_mode=calc_mode,
                frame_size=frame_size,
                hop_length=hop_length,
                eps=eps,
            )
        raise ValueError(f"Unsupported method: {method_value!r}")

    def _frequency_stepped_output_spl(
        self,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Dict,
        v2pa_factor: Optional[float],
        reference_pressure_pa: float,
        splf_calc_mode: str,
        eps: float,
    ) -> SplFrequencyResult:
        schedule, segment_pairs = _frequency_stepped_analysis_segments(
            recorded_signal,
            stimulus_metadata,
            self.sample_rate,
        )
        if not segment_pairs:
            return SplFrequencyResult(np.array([]), np.array([]))

        v2pa_value = 1.0 if not v2pa_factor else float(v2pa_factor)
        rms_sq_by_step: Dict[int, float] = {}
        count_by_step: Dict[int, int] = {}
        frequency_by_step: Dict[int, float] = {}

        for segment, segment_signal_v in segment_pairs:
            segment_signal_pa = np.asarray(segment_signal_v, dtype=np.float64) * v2pa_value
            if segment_signal_pa.size <= 1:
                continue

            if splf_calc_mode == "fundamental":
                n_idx = np.arange(segment_signal_pa.size, dtype=np.float64)
                basis = np.exp(-1j * (2.0 * np.pi * float(segment.frequency_hz) / self.sample_rate) * n_idx)
                y_k = np.dot(segment_signal_pa, basis)
                peak_amp_pa = 2.0 * np.abs(y_k) / float(segment_signal_pa.size)
                rms_pa = float(peak_amp_pa) / np.sqrt(2.0)
            else:
                y0 = segment_signal_pa - float(np.mean(segment_signal_pa))
                rms_pa = float(np.sqrt(np.mean(y0 * y0)))

            step_index = int(segment.step_index)
            rms_sq_by_step[step_index] = rms_sq_by_step.get(step_index, 0.0) + rms_pa * rms_pa
            count_by_step[step_index] = count_by_step.get(step_index, 0) + 1
            frequency_by_step.setdefault(step_index, float(segment.frequency_hz))

        if not count_by_step:
            return SplFrequencyResult(np.array([]), np.array([]))

        step_indices = sorted(count_by_step)
        freqs = np.asarray([frequency_by_step[index] for index in step_indices], dtype=np.float64)
        rms_pa = np.asarray(
            [np.sqrt(rms_sq_by_step[index] / float(count_by_step[index])) for index in step_indices],
            dtype=np.float64,
        )
        sort_idx = np.argsort(freqs, kind="stable")
        freqs = freqs[sort_idx]
        rms_pa = rms_pa[sort_idx]
        spl_db = _spl_db_from_pa_rms(rms_pa + float(eps), reference_pressure_pa)
        return SplFrequencyResult(freqs, spl_db)

    def _step_output_spl(
        self,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Dict,
        v2pa_factor: Optional[float],
        reference_pressure_pa: float,
        splf_calc_mode: str,
        eps: float,
    ) -> SplFrequencyResult:
        y_full_v = _as_1d_float_array(recorded_signal)
        n_total = int(y_full_v.size)
        if n_total <= 1:
            return SplFrequencyResult(np.array([]), np.array([]))

        v2pa_value = 1.0 if not v2pa_factor else float(v2pa_factor)
        y_full_pa = y_full_v * v2pa_value

        repeat_times = int(stimulus_metadata.get("repeat_times", 1) or 1)
        repeat_times = max(1, repeat_times)
        num_steps = int(stimulus_metadata.get("num_steps", 0) or 0)
        if num_steps <= 0:
            raise ValueError("SPL-Frequency(step) requires a positive num_steps")

        start_freq = float(stimulus_metadata["start_freq"])
        stop_freq = float(stimulus_metadata["stop_freq"])
        stimulus_type = str(stimulus_metadata.get("stimulus_type", "linear"))
        if stimulus_type == "linear":
            step_freqs = np.linspace(start_freq, stop_freq, num_steps)
        elif stimulus_type == "log":
            step_freqs = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_steps)
        else:
            raise ValueError(f"Unsupported step stimulus_type: {stimulus_type}")

        samples_per_rep_from_len = n_total // repeat_times
        total_time = stimulus_metadata.get("total_time")
        if total_time is not None:
            expected_per_rep = int(round(float(total_time) / float(repeat_times) * float(self.sample_rate)))
            samples_per_rep = expected_per_rep if samples_per_rep_from_len >= expected_per_rep else samples_per_rep_from_len
        else:
            samples_per_rep = samples_per_rep_from_len

        step_samples = samples_per_rep // num_steps
        if step_samples <= 4:
            return SplFrequencyResult(np.array([]), np.array([]))

        window = np.hanning(step_samples).astype(np.float64, copy=False)
        w_sum = float(np.sum(window))
        if w_sum <= 0.0:
            raise ValueError("Invalid window normalization (sum(window) <= 0)")
        w2_sum = float(np.sum(window * window))
        if w2_sum <= 0.0:
            raise ValueError("Invalid window normalization (sum(window^2) <= 0)")
        n_idx = np.arange(step_samples, dtype=np.float64)

        rms_sq_accum = np.zeros(num_steps, dtype=np.float64)
        for rep in range(repeat_times):
            rep_start = rep * samples_per_rep_from_len
            for step_idx, freq_hz in enumerate(step_freqs):
                seg_start = rep_start + step_idx * step_samples
                seg_end = seg_start + step_samples
                y_seg = y_full_pa[seg_start:seg_end]
                if y_seg.size != step_samples:
                    continue

                if splf_calc_mode == "fundamental":
                    basis = np.exp(-1j * (2.0 * np.pi * float(freq_hz) / self.sample_rate) * n_idx)
                    y_k = np.dot(y_seg * window, basis)

                    peak_amp_pa = 2.0 * np.abs(y_k) / w_sum
                    rms_pa = float(peak_amp_pa) / np.sqrt(2.0)
                else:
                    y0 = y_seg - float(np.mean(y_seg))
                    rms_pa = float(np.sqrt(np.sum((y0 * window) ** 2) / w2_sum))
                rms_sq_accum[step_idx] += rms_pa * rms_pa

        rms_pa_mean = np.sqrt(rms_sq_accum / float(repeat_times))
        spl_db = _spl_db_from_pa_rms(rms_pa_mean + float(eps), reference_pressure_pa)

        sort_idx = np.argsort(step_freqs)
        return SplFrequencyResult(step_freqs[sort_idx], spl_db[sort_idx])

    def _chirp_output_spl(
        self,
        recorded_signal: np.ndarray,
        *,
        stimulus_metadata: Dict,
        v2pa_factor: Optional[float],
        reference_pressure_pa: float,
        splf_calc_mode: str,
        frame_size: int,
        hop_length: int,
        eps: float,
    ) -> SplFrequencyResult:
        y_full_v = _as_1d_float_array(recorded_signal)
        n_total = int(y_full_v.size)
        if n_total <= 1:
            return SplFrequencyResult(np.array([]), np.array([]))

        v2pa_value = 1.0 if not v2pa_factor else float(v2pa_factor)
        y_full_pa = y_full_v * v2pa_value

        repeat_times = int(stimulus_metadata.get("repeat_times", 1) or 1)
        repeat_times = max(1, repeat_times)
        samples_per_rep_from_len = n_total // repeat_times

        frame_size = int(frame_size)
        hop_length = int(hop_length)
        if frame_size <= 0 or hop_length <= 0:
            raise ValueError("frame_size and hop_length must be positive")
        if samples_per_rep_from_len <= frame_size:
            return SplFrequencyResult(np.array([]), np.array([]))

        total_time = stimulus_metadata.get("total_time")
        if total_time is None:
            raise ValueError("SPL-Frequency(chirp) requires total_time in stimulus_metadata")
        single_rep_duration_s = float(total_time) / float(repeat_times)

        window = np.hanning(frame_size).astype(np.float64, copy=False)
        w_sum = float(np.sum(window))
        if w_sum <= 0.0:
            raise ValueError("Invalid window normalization (sum(window) <= 0)")
        w2_sum = float(np.sum(window * window))
        if w2_sum <= 0.0:
            raise ValueError("Invalid window normalization (sum(window^2) <= 0)")
        half = frame_size // 2

        num_frames = 1 + (samples_per_rep_from_len - frame_size) // hop_length
        time_s = (np.arange(num_frames) * hop_length + frame_size / 2) / self.sample_rate
        f_inst = _instantaneous_frequency(time_s, stimulus_metadata, single_rep_duration_s)

        n_idx = np.arange(frame_size, dtype=np.float64)
        rms_sq_accum = np.zeros(num_frames, dtype=np.float64)
        for rep in range(repeat_times):
            rep_start = rep * samples_per_rep_from_len
            rep_end = rep_start + samples_per_rep_from_len
            y = y_full_pa[rep_start:rep_end]
            for frame_idx, freq_hz in enumerate(f_inst):
                center = frame_idx * hop_length + half
                start = center - half
                end = start + frame_size
                y_seg = y[start:end]
                if y_seg.size != frame_size:
                    continue

                if splf_calc_mode == "fundamental":
                    basis = np.exp(-1j * (2.0 * np.pi * float(freq_hz) / self.sample_rate) * n_idx)
                    y_k = np.dot(y_seg * window, basis)

                    peak_amp_pa = 2.0 * np.abs(y_k) / w_sum
                    rms_pa = float(peak_amp_pa) / np.sqrt(2.0)
                else:
                    y0 = y_seg - float(np.mean(y_seg))
                    rms_pa = float(np.sqrt(np.sum((y0 * window) ** 2) / w2_sum))
                rms_sq_accum[frame_idx] += rms_pa * rms_pa

        rms_pa_mean = np.sqrt(rms_sq_accum / float(repeat_times))

        freqs_out = np.asarray(f_inst, dtype=np.float64)
        stimulus_type = str(stimulus_metadata.get("stimulus_type", ""))
        if "mirror" in stimulus_type:
            mid = rms_pa_mean.size // 2
            back = rms_pa_mean[:mid][::-1]
            fwd = rms_pa_mean[mid:]
            f_fwd = freqs_out[mid:]
            min_len = int(min(back.size, fwd.size, f_fwd.size))
            if min_len <= 0:
                return SplFrequencyResult(np.array([]), np.array([]))
            rms_pa_mean = np.sqrt(0.5 * (back[:min_len] ** 2 + fwd[:min_len] ** 2))
            freqs_out = np.asarray(f_fwd[:min_len], dtype=np.float64)

        sort_idx = np.argsort(freqs_out)
        freqs_out = freqs_out[sort_idx]
        rms_pa_mean = rms_pa_mean[sort_idx]

        spl_db = _spl_db_from_pa_rms(rms_pa_mean + float(eps), reference_pressure_pa)
        return SplFrequencyResult(freqs_out, spl_db)
