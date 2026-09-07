"""Pure SPL trace computation shared by active and detached analysis."""
from dataclasses import dataclass

import numpy as np

from base.pre_processing.audio_thd_frequency_response_analysis import (
    AudioThdFrequencyResponseAnalysis,
)
from base.pre_processing.spl_runtime_config import (
    apply_spl_analysis_time_range,
    calculate_overall_spl,
)
from base.core_algorithm.harmonic_distortion.weighted import (
    apply_weighting_filter,
)
from base.utils.smooth import smooth


@dataclass(frozen=True)
class SplTrace:
    analysis_signal: np.ndarray
    signal_duration: np.ndarray
    signal_spl: np.ndarray
    overall_spl: float


def compute_spl_trace(
        signal, sample_rate, config, *, v2pa_factor,
        weighting_filter=apply_weighting_filter,
        spl_calculator=None,
        overall_calculator=calculate_overall_spl):
    sample_rate = int(sample_rate)
    config = dict(config or {})
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    weighting = str(config.get("weighting", "Z") or "Z").upper()
    if weighting not in ("NONE", "Z"):
        values = weighting_filter(
            values, sample_rate, weighting=weighting, zero_phase=False)
    analysis_signal, analysis_start_sample = apply_spl_analysis_time_range(
        values, sample_rate, config)
    overall_spl = overall_calculator(
        analysis_signal, 20e-6, v2pa_factor=v2pa_factor)
    window_size = 1201
    if spl_calculator is None:
        spl_calculator = AudioThdFrequencyResponseAnalysis().spl_calculation
    signal_spl = spl_calculator(
        analysis_signal, 20e-6, window_size=window_size,
        v2pa_factor=v2pa_factor, trim_edges=True)
    start_index = (
        0 if len(signal_spl) == len(analysis_signal) else window_size // 2)
    signal_duration = (
        np.arange(len(signal_spl), dtype=float) + float(start_index)
    ) / float(sample_rate)
    signal_duration += float(analysis_start_sample) / float(sample_rate)
    if config.get("smooth_checked"):
        signal_spl = smooth(signal_spl, window_size=1102, method="savgol")
    return SplTrace(
        np.asarray(analysis_signal), np.asarray(signal_duration),
        np.asarray(signal_spl), float(overall_spl))
