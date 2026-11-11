"""Daniel-Weber psychoacoustic roughness.

The implementation follows MoSQITo's ``roughness_dw`` path intentionally,
including its bin numbering and spectrum mirroring conventions. Those details
look unusual compared with a standard ``rfft`` implementation, but changing
them breaks point-by-point compatibility with MoSQITo.
"""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .stat_utils import iso_c_percentile

from .roughness_constants import (
    DEFAULT_ROUGHNESS_OVERLAP,
    MOSQITO_ROUGHNESS_MODEL_NAME,
    ROUGHNESS_BARK_AXIS,
    ROUGHNESS_BARK_BAND_COUNT,
    ROUGHNESS_CALIBRATION_FACTOR,
    ROUGHNESS_EAR_FILTER_BARK_POINTS,
    ROUGHNESS_EAR_FILTER_DB,
    ROUGHNESS_FFT_AMPLITUDE_CORRECTION,
    ROUGHNESS_FRAME_DURATION_S,
    ROUGHNESS_GZI_BARK_POINTS,
    ROUGHNESS_GZI_VALUES,
    ROUGHNESS_HALF_BARK_STEP,
    ROUGHNESS_H_WEIGHTING_CURVES,
    ROUGHNESS_H_WEIGHTING_START_BIN,
    ROUGHNESS_LTQ_BARK_POINTS,
    ROUGHNESS_LTQ_DB,
    ROUGHNESS_LOWER_SLOPE_DB_PER_BARK,
    ROUGHNESS_MODEL_NAME,
    ROUGHNESS_UPPER_SLOPE_BASE_DB_PER_BARK,
    ROUGHNESS_UPPER_SLOPE_FREQ_FACTOR,
    ROUGHNESS_UPPER_SLOPE_LEVEL_FACTOR,
    ZWICKER_BARK_FREQUENCY_TABLE_HZ,
    ZWICKER_HALF_BARK_AXIS,
)


@dataclass(frozen=True)
class RoughnessSummary:
    """Representative roughness values in asper."""

    r_mean_asper: float
    r_max_asper: float
    r5_asper: float


@dataclass(frozen=True)
class RoughnessResult:
    """Output of Daniel-Weber roughness over 200 ms frames."""

    time_s: np.ndarray
    roughness_asper: np.ndarray
    specific_roughness_asper_per_bark: np.ndarray
    bark_axis: np.ndarray
    summary: RoughnessSummary
    metadata: Dict[str, object]


def compute_roughness_daniel_weber(
    signal_pa: np.ndarray,
    sample_rate: int,
    *,
    overlap: float = DEFAULT_ROUGHNESS_OVERLAP,
) -> RoughnessResult:
    """Compute Daniel-Weber roughness from a mono pressure signal in Pa.

    Parameters
    ----------
    signal_pa:
        Mono time-domain sound pressure in Pa.
    sample_rate:
        Original sample rate in Hz. The signal is not resampled.
    overlap:
        MoSQITo-compatible segmentation coefficient. ``0.5`` means a hop of
        ``0.5 * nperseg``. ``0`` is the MoSQITo special case for no overlap.
    """

    fs = int(sample_rate)
    if fs <= 0:
        raise ValueError("sample_rate must be > 0")

    signal = np.asarray(signal_pa, dtype=np.float64)
    if signal.ndim != 1:
        signal = np.squeeze(signal)
    if signal.ndim != 1:
        raise ValueError("signal_pa must be a mono 1-D signal")
    if signal.size == 0:
        raise ValueError("signal_pa is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal_pa contains NaN or inf values")

    nperseg = int(ROUGHNESS_FRAME_DURATION_S * fs)
    if nperseg <= 0:
        raise ValueError("sample_rate is too low for a 200 ms roughness frame")
    if signal.size < nperseg:
        raise ValueError("signal is shorter than the required 200 ms roughness frame")

    hop = _mosqito_hop_from_overlap(nperseg, overlap)
    frames, time_s = _time_segmentation(signal, fs, nperseg, hop)
    spectrum, freq_axis = _comp_spectrum(frames, fs, nperseg)

    h_weight = _h_weighting(nperseg, fs)
    gzi = _gzi_weighting(ROUGHNESS_BARK_AXIS)

    invariants = _precompute_frame_invariants(freq_axis, fs, nperseg)

    nseg = spectrum.shape[1] if spectrum.ndim > 1 else 1
    roughness = np.empty(nseg, dtype=np.float64)
    specific = np.empty((ROUGHNESS_BARK_BAND_COUNT, nseg), dtype=np.float64)
    bark_axis = ROUGHNESS_BARK_AXIS

    if spectrum.ndim == 1:
        roughness[0], specific[:, 0], bark_axis = _roughness_dw_main_calc(
            spectrum, freq_axis, fs, gzi, h_weight, invariants
        )
    else:
        for idx in range(nseg):
            roughness[idx], specific[:, idx], bark_axis = _roughness_dw_main_calc(
                spectrum[:, idx], freq_axis, fs, gzi, h_weight, invariants
            )

    roughness = np.where(np.isfinite(roughness), roughness, np.nan)
    specific = np.where(np.isfinite(specific), specific, 0.0)
    summary = _build_summary(roughness)

    return RoughnessResult(
        time_s=np.asarray(time_s, dtype=np.float64),
        roughness_asper=roughness,
        specific_roughness_asper_per_bark=specific,
        bark_axis=np.asarray(bark_axis, dtype=np.float64),
        summary=summary,
        metadata={
            "model": ROUGHNESS_MODEL_NAME,
            "backend": "project_engineering",
            "sample_rate_hz": fs,
            "nperseg": nperseg,
            "overlap": float(overlap),
            "hop_samples": hop,
            "frame_duration_s": ROUGHNESS_FRAME_DURATION_S,
            "window": "blackman",
            "window_normalization": "divide_by_window_sum",
            "fft_amplitude_correction": ROUGHNESS_FFT_AMPLITUDE_CORRECTION,
            "calibration_factor": ROUGHNESS_CALIBRATION_FACTOR,
            "frequency_axis": "np.arange(1, nperseg//2 + 1) * fs / nperseg",
            "spectrum_mirroring": "np.concatenate((spec, spec[len(spec)::-1]))",
            "unit": "asper",
        },
    )

def compute_roughness_mosqito_dw(
    signal_pa: np.ndarray,
    sample_rate: int,
    *,
    overlap: float = DEFAULT_ROUGHNESS_OVERLAP,
) -> RoughnessResult:
    """Compute Daniel-Weber roughness through MoSQITo's public backend."""

    try:
        from mosqito.sq_metrics import roughness_dw
    except ImportError as exc:
        raise RuntimeError("MoSQITo roughness backend is unavailable; install mosqito to run ROUGH.") from exc

    fs = int(sample_rate)
    if fs <= 0:
        raise ValueError("sample_rate must be > 0")

    signal = np.asarray(signal_pa, dtype=np.float64)
    if signal.ndim != 1:
        signal = np.squeeze(signal)
    if signal.ndim != 1:
        raise ValueError("signal_pa must be a mono 1-D signal")
    if signal.size == 0:
        raise ValueError("signal_pa is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal_pa contains NaN or inf values")

    nperseg = int(ROUGHNESS_FRAME_DURATION_S * fs)
    if nperseg <= 0:
        raise ValueError("sample_rate is too low for a 200 ms roughness frame")
    if signal.size < nperseg:
        raise ValueError("signal is shorter than the required 200 ms roughness frame")

    overlap_value = float(overlap)
    hop = _mosqito_hop_from_overlap(nperseg, overlap_value)
    roughness, specific, bark_axis, time_s = roughness_dw(signal, fs, overlap=overlap_value)
    roughness = np.nan_to_num(np.asarray(roughness, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    specific = np.nan_to_num(np.asarray(specific, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    bark_axis = np.asarray(bark_axis, dtype=np.float64).reshape(-1)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if specific.ndim == 1:
        specific = specific.reshape(-1, 1)

    return RoughnessResult(
        time_s=time_s,
        roughness_asper=roughness,
        specific_roughness_asper_per_bark=specific,
        bark_axis=bark_axis,
        summary=_build_summary(roughness),
        metadata={
            "model": MOSQITO_ROUGHNESS_MODEL_NAME,
            "backend": "mosqito",
            "backend_function": "mosqito.sq_metrics.roughness_dw",
            "sample_rate_hz": fs,
            "nperseg": nperseg,
            "overlap": overlap_value,
            "hop_samples": hop,
            "frame_duration_s": ROUGHNESS_FRAME_DURATION_S,
            "unit": "asper",
        },
    )


def _mosqito_hop_from_overlap(nperseg: int, overlap: float) -> int:
    try:
        overlap_value = float(overlap)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap must be numeric") from exc
    if overlap_value < 0.0 or overlap_value >= 1.0:
        raise ValueError("overlap must be in [0, 1)")
    hop = int(overlap_value * nperseg)
    if hop == 0:
        hop = nperseg
    if hop <= 0:
        raise ValueError("roughness hop must be > 0")
    return hop


def _time_segmentation(signal: np.ndarray, fs: int, nperseg: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, signal.size - nperseg + 1, hop, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("signal is shorter than the required 200 ms roughness frame")

    offsets = np.arange(nperseg, dtype=np.int64)
    frames = signal[starts[:, None] + offsets].T  # (nperseg, n_seg)

    time_s = (starts.astype(np.float64) + (nperseg - 1) / 2.0) / float(fs)
    return frames, time_s


def _comp_spectrum(frames: np.ndarray, fs: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.blackman(nperseg).astype(np.float64, copy=False)
    window = window / np.sum(window)
    if frames.ndim > 1:
        window = window[:, np.newaxis]

    spectrum = (
        np.fft.fft(frames * window, n=nperseg, axis=0)[0 : nperseg // 2]
        * ROUGHNESS_FFT_AMPLITUDE_CORRECTION
    )
    freq_axis = np.arange(1, nperseg // 2 + 1, 1, dtype=np.float64) * (float(fs) / nperseg)
    return spectrum, freq_axis


def _precompute_frame_invariants(freq_axis: np.ndarray, fs: int, nperseg: int) -> dict:
    """Compute values that are identical for every frame."""
    n = nperseg
    bark_axis = _freq2bark(freq_axis)
    nz = np.arange(1, n // 2 + 1, 1)

    a0 = np.zeros(n, dtype=np.float64)
    a0[nz - 1] = _db2amp(_ear_filter_coeff(bark_axis), ref=1.0)

    threshold = _ltq_roughness(bark_axis)
    zi = ROUGHNESS_BARK_AXIS
    zb = _bark2freq(zi) * n / fs
    min_excit_db = np.interp(zb, nz, threshold)

    return {
        "bark_axis": bark_axis,
        "a0": a0,
        "threshold": threshold,
        "min_excit_db": min_excit_db,
        "n": n,
        "nz": nz,
    }


def _roughness_dw_main_calc(
    spec: np.ndarray,
    freq_axis: np.ndarray,
    fs: int,
    gzi: np.ndarray,
    h_weight: np.ndarray,
    inv: dict,
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(spec) != len(freq_axis):
        raise ValueError("spectrum and frequency axis should have the same number of points")

    spec = np.concatenate((spec, spec[len(spec) :: -1]))
    n = inv["n"]
    bark_axis = inv["bark_axis"]
    threshold = inv["threshold"]
    min_excit_db = inv["min_excit_db"]

    spec = inv["a0"] * spec

    module = np.abs(spec[0 : n // 2])
    module_safe = module.copy()
    module_safe[module_safe == 0] = 2e-12
    spec_db = 20.0 * np.log10(module_safe / 2e-5)
    audible_index = np.where(spec_db > threshold)[0]
    n_aud = len(audible_index)

    n_channel = ROUGHNESS_BARK_BAND_COUNT
    zi = ROUGHNESS_BARK_AXIS
    if n_aud == 0:
        return 0.0, np.zeros(n_channel, dtype=np.float64), zi

    aud_idx = audible_index
    aud_freqs = freq_axis[aud_idx]
    aud_levels = spec_db[aud_idx]
    aud_barks = bark_axis[aud_idx]

    s1 = ROUGHNESS_LOWER_SLOPE_DB_PER_BARK
    s2 = np.minimum(
        ROUGHNESS_UPPER_SLOPE_BASE_DB_PER_BARK
        - ROUGHNESS_UPPER_SLOPE_FREQ_FACTOR / aud_freqs
        + ROUGHNESS_UPPER_SLOPE_LEVEL_FACTOR * aud_levels,
        0.0,
    )

    ch_low = np.floor(aud_barks / ROUGHNESS_HALF_BARK_STEP).astype(np.intp) - 1
    ch_high = np.ceil(aud_barks / ROUGHNESS_HALF_BARK_STEP).astype(np.intp) - 1

    half_step = ROUGHNESS_HALF_BARK_STEP
    channels = np.arange(n_channel, dtype=np.intp)
    channel_positions = (channels.astype(np.float64) + 1.0) * half_step
    channel_indices = channels[None, :]
    bark_2d = aud_barks[:, None]
    level_2d = aud_levels[:, None]
    min_excit_2d = min_excit_db[None, :]

    low_slope_db = s1 * (bark_2d - channel_positions[None, :]) + level_2d
    high_slope_db = s2[:, None] * (channel_positions[None, :] - bark_2d) + level_2d

    low_mask = (channel_indices <= ch_low[:, None]) & (low_slope_db > min_excit_2d)
    high_mask = (channel_indices >= ch_high[:, None]) & (high_slope_db > min_excit_2d)

    slopes = np.zeros((n_aud, n_channel), dtype=np.float64)
    slopes[low_mask] = 10.0 ** (0.05 * low_slope_db[low_mask]) * 2e-5
    slopes[high_mask] = 10.0 ** (0.05 * high_slope_db[high_mask]) * 2e-5

    # --- Build excitation for all 47 channels at once, then batch-FFT ---
    ch_low_2d = ch_low[None, :]       # (1, n_aud)
    ch_high_2d = ch_high[None, :]     # (1, n_aud)
    channels_2d = channels[:, None]   # (n_channel, 1)

    is_low = ch_low_2d == channels_2d
    is_high = ch_high_2d == channels_2d
    is_boundary = is_low | is_high
    is_above = (ch_high_2d > channels_2d) & ~is_boundary
    is_below = ~is_above & ~is_boundary

    module_aud = module[aud_idx]
    module_aud_safe = np.where(module_aud > 0, module_aud, 1.0)

    j_indices = np.arange(n_aud, dtype=np.intp)
    slope_idx_above = np.minimum(channels_2d + 1, n_channel - 1)
    slope_idx_below = np.maximum(channels_2d - 1, 0)

    ampl = np.zeros((n_channel, n_aud), dtype=np.float64)
    ampl[is_boundary] = 1.0
    above_vals = slopes[j_indices, slope_idx_above] / module_aud_safe[None, :]
    below_vals = slopes[j_indices, slope_idx_below] / module_aud_safe[None, :]
    ampl = np.where(is_above, above_vals, ampl)
    ampl = np.where(is_below, below_vals, ampl)

    exc_all = np.zeros((n_channel, n), dtype=complex)
    exc_all[:, aud_idx] = ampl * spec[aud_idx][None, :]

    temporal_all = np.abs(n * np.real(np.fft.ifft(exc_all, axis=1)))
    h0_all = np.mean(temporal_all, axis=1)                          # (n_channel,)
    envelope_all = np.fft.fft(temporal_all - h0_all[:, None], axis=1)
    envelope_all *= h_weight
    hbp = 2.0 * np.real(np.fft.ifft(envelope_all, axis=1))

    hbp_rms = np.sqrt(np.mean(hbp * hbp, axis=1))
    h0_safe = np.where(h0_all > 0, h0_all, 1.0)
    mod_depth = np.where(h0_all > 0, np.minimum(hbp_rms / h0_safe, 1.0), 0.0)

    # --- Batch Pearson correlation between channels i and i+2 ---
    ki = np.zeros(n_channel, dtype=np.float64)
    a_rows = hbp[0 : n_channel - 2]                                # (45, n)
    b_rows = hbp[2 : n_channel]                                    # (45, n)
    a_mean = a_rows.mean(axis=1, keepdims=True)
    b_mean = b_rows.mean(axis=1, keepdims=True)
    a_c = a_rows - a_mean
    b_c = b_rows - b_mean
    num = np.sum(a_c * b_c, axis=1)
    denom = np.sqrt(np.sum(a_c * a_c, axis=1) * np.sum(b_c * b_c, axis=1))
    valid = denom > 0
    ki[: n_channel - 2] = np.where(valid, num / np.where(valid, denom, 1.0), 0.0)

    r_spec = np.empty(n_channel, dtype=np.float64)
    r_spec[0] = gzi[0] * (mod_depth[0] * ki[0]) ** 2
    r_spec[1] = gzi[1] * (mod_depth[1] * ki[1]) ** 2
    r_spec[2 : n_channel - 2] = (
        gzi[2 : n_channel - 2]
        * (mod_depth[2 : n_channel - 2] * ki[2 : n_channel - 2] * ki[0 : n_channel - 4])
        ** 2
    )
    r_spec[n_channel - 2] = gzi[n_channel - 2] * (mod_depth[n_channel - 2] * ki[n_channel - 4]) ** 2
    r_spec[n_channel - 1] = gzi[n_channel - 1] * (mod_depth[n_channel - 1] * ki[n_channel - 3]) ** 2
    r_spec = np.nan_to_num(r_spec, nan=0.0, posinf=0.0, neginf=0.0)

    roughness = ROUGHNESS_CALIBRATION_FACTOR * float(np.sum(r_spec))
    return roughness, r_spec, zi


def _build_summary(roughness: np.ndarray) -> RoughnessSummary:
    values = np.asarray(roughness, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return RoughnessSummary(float("nan"), float("nan"), float("nan"))
    return RoughnessSummary(
        r_mean_asper=float(np.mean(values)),
        r_max_asper=float(np.max(values)),
        r5_asper=iso_c_percentile(values, percentile=5),
    )


def _h_weighting(n: int, fs: int) -> np.ndarray:
    cut = ROUGHNESS_H_WEIGHTING_START_BIN
    h = np.zeros((ROUGHNESS_BARK_BAND_COUNT, n), dtype=np.float64)

    for band_index, max_freq_hz, x_points, y_points in ROUGHNESS_H_WEIGHTING_CURVES:
        last = math.floor((float(max_freq_hz) / fs) * n)
        j = np.arange(cut, last)
        freq = j * fs / n
        h[band_index, j] = np.interp(freq[j - cut], x_points, y_points)

    h[0, :] = h[2, :] = h[3, :] = h[1, :]
    for i in range(5, 15):
        h[i, :] = h[4, :]
    for i in range(16, 20):
        h[i, :] = h[15, :]
    for i in range(21, 41):
        h[i, :] = h[20, :]
    for i in range(42, ROUGHNESS_BARK_BAND_COUNT):
        h[i, :] = h[41, :]
    return h


def _gzi_weighting(center_freq: np.ndarray) -> np.ndarray:
    return np.interp(center_freq, ROUGHNESS_GZI_BARK_POINTS, ROUGHNESS_GZI_VALUES)


def _ear_filter_coeff(bark_axis: np.ndarray) -> np.ndarray:
    return np.interp(bark_axis, ROUGHNESS_EAR_FILTER_BARK_POINTS, ROUGHNESS_EAR_FILTER_DB)


def _ltq_roughness(bark_axis: np.ndarray) -> np.ndarray:
    return np.interp(bark_axis, ROUGHNESS_LTQ_BARK_POINTS, ROUGHNESS_LTQ_DB)


def _freq2bark(freq_axis: np.ndarray) -> np.ndarray:
    return np.interp(freq_axis, ZWICKER_BARK_FREQUENCY_TABLE_HZ, ZWICKER_HALF_BARK_AXIS)


def _bark2freq(bark_axis: np.ndarray) -> np.ndarray:
    return np.interp(bark_axis, ZWICKER_HALF_BARK_AXIS, ZWICKER_BARK_FREQUENCY_TABLE_HZ)


def _zwicker_freq_table() -> np.ndarray:
    return ZWICKER_BARK_FREQUENCY_TABLE_HZ


def _db2amp(db: np.ndarray | float, ref: float = 1.0) -> np.ndarray | float:
    if ref == 0:
        raise ValueError("Reference must be different from 0")
    return np.power(10.0, 0.05 * db) * ref


def _amp2db(amp: np.ndarray, ref: float = 1.0) -> np.ndarray:
    if ref == 0:
        raise ValueError("Reference must be different from 0")
    arr = np.asarray(amp, dtype=np.float64)
    arr[arr == 0] = 2e-12
    return 20.0 * np.log10(arr / ref)
