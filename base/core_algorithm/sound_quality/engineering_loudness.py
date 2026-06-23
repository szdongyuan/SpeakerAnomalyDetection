"""
Engineering Zwicker-like loudness analysis.

This module is a project-local implementation shaped by the public MoSQITo
structure and the SQAT paper description of ISO 532-1:2017. It intentionally
does not import or copy MoSQITo. The first version is meant for product
integration and trend analysis; strict ISO conformance still needs validation
against the ISO 532-1 electronic test package.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
from scipy import signal as scipy_signal

from .psychoacoustic_constants import (
    BARK_AXIS_24_BARK_0_1_STEP,
    LOUDNESS_CORE_LEVEL_EXPONENT_SCALE,
    LOUDNESS_CORE_LTQ_EXPONENT_SCALE,
    LOUDNESS_CORE_MP1_SCALE,
    LOUDNESS_CORE_POWER,
    LOUDNESS_CORE_S,
    LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S,
    LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S,
    LOUDNESS_INTERNAL_TIME_RESOLUTION_S,
    LOUDNESS_TV_SUMMARY_SKIP_S,
    LOUDNESS_A0_TABLE as _A0_TABLE,
    LOUDNESS_COMPARISON_DECIMALS,
    LOUDNESS_DCB_TABLE as _DCB_TABLE,
    LOUDNESS_DDF_TABLE as _DDF_TABLE,
    LOUDNESS_DLL_TABLE as _DLL_TABLE,
    LOUDNESS_LOW_BAND_CORRECTION_BASE,
    LOUDNESS_LOW_BAND_CORRECTION_POWER,
    LOUDNESS_LOW_BAND_CORRECTION_SCALE,
    LOUDNESS_LTQ_TABLE as _LTQ_TABLE,
    LOUDNESS_RAP_TABLE as _RAP_TABLE,
    LOUDNESS_RNS_TABLE,
    LOUDNESS_SPECIFIC_BARK_BIN_COUNT,
    LOUDNESS_STATIONARY_WINDOW_EDGE_TRIM_RATIO,
    LOUDNESS_TEMPORAL_FAST_TAU_S,
    LOUDNESS_TEMPORAL_FAST_WEIGHT,
    LOUDNESS_TEMPORAL_SLOW_TAU_S,
    LOUDNESS_TEMPORAL_SLOW_WEIGHT,
    LOUDNESS_TIME_VARYING_FILTER_DELTA,
    LOUDNESS_TIME_VARYING_FILTER_GAIN,
    LOUDNESS_TIME_VARYING_FILTER_REF,
    LOUDNESS_USL_FULL_TABLE,
    LOUDNESS_ZUP_TABLE,
    ONE_THIRD_CENTERS_HZ,
    P_REF,
    TARGET_FS_HZ,
)

_NUMBA_AVAILABLE = False


def _maybe_njit(*njit_args, **njit_kwargs):
    """No-op decorator — numba JIT is disabled.

    The pure-Python fallback is used unconditionally. This avoids the
    heavy first-call JIT compilation penalty (~10 s) and removes the
    numba dependency from the build.
    """

    def _wrap(func):
        return func

    return _wrap




class LoudnessMethod(str, Enum):
    """Supported loudness modes."""

    PER_SEGMENT = "per_segment"
    TIME_VARYING_ISO532_1 = "time_varying_iso532_1"


@dataclass(frozen=True)
class LoudnessSummary:
    mean_sone: float
    max_sone: float
    mean_phon: float
    max_phon: float


@dataclass(frozen=True)
class LoudnessResult:
    time_s: np.ndarray
    loudness_sone: np.ndarray
    loudness_level_phon: np.ndarray
    specific_loudness: np.ndarray
    bark_axis: np.ndarray
    summary: LoudnessSummary
    metadata: Dict[str, object]


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def sones_to_phons(loudness_sone: np.ndarray) -> np.ndarray:
    """Convert loudness from sone to phon using the common ISO 532 relation."""

    n = np.asarray(loudness_sone, dtype=np.float64)
    n_safe = np.maximum(n, 0.0)
    return np.where(
        n_safe >= 1.0,
        40.0 + 10.0 * np.log2(np.maximum(n_safe, np.finfo(np.float64).tiny)),
        40.0 * np.power(n_safe + 0.0005, 0.35),
    )


def _summary_from_loudness(loudness_sone: np.ndarray, loudness_level_phon: np.ndarray) -> LoudnessSummary:
    loudness = np.asarray(loudness_sone, dtype=np.float64)
    phon = np.asarray(loudness_level_phon, dtype=np.float64)
    if loudness.size == 0:
        return LoudnessSummary(np.nan, np.nan, np.nan, np.nan)
    return LoudnessSummary(
        mean_sone=float(np.mean(loudness)),
        max_sone=float(np.max(loudness)),
        mean_phon=float(np.mean(phon)),
        max_phon=float(np.max(phon)),
    )


def _resample_if_needed(x: np.ndarray, fs: int) -> tuple[np.ndarray, int, bool]:
    if int(fs) == TARGET_FS_HZ:
        return x, int(fs), False
    n_out = int(round(x.size * TARGET_FS_HZ / float(fs)))
    if n_out <= 1:
        return x, int(fs), False
    return scipy_signal.resample(x, n_out), TARGET_FS_HZ, True


def _third_octave_levels_sliding_iso_frontend(
    signal_pa: np.ndarray,
    fs: int,
    frame_size: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate sliding steady-state 1/3-octave levels through the ISO filterbank.

    A rectangular FFT band sum is phase-stable, but it treats a pure tone as if
    it lived only in one ideal 1/3-octave bucket. The Zwicker loudness tables are
    calibrated against the ISO-style filterbank response, where neighbouring
    filters receive finite energy from a 1 kHz tone. Reusing the same filterbank
    as the time-varying path keeps stationary and time-varying results aligned.
    """

    if int(fs) != TARGET_FS_HZ:
        raise ValueError("sliding ISO loudness frontend expects an internal 48 kHz signal")
    x = _as_1d_float_array(signal_pa)
    if x.size == 0:
        raise ValueError("signal_pa must not be empty")

    frame_size = min(int(frame_size), int(x.size))
    starts = np.arange(0, x.size - frame_size + 1, int(hop), dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)
    time_s = (starts + frame_size / 2.0) / float(fs)

    high_res_levels, high_res_time = _third_octave_levels_time_varying(x, fs)
    levels = np.empty((ONE_THIRD_CENTERS_HZ.size, starts.size), dtype=np.float64)
    for col, start in enumerate(starts):
        # Use the central half of each analysis window to avoid filter startup
        # and window-edge transients contaminating a steady-state estimate.
        trim = LOUDNESS_STATIONARY_WINDOW_EDGE_TRIM_RATIO
        lo = (float(start) + trim * float(frame_size)) / float(fs)
        hi = (float(start) + (1.0 - trim) * float(frame_size)) / float(fs)
        mask = (high_res_time >= lo) & (high_res_time < hi)
        if not np.any(mask):
            lo = float(start) / float(fs)
            hi = float(start + frame_size) / float(fs)
            mask = (high_res_time >= lo) & (high_res_time < hi)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(high_res_time - time_s[col])))
            levels[:, col] = high_res_levels[:, idx]
            continue
        band_power = np.mean(10.0 ** (high_res_levels[:, mask] / 10.0), axis=1)
        levels[:, col] = 10.0 * np.log10(np.maximum(band_power, np.finfo(np.float64).tiny))
    return levels, time_s


def _smooth_squared_band(x: np.ndarray, center_freq_hz: float, fs: int) -> np.ndarray:
    """Square a band signal and smooth it with three first-order low-pass filters."""

    tau = 2.0 / (3.0 * min(float(center_freq_hz), 1000.0))
    a1 = np.exp(-1.0 / (float(fs) * tau))
    b0 = 1.0 - a1
    y = np.asarray(x, dtype=np.float64) ** 2
    for _ in range(3):
        y = scipy_signal.lfilter([b0], [1.0, -a1], y)
    return y


def _third_octave_levels_time_varying(signal_pa: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 28 one-third-octave SPL time series at 0.5 ms resolution.

    This follows the `loudness_zwtv` chain shape: ISO table band-pass filters,
    square, third-order smoothing, SPL conversion, then decimation from 48 kHz
    to 2 kHz.
    """

    if int(fs) != TARGET_FS_HZ:
        raise ValueError("time-varying loudness expects an internal 48 kHz signal")
    x = _as_1d_float_array(signal_pa)
    dec_factor = int(TARGET_FS_HZ / 2000)
    n_time = len(x[::dec_factor])
    time_axis = np.arange(n_time, dtype=np.float64) * dec_factor / float(fs)
    levels = np.empty((ONE_THIRD_CENTERS_HZ.size, n_time), dtype=np.float64)
    for idx, fc in enumerate(ONE_THIRD_CENTERS_HZ):
        sos = LOUDNESS_TIME_VARYING_FILTER_REF - LOUDNESS_TIME_VARYING_FILTER_DELTA[idx]
        band = LOUDNESS_TIME_VARYING_FILTER_GAIN[idx] * scipy_signal.sosfilt(sos, x)
        smoothed = _smooth_squared_band(band, float(fc), fs)
        levels[idx, :] = 10.0 * np.log10((smoothed[::dec_factor] + 1e-12) / (P_REF**2))
    return levels, time_axis


@_maybe_njit(cache=True)
def _core_loudness_kernel_single(
    spec: np.ndarray,
    field_is_diffuse: bool,
    rap: np.ndarray,
    dll: np.ndarray,
    ltq: np.ndarray,
    a0: np.ndarray,
    ddf: np.ndarray,
    dcb: np.ndarray,
) -> np.ndarray:
    """Compute one Zwicker core-loudness column from 28 third-octave levels."""

    s = LOUDNESS_CORE_S

    corrected_low = np.empty(11, dtype=np.float64)
    for band_idx in range(11):
        level = spec[band_idx]
        correction = dll[7, band_idx]
        for range_idx in range(rap.size):
            if level <= rap[range_idx] - dll[range_idx, band_idx]:
                correction = dll[range_idx, band_idx]
                break
        corrected_low[band_idx] = level + correction

    s0 = 0.0
    for i in range(0, 6):
        s0 += 10.0 ** (corrected_low[i] / 10.0)
    s1 = 0.0
    for i in range(6, 9):
        s1 += 10.0 ** (corrected_low[i] / 10.0)
    s2 = 0.0
    for i in range(9, 11):
        s2 += 10.0 ** (corrected_low[i] / 10.0)
    lcb0 = 10.0 * np.log10(s0) if s0 > 0.0 else 0.0
    lcb1 = 10.0 * np.log10(s1) if s1 > 0.0 else 0.0
    lcb2 = 10.0 * np.log10(s2) if s2 > 0.0 else 0.0

    le = np.empty(20, dtype=np.float64)
    le[0] = lcb0
    le[1] = lcb1
    le[2] = lcb2
    for i in range(3, 20):
        le[i] = spec[8 + i]
    for i in range(20):
        le[i] -= a0[i]
        if field_is_diffuse:
            le[i] += ddf[i]

    core = np.zeros(21, dtype=np.float64)
    for i in range(20):
        if le[i] > ltq[i]:
            le_adj = le[i] - dcb[i]
            mp1 = LOUDNESS_CORE_MP1_SCALE * (
                10.0 ** (LOUDNESS_CORE_LTQ_EXPONENT_SCALE * ltq[i])
            )
            mp2 = (
                1.0
                - s
                + s * (10.0 ** (LOUDNESS_CORE_LEVEL_EXPONENT_SCALE * (le_adj - ltq[i])))
            ) ** LOUDNESS_CORE_POWER - 1.0
            val = mp1 * mp2
            if val < 0.0:
                val = 0.0
            core[i] = val

    low_band_correction = LOUDNESS_LOW_BAND_CORRECTION_BASE + (
        LOUDNESS_LOW_BAND_CORRECTION_SCALE
        * (max(core[0], 0.0) ** LOUDNESS_LOW_BAND_CORRECTION_POWER)
    )
    if low_band_correction <= 1.0:
        core[0] *= low_band_correction
    return core


@_maybe_njit(cache=True)
def _core_loudness_kernel_batch(
    spec: np.ndarray,
    field_is_diffuse: bool,
    rap: np.ndarray,
    dll: np.ndarray,
    ltq: np.ndarray,
    a0: np.ndarray,
    ddf: np.ndarray,
    dcb: np.ndarray,
) -> np.ndarray:
    """Vectorized Zwicker core-loudness over time. ``spec`` is shape (28, T)."""

    n_time = spec.shape[1]
    out = np.zeros((21, n_time), dtype=np.float64)
    spec_col = np.empty(28, dtype=np.float64)
    for t in range(n_time):
        for band in range(28):
            spec_col[band] = spec[band, t]
        out[:, t] = _core_loudness_kernel_single(
            spec_col, field_is_diffuse, rap, dll, ltq, a0, ddf, dcb
        )
    return out


def _core_loudness_from_third_octaves_batch(levels_db: np.ndarray, field_type: str) -> np.ndarray:
    """Compute Zwicker core-loudness for all time frames via NumPy batch ops.

    Input shape: (28, T), output shape: (21, T).
    """
    spec = np.asarray(levels_db, dtype=np.float64)
    if spec.ndim != 2 or spec.shape[0] != 28:
        raise ValueError(f"Expected (28, T) levels, got shape {spec.shape}")
    T = spec.shape[1]
    if T == 0:
        return np.zeros((21, 0), dtype=np.float64)
    max_low = np.nanmax(spec[:11])
    if max_low > 120.0:
        raise ValueError("Zwicker loudness is not valid above 120 dB in the first 11 bands")

    field_is_diffuse = str(field_type).lower() == "diffuse"
    rap = _RAP_TABLE
    dll = _DLL_TABLE
    ltq = _LTQ_TABLE
    a0 = _A0_TABLE
    ddf = _DDF_TABLE
    dcb = _DCB_TABLE
    s = LOUDNESS_CORE_S

    low_spec = spec[:11, :]  # (11, T)
    corrected_low = np.empty((11, T), dtype=np.float64)
    for band_idx in range(11):
        level = low_spec[band_idx, :]
        correction = np.full(T, dll[7, band_idx], dtype=np.float64)
        still_default = np.ones(T, dtype=bool)
        for range_idx in range(rap.size):
            threshold = rap[range_idx] - dll[range_idx, band_idx]
            matched = still_default & (level <= threshold)
            correction[matched] = dll[range_idx, band_idx]
            still_default &= ~matched
        corrected_low[band_idx, :] = level + correction

    s0 = np.sum(10.0 ** (corrected_low[:6, :] / 10.0), axis=0)
    s1 = np.sum(10.0 ** (corrected_low[6:9, :] / 10.0), axis=0)
    s2 = np.sum(10.0 ** (corrected_low[9:11, :] / 10.0), axis=0)

    lcb0 = np.where(s0 > 0.0, 10.0 * np.log10(s0), 0.0)
    lcb1 = np.where(s1 > 0.0, 10.0 * np.log10(s1), 0.0)
    lcb2 = np.where(s2 > 0.0, 10.0 * np.log10(s2), 0.0)

    le = np.empty((20, T), dtype=np.float64)
    le[0, :] = lcb0
    le[1, :] = lcb1
    le[2, :] = lcb2
    le[3:, :] = spec[11:, :]
    le -= a0[:, np.newaxis]
    if field_is_diffuse:
        le += ddf[:, np.newaxis]

    above_threshold = le > ltq[:, np.newaxis]
    le_adj = le - dcb[:, np.newaxis]
    mp1 = LOUDNESS_CORE_MP1_SCALE * (10.0 ** (LOUDNESS_CORE_LTQ_EXPONENT_SCALE * ltq))
    mp2_base = 1.0 - s + s * (10.0 ** (LOUDNESS_CORE_LEVEL_EXPONENT_SCALE * (le_adj - ltq[:, np.newaxis])))
    mp2 = mp2_base ** LOUDNESS_CORE_POWER - 1.0
    core_vals = np.maximum(mp1[:, np.newaxis] * mp2, 0.0)
    core_vals[~above_threshold] = 0.0

    core = np.zeros((21, T), dtype=np.float64)
    core[:20, :] = core_vals

    low_band_correction = LOUDNESS_LOW_BAND_CORRECTION_BASE + (
        LOUDNESS_LOW_BAND_CORRECTION_SCALE
        * (np.maximum(core[0, :], 0.0) ** LOUDNESS_LOW_BAND_CORRECTION_POWER)
    )
    mask_correct = low_band_correction <= 1.0
    core[0, :] = np.where(mask_correct, core[0, :] * low_band_correction, core[0, :])
    return core


_NEG_RNS_SORTED = np.sort(-np.round(LOUDNESS_RNS_TABLE, 8))


def _get_rns_index(values: np.ndarray, rns: np.ndarray, *, equal_too: bool = False) -> np.ndarray:
    """Return slope-table range indexes for core/specific loudness values.

    For each value *v* the index equals the count of ``rns`` entries that are
    strictly greater than *v* (or >= when *equal_too* is True).  ``rns`` is
    assumed descending; the look-up is done via ``np.searchsorted`` on the
    negated/sorted table, avoiding the large temporary arrays that a
    broadcast-tile approach would create.
    """
    arr = np.round(np.asarray(values, dtype=np.float64), 8)
    neg_rns = _NEG_RNS_SORTED
    side = "right" if equal_too else "left"
    indexes = np.searchsorted(neg_rns, -arr, side=side)
    return np.minimum(indexes, rns.size - 1)


def _specific_loudness_from_core(core_loudness: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Map core loudness to specific loudness using the `zup/rns/usl` slope tables."""

    nm = np.asarray(core_loudness, dtype=np.float64)
    input_was_1d = nm.ndim == 1
    if input_was_1d:
        nm = np.vstack([nm, nm]).T
    if nm.shape[0] != 21:
        raise ValueError(f"Expected 21 core-loudness bands, got {nm.shape[0]}")

    zup = LOUDNESS_ZUP_TABLE
    rns = LOUDNESS_RNS_TABLE
    usl_full = LOUDNESS_USL_FULL_TABLE

    data_length = nm.shape[1]
    nm_wide = nm.shape[0]
    spec_length = LOUDNESS_SPECIFIC_BARK_BIN_COUNT
    dec_compare = LOUDNESS_COMPARISON_DECIMALS
    zup_idx = (zup * 10).astype(np.int32)
    zup_idx = np.append(zup_idx, 0)
    zup_array = (np.ones((nm_wide, data_length)).T * zup).T

    n_specific = np.zeros((spec_length, data_length), dtype=np.float64)
    for idx in range(nm_wide - 1):
        n_specific[zup_idx[idx - 1] : zup_idx[idx], :] = nm[idx]

    loudness = np.zeros(data_length, dtype=np.float64)
    rns_ind = _get_rns_index(nm, rns)
    rns_values = rns[rns_ind]
    usl_array_ind = np.array([rns_ind.T, (np.ones((zup.shape[0], data_length)).T * np.arange(21))], dtype=int).transpose(1, 2, 0)
    usl_array = usl_full[usl_array_ind[:, :, 0], usl_array_ind[:, :, 1]].T

    n2 = np.ones((spec_length, data_length), dtype=np.float64)
    z2 = np.ones((spec_length, data_length), dtype=np.float64)
    usl_specific = np.ones((spec_length, data_length), dtype=np.float64)
    dz_specific = np.ones((spec_length, data_length), dtype=np.float64)
    rns_specific = np.ones((spec_length, data_length), dtype=np.float64)
    dz = zup_array - np.roll(zup_array, 1, axis=0)
    dz[0, :] = zup[1]
    dz[1, :] = zup[1]

    for idx in range(nm_wide):
        n2[zup_idx[idx - 1] : zup_idx[idx], :] = nm[idx]
        dz_specific[zup_idx[idx - 1] : zup_idx[idx], :] = dz[idx]
        z2[zup_idx[idx - 1] : zup_idx[idx], :] = zup_array[idx]
        usl_specific[zup_idx[idx - 1] : zup_idx[idx], :] = usl_array[idx]
        rns_specific[zup_idx[idx - 1] : zup_idx[idx], :] = rns_values[idx]

    n1_aux = np.zeros(data_length, dtype=np.float64)
    z1_aux = np.zeros(data_length, dtype=np.float64)
    for idx in range(nm_wide):
        j_start = zup_idx[idx - 1]
        indexes = _get_rns_index(n2[j_start - 1], rns)
        rns_specific[j_start] = rns[indexes]
        usl_specific[j_start] = usl_full[indexes, idx - 1]

        mask_descending = np.round(n2[j_start - 1], dec_compare) > np.round(nm[idx], dec_compare)
        not_descending = ~mask_descending
        loudness[not_descending] += (n2[j_start] * (z2[j_start] - z1_aux))[not_descending]
        n1_aux[not_descending] = n2[j_start, not_descending]
        z1_aux[not_descending] = z2[j_start, not_descending]

        if mask_descending.any():
            max_rns_nm = np.maximum(rns_specific[j_start - 1], nm[idx])
            z2[j_start, mask_descending] = np.minimum((n1_aux - max_rns_nm) / usl_specific[j_start] + z1_aux, zup[idx])[mask_descending]
            dz_specific[j_start, mask_descending] = (z2[j_start] - z1_aux)[mask_descending]
            n2[j_start, mask_descending] = (n1_aux - dz_specific[j_start] * usl_specific[j_start])[mask_descending]
            loudness[mask_descending] += (dz_specific[j_start] * (n1_aux + n2[j_start]) / 2.0)[mask_descending]

        z_array = np.ones(data_length, dtype=np.float64) * zup[idx - 1] + 0.1
        last_j = j_start
        mask_z_bigger_z2 = np.zeros(data_length, dtype=bool)
        for j in range(zup_idx[idx - 1], zup_idx[idx]):
            last_j = j
            if j != zup_idx[idx - 1]:
                z2[j, mask_descending] = z2[j - 1, mask_descending]
                n2[j, mask_descending] = n2[j - 1, mask_descending]
                dz_specific[j, mask_descending] = dz_specific[j - 1, mask_descending]
                usl_specific[j, mask_descending] = usl_specific[j - 1, mask_descending]
                rns_specific[j, mask_descending] = rns_specific[j - 1, mask_descending]

            mask_z_bigger_z2 = mask_descending & (np.round(z2[j], dec_compare) <= np.round(z_array, dec_compare))
            if mask_z_bigger_z2.any():
                indexes = _get_rns_index(n2[j, mask_z_bigger_z2], rns, equal_too=True)
                rns_specific[j, mask_z_bigger_z2] = rns[indexes]
                usl_specific[j, mask_z_bigger_z2] = usl_full[indexes, idx - 1]
                n1_aux[mask_z_bigger_z2] = n2[j, mask_z_bigger_z2]
                z1_aux[mask_z_bigger_z2] = z2[j, mask_z_bigger_z2]

                mask_low = mask_z_bigger_z2 & (np.round(n1_aux, dec_compare) <= np.round(nm[idx], dec_compare))
                n2[j, mask_low] = nm[idx, mask_low]
                z2[j, mask_low] = zup[idx]
                dz_specific[j, mask_low] = (z2[j] - z1_aux)[mask_low]
                loudness[mask_low] += (n2[j] * (z2[j] - z1_aux))[mask_low]

                mask_high = mask_z_bigger_z2 & (np.round(n1_aux, dec_compare) > np.round(nm[idx], dec_compare))
                max_rns_nm = np.maximum(rns_specific[j], nm[idx])
                z2[j, mask_high] = np.minimum((n1_aux - max_rns_nm) / usl_specific[j] + z1_aux, zup[idx])[mask_high]
                dz_specific[j, mask_high] = (z2[j] - z1_aux)[mask_high]
                n2[j, mask_high] = (n1_aux - dz_specific[j] * usl_specific[j])[mask_high]
                loudness[mask_high] += (dz_specific[j] * (n1_aux + n2[j]) / 2.0)[mask_high]
                n_specific[j, mask_high] = (n1_aux - (z_array - z1_aux) * usl_specific[j])[mask_high]

                mask_rest = np.logical_xor(mask_z_bigger_z2, mask_descending)
                n_specific[j, mask_rest] = (n1_aux - (z_array - z1_aux) * usl_specific[j])[mask_rest]
                mask_descending = np.logical_xor(mask_descending, mask_low)
            else:
                n_specific[j, mask_descending] = (n1_aux - (z_array - z1_aux) * usl_specific[j])[mask_descending]

            z_array += 0.1
            z1_aux = z2[j].copy()
            n1_aux = n2[j].copy()
            if not mask_descending.any():
                break

        z1_aux = z2[zup_idx[idx] - 1].copy()
        n1_aux = n2[zup_idx[idx] - 1].copy()
        if mask_z_bigger_z2.any():
            indexes = _get_rns_index(n2[last_j, mask_z_bigger_z2], rns, equal_too=True)
            rns_specific[last_j, mask_z_bigger_z2] = rns[indexes]
            usl_specific[last_j, mask_z_bigger_z2] = usl_full[indexes, idx - 1]

    loudness[loudness < 0.0] = 0.0
    loudness[loudness <= 16.0] = np.floor(loudness[loudness <= 16.0] * 1000.0 + 0.5) / 1000.0
    loudness[loudness > 16.0] = np.floor(loudness[loudness > 16.0] * 100.0 + 0.5) / 100.0
    bark_axis = BARK_AXIS_24_BARK_0_1_STEP
    if input_was_1d:
        return float(loudness[0]), n_specific[:, 0], bark_axis
    return loudness, n_specific, bark_axis


@_maybe_njit(cache=True)
def _nonlinear_decay_kernel(
    core: np.ndarray,
    nl_iter: int,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> np.ndarray:
    """Numba-friendly inner loop of the Zwicker nonlinear temporal decay.

    Computes ``uo`` (same shape as the interpolated grid) and returns the
    first interpolation step of every input column, matching the original
    ``uo.reshape(..., nl_iter)[:, :, 0]`` slice in pure-Python form.
    """

    n_bands = core.shape[0]
    n_time = core.shape[1]
    n_cols = n_time * nl_iter

    delta = np.empty((n_bands, n_time), dtype=np.float64)
    for band in range(n_bands):
        for t in range(n_time - 1):
            delta[band, t] = (core[band, t + 1] - core[band, t]) / nl_iter
        delta[band, n_time - 1] = -core[band, n_time - 1] / nl_iter

    ui = np.empty((n_bands, n_cols), dtype=np.float64)
    for band in range(n_bands):
        for t in range(n_time):
            base = core[band, t]
            d = delta[band, t]
            offset = t * nl_iter
            for inner in range(nl_iter):
                ui[band, offset + inner] = base + d * inner

    out = np.empty((n_bands, n_time), dtype=np.float64)
    for band in range(n_bands):
        uo_prev = ui[band, 0]
        if core[band, 0] >= 1e-5:
            u2_prev = core[band, 0] * (1.0 - b5)
        else:
            u2_prev = 0.0
        out[band, 0] = uo_prev

        for col in range(1, n_cols):
            ui_cur = ui[band, col]

            uo_cur = ui_cur
            uo2_a = uo_prev * b2 - u2_prev * b3
            if (uo_prev > u2_prev) and (uo2_a >= ui_cur):
                uo_cur = uo2_a
            uo2_b = uo_prev * b4
            if (uo_prev <= u2_prev) and (uo2_b >= ui_cur):
                uo_cur = uo2_b

            u2_cur = uo_cur
            u22 = uo_prev * b0 - u2_prev * b1
            if (ui_cur < uo_prev) and (uo_prev > u2_prev) and (u22 <= uo_cur):
                u2_cur = u22

            u2_2 = (u2_prev - ui_cur) * b5 + ui_cur
            near_previous = (abs(ui_cur - uo_prev) < 1e-5) and (uo_cur <= u2_prev)
            if (ui_cur >= uo_prev) and (not near_previous):
                u2_cur = u2_2

            if col % nl_iter == 0:
                out[band, col // nl_iter] = uo_cur

            uo_prev = uo_cur
            u2_prev = u2_cur

    return out


def _nonlinear_decay_kernel_python(
    core: np.ndarray,
    nl_iter: int,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> np.ndarray:
    """Pure-Python fallback for the Zwicker nonlinear temporal decay.

    Optimisations vs. a naive port of the numba kernel:
    - core rows converted to Python lists (.tolist()) to dodge numpy scalar overhead
    - incremental ``ui_cur += delta`` instead of ``base + delta * inner``
    - ``abs()`` replaced with direct difference comparison
    - inner iteration count cached as local int
    """

    n_bands = core.shape[0]
    n_time = core.shape[1]
    out = np.empty((n_bands, n_time), dtype=np.float64)
    inv_iter = 1.0 / float(nl_iter)
    one_minus_b5 = 1.0 - b5
    n_inner_tail = nl_iter - 1
    eps = 1e-5

    for band in range(n_bands):
        core_band = core[band].tolist()
        c0 = core_band[0]
        uo_prev = c0
        u2_prev = c0 * one_minus_b5 if c0 >= eps else 0.0
        out[band, 0] = uo_prev

        base = c0
        delta = (core_band[1] - base) * inv_iter if n_time > 1 else -base * inv_iter
        ui_cur = base + delta
        for _ in range(n_inner_tail):
            uo_cur = ui_cur
            if uo_prev > u2_prev:
                uo2_a = uo_prev * b2 - u2_prev * b3
                if uo2_a >= ui_cur:
                    uo_cur = uo2_a
                u2_cur = uo_cur
                if ui_cur < uo_prev:
                    u22 = uo_prev * b0 - u2_prev * b1
                    if u22 <= uo_cur:
                        u2_cur = u22
            else:
                uo2_b = uo_prev * b4
                if uo2_b >= ui_cur:
                    uo_cur = uo2_b
                u2_cur = uo_cur

            if ui_cur >= uo_prev:
                if not ((ui_cur - uo_prev < eps) and (uo_cur <= u2_prev)):
                    u2_cur = (u2_prev - ui_cur) * b5 + ui_cur

            uo_prev = uo_cur
            u2_prev = u2_cur
            ui_cur += delta

        for time_idx in range(1, n_time):
            base = core_band[time_idx]
            delta = (core_band[time_idx + 1] - base) * inv_iter if time_idx + 1 < n_time else -base * inv_iter

            ui_cur = base
            uo_cur = ui_cur
            if uo_prev > u2_prev:
                uo2_a = uo_prev * b2 - u2_prev * b3
                if uo2_a >= ui_cur:
                    uo_cur = uo2_a
                u2_cur = uo_cur
                if ui_cur < uo_prev:
                    u22 = uo_prev * b0 - u2_prev * b1
                    if u22 <= uo_cur:
                        u2_cur = u22
            else:
                uo2_b = uo_prev * b4
                if uo2_b >= ui_cur:
                    uo_cur = uo2_b
                u2_cur = uo_cur

            if ui_cur >= uo_prev:
                if not ((ui_cur - uo_prev < eps) and (uo_cur <= u2_prev)):
                    u2_cur = (u2_prev - ui_cur) * b5 + ui_cur

            out[band, time_idx] = uo_cur
            uo_prev = uo_cur
            u2_prev = u2_cur

            ui_cur = base + delta
            for _ in range(n_inner_tail):
                uo_cur = ui_cur
                if uo_prev > u2_prev:
                    uo2_a = uo_prev * b2 - u2_prev * b3
                    if uo2_a >= ui_cur:
                        uo_cur = uo2_a
                    u2_cur = uo_cur
                    if ui_cur < uo_prev:
                        u22 = uo_prev * b0 - u2_prev * b1
                        if u22 <= uo_cur:
                            u2_cur = u22
                else:
                    uo2_b = uo_prev * b4
                    if uo2_b >= ui_cur:
                        uo_cur = uo2_b
                    u2_cur = uo_cur

                if ui_cur >= uo_prev:
                    if not ((ui_cur - uo_prev < eps) and (uo_cur <= u2_prev)):
                        u2_cur = (u2_prev - ui_cur) * b5 + ui_cur

                uo_prev = uo_cur
                u2_prev = u2_cur
                ui_cur += delta

    return out


NONLINEAR_DECAY_INTERPOLATION_STEPS = 12


def _nonlinear_decay(core_loudness: np.ndarray, nl_iter: int | None = None) -> np.ndarray:
    """Simulate nonlinear temporal decay used by the Zwicker time-varying path.

    Parameters
    ----------
    nl_iter : int, optional
        Number of linear-interpolation sub-steps between consecutive 2 kHz
        core-loudness frames.  Higher values give a smoother decay at the
        cost of more computation.  ``None`` uses the module-level default
        ``NONLINEAR_DECAY_INTERPOLATION_STEPS`` (12).  The original Zwicker
        reference uses 24; values >= 8 yield < 0.002 sone deviation.
    """

    sample_rate = 2000
    nl_iter = nl_iter if nl_iter is not None else NONLINEAR_DECAY_INTERPOLATION_STEPS
    t_short = 0.005
    t_long = 0.015
    t_var = 0.075
    core = np.asarray(core_loudness, dtype=np.float64)
    if core.ndim != 2 or core.shape[1] == 0:
        return core.copy()

    delta_t = 1.0 / (sample_rate * nl_iter)
    p_val = (t_var + t_long) / (t_var * t_short)
    q_val = 1.0 / (t_short * t_var)
    lambda_1 = -p_val / 2.0 + np.sqrt(p_val * p_val / 4.0 - q_val)
    lambda_2 = -p_val / 2.0 - np.sqrt(p_val * p_val / 4.0 - q_val)
    den = t_var * (lambda_1 - lambda_2)
    e1 = np.exp(lambda_1 * delta_t)
    e2 = np.exp(lambda_2 * delta_t)
    b0 = (e1 - e2) / den
    b1 = ((t_var * lambda_2 + 1.0) * e1 - (t_var * lambda_1 + 1.0) * e2) / den
    b2 = ((t_var * lambda_1 + 1.0) * e1 - (t_var * lambda_2 + 1.0) * e2) / den
    b3 = (t_var * lambda_1 + 1.0) * (t_var * lambda_2 + 1.0) * (e1 - e2) / den
    b4 = float(np.exp(-delta_t / t_long))
    b5 = float(np.exp(-delta_t / t_var))

    kernel = _nonlinear_decay_kernel if _NUMBA_AVAILABLE else _nonlinear_decay_kernel_python
    return kernel(
        np.ascontiguousarray(core),
        nl_iter,
        float(b0),
        float(b1),
        float(b2),
        float(b3),
        float(b4),
        float(b5),
    )


def _lowpass_interpolated(x: np.ndarray, tau: float, sample_rate: int) -> np.ndarray:
    """First-order low-pass with 24x linear interpolation for temporal weighting."""

    values = np.asarray(x, dtype=np.float64).reshape(-1)
    lp_iter = 24
    a1 = np.exp(-1.0 / (float(sample_rate) * lp_iter * float(tau)))
    b0 = 1.0 - a1
    delta = np.roll(values, -1)
    delta[-1] = 0.0
    delta = (delta - values) / float(lp_iter)
    ui = np.zeros((values.size, lp_iter), dtype=np.float64)
    ui[:, 0] = values
    for inner in range(1, lp_iter):
        ui[:, inner] = ui[:, inner - 1] + delta
    filtered = scipy_signal.lfilter([b0], [1.0, -a1], ui.reshape(-1))
    return filtered.reshape(values.size, lp_iter)[:, 0]


def _temporal_weighting(loudness_sone: np.ndarray) -> np.ndarray:
    """Apply the two time-constant temporal weighting from the time-varying path."""

    fast = _lowpass_interpolated(loudness_sone, LOUDNESS_TEMPORAL_FAST_TAU_S, 2000)
    slow = _lowpass_interpolated(loudness_sone, LOUDNESS_TEMPORAL_SLOW_TAU_S, 2000)
    return LOUDNESS_TEMPORAL_FAST_WEIGHT * fast + LOUDNESS_TEMPORAL_SLOW_WEIGHT * slow


class LoudnessAnalyzer:
    """Compute project-local loudness metrics for calibrated acoustic signals."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)

    def compute(
        self,
        recorded_signal: np.ndarray,
        *,
        method: str | LoudnessMethod = LoudnessMethod.PER_SEGMENT,
        field_type: str = "free",
        v2pa_factor: Optional[float] = None,
        reference_pressure_pa: float = P_REF,
        frame_size: Optional[int] = None,
        hop_length: Optional[int] = None,
        frame_duration_s: Optional[float] = None,
        hop_duration_s: Optional[float] = None,
        output_time_resolution_s: Optional[float] = None,
    ) -> LoudnessResult:
        method_value = method if isinstance(method, LoudnessMethod) else LoudnessMethod(str(method))
        field_value = str(field_type).lower().strip()
        if field_value not in {"free", "diffuse"}:
            raise ValueError(f"field_type must be 'free' or 'diffuse', got {field_type!r}")

        signal_v = _as_1d_float_array(recorded_signal)
        if signal_v.size == 0:
            raise ValueError("recorded_signal must not be empty")
        v2pa = 1.0 if v2pa_factor is None else float(v2pa_factor)
        signal_pa = signal_v * v2pa
        signal_pa, fs, was_resampled = _resample_if_needed(signal_pa, self.sample_rate)
        metadata_base: Dict[str, object] = {
            "method": method_value.value,
            "reference": "ISO 532-1:2017 / Zwicker",
            "field_type": field_value,
            "sample_rate_input_hz": self.sample_rate,
            "sample_rate_internal_hz": fs,
            "resampled_to_48k": was_resampled,
            "v2pa_factor": v2pa,
            "reference_pressure_pa": float(reference_pressure_pa),
        }

        if method_value == LoudnessMethod.TIME_VARYING_ISO532_1:
            third_octave_db, time_s_full = _third_octave_levels_time_varying(signal_pa, fs)
            core_loudness = _core_loudness_from_third_octaves_batch(third_octave_db, field_value)
            core_loudness = _nonlinear_decay(core_loudness)
            loudness_raw, specific_loudness_full, bark_axis = _specific_loudness_from_core(core_loudness)
            loudness_sone_full = _temporal_weighting(np.asarray(loudness_raw, dtype=np.float64))
            output_step_s = (
                float(output_time_resolution_s)
                if output_time_resolution_s is not None and float(output_time_resolution_s) > 0.0
                else LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S
            )
            dec_factor = max(1, int(round(output_step_s / LOUDNESS_INTERNAL_TIME_RESOLUTION_S)))
            time_s = time_s_full[::dec_factor]
            loudness_sone = loudness_sone_full[::dec_factor]
            loudness_level_phon = sones_to_phons(loudness_sone)
            specific_loudness = specific_loudness_full[:, ::dec_factor]
            skip_idx = int(np.searchsorted(time_s, LOUDNESS_TV_SUMMARY_SKIP_S, side="left"))
            summary = _summary_from_loudness(
                loudness_sone[skip_idx:], loudness_level_phon[skip_idx:]
            )
            metadata = dict(metadata_base)
            metadata.update({
                "implementation": "project_time_varying_zwicker_chain",
                "reference": "ISO 532-1:2017 / Zwicker, MoSQITo and SQAT used as public references",
                "implementation_level": "time_varying_chain_with_iso_filter_tables_and_full_slope_mapping_pending_iso_package_validation",
                "backend": "project_engineering",
                "internal_time_resolution_s": LOUDNESS_INTERNAL_TIME_RESOLUTION_S,
                "output_time_resolution_s": dec_factor * LOUDNESS_INTERNAL_TIME_RESOLUTION_S,
                "third_octave_filter": "iso_532_1_table_a1_a2_iir_coefficients",
                "specific_loudness_mapping": "full_zup_rns_usl_high_frequency_slope_mapping",
            })
            return LoudnessResult(
                time_s=time_s,
                loudness_sone=loudness_sone,
                loudness_level_phon=loudness_level_phon,
                specific_loudness=specific_loudness,
                bark_axis=bark_axis,
                summary=summary,
                metadata=metadata,
            )

        if frame_duration_s is not None:
            frame_size = int(round(float(frame_duration_s) * float(fs)))
        elif frame_size is None:
            frame_size = int(round(LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S * float(fs)))
        else:
            frame_size = int(frame_size)
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        frame_size = min(frame_size, int(signal_pa.size))
        if hop_duration_s is not None:
            hop = int(round(float(hop_duration_s) * float(fs)))
        elif hop_length is not None:
            hop = int(hop_length)
        else:
            hop = max(1, int(round(LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S * float(fs))))
        if hop <= 0:
            raise ValueError("hop_length must be positive")

        third_octave_db, time_s = _third_octave_levels_sliding_iso_frontend(
            signal_pa, fs, frame_size, hop
        )
        core_loudness = _core_loudness_from_third_octaves_batch(third_octave_db, field_value)
        loudness_raw, specific_loudness, bark_axis = _specific_loudness_from_core(core_loudness)
        loudness_sone = np.atleast_1d(np.asarray(loudness_raw, dtype=np.float64))
        loudness_level_phon = sones_to_phons(loudness_sone)

        summary = _summary_from_loudness(loudness_sone, loudness_level_phon)

        metadata = dict(metadata_base)
        metadata.update({
            "display_name": "segmented_loudness",
            "implementation": "project_sliding_iso_frontend_per_segment",
            "reference": "ISO 532-1:2017 / Zwicker (sliding window over ISO time-varying filterbank)",
            "implementation_level": "iso_tv_filterbank_sliding_window_power_average",
            "backend": "project_engineering",
            "frame_size": int(frame_size),
            "hop_length": int(hop),
            "frame_duration_s": float(frame_size) / float(fs),
            "hop_duration_s": float(hop) / float(fs),
            "time_axis": "window_center",
            "third_octave_estimator": "iso_532_1_tv_filterbank_windowed_power_mean",
            "bark_resolution": 0.1,
        })
        return LoudnessResult(
            time_s=time_s,
            loudness_sone=loudness_sone,
            loudness_level_phon=loudness_level_phon,
            specific_loudness=specific_loudness,
            bark_axis=bark_axis,
            summary=summary,
            metadata=metadata,
        )
