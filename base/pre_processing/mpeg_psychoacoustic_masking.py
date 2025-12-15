"""
MPEG-style psychoacoustic masking (Model-1-like) utilities.

This module builds a masking threshold curve T(f) from a full-spectrum SPL frame
using:
  - Tonal maskers (peak-picked and de-duplicated per critical band)
  - Noise maskers (residual energy per critical band)
  - Bark-domain spreading and power-domain threshold summation

It is intended to support PRB by computing masking at harmonic frequencies while
keeping the masking model consistent with full-spectrum psychoacoustics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from base.pre_processing.psychoacoustic_utils import freq_to_bark, absolute_threshold_of_hearing_db


@dataclass(frozen=True)
class MaskerSet:
    tonal_freqs_hz: np.ndarray
    tonal_levels_db: np.ndarray
    noise_freqs_hz: np.ndarray
    noise_levels_db: np.ndarray

    def all_freqs_hz(self) -> np.ndarray:
        if self.tonal_freqs_hz.size == 0:
            return self.noise_freqs_hz
        if self.noise_freqs_hz.size == 0:
            return self.tonal_freqs_hz
        return np.concatenate([self.tonal_freqs_hz, self.noise_freqs_hz])

    def all_levels_db(self) -> np.ndarray:
        if self.tonal_levels_db.size == 0:
            return self.noise_levels_db
        if self.noise_levels_db.size == 0:
            return self.tonal_levels_db
        return np.concatenate([self.tonal_levels_db, self.noise_levels_db])

    def all_is_tonal(self) -> np.ndarray:
        if self.tonal_levels_db.size == 0:
            return np.zeros(self.noise_levels_db.size, dtype=bool)
        if self.noise_levels_db.size == 0:
            return np.ones(self.tonal_levels_db.size, dtype=bool)
        return np.concatenate(
            [np.ones(self.tonal_levels_db.size, dtype=bool), np.zeros(self.noise_levels_db.size, dtype=bool)]
        )


def _db_to_power(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 10.0)


def _power_to_db(power: np.ndarray) -> np.ndarray:
    power = np.maximum(power, 1e-30)
    return 10.0 * np.log10(power)


def _critical_band_index_from_bark(bark: np.ndarray) -> np.ndarray:
    # MPEG-style critical bands span roughly 0..24 Bark.
    idx = np.floor(bark).astype(int)
    return np.clip(idx, 0, 24)


def _partition_index_from_bark(bark: np.ndarray, *, partitions_per_bark: int = 3) -> np.ndarray:
    """
    Map Bark values to MPEG-style partitions (roughly 0.33 Bark for partitions_per_bark=3).
    """
    if partitions_per_bark <= 0:
        raise ValueError(f"partitions_per_bark must be positive, got {partitions_per_bark}")
    idx = np.floor(np.asarray(bark, dtype=float) * float(partitions_per_bark)).astype(int)
    return np.clip(idx, 0, int(24 * partitions_per_bark))


def _spreading_function_mpeg1_model1(dz_bark: np.ndarray) -> np.ndarray:
    """
    MPEG-1 psychoacoustic model 1 spreading function approximation in dB (Johnston-style).

    The exact MPEG reference includes additional details (partition tables, further decimation rules).
    This captures the characteristic asymmetric Bark-domain spread used by model 1.
    """
    dz = np.asarray(dz_bark, dtype=float)
    x = dz + 0.474
    return 15.81 + 7.5 * x - 17.5 * np.sqrt(1.0 + x * x)


def pick_maskers_model1_like(
    spl_spectrum_db: np.ndarray,
    freqs_hz: np.ndarray,
    bark_bins: np.ndarray | None = None,
    band_index: np.ndarray | None = None,
    *,
    min_over_ath_db: float = 5.0,
    tonal_neighbor_merge_bins: int = 1,
    max_tonal_per_band: int = 1,
    enable_noise_maskers: bool = True,
    min_noise_over_ath_db: float = 0.0,
) -> MaskerSet:
    """
    Pick tonal and noise maskers from a full-spectrum SPL frame.

    This is "Model-1-like" in the sense that it:
      - selects tonal peaks (local maxima) and de-duplicates per critical band
      - uses residual (non-tonal) energy per critical band as a noise masker

    Args:
        spl_spectrum_db: (n_bins,) SPL per FFT bin (dB)
        freqs_hz: (n_bins,) frequency axis (Hz), matching spl_spectrum_db
        bark_bins: Optional precomputed Bark values for freqs_hz
        band_index: Optional precomputed critical band indices for freqs_hz
        min_over_ath_db: tonal peak must exceed ATH by this margin
        tonal_neighbor_merge_bins: merge +/-N bins into the tonal masker power
        max_tonal_per_band: keep up to N strongest tonal maskers per band
        enable_noise_maskers: whether to add noise maskers per band
        min_noise_over_ath_db: noise masker must exceed ATH(band center) by this margin

    Returns:
        MaskerSet with tonal and noise masker arrays.
    """
    spl = np.asarray(spl_spectrum_db, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    if spl.ndim != 1 or freqs.ndim != 1 or spl.shape[0] != freqs.shape[0]:
        raise ValueError("spl_spectrum_db and freqs_hz must be 1D arrays with the same length")

    if bark_bins is None:
        bark_bins = freq_to_bark(freqs)
    if band_index is None:
        band_index = _critical_band_index_from_bark(bark_bins)

    # Absolute threshold of hearing for gating candidate maskers
    ath_db = absolute_threshold_of_hearing_db(freqs)

    # Local maxima peak picking (exclude DC and Nyquist bins)
    if spl.size < 3:
        return MaskerSet(
            tonal_freqs_hz=np.array([], dtype=float),
            tonal_levels_db=np.array([], dtype=float),
            noise_freqs_hz=np.array([], dtype=float),
            noise_levels_db=np.array([], dtype=float),
        )

    center = spl[1:-1]
    left = spl[:-2]
    right = spl[2:]
    peak_mask = (center > left) & (center >= right)
    peak_bins = np.nonzero(peak_mask)[0] + 1
    if peak_bins.size == 0:
        peak_bins = np.array([], dtype=int)

    # Gate peaks by ATH margin
    peak_bins = peak_bins[spl[peak_bins] > (ath_db[peak_bins] + min_over_ath_db)]

    # Convert spectrum into power for summations
    power = _db_to_power(spl)
    residual_power = power.copy()

    # Build tonal candidates with neighborhood power merge
    tonal_candidates = []
    tonal_used_bins = np.zeros(spl.size, dtype=bool)
    for k in peak_bins:
        lo = max(int(k) - tonal_neighbor_merge_bins, 0)
        hi = min(int(k) + tonal_neighbor_merge_bins + 1, spl.size)
        region_power = float(np.sum(power[lo:hi]))
        if region_power <= 0.0:
            continue
        tonal_candidates.append((int(k), _power_to_db(region_power)))
        tonal_used_bins[lo:hi] = True
        residual_power[lo:hi] = 0.0

    if not tonal_candidates:
        tonal_bins = np.array([], dtype=int)
        tonal_levels_db = np.array([], dtype=float)
    else:
        # De-duplicate per critical band: keep strongest peaks per band
        tonal_bins = np.array([k for k, _ in tonal_candidates], dtype=int)
        tonal_levels_db = np.array([lvl for _, lvl in tonal_candidates], dtype=float)
        tonal_bands = band_index[tonal_bins]

        kept_mask = np.zeros(tonal_bins.size, dtype=bool)
        for b in np.unique(tonal_bands):
            idx = np.where(tonal_bands == b)[0]
            if idx.size == 0:
                continue
            # Keep top-N by level
            order = np.argsort(tonal_levels_db[idx])[::-1]
            keep_n = idx[order[:max_tonal_per_band]]
            kept_mask[keep_n] = True

        tonal_bins = tonal_bins[kept_mask]
        tonal_levels_db = tonal_levels_db[kept_mask]

    tonal_freqs = freqs[tonal_bins] if tonal_bins.size else np.array([], dtype=float)

    # Noise maskers: residual energy per critical band
    noise_freqs = []
    noise_levels = []
    if enable_noise_maskers:
        for b in range(0, 25):
            band_bins = np.where(band_index == b)[0]
            if band_bins.size == 0:
                continue
            band_power = float(np.sum(residual_power[band_bins]))
            if band_power <= 0.0:
                continue

            # Choose a representative frequency for the band (power-weighted centroid)
            band_freqs = freqs[band_bins]
            w = residual_power[band_bins]
            freq_centroid = float(np.sum(band_freqs * w) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(band_freqs))

            level_db = float(_power_to_db(band_power))
            ath_at_centroid = float(absolute_threshold_of_hearing_db(np.array([freq_centroid]))[0])
            if level_db > (ath_at_centroid + min_noise_over_ath_db):
                noise_freqs.append(freq_centroid)
                noise_levels.append(level_db)

    return MaskerSet(
        tonal_freqs_hz=np.asarray(tonal_freqs, dtype=float),
        tonal_levels_db=np.asarray(tonal_levels_db, dtype=float),
        noise_freqs_hz=np.asarray(noise_freqs, dtype=float),
        noise_levels_db=np.asarray(noise_levels, dtype=float),
    )


def masking_threshold_curve_model1_like(
    spl_spectrum_db: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    bark_bins: np.ndarray | None = None,
    band_index: np.ndarray | None = None,
    min_over_ath_db: float = 5.0,
    tonal_neighbor_merge_bins: int = 1,
    max_tonal_per_band: int = 1,
    enable_noise_maskers: bool = True,
    min_noise_over_ath_db: float = 0.0,
    max_total_maskers: int = 64,
    include_ath: bool = True,
) -> np.ndarray:
    """
    Compute a full masking threshold curve T(f_bin) in dB SPL for one frame.

    Returns:
        threshold_db: (n_bins,) array.
            If include_ath=True (default), the result is max(ATH, combined masking thresholds).
            If include_ath=False, the result contains masking thresholds due to maskers only.
    """
    spl = np.asarray(spl_spectrum_db, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    if bark_bins is None:
        bark_bins = freq_to_bark(freqs)
    if band_index is None:
        band_index = _critical_band_index_from_bark(bark_bins)

    ath_db = absolute_threshold_of_hearing_db(freqs) if include_ath else None

    maskers = pick_maskers_model1_like(
        spl,
        freqs,
        bark_bins=bark_bins,
        band_index=band_index,
        min_over_ath_db=min_over_ath_db,
        tonal_neighbor_merge_bins=tonal_neighbor_merge_bins,
        max_tonal_per_band=max_tonal_per_band,
        enable_noise_maskers=enable_noise_maskers,
        min_noise_over_ath_db=min_noise_over_ath_db,
    )

    masker_freqs = maskers.all_freqs_hz()
    masker_levels = maskers.all_levels_db()
    is_tonal = maskers.all_is_tonal()

    if masker_freqs.size == 0:
        return absolute_threshold_of_hearing_db(freqs) if include_ath else np.zeros_like(freqs, dtype=float)

    # Keep the strongest maskers only
    if masker_levels.size > max_total_maskers:
        idx = np.argsort(masker_levels)[-max_total_maskers:]
        masker_freqs = masker_freqs[idx]
        masker_levels = masker_levels[idx]
        is_tonal = is_tonal[idx]

    combined_db = masking_threshold_from_maskers_model1_like(
        masker_freqs_hz=masker_freqs,
        masker_levels_db=masker_levels,
        is_tonal=is_tonal,
        target_freqs_hz=freqs,
        target_bark=bark_bins,
    )

    if include_ath:
        return np.maximum(ath_db, combined_db)
    return combined_db


def masking_threshold_from_maskers_model1_like(
    *,
    masker_freqs_hz: np.ndarray,
    masker_levels_db: np.ndarray,
    is_tonal: np.ndarray,
    target_freqs_hz: np.ndarray,
    target_bark: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a combined masking threshold in dB SPL at arbitrary target frequencies.

    This returns the masking threshold due to the provided maskers only (no ATH max).
    """
    masker_freqs = np.asarray(masker_freqs_hz, dtype=float).reshape(-1)
    masker_levels = np.asarray(masker_levels_db, dtype=float).reshape(-1)
    tonal_flags = np.asarray(is_tonal, dtype=bool).reshape(-1)
    targets = np.asarray(target_freqs_hz, dtype=float).reshape(-1)

    if masker_freqs.size == 0:
        return np.zeros_like(targets, dtype=float)
    if masker_freqs.size != masker_levels.size or masker_freqs.size != tonal_flags.size:
        raise ValueError("masker_freqs_hz, masker_levels_db, and is_tonal must have the same length")

    if target_bark is None:
        target_bark = freq_to_bark(targets)
    else:
        target_bark = np.asarray(target_bark, dtype=float).reshape(-1)
        if target_bark.size != targets.size:
            raise ValueError("target_bark must have the same length as target_freqs_hz")

    # Bark values for maskers and targets
    masker_bark = freq_to_bark(masker_freqs)
    dz = target_bark[None, :] - masker_bark[:, None]
    abs_dz = np.abs(dz)

    # Spreading slopes (MPEG-like heuristic): steeper upward masking than downward
    # Upward spread (dz>=0): about -27..-30 dB/Bark
    slopes_up = -27.0 - 3.0 * (masker_freqs / 5000.0)
    slopes_up = np.maximum(slopes_up, -30.0)
    # Downward spread (dz<0): gentler, about -24..-20 dB/Bark
    slopes_down = -24.0 + 3.0 * (masker_freqs / 5000.0)
    slopes_down = np.minimum(slopes_down, -20.0)
    slopes = np.where(dz >= 0.0, slopes_up[:, None], slopes_down[:, None])

    # MPEG-like tonal vs noise masker correction (in dB)
    tonal_corr = -0.275 * masker_bark - 6.0
    noise_corr = -0.175 * masker_bark - 2.0
    corr = np.where(tonal_flags, tonal_corr, noise_corr).astype(float)

    # Level-dependent offset (stronger masker => stronger masking)
    offsets = np.where(masker_levels >= 60.0, 3.0, np.where(masker_levels >= 40.0, 6.0, 10.0))

    thr = (masker_levels + corr - offsets)[:, None] + slopes * abs_dz
    thr = np.maximum(thr, 0.0)

    combined_power = np.sum(_db_to_power(thr), axis=0)
    return _power_to_db(combined_power)


def pick_maskers_mpeg1_model1(
    spl_spectrum_db: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    bark_bins: np.ndarray | None = None,
    partition_index: np.ndarray | None = None,
    partitions_per_bark: int = 3,
    forced_tonal_bins: np.ndarray | None = None,
    tonal_peak_prominence_db: float = 7.0,
    min_over_ath_db: float = 0.0,
    tonal_neighbor_merge_bins: int = 1,
    max_tonal_per_partition: int = 1,
    enable_noise_maskers: bool = True,
    min_noise_over_ath_db: float = 0.0,
) -> MaskerSet:
    """
    Pick tonal and noise maskers using an MPEG-1 psychoacoustic model-1 style workflow.

    Differences vs `pick_maskers_model1_like`:
      - Uses finer Bark partitions (~0.33 Bark by default).
      - Uses MPEG-like tonal detection (local max + neighbor prominence checks).
      - Computes noise maskers from residual energy per partition using bincount.
    """
    spl = np.asarray(spl_spectrum_db, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    if spl.ndim != 1 or freqs.ndim != 1 or spl.shape[0] != freqs.shape[0]:
        raise ValueError("spl_spectrum_db and freqs_hz must be 1D arrays with the same length")
    if spl.size < 3:
        return MaskerSet(
            tonal_freqs_hz=np.array([], dtype=float),
            tonal_levels_db=np.array([], dtype=float),
            noise_freqs_hz=np.array([], dtype=float),
            noise_levels_db=np.array([], dtype=float),
        )

    if bark_bins is None:
        bark_bins = freq_to_bark(freqs)
    bark_bins = np.asarray(bark_bins, dtype=float)
    if bark_bins.shape != freqs.shape:
        raise ValueError("bark_bins must have the same shape as freqs_hz")

    if partition_index is None:
        partition_index = _partition_index_from_bark(bark_bins, partitions_per_bark=partitions_per_bark)
    partition_index = np.asarray(partition_index, dtype=int)
    if partition_index.shape != freqs.shape:
        raise ValueError("partition_index must have the same shape as freqs_hz")

    n_partitions = int(np.max(partition_index)) + 1 if partition_index.size else 0
    if n_partitions <= 0:
        return MaskerSet(
            tonal_freqs_hz=np.array([], dtype=float),
            tonal_levels_db=np.array([], dtype=float),
            noise_freqs_hz=np.array([], dtype=float),
            noise_levels_db=np.array([], dtype=float),
        )

    ath_db = absolute_threshold_of_hearing_db(freqs)

    # Tonal peak picking with neighbor prominence (MPEG-like).
    peak_bins = np.array([], dtype=int)
    if spl.size >= 5:
        mid = np.arange(2, spl.size - 2)
        is_peak = (
            (spl[mid] > spl[mid - 1])
            & (spl[mid] >= spl[mid + 1])
            & ((spl[mid] - spl[mid - 1]) >= tonal_peak_prominence_db)
            & ((spl[mid] - spl[mid + 1]) >= tonal_peak_prominence_db)
            & ((spl[mid] - spl[mid - 2]) >= tonal_peak_prominence_db)
            & ((spl[mid] - spl[mid + 2]) >= tonal_peak_prominence_db)
        )
        peak_bins = mid[is_peak]
    else:
        center = spl[1:-1]
        left = spl[:-2]
        right = spl[2:]
        peak_mask = (center > left) & (center >= right)
        peak_bins = np.nonzero(peak_mask)[0] + 1

    # Force-insert known tonal bins (e.g., fundamental bin), and treat them as tonal components.
    if forced_tonal_bins is not None:
        forced = np.asarray(forced_tonal_bins, dtype=int).reshape(-1)
        forced = forced[(forced >= 1) & (forced < spl.size - 1)]
        if forced.size:
            if peak_bins.size:
                peak_bins = np.unique(np.concatenate([peak_bins, forced]))
            else:
                peak_bins = np.unique(forced)

    if peak_bins.size == 0 and not enable_noise_maskers:
        return MaskerSet(
            tonal_freqs_hz=np.array([], dtype=float),
            tonal_levels_db=np.array([], dtype=float),
            noise_freqs_hz=np.array([], dtype=float),
            noise_levels_db=np.array([], dtype=float),
        )

    # ATH gate
    peak_bins = peak_bins[spl[peak_bins] > (ath_db[peak_bins] + min_over_ath_db)]

    power = _db_to_power(spl)
    residual_power = power.copy()

    # Build tonal candidates by merging neighborhood power and removing from residual.
    tonal_candidates_bins = []
    tonal_candidates_levels = []
    if peak_bins.size:
        for k in peak_bins:
            lo = max(int(k) - tonal_neighbor_merge_bins, 0)
            hi = min(int(k) + tonal_neighbor_merge_bins + 1, spl.size)
            region_power = float(np.sum(power[lo:hi]))
            if region_power <= 0.0:
                continue
            tonal_candidates_bins.append(int(k))
            tonal_candidates_levels.append(float(_power_to_db(region_power)))
            residual_power[lo:hi] = 0.0

    tonal_bins = np.asarray(tonal_candidates_bins, dtype=int)
    tonal_levels_db = np.asarray(tonal_candidates_levels, dtype=float)
    if tonal_bins.size:
        tonal_partitions = partition_index[tonal_bins]
        kept = np.zeros(tonal_bins.size, dtype=bool)
        for p in np.unique(tonal_partitions):
            idx = np.where(tonal_partitions == p)[0]
            if idx.size == 0:
                continue
            order = np.argsort(tonal_levels_db[idx])[::-1]
            keep_idx = idx[order[:max_tonal_per_partition]]
            kept[keep_idx] = True
        tonal_bins = tonal_bins[kept]
        tonal_levels_db = tonal_levels_db[kept]

    tonal_freqs = freqs[tonal_bins] if tonal_bins.size else np.array([], dtype=float)

    # Noise maskers from residual energy per partition (bincount for performance).
    noise_freqs = np.array([], dtype=float)
    noise_levels_db = np.array([], dtype=float)
    if enable_noise_maskers:
        part_power = np.bincount(partition_index, weights=residual_power, minlength=n_partitions).astype(float)
        part_freq_power = np.bincount(partition_index, weights=residual_power * freqs, minlength=n_partitions).astype(float)
        valid = part_power > 0.0
        if np.any(valid):
            centroids = np.zeros(n_partitions, dtype=float)
            centroids[valid] = part_freq_power[valid] / part_power[valid]
            part_levels_db = np.full(n_partitions, -np.inf, dtype=float)
            part_levels_db[valid] = _power_to_db(part_power[valid])

            # Select only local maxima of the noise-energy partitions to avoid over-counting
            # many highly correlated noise maskers (closer to MPEG model behavior).
            if n_partitions >= 3:
                center = part_levels_db[1:-1]
                left = part_levels_db[:-2]
                right = part_levels_db[2:]
                peak_mask = (center > left) & (center >= right)
                peak_parts = np.nonzero(peak_mask)[0] + 1
            else:
                peak_parts = np.where(valid)[0]

            peak_parts = peak_parts[valid[peak_parts]]
            if peak_parts.size:
                freqs_centroid = centroids[peak_parts]
                levels = part_levels_db[peak_parts]
                ath_centroid = absolute_threshold_of_hearing_db(freqs_centroid)
                keep = levels > (ath_centroid + min_noise_over_ath_db)
                noise_freqs = freqs_centroid[keep]
                noise_levels_db = levels[keep]

    return MaskerSet(
        tonal_freqs_hz=np.asarray(tonal_freqs, dtype=float),
        tonal_levels_db=np.asarray(tonal_levels_db, dtype=float),
        noise_freqs_hz=np.asarray(noise_freqs, dtype=float),
        noise_levels_db=np.asarray(noise_levels_db, dtype=float),
    )


def masking_threshold_from_maskers_mpeg1_model1(
    *,
    masker_freqs_hz: np.ndarray,
    masker_levels_db: np.ndarray,
    is_tonal: np.ndarray,
    target_freqs_hz: np.ndarray,
    target_bark: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a combined masking threshold in dB SPL at arbitrary target frequencies (maskers only, no ATH max).
    """
    masker_freqs = np.asarray(masker_freqs_hz, dtype=float).reshape(-1)
    masker_levels = np.asarray(masker_levels_db, dtype=float).reshape(-1)
    tonal_flags = np.asarray(is_tonal, dtype=bool).reshape(-1)
    targets = np.asarray(target_freqs_hz, dtype=float).reshape(-1)

    if masker_freqs.size == 0:
        return np.zeros_like(targets, dtype=float)
    if masker_freqs.size != masker_levels.size or masker_freqs.size != tonal_flags.size:
        raise ValueError("masker_freqs_hz, masker_levels_db, and is_tonal must have the same length")

    if target_bark is None:
        target_bark = freq_to_bark(targets)
    else:
        target_bark = np.asarray(target_bark, dtype=float).reshape(-1)
        if target_bark.size != targets.size:
            raise ValueError("target_bark must have the same length as target_freqs_hz")

    masker_bark = freq_to_bark(masker_freqs)
    dz = target_bark[None, :] - masker_bark[:, None]

    sf = _spreading_function_mpeg1_model1(dz)

    # Tonal vs noise masker offsets (MPEG model-1 style).
    tonal_base = masker_levels - 0.275 * masker_bark - 6.025
    noise_base = masker_levels - 0.175 * masker_bark - 2.025
    base = np.where(tonal_flags, tonal_base, noise_base).astype(float)

    thr = base[:, None] + sf
    thr = np.maximum(thr, 0.0)

    combined_power = np.sum(_db_to_power(thr), axis=0)
    return _power_to_db(combined_power)
