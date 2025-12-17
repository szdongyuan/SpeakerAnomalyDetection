import numpy as np

from base.pre_processing.mpeg_psychoacoustic_masking import (
    pick_maskers_mpeg1_model1,
    masking_threshold_from_maskers_mpeg1_model1,
)
from base.pre_processing.psychoacoustic_utils import freq_to_bark, absolute_threshold_of_hearing_db


def _pa_for_spl(spl_db: float) -> float:
    return float(20e-6 * np.power(10.0, float(spl_db) / 20.0))


def _db_to_power(db: float) -> float:
    return float(np.power(10.0, float(db) / 10.0))


def _power_to_db(power: float) -> float:
    power = float(max(power, 1e-30))
    return float(10.0 * np.log10(power))


def _build_frame_spls_with_code_path_floor(
    *,
    sample_rate: int,
    n_fft: int,
    fund_bin: int,
    fund_spl_db: float,
    harmonic_bin: int,
    harmonic_spl_db: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 1D SPL-per-bin frame exactly like the current PRB code path:
    - time-domain is bypassed; we synthesize per-bin 'amplitudes' in Pa directly
    - convert to dB SPL re 20 uPa with a small amplitude floor
    - clip SPL below 0 dB to 0 dB (current behavior in compute_perceptual_thd_batch)
    """
    sr = int(sample_rate)
    n_fft = int(n_fft)
    rfft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    frame_amplitudes = np.zeros_like(rfft_freqs, dtype=float)
    frame_amplitudes[int(fund_bin)] = _pa_for_spl(fund_spl_db)
    if harmonic_spl_db is not None:
        frame_amplitudes[int(harmonic_bin)] = _pa_for_spl(harmonic_spl_db)

    reference_pressure = 20e-6
    min_amplitude = 1e-12
    frame_spls = 20.0 * np.log10(np.maximum(frame_amplitudes / reference_pressure, min_amplitude))
    frame_spls = np.maximum(frame_spls, 0.0)

    bark_bins = freq_to_bark(rfft_freqs)
    partitions_per_bark = 3
    partition_index = np.clip(
        np.floor(bark_bins * float(partitions_per_bark)).astype(int),
        0,
        int(24 * partitions_per_bark),
    )
    return frame_spls, rfft_freqs, partition_index


def _masked_spl_after_threshold(harmonic_spl_db: float, threshold_db: float) -> float:
    harmonic_spl_db = float(harmonic_spl_db)
    threshold_db = float(threshold_db)
    if threshold_db <= 0.0:
        return max(harmonic_spl_db, 0.0)
    residual_power = _db_to_power(harmonic_spl_db) - _db_to_power(threshold_db)
    if residual_power <= 0.0:
        return 0.0
    return max(_power_to_db(residual_power), 0.0)


def _compute_threshold_and_audibility(
    *,
    sample_rate: int = 48000,
    n_fft: int = 9600,
    f0_hz: float = 200.0,
    harmonic_order: int = 5,
    fundamental_spl_db: float = 80.0,
    harmonic_spl_db: float | None = 70.0,
    tonal_peak_prominence_db: float = 7.0,
) -> tuple[float, float, float, bool]:
    """
    Returns: (threshold_db, masked_harmonic_spl_db, ath_db, audible)
    """
    sr = int(sample_rate)
    harm_freq_hz = float(f0_hz) * int(harmonic_order)

    # Choose FFT so 200 Hz and 1000 Hz land exactly on bins: with n_fft=9600 and sr=48k, bin spacing is 5 Hz.
    fund_bin = int(round(float(f0_hz) * n_fft / sr))
    harmonic_bin = int(round(harm_freq_hz * n_fft / sr))
    if abs(rharm := (harm_freq_hz - (harmonic_bin * sr / n_fft))) > 1e-9:
        raise AssertionError(f"harmonic frequency not bin-centered, residual={rharm}")

    frame_spls, rfft_freqs, partition_index = _build_frame_spls_with_code_path_floor(
        sample_rate=sr,
        n_fft=n_fft,
        fund_bin=fund_bin,
        fund_spl_db=fundamental_spl_db,
        harmonic_bin=harmonic_bin,
        harmonic_spl_db=harmonic_spl_db,
    )

    bark_bins = freq_to_bark(rfft_freqs)
    maskers = pick_maskers_mpeg1_model1(
        frame_spls,
        rfft_freqs,
        bark_bins=bark_bins,
        partition_index=partition_index,
        partitions_per_bark=3,
        forced_tonal_bins=np.array([fund_bin], dtype=int),
        tonal_peak_prominence_db=float(tonal_peak_prominence_db),
        min_over_ath_db=0.0,
        tonal_neighbor_merge_bins=1,
        max_tonal_per_partition=1,
        enable_noise_maskers=True,
        min_noise_over_ath_db=0.0,
    )

    target_freq = float(rfft_freqs[harmonic_bin])
    threshold_db = float(
        masking_threshold_from_maskers_mpeg1_model1(
            masker_freqs_hz=maskers.all_freqs_hz(),
            masker_levels_db=maskers.all_levels_db(),
            is_tonal=maskers.all_is_tonal(),
            target_freqs_hz=np.array([target_freq], dtype=float),
        )[0]
    )

    harmonic_level_db = float(harmonic_spl_db or 0.0)
    masked_harmonic_db = _masked_spl_after_threshold(harmonic_level_db, threshold_db)
    ath_db = float(absolute_threshold_of_hearing_db(np.array([target_freq], dtype=float))[0])
    audible = masked_harmonic_db > ath_db
    return threshold_db, masked_harmonic_db, ath_db, audible


def test_prb_masking_200hz_5th_harmonic_not_overkilled_at_reasonable_levels():
    # For a clearly audible 1 kHz tone (~70 dB SPL), fundamentals up to ~120 dB SPL
    # should not completely mask it.
    for fundamental_spl_db in (60.0, 80.0, 100.0, 120.0):
        thr, masked_db, ath_db, audible = _compute_threshold_and_audibility(
            f0_hz=200.0,
            harmonic_order=5,
            fundamental_spl_db=fundamental_spl_db,
            harmonic_spl_db=70.0,
        )
        assert thr < 70.0
        assert masked_db > ath_db
        assert audible is True


def test_prb_masking_200hz_5th_harmonic_can_be_fully_masked_when_fundamental_is_extreme():
    # With extremely high fundamental level, the masking threshold at 1 kHz should exceed the harmonic level.
    thr, masked_db, ath_db, audible = _compute_threshold_and_audibility(
        f0_hz=200.0,
        harmonic_order=5,
        fundamental_spl_db=140.0,
        harmonic_spl_db=70.0,
    )
    assert thr > 70.0
    assert masked_db == 0.0
    assert audible is False


def test_prb_masking_boundary_for_200hz_5th_harmonic_reasonable():
    # Find the minimum harmonic level that remains audible when the fundamental is 80 dB SPL.
    # This should be on the order of ~20 dB SPL, not tens of dB higher.
    min_audible = None
    for harmonic_spl_db in range(0, 61):
        _, _, _, audible = _compute_threshold_and_audibility(
            f0_hz=200.0,
            harmonic_order=5,
            fundamental_spl_db=80.0,
            harmonic_spl_db=float(harmonic_spl_db) if harmonic_spl_db > 0 else None,
        )
        if audible:
            min_audible = harmonic_spl_db
            break

    assert min_audible is not None
    assert 10 <= min_audible <= 30

