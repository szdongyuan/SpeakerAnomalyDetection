# base/pre_processing/psychoacoustic_utils.py
"""
Psychoacoustic utilities for perceptual audio analysis.

Implements ISO 226:2003 equal-loudness contours and simultaneous masking models.
"""
import numpy as np
from scipy.interpolate import interp1d


# ISO 226:2003 equal-loudness contour data (40 phon curve)
# Format: (frequency_Hz, SPL_dB)
ISO_226_40_PHON = np.array([
    [20, 84.9], [25, 79.8], [31.5, 74.7], [40, 69.8], [50, 65.4],
    [63, 61.2], [80, 57.2], [100, 53.7], [125, 50.7], [160, 47.9],
    [200, 45.5], [250, 43.4], [315, 41.5], [400, 39.7], [500, 38.2],
    [630, 36.8], [800, 35.6], [1000, 40.0], [1250, 38.9], [1600, 37.7],
    [2000, 36.6], [2500, 35.7], [3150, 35.0], [4000, 34.6], [5000, 34.8],
    [6300, 35.5], [8000, 37.0], [10000, 40.0], [12500, 47.5]
])

# Pre-compute interpolation function for SPL to phons conversion
_freq_to_spl_40_phon = interp1d(
    ISO_226_40_PHON[:, 0], ISO_226_40_PHON[:, 1],
    kind='cubic', fill_value='extrapolate'
)


def spl_to_phons(frequencies: np.ndarray, spl_values: np.ndarray) -> np.ndarray:
    """
    Convert SPL (dB) to phons using ISO 226:2003 equal-loudness contours.

    Uses the 40-phon curve as reference and linearly scales for other loudness levels.

    Args:
        frequencies: Frequency values in Hz (n,)
        spl_values: SPL values in dB (n,)

    Returns:
        phons: Perceived loudness in phons (n,)
    """
    # Get SPL required for 40 phons at each frequency
    spl_at_40_phon = _freq_to_spl_40_phon(frequencies)

    # Linear approximation: phons = 40 + (spl - spl_at_40_phon)
    # This assumes equal loudness contours are parallel (good approximation for moderate levels)
    # Apply a compression factor to account for contour convergence
    phon_diff = spl_values - spl_at_40_phon

    # Compression factor: equal-loudness contours converge at higher loudness levels
    # At 1 kHz, no compression (factor = 1.0)
    # At frequencies far from 1 kHz, more compression
    log_freq_ratio = np.abs(np.log10(frequencies / 1000.0))
    compression = 1.0 - 0.21 * log_freq_ratio
    compression = np.clip(compression, 0.75, 1.0)

    phons = 40.0 + (phon_diff * compression)

    # Clip negative phons to 0 (below threshold of hearing)
    phons = np.maximum(phons, 0.0)

    return phons


def compute_simultaneous_masking_threshold(
    masker_freq: float,
    masker_level: float,
    maskee_freq: float
) -> float:
    """
    Compute simultaneous masking threshold using simplified spreading function.

    Based on psychoacoustic masking models (Zwicker & Fastl, 1999).
    Masking spreads asymmetrically: stronger upward in frequency than downward.

    Args:
        masker_freq: Frequency of masker in Hz
        masker_level: SPL of masker in dB
        maskee_freq: Frequency of signal being masked in Hz

    Returns:
        threshold: Masking threshold in dB SPL at maskee_freq
    """
    # Convert frequencies to Bark scale (critical band scale)
    def freq_to_bark(f):
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    masker_bark = freq_to_bark(masker_freq)
    maskee_bark = freq_to_bark(maskee_freq)
    bark_distance = maskee_bark - masker_bark

    # Spreading function (asymmetric slopes)
    if bark_distance >= 0:
        # Upward spread: -27 dB/Bark slope
        slope = -27.0
    else:
        # Downward spread: -24 dB/Bark slope (gentler)
        slope = -24.0

    # Masking threshold = masker_level + slope * distance - offset
    # Offset accounts for masker bandwidth and other factors
    offset = 6.0  # Typical value for tonal masker

    threshold = masker_level + slope * abs(bark_distance) - offset

    # Threshold cannot be negative
    threshold = max(threshold, 0.0)

    return threshold


def apply_masking(
    fundamental_freq: float,
    fundamental_spl: float,
    harmonic_freqs: np.ndarray,
    harmonic_spls: np.ndarray
) -> np.ndarray:
    """
    Apply simultaneous masking from fundamental to higher harmonics.

    For each harmonic:
    1. Compute masking threshold from fundamental
    2. If harmonic SPL < threshold, set to 0 (fully masked)
    3. Otherwise, reduce harmonic SPL by masking effect

    Args:
        fundamental_freq: Fundamental frequency in Hz
        fundamental_spl: Fundamental SPL in dB
        harmonic_freqs: Harmonic frequencies in Hz (n,)
        harmonic_spls: Harmonic SPLs in dB (n,)

    Returns:
        masked_spls: Harmonic SPLs after masking in dB (n,)
    """
    masked_spls = np.zeros_like(harmonic_spls)

    for i, (h_freq, h_spl) in enumerate(zip(harmonic_freqs, harmonic_spls)):
        # Compute masking threshold from fundamental
        threshold = compute_simultaneous_masking_threshold(
            fundamental_freq, fundamental_spl, h_freq
        )

        # If harmonic is below threshold, fully masked
        if h_spl <= threshold:
            masked_spls[i] = 0.0
        else:
            # Harmonic is above threshold, partially audible
            # Apply masking reduction based on threshold
            if threshold > 0:
                masking_reduction = threshold * 0.3  # 30% of threshold
                masked_spls[i] = max(h_spl - masking_reduction, 0.0)
            else:
                # Even with zero threshold, apply minimal masking from strong fundamental
                # This accounts for general masking effects beyond critical bands
                if fundamental_spl > 60.0:  # Strong fundamental
                    # Apply small reduction (1-5 dB) based on fundamental level
                    general_masking = (fundamental_spl - 60.0) * 0.05  # 0.05 dB per dB above 60
                    masked_spls[i] = max(h_spl - general_masking, 0.0)
                else:
                    masked_spls[i] = h_spl

    return masked_spls
