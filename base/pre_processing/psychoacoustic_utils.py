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


def freq_to_bark(f: float) -> float:
    """
    Convert frequency to Bark scale (critical band scale).

    Uses Traunmüller's formula for the Bark scale.

    Args:
        f: Frequency in Hz

    Returns:
        bark: Critical band rate in Bark
    """
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


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


def compute_bark_weight(bark_distance: float, function: str) -> float:
    """
    Compute weight based on Bark distance between masker and maskee.

    Weighting functions model how masking effectiveness decreases with
    frequency distance. Nearby frequencies mask more effectively.

    Args:
        bark_distance: Distance in Bark scale (will be normalized to positive)
        function: Weight function type
            - 'exponential': exp(-distance/2.0) [standard, recommended]
            - 'gaussian': exp(-(distance²)/2.0)
            - 'linear': max(0, 1 - distance/5.0)
            - 'inverse': 1/(1 + distance)

    Returns:
        weight: Weighting factor in [0, 1]

    Raises:
        ValueError: If function type is unknown
    """
    # Normalize to positive distance (Bark distance is symmetric)
    distance = abs(bark_distance)

    if function == 'exponential':
        return np.exp(-distance / 2.0)
    elif function == 'gaussian':
        return np.exp(-(distance ** 2) / 2.0)
    elif function == 'linear':
        return max(0.0, 1.0 - distance / 5.0)
    elif function == 'inverse':
        return 1.0 / (1.0 + distance)
    else:
        raise ValueError(f"Unknown weight function: {function}")


def apply_cumulative_masking(
    masker_freqs: np.ndarray,
    masker_spls: np.ndarray,
    maskee_freqs: np.ndarray,
    maskee_spls: np.ndarray,
    weight_function: str = 'exponential'
) -> np.ndarray:
    """
    Apply cumulative masking from multiple maskers to maskees.

    Uses distance-weighted power summation of masking thresholds.
    Each masker contributes to masking based on its Bark distance from
    the maskee, with closer maskers having more influence.

    Args:
        masker_freqs: Frequencies of all maskers (fundamental + harmonics 1-9) in Hz
        masker_spls: SPL levels of all maskers in dB
        maskee_freqs: Frequencies of harmonics being analyzed in Hz
        maskee_spls: SPL levels of harmonics being analyzed in dB
        weight_function: 'exponential', 'gaussian', 'linear', or 'inverse'

    Returns:
        masked_spls: Masked SPL values for each maskee in dB (n_maskees,)

    Raises:
        ValueError: If masker_freqs and masker_spls have different lengths
        ValueError: If maskee_freqs and maskee_spls have different lengths
    """
    # Validate input array lengths
    if len(masker_freqs) != len(masker_spls):
        raise ValueError(
            f"masker_freqs and masker_spls must have same length: "
            f"{len(masker_freqs)} != {len(masker_spls)}"
        )
    if len(maskee_freqs) != len(maskee_spls):
        raise ValueError(
            f"maskee_freqs and maskee_spls must have same length: "
            f"{len(maskee_freqs)} != {len(maskee_spls)}"
        )

    n_maskees = len(maskee_freqs)
    n_maskers = len(masker_freqs)
    masked_spls = np.zeros(n_maskees)

    # Convert frequencies to Bark scale
    masker_barks = np.array([freq_to_bark(f) for f in masker_freqs])
    maskee_barks = np.array([freq_to_bark(f) for f in maskee_freqs])

    # Compute weight matrix (n_maskees × n_maskers)
    weight_matrix = np.zeros((n_maskees, n_maskers))
    for i, maskee_bark in enumerate(maskee_barks):
        for j, masker_bark in enumerate(masker_barks):
            bark_distance = abs(maskee_bark - masker_bark)
            weight_matrix[i, j] = compute_bark_weight(bark_distance, weight_function)

    # Compute threshold matrix (n_maskees × n_maskers)
    threshold_matrix = np.zeros((n_maskees, n_maskers))
    for i, (maskee_freq, maskee_bark) in enumerate(zip(maskee_freqs, maskee_barks)):
        for j, (masker_freq, masker_spl) in enumerate(zip(masker_freqs, masker_spls)):
            threshold = compute_simultaneous_masking_threshold(
                masker_freq, masker_spl, maskee_freq
            )
            threshold_matrix[i, j] = threshold

    # Apply weighted power summation for each maskee
    for i in range(n_maskees):
        # Weighted power summation
        weights = weight_matrix[i, :]
        thresholds = threshold_matrix[i, :]

        # Convert to linear power, weight, sum, convert back
        powers = weights * np.power(10.0, thresholds / 10.0)
        total_power = np.sum(powers)

        # Guard against zero total power (would cause log10(0) -> -inf)
        if total_power <= 0:
            # No masking effect if total power is zero or negative
            masked_spls[i] = maskee_spls[i]
            continue

        combined_threshold = 10.0 * np.log10(total_power)

        # Apply masking
        if maskee_spls[i] <= combined_threshold:
            # Fully masked
            masked_spls[i] = 0.0
        else:
            # Partial masking (30% of threshold)
            # Only apply reduction if threshold is positive
            if combined_threshold > 0:
                masking_reduction = combined_threshold * 0.3
                masked_spls[i] = max(maskee_spls[i] - masking_reduction, 0.0)
            else:
                # No masking effect if threshold is negative or zero
                masked_spls[i] = maskee_spls[i]

    return masked_spls
