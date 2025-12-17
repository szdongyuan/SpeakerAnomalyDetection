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


def absolute_threshold_of_hearing_db(frequencies_hz: np.ndarray) -> np.ndarray:
    """
    Approximate absolute threshold of hearing (ATH) in dB SPL as a function of frequency.

    This is a commonly used analytic approximation of the minimum audible SPL curve
    (e.g., in psychoacoustic models for perceptual audio coding).

    Args:
        frequencies_hz: Frequencies in Hz (array-like)

    Returns:
        ath_db_spl: ATH values in dB SPL, same shape as input.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    f = np.maximum(f, 1.0)  # avoid divide-by-zero at DC
    f_khz = f / 1000.0

    # Standard ATH approximation:
    # 3.64*(f/1k)^(-0.8) - 6.5*exp(-0.6*(f/1k - 3.3)^2) + 1e-3*(f/1k)^4
    ath = (
        3.64 * np.power(f_khz, -0.8)
        - 6.5 * np.exp(-0.6 * np.square(f_khz - 3.3))
        + 1e-3 * np.power(f_khz, 4.0)
    )

    # We clamp to a reasonable range to avoid pathological values outside the
    # typical validity region of the approximation.
    return np.clip(ath, 0.0, 120.0)


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
    Compute simultaneous masking threshold using improved spreading function.

    Based on:
    - Zwicker & Fastl (1999) psychoacoustic masking model
    - MPEG psychoacoustic model 1 (ISO/IEC 11172-3)
    - Terhardt (1979) frequency-dependent masking slopes

    Improvements over simplified model:
    1. Frequency-dependent slopes (varies with masker frequency)
    2. Level-dependent offset (stronger maskers have lower offset)
    3. Asymmetric spreading (different slopes above/below)

    Args:
        masker_freq: Frequency of masker in Hz
        masker_level: SPL of masker in dB
        maskee_freq: Frequency of signal being masked in Hz

    Returns:
        threshold: Masking threshold in dB SPL at maskee_freq

    References:
        - Zwicker & Fastl (1999) "Psychoacoustics: Facts and Models"
        - ISO/IEC 11172-3 (MPEG-1 Audio)
        - Terhardt (1979) "Calculating virtual pitch"
    """
    # Convert frequencies to Bark scale (critical band scale)
    masker_bark = freq_to_bark(masker_freq)
    maskee_bark = freq_to_bark(maskee_freq)
    bark_distance = maskee_bark - masker_bark

    # Frequency-dependent spreading function slopes
    # Based on Terhardt (1979) and MPEG psychoacoustic model
    if bark_distance >= 0:
        # Upward spread (masker masks higher frequencies)
        # Slope becomes steeper at higher frequencies
        # Low freq: -27 dB/Bark, High freq: -30 dB/Bark
        slope = -27.0 - 3.0 * (masker_freq / 5000.0)
        slope = max(slope, -30.0)  # Limit to -30 dB/Bark
    else:
        # Downward spread (masker masks lower frequencies)
        # MPEG model uses positive slope for downward masking
        # But Zwicker uses negative (gentler attenuation)
        # We use Zwicker's approach for consistency
        slope = -24.0 + 3.0 * (masker_freq / 5000.0)
        slope = min(slope, -20.0)  # Limit to -20 dB/Bark (gentler)

    # Level-dependent offset (based on MPEG psychoacoustic model)
    # Stronger maskers (high SPL) have lower offset → stronger masking
    # Weaker maskers (low SPL) have higher offset → weaker masking
    if masker_level >= 60:
        # Strong masker: offset 3 dB
        offset = 3.0
    elif masker_level >= 40:
        # Medium masker: offset 6 dB (original value)
        offset = 6.0
    else:
        # Weak masker: offset 10 dB
        offset = 10.0

    # Compute masking threshold
    threshold = masker_level + slope * abs(bark_distance) - offset

    # Threshold cannot be negative (below absolute threshold of hearing)
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

    Returns an *effective* SPL after masking by removing the masking threshold
    in the power domain:

        P_eff = max(P_harmonic - P_threshold, 0)
        SPL_eff = 10 * log10(P_eff)

    This produces a smooth reduction when the harmonic is near the masking
    threshold, while still yielding 0 dB SPL when fully masked.

    Args:
        fundamental_freq: Fundamental frequency in Hz
        fundamental_spl: Fundamental SPL in dB
        harmonic_freqs: Harmonic frequencies in Hz (n,)
        harmonic_spls: Harmonic SPLs in dB (n,)

    Returns:
        masked_spls: Effective harmonic SPLs after masking in dB (n,)
                     In [0, harmonic_spl] for each harmonic.

    References:
        - Zwicker & Fastl (1999) "Psychoacoustics: Facts and Models"
        - ISO 226:2003 (equal-loudness contours)
    """
    masked_spls = np.zeros_like(harmonic_spls)

    for i, (h_freq, h_spl) in enumerate(zip(harmonic_freqs, harmonic_spls)):
        # Compute masking threshold from fundamental using improved spreading function
        threshold = compute_simultaneous_masking_threshold(
            fundamental_freq, fundamental_spl, h_freq
        )

        # No masking effect at/below 0 dB SPL threshold
        if threshold <= 0.0:
            masked_spls[i] = h_spl
            continue

        # Fully masked
        if h_spl <= threshold:
            masked_spls[i] = 0.0
            continue

        # Partial masking: remove threshold in power domain
        harmonic_power = np.power(10.0, h_spl / 10.0)
        threshold_power = np.power(10.0, threshold / 10.0)
        residual_power = harmonic_power - threshold_power

        if residual_power <= 0.0:
            masked_spls[i] = 0.0
        else:
            masked_spls[i] = max(10.0 * np.log10(residual_power), 0.0)

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

    Uses power summation of individual masking thresholds (in the power domain).

    NOTE: Each individual masking threshold is already a function of Bark distance
    via `compute_simultaneous_masking_threshold` (spreading function). Applying an
    additional distance-based weight here would effectively "double count" the
    distance attenuation. The `weight_function` parameter is retained for backward
    compatibility but is not used.

    Args:
        masker_freqs: Frequencies of all maskers (fundamental + harmonics 1-9) in Hz
        masker_spls: SPL levels of all maskers in dB
        maskee_freqs: Frequencies of harmonics being analyzed in Hz
        maskee_spls: SPL levels of harmonics being analyzed in dB
        weight_function: Kept for backward compatibility (currently unused).

    Returns:
        masked_spls: Effective masked SPL values for each maskee in dB (n_maskees,)

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

    # Compute threshold matrix (n_maskees × n_maskers)
    threshold_matrix = np.zeros((n_maskees, n_maskers))
    for i, maskee_freq in enumerate(maskee_freqs):
        for j, (masker_freq, masker_spl) in enumerate(zip(masker_freqs, masker_spls)):
            threshold = compute_simultaneous_masking_threshold(
                masker_freq, masker_spl, maskee_freq
            )
            threshold_matrix[i, j] = threshold

    # Apply weighted power summation for each maskee
    for i in range(n_maskees):
        thresholds = threshold_matrix[i, :]

        # Convert to linear power, sum, convert back
        powers = np.power(10.0, thresholds / 10.0)
        total_power = np.sum(powers)

        # Guard against zero total power (would cause log10(0) -> -inf)
        if total_power <= 0:
            # No masking effect if total power is zero or negative
            masked_spls[i] = maskee_spls[i]
            continue

        combined_threshold = 10.0 * np.log10(total_power)

        # No masking effect at/below 0 dB SPL threshold
        if combined_threshold <= 0.0:
            masked_spls[i] = maskee_spls[i]
            continue

        # Fully masked
        if maskee_spls[i] <= combined_threshold:
            masked_spls[i] = 0.0
            continue

        # Partial masking: remove combined threshold in power domain
        maskee_power = np.power(10.0, maskee_spls[i] / 10.0)
        threshold_power = np.power(10.0, combined_threshold / 10.0)
        residual_power = maskee_power - threshold_power

        if residual_power <= 0.0:
            masked_spls[i] = 0.0
        else:
            masked_spls[i] = max(10.0 * np.log10(residual_power), 0.0)

    return masked_spls
