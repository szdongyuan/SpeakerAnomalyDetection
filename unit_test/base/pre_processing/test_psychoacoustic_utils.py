# unit_test/base/pre_processing/test_psychoacoustic_utils.py
import pytest
import numpy as np
from base.pre_processing.psychoacoustic_utils import (
    spl_to_phons,
    compute_simultaneous_masking_threshold,
    apply_masking,
    compute_bark_weight
)


def test_spl_to_phons_1khz():
    """Verify 1 kHz reference: 40 dB SPL = 40 phons"""
    frequencies = np.array([1000.0])
    spl_values = np.array([40.0])

    phons = spl_to_phons(frequencies, spl_values)

    # At 1 kHz, phons should equal SPL
    assert np.isclose(phons[0], 40.0, atol=0.1)


def test_spl_to_phons_low_frequency():
    """Verify low frequencies require more SPL for same loudness"""
    frequencies = np.array([100.0, 1000.0])
    spl_values = np.array([60.0, 40.0])  # Same loudness level

    phons = spl_to_phons(frequencies, spl_values)

    # 100 Hz at 60 dB ≈ 40 phons (from ISO 226)
    assert np.isclose(phons[0], 40.0, atol=5.0)
    assert np.isclose(phons[1], 40.0, atol=0.1)


def test_compute_simultaneous_masking_threshold():
    """Verify masking threshold increases with masker level"""
    masker_freq = 1000.0
    masker_level = 60.0  # dB SPL
    maskee_freq = 1100.0  # Nearby frequency

    threshold = compute_simultaneous_masking_threshold(
        masker_freq, masker_level, maskee_freq
    )

    # Threshold should be positive (masking occurs)
    assert threshold > 0
    # Threshold should be less than masker level
    assert threshold < masker_level


def test_compute_simultaneous_masking_threshold_far_frequency():
    """Verify masking decreases with frequency distance"""
    masker_freq = 1000.0
    masker_level = 60.0
    near_freq = 1100.0
    far_freq = 2000.0

    near_threshold = compute_simultaneous_masking_threshold(
        masker_freq, masker_level, near_freq
    )
    far_threshold = compute_simultaneous_masking_threshold(
        masker_freq, masker_level, far_freq
    )

    # Near frequency should have higher masking threshold
    assert near_threshold > far_threshold


def test_apply_masking():
    """Verify masking reduces perceived loudness of harmonics"""
    # Setup: fundamental + 10th harmonic
    fundamental_freq = 100.0
    fundamental_spl = 70.0

    harmonic_freqs = np.array([1000.0, 1100.0])  # 10th, 11th harmonics
    harmonic_spls = np.array([40.0, 35.0])

    masked_spls = apply_masking(
        fundamental_freq, fundamental_spl,
        harmonic_freqs, harmonic_spls
    )

    # Masked SPLs should be <= original SPLs
    assert np.all(masked_spls <= harmonic_spls)
    # Some masking should occur
    assert np.any(masked_spls < harmonic_spls)


def test_bark_weight_exponential_close_distance():
    """Test exponential weighting for close harmonics (< 1 Bark)"""
    weight = compute_bark_weight(0.7, 'exponential')
    assert weight > 0.6, "Close harmonics should have high weight"
    assert weight < 1.0, "Weight should be less than 1.0"
    # exp(-0.7/2.0) ≈ 0.705
    assert abs(weight - 0.705) < 0.01


def test_bark_weight_exponential_far_distance():
    """Test exponential weighting for distant harmonics (> 5 Bark)"""
    weight = compute_bark_weight(5.0, 'exponential')
    assert weight < 0.1, "Distant harmonics should have low weight"
    # exp(-5.0/2.0) ≈ 0.082
    assert abs(weight - 0.082) < 0.01


def test_bark_weight_zero_distance():
    """Test weighting at zero distance (same frequency)"""
    weight = compute_bark_weight(0.0, 'exponential')
    assert weight == 1.0, "Same frequency should have weight 1.0"


def test_bark_weight_gaussian():
    """Test Gaussian weighting function"""
    weight = compute_bark_weight(1.0, 'gaussian')
    # exp(-(1.0**2)/2.0) ≈ 0.606
    assert abs(weight - 0.606) < 0.01


def test_bark_weight_linear():
    """Test linear weighting function"""
    weight = compute_bark_weight(2.0, 'linear')
    # max(0, 1 - 2.0/5.0) = 0.6
    assert weight == 0.6


def test_bark_weight_linear_clipping():
    """Test linear weighting clips to 0"""
    weight = compute_bark_weight(10.0, 'linear')
    assert weight == 0.0


def test_bark_weight_inverse():
    """Test inverse weighting function"""
    weight = compute_bark_weight(1.0, 'inverse')
    # 1/(1+1.0) = 0.5
    assert weight == 0.5


def test_bark_weight_invalid_function():
    """Test invalid weight function raises error"""
    with pytest.raises(ValueError, match="Unknown weight function"):
        compute_bark_weight(1.0, 'invalid')
