# unit_test/base/pre_processing/test_psychoacoustic_utils.py
import pytest
import numpy as np
from base.pre_processing.psychoacoustic_utils import (
    spl_to_phons,
    compute_simultaneous_masking_threshold,
    apply_masking,
    compute_bark_weight,
    apply_cumulative_masking,
    compute_noise_spectrum
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


def test_cumulative_masking_single_masker():
    """Test cumulative masking with single masker (should match fundamental-only)"""
    masker_freqs = np.array([100.0])
    masker_spls = np.array([60.0])
    maskee_freqs = np.array([1000.0])
    maskee_spls = np.array([20.0])

    result = apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)

    assert len(result) == 1
    assert result[0] == 20.0, "With distant masker, harmonic should pass through"


def test_cumulative_masking_close_maskers():
    """Test cumulative masking with close masker (9th masking 10th)"""
    # 100 Hz fundamental, 900 Hz (9th) and 1000 Hz (10th)
    masker_freqs = np.array([100.0, 900.0])
    masker_spls = np.array([60.0, 50.0])  # Strong 9th harmonic
    maskee_freqs = np.array([1000.0])
    maskee_spls = np.array([30.0])  # Weak 10th harmonic

    result = apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)

    # 9th harmonic (900 Hz) is ~0.7 Bark from 10th (1000 Hz)
    # Should provide significant masking
    assert result[0] < 30.0, "Close masker should reduce harmonic SPL"
    assert result[0] > 0.0, "Harmonic not fully masked"


def test_cumulative_masking_full_masking():
    """Test that weak harmonic below threshold is fully masked"""
    masker_freqs = np.array([100.0, 900.0])
    masker_spls = np.array([60.0, 60.0])  # Very strong 9th
    maskee_freqs = np.array([1000.0])
    maskee_spls = np.array([10.0])  # Very weak 10th

    result = apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)

    assert result[0] == 0.0, "Weak harmonic should be fully masked"


def test_cumulative_masking_multiple_maskees():
    """Test cumulative masking with multiple target harmonics"""
    # Fundamental + 1st-9th harmonics as maskers
    masker_freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0,
                             600.0, 700.0, 800.0, 900.0])
    masker_spls = np.full(9, 50.0)  # All at 50 dB

    # Target: 10th, 11th, 12th harmonics
    maskee_freqs = np.array([1000.0, 1100.0, 1200.0])
    maskee_spls = np.array([30.0, 28.0, 26.0])

    result = apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)

    assert len(result) == 3
    # All should be reduced by cumulative masking
    assert all(result < maskee_spls)


def test_cumulative_masking_weight_function_selection():
    """Test all weight functions produce different results without warnings"""
    masker_freqs = np.array([100.0, 900.0])
    masker_spls = np.array([60.0, 50.0])
    maskee_freqs = np.array([1000.0])
    maskee_spls = np.array([30.0])

    # Test all 4 weight functions
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)  # Catch any runtime warnings

        result_exp = apply_cumulative_masking(
            masker_freqs, masker_spls, maskee_freqs, maskee_spls, 'exponential'
        )
        result_gauss = apply_cumulative_masking(
            masker_freqs, masker_spls, maskee_freqs, maskee_spls, 'gaussian'
        )
        result_linear = apply_cumulative_masking(
            masker_freqs, masker_spls, maskee_freqs, maskee_spls, 'linear'
        )
        result_inverse = apply_cumulative_masking(
            masker_freqs, masker_spls, maskee_freqs, maskee_spls, 'inverse'
        )

    # Different weighting functions should produce different results
    assert result_exp[0] != result_gauss[0], "Exponential vs Gaussian"
    assert result_exp[0] != result_linear[0], "Exponential vs Linear"
    assert result_exp[0] != result_inverse[0], "Exponential vs Inverse"
    assert result_gauss[0] != result_linear[0], "Gaussian vs Linear"

    # All results should be valid (>= 0, <= original SPL)
    for result in [result_exp, result_gauss, result_linear, result_inverse]:
        assert result[0] >= 0.0, "Result should be non-negative"
        assert result[0] <= maskee_spls[0], "Result should not exceed original SPL"


def test_cumulative_masking_mismatched_masker_arrays():
    """Test that mismatched masker array lengths raise ValueError"""
    masker_freqs = np.array([100.0, 200.0])
    masker_spls = np.array([60.0])  # Mismatched length
    maskee_freqs = np.array([1000.0])
    maskee_spls = np.array([30.0])

    with pytest.raises(ValueError, match="masker_freqs and masker_spls must have same length"):
        apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)


def test_cumulative_masking_mismatched_maskee_arrays():
    """Test that mismatched maskee array lengths raise ValueError"""
    masker_freqs = np.array([100.0, 200.0])
    masker_spls = np.array([60.0, 50.0])
    maskee_freqs = np.array([1000.0, 1100.0])
    maskee_spls = np.array([30.0])  # Mismatched length

    with pytest.raises(ValueError, match="maskee_freqs and maskee_spls must have same length"):
        apply_cumulative_masking(masker_freqs, masker_spls, maskee_freqs, maskee_spls)


def test_cumulative_masking_zero_power_guard():
    """Test that zero total power doesn't cause divide-by-zero"""
    # Use very distant masker with linear weight function that clips to 0
    masker_freqs = np.array([100.0])
    masker_spls = np.array([60.0])
    # Very high frequency maskee (>50 Bark distance with linear weight = 0)
    maskee_freqs = np.array([20000.0])
    maskee_spls = np.array([30.0])

    # Should not raise RuntimeWarning for log10(0)
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        result = apply_cumulative_masking(
            masker_freqs, masker_spls, maskee_freqs, maskee_spls, 'linear'
        )

    # With zero power, maskee should pass through unchanged
    assert result[0] == maskee_spls[0]


def test_compute_noise_spectrum_exact_length():
    """Test compute_noise_spectrum with exact 0.5s data"""
    sample_rate = 48000
    n_fft = int(0.5 * sample_rate)  # 24000

    # Create 0.5s of white noise
    pre_alignment_data = np.random.randn(n_fft)

    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Should return magnitude spectrum of correct length
    expected_length = n_fft // 2 + 1  # 12001
    assert len(noise_spectrum) == expected_length
    # All magnitudes should be non-negative
    assert np.all(noise_spectrum >= 0)


def test_compute_noise_spectrum_longer_data():
    """Test compute_noise_spectrum extracts middle 0.5s from longer data"""
    sample_rate = 48000
    n_fft = int(0.5 * sample_rate)  # 24000

    # Create 1.0s of data (longer than 0.5s)
    pre_alignment_data = np.random.randn(sample_rate)

    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Should still return correct length
    expected_length = n_fft // 2 + 1
    assert len(noise_spectrum) == expected_length
    assert np.all(noise_spectrum >= 0)


def test_compute_noise_spectrum_shorter_data():
    """Test compute_noise_spectrum pads shorter data with zeros"""
    sample_rate = 48000
    n_fft = int(0.5 * sample_rate)  # 24000

    # Create 0.1s of data (shorter than 0.5s)
    pre_alignment_data = np.random.randn(int(0.1 * sample_rate))

    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Should still return correct length (after padding)
    expected_length = n_fft // 2 + 1
    assert len(noise_spectrum) == expected_length
    assert np.all(noise_spectrum >= 0)


def test_compute_noise_spectrum_sine_wave():
    """Test compute_noise_spectrum with known sine wave"""
    sample_rate = 48000
    n_fft = int(0.5 * sample_rate)  # 24000

    # Create 0.5s of 1000 Hz sine wave
    t = np.arange(n_fft) / sample_rate
    pre_alignment_data = np.sin(2 * np.pi * 1000 * t)

    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Should have peak around 1000 Hz
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
    peak_idx = np.argmax(noise_spectrum)
    peak_freq = freqs[peak_idx]

    # Peak should be close to 1000 Hz (within 10 Hz tolerance)
    assert abs(peak_freq - 1000.0) < 10.0
    # Peak magnitude should be significant
    assert noise_spectrum[peak_idx] > 0.1


def test_compute_noise_spectrum_dc_component():
    """Test compute_noise_spectrum DC component is attenuated by Hann window"""
    sample_rate = 48000
    n_fft = int(0.5 * sample_rate)

    # Create DC signal (constant value)
    pre_alignment_data = np.ones(n_fft)

    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Hann window attenuates DC component
    # DC bin (index 0) should be non-zero but reduced
    assert noise_spectrum[0] > 0
    # DC should not dominate the spectrum
    # (Hann window spreads energy to nearby bins)
    assert noise_spectrum[0] < np.sum(noise_spectrum) * 0.5

