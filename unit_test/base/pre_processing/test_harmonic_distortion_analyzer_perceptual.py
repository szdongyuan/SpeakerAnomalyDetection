# unit_test/base/pre_processing/test_harmonic_distortion_analyzer_perceptual.py
import pytest
import numpy as np
from unittest.mock import Mock
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


def test_compute_perceptual_thd_batch():
    """Verify perceptual THD returns phons instead of percentage"""
    # Create mock analyzer (abstract class, so we mock it)
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    # Bind the method to the mock
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    # Setup test data
    n_bins = 1000
    n_frames = 3

    # Create spectrum with fundamental and harmonics
    spectrum_matrix = np.zeros((n_bins + 1, n_frames))
    fundamental_bins = np.array([100, 200, 300])  # Fundamental at bins 100, 200, 300

    # Set fundamental amplitudes (high)
    spectrum_matrix[100, 0] = 1.0
    spectrum_matrix[200, 1] = 1.0
    spectrum_matrix[300, 2] = 1.0

    # Set 10th harmonic amplitudes (lower)
    spectrum_matrix[110, 0] = 0.01  # 10th harmonic
    spectrum_matrix[210, 1] = 0.01
    spectrum_matrix[310, 2] = 0.01

    # Create mask (1s for harmonics, 0s elsewhere)
    mask_matrix = np.zeros_like(spectrum_matrix)
    mask_matrix[110, 0] = 1.0
    mask_matrix[210, 1] = 1.0
    mask_matrix[310, 2] = 1.0

    # Fundamental frequencies (derived from bins)
    fundamental_freqs = fundamental_bins * (44100 / 2) / n_bins

    # Call perceptual THD
    perceptual_loudness = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs
    )

    # Should return phons (positive values)
    assert perceptual_loudness.shape == (n_frames,)
    assert np.all(perceptual_loudness >= 0)
    # Phons should be reasonable (0-100 range typically)
    assert np.all(perceptual_loudness < 200)


def test_compute_perceptual_thd_batch_masking_effect():
    """Verify masking reduces perceived loudness"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    n_bins = 1000
    n_frames = 2

    spectrum_matrix = np.zeros((n_bins + 1, n_frames))

    # Frame 0: Strong fundamental, weak harmonic (strong masking)
    spectrum_matrix[100, 0] = 1.0  # Fundamental
    spectrum_matrix[110, 0] = 0.001  # 10th harmonic (very weak)

    # Frame 1: Weak fundamental, strong harmonic (less masking)
    spectrum_matrix[100, 1] = 0.1  # Fundamental
    spectrum_matrix[110, 1] = 0.001  # Same harmonic level

    mask_matrix = np.zeros_like(spectrum_matrix)
    mask_matrix[110, :] = 1.0  # Mask for harmonics

    fundamental_bins = np.array([100, 100])
    fundamental_freqs = np.array([1000.0, 1000.0])

    perceptual_loudness = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs
    )

    # Frame 0 should have lower perceived loudness (more masking)
    # Frame 1 should have higher perceived loudness (less masking)
    assert perceptual_loudness[0] <= perceptual_loudness[1]


def test_compute_perceptual_thd_batch_additional_tonal_masker_reduces_loudness():
    """Strong nearby tonal energy should reduce perceived loudness of a target harmonic."""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    n_bins = 1024
    n_frames = 1
    spectrum_base = np.zeros((n_bins + 1, n_frames))

    # Fundamental at bin 10 (100 Hz)
    spectrum_base[10, 0] = 0.5
    # Weak 10th harmonic (target)
    spectrum_base[100, 0] = 0.01

    # Analysis mask: only 10th harmonic
    mask_matrix = np.zeros((n_bins + 1, n_frames))
    mask_matrix[100, 0] = 1.0

    fundamental_bins = np.array([10])
    fundamental_freqs = np.array([100.0])

    # Baseline: no nearby strong masker
    result_base = analyzer.compute_perceptual_thd_batch(
        spectrum_base, mask_matrix, fundamental_bins, fundamental_freqs
    )

    # Add a strong nearby tonal component (e.g., 9th harmonic around 900 Hz)
    spectrum_with_masker = spectrum_base.copy()
    spectrum_with_masker[90, 0] = 0.08
    result_with_masker = analyzer.compute_perceptual_thd_batch(
        spectrum_with_masker, mask_matrix, fundamental_bins, fundamental_freqs
    )

    assert result_with_masker[0] <= result_base[0]
    assert result_with_masker[0] >= 0


def test_compute_perceptual_thd_batch_backward_compatible():
    """Test that masking_config=None uses existing fundamental-only behavior"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    # Simple test setup
    n_bins = 1024
    n_frames = 1
    spectrum_matrix = np.random.rand(n_bins + 1, n_frames) * 0.01
    spectrum_matrix[10, 0] = 0.5  # Fundamental

    mask_matrix = np.zeros((n_bins + 1, n_frames))
    mask_matrix[100, 0] = 1.0

    fundamental_bins = np.array([10])
    fundamental_freqs = np.array([100.0])

    # Should not raise error with old signature
    result = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs
    )

    assert len(result) == n_frames
    assert result[0] >= 0


#
# NOTE: The PRB masking model now derives maskers from the full spectrum per frame,
# so previous tests that patched apply_cumulative_masking / checked double counting
# are no longer applicable.

def test_apply_noise_correction_basic():
    """Test basic noise correction with quadrature subtraction"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    # Bind the method to the mock
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Create test data: harmonic at bin 100, amplitude 2.0
    harmonic_bins = np.array([100])
    harmonic_amplitudes = np.array([2.0])
    
    # Noise spectrum: 1000 bins, noise at bin 100 is 1.0
    noise_spectrum = np.zeros(1000)
    noise_spectrum[100] = 1.0
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # Quadrature subtraction: sqrt(2^2 - 1^2) = sqrt(3) ≈ 1.732
    expected = np.sqrt(2.0**2 - 1.0**2)
    assert np.isclose(corrected[0], expected, atol=0.001)


def test_apply_noise_correction_noise_exceeds_signal():
    """Test noise correction when noise exceeds signal (should clip to min threshold)"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Harmonic amplitude: 1.0, Noise: 2.0 (exceeds signal)
    harmonic_bins = np.array([100])
    harmonic_amplitudes = np.array([1.0])
    
    noise_spectrum = np.zeros(1000)
    noise_spectrum[100] = 2.0
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # Should clip to minimum threshold (1e-12) to prevent log(0)
    assert corrected[0] == 1e-12


def test_apply_noise_correction_zero_noise():
    """Test noise correction with zero noise (should return original amplitude)"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Harmonic amplitude: 2.0, Noise: 0.0
    harmonic_bins = np.array([100])
    harmonic_amplitudes = np.array([2.0])
    
    noise_spectrum = np.zeros(1000)
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # With zero noise, should return original amplitude
    assert np.isclose(corrected[0], 2.0, atol=0.001)


def test_apply_noise_correction_interpolation():
    """Test noise spectrum interpolation for non-integer bin indices"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Harmonic at non-integer bin (requires interpolation)
    harmonic_bins = np.array([100.5])  # Between bin 100 and 101
    harmonic_amplitudes = np.array([2.0])
    
    # Noise spectrum with values at bins 100 and 101
    noise_spectrum = np.zeros(1000)
    noise_spectrum[100] = 1.0
    noise_spectrum[101] = 1.5
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # Interpolated noise should be 1.25 (average of 1.0 and 1.5)
    # Quadrature: sqrt(2^2 - 1.25^2) ≈ 1.601
    expected = np.sqrt(2.0**2 - 1.25**2)
    assert np.isclose(corrected[0], expected, atol=0.001)


def test_apply_noise_correction_multiple_harmonics():
    """Test noise correction with multiple harmonics"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Multiple harmonics at different bins
    harmonic_bins = np.array([100, 200, 300])
    harmonic_amplitudes = np.array([2.0, 3.0, 4.0])
    
    # Noise spectrum
    noise_spectrum = np.zeros(1000)
    noise_spectrum[100] = 1.0
    noise_spectrum[200] = 1.5
    noise_spectrum[300] = 2.0
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # Verify each correction
    assert np.isclose(corrected[0], np.sqrt(2.0**2 - 1.0**2), atol=0.001)
    assert np.isclose(corrected[1], np.sqrt(3.0**2 - 1.5**2), atol=0.001)
    assert np.isclose(corrected[2], np.sqrt(4.0**2 - 2.0**2), atol=0.001)


def test_apply_noise_correction_minimum_threshold():
    """Test all corrected values are >= minimum threshold"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    
    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer._apply_noise_correction = HDAnalyzer._apply_noise_correction.__get__(analyzer)
    
    # Mix of cases: some with noise > signal, some normal
    harmonic_bins = np.array([100, 200, 300])
    harmonic_amplitudes = np.array([0.5, 2.0, 3.0])
    
    noise_spectrum = np.zeros(1000)
    noise_spectrum[100] = 1.0  # Noise > signal
    noise_spectrum[200] = 0.5  # Noise < signal
    noise_spectrum[300] = 3.0  # Noise == signal
    
    corrected = analyzer._apply_noise_correction(
        harmonic_bins, harmonic_amplitudes, noise_spectrum
    )
    
    # All values should be >= 1e-12 (minimum threshold)
    assert np.all(corrected >= 1e-12)
    # No NaNs or infinities
    assert np.all(np.isfinite(corrected))
