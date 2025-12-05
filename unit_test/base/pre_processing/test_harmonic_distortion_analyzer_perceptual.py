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


def test_compute_perceptual_thd_batch_with_cumulative_masking():
    """Test cumulative masking reduces phon values compared to fundamental-only"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    # Create spectrum with strong 9th harmonic, weak 10th
    n_bins = 1024
    n_frames = 1
    spectrum_matrix = np.zeros((n_bins + 1, n_frames))

    # Fundamental at bin 10 (100 Hz)
    spectrum_matrix[10, 0] = 0.5

    # Strong 9th harmonic (900 Hz) at bin 90
    spectrum_matrix[90, 0] = 0.05

    # Weak 10th harmonic (1000 Hz) at bin 100
    spectrum_matrix[100, 0] = 0.01

    # Analysis mask: only 10th harmonic
    mask_matrix = np.zeros((n_bins + 1, n_frames))
    mask_matrix[100, 0] = 1.0

    # Masking mask: 1st-9th harmonics
    masking_mask_matrix = np.zeros((n_bins + 1, n_frames))
    for h in range(1, 10):
        masking_mask_matrix[10 * h, 0] = 1.0

    fundamental_bins = np.array([10])
    fundamental_freqs = np.array([100.0])

    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    # Test with cumulative masking
    result_cumulative = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs,
        masking_mask_matrix=masking_mask_matrix,
        masking_config=masking_config
    )

    # Test without cumulative masking (fundamental only)
    result_fundamental = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs,
        masking_mask_matrix=None,
        masking_config=None
    )

    # Cumulative masking should reduce phon value (9th masks 10th)
    assert result_cumulative[0] < result_fundamental[0]
    assert result_cumulative[0] >= 0


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
