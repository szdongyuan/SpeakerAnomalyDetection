# unit_test/base/pre_processing/test_harmonic_distortion_analyzer_perceptual.py
import pytest
import numpy as np
from unittest.mock import Mock
from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


def test_compute_perceptual_thd_batch():
    """Verify perceptual THD returns phons instead of percentage"""
    # Create mock analyzer (abstract class, so we mock it)
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000

    # Bind the method to the mock
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
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
    fundamental_freqs = fundamental_bins * (analyzer.sample_rate / 2) / n_bins

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
    analyzer.sample_rate = 48000

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
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
        spectrum_matrix,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_loudness_method": "masked_harmonics"},
        n_fft=2 * (n_bins - 1),
    )

    # Frame 0 should have lower perceived loudness (more masking)
    # Frame 1 should have higher perceived loudness (less masking)
    assert perceptual_loudness[0] <= perceptual_loudness[1]


def test_compute_perceptual_thd_batch_additional_tonal_masker_reduces_loudness():
    """Strong nearby tonal energy should reduce perceived loudness of a target harmonic."""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
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
        spectrum_base,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_loudness_method": "masked_harmonics"},
        n_fft=2 * (n_bins - 1),
    )

    # Add a strong nearby tonal component (e.g., 9th harmonic around 900 Hz)
    spectrum_with_masker = spectrum_base.copy()
    spectrum_with_masker[90, 0] = 0.08
    result_with_masker = analyzer.compute_perceptual_thd_batch(
        spectrum_with_masker,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_loudness_method": "masked_harmonics"},
        n_fft=2 * (n_bins - 1),
    )

    assert result_with_masker[0] <= result_base[0]
    assert result_with_masker[0] >= 0


def test_compute_perceptual_thd_batch_backward_compatible():
    """Test that masking_config=None uses existing fundamental-only behavior"""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
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


def test_compute_perceptual_thd_batch_sc_method_basic_sanity():
    """SC method: fundamental-only should yield ~0, adding a harmonic should increase PRB."""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    n_bins = 1024
    n_frames = 1
    n_fft = 2 * (n_bins - 1)

    spectrum_matrix = np.zeros((n_bins + 1, n_frames), dtype=np.float64)
    mask_matrix = np.zeros_like(spectrum_matrix)

    # Fundamental at row 10 (dummy row at 0).
    f0_row = 10
    spectrum_matrix[f0_row, 0] = 1.0
    fundamental_bins = np.array([f0_row], dtype=int)
    fundamental_freqs = np.array([100.0], dtype=float)

    out_f0 = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_method": "sc", "sc_metric": "totalnl"},
        n_fft=n_fft,
    )
    assert out_f0.shape == (1,)
    assert out_f0[0] == pytest.approx(0.0, abs=1e-12)

    # Add a harmonic.
    spectrum_matrix2 = spectrum_matrix.copy()
    spectrum_matrix2[100, 0] = 0.1
    out_h = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix2,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_method": "sc", "sc_metric": "totalnl"},
        n_fft=n_fft,
    )
    assert out_h[0] > out_f0[0]


def test_compute_perceptual_thd_batch_sc_metric_ehs_detects_harmonic_family():
    """SC method: EHS should increase when a dense harmonic series is present."""
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer

    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    n_bins = 1024
    n_frames = 1
    n_fft = 2 * (n_bins - 1)
    rfft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / analyzer.sample_rate)

    # Pick an f0 aligned with an FFT bin so the harmonic pattern is clean in frequency domain.
    k0 = 20  # rFFT bin index (excluding dummy row offset)
    f0_hz = float(rfft_freqs[k0])
    f0_row = k0 + 1  # spectrum_matrix has dummy row at 0

    spectrum_f0 = np.zeros((n_bins + 1, n_frames), dtype=np.float64)
    spectrum_f0[f0_row, 0] = 1.0

    spectrum_h = spectrum_f0.copy()
    for h in range(2, 30):
        k = h * k0
        if k >= n_bins:
            break
        spectrum_h[k + 1, 0] = 0.3

    mask_matrix = np.zeros_like(spectrum_f0)
    fundamental_bins = np.array([f0_row], dtype=int)
    fundamental_freqs = np.array([f0_hz], dtype=float)

    out_ehs_f0 = analyzer.compute_perceptual_thd_batch(
        spectrum_f0,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_method": "sc", "sc_metric": "ehs"},
        n_fft=n_fft,
    )
    out_ehs_h = analyzer.compute_perceptual_thd_batch(
        spectrum_h,
        mask_matrix,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_method": "sc", "sc_metric": "ehs"},
        n_fft=n_fft,
    )
    assert out_ehs_h[0] > out_ehs_f0[0]


def test_compute_perceptual_thd_batch_requires_48khz_or_higher():
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100

    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HDAnalyzer
    analyzer.compute_perceptual_thd_batch = HDAnalyzer.compute_perceptual_thd_batch.__get__(analyzer)

    spectrum_matrix = np.zeros((33, 1))
    mask_matrix = np.zeros_like(spectrum_matrix)
    fundamental_bins = np.array([1])
    fundamental_freqs = np.array([100.0])
    with pytest.raises(ValueError, match=">= 48000"):
        analyzer.compute_perceptual_thd_batch(spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs)


#
# NOTE: Noise spectrum subtraction was removed because it is hard to control and can
# produce inconsistent results depending on the capture conditions.
