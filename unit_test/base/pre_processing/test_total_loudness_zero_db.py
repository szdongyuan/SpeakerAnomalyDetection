import numpy as np
from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class _DummyAnalyzer(HarmonicDistortionAnalyzer):
    """Minimal concrete analyzer to access compute_perceptual_thd_batch."""

    def compute_distortion(self, recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask, **kwargs):
        raise NotImplementedError


def test_total_loudness_at_zero_db_is_near_zero():
    """
    Verify that 0 dB SPL content yields near-0 phon after proportional allocation.

    A single harmonic at 0 dB SPL (20 µPa) should produce ~0 phon total loudness.
    """
    sample_rate = 48000
    analyzer = _DummyAnalyzer(sample_rate=sample_rate)

    # One frame, dummy bin + fundamental bin + one harmonic bin
    # Amplitude corresponding to 0 dB SPL relative to 20 µPa
    amp_zero_db = 20e-6
    spectrum_matrix = np.array([
        [0.0],           # dummy bin
        [0.0],           # fundamental bin (set to 0 to avoid coupling in anchor)
        [amp_zero_db],   # one harmonic bin at 0 dB SPL
    ])

    # Mask selects harmonic bin only (bin index 2)
    mask_matrix = np.array([
        [0.0],
        [0.0],
        [1.0],
    ])
    fundamental_bins = np.array([1])  # fundamental bin index
    fundamental_freqs = np.array([1000.0])  # anchor frequency (not used in fullband sum)

    loudness = analyzer.compute_perceptual_thd_batch(
        spectrum_matrix=spectrum_matrix,
        mask_matrix=mask_matrix,
        fundamental_bins=fundamental_bins,
        fundamental_freqs=fundamental_freqs,
        spl_calibration_db=0.0,
    )

    # Expect near-0 phon (sones=1/16) -> phon=0
    assert loudness[0] == 0.0
