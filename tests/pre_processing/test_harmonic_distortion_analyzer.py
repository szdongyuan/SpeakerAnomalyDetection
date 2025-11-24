# tests/pre_processing/test_harmonic_distortion_analyzer.py
import numpy as np
import pytest
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class ConcreteAnalyzer(HarmonicDistortionAnalyzer):
    """Concrete implementation for testing."""
    def compute_distortion(self, recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask, **kwargs):
        return {}


class TestHarmonicDistortionAnalyzer:
    def test_compute_thd_batch(self):
        """Test vectorized THD computation using pre-built mask"""
        analyzer = ConcreteAnalyzer(sample_rate=44100)

        # Create synthetic spectrum (n_bins+1, n_steps)
        n_bins = 100
        n_steps = 16
        spectrum_matrix = np.random.rand(n_bins + 1, n_steps) * 100
        spectrum_matrix[0, :] = 0  # Dummy bin

        # Create mask: fundamental at bins [10, 11, 12, ...], 2nd harmonic at [20, 22, 24, ...]
        mask_matrix = np.zeros((n_bins + 1, n_steps))
        fundamental_bins = np.arange(10, 10 + n_steps)
        for i in range(n_steps):
            mask_matrix[fundamental_bins[i], i] = 1.0  # Fundamental
            mask_matrix[fundamental_bins[i] * 2, i] = 1.0  # 2nd harmonic

        thd = analyzer.compute_thd_batch(spectrum_matrix, mask_matrix, fundamental_bins)

        # THD should be array of percentages
        assert thd.shape == (n_steps,)
        assert np.all(thd >= 0)
        assert np.all(thd <= 100)
