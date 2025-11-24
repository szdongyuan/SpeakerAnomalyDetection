import numpy as np
import pytest
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestHarmonicIndexBuilder:
    def test_build_step_signal_index_matrix_shape(self):
        """Test that overall index matrix has correct shape (num_steps, max_order+1)"""
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 500.0,
            'stop_freq': 2000.0,
            'num_steps': 16,
            'total_time': 4.0,
            'repeat_times': 3,
            'sample_rate': 44100
        }

        n_fft = 35280
        max_order = 35

        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=n_fft, max_harmonic_order=max_order
        )

        # Column 0 should be sentinel (all zeros)
        assert index_matrix.shape == (16, 36)  # 36 = max_order + 1
        assert np.all(index_matrix[:, 0] == 0)
        # Column 1 should have fundamental bins (non-zero, +1 offset)
        assert np.all(index_matrix[:, 1] > 0)
        # Fund freqs should match stimulus config
        assert len(fund_freqs) == 16
        assert fund_freqs[0] == pytest.approx(500.0, abs=1.0)
        assert fund_freqs[-1] == pytest.approx(2000.0, abs=1.0)
