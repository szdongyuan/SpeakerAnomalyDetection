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

    def test_build_chirp_signal_index_matrix_shape(self):
        """Test that chirp index matrix has correct shape (num_frames, max_order+1)"""
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 4.0,
            'repeat_times': 2,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024
        max_order = 35

        index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
            stimulus_metadata,
            sr=44100,
            n_fft=stft_window_size,
            hop_length=stft_hop_size,
            max_harmonic_order=max_order
        )

        # Should have time-varying frames
        assert index_matrix.shape[1] == 36  # max_order + 1
        assert index_matrix.shape[0] > 50  # Many frames for time-varying signal
        # Column 0 should be sentinel (all zeros)
        assert np.all(index_matrix[:, 0] == 0)
        # Column 1 should have fundamental bins (non-zero)
        assert np.all(index_matrix[:, 1] > 0)
        # Fund freqs should be time-varying
        assert len(fund_freqs) == index_matrix.shape[0]
        assert fund_freqs[0] < fund_freqs[-1]  # Log chirp: low to high

    def test_create_mask_from_indices(self):
        """Test converting selected harmonic indices to binary mask matrix"""
        builder = HarmonicIndexBuilder()

        # Build overall index first
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
        index_matrix, _, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=n_fft, max_harmonic_order=35
        )

        # User selects harmonics [2, 3, 4, 5]
        harmonic_orders = [2, 3, 4, 5]
        n_bins_with_dummy = len(fft_freqs)

        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, n_bins_with_dummy
        )

        # Mask should be binary (n_bins+1, num_steps)
        assert mask_matrix.shape == (n_bins_with_dummy, 16)
        assert np.all((mask_matrix == 0) | (mask_matrix == 1))
        # Row 0 (dummy bin) should be all zeros
        assert np.all(mask_matrix[0, :] == 0)
        # Should have exactly 5 ones per step (fundamental + 4 harmonics)
        assert np.all(np.sum(mask_matrix, axis=0) == 5)

    def test_build_step_signal_index_matrix_with_stft_params(self):
        """Test that step signal index matrix can be built with STFT window parameters"""
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

        # Calculate step duration for window size
        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * stimulus_metadata['sample_rate'])

        # For STFT: window_size = step_samples (no trimming)
        stft_window_size = step_samples
        max_order = 35

        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=stft_window_size, max_harmonic_order=max_order
        )

        # Verify shape and structure
        assert index_matrix.shape == (16, 36)  # 36 = max_order + 1
        assert np.all(index_matrix[:, 0] == 0)  # Column 0 is sentinel
        assert np.all(index_matrix[:, 1] > 0)    # Column 1 has fundamental bins
        assert len(fund_freqs) == 16
        assert fund_freqs[0] == pytest.approx(500.0, abs=1.0)
        assert fund_freqs[-1] == pytest.approx(2000.0, abs=1.0)

        # Verify FFT freqs match STFT expectations
        expected_n_bins = stft_window_size // 2 + 1
        assert len(fft_freqs) == expected_n_bins + 1  # +1 for dummy bin

    def test_chirp_invalid_repeat_times_zero_raises_error(self):
        """Test that repeat_times=0 raises ValueError in chirp index builder"""
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 4.0,
            'repeat_times': 0,  # Invalid
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024

        with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
            builder.build_chirp_signal_index_matrix(
                stimulus_metadata,
                sr=44100,
                n_fft=stft_window_size,
                hop_length=stft_hop_size,
                max_harmonic_order=35
            )

    def test_chirp_invalid_repeat_times_negative_raises_error(self):
        """Test that negative repeat_times raises ValueError in chirp index builder"""
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 4.0,
            'repeat_times': -1,  # Invalid
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024

        with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
            builder.build_chirp_signal_index_matrix(
                stimulus_metadata,
                sr=44100,
                n_fft=stft_window_size,
                hop_length=stft_hop_size,
                max_harmonic_order=35
            )

    def test_chirp_invalid_repeat_times_float_raises_error(self):
        """Test that float repeat_times raises ValueError in chirp index builder"""
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 4.0,
            'repeat_times': 2.5,  # Invalid - float
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024

        with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
            builder.build_chirp_signal_index_matrix(
                stimulus_metadata,
                sr=44100,
                n_fft=stft_window_size,
                hop_length=stft_hop_size,
                max_harmonic_order=35
            )
