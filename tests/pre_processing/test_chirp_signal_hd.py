# tests/pre_processing/test_chirp_signal_hd.py
import numpy as np
import pytest
from base.pre_processing.chirp_signal_hd import ChirpSignalHD
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestChirpSignalHD:
    def test_compute_distortion_with_prebuilt_mask(self):
        """Test THD computation for chirp signal using pre-built mask"""
        # Build mask in Phase 1
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024

        # Phase 1A: Build overall index
        index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
            stimulus_metadata,
            sr=44100,
            n_fft=stft_window_size,
            hop_length=stft_hop_size,
            max_harmonic_order=35
        )

        # Phase 1B: Select harmonics and build mask
        harmonic_orders = [2, 3]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Create synthetic recorded signal
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        # Phase 2: Compute THD
        analyzer = ChirpSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, fund_freqs, time_array, fundamental_bins),
            stft_window_size=stft_window_size,
            stft_hop_size=stft_hop_size
        )

        assert 'frequencies' in result
        assert 'thd' in result
        assert 'times' in result
        assert len(result['frequencies']) > 0
        assert len(result['thd']) == len(result['frequencies'])
