import numpy as np
import pytest
from base.pre_processing.step_signal_hd import StepSignalHD
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestStepSignalHD:
    def test_compute_distortion_with_prebuilt_mask(self):
        """Test THD computation for step signal using pre-built mask"""
        # Build mask in Phase 1
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 500.0,
            'stop_freq': 2000.0,
            'num_steps': 4,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        trim_samples = 2205
        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * stimulus_metadata['sample_rate'])
        n_fft = step_samples - 2 * trim_samples

        # Phase 1A: Build overall index
        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=n_fft, max_harmonic_order=35
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
        analyzer = StepSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
            trim_samples=trim_samples
        )

        assert 'frequencies' in result
        assert 'thd' in result
        assert len(result['frequencies']) == 4
        assert len(result['thd']) == 4
        assert np.all(result['thd'] >= 0)

    def test_compute_distortion_with_stft(self):
        """Test THD computation for step signal using STFT instead of batch FFT"""
        # Build mask in Phase 1
        from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 500.0,
            'stop_freq': 2000.0,
            'num_steps': 8,
            'total_time': 2.0,
            'repeat_times': 2,
            'sample_rate': 44100
        }

        # Calculate STFT parameters (no trimming)
        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * stimulus_metadata['sample_rate'])
        stft_window_size = step_samples
        stft_hop_size = step_samples  # No overlap - hop equals window

        # Phase 1A: Build overall index (using STFT window size)
        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=stft_window_size, max_harmonic_order=35
        )

        # Phase 1B: Select harmonics and build mask
        harmonic_orders = [2, 3]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Create synthetic recorded signal
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        # Phase 2: Compute THD using STFT
        analyzer = StepSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
            use_stft=True,  # NEW PARAMETER
            stft_window_type='hann'  # NEW PARAMETER
        )

        assert 'frequencies' in result
        assert 'thd' in result
        assert len(result['frequencies']) == 8
        assert len(result['thd']) == 8
        assert np.all(result['thd'] >= 0)
        assert np.all(result['thd'] <= 100)
