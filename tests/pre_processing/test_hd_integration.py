"""Integration tests for three-phase HD workflow."""
import numpy as np
import pytest
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder
from base.pre_processing.step_signal_hd import StepSignalHD
from base.pre_processing.chirp_signal_hd import ChirpSignalHD


class TestHDIntegration:
    def test_three_phase_step_signal_workflow(self):
        """Test complete workflow: Phase 1A → Phase 1B → Phase 2 for step signals"""
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1A: Build Overall Index Matrix (STFT window size)
        # ═══════════════════════════════════════════════════════════════════
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

        builder = HarmonicIndexBuilder()

        # Calculate STFT window size (full step duration)
        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * stimulus_metadata['sample_rate'])
        stft_window_size = step_samples

        # Build overall index with ALL harmonics (1-35)
        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=stft_window_size, max_harmonic_order=35
        )

        assert index_matrix.shape == (16, 36)  # All harmonics

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1B: Select User Configuration
        # ═══════════════════════════════════════════════════════════════════
        harmonic_orders = [2, 3, 4, 5]

        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        assert mask_matrix.shape[1] == 16
        assert np.sum(mask_matrix, axis=0)[0] == 5  # Fund + 4 harmonics

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Calculation (STFT only)
        # ═══════════════════════════════════════════════════════════════════
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        analyzer = StepSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
            stft_window_type='hann'
        )

        assert len(result['frequencies']) == 16
        assert len(result['thd']) == 16
        assert result['num_repetitions'] == 3

    def test_three_phase_chirp_signal_workflow(self):
        """Test complete workflow for chirp signals"""
        # Phase 1A
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 4.0,
            'repeat_times': 2,
            'sample_rate': 44100
        }

        builder = HarmonicIndexBuilder()
        stft_window_size = 2048
        stft_hop_size = 1024

        index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
            stimulus_metadata,
            sr=44100,
            n_fft=stft_window_size,
            hop_length=stft_hop_size,
            max_harmonic_order=35
        )

        # Phase 1B
        harmonic_orders = [2, 3]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Phase 2
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        analyzer = ChirpSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, None, fund_freqs, time_array, fundamental_bins),
            stft_window_size=stft_window_size,
            stft_hop_size=stft_hop_size
        )

        assert len(result['frequencies']) > 0
        assert len(result['thd']) == len(result['frequencies'])
        assert len(result['times']) == len(result['frequencies'])
        assert np.all(result['thd'] >= 0)
