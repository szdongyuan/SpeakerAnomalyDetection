"""
Comparison tests between FFT (with trimming) and STFT (without trimming) approaches
for step signal harmonic distortion analysis.
"""
import numpy as np
import pytest
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder
from base.pre_processing.step_signal_hd import StepSignalHD


class TestStepSignalFFTvsSTFT:
    def test_both_approaches_produce_valid_results(self):
        """Test that both FFT and STFT approaches produce valid THD measurements"""
        # Setup common parameters
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

        harmonic_orders = [2, 3, 4]
        builder = HarmonicIndexBuilder()

        # Create synthetic test signal with known harmonics
        sr = 44100
        total_samples = int(stimulus_metadata['total_time'] * sr)
        recorded_signal = np.zeros(total_samples)

        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * sr)

        t = np.arange(step_samples) / sr
        fund_freq = 500.0

        # Create signal with fundamental + 2nd harmonic (20% amplitude)
        for rep in range(2):
            for step in range(8):
                start_idx = (rep * 8 + step) * step_samples
                step_signal = np.sin(2 * np.pi * fund_freq * t) + 0.2 * np.sin(2 * np.pi * 2 * fund_freq * t)
                recorded_signal[start_idx:start_idx + step_samples] = step_signal

        # ═══════════════════════════════════════════════════════════════════
        # Approach 1: FFT with trimming
        # ═══════════════════════════════════════════════════════════════════
        trim_samples = 2205
        n_fft_trimmed = step_samples - 2 * trim_samples

        index_matrix_fft, fund_freqs, fft_freqs_fft = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=sr, n_fft=n_fft_trimmed, max_harmonic_order=35
        )

        mask_matrix_fft = builder.create_mask_from_indices(
            index_matrix_fft, harmonic_orders, len(fft_freqs_fft)
        )
        fundamental_bins_fft = index_matrix_fft[:, 1]

        analyzer = StepSignalHD(sample_rate=sr)
        result_fft = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix_fft, fund_freqs, fundamental_bins_fft),
            trim_samples=trim_samples,
            use_stft=False
        )

        # ═══════════════════════════════════════════════════════════════════
        # Approach 2: STFT without trimming
        # ═══════════════════════════════════════════════════════════════════
        n_fft_stft = step_samples

        index_matrix_stft, fund_freqs, fft_freqs_stft = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=sr, n_fft=n_fft_stft, max_harmonic_order=35
        )

        mask_matrix_stft = builder.create_mask_from_indices(
            index_matrix_stft, harmonic_orders, len(fft_freqs_stft)
        )
        fundamental_bins_stft = index_matrix_stft[:, 1]

        result_stft = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix_stft, fund_freqs, fundamental_bins_stft),
            use_stft=True,
            stft_window_type='hann'
        )

        # ═══════════════════════════════════════════════════════════════════
        # Verify both approaches produce valid results
        # ═══════════════════════════════════════════════════════════════════
        assert len(result_fft['thd']) == 8
        assert len(result_stft['thd']) == 8

        # Both should produce non-negative THD values
        assert np.all(result_fft['thd'] >= 0), "FFT THD values should be non-negative"
        assert np.all(result_stft['thd'] >= 0), "STFT THD values should be non-negative"

        # FFT approach (no windowing) should detect harmonics strongly
        # At least the first step should show clear harmonic content
        assert result_fft['thd'][0] > 10, f"FFT first step THD too low: {result_fft['thd'][0]}"

        # STFT approach (with Hann window) will be smaller due to windowing
        # but first step should still show some harmonic content
        assert result_stft['thd'][0] > 5, f"STFT first step THD too low: {result_stft['thd'][0]}"

        # Both should have same number of valid measurements
        assert len(result_fft['thd']) == len(result_stft['thd'])
