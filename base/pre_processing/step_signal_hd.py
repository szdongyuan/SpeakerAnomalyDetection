"""
StepSignalHD - Phase 2 analyzer for step signals

Computes THD for step signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class StepSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for step signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray],
        trim_samples: int = 2205,
        **kwargs
    ) -> Dict:
        """
        Compute THD for step signals using pre-built mask.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: (mask_matrix, fundamental_freqs, fundamental_bins) from Phase 1B
            trim_samples: Samples to trim from step boundaries

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'num_repetitions': repeat_times
            }
        """
        mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask

        num_steps = stimulus_metadata['num_steps']
        repeat_times = stimulus_metadata['repeat_times']
        total_time = stimulus_metadata['total_time']

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        for repetition_signal in repetitions:
            # Split into steps and trim
            step_segments = self._split_and_trim_steps(
                repetition_signal, num_steps, trim_samples
            )

            # Batch FFT (vectorized)
            spectrum_matrix = self._compute_batch_fft(step_segments)

            # Add dummy bin
            spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)

            # Compute THD using pre-built mask
            thd = self.compute_thd_batch(spectrum_with_dummy, mask_matrix, fundamental_bins)
            thd_per_rep.append(thd)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)

        return {
            'frequencies': fundamental_freqs,
            'thd': averaged_thd,
            'num_repetitions': repeat_times
        }

    def _split_repetitions(self, signal: np.ndarray, repeat_times: int) -> list:
        """Split signal into repetitions."""
        if repeat_times == 1:
            return [signal]

        rep_length = len(signal) // repeat_times
        return [signal[i*rep_length:(i+1)*rep_length] for i in range(repeat_times)]

    def _split_and_trim_steps(
        self, signal: np.ndarray, num_steps: int, trim_samples: int
    ) -> list:
        """Split repetition into steps and trim boundaries."""
        step_samples = len(signal) // num_steps
        step_segments = []

        for step_idx in range(num_steps):
            start = step_idx * step_samples
            step_signal = signal[start:start + step_samples]
            trimmed = step_signal[trim_samples:-trim_samples]
            step_segments.append(trimmed)

        return step_segments

    def _compute_batch_fft(self, segments: list) -> np.ndarray:
        """Compute FFT for all segments in batch (vectorized)."""
        max_len = max(len(seg) for seg in segments)
        n_steps = len(segments)

        # Create zero-padded matrix
        step_matrix = np.zeros((max_len, n_steps))
        for i, seg in enumerate(segments):
            step_matrix[:len(seg), i] = seg

        # Batch FFT
        spectrum_matrix = np.abs(np.fft.rfft(step_matrix, axis=0))

        return spectrum_matrix
