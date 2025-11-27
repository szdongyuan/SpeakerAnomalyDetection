"""
StepSignalHD - Phase 2 analyzer for step signals

Computes THD for step signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class StepSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for step signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray],
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for step signals using pre-built mask and STFT.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: (mask_matrix, fundamental_freqs, fundamental_bins) from Phase 1B
            stft_window_type: Window function for STFT (default 'hann')

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'num_repetitions': repeat_times,
                'spectrum_matrix': averaged_spectrum (with dummy bin at index 0)
            }
        """
        mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask

        num_steps = stimulus_metadata['num_steps']
        repeat_times = stimulus_metadata['repeat_times']
        total_time = stimulus_metadata['total_time']

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        spectrum_per_rep = []
        for repetition_signal in repetitions:
            # Calculate STFT parameters
            single_rep_duration = total_time / repeat_times
            step_duration = single_rep_duration / num_steps
            step_samples = int(step_duration * self.sample_rate)
            stft_window_size = step_samples
            stft_hop_size = step_samples  # No overlap

            # Compute STFT (results in exactly num_steps frames)
            spectrum_matrix = self._compute_stft(
                repetition_signal, stft_window_size, stft_hop_size, stft_window_type
            )

            # Add dummy bin
            spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)

            # Validate and align frame counts (defensive programming)
            num_frames = min(spectrum_with_dummy.shape[1], mask_matrix.shape[1])
            spectrum_trimmed = spectrum_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]

            # Compute THD using pre-built mask
            thd = self.compute_thd_batch(spectrum_trimmed, mask_trimmed, fund_bins_trimmed)
            thd_per_rep.append(thd)
            spectrum_per_rep.append(spectrum_trimmed)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)
        averaged_spectrum = np.mean(spectrum_per_rep, axis=0)

        return {
            'frequencies': fundamental_freqs,
            'thd': averaged_thd,
            'num_repetitions': repeat_times,
            'spectrum_matrix': averaged_spectrum
        }

    def _split_repetitions(self, signal: np.ndarray, repeat_times: int) -> list:
        """Split signal into repetitions."""
        if repeat_times == 1:
            return [signal]

        rep_length = len(signal) // repeat_times
        return [signal[i*rep_length:(i+1)*rep_length] for i in range(repeat_times)]

    def _compute_stft(
        self,
        signal: np.ndarray,
        window_size: int,
        hop_size: int,
        window_type: str
    ) -> np.ndarray:
        """
        Compute STFT magnitude for step signal.

        For step signals with STFT, window_size = step_duration and
        hop_size = step_duration (no overlap), resulting in exactly
        one STFT frame per step.

        Args:
            signal: Input signal (one repetition)
            window_size: STFT window size (equals step duration in samples)
            hop_size: STFT hop size (equals step duration - no overlap)
            window_type: Window function type (e.g., 'hann')

        Returns:
            stft_magnitude: (n_bins, n_steps) magnitude spectrum
        """
        freqs, times, Zxx = scipy_signal.stft(
            signal,
            fs=self.sample_rate,
            window=window_type,
            nperseg=window_size,
            noverlap=window_size - hop_size,
            nfft=window_size,
            return_onesided=True,
            boundary=None,
            padded=False
        )

        return np.abs(Zxx)
