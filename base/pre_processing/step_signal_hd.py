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
        trim_samples: int = 2205,
        use_stft: bool = False,
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for step signals using pre-built mask.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: (mask_matrix, fundamental_freqs, fundamental_bins) from Phase 1B
            trim_samples: Samples to trim from step boundaries (only used if use_stft=False)
            use_stft: If True, use STFT instead of batch FFT (no trimming)
            stft_window_type: Window function for STFT (default 'hann')

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
            if use_stft:
                # STFT approach: no splitting, no trimming
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
            else:
                # Original FFT approach: split, trim, batch FFT
                step_segments = self._split_and_trim_steps(
                    repetition_signal, num_steps, trim_samples
                )
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
