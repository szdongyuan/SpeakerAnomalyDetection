# base/pre_processing/chirp_signal_hd.py
"""
ChirpSignalHD - Phase 2 analyzer for chirp signals

Computes THD for chirp signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class ChirpSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for chirp signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        stft_window_size: int = 2048,
        stft_hop_size: int = 1024,
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for chirp signals using pre-built mask.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference)
            harmonic_mask: (mask_matrix, fund_freqs, time_array, fund_bins) from Phase 1B
            stft_window_size: STFT window size
            stft_hop_size: STFT hop size
            stft_window_type: Window function type

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'times': time_array,
                'num_repetitions': repeat_times
            }
        """
        mask_matrix, fundamental_freqs, time_array, fundamental_bins = harmonic_mask
        repeat_times = stimulus_metadata['repeat_times']

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        for repetition_signal in repetitions:
            # Compute STFT
            stft_magnitude = self._compute_stft(
                repetition_signal, stft_window_size, stft_hop_size, stft_window_type
            )

            # Add dummy bin
            stft_with_dummy = np.insert(stft_magnitude, 0, 0.0, axis=0)

            # Align frame counts (handle boundary effects)
            num_frames = min(stft_with_dummy.shape[1], mask_matrix.shape[1])
            stft_trimmed = stft_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]

            # Compute THD using pre-built mask
            thd = self.compute_thd_batch(stft_trimmed, mask_trimmed, fund_bins_trimmed)
            thd_per_rep.append(thd)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)

        # Trim time and frequency arrays to match
        num_frames = len(averaged_thd)

        return {
            'frequencies': fundamental_freqs[:num_frames],
            'thd': averaged_thd,
            'times': time_array[:num_frames],
            'num_repetitions': repeat_times
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
        """Compute STFT magnitude."""
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
