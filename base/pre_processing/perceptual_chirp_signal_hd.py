"""
PerceptualChirpSignalHD - Phase 2 perceptual analyzer for chirp signals

Computes perceptual loudness (phons) for chirp signals using psychoacoustic models.
Extends ChirpSignalHD with perceptual THD computation.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from base.pre_processing.chirp_signal_hd import ChirpSignalHD


class PerceptualChirpSignalHD(ChirpSignalHD):
    """Perceptual loudness analyzer for chirp signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray],
        stft_window_size: int = 2048,
        stft_hop_size: int = 1024,
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute perceptual loudness (phons) for chirp signals using psychoacoustic models.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with start_freq, stop_freq, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: (mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins) from Phase 1B
            stft_window_size: STFT window size (default 2048)
            stft_hop_size: STFT hop size (default 512)
            stft_window_type: Window function for STFT (default 'hann')

        Returns:
            {
                'frequencies': fundamental_freqs,
                'times': time_values,
                'perceptual_loudness': loudness_values_in_phons,
                'spectrum_matrix': spectrum
            }
        """
        mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins = harmonic_mask

        # Compute STFT (reuse parent method)
        spectrum_matrix = self._compute_stft(
            recorded_signal, stft_window_size, stft_hop_size, stft_window_type
        )

        # Add dummy bin
        spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)

        # Validate and align frame counts
        num_frames = min(spectrum_with_dummy.shape[1], mask_matrix.shape[1])
        spectrum_trimmed = spectrum_with_dummy[:, :num_frames]
        mask_trimmed = mask_matrix[:, :num_frames]
        fund_bins_trimmed = fundamental_bins[:num_frames]
        fund_freqs_trimmed = fundamental_freqs[:num_frames]

        # Compute perceptual loudness
        perceptual_loudness = self.compute_perceptual_thd_batch(
            spectrum_trimmed, mask_trimmed, fund_bins_trimmed, fund_freqs_trimmed
        )

        # Compute time values
        times = np.arange(num_frames) * stft_hop_size / self.sample_rate

        return {
            'frequencies': fund_freqs_trimmed,
            'times': times,
            'perceptual_loudness': perceptual_loudness,
            'spectrum_matrix': spectrum_trimmed,
            'num_repetitions': 1
        }
