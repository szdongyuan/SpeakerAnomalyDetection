"""
PerceptualChirpSignalHD - Phase 2 perceptual analyzer for chirp signals

Computes perceptual loudness (phons) for chirp signals using psychoacoustic models.
Extends ChirpSignalHD with perceptual THD computation.
"""
import numpy as np
from typing import Dict, Tuple
from base.pre_processing.chirp_signal_hd import ChirpSignalHD


class PerceptualChirpSignalHD(ChirpSignalHD):
    """Perceptual loudness analyzer for chirp signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple = None,
        stft_window_size: int = 2048,
        stft_hop_size: int = 1024,
        stft_window_type: str = 'hann',
        masking_config: Dict = None,
        spl_calibration_db: float = 0.0,
        noise_spectrum: np.ndarray = None,
        **kwargs
    ) -> Dict:
        """
        Compute perceptual loudness (phons) for chirp signals using psychoacoustic models.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with start_freq, stop_freq, total_time.
                Optional fields: repeat_times (default 1), stimulus_type (default 'linear')
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: Either:
                - 4-tuple: (mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins) - legacy format
                - 5-tuple: (mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins)
                - None to create automatically
            stft_window_size: STFT window size (default 2048)
            stft_hop_size: STFT hop size (default 1024)
            stft_window_type: Window function for STFT (default 'hann')
            masking_config: Optional masking configuration dict with keys:
                - 'masking_range': (start, end) harmonic orders for masking
                - 'enable_cumulative': bool to enable cumulative masking
                - 'weight_function': str ('exponential', 'gaussian', etc.)
            spl_calibration_db: Calibration offset in dB (default 0.0)
            noise_spectrum: Optional (n_fft//2 + 1,) background noise magnitude spectrum

        Returns:
            {
                'frequencies': fundamental_freqs,
                'times': time_values,
                'perceptual_loudness': loudness_values_in_phons,
                'spectrum_matrix': spectrum,
                'num_repetitions': repeat_times from metadata (default 1)
            }
        """
        # Create harmonic mask if not provided
        if harmonic_mask is None:
            # Validate repeat_times if provided
            if 'repeat_times' in stimulus_metadata:
                rt = stimulus_metadata['repeat_times']
                if not isinstance(rt, int) or rt <= 0:
                    raise ValueError(f"repeat_times must be a positive integer, got {rt}")
            else:
                stimulus_metadata['repeat_times'] = 1  # Default to single repetition

            if 'stimulus_type' not in stimulus_metadata:
                stimulus_metadata['stimulus_type'] = 'linear'  # Default to linear chirp

            harmonic_mask = self._create_harmonic_mask(
                stimulus_metadata, harmonic_orders,
                stft_window_size, stft_hop_size,
                masking_config=masking_config
            )

        # Support both old 4-tuple and new 5-tuple for backward compatibility
        if len(harmonic_mask) == 4:
            # Old format without time_array
            mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask
            time_array = None
        elif len(harmonic_mask) == 5:
            # New format with time_array
            mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins = harmonic_mask
        else:
            raise ValueError(f"harmonic_mask must be 4-tuple or 5-tuple, got {len(harmonic_mask)}")

        # If cumulative masking is enabled, ensure masking_mask_matrix is available
        if masking_config and masking_config.get('enable_cumulative', False):
            if masking_mask_matrix is None:
                raise ValueError("enable_cumulative=True requires masking_mask_matrix (4-tuple or 5-tuple harmonic_mask)")

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

        # Trim masking mask if present
        masking_mask_trimmed = None
        if masking_mask_matrix is not None:
            masking_mask_trimmed = masking_mask_matrix[:, :num_frames]

        # Compute perceptual loudness with masking config and calibration
        perceptual_loudness = self.compute_perceptual_thd_batch(
            spectrum_trimmed,
            mask_trimmed,
            fund_bins_trimmed,
            fund_freqs_trimmed,
            masking_mask_matrix=masking_mask_trimmed,
            masking_config=masking_config,
            spl_calibration_db=spl_calibration_db,
            noise_spectrum=noise_spectrum
        )

        # Compute time values
        times = np.arange(num_frames) * stft_hop_size / self.sample_rate

        # Get actual repeat_times from metadata (default to 1 if not present)
        # Note: Averaging across repetitions happens in higher-level callers if needed
        num_repetitions = stimulus_metadata.get('repeat_times', 1)

        return {
            'frequencies': fund_freqs_trimmed,
            'times': times,
            'perceptual_loudness': perceptual_loudness,
            'spectrum_matrix': spectrum_trimmed,
            'num_repetitions': num_repetitions
        }
