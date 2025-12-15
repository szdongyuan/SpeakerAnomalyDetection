"""
PerceptualStepSignalHD - Phase 2 perceptual analyzer for step signals

Computes perceptual loudness (phons) for step signals using psychoacoustic models.
Extends StepSignalHD with perceptual THD computation.
"""
import numpy as np
from typing import Dict, Tuple
from base.pre_processing.step_signal_hd import StepSignalHD


class PerceptualStepSignalHD(StepSignalHD):
    """Perceptual loudness analyzer for step signals."""

    @staticmethod
    def _pad_or_trim_mask_rows(mask_matrix: np.ndarray, target_rows: int) -> np.ndarray:
        """Pad (with zeros) or trim a mask matrix to a target number of rows."""
        if mask_matrix.shape[0] == target_rows:
            return mask_matrix
        if mask_matrix.shape[0] > target_rows:
            return mask_matrix[:target_rows, :]
        padded = np.zeros((target_rows, mask_matrix.shape[1]), dtype=mask_matrix.dtype)
        padded[:mask_matrix.shape[0], :] = mask_matrix
        return padded

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        stft_window_type: str = 'hann',
        masking_config: Dict = None,
        spl_calibration_db: float = 0.0,
        noise_spectrum: np.ndarray = None,
        **kwargs
    ) -> Dict:
        """
        Compute perceptual loudness (phons) for step signals using psychoacoustic models.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: Either:
                - 3-tuple: (mask_matrix, fundamental_freqs, fundamental_bins) - legacy format
                - 4-tuple: (mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins)
                - None to create automatically
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
                'perceptual_loudness': loudness_values_in_phons,
                'num_repetitions': repeat_times,
                'spectrum_matrix': averaged_spectrum
            }
        """
        # Create harmonic mask if not provided
        if harmonic_mask is None:
            harmonic_mask = self._create_harmonic_mask(
                stimulus_metadata, harmonic_orders, masking_config=masking_config
            )

        # Support both old 3-tuple and new 4-tuple for backward compatibility
        if len(harmonic_mask) == 3:
            mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask
            masking_mask_matrix = None
        elif len(harmonic_mask) == 4:
            mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask
        else:
            raise ValueError(f"harmonic_mask must be 3-tuple or 4-tuple, got {len(harmonic_mask)}")

        # If cumulative masking is enabled, ensure masking_mask_matrix is available
        if masking_config and masking_config.get('enable_cumulative', False):
            if masking_mask_matrix is None:
                raise ValueError("enable_cumulative=True requires masking_mask_matrix (4-tuple harmonic_mask)")

        num_steps = stimulus_metadata['num_steps']
        repeat_times = stimulus_metadata['repeat_times']
        total_time = stimulus_metadata['total_time']

        # Split into repetitions (reuse parent method)
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        perceptual_loudness_per_rep = []
        spectrum_per_rep = []

        for repetition_signal in repetitions:
            # Calculate STFT parameters (reuse parent logic)
            single_rep_duration = total_time / repeat_times
            step_duration = single_rep_duration / num_steps
            step_samples = int(step_duration * self.sample_rate)
            if step_samples < 2:
                raise ValueError(
                    f"Step duration too short for STFT: step_samples={step_samples}. "
                    f"Check total_time={total_time}, repeat_times={repeat_times}, num_steps={num_steps}."
                )
            stft_window_size = step_samples
            stft_hop_size = step_samples

            # Compute STFT (reuse parent method)
            spectrum_matrix = self._compute_stft(
                repetition_signal, stft_window_size, stft_hop_size, stft_window_type
            )

            # Add dummy bin
            spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)
            target_rows = spectrum_with_dummy.shape[0]
            mask_matrix_aligned = self._pad_or_trim_mask_rows(mask_matrix, target_rows)
            masking_mask_matrix_aligned = None
            if masking_mask_matrix is not None:
                masking_mask_matrix_aligned = self._pad_or_trim_mask_rows(masking_mask_matrix, target_rows)

            # Validate and align frame counts
            num_frames = min(spectrum_with_dummy.shape[1], mask_matrix_aligned.shape[1])
            spectrum_trimmed = spectrum_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix_aligned[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]
            fund_freqs_trimmed = fundamental_freqs[:num_frames]

            # Trim masking mask if present
            masking_mask_trimmed = None
            if masking_mask_matrix_aligned is not None:
                masking_mask_trimmed = masking_mask_matrix_aligned[:, :num_frames]

            # Compute perceptual loudness with masking config and calibration
            perceptual_loudness = self.compute_perceptual_thd_batch(
                spectrum_trimmed,
                mask_trimmed,
                fund_bins_trimmed,
                fund_freqs_trimmed,
                masking_mask_matrix=masking_mask_trimmed,
                masking_config=masking_config,
                spl_calibration_db=spl_calibration_db,
                noise_spectrum=noise_spectrum,
                n_fft=stft_window_size
            )

            perceptual_loudness_per_rep.append(perceptual_loudness)
            spectrum_per_rep.append(spectrum_trimmed)

        # Average across repetitions
        averaged_loudness = np.mean(perceptual_loudness_per_rep, axis=0)
        averaged_spectrum = np.mean(spectrum_per_rep, axis=0)
        num_frames = len(averaged_loudness)

        return {
            'frequencies': fundamental_freqs[:num_frames],
            'perceptual_loudness': averaged_loudness,
            'num_repetitions': repeat_times,
            'spectrum_matrix': averaged_spectrum
        }
