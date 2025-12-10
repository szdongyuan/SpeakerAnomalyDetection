"""
HarmonicDistortionAnalyzer - Base class for Phase 2: THD Calculation

Computes THD using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from base.pre_processing.psychoacoustic_utils import spl_to_phons, apply_masking


class HarmonicDistortionAnalyzer(ABC):
    """Base analyzer for THD calculation with pre-built masks."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @abstractmethod
    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: tuple,
        **kwargs
    ) -> Dict:
        """
        Compute THD using pre-built mask. Must be implemented by subclasses.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config dict
            harmonic_orders: Selected harmonics
            harmonic_mask: Pre-built mask data from Phase 1B
            **kwargs: Additional parameters

        Returns:
            Result dict with 'frequencies', 'thd', etc.
        """
        pass

    def compute_thd_batch(
        self,
        spectrum_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        fundamental_bins: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized THD computation using pre-built mask.

        Formula: THD = sqrt(sum(H_i²)) / sqrt(F² + sum(H_i²)) × 100%

        Args:
            spectrum_matrix: (n_bins+1, n_steps_or_frames) magnitude spectrum with dummy bin
            mask_matrix: (n_bins+1, n_steps_or_frames) binary mask for selected harmonics
            fundamental_bins: (n_steps_or_frames,) indices of fundamental in spectrum

        Returns:
            thd_percentage: (n_steps_or_frames,) THD values in percent
        """
        n_cols = spectrum_matrix.shape[1]

        # Extract fundamental amplitudes (vectorized)
        row_indices = fundamental_bins.astype(int)
        col_indices = np.arange(n_cols)
        fundamental_amplitudes = spectrum_matrix[row_indices, col_indices]

        # Create harmonic-only mask (exclude fundamental)
        harmonic_mask = mask_matrix.copy()
        harmonic_mask[row_indices, col_indices] = 0.0

        # Compute harmonic power (vectorized)
        harmonic_amplitudes_squared = (spectrum_matrix ** 2) * harmonic_mask
        harmonic_power = np.sum(harmonic_amplitudes_squared, axis=0)

        # Compute THD (vectorized)
        fundamental_power = fundamental_amplitudes ** 2
        total_power = fundamental_power + harmonic_power
        safe_total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero

        thd_ratio = np.sqrt(harmonic_power / safe_total_power)
        thd_percentage = thd_ratio * 100.0

        return thd_percentage

    def compute_perceptual_thd_batch(
        self,
        spectrum_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        fundamental_bins: np.ndarray,
        fundamental_freqs: np.ndarray,
        masking_mask_matrix: np.ndarray = None,
        masking_config: dict = None,
        spl_calibration_db: float = 0.0
    ) -> np.ndarray:
        """
        Compute perceptual loudness (in phons) of harmonics using psychoacoustic models.

        Applies ISO 226 equal-loudness contours and simultaneous masking from fundamental.
        Only harmonics above masking threshold contribute to perceived loudness.

        Args:
            spectrum_matrix: (n_bins+1, n_frames) magnitude spectrum with dummy bin
            mask_matrix: (n_bins+1, n_frames) binary mask for selected harmonics
            fundamental_bins: (n_frames,) indices of fundamental in spectrum
            fundamental_freqs: (n_frames,) fundamental frequencies in Hz
            masking_mask_matrix: Optional (n_bins+1, n_frames) binary mask for masking harmonics
            masking_config: Optional dict with keys:
                - 'masking_range': (start, end) harmonic orders
                - 'enable_cumulative': bool
                - 'weight_function': str ('exponential', 'gaussian', etc.)
            spl_calibration_db: Calibration offset in dB (default 0.0).
                Applied in amplitude domain: calibrated_amp = amp * 10^(calibration_db/20)

        Returns:
            perceptual_loudness: (n_frames,) perceived loudness in phons
        """
        n_cols = spectrum_matrix.shape[1]
        perceptual_loudness = np.zeros(n_cols)
        # Convert calibration offset (dB) to linear multiplier so calibration happens
        # in the amplitude domain before the log transform. This prevents very small
        # bins from being artificially lifted by adding a constant dB offset.
        calibration_multiplier = np.power(10.0, spl_calibration_db / 20.0) if spl_calibration_db != 0.0 else 1.0
        min_amplitude = 1e-12  # avoid log(0) after calibration

        # Extract fundamental amplitudes
        row_indices = fundamental_bins.astype(int)
        col_indices = np.arange(n_cols)
        raw_fundamental_amplitudes = spectrum_matrix[row_indices, col_indices]
        fundamental_amplitudes = raw_fundamental_amplitudes * calibration_multiplier

        # Convert amplitude to SPL (dB re 20 μPa) - standard acoustic reference
        reference_pressure = 20e-6
        fundamental_spl_uncalibrated = 20.0 * np.log10(
            np.maximum(raw_fundamental_amplitudes / reference_pressure, min_amplitude)
        )
        fundamental_spl = 20.0 * np.log10(
            np.maximum(fundamental_amplitudes / reference_pressure, min_amplitude)
        )
        # Clip calibrated SPL below 0 dB to 0 (post-calibration floor)
        fundamental_spl = np.maximum(fundamental_spl, 0.0)
        silence_spl_threshold_calibrated = 0.0  # dB SPL (calibrated floor)

        # Process each frame independently
        for frame_idx in range(n_cols):

            # Extract harmonic amplitudes for this frame
            harmonic_mask_col = mask_matrix[:, frame_idx]

            # Exclude fundamental from harmonic mask
            harmonic_mask_col = harmonic_mask_col.copy()
            harmonic_mask_col[row_indices[frame_idx]] = 0.0

            # Find which bins have harmonics
            harmonic_bin_indices = np.where(harmonic_mask_col > 0)[0]

            if len(harmonic_bin_indices) == 0:
                # No harmonics selected
                perceptual_loudness[frame_idx] = 0.0
                continue

            # --- Fullband sones anchor (per-bin, frequency-aware, calibrated) ---
            calibrated_frame_amplitudes = spectrum_matrix[:, frame_idx] * calibration_multiplier
            # Remove fundamental from anchor to avoid coupling harmonic loudness to fundamental level
            fund_bin = row_indices[frame_idx]
            calibrated_frame_amplitudes = calibrated_frame_amplitudes.copy()
            if fund_bin < calibrated_frame_amplitudes.shape[0]:
                calibrated_frame_amplitudes[fund_bin] = 0.0

            calibrated_frame_spl = 20.0 * np.log10(
                np.maximum(calibrated_frame_amplitudes / reference_pressure, min_amplitude)
            )
            n_bins = spectrum_matrix.shape[0] - 1  # Subtract dummy bin
            bin_freqs = np.arange(spectrum_matrix.shape[0]) * (self.sample_rate / 2.0) / n_bins
            fullband_phons = spl_to_phons(bin_freqs, calibrated_frame_spl)
            # Only count audible components for total sones anchor
            audible_fullband = fullband_phons > 0.0
            fullband_sones = np.zeros_like(fullband_phons)
            if np.any(audible_fullband):
                fullband_sones[audible_fullband] = np.power(
                    2.0, (fullband_phons[audible_fullband] - 40.0) / 10.0
                )
            total_sones_fullband = np.sum(fullband_sones)

            # Get harmonic amplitudes
            raw_harmonic_amplitudes = spectrum_matrix[harmonic_bin_indices, frame_idx]
            harmonic_amplitudes = raw_harmonic_amplitudes * calibration_multiplier
            # Convert to SPL (dB re 20 μPa) after calibration
            harmonic_spls = 20.0 * np.log10(
                np.maximum(harmonic_amplitudes / reference_pressure, min_amplitude)
            )
            # Clip calibrated SPL below 0 dB to 0
            harmonic_spls = np.maximum(harmonic_spls, 0.0)

            # If both fundamental and harmonics are at/below calibrated floor, treat frame as silence
            if (
                fundamental_spl[frame_idx] <= silence_spl_threshold_calibrated
                and np.max(harmonic_spls) <= silence_spl_threshold_calibrated
            ):
                perceptual_loudness[frame_idx] = 0.0
                continue

            # Compute harmonic frequencies (from FFT bin index)
            # Frequency = bin_index * sample_rate / (2 * n_bins)
            n_bins = spectrum_matrix.shape[0] - 1  # Subtract dummy bin
            harmonic_freqs = harmonic_bin_indices * (self.sample_rate / 2.0) / n_bins

            # Apply masking (cumulative or fundamental-only)
            if masking_mask_matrix is not None and masking_config and masking_config.get('enable_cumulative'):
                # Extract masking harmonics
                masking_mask_col = masking_mask_matrix[:, frame_idx]
                masking_bin_indices = np.where(masking_mask_col > 0)[0]

                # Filter out fundamental bin to avoid double-counting
                # (fundamental will be prepended explicitly below)
                fundamental_bin = fundamental_bins[frame_idx]
                masking_bin_indices = masking_bin_indices[masking_bin_indices != fundamental_bin]

                if len(masking_bin_indices) > 0:
                    # Extract amplitudes
                    masking_amplitudes = spectrum_matrix[masking_bin_indices, frame_idx] * calibration_multiplier

                    # Convert to SPL (dB re 20 μPa) after calibration
                    masking_spls = 20.0 * np.log10(
                        np.maximum(masking_amplitudes / reference_pressure, min_amplitude)
                    )

                    # Compute frequencies
                    masking_freqs = masking_bin_indices * (self.sample_rate / 2.0) / n_bins

                    # Combine fundamental + masking harmonics
                    all_masker_freqs = np.concatenate([[fundamental_freqs[frame_idx]], masking_freqs])
                    all_masker_spls = np.concatenate([[fundamental_spl[frame_idx]], masking_spls])
                else:
                    # No masking harmonics found, use fundamental only
                    all_masker_freqs = np.array([fundamental_freqs[frame_idx]])
                    all_masker_spls = np.array([fundamental_spl[frame_idx]])

                # Apply cumulative masking
                from base.pre_processing.psychoacoustic_utils import apply_cumulative_masking
                masked_spls = apply_cumulative_masking(
                    all_masker_freqs,
                    all_masker_spls,
                    harmonic_freqs,
                    harmonic_spls,
                    masking_config.get('weight_function', 'exponential')
                )
            else:
                # Use existing fundamental-only masking
                masked_spls = apply_masking(
                    fundamental_freqs[frame_idx],
                    fundamental_spl[frame_idx],
                    harmonic_freqs,
                    harmonic_spls
                )

            # Convert masked SPLs to phons
            audible_indices = masked_spls > 0
            if np.any(audible_indices):
                audible_freqs = harmonic_freqs[audible_indices]
                audible_spls = masked_spls[audible_indices]

                phons_values = spl_to_phons(audible_freqs, audible_spls)

                # Total perceived loudness using sones summation (psychoacoustically correct)
                # Phons are NOT additive; must convert to sones, sum, then convert back
                # Sones = 2^((phons - 40) / 10), Phons = 40 + 10*log2(sones)
                sones_values = np.power(2.0, (phons_values - 40.0) / 10.0)
                harmonic_sones_sum = np.sum(sones_values)

                if harmonic_sones_sum > 0 and total_sones_fullband > 0:
                    # Pure proportional allocation of total sones to unmasked harmonics
                    weights = sones_values / harmonic_sones_sum
                    sones_values = weights * total_sones_fullband
                    total_sones = total_sones_fullband
                else:
                    total_sones = 0.0

                perceptual_loudness[frame_idx] = 40.0 + 10.0 * np.log2(total_sones) if total_sones > 0 else 0.0
            else:
                perceptual_loudness[frame_idx] = 0.0

        return perceptual_loudness
