"""
HarmonicDistortionAnalyzer - Base class for Phase 2: THD Calculation

Computes THD using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
from base.pre_processing.psychoacoustic_utils import spl_to_phons, apply_masking
try:
    from mosqito.sq_metrics.loudness import loudness_zwst_freq
except Exception:
    loudness_zwst_freq = None


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

        Formula: THD = sqrt(sum(H_i²)) / F × 100%

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

        # Compute THD (vectorized): sqrt(sum(H^2)) / F
        fundamental_amplitudes_safe = np.maximum(np.abs(fundamental_amplitudes), 1e-12)
        thd_ratio = np.sqrt(harmonic_power) / fundamental_amplitudes_safe
        thd_percentage = thd_ratio * 100.0

        return thd_percentage

    def _apply_noise_correction(
        self,
        harmonic_bins: np.ndarray,
        harmonic_amplitudes: np.ndarray,
        noise_spectrum: np.ndarray
    ) -> np.ndarray:
        """
        Apply background noise correction to harmonic amplitudes using quadrature subtraction.

        Interpolates noise spectrum to harmonic frequencies and subtracts noise power
        from signal power in quadrature (sqrt(signal^2 - noise^2)).

        Args:
            harmonic_bins: (n_harmonics,) bin indices of harmonics in spectrum
            harmonic_amplitudes: (n_harmonics,) amplitude values at harmonic bins
            noise_spectrum: (n_fft//2 + 1,) background noise magnitude spectrum

        Returns:
            corrected_amplitudes: (n_harmonics,) noise-corrected amplitude values
        """
        # Interpolate noise spectrum to harmonic bin indices
        noise_interp = np.interp(
            harmonic_bins,
            np.arange(len(noise_spectrum)),
            noise_spectrum
        )

        # Quadrature subtraction: corrected = sqrt(signal^2 - noise^2)
        signal_power = harmonic_amplitudes ** 2
        noise_power = noise_interp ** 2

        # Clip to prevent negative values (when noise > signal)
        corrected_power = np.maximum(signal_power - noise_power, 0.0)
        corrected_amplitudes = np.sqrt(corrected_power)

        # Apply minimum threshold to prevent log(0) later
        min_amplitude = 1e-12
        corrected_amplitudes = np.maximum(corrected_amplitudes, min_amplitude)

        return corrected_amplitudes

    def compute_perceptual_thd_batch(
        self,
        spectrum_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        fundamental_bins: np.ndarray,
        fundamental_freqs: np.ndarray,
        masking_mask_matrix: np.ndarray = None,
        masking_config: dict = None,
        spl_calibration_db: float = 0.0,
        noise_spectrum: np.ndarray = None,
        n_fft: Optional[int] = None
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
            spl_calibration_db: SPL calibration offset in dB (default 0.0).
                This is expected to come from microphone SPL calibration (e.g., a 94 dB / 114 dB calibrator).
                It is applied in the amplitude domain:
                    calibrated_pressure_like = raw_voltage_like * 10^(calibration_db/20)
                After proper SPL calibration, the calibrated amplitude can be treated as being in Pascals
                up to a constant that depends on the exact FFT/STFT magnitude scaling.
            noise_spectrum: Optional (n_fft//2 + 1,) background noise magnitude spectrum.
                If provided, applies background noise correction to harmonics.

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

        # Precompute FFT frequency axis (without the dummy bin).
        # spectrum_matrix uses a dummy bin at row 0, so FFT bin k maps to row k+1.
        n_rfft_bins = spectrum_matrix.shape[0] - 1
        if n_rfft_bins <= 0:
            return perceptual_loudness

        if n_fft is None:
            # Best-effort inference. Prefer passing the actual STFT/FFT size via `n_fft`,
            # since `n_rfft_bins` alone cannot disambiguate even vs odd FFT lengths.
            n_fft = max(2 * (n_rfft_bins - 1), 1)

        if not isinstance(n_fft, int) or n_fft <= 0:
            raise ValueError(f"n_fft must be a positive integer, got {n_fft}")

        expected_rfft_bins = (n_fft // 2) + 1
        if expected_rfft_bins != n_rfft_bins:
            raise ValueError(
                "Inconsistent `n_fft` vs `spectrum_matrix` shape: "
                f"n_fft={n_fft} implies {expected_rfft_bins} rFFT bins, "
                f"but spectrum_matrix has {n_rfft_bins} (excluding dummy row)."
            )

        rfft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        noise_n_fft = None
        if noise_spectrum is not None:
            noise_n_fft = max(2 * (len(noise_spectrum) - 1), 1)

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
            # Exclude dummy/sentinel bins (row 0) and any out-of-range indices.
            harmonic_bin_indices = harmonic_bin_indices[
                (harmonic_bin_indices > 0) & (harmonic_bin_indices <= n_rfft_bins)
            ]

            if len(harmonic_bin_indices) == 0:
                # No harmonics selected
                perceptual_loudness[frame_idx] = 0.0
                continue

            harmonic_rfft_bins = harmonic_bin_indices - 1
            harmonic_freqs = rfft_freqs[harmonic_rfft_bins]

            # Get harmonic amplitudes
            raw_harmonic_amplitudes = spectrum_matrix[harmonic_bin_indices, frame_idx]
            harmonic_amplitudes = raw_harmonic_amplitudes * calibration_multiplier

            # Apply noise correction if noise spectrum is provided
            if noise_spectrum is not None:
                noise_bin_positions = harmonic_freqs * noise_n_fft / self.sample_rate
                noise_bin_positions = np.clip(noise_bin_positions, 0.0, len(noise_spectrum) - 1.0)
                harmonic_amplitudes = self._apply_noise_correction(
                    noise_bin_positions,
                    harmonic_amplitudes,
                    noise_spectrum
                )

            # Convert to SPL (dB re 20 μPa) after calibration (and optional noise correction)
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

            # Apply masking (cumulative or fundamental-only)
            if masking_mask_matrix is not None and masking_config and masking_config.get('enable_cumulative'):
                # Extract masking harmonics
                masking_mask_col = masking_mask_matrix[:, frame_idx]
                masking_bin_indices = np.where(masking_mask_col > 0)[0]
                masking_bin_indices = masking_bin_indices[
                    (masking_bin_indices > 0) & (masking_bin_indices <= n_rfft_bins)
                ]

                # Filter out fundamental bin to avoid double-counting
                # (fundamental will be prepended explicitly below)
                fundamental_bin = fundamental_bins[frame_idx]
                masking_bin_indices = masking_bin_indices[masking_bin_indices != fundamental_bin]

                if len(masking_bin_indices) > 0:
                    masking_rfft_bins = masking_bin_indices - 1
                    masking_freqs = rfft_freqs[masking_rfft_bins]

                    # Extract amplitudes
                    masking_amplitudes = spectrum_matrix[masking_bin_indices, frame_idx] * calibration_multiplier
                    if noise_spectrum is not None:
                        noise_bin_positions = masking_freqs * noise_n_fft / self.sample_rate
                        noise_bin_positions = np.clip(noise_bin_positions, 0.0, len(noise_spectrum) - 1.0)
                        masking_amplitudes = self._apply_noise_correction(
                            noise_bin_positions,
                            masking_amplitudes,
                            noise_spectrum
                        )

                    # Convert to SPL (dB re 20 μPa) after calibration
                    masking_spls = 20.0 * np.log10(
                        np.maximum(masking_amplitudes / reference_pressure, min_amplitude)
                    )
                    masking_spls = np.maximum(masking_spls, 0.0)

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
                    harmonic_spls,
                )

            # Convert masked SPLs into an amplitude-domain attenuation, so masking affects
            # the loudness computation (not only the audibility gating).
            attenuation_db = masked_spls - harmonic_spls
            attenuation_factors = np.power(10.0, attenuation_db / 20.0)

            # Convert masked SPLs to phons
            audible_indices = masked_spls > 20
            if np.any(audible_indices):
                # Build calibrated spectrum containing only audible harmonics (fundamental excluded)
                # NOTE: Use the calibrated (and optionally noise-corrected) harmonic amplitudes so
                # noise correction affects loudness, not only audibility gating.
                masked_spectrum = np.zeros_like(rfft_freqs)
                audible_bins = harmonic_rfft_bins[audible_indices]
                masked_spectrum[audible_bins] = (
                    harmonic_amplitudes[audible_indices] * attenuation_factors[audible_indices]
                )

                total_sones = 0.0
                if loudness_zwst_freq is not None:
                    try:
                        total_sones, _, _ = loudness_zwst_freq(
                            masked_spectrum,
                            rfft_freqs,
                            field_type="free"
                        )
                        if isinstance(total_sones, np.ndarray):
                            total_sones = float(np.mean(total_sones))
                    except Exception:
                        total_sones = 0.0

                # Fallback: sum sones of audible harmonics only
                if total_sones == 0.0:
                    audible_freqs = harmonic_freqs[audible_indices]
                    audible_spls = masked_spls[audible_indices]
                    phons_values = spl_to_phons(audible_freqs, audible_spls)
                    # ISO 532 / Stevens: phons<40 follow a power law; >=40 use doubling per 10 phons.
                    sones_values = np.where(
                        phons_values < 40.0,
                        np.power(phons_values / 40.0, 2.5),
                        np.power(2.0, (phons_values - 40.0) / 10.0),
                    )
                    total_sones = np.sum(sones_values)

                if total_sones > 0:
                    # Convert sones -> phons with the standard piecewise mapping:
                    # - N < 1 sone: Ln = 40 * N^0.4
                    # - N >= 1 sone: Ln = 40 + 10*log2(N)
                    if total_sones < 1.0:
                        perceptual_loudness[frame_idx] = 40.0 * np.power(total_sones, 0.4)
                    else:
                        perceptual_loudness[frame_idx] = 40.0 + 10.0 * np.log2(total_sones)
                else:
                    perceptual_loudness[frame_idx] = 0.0
            else:
                perceptual_loudness[frame_idx] = 0.0

        return perceptual_loudness
