"""
HarmonicDistortionAnalyzer - Base class for Phase 2: THD Calculation

Computes THD using pre-built masks from Phase 1B.
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod

from base.core_algorithm.harmonic_distortion.peaq_sc_loudness_model import PEAQLoudnessConfig, PEAQLoudnessModel


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
        harmonic_amplitudes_squared = (spectrum_matrix**2) * harmonic_mask
        harmonic_power = np.sum(harmonic_amplitudes_squared, axis=0)

        # Compute THD (vectorized): sqrt(sum(H^2)) / F
        fundamental_amplitudes_safe = np.maximum(np.abs(fundamental_amplitudes), 1e-12)
        thd_ratio = np.sqrt(harmonic_power) / fundamental_amplitudes_safe
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
        v2pa_factor: float = 1.0,
        n_fft: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute perceptual rub & buzz indicator using the SoundCheck/Listen (SC) loudness model.

        Args:
            spectrum_matrix: (n_bins+1, n_frames) magnitude spectrum with dummy bin
            mask_matrix: (n_bins+1, n_frames) binary mask for selected harmonics
            fundamental_bins: (n_frames,) indices of fundamental in spectrum
            fundamental_freqs: (n_frames,) fundamental frequencies in Hz
            masking_mask_matrix: Optional (n_bins+1, n_frames) binary mask for masking harmonics
            masking_config: Optional dict for SC tuning parameters.
            v2pa_factor: Microphone calibration multiplier (V -> Pa), default 1.0.
                Applied in the amplitude domain:
                    calibrated_pressure_like = raw_voltage_like * v2pa_factor

        Returns:
            perceptual_loudness: (n_frames,) perceived loudness in phons
        """
        masking_config = masking_config or {}
        prb_method = str(masking_config.get("prb_method", "sc")).strip().lower()
        if prb_method in {"", "soundcheck", "peaq_sc", "peaq-sc"}:
            prb_method = "sc"
        if prb_method != "sc":
            raise ValueError(f"Unsupported PRB method: {prb_method!r}. Only 'sc' is supported.")
        if int(self.sample_rate) < 48000:
            raise ValueError(
                "PRB loudness requires analysis sample rate >= 48000 Hz. "
                "If the recording is 44100 Hz, resample to 48000 Hz before running PRB."
            )

        n_cols = spectrum_matrix.shape[1]
        perceptual_loudness = np.zeros(n_cols, dtype=float)
        if not v2pa_factor:
            v2pa_factor = 1.0
        calibration_multiplier = float(v2pa_factor)

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

        sc_cfg_kwargs = {}
        if "sc_ear_term3_exponent" in masking_config:
            sc_cfg_kwargs["ear_weighting_term3_exponent"] = float(masking_config["sc_ear_term3_exponent"])
        if "sc_ear_term3_coeff" in masking_config:
            sc_cfg_kwargs["ear_weighting_term3_coeff"] = float(masking_config["sc_ear_term3_coeff"])

        sc_model = PEAQLoudnessModel(rfft_freqs, config=PEAQLoudnessConfig(**sc_cfg_kwargs))

        # SoundCheck/Listen ecosystem compatibility:
        # apply an additional fixed amplitude scaling after SPL calibration.
        sc_post_calibration_multiplier = float(masking_config.get("sc_post_calibration_multiplier", 0.075))
        if not np.isfinite(sc_post_calibration_multiplier) or sc_post_calibration_multiplier <= 0.0:
            raise ValueError(
                "sc_post_calibration_multiplier must be a finite positive number, "
                f"got {sc_post_calibration_multiplier}"
            )
        sc_total_multiplier = float(calibration_multiplier) * sc_post_calibration_multiplier

        full_spectra = (np.asarray(spectrum_matrix[1:, :], dtype=np.float64) * sc_total_multiplier).copy()
        # Drop very low-frequency rFFT bins to avoid bias from drift / rumble.
        # `low_freq_zero_bins=2` means: set bins [0,1] (DC + first bin) to 0.
        low_freq_zero_bins = int(masking_config.get("low_freq_zero_bins", 2))
        if low_freq_zero_bins < 0:
            raise ValueError(f"low_freq_zero_bins must be >= 0, got {low_freq_zero_bins}")
        if full_spectra.shape[0] > 0 and low_freq_zero_bins > 0:
            full_spectra[: min(low_freq_zero_bins, full_spectra.shape[0]), :] = 0.0

        # Fundamental-only reference spectrum (optionally include a small neighborhood around the bin).
        fund_neighbor_bins = int(masking_config.get("fundamental_neighbor_bins", 2))
        if fund_neighbor_bins < 0:
            raise ValueError(f"fundamental_neighbor_bins must be >= 0, got {fund_neighbor_bins}")
        fund_spectra = np.zeros_like(full_spectra, dtype=np.float64)
        for frame_idx in range(n_cols):
            fbin_row = int(fundamental_bins[frame_idx])
            if fbin_row <= 0:
                continue
            fbin = fbin_row - 1  # drop dummy row offset
            if fbin < 0 or fbin >= n_rfft_bins:
                continue
            lo = max(0, fbin - fund_neighbor_bins)
            hi = min(n_rfft_bins, fbin + fund_neighbor_bins + 1)
            fund_spectra[lo:hi, frame_idx] = full_spectra[lo:hi, frame_idx]

        # Optional paper section 4.10: Error Harmonic Structure (EHS) based on cepstrum peak at 1/f0.
        # This helps separate "harmonic family" buzz from low-order harmonics and noise.
        # Paper-aligned default: use the combined indicator TotalNL×EHS unless explicitly overridden.
        sc_metric = str(masking_config.get("sc_metric", "totalnl_x_ehs")).strip().lower()
        if sc_metric not in {"totalnl", "totalnl_phons", "ehs", "totalnl_x_ehs"}:
            sc_metric = "totalnl_x_ehs"

        totalnl_phons: np.ndarray | None = None
        if sc_metric in {"totalnl", "totalnl_phons", "totalnl_x_ehs"}:
            out = sc_model.compute_partial_loudness_from_spectra(full_spectra, fund_spectra)
            totalnl_phons = np.asarray(out.n_total_phons, dtype=np.float64).reshape(-1)

        if sc_metric in {"ehs", "totalnl_x_ehs"}:
            f0 = np.asarray(fundamental_freqs, dtype=np.float64).reshape(-1)
            if f0.size != n_cols:
                raise ValueError(f"fundamental_freqs must have length n_frames={n_cols}, got {f0.shape}")

            # Compute EHS on a harmonic-line spectrum (yellow curve in cepstrum plots) by default.
            # This suppresses the broadband spectral envelope / cepstrum low-quefrency ramp that makes
            # absolute cepstrum magnitudes hard to compare across different f0 values.
            use_harmonic_line = bool(masking_config.get("sc_ehs_use_harmonic_line_spectrum", True))
            ehs_input_spectra = full_spectra
            if use_harmonic_line:
                max_order = int(masking_config.get("sc_ehs_max_harmonic_order", 200))
                if max_order <= 0:
                    raise ValueError(f"sc_ehs_max_harmonic_order must be > 0, got {max_order}")
                min_order = int(masking_config.get("sc_ehs_min_harmonic_order", 1))
                if min_order <= 0:
                    raise ValueError(f"sc_ehs_min_harmonic_order must be > 0, got {min_order}")
                if min_order > max_order:
                    raise ValueError(
                        "sc_ehs_min_harmonic_order must be <= sc_ehs_max_harmonic_order, "
                        f"got min={min_order} max={max_order}"
                    )
                neighbor_bins = int(masking_config.get("sc_ehs_harmonic_neighbor_bins", 1))
                if neighbor_bins < 0:
                    raise ValueError(f"sc_ehs_harmonic_neighbor_bins must be >= 0, got {neighbor_bins}")

                bin_hz = float(self.sample_rate) / float(n_fft)
                if not np.isfinite(bin_hz) or bin_hz <= 0.0:
                    raise ValueError(f"Unable to infer FFT bin spacing from sample_rate={self.sample_rate}, n_fft={n_fft}")

                nyquist_hz = float(self.sample_rate) / 2.0
                offsets = np.arange(-neighbor_bins, neighbor_bins + 1, dtype=np.int64)
                max_bin = int(n_rfft_bins - 1)
                harmonic_line = np.zeros_like(full_spectra, dtype=np.float64)
                for frame_idx in range(n_cols):
                    f0_i = float(f0[frame_idx])
                    if not np.isfinite(f0_i) or f0_i <= 0.0:
                        continue
                    max_order_frame = min(max_order, int(np.floor(nyquist_hz / f0_i)))
                    if max_order_frame < min_order:
                        continue
                    orders = np.arange(min_order, max_order_frame + 1, dtype=np.float64)
                    harmonic_bins = np.rint((orders * f0_i) / bin_hz).astype(np.int64, copy=False)
                    harmonic_bins = np.clip(harmonic_bins, 1, max_bin)
                    if harmonic_bins.size == 0:
                        continue
                    if offsets.size == 1:
                        bins = np.unique(harmonic_bins)
                    else:
                        bins = np.unique(np.clip(harmonic_bins.reshape(-1, 1) + offsets.reshape(1, -1), 1, max_bin))
                        bins = bins.reshape(-1)
                    harmonic_line[bins, frame_idx] = full_spectra[bins, frame_idx]
                ehs_input_spectra = harmonic_line

            # Robust low-frequency filtering: if sc_ehs_min_freq_hz is not specified or is None,
            # automatically compute as max(40.0, min(f0) * sc_ehs_min_f0_ratio).
            ehs_f_min = masking_config.get("sc_ehs_min_freq_hz")
            if ehs_f_min is not None:
                ehs_f_min = float(ehs_f_min)
            ehs_f0_ratio = float(masking_config.get("sc_ehs_min_f0_ratio", 0.8))

            ehs = sc_model.compute_error_harmonic_structure(
                ehs_input_spectra,
                f0,
                f_min_hz=ehs_f_min,
                f_min_f0_ratio=ehs_f0_ratio,
                f_max_hz=masking_config.get("sc_ehs_max_freq_hz"),
                peak_search_width_bins=int(masking_config.get("sc_ehs_peak_width_bins", 1)),
                remove_dc=bool(masking_config.get("sc_ehs_remove_dc", False)),
                subtract_f0_masking=bool(masking_config.get("sc_ehs_subtract_f0_masking", True)),
            )

            if sc_metric == "ehs":
                return np.asarray(ehs, dtype=np.float64).reshape(-1)
            # totalnl_x_ehs: paper later multiplies "Partial Loudness overall level" by "Harmonic Structure overall level".
            if totalnl_phons is None:
                raise ValueError("Internal error: totalnl_phons was not computed for sc_metric='totalnl_x_ehs'")
            return np.asarray(totalnl_phons * ehs, dtype=np.float64).reshape(-1)

        if totalnl_phons is None:
            raise ValueError("Internal error: totalnl_phons was not computed for sc_metric='totalnl'")
        return totalnl_phons
