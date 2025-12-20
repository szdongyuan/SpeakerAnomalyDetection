"""
HarmonicDistortionAnalyzer - Base class for Phase 2: THD Calculation

Computes THD using pre-built masks from Phase 1B.
"""
import numpy as np
import os
from typing import Dict, Optional
from abc import ABC, abstractmethod
from base.core_algorithm.psychoacoustic.psychoacoustic_utils import (
    spl_to_phons,
    freq_to_bark,
)
from base.core_algorithm.psychoacoustic.mpeg_psychoacoustic_masking import (
    pick_maskers_mpeg1_model1,
    masking_threshold_from_maskers_mpeg1_model1,
)
try:
    from mosqito.sq_metrics.loudness import loudness_zwst_freq
except Exception:
    loudness_zwst_freq = None
else:
    # Allow disabling mosqito for performance testing / environments without a full audio-range freq axis.
    if os.environ.get("PRB_DISABLE_MOSQITO") == "1":
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

    def compute_perceptual_thd_batch(
        self,
        spectrum_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        fundamental_bins: np.ndarray,
        fundamental_freqs: np.ndarray,
        masking_mask_matrix: np.ndarray = None,
        masking_config: dict = None,
        spl_calibration_db: float = 0.0,
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

        Returns:
            perceptual_loudness: (n_frames,) perceived loudness in phons
        """
        n_cols = spectrum_matrix.shape[1]
        masking_config = masking_config or {}
        prb_method = str(masking_config.get("prb_method", "iso")).strip().lower()
        if prb_method not in {"sc", "iso"}:
            prb_method = "sc" if "sc" in prb_method else "iso"

        # ISO method requires mosqito; SC method does not.
        if prb_method == "iso":
            # Strict mode: PRB must be computed via mosqito loudness only (no fallback implementation),
            # otherwise results may differ between environments.
            if loudness_zwst_freq is None:
                raise ImportError(
                    "mosqito is required for PRB loudness computation, but it is not available. "
                    "Install mosqito or remove PRB_DISABLE_MOSQITO."
                )
        if int(self.sample_rate) < 48000:
            raise ValueError(
                "PRB loudness requires analysis sample rate >= 48000 Hz. "
                "If the recording is 44100 Hz, resample to 48000 Hz before running PRB."
            )

        perceptual_loudness = np.zeros(n_cols, dtype=float)
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
        rfft_bark_bins = freq_to_bark(rfft_freqs)

        if prb_method == "sc":
            # "sc" method: Listen/SoundCheck simplified perceptual model (paper PEAQ-SC path).
            # Compute TotalNL (phons) from the full spectrum vs a fundamental-only reference.
            from base.core_algorithm.perceptual_rubbuzz_sc.peaq_loudness_model import PEAQLoudnessConfig, PEAQLoudnessModel

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

            out = sc_model.compute_partial_loudness_from_spectra(full_spectra, fund_spectra)
            totalnl_phons = np.asarray(out.n_total_phons, dtype=np.float64).reshape(-1)

            # Optional paper section 4.10: Error Harmonic Structure (EHS) based on cepstrum peak at 1/f0.
            # This helps separate "harmonic family" buzz from low-order harmonics and noise.
            sc_metric = str(masking_config.get("sc_metric", "totalnl")).strip().lower()
            if sc_metric not in {"totalnl", "totalnl_phons", "ehs", "totalnl_x_ehs"}:
                sc_metric = "totalnl"

            if sc_metric in {"ehs", "totalnl_x_ehs"}:
                f0 = np.asarray(fundamental_freqs, dtype=np.float64).reshape(-1)
                if f0.size != n_cols:
                    raise ValueError(f"fundamental_freqs must have length n_frames={n_cols}, got {f0.shape}")

                ehs = sc_model.compute_error_harmonic_structure(
                    full_spectra,
                    f0,
                    f_min_hz=float(masking_config.get("sc_ehs_min_freq_hz", 20.0)),
                    f_max_hz=masking_config.get("sc_ehs_max_freq_hz"),
                    peak_search_width_bins=int(masking_config.get("sc_ehs_peak_width_bins", 1)),
                    remove_dc=bool(masking_config.get("sc_ehs_remove_dc", False)),
                )

                if sc_metric == "ehs":
                    return np.asarray(ehs, dtype=np.float64).reshape(-1)
                # totalnl_x_ehs: paper later multiplies "Partial Loudness overall level" by "Harmonic Structure overall level".
                return np.asarray(totalnl_phons * ehs, dtype=np.float64).reshape(-1)

            return totalnl_phons

        # Default PRB method: use spreaded specific loudness sampled at the selected harmonic locations.
        # This matches the ecosystem requirement that "per-frequency" loudness is taken from the spreaded
        # Bark-domain loudness profile at the target harmonic points (rather than a full Bark integration).
        prb_loudness_method = str(masking_config.get("prb_loudness_method", "delta_specific")).strip().lower()
        partitions_per_bark = int(masking_config.get("partitions_per_bark", 3))
        rfft_partition_index = np.clip(
            np.floor(rfft_bark_bins * float(partitions_per_bark)).astype(int),
            0,
            int(24 * partitions_per_bark),
        )
        tonal_peak_prominence_db = float(masking_config.get("tonal_peak_prominence_db", 7.0))
        masker_min_over_ath_db = float(masking_config.get("min_over_ath_db", 0.0))
        tonal_neighbor_merge_bins = int(masking_config.get("tonal_neighbor_merge_bins", 1))
        max_tonal_per_partition = int(
            masking_config.get("max_tonal_per_partition", masking_config.get("max_tonal_per_band", 1))
        )
        enable_noise_maskers = bool(masking_config.get("enable_noise_maskers", True))
        min_noise_over_ath_db = float(masking_config.get("min_noise_over_ath_db", 0.0))
        max_total_maskers = int(masking_config.get("max_total_maskers", 64))

        if prb_loudness_method in {"delta_specific", "delta_specific_loudness", "delta_specific_loudness_zwicker"}:
            # Loudness contour difference method:
            # 1) Compute specific loudness N'(z,t) for the full spectrum (Zwicker spreading/filters inside mosqito)
            # 2) Compute specific loudness for the fundamental-only spectrum
            # 3) Subtract contours: N'_dist(z,t) = max(N'_total - N'_f0, 0)
            # 4) For ecosystem-compatibility, *sample* the spreaded specific loudness at the harmonic Bark locations
            #    (rather than integrating over Bark), and convert the resulting pseudo-sones -> phons.

            # Build calibrated full spectrum (exclude dummy row), and drop DC to avoid spurious low-frequency bias.
            full_spectra = (np.asarray(spectrum_matrix[1:, :], dtype=np.float32) * float(calibration_multiplier)).copy()
            low_freq_zero_bins = int(masking_config.get("low_freq_zero_bins", 2))
            if low_freq_zero_bins < 0:
                raise ValueError(f"low_freq_zero_bins must be >= 0, got {low_freq_zero_bins}")
            if full_spectra.shape[0] > 0 and low_freq_zero_bins > 0:
                full_spectra[: min(low_freq_zero_bins, full_spectra.shape[0]), :] = 0.0

            # Fundamental-only spectrum: keep the fundamental bin (and optionally a small neighborhood) per frame.
            fund_neighbor_bins = int(masking_config.get("fundamental_neighbor_bins", 2))
            if fund_neighbor_bins < 0:
                raise ValueError(f"fundamental_neighbor_bins must be >= 0, got {fund_neighbor_bins}")
            fund_spectra = np.zeros_like(full_spectra, dtype=np.float32)

            # spectrum_matrix has dummy row at 0, so FFT bin k maps to row k+1.
            for frame_idx in range(n_cols):
                fbin_row = int(fundamental_bins[frame_idx])
                if fbin_row <= 0:
                    continue
                fbin = fbin_row - 1
                if fbin < 0 or fbin >= n_rfft_bins:
                    continue
                lo = max(0, fbin - fund_neighbor_bins)
                hi = min(n_rfft_bins, fbin + fund_neighbor_bins + 1)
                fund_spectra[lo:hi, frame_idx] = full_spectra[lo:hi, frame_idx]

            # Compute loudness (sones and specific loudness in sones/bark).
            n_total, n_spec_total, bark_axis = loudness_zwst_freq(full_spectra, rfft_freqs, field_type="free")
            n_f0, n_spec_f0, bark_axis_f0 = loudness_zwst_freq(fund_spectra, rfft_freqs, field_type="free")

            n_total = np.asarray(n_total, dtype=np.float64).reshape(-1)
            n_f0 = np.asarray(n_f0, dtype=np.float64).reshape(-1)
            n_spec_total = np.asarray(n_spec_total, dtype=np.float64)
            n_spec_f0 = np.asarray(n_spec_f0, dtype=np.float64)
            # mosqito may return 1D specific loudness for a single frame; normalize to (Nbark, Ntime).
            if n_spec_total.ndim == 1:
                n_spec_total = n_spec_total.reshape(-1, 1)
            if n_spec_f0.ndim == 1:
                n_spec_f0 = n_spec_f0.reshape(-1, 1)
            bark_axis = np.asarray(bark_axis, dtype=np.float64).reshape(-1)
            bark_axis_f0 = np.asarray(bark_axis_f0, dtype=np.float64).reshape(-1)

            if n_total.size != n_cols:
                raise ValueError(f"mosqito returned {n_total.size} frames for total loudness, expected {n_cols}")
            if n_f0.size != n_cols:
                raise ValueError(f"mosqito returned {n_f0.size} frames for f0 loudness, expected {n_cols}")
            if bark_axis.size != bark_axis_f0.size or not np.allclose(bark_axis, bark_axis_f0, atol=0.0, rtol=0.0):
                raise ValueError("mosqito returned inconsistent bark axes for total vs f0 loudness")
            if n_spec_total.shape != n_spec_f0.shape:
                raise ValueError(
                    f"mosqito returned inconsistent specific loudness shapes: total={n_spec_total.shape} f0={n_spec_f0.shape}"
                )

            # Strictness: if we provided non-zero energy but mosqito returned 0 sones for the total spectrum, fail-fast.
            total_energy = np.sum(np.square(full_spectra), axis=0, dtype=np.float64)
            unexpected_zero = (total_energy > 0.0) & (n_total <= 0.0)
            if np.any(unexpected_zero):
                bad = int(np.flatnonzero(unexpected_zero)[0])
                raise ValueError(
                    "mosqito returned 0 sones for a non-silent full-spectrum frame. "
                    f"First bad frame index={bad}, energy={float(total_energy[bad]):.3e}."
                )

            # Specific loudness contour difference (clipped to >= 0).
            n_spec_dist = np.maximum(n_spec_total - n_spec_f0, 0.0)

            def _sones_to_phons(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=np.float64)
                ph = np.zeros_like(arr)
                positive = arr > 0.0
                lt1 = positive & (arr < 1.0)
                ge1 = positive & ~lt1
                ph[lt1] = 40.0 * np.power(arr[lt1], 0.4)
                ph[ge1] = 40.0 + 10.0 * np.log2(arr[ge1])
                return ph

            def _sample_specific_loudness_sones_per_bark(
                n_specific_1d: np.ndarray,
                bark_axis_1d: np.ndarray,
                target_bark: float,
                *,
                edge_window_bark: float,
                edge_aggregation: str,
            ) -> float:
                n_specific_1d = np.asarray(n_specific_1d, dtype=np.float64).reshape(-1)
                bark_axis_1d = np.asarray(bark_axis_1d, dtype=np.float64).reshape(-1)
                if n_specific_1d.size == 0 or bark_axis_1d.size == 0 or n_specific_1d.size != bark_axis_1d.size:
                    raise ValueError(
                        f"Invalid specific loudness arrays: {n_specific_1d.shape} vs {bark_axis_1d.shape}"
                    )
                z0 = float(target_bark)
                bark_min = float(bark_axis_1d[0])
                bark_max = float(bark_axis_1d[-1])

                if bark_min <= z0 <= bark_max:
                    return float(np.interp(z0, bark_axis_1d, n_specific_1d))

                window = float(max(edge_window_bark, 0.0))
                if window <= 0.0:
                    return float(n_specific_1d[0] if z0 < bark_min else n_specific_1d[-1])

                half = 0.5 * window
                if (bark_max - bark_min) <= window:
                    return float(np.mean(n_specific_1d))

                if z0 < bark_min:
                    center = bark_min + half
                else:
                    center = bark_max - half

                lo = center - half
                hi = center + half
                mask = (bark_axis_1d >= lo) & (bark_axis_1d <= hi)
                if not np.any(mask):
                    return float(n_specific_1d[0] if z0 < bark_min else n_specific_1d[-1])

                agg = str(edge_aggregation).strip().lower()
                if agg in {"max", "maximum"}:
                    return float(np.max(n_specific_1d[mask]))
                return float(np.mean(n_specific_1d[mask]))

            # Sample N'_dist at the selected harmonic Bark positions (mask_matrix),
            # using a windowed mean/max near the Bark axis edges to avoid hard-clamping to 24 Bark.
            edge_window_bark = float(masking_config.get("specific_edge_window_bark", 0.6))
            edge_aggregation = str(masking_config.get("specific_edge_aggregation", "mean"))

            pseudo_sones = np.zeros(n_cols, dtype=np.float64)
            for frame_idx in range(n_cols):
                harmonic_mask_col = mask_matrix[:, frame_idx].copy()
                fbin_row = int(fundamental_bins[frame_idx])
                if 0 <= fbin_row < harmonic_mask_col.size:
                    harmonic_mask_col[fbin_row] = 0.0
                harmonic_bin_indices = np.where(harmonic_mask_col > 0)[0]
                harmonic_bin_indices = harmonic_bin_indices[
                    (harmonic_bin_indices > 0) & (harmonic_bin_indices <= n_rfft_bins)
                ]
                if harmonic_bin_indices.size == 0:
                    continue

                # Convert to rFFT bin indices and Bark locations.
                harmonic_rfft_bins = harmonic_bin_indices - 1
                harmonic_barks = rfft_bark_bins[harmonic_rfft_bins]
                n_spec_dist_col = n_spec_dist[:, frame_idx]

                s = 0.0
                for z0 in harmonic_barks:
                    s += _sample_specific_loudness_sones_per_bark(
                        n_spec_dist_col,
                        bark_axis,
                        float(z0),
                        edge_window_bark=edge_window_bark,
                        edge_aggregation=edge_aggregation,
                    )
                pseudo_sones[frame_idx] = s

            return _sones_to_phons(pseudo_sones)

        # Compute loudness in batch using a 2D spectrum (n_freq_bins, n_frames).
        masked_spectra = np.zeros((n_rfft_bins, n_cols), dtype=np.float32)
        # Track per-frame energy to sanity-check mosqito output.
        frame_energy = np.zeros(n_cols, dtype=np.float64)

        # Extract fundamental amplitudes
        row_indices = fundamental_bins.astype(int)
        col_indices = np.arange(n_cols)
        raw_fundamental_amplitudes = spectrum_matrix[row_indices, col_indices]
        fundamental_amplitudes = raw_fundamental_amplitudes * calibration_multiplier

        # Convert amplitude to SPL (dB re 20 μPa) - standard acoustic reference
        reference_pressure = 20e-6
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

            # Masking model (PRB):
            # Build tonal maskers from the *full spectrum* of the current frame, then compute
            # combined masking thresholds at each selected harmonic frequency.
            #
            # This better matches the psychoacoustic notion of "a loud sound masks a quiet one"
            # than restricting maskers to the harmonic subset only.
            frame_amplitudes = spectrum_matrix[1:, frame_idx] * calibration_multiplier
            frame_spls = 20.0 * np.log10(np.maximum(frame_amplitudes / reference_pressure, min_amplitude))
            frame_spls = np.maximum(frame_spls, 0.0)

            fundamental_rfft_bin = int(max(row_indices[frame_idx] - 1, 0))
            forced_bins = None
            if 0 <= fundamental_rfft_bin < rfft_freqs.size:
                forced_bins = np.array([fundamental_rfft_bin], dtype=int)

            maskers = pick_maskers_mpeg1_model1(
                frame_spls,
                rfft_freqs,
                bark_bins=rfft_bark_bins,
                partition_index=rfft_partition_index,
                partitions_per_bark=partitions_per_bark,
                forced_tonal_bins=forced_bins,
                tonal_peak_prominence_db=tonal_peak_prominence_db,
                min_over_ath_db=masker_min_over_ath_db,
                tonal_neighbor_merge_bins=tonal_neighbor_merge_bins,
                max_tonal_per_partition=max_tonal_per_partition,
                enable_noise_maskers=enable_noise_maskers,
                min_noise_over_ath_db=min_noise_over_ath_db,
            )

            masker_freqs = maskers.all_freqs_hz()
            masker_levels = maskers.all_levels_db()
            is_tonal = maskers.all_is_tonal()

            if masker_freqs.size == 0:
                combined_thresholds = np.zeros_like(harmonic_spls)
            else:
                if masker_levels.size > max_total_maskers:
                    keep = np.argsort(masker_levels)[-max_total_maskers:]
                    masker_freqs = masker_freqs[keep]
                    masker_levels = masker_levels[keep]
                    is_tonal = is_tonal[keep]

                combined_thresholds = masking_threshold_from_maskers_mpeg1_model1(
                    masker_freqs_hz=masker_freqs,
                    masker_levels_db=masker_levels,
                    is_tonal=is_tonal,
                    target_freqs_hz=harmonic_freqs,
                )

            # Convert masking thresholds into an *effective* SPL by subtracting in the power domain:
            #   P_eff = max(P_harm - P_thr, 0), SPL_eff = 10*log10(P_eff)
            masked_spls = harmonic_spls.copy()
            has_threshold = combined_thresholds > 0.0
            if np.any(has_threshold):
                harmonic_power = np.power(10.0, harmonic_spls / 10.0)
                threshold_power = np.power(10.0, combined_thresholds / 10.0)
                residual_power = harmonic_power - threshold_power
                residual_power = np.maximum(residual_power, 0.0)
                masked = np.zeros_like(harmonic_spls)
                positive = residual_power > 0.0
                masked[positive] = 10.0 * np.log10(residual_power[positive])
                masked_spls[has_threshold] = masked[has_threshold]
                masked_spls = np.maximum(masked_spls, 0.0)

            # Convert masked SPLs into an amplitude-domain attenuation, so masking affects
            # the loudness computation (not only the audibility gating).
            attenuation_db = masked_spls - harmonic_spls
            attenuation_factors = np.power(10.0, attenuation_db / 20.0)

            # Do not pre-gate by ATH here; mosqito handles inaudible contributions internally.
            # Keep only components with non-zero residual after masking subtraction.
            keep = masked_spls > 0.0
            if np.any(keep):
                masked_spectrum = masked_spectra[:, frame_idx]
                kept_bins = harmonic_rfft_bins[keep]
                masked_spectrum[kept_bins] = harmonic_amplitudes[keep] * attenuation_factors[keep]
                frame_energy[frame_idx] = float(np.sum(np.square(masked_spectrum)))
            else:
                frame_energy[frame_idx] = 0.0

        total_sones, _, _ = loudness_zwst_freq(masked_spectra, rfft_freqs, field_type="free")
        total_sones = np.asarray(total_sones, dtype=np.float64).reshape(-1)
        if total_sones.size != n_cols:
            raise ValueError(f"mosqito returned {total_sones.size} frames, expected {n_cols}")

        # Strictness: if we provided non-zero energy to mosqito but it returned 0 sones,
        # treat this as an invalid computation (often caused by an incompatible `freqs` axis).
        unexpected_zero = (frame_energy > 0.0) & (total_sones <= 0.0)
        if np.any(unexpected_zero):
            bad = int(np.flatnonzero(unexpected_zero)[0])
            raise ValueError(
                "mosqito returned 0 sones for a non-silent frame. "
                f"First bad frame index={bad}, energy={frame_energy[bad]:.3e}. "
                "This indicates an invalid loudness computation (e.g., incomplete frequency coverage)."
            )

        # Convert sones -> phons with the standard piecewise mapping:
        # - N < 1 sone: Ln = 40 * N^0.4
        # - N >= 1 sone: Ln = 40 + 10*log2(N)
        phons = np.zeros_like(total_sones)
        positive = total_sones > 0.0
        lt1 = positive & (total_sones < 1.0)
        ge1 = positive & ~lt1
        phons[lt1] = 40.0 * np.power(total_sones[lt1], 0.4)
        phons[ge1] = 40.0 + 10.0 * np.log2(total_sones[ge1])
        perceptual_loudness = phons

        return perceptual_loudness
