"""
Listen / AES 127th (2009) simplified PEAQ loudness model (FFT version).

Used by perceptual harmonic distortion (PRB-SC) analysis (SC-only).

Source (paper in this repo):
`docs/paper/perceptualRub&Buzz.md`
  "Practical Measurement of Loudspeaker Distortion Using a Simplified Auditory Perceptual Model"
  Steve Temme, Pascal Brunet, Don Keele Jr.

This module intentionally does NOT use mosqito or the existing masking code paths.
It implements the paper's processing chain starting from FFT/STFT spectra:

  1) Level adaptation (4.3)                     [implemented as max-bin matching]
  2) Ear frequency weighting W[k]              (Eq.1) + apply to spectrum (Eq.2)
  3) Bark bands / auditory filter groups        (Eq.3) + energy mapping to Pe[k]
  4) Add internal noise P_thres[k]              (Eq.4) to get pitch patterns Pp[k]
  5) Frequency spreading -> excitation E[k]     (Eq.5-8, gamma=0.4, res=0.25 Bark)
  6) Loudness N[k] (specific) + N_total         (Eq.9-12)
  7) Partial loudness NL[k] + TotalNL           (Eq.13-15)

The algorithm is defined for stationary signals; for STFT usage, run it per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from base.core_algorithm.harmonic_distortion.peaq_sc_loudness_result import PEAQLoudnessResult
from base.core_algorithm.harmonic_distortion.peaq_sc_utils import (
    bark_to_hz_peaq,
    hz_to_bark_peaq,
    sones_to_phons,
    _ear_weighting_db_parametric,
)


@dataclass(frozen=True)
class PEAQLoudnessConfig:
    # Paper parameters / defaults.
    z_min_hz: float = 91.7
    # Upper frequency coverage for Bark bands.
    # - None: derive from the provided rFFT frequency axis (up to Nyquist).
    # - float: cap at min(z_max_hz, Nyquist).
    z_max_hz: Optional[float] = None

    # Bark band layout. Paper uses 0.25 Bark and 109 bands (~91.7 Hz to ~17.7 kHz).
    # If n_bands is None, it is derived from z_min_hz/z_max_hz and res_bark to cover the desired range.
    n_bands: Optional[int] = None
    res_bark: float = 0.25
    gamma_mix: float = 0.4
    alpha_mask: float = 1.5  # Eq.14
    beta_pow: float = 0.23  # paper: growth exponent for loudness

    # Eq.1 ear weighting. Default matches the paper; can be tweaked to emulate other implementations.
    ear_weighting_term3_coeff: float = -1.0e-3
    ear_weighting_term3_exponent: float = 3.6

    # SPL reference for converting SPL(dB) <-> Pa.
    reference_pressure_pa: float = 20e-6

    # Calibration: choose "const" so that a 1 kHz, 100 dB SPL tone yields 64 sones (== 100 phons).
    calibrate_const: bool = True
    calibration_tone_freq_hz: float = 1000.0
    calibration_tone_spl_db: float = 100.0
    calibration_target_sones: float = 64.0

    # Level adaptation (4.3) strategy when a reference spectrum is provided.
    level_adaptation: Literal["none", "match_max_bin"] = "match_max_bin"


class PEAQLoudnessModel:
    """
    Per-frame simplified PEAQ loudness model from the Listen paper.

    Typical usage for STFT frames:
        model = PEAQLoudnessModel(rfft_freqs_hz, config=PEAQLoudnessConfig())
        out = model.compute_partial_loudness_from_spectra(test_pa, ref_pa)
        total_nl_sones = out.total_nl_sones
    """

    def __init__(self, rfft_freqs_hz: np.ndarray, *, config: Optional[PEAQLoudnessConfig] = None):
        self.config = config or PEAQLoudnessConfig()

        freqs = np.asarray(rfft_freqs_hz, dtype=np.float64).reshape(-1)
        if freqs.ndim != 1 or freqs.size < 2:
            raise ValueError("rfft_freqs_hz must be a 1D array with at least 2 entries")
        if np.any(freqs < 0.0) or not np.all(np.isfinite(freqs)):
            raise ValueError("rfft_freqs_hz must be finite and >= 0")
        if not np.all(np.diff(freqs) >= 0.0):
            raise ValueError("rfft_freqs_hz must be non-decreasing")
        self.rfft_freqs_hz = freqs

        # SPL reference used to convert Pa <-> dB SPL. Internally we work in *power ratio* units (p^2/p_ref^2).
        self._p_ref = float(self.config.reference_pressure_pa)
        self._p_ref2 = max(self._p_ref**2, 1e-300)

        self._w_db = _ear_weighting_db_parametric(
            self.rfft_freqs_hz,
            term3_coeff=float(self.config.ear_weighting_term3_coeff),
            term3_exponent=float(self.config.ear_weighting_term3_exponent),
        )

        # Bark-domain auditory filter groups (paper 4.5). The paper uses 109 bands up to ~17.7 kHz,
        # but we can extend up to Nyquist so tones like 20 kHz are not silently dropped.
        z_min = float(hz_to_bark_peaq(np.array([self.config.z_min_hz], dtype=np.float64))[0])
        res = float(self.config.res_bark)
        if res <= 0.0:
            raise ValueError(f"res_bark must be > 0, got {res}")

        n_bands_cfg = self.config.n_bands
        if n_bands_cfg is not None:
            n_bands = int(n_bands_cfg)
            if n_bands <= 0:
                raise ValueError(f"n_bands must be > 0, got {n_bands}")
        else:
            z_max_hz_cfg = self.config.z_max_hz
            z_max_hz = float(z_max_hz_cfg) if z_max_hz_cfg is not None else float(self.rfft_freqs_hz[-1])
            z_max_hz = min(z_max_hz, float(self.rfft_freqs_hz[-1]))
            z_max_bark = float(hz_to_bark_peaq(np.array([z_max_hz], dtype=np.float64))[0])
            # Choose band count so that the last band's upper half-step boundary covers z_max_bark:
            #   z_min + (n_bands - 0.5) * res >= z_max_bark
            # -> n_bands >= (z_max_bark - z_min)/res + 0.5
            n_bands = int(np.ceil(((z_max_bark - z_min) / res) + 0.5))
            n_bands = max(n_bands, 1)

        self._band_z0 = z_min
        self._band_z = z_min + np.arange(n_bands, dtype=np.float64) * res
        self._band_fc_hz = bark_to_hz_peaq(self._band_z)

        # Map rFFT bins -> Bark groups via 0.25-Bark intervals around the centers.
        bin_bark = hz_to_bark_peaq(self.rfft_freqs_hz)
        # Centers are z0 + i*res; assign bins using half-step boundaries.
        band_idx = np.floor((bin_bark - (z_min - 0.5 * res)) / res).astype(int)
        valid = (band_idx >= 0) & (band_idx < n_bands)
        self._bin_valid = valid
        self._bin_band = band_idx
        self._valid_bin_indices = np.flatnonzero(valid)
        self._valid_band_indices = self._bin_band[self._valid_bin_indices].astype(np.int64, copy=False)

        # Internal noise (paper Eq.4), interpreted as dB SPL and converted to linear power ratio (p^2/p_ref^2).
        fc_khz = np.maximum(self._band_fc_hz / 1000.0, 1e-12)
        p_thres_db = 0.4 * 3.64 * np.power(fc_khz, -0.8)
        self._p_thres_ratio = np.power(10.0, p_thres_db / 10.0)

        # Loudness threshold terms (Eq.10 / Eq.11), interpreted as dB SPL and converted to linear power ratio.
        e_thres_db = 0.364 * np.power(fc_khz, -0.8)
        self._e_thres_ratio = np.power(10.0, e_thres_db / 10.0)

        f = np.maximum(self._band_fc_hz, 1e-12)
        s_db = -2.0 - 2.05 * np.arctan(f / 4000.0) - 0.75 * np.arctan(np.square(f / 1600.0))
        # In the loudness equations, s[k] is used as a linear factor inside (1 - s + ...),
        # so we convert from dB to linear.
        self._s_lin = np.power(10.0, s_db / 10.0)

        # Precompute constant pieces for frequency spreading (Eq.5-8).
        self._gamma = float(self.config.gamma_mix)
        if not (0.0 < self._gamma < 2.0):
            raise ValueError(f"gamma_mix must be in (0,2), got {self._gamma}")
        self._res = res
        self._z = int(n_bands)

        k = np.arange(self._z, dtype=np.float64)
        # d_idx[k,j] = res*(k-j) in Bark.
        self._d_idx = (k[:, None] - k[None, :]) * self._res
        self._lower_mask = self._d_idx < 0.0  # k < j
        self._slope_l_db_per_bark = 27.0  # Eq.6
        # Lower-side terms are constant (depend only on index difference and slope_l).
        self._term_lower = np.power(10.0, (self._d_idx * self._slope_l_db_per_bark) / 10.0)  # ok for k<j
        self._term_lower = np.where(self._lower_mask, self._term_lower, 0.0)

        # NormSP[k] base curve (paper text after Eq.8), computed with L[j] == 0 for all j.
        self._norm_sp = self._compute_norm_sp()
        if np.any(self._norm_sp <= 0.0) or not np.all(np.isfinite(self._norm_sp)):
            raise ValueError("Computed NormSP is not finite/positive; check configuration")

        # Calibration constant "const" (paper 4.8 / 4.9).
        # Default to 1.0 so helper methods can be used during calibration.
        self._const = 1.0
        if self.config.calibrate_const:
            self._const = float(self._calibrate_const())

    @property
    def scaling_const(self) -> float:
        return float(self._const)

    def _compute_slope_u_db_per_bark(self, l_db: np.ndarray, fc_hz: np.ndarray) -> np.ndarray:
        # Paper Eq.5: Su = min(0, -24 - 230/f + 0.2*L)
        f = np.asarray(fc_hz, dtype=np.float64)
        l_db = np.asarray(l_db, dtype=np.float64)
        su = -24.0 - (230.0 / np.maximum(f, 1e-12)) + 0.2 * l_db
        return np.minimum(su, 0.0)

    def _compute_norm_sp(self) -> np.ndarray:
        # L[j] == 0 dB -> 10^(L/10) == 1, and Su depends only on frequency.
        su0 = self._compute_slope_u_db_per_bark(np.zeros(self._z, dtype=np.float64), self._band_fc_hz)
        # Precompute A[j] for the base case.
        a0 = np.zeros(self._z, dtype=np.float64)
        # Compute A[j] = sum_k 10^(d*slope/10), using slope_l for k<j and su0[j] for k>=j.
        for j in range(self._z):
            # Upper side includes k==j (d=0 => 1)
            d = self._d_idx[:, j]
            upper = d >= 0.0
            t = np.zeros(self._z, dtype=np.float64)
            t[self._lower_mask[:, j]] = self._term_lower[self._lower_mask[:, j], j]
            t[upper] = np.power(10.0, (d[upper] * su0[j]) / 10.0)
            a0[j] = float(np.sum(t))
        # Now compute E[k] for the base curve (single 'frame'): Eq.7 with amp_j==1/a0[j].
        acc = np.zeros(self._z, dtype=np.float64)
        gamma = self._gamma
        for j in range(self._z):
            amp = 1.0 / max(a0[j], 1e-300)
            # Lower contributions
            if j > 0:
                acc[:j] += np.power(self._term_lower[:j, j] * amp, gamma)
            # Upper contributions (including j)
            d_up = self._d_idx[j:, j]
            t_up = np.power(10.0, (d_up * su0[j]) / 10.0)
            acc[j:] += np.power(t_up * amp, gamma)
        e0 = np.power(acc, 1.0 / gamma)
        # NormSP[k] == base excitation, so that flat L==0 yields E==1 after division.
        return np.maximum(e0, 1e-300)

    def _calibrate_const(self) -> float:
        """
        Choose 'const' so that a 1 kHz, 100 dB SPL tone produces ~64 sones.

        This follows the paper statement:
          "The scaling constant is chosen in order to give an overall loudness of 64 tones = 100 phons
           for a 100 dB SPL sine tone at 1 kHz."
        """
        # Synthetic "test" spectrum: all energy in the closest FFT bin to 1 kHz.
        f0 = float(self.config.calibration_tone_freq_hz)
        spl0 = float(self.config.calibration_tone_spl_db)
        p_rms = float(self.config.reference_pressure_pa) * float(np.power(10.0, spl0 / 20.0))

        idx = int(np.argmin(np.abs(self.rfft_freqs_hz - f0)))
        spec = np.zeros((self.rfft_freqs_hz.size, 1), dtype=np.float64)
        spec[idx, 0] = p_rms

        # Compute overall loudness with const=1.0
        n_total = float(self.compute_loudness_from_single_spectrum(spec, apply_level_adaptation=False).n_total_sones[0])
        if not np.isfinite(n_total) or n_total <= 0.0:
            raise ValueError(f"Calibration failed: computed N_total={n_total} for 100 dB @ 1 kHz")
        return float(self.config.calibration_target_sones) / n_total

    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D spectrum array, got shape {x.shape}")
        return x

    def _apply_level_adaptation(self, test_pa: np.ndarray, ref_pa: np.ndarray) -> np.ndarray:
        mode = self.config.level_adaptation
        if mode == "none":
            return ref_pa
        if mode != "match_max_bin":
            raise ValueError(f"Unsupported level_adaptation={mode!r}")

        # Paper 4.3: scale reference spectrum level to match response level.
        # For a pure-tone reference, matching the max bin is equivalent to matching the tone level.
        test_peak = np.max(np.abs(test_pa), axis=0)
        ref_peak = np.max(np.abs(ref_pa), axis=0)
        scale = np.ones_like(test_peak, dtype=np.float64)
        # If the reference is (near) silence, level adaptation must NOT amplify numerical noise.
        good = ref_peak > 1e-12
        scale[good] = test_peak[good] / ref_peak[good]
        return ref_pa * scale.reshape(1, -1)

    def _weighted_bin_power_ratio(self, spectrum_pa: np.ndarray) -> np.ndarray:
        """
        Apply ear weighting in amplitude domain (Eq.2) and return per-bin *power ratio* (p^2/p_ref^2).

        The public API accepts spectra in Pa RMS per bin; internal processing uses power ratios to match
        the paper's dB SPL conventions.
        """
        spectrum_pa = np.asarray(spectrum_pa, dtype=np.float64)
        weight = np.power(10.0, self._w_db / 20.0).reshape(-1, 1)
        weighted = spectrum_pa * weight
        return np.square(np.abs(weighted)) / self._p_ref2

    def _to_pitch_patterns_power_ratio(self, spectrum_pa: np.ndarray) -> np.ndarray:
        # Energy mapping to Bark bands + internal noise (Eq.4), all in power ratio units.
        e_bin = self._weighted_bin_power_ratio(spectrum_pa)
        t_frames = int(e_bin.shape[1])
        pe = np.zeros((self._z, t_frames), dtype=np.float64)
        if self._valid_bin_indices.size > 0:
            weights = e_bin[self._valid_bin_indices, :]  # (n_valid, T)
            for t in range(t_frames):
                pe[:, t] = np.bincount(
                    self._valid_band_indices,
                    weights=weights[:, t],
                    minlength=self._z,
                )
        return pe + self._p_thres_ratio.reshape(-1, 1)

    def _excitation_from_pitch_patterns(self, pp_ratio: np.ndarray) -> np.ndarray:
        """
        Frequency spreading (Eq.5-8) to compute excitation patterns E[k,t].

        pp_ratio: (Z, T) pitch patterns in power ratio (p^2/p_ref^2, must be > 0).
        """
        pp = np.asarray(pp_ratio, dtype=np.float64)
        if pp.shape[0] != self._z:
            raise ValueError(f"pp_ratio first dim must be Z={self._z}, got {pp.shape}")
        pp = np.maximum(pp, 1e-300)

        # Paper Eq.5 uses L[k]/dB = 10*log10(Pp[k]) where Pp is a linear power ratio (dB SPL).
        l_db = 10.0 * np.log10(pp)
        su = self._compute_slope_u_db_per_bark(l_db, self._band_fc_hz.reshape(-1, 1))  # (Z, T)

        gamma = self._gamma
        t_frames = int(pp.shape[1])
        acc = np.zeros((self._z, t_frames), dtype=np.float64)

        # Per-source band loop (Z bands). Keeps memory bounded (no ZxZxT tensor).
        for j in range(self._z):
            # A[j,t] = sum_k 10^(d*slope/10), lower side uses fixed slope_l, upper uses su[j,t].
            a = np.zeros((t_frames,), dtype=np.float64)

            # Lower contributions (k<j): constant in k, no dependence on su.
            if j > 0:
                a += float(np.sum(self._term_lower[:j, j]))

            # Upper contributions (k>=j): depends on su[j,t].
            d_up = self._d_idx[j:, j].reshape(-1, 1)  # (Z-j, 1)
            t_up = np.power(10.0, (d_up * su[j : j + 1, :]) / 10.0)  # (Z-j, T)
            a += np.sum(t_up, axis=0)
            a = np.maximum(a, 1e-300)

            amp = (np.power(10.0, l_db[j, :] / 10.0) / a).reshape(1, -1)  # (1, T)

            # Lower part: E_line[k<j] = amp * term_lower[k,j]
            if j > 0:
                acc[:j, :] += np.power(self._term_lower[:j, j].reshape(-1, 1) * amp, gamma)

            # Upper part (including k==j): E_line[k>=j] = amp * 10^(d_up*su/10)
            acc[j:, :] += np.power(t_up * amp, gamma)

        e = np.power(np.maximum(acc, 0.0), 1.0 / gamma)
        e /= self._norm_sp.reshape(-1, 1)
        return e

    def compute_loudness_from_single_spectrum(
        self,
        spectrum_pa: np.ndarray,
        *,
        apply_level_adaptation: bool = False,
        reference_spectrum_pa: Optional[np.ndarray] = None,
    ) -> "PEAQLoudnessResult":
        """
        Compute paper section 4.8 loudness N[k] and N_total for a single signal spectrum.

        If reference_spectrum_pa is given and apply_level_adaptation is True, it will be scaled to match
        spectrum_pa, then used as the input (useful for building an adapted pure-tone reference).
        """
        test = self._ensure_2d(spectrum_pa)
        if reference_spectrum_pa is not None:
            ref = self._ensure_2d(reference_spectrum_pa)
            if ref.shape != test.shape:
                raise ValueError(f"reference_spectrum_pa shape {ref.shape} != spectrum_pa shape {test.shape}")
            if apply_level_adaptation:
                test = self._apply_level_adaptation(test, ref)

        pp = self._to_pitch_patterns_power_ratio(test)
        e = self._excitation_from_pitch_patterns(pp)

        beta_pow = float(self.config.beta_pow)
        e_thres = self._e_thres_ratio.reshape(-1, 1)
        s = self._s_lin.reshape(-1, 1)

        # Eq.9 (const handled later via scaling_const)
        term = 1.0 - s + (s * e / np.maximum(e_thres, 1e-300))
        n_specific = np.power(np.maximum(e_thres / np.maximum(s, 1e-300), 1e-300), beta_pow) * (
            np.power(np.maximum(term, 0.0), beta_pow) - 1.0
        )
        n_specific = np.maximum(n_specific, 0.0)

        # Eq.12: overall loudness
        n_total = (24.0 / float(self._z)) * np.sum(n_specific, axis=0)

        n_total *= self._const
        n_specific *= self._const
        return PEAQLoudnessResult(
            n_total_sones=n_total,
            n_total_phons=sones_to_phons(n_total),
            n_specific_sones_per_band=n_specific,
            band_center_hz=self._band_fc_hz,
            band_center_bark=self._band_z,
        )

    def compute_partial_loudness_from_spectra(
        self,
        test_spectrum_pa: np.ndarray,
        ref_spectrum_pa: np.ndarray,
        *,
        apply_level_adaptation: Optional[bool] = None,
    ) -> "PEAQLoudnessResult":
        """
        Compute the paper section 4.9 partial noise loudness (Eq.13-15).

        Inputs must be aligned spectra (same frequency axis and number of frames).

        apply_level_adaptation:
          - None: follow `config.level_adaptation` (default behaviour)
          - False: do NOT scale the reference spectrum
          - True: scale reference -> test level (paper 4.3)
        """
        test = self._ensure_2d(test_spectrum_pa)
        ref = self._ensure_2d(ref_spectrum_pa)
        if test.shape != ref.shape:
            raise ValueError(f"test_spectrum_pa shape {test.shape} != ref_spectrum_pa shape {ref.shape}")

        # 4.3 Level Adaptation
        do_adapt = apply_level_adaptation
        if do_adapt is None:
            do_adapt = self.config.level_adaptation != "none"
        ref_adapted = self._apply_level_adaptation(test, ref) if do_adapt else ref

        pp_test = self._to_pitch_patterns_power_ratio(test)
        pp_ref = self._to_pitch_patterns_power_ratio(ref_adapted)

        e_test = self._excitation_from_pitch_patterns(pp_test)
        e_ref = self._excitation_from_pitch_patterns(pp_ref)

        # Eq.14 masking coefficient (avoid division by zero).
        diff = e_test - e_ref
        e_ref_safe = np.maximum(e_ref, 1e-300)
        # NL uses max(E_test - E_ref, 0); apply the same for stability so beta_mask stays in (0, 1].
        diff_pos = np.maximum(diff, 0.0)
        beta_mask = np.exp(-float(self.config.alpha_mask) * (diff_pos / e_ref_safe))

        # Eq.13 (partial loudness)
        # Paper note under Eq.13: E_thres is the internal noise function P_thres[k] (section 4.6).
        e_thres = self._p_thres_ratio.reshape(-1, 1)
        beta_pow = float(self.config.beta_pow)
        num = diff_pos
        denom = e_thres + e_ref * beta_mask
        denom = np.maximum(denom, 1e-300)
        inner = 1.0 + (num / denom)
        nl_specific = np.power(np.maximum(e_thres, 1e-300), beta_pow) * (np.power(inner, beta_pow) - 1.0)
        nl_specific = np.maximum(nl_specific, 0.0)

        # Eq.15: TotalNL
        total_nl = (24.0 / float(self._z)) * np.sum(nl_specific, axis=0)

        total_nl *= self._const
        nl_specific *= self._const
        return PEAQLoudnessResult(
            n_total_sones=total_nl,
            n_total_phons=sones_to_phons(total_nl),
            n_specific_sones_per_band=nl_specific,
            band_center_hz=self._band_fc_hz,
            band_center_bark=self._band_z,
        )

    def compute_error_harmonic_structure(
        self,
        test_spectrum_pa: np.ndarray,
        fundamental_freqs_hz: np.ndarray,
        *,
        normalize_to_n_fft: int = 2048,
        f_min_hz: Optional[float] = None,
        f_min_f0_ratio: float = 0.8,
        f_max_hz: Optional[float] = None,
        peak_search_width_bins: int = 1,
        remove_dc: bool = False,
        subtract_f0_masking: bool = True,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Compute the Error Harmonic Structure (EHS) described in paper section 4.10.

        The paper defines EHS via the (power) cepstrum of the test spectrum:
          1) Apply the ear weighting (Eq.1/Eq.2) to the spectrum magnitude
          2) Optionally subtract the fundamental's frequency spreading masking (Eq.5-8):
             For each frequency bin, compute the masking from f0 using the spreading slopes
             (Sl=27 dB/Bark for lower side, Su for upper side), then compute the excess
             above masking in dB SPL. The lower bound is set to 0 dB SPL (audible threshold).
          3) Normalize per frame
          4) Take the log magnitude spectrum (using natural log)
          5) Compute the FFT across frequency bins (cepstrum)
          6) Select the EHS peak near q0 = 1/f0:
             - pick the nearest quefrency bin to q0
             - if peak_search_width_bins > 0, take the max magnitude within +/- that many bins around it

        Args:
            f_min_hz: Low-frequency cutoff in Hz. If None, automatically computed as
                      max(40.0, min(fundamental_freqs_hz) * f_min_f0_ratio) to robustly
                      filter sub-fundamental noise.
            f_min_f0_ratio: Ratio for automatic f_min computation (default 0.8 means f_min = 0.8 * f0_min).
                           Only used when f_min_hz is None.

        This implementation returns a dimensionless per-frame scalar (shape (T,)).
        """
        test = self._ensure_2d(test_spectrum_pa)
        if test.shape[0] != self.rfft_freqs_hz.size:
            raise ValueError(
                f"test_spectrum_pa first dim must match rfft_freqs_hz ({self.rfft_freqs_hz.size}), got {test.shape}"
            )

        # Auto-determine f_min based on minimum fundamental frequency (if not explicitly provided)
        if f_min_hz is None:
            f0_array = np.asarray(fundamental_freqs_hz, dtype=np.float64).reshape(-1)
            valid_f0 = f0_array[np.isfinite(f0_array) & (f0_array > 0.0)]
            if valid_f0.size == 0:
                f_min = 40.0  # fallback to default
            else:
                f0_min = float(np.min(valid_f0))
                f_min = max(40.0, f0_min * float(f_min_f0_ratio))
        else:
            f_min = float(f_min_hz)
            if not np.isfinite(f_min) or f_min < 0.0:
                raise ValueError(f"f_min_hz must be finite and >= 0, got {f_min_hz}")

        f_max = float(f_max_hz) if f_max_hz is not None else float(self.rfft_freqs_hz[-1])
        if not np.isfinite(f_max) or f_max <= 0.0:
            raise ValueError(f"f_max_hz must be finite and > 0, got {f_max_hz}")

        # Choose a contiguous frequency slice to preserve a uniform dF for quefrency mapping.
        start = int(np.searchsorted(self.rfft_freqs_hz, f_min, side="left"))
        start = max(start, 1)  # avoid DC
        end = int(np.searchsorted(self.rfft_freqs_hz, f_max, side="right")) - 1
        end = min(end, int(self.rfft_freqs_hz.size - 1))
        if end <= start:
            raise ValueError(f"Invalid EHS frequency slice: start={start}, end={end}")

        freqs = self.rfft_freqs_hz[start : end + 1]
        df = np.diff(freqs)
        d_f = float(np.median(df)) if df.size > 0 else 0.0
        if not np.isfinite(d_f) or d_f <= 0.0:
            raise ValueError("Unable to infer frequency resolution dF for EHS computation")
        # EHS assumes a linear frequency grid (FFT bins).
        if df.size > 1 and not np.allclose(df, d_f, rtol=1e-6, atol=1e-12):
            raise ValueError("EHS requires uniformly spaced rFFT frequency bins")

        n = int(freqs.size)
        if n < 4:
            raise ValueError(f"EHS requires at least 4 frequency bins, got {n}")

        t_frames = int(test.shape[1])
        f0 = np.asarray(fundamental_freqs_hz, dtype=np.float64)
        if f0.ndim == 0:
            f0 = np.full((t_frames,), float(f0), dtype=np.float64)
        else:
            f0 = f0.reshape(-1)
        if f0.size != t_frames:
            raise ValueError(
                f"fundamental_freqs_hz must have length T={t_frames}, got shape {np.asarray(fundamental_freqs_hz).shape}"
            )

        valid_frame = np.isfinite(f0) & (f0 > 0.0)

        if peak_search_width_bins < 0:
            raise ValueError(f"peak_search_width_bins must be >= 0, got {peak_search_width_bins}")
        w_bins = int(peak_search_width_bins)

        eps_val = float(eps)
        if not np.isfinite(eps_val) or eps_val <= 0.0:
            raise ValueError(f"eps must be finite and > 0, got {eps}")

        do_masking = bool(subtract_f0_masking)

        # Reference pressure for 0 dB SPL.
        p0_pa = 2.0e-5

        # 1) Ear weighting (Eq.2) applied to magnitude spectrum.
        weight = np.power(10.0, self._w_db[start : end + 1] / 20.0).reshape(-1, 1)
        mag_w = np.abs(test[start : end + 1, :]) * weight  # (n, T) in Pa

        # Convert to dB SPL for masking computation.
        mag_db = 20.0 * np.log10(np.maximum(mag_w, eps_val) / p0_pa)  # (n, T)

        # 2) Optional f0 frequency spreading masking subtraction.
        #    Use the paper's spreading slopes (Eq.5-8) to compute the fundamental's masking
        #    at each frequency bin. For harmonics only, compute excess above masking.
        #    The fundamental itself is preserved (no masking applied to f0).
        if do_masking:
            df_full = float(self.rfft_freqs_hz[1] - self.rfft_freqs_hz[0]) if self.rfft_freqs_hz.size > 1 else 0.0
            if not np.isfinite(df_full) or df_full <= 0.0:
                raise ValueError("Unable to infer FFT bin spacing for f0 masking computation")

            # Find fundamental bin index in the full spectrum.
            k0 = np.zeros((t_frames,), dtype=np.int64)
            k0[valid_frame] = np.rint(f0[valid_frame] / df_full).astype(np.int64, copy=False)

            valid_k0 = (k0 >= start) & (k0 <= end)
            valid_frame &= valid_k0

            k0_clamped = np.clip(k0, start, end)
            idx0 = (k0_clamped - start).astype(np.int64, copy=False)
            idx0 = np.clip(idx0, 0, n - 1)

            # Get fundamental level L0 (dB SPL) using a small neighborhood max.
            idxs = np.stack(
                [
                    np.clip(idx0 - 1, 0, n - 1),
                    idx0,
                    np.clip(idx0 + 1, 0, n - 1),
                ],
                axis=0,
            )  # (3, T)
            l0_db = np.max(np.take_along_axis(mag_db, idxs, axis=0), axis=0)  # (T,)
            l0_db = np.where(valid_k0, l0_db, 0.0)

            # Compute Bark positions for all bins in the slice and for f0.
            z_bins = hz_to_bark_peaq(freqs).reshape(-1, 1)  # (n, 1)
            z_f0 = hz_to_bark_peaq(f0).reshape(1, -1)  # (1, T)
            dz = z_bins - z_f0  # (n, T), positive means bin is above f0

            # Identify fundamental region (|dz| < 0.5 Bark) - these bins keep original level.
            near_f0 = np.abs(dz) < 0.5

            # Spreading slopes (Eq.5-6):
            #   Lower side (f < f0, dz < 0): Sl = 27 dB/Bark
            #   Upper side (f > f0, dz >= 0): Su = min(0, -24 - 230/f0 + 0.2*L0)
            slope_l = 27.0  # dB/Bark
            f0_safe = np.maximum(f0, 1e-12).reshape(1, -1)
            slope_u = np.minimum(0.0, -24.0 - 230.0 / f0_safe + 0.2 * l0_db.reshape(1, -1))  # (1, T)

            # Masking level at each bin: M = L0 + slope * dz
            # For lower side (dz < 0): masking decreases as we go lower
            # For upper side (dz >= 0): masking decreases as we go higher
            lower_mask = dz < 0.0
            masking_db = np.where(
                lower_mask,
                l0_db.reshape(1, -1) + slope_l * dz,  # dz is negative, so masking decreases
                l0_db.reshape(1, -1) + slope_u * dz,  # slope_u is negative, dz positive, masking decreases
            )  # (n, T)

            # For invalid frames, set masking to -inf so residual equals original.
            masking_db = np.where(valid_frame.reshape(1, -1), masking_db, -np.inf)

            # Only consider masking >= 0 dB. Negative masking does not enhance harmonics.
            # effective_masking = max(masking_db, 0)
            effective_masking_db = np.maximum(masking_db, 0.0)  # (n, T)

            # Compute excess above effective masking in dB for harmonics only.
            # residual = mag_db - max(masking, 0)
            harmonic_residual_db = mag_db - effective_masking_db  # (n, T)

            # Final result: fundamental keeps original level, harmonics use excess above masking.
            result_db = np.where(near_f0, mag_db, harmonic_residual_db)

            # Apply 0 dB SPL floor to all bins (audible threshold).
            result_db = np.maximum(result_db, 0.0)

        else:
            # No masking subtraction: use original dB values, clamped to 0 dB floor.
            result_db = np.maximum(mag_db, 0.0)

        # 3) Normalize per frame using the maximum value (typically the fundamental).
        #    Convert dB to linear for normalization.
        result_linear = np.power(10.0, result_db / 20.0)  # (n, T)
        max_linear = np.max(result_linear, axis=0, keepdims=True)
        max_linear = np.maximum(max_linear, eps_val)
        valid_frame &= (max_linear.reshape(-1) > eps_val)
        norm_mag = result_linear / max_linear
        norm_mag = np.maximum(norm_mag, eps_val)

        # 4) Log magnitude spectrum (natural log for cepstrum).
        log_mag = np.log(norm_mag)

        # Apply log-domain floor corresponding to 0 dB SPL after normalization.
        # 0 dB SPL -> linear = 10^(0/20) = 1.0
        # After normalization: 1.0 / max_linear
        # After log: ln(1.0 / max_linear) = -ln(max_linear)
        log_floor = -np.log(max_linear)  # (1, T)
        log_mag = np.maximum(log_mag, log_floor)

        if bool(remove_dc):
            log_mag = log_mag - np.mean(log_mag, axis=0, keepdims=True)

        # 5) Cepstrum via FFT across frequency bins.
        cep = np.fft.rfft(log_mag, axis=0)
        cep_mag = np.abs(cep) / float(n)

        # 6) Peak at quefrency ~= 1/f0.
        q_axis = np.fft.rfftfreq(n, d=d_f)
        target_q = np.zeros((t_frames,), dtype=np.float64)
        target_q[valid_frame] = 1.0 / f0[valid_frame]

        # Nearest index on the quefrency axis for each frame.
        idx_right = np.searchsorted(q_axis, target_q, side="left")
        idx_left = np.clip(idx_right - 1, 0, q_axis.size - 1)
        idx_right = np.clip(idx_right, 0, q_axis.size - 1)
        dist_left = np.abs(q_axis[idx_left] - target_q)
        dist_right = np.abs(q_axis[idx_right] - target_q)
        use_right = dist_right < dist_left
        idx = np.where(use_right, idx_right, idx_left).astype(np.int64, copy=False)

        # If the target quefrency is outside representable range, drop the frame.
        valid_frame &= target_q <= float(q_axis[-1])

        # Peak search around the target bin.
        t_idx = np.arange(t_frames, dtype=np.int64)
        if w_bins <= 0:
            ehs = cep_mag[idx, t_idx]
        else:
            offsets = np.arange(-w_bins, w_bins + 1, dtype=np.int64)
            max_idx = int(cep_mag.shape[0] - 1)
            # Exclude DC bin (0) by clamping to [1, max_idx].
            vals = np.stack(
                [cep_mag[np.clip(idx + off, 1, max_idx), t_idx] for off in offsets],
                axis=0,
            )
            ehs = np.max(vals, axis=0)

        ehs = np.where(valid_frame, ehs, 0.0)

        # Empirical normalization: EHS magnitude scales approximately inversely with the number of FFT bins.
        # To reduce window-length dependence (especially for step signals where n_fft varies per step),
        # scale EHS to a fixed reference FFT size (default 2048).
        ref_n_fft = int(normalize_to_n_fft)
        if ref_n_fft <= 0:
            raise ValueError(f"normalize_to_n_fft must be a positive integer, got {normalize_to_n_fft!r}")

        # Best-effort inference from the rFFT frequency axis length.
        # For even n_fft: len(rfft) = n_fft/2 + 1 -> n_fft = 2*(len-1).
        # For odd  n_fft: this underestimates by 1; the scaling error is negligible for our use case.
        n_fft_full = 2 * int(self.rfft_freqs_hz.size - 1)
        if n_fft_full <= 0:
            raise ValueError(f"Invalid rfft_freqs_hz axis length: {self.rfft_freqs_hz.size}")

        ehs *= float(n_fft_full) / float(ref_n_fft)

        # Hard calibration factor: scale final EHS by 0.7
        ehs *= 0.7

        return np.asarray(ehs, dtype=np.float64).reshape(-1)
