"""
Listen / AES 127th (2009) simplified PEAQ loudness model (FFT version).

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


def hz_to_bark_peaq(freq_hz: np.ndarray) -> np.ndarray:
    """PEAQ pitch scale approximation (paper Eq.3): z = 7 * asinh(f/650)."""
    f = np.asarray(freq_hz, dtype=np.float64)
    f = np.maximum(f, 0.0)
    return 7.0 * np.arcsinh(f / 650.0)


def bark_to_hz_peaq(bark: np.ndarray) -> np.ndarray:
    """Inverse of hz_to_bark_peaq: f = 650 * sinh(z/7)."""
    z = np.asarray(bark, dtype=np.float64)
    return 650.0 * np.sinh(z / 7.0)


def sones_to_phons(sones: np.ndarray) -> np.ndarray:
    """
    Standard sones -> phons mapping (Zwicker-style piecewise, used in the rest of this codebase).

    - N < 1: phon = 40 * N^0.4
    - N >= 1: phon = 40 + 10*log2(N)
    """
    n = np.asarray(sones, dtype=np.float64)
    ph = np.zeros_like(n)
    positive = n > 0.0
    lt1 = positive & (n < 1.0)
    ge1 = positive & ~lt1
    ph[lt1] = 40.0 * np.power(n[lt1], 0.4)
    ph[ge1] = 40.0 + 10.0 * np.log2(n[ge1])
    return ph


def _ear_weighting_db(freqs_hz: np.ndarray) -> np.ndarray:
    """
    Outer/middle ear frequency weighting from the paper (Eq.1).

    W(f)/dB = -0.6*3.64*(f/1k)^(-0.8)
              + 6.5*exp(-0.6*(f/1k - 3.3)^2)
              - 1e-3*(f/1k)^(3.6)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    f_khz = np.maximum(f / 1000.0, 1e-12)
    term1 = -0.6 * 3.64 * np.power(f_khz, -0.8)
    term2 = 6.5 * np.exp(-0.6 * np.square(f_khz - 3.3))
    term3 = -1.0e-3 * np.power(f_khz, 3.6)
    return term1 + term2 + term3


@dataclass(frozen=True)
class PEAQLoudnessConfig:
    # Paper parameters / defaults.
    z_min_hz: float = 91.7
    z_max_hz: float = 17700.0
    n_bands: int = 109
    res_bark: float = 0.25
    gamma_mix: float = 0.4
    alpha_mask: float = 1.5  # Eq.14
    beta_pow: float = 0.23  # paper: growth exponent for loudness

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

        self._w_db = _ear_weighting_db(self.rfft_freqs_hz)

        # Bark-domain auditory filter groups (paper 4.5).
        z_min = float(hz_to_bark_peaq(np.array([self.config.z_min_hz], dtype=np.float64))[0])
        self._band_z0 = z_min
        self._band_z = z_min + np.arange(int(self.config.n_bands), dtype=np.float64) * float(self.config.res_bark)
        self._band_fc_hz = bark_to_hz_peaq(self._band_z)

        # Map rFFT bins -> Bark groups via 0.25-Bark intervals around the centers.
        bin_bark = hz_to_bark_peaq(self.rfft_freqs_hz)
        # Centers are z0 + i*res; assign bins using half-step boundaries.
        band_idx = np.floor((bin_bark - (z_min - 0.5 * float(self.config.res_bark))) / float(self.config.res_bark)).astype(
            int
        )
        valid = (band_idx >= 0) & (band_idx < int(self.config.n_bands))
        self._bin_valid = valid
        self._bin_band = band_idx
        self._valid_bin_indices = np.flatnonzero(valid)
        self._valid_band_indices = self._bin_band[self._valid_bin_indices].astype(np.int64, copy=False)

        # Internal noise (paper Eq.4), interpreted as dB SPL and converted to Pa^2.
        fc_khz = np.maximum(self._band_fc_hz / 1000.0, 1e-12)
        p_thres_db = 0.4 * 3.64 * np.power(fc_khz, -0.8)
        self._p_thres_pa2 = (float(self.config.reference_pressure_pa) ** 2) * np.power(10.0, p_thres_db / 10.0)

        # Loudness threshold terms (Eq.10 / Eq.11), interpreted as dB SPL and converted to linear.
        e_thres_db = 0.364 * np.power(fc_khz, -0.8)
        self._e_thres_pa2 = (float(self.config.reference_pressure_pa) ** 2) * np.power(10.0, e_thres_db / 10.0)

        f = np.maximum(self._band_fc_hz, 1e-12)
        s_db = -2.0 - 2.05 * np.arctan(f / 4000.0) - 0.75 * np.arctan(np.square(f / 1600.0))
        # In the loudness equations, s[k] is used as a linear factor inside (1 - s + ...),
        # so we convert from dB to linear.
        self._s_lin = np.power(10.0, s_db / 10.0)

        # Precompute constant pieces for frequency spreading (Eq.5-8).
        self._gamma = float(self.config.gamma_mix)
        if not (0.0 < self._gamma < 2.0):
            raise ValueError(f"gamma_mix must be in (0,2), got {self._gamma}")
        self._res = float(self.config.res_bark)
        if self._res <= 0.0:
            raise ValueError(f"res_bark must be > 0, got {self._res}")
        self._z = int(self.config.n_bands)

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
        good = ref_peak > 0.0
        scale[good] = test_peak[good] / ref_peak[good]
        return ref_pa * scale.reshape(1, -1)

    def _weighted_bin_energy_pa2(self, spectrum_pa: np.ndarray) -> np.ndarray:
        # Apply ear weighting in amplitude domain (Eq.2) and return per-bin energy (Pa^2).
        spectrum_pa = np.asarray(spectrum_pa, dtype=np.float64)
        weight = np.power(10.0, self._w_db / 20.0).reshape(-1, 1)
        weighted = spectrum_pa * weight
        return np.square(np.abs(weighted))

    def _to_pitch_patterns_pa2(self, spectrum_pa: np.ndarray) -> np.ndarray:
        # Energy mapping to Bark bands + internal noise (Eq.4).
        e_bin = self._weighted_bin_energy_pa2(spectrum_pa)
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
        return pe + self._p_thres_pa2.reshape(-1, 1)

    def _excitation_from_pitch_patterns(self, pp_pa2: np.ndarray) -> np.ndarray:
        """
        Frequency spreading (Eq.5-8) to compute excitation patterns E[k,t].

        pp_pa2: (Z, T) pitch patterns in Pa^2 (must be > 0).
        """
        pp = np.asarray(pp_pa2, dtype=np.float64)
        if pp.shape[0] != self._z:
            raise ValueError(f"pp_pa2 first dim must be Z={self._z}, got {pp.shape}")
        pp = np.maximum(pp, 1e-300)

        # L[j,t] in dB, Eq. text before Eq.7.
        l_db = 10.0 * np.log10(pp)
        su = self._compute_slope_u_db_per_bark(l_db, self._band_fc_hz.reshape(-1, 1))  # (Z, T)

        gamma = self._gamma
        t_frames = int(pp.shape[1])
        acc = np.zeros((self._z, t_frames), dtype=np.float64)

        # Per-source band loop (Z=109). Keeps memory bounded (no ZxZxT tensor).
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

        pp = self._to_pitch_patterns_pa2(test)
        e = self._excitation_from_pitch_patterns(pp)

        beta_pow = float(self.config.beta_pow)
        e_thres = self._e_thres_pa2.reshape(-1, 1)
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
    ) -> "PEAQLoudnessResult":
        """
        Compute the paper section 4.9 partial noise loudness (Eq.13-15).

        Inputs must be aligned spectra (same frequency axis and number of frames).
        """
        test = self._ensure_2d(test_spectrum_pa)
        ref = self._ensure_2d(ref_spectrum_pa)
        if test.shape != ref.shape:
            raise ValueError(f"test_spectrum_pa shape {test.shape} != ref_spectrum_pa shape {ref.shape}")

        # 4.3 Level Adaptation
        ref_adapted = self._apply_level_adaptation(test, ref)

        pp_test = self._to_pitch_patterns_pa2(test)
        pp_ref = self._to_pitch_patterns_pa2(ref_adapted)

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
        e_thres = self._p_thres_pa2.reshape(-1, 1)
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


@dataclass(frozen=True)
class PEAQLoudnessResult:
    """
    Result container used for both "loudness" (Eq.9-12) and "partial loudness" (Eq.13-15).

    The paper's naming:
      - N_total / TotalNL correspond to `n_total_sones`
      - N[k] / NL[k] correspond to `n_specific_sones_per_band` (per band index k)
    """

    n_total_sones: np.ndarray  # (T,)
    n_total_phons: np.ndarray  # (T,)
    n_specific_sones_per_band: np.ndarray  # (Z, T)
    band_center_hz: np.ndarray  # (Z,)
    band_center_bark: np.ndarray  # (Z,)
