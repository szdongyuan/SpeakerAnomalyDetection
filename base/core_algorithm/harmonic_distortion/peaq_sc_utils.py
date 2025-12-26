"""
PEAQ-SC helper utilities: Bark scale, sones/phons mapping, ear weighting, and window leakage ratio.
"""

from __future__ import annotations

from functools import lru_cache

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


def _ear_weighting_db_parametric(
    freqs_hz: np.ndarray,
    *,
    term3_coeff: float = -1.0e-3,
    term3_exponent: float = 3.6,
) -> np.ndarray:
    """
    Parametric form of Eq.1 so we can tweak small coefficient/exponent differences (e.g. SoundCheck vs paper).

    Only the high-frequency roll-off term is parameterized because it dominates above ~10 kHz:
      term3 = term3_coeff * (f/1k)^(term3_exponent)

    The defaults match the paper (Eq.1).
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    f_khz = np.maximum(f / 1000.0, 1e-12)
    term1 = -0.6 * 3.64 * np.power(f_khz, -0.8)
    term2 = 6.5 * np.exp(-0.6 * np.square(f_khz - 3.3))
    term3 = float(term3_coeff) * np.power(f_khz, float(term3_exponent))
    return term1 + term2 + term3


@lru_cache(maxsize=8)
def _window_leakage_ratio(n_fft: int, window_type: str) -> np.ndarray:
    """
    Precompute the magnitude leakage ratio |W[m]|/|W[0]| for the analysis window of length n_fft.

    For a bin-centered sinusoid windowed by w[n], the FFT bin magnitudes follow the window's DFT samples.
    This ratio is used as a simple "f0 spread" model: leakage at an offset of m bins is approximately
    A0 * ratio[m], where A0 is the fundamental bin magnitude.
    """
    n = int(n_fft)
    if n <= 0:
        raise ValueError(f"n_fft must be > 0, got {n_fft}")

    wtype = str(window_type).strip().lower()
    if wtype in {"hann", "hanning"}:
        win = np.hanning(n).astype(np.float64, copy=False)
    elif wtype in {"boxcar", "rect", "rectangular"}:
        win = np.ones((n,), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window_type={window_type!r} for f0 spread baseline")

    w = np.abs(np.fft.rfft(win))
    w0 = float(w[0]) if w.size else 0.0
    w0 = w0 if np.isfinite(w0) and w0 > 0.0 else 1.0
    return (w / w0).astype(np.float64, copy=False)
