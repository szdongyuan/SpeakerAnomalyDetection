"""Frequency-domain A, C, and Z weighting corrections."""

from __future__ import annotations

import numpy as np


def a_weighting(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return A-weighting corrections in dB."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    frequency_squared = frequencies**2
    numerator = 12194.0**2 * frequency_squared**2
    denominator = (
        (frequency_squared + 20.6**2)
        * (frequency_squared + 12194.0**2)
        * np.sqrt(
            (frequency_squared + 107.7**2)
            * (frequency_squared + 737.9**2)
        )
    )
    response = numerator / (denominator + 1e-30)
    return 20 * np.log10(response + 1e-30) + 2.0


def c_weighting(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return C-weighting corrections in dB."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    frequency_squared = frequencies**2
    numerator = 12194.0**2 * frequency_squared
    denominator = (
        (frequency_squared + 20.6**2)
        * (frequency_squared + 12194.0**2)
    )
    response = numerator / (denominator + 1e-30)
    return 20 * np.log10(response + 1e-30) + 0.062


def z_weighting(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return zero corrections for Z weighting."""
    return np.zeros_like(frequencies_hz, dtype=float)


WEIGHTING_FUNCTIONS = {
    "A": a_weighting,
    "C": c_weighting,
    "Z": z_weighting,
}


def get_weighting_fn(name: str = "Z"):
    """Resolve a frequency weighting function, defaulting to Z weighting."""
    normalized = str(name or "Z").upper()
    if normalized in ("NONE", "Z（NONE）"):
        normalized = "Z"
    return WEIGHTING_FUNCTIONS.get(normalized, z_weighting)
