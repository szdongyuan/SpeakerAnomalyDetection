"""Helpers for SPL runtime calculation and display compatibility."""

from __future__ import annotations

from typing import Any

import numpy as np


def calculate_overall_spl(
    recorded_signal,
    reference_pressure: float = 20e-6,
    v2pa_factor: float | None = None,
) -> float:
    """Compute the RMS sound pressure level over the full input signal."""
    signal_float = np.asarray(recorded_signal, dtype=float)
    if signal_float.size == 0:
        return float("nan")

    factor = 1.0 if v2pa_factor is None else float(v2pa_factor)
    pressure_pa = signal_float * factor
    pressure_rms = float(np.sqrt(np.mean(pressure_pa**2)))
    pressure_rms = max(pressure_rms, 1.0e-10)
    return float(20 * np.log10(pressure_rms / float(reference_pressure)))


def resolve_spl_unit(weighting: Any) -> str:
    """Return the SPL display unit for a configured frequency weighting."""
    normalized = str(weighting or "Z").strip().upper()
    return {
        "A": "dBA",
        "B": "dBB",
        "C": "dBC",
        "D": "dBD",
    }.get(normalized, "dB")
