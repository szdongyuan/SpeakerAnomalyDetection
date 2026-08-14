"""Helpers for SPL runtime calculation and display compatibility."""

from __future__ import annotations

from typing import Any, Mapping

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


def resolve_free_field_distance_correction_db(
    config: Mapping[str, Any] | None,
) -> float:
    """Return the free-field spherical-spreading correction in decibels."""
    cfg = config or {}
    if not cfg.get("free_field_distance_enabled", False):
        return 0.0
    distances = (
        ("measurement_distance_m", "测量距离"),
        ("target_distance_m", "目标距离"),
    )
    resolved = {}
    for key, label in distances:
        try:
            distance = float(cfg.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}必须为有限正数。") from exc
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"{label}必须为有限正数。")
        resolved[key] = distance

    return float(
        -20.0
        * np.log10(
            resolved["target_distance_m"]
            / resolved["measurement_distance_m"]
        )
    )


def resolve_directional_additional_correction_db(
    config: Mapping[str, Any] | None,
) -> float:
    """Return the configured manual directional correction in decibels."""
    cfg = config or {}
    if not cfg.get("directional_correction_enabled", False):
        return 0.0

    try:
        correction_db = float(
            cfg.get("directional_additional_correction_db", 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("方向修正必须为有限数值。") from exc
    if not np.isfinite(correction_db):
        raise ValueError("方向修正必须为有限数值。")
    return correction_db


def apply_spl_analysis_time_range(
    recorded_signal,
    sample_rate: float,
    config: Mapping[str, Any] | None,
):
    """Slice SPL input to the configured time range and return its source offset."""
    cfg = config or {}
    if not cfg.get("analysis_time_range_enabled", False):
        return recorded_signal, 0

    signal = np.asarray(recorded_signal)
    start_sec = max(
        0.0,
        float(cfg.get("analysis_start_time_sec", 0.0) or 0.0),
    )
    end_sec = max(
        0.0,
        float(cfg.get("analysis_end_time_sec", 0.0) or 0.0),
    )
    start_sample = min(
        int(np.floor(start_sec * float(sample_rate))),
        len(signal),
    )
    end_sample = (
        len(signal)
        if end_sec == 0.0
        else min(
            int(np.ceil(end_sec * float(sample_rate))),
            len(signal),
        )
    )
    if end_sample <= start_sample:
        return recorded_signal, 0
    return signal[start_sample:end_sample], start_sample
