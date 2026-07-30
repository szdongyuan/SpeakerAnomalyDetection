"""Stable config contract for standard HD/RB harmonic detection methods."""

from __future__ import annotations

from typing import Any


HARMONIC_DETECTION_METHOD_KEY = "harmonic_detection_method"
HARMONIC_DETECTION_METHOD_SYNCHRONOUS = "synchronous"
HARMONIC_DETECTION_METHOD_FOURIER = "fourier"
HARMONIC_DETECTION_METHOD_DEFAULT = HARMONIC_DETECTION_METHOD_SYNCHRONOUS

HARMONIC_DETECTION_METHOD_LABELS = {
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS: "同步检波",
    HARMONIC_DETECTION_METHOD_FOURIER: "傅里叶变换",
}

HARMONIC_DETECTION_METHOD_ALIASES = {
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS: HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    "sync": HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    "synchronous_detection": HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    "同步检波": HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    HARMONIC_DETECTION_METHOD_FOURIER: HARMONIC_DETECTION_METHOD_FOURIER,
    "fft": HARMONIC_DETECTION_METHOD_FOURIER,
    "fourier_transform": HARMONIC_DETECTION_METHOD_FOURIER,
    "傅里叶变换": HARMONIC_DETECTION_METHOD_FOURIER,
}


def normalize_harmonic_detection_method(
    value: Any,
    default: str = HARMONIC_DETECTION_METHOD_DEFAULT,
    *,
    strict: bool = False,
) -> str:
    normalized_default = str(default or HARMONIC_DETECTION_METHOD_DEFAULT).strip().lower()
    if normalized_default not in HARMONIC_DETECTION_METHOD_LABELS:
        normalized_default = HARMONIC_DETECTION_METHOD_DEFAULT

    if value in (None, ""):
        return normalized_default

    key = str(value).strip()
    normalized = HARMONIC_DETECTION_METHOD_ALIASES.get(key)
    if normalized is None:
        normalized = HARMONIC_DETECTION_METHOD_ALIASES.get(key.lower())
    if normalized is not None:
        return normalized

    if strict:
        raise ValueError(f"Unsupported harmonic detection method: {value!r}")
    return normalized_default
