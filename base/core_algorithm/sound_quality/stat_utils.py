"""Shared statistical utilities for sound quality metrics."""

import numpy as np

__all__ = ["iso_c_percentile"]


def iso_c_percentile(values: np.ndarray, percentile: int) -> float:
    """ISO 532-1 Annex C style percentile (exceedance level).

    ``percentile=5`` corresponds to N5 / R5 / S5: the value exceeded
    by 5 % of the time series samples.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    if percentile == 0:
        return float(arr[-1])
    if percentile == 100:
        return float(arr[0])
    idx = int((1.0 - percentile / 100.0) * arr.size)
    idx = min(max(idx, 1), arr.size - 1)
    return float((arr[idx - 1] + arr[idx]) / 2.0)
