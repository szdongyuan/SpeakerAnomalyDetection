"""Bounded display payload helpers; calculations continue to use full data."""

from __future__ import annotations

import numpy as np


DEFAULT_MAX_DISPLAY_POINTS = 20_000


def min_max_envelope(x_values, y_values, *, max_points=DEFAULT_MAX_DISPLAY_POINTS):
    """Keep extrema in ordered buckets while bounding IPC/display size."""
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y lengths differ")
    if type(max_points) is not int or max_points < 2:
        raise ValueError("max_points must be an integer >= 2")
    if x.size <= max_points:
        return x.tolist(), y.tolist()
    bucket_count = max(1, max_points // 2)
    edges = np.linspace(0, x.size, bucket_count + 1, dtype=np.int64)
    selected = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        bucket_y = y[start:stop]
        finite = np.isfinite(bucket_y)
        if not np.any(finite):
            selected.append(start)
            continue
        finite_indices = np.flatnonzero(finite)
        local_min = int(finite_indices[np.argmin(bucket_y[finite])])
        local_max = int(finite_indices[np.argmax(bucket_y[finite])])
        selected.extend(sorted({start + local_min, start + local_max}))
    selected = np.asarray(selected[:max_points], dtype=np.int64)
    return x[selected].tolist(), y[selected].tolist()


def sample_curve_at_interval(x_values, y_values, *, interval_seconds=0.01):
    """Sample a time curve at a fixed interval for CSV output only."""
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y lengths differ")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return [], []
    if not np.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be greater than zero")
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    y = y[unique_indices]
    if unique_x.size == 1:
        return unique_x.tolist(), y.tolist()
    target = np.arange(
        unique_x[0],
        unique_x[-1] + interval_seconds * 0.5,
        interval_seconds,
        dtype=np.float64,
    )
    sampled = np.interp(target, unique_x, y)
    return target.tolist(), sampled.tolist()
