"""Pure comparison helpers for golden-sample curve display modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import (
    DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
    GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
)


def normalize_golden_sample_display_mode(analysis_config: Mapping[str, Any] | None) -> str:
    if not isinstance(analysis_config, Mapping):
        return DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE
    mode = str(
        analysis_config.get(
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
            DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
        )
        or DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE
    ).lower()
    if mode == GOLDEN_SAMPLE_DISPLAY_ENVELOPE:
        return mode
    return DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE


def has_valid_golden_overlap(baseline_aligned, current_y=None) -> bool:
    """Return whether the aligned golden curve contains any comparable point."""
    if baseline_aligned is None:
        return False
    try:
        baseline = np.asarray(baseline_aligned, dtype=float)
    except (TypeError, ValueError):
        return False
    if baseline.size == 0:
        return False
    valid = np.isfinite(baseline)
    if current_y is not None:
        try:
            current = np.asarray(current_y, dtype=float)
        except (TypeError, ValueError):
            return False
        if current.shape != baseline.shape:
            return False
        valid &= np.isfinite(current)
    return bool(np.any(valid))


def is_invalid_golden_envelope_limit_comparison(
    analysis_config: Mapping[str, Any] | None,
    baseline_aligned,
    current_y=None,
) -> bool:
    """Return whether envelope-limit judgment must stop because golden data is invalid."""
    if not isinstance(analysis_config, Mapping):
        return False
    envelope_limit_enabled = (
        bool(analysis_config.get(GOLDEN_SAMPLE_CHECKED_KEY))
        and bool(analysis_config.get("limit_checked"))
        and normalize_golden_sample_display_mode(analysis_config) == GOLDEN_SAMPLE_DISPLAY_ENVELOPE
    )
    return envelope_limit_enabled and not has_valid_golden_overlap(baseline_aligned, current_y)


def build_golden_curve_comparison(x_current, y_current, x_base, y_base):
    """Return signed deviation and the golden curve aligned to current X values."""
    x_c = np.asarray(x_current, dtype=float)
    y_c = np.asarray(y_current, dtype=float)
    x_b = np.asarray(x_base, dtype=float)
    y_b = np.asarray(y_base, dtype=float)

    if x_c.size == 0 or y_c.size == 0 or x_b.size == 0 or y_b.size == 0:
        return y_c, None

    baseline_mask = np.isfinite(x_b) & np.isfinite(y_b)
    x_b = x_b[baseline_mask]
    y_b = y_b[baseline_mask]
    if x_b.size < 2:
        return y_c, None

    if np.unique(x_b).size != x_b.size and x_c.size == y_c.size:
        x_c_flat = np.ravel(x_c)
        y_c_flat = np.ravel(y_c)
        current_mask = np.isfinite(x_c_flat) & np.isfinite(y_c_flat)
        if int(np.count_nonzero(current_mask)) == x_b.size:
            current_order = np.argsort(x_c_flat[current_mask], kind="stable")
            baseline_order = np.argsort(x_b, kind="stable")
            current_sorted_x = x_c_flat[current_mask][current_order]
            baseline_sorted_x = x_b[baseline_order]
            if current_sorted_x.shape == baseline_sorted_x.shape and np.allclose(
                current_sorted_x,
                baseline_sorted_x,
                rtol=1e-9,
                atol=1e-9,
            ):
                paired_deviation = y_c_flat[current_mask][current_order] - y_b[baseline_order]
                deviation = np.asarray(y_c_flat, dtype=float).copy()
                baseline_aligned = np.full(y_c_flat.shape, np.nan, dtype=float)
                current_indices = np.flatnonzero(current_mask)
                deviation[current_indices[current_order]] = paired_deviation
                baseline_aligned[current_indices[current_order]] = y_b[baseline_order]
                return deviation.reshape(y_c.shape), baseline_aligned.reshape(y_c.shape)

    sort_indices = np.argsort(x_b, kind="stable")
    x_b = x_b[sort_indices]
    y_b = y_b[sort_indices]
    x_b, unique_indices = np.unique(x_b, return_index=True)
    y_b = y_b[unique_indices]
    if x_b.size < 2:
        return y_c, None

    baseline_aligned = np.interp(x_c, x_b, y_b)
    in_range = (x_c >= float(np.min(x_b))) & (x_c <= float(np.max(x_b)))
    baseline_aligned = np.where(in_range, baseline_aligned, np.nan)
    return y_c - baseline_aligned, baseline_aligned


def interpolate_relative_limits(data_x, limit_x, upper_limits, lower_limits):
    """Match relative limit curves to data X values using linear interpolation."""
    data_x_arr = np.asarray(data_x, dtype=float)
    limit_x_arr = np.asarray(limit_x, dtype=float)
    upper_arr = np.asarray(upper_limits, dtype=float)
    lower_arr = np.asarray(lower_limits, dtype=float)
    upper_at_data = np.full(data_x_arr.shape, np.nan, dtype=float)
    lower_at_data = np.full(data_x_arr.shape, np.nan, dtype=float)
    in_band = (data_x_arr >= limit_x_arr.min()) & (data_x_arr <= limit_x_arr.max())
    if np.any(in_band):
        upper_at_data[in_band] = np.interp(data_x_arr[in_band], limit_x_arr, upper_arr)
        lower_at_data[in_band] = np.interp(data_x_arr[in_band], limit_x_arr, lower_arr)
    return upper_at_data, lower_at_data


def match_nearest_relative_limits(data_x, limit_x, upper_limits, lower_limits):
    """Match relative limits with the HD/RB/PRB nearest-point compatibility rule."""
    data_x_arr = np.asarray(data_x, dtype=float)
    limit_x_arr = np.asarray(limit_x, dtype=float)
    upper_arr = np.asarray(upper_limits, dtype=float)
    lower_arr = np.asarray(lower_limits, dtype=float)
    upper_at_data = np.full(data_x_arr.shape, np.nan, dtype=float)
    lower_at_data = np.full(data_x_arr.shape, np.nan, dtype=float)
    min_limit_x = float(np.min(limit_x_arr))
    max_limit_x = float(np.max(limit_x_arr))

    for index, frequency in enumerate(data_x_arr):
        if not np.isfinite(frequency):
            continue
        limit_index = int(np.argmin(np.abs(limit_x_arr - frequency)))
        if index + 1 < data_x_arr.size:
            next_limit_index = int(np.argmin(np.abs(limit_x_arr - data_x_arr[index + 1])))
        else:
            next_limit_index = limit_index
        if frequency < min_limit_x and limit_index == next_limit_index:
            continue
        if frequency > max_limit_x and limit_index == next_limit_index:
            continue
        upper_at_data[index] = upper_arr[limit_index]
        lower_at_data[index] = lower_arr[limit_index]

    return upper_at_data, lower_at_data


def build_golden_envelope_limits(baseline_aligned, upper_offset, lower_offset):
    """Build absolute envelope curves from signed golden-sample offsets."""
    baseline = np.asarray(baseline_aligned, dtype=float)
    upper = baseline + np.asarray(upper_offset, dtype=float)
    lower = baseline + np.asarray(lower_offset, dtype=float)
    return upper, lower


def build_golden_offset_deviation_limits(upper_offset, lower_offset):
    """Return deviation-coordinate limits equivalent to the absolute envelope."""
    upper = np.asarray(upper_offset, dtype=float)
    lower = np.asarray(lower_offset, dtype=float)
    return upper, lower


def golden_offset_comparison_mask(deviation, upper_offset, lower_offset):
    """Return points where deviation and at least one matched offset are valid."""
    deviation_arr = np.asarray(deviation, dtype=float)
    upper_arr = np.asarray(upper_offset, dtype=float)
    lower_arr = np.asarray(lower_offset, dtype=float)
    if deviation_arr.shape != upper_arr.shape or deviation_arr.shape != lower_arr.shape:
        raise ValueError("偏差曲线与上下框线偏移量数组长度不一致")
    return np.isfinite(deviation_arr) & (
        np.isfinite(upper_arr) | np.isfinite(lower_arr)
    )


def build_interpolated_golden_envelope_plot(
    data_x,
    raw_y,
    baseline_aligned,
    limit_x,
    upper_limits,
    lower_limits,
):
    """Return a full raw curve and absolute limits for FR/SPLF plotting."""
    x_arr = np.asarray(data_x, dtype=float)
    raw_arr = np.asarray(raw_y, dtype=float)
    baseline_arr = np.asarray(baseline_aligned, dtype=float)
    display_mask = np.isfinite(x_arr) & np.isfinite(raw_arr) & (x_arr > 0)
    display_x = x_arr[display_mask]
    display_y = raw_arr[display_mask]
    display_baseline = baseline_arr[display_mask]

    if display_x.size > 1:
        sort_indices = np.argsort(display_x)
        display_x = display_x[sort_indices]
        display_y = display_y[sort_indices]
        display_baseline = display_baseline[sort_indices]

    upper_offset, lower_offset = interpolate_relative_limits(
        display_x,
        limit_x,
        upper_limits,
        lower_limits,
    )
    absolute_upper, absolute_lower = build_golden_envelope_limits(
        display_baseline,
        upper_offset,
        lower_offset,
    )
    return display_x, display_y, absolute_upper, absolute_lower
