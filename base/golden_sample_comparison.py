"""Pure comparison helpers for golden-sample curve display modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import (
    DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
    DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODES,
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
    GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
    GOLDEN_SAMPLE_DISPLAY_MODES,
    GOLDEN_SAMPLE_DISPLAY_MODES_KEY,
)


def normalize_golden_sample_display_modes(
    analysis_config: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not isinstance(analysis_config, Mapping):
        return DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODES

    raw_modes = analysis_config.get(GOLDEN_SAMPLE_DISPLAY_MODES_KEY)
    if isinstance(raw_modes, list):
        selected = {
            value
            for value in raw_modes
            if isinstance(value, str) and value in GOLDEN_SAMPLE_DISPLAY_MODES
        }
        if selected:
            return tuple(
                mode for mode in GOLDEN_SAMPLE_DISPLAY_MODES if mode in selected
            )

    legacy_mode = analysis_config.get(
        GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
        DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
    )
    if legacy_mode == GOLDEN_SAMPLE_DISPLAY_ENVELOPE:
        return (GOLDEN_SAMPLE_DISPLAY_ENVELOPE,)
    return DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODES


def normalize_golden_sample_display_mode(
    analysis_config: Mapping[str, Any] | None,
) -> str:
    modes = normalize_golden_sample_display_modes(analysis_config)
    return modes[0]


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
        and GOLDEN_SAMPLE_DISPLAY_ENVELOPE
        in normalize_golden_sample_display_modes(analysis_config)
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


def _sorted_unique_baseline(data_x, baseline_aligned):
    x_arr = np.asarray(data_x, dtype=float)
    baseline_arr = np.asarray(baseline_aligned, dtype=float)
    baseline_mask = np.isfinite(x_arr) & np.isfinite(baseline_arr) & (x_arr > 0)
    baseline_x = x_arr[baseline_mask]
    baseline_y = baseline_arr[baseline_mask]
    if baseline_x.size > 1:
        sort_indices = np.argsort(baseline_x, kind="stable")
        baseline_x = baseline_x[sort_indices]
        baseline_y = baseline_y[sort_indices]
        baseline_x, unique_indices = np.unique(baseline_x, return_index=True)
        baseline_y = baseline_y[unique_indices]
    return baseline_x, baseline_y


def _manual_value_between(start_x, start_y, end_x, end_y, target_x):
    if not np.isfinite(start_y) or not np.isfinite(end_y):
        return np.nan
    if start_x == end_x:
        return float(end_y)
    ratio = (target_x - start_x) / (end_x - start_x)
    return float(start_y + ratio * (end_y - start_y))


def _enrich_manual_geometry_run(
    run_x,
    run_upper,
    run_lower,
    baseline_x,
):
    enriched_x = [float(run_x[0])]
    enriched_upper = [float(run_upper[0])]
    enriched_lower = [float(run_lower[0])]

    for index in range(1, run_x.size):
        start_x = float(run_x[index - 1])
        end_x = float(run_x[index])
        low_x = min(start_x, end_x)
        high_x = max(start_x, end_x)
        interior_x = baseline_x[(baseline_x > low_x) & (baseline_x < high_x)]
        if end_x < start_x:
            interior_x = interior_x[::-1]

        for sample_x in interior_x:
            sample_x = float(sample_x)
            enriched_x.append(sample_x)
            enriched_upper.append(
                _manual_value_between(
                    start_x,
                    run_upper[index - 1],
                    end_x,
                    run_upper[index],
                    sample_x,
                )
            )
            enriched_lower.append(
                _manual_value_between(
                    start_x,
                    run_lower[index - 1],
                    end_x,
                    run_lower[index],
                    sample_x,
                )
            )

        enriched_x.append(end_x)
        enriched_upper.append(float(run_upper[index]))
        enriched_lower.append(float(run_lower[index]))

    return enriched_x, enriched_upper, enriched_lower


def build_manual_endpoint_golden_envelope_plot(
    data_x,
    raw_y,
    baseline_aligned,
    plot_x,
    upper_offsets,
    lower_offsets,
):
    """Build a raw display curve and endpoint-aware manual golden envelope."""
    x_arr = np.asarray(data_x, dtype=float)
    raw_arr = np.asarray(raw_y, dtype=float)
    display_mask = np.isfinite(x_arr) & np.isfinite(raw_arr) & (x_arr > 0)
    display_x = x_arr[display_mask]
    display_y = raw_arr[display_mask]
    if display_x.size > 1:
        sort_indices = np.argsort(display_x, kind="stable")
        display_x = display_x[sort_indices]
        display_y = display_y[sort_indices]

    baseline_x, baseline_y = _sorted_unique_baseline(data_x, baseline_aligned)
    geometry_x = np.asarray(plot_x, dtype=float)
    geometry_upper = np.asarray(upper_offsets, dtype=float)
    geometry_lower = np.asarray(lower_offsets, dtype=float)

    limit_x = []
    enriched_upper = []
    enriched_lower = []
    run_start = 0
    while run_start < geometry_x.size:
        if not np.isfinite(geometry_x[run_start]):
            if limit_x and not np.isnan(limit_x[-1]):
                limit_x.append(np.nan)
                enriched_upper.append(np.nan)
                enriched_lower.append(np.nan)
            run_start += 1
            continue

        run_end = run_start + 1
        while run_end < geometry_x.size and np.isfinite(geometry_x[run_end]):
            run_end += 1
        run_values = _enrich_manual_geometry_run(
            geometry_x[run_start:run_end],
            geometry_upper[run_start:run_end],
            geometry_lower[run_start:run_end],
            baseline_x,
        )
        limit_x.extend(run_values[0])
        enriched_upper.extend(run_values[1])
        enriched_lower.extend(run_values[2])
        run_start = run_end

    limit_x_arr = np.asarray(limit_x, dtype=float)
    upper_offset_arr = np.asarray(enriched_upper, dtype=float)
    lower_offset_arr = np.asarray(enriched_lower, dtype=float)
    baseline_at_limit = np.full(limit_x_arr.shape, np.nan, dtype=float)
    if baseline_x.size:
        finite_limit = np.isfinite(limit_x_arr)
        in_baseline_range = (
            finite_limit
            & (limit_x_arr >= baseline_x[0])
            & (limit_x_arr <= baseline_x[-1])
        )
        baseline_at_limit[in_baseline_range] = np.interp(
            limit_x_arr[in_baseline_range],
            baseline_x,
            baseline_y,
        )
    absolute_upper, absolute_lower = build_golden_envelope_limits(
        baseline_at_limit,
        upper_offset_arr,
        lower_offset_arr,
    )
    return display_x, display_y, limit_x_arr, absolute_upper, absolute_lower
