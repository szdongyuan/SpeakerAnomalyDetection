"""Qt-independent compatibility helpers shared by active and detached analysis."""

import math

import numpy as np


def interpolate_spl_limit_side(target_x, limit_x, limit_values):
    """Interpolate adjacent finite SPL limit rows without crossing a gap."""
    target = np.asarray(target_x, dtype=float)
    x_values = np.asarray(limit_x, dtype=float)
    y_values = np.asarray(limit_values, dtype=float)
    interpolated = np.full(target.shape, np.nan, dtype=float)
    finite_target = np.isfinite(target)

    right_indices = np.searchsorted(x_values, target, side="right")
    left_indices = right_indices - 1
    between_points = (
        finite_target
        & (left_indices >= 0)
        & (right_indices < x_values.size)
    )
    safe_left = np.clip(left_indices, 0, x_values.size - 1)
    safe_right = np.clip(right_indices, 0, x_values.size - 1)
    left_x = x_values[safe_left]
    right_x = x_values[safe_right]
    left_y = y_values[safe_left]
    right_y = y_values[safe_right]
    segment_mask = (
        between_points
        & (right_x > left_x)
        & np.isfinite(left_y)
        & np.isfinite(right_y)
    )
    ratio = (target[segment_mask] - left_x[segment_mask]) / (
        right_x[segment_mask] - left_x[segment_mask]
    )
    interpolated[segment_mask] = left_y[segment_mask] + ratio * (
        right_y[segment_mask] - left_y[segment_mask]
    )

    finite_rows = np.isfinite(y_values)
    if np.any(finite_rows):
        finite_x = x_values[finite_rows]
        finite_y = y_values[finite_rows]
        exact_x, first_indices = np.unique(finite_x, return_index=True)
        exact_y = finite_y[first_indices]
        exact_indices = np.searchsorted(exact_x, target, side="left")
        safe_exact = np.clip(exact_indices, 0, exact_x.size - 1)
        exact_mask = finite_target & np.isclose(
            target,
            exact_x[safe_exact],
            rtol=1e-12,
            atol=1e-12,
        )
        interpolated[exact_mask] = exact_y[safe_exact[exact_mask]]

    return interpolated


def interpolate_spl_limit_curves(
    target_x,
    limit_x,
    upper_limits,
    lower_limits,
):
    """Match both sparse SPL CSV polylines to measured time points."""
    target = np.asarray(target_x, dtype=float)
    x_values = np.asarray(limit_x, dtype=float)
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    upper_values = np.asarray(upper_limits, dtype=float)[order]
    lower_values = np.asarray(lower_limits, dtype=float)[order]
    return (
        interpolate_spl_limit_side(target, x_values, upper_values),
        interpolate_spl_limit_side(target, x_values, lower_values),
    )


def parse_fba_custom_bands_text(text):
    """Parse the legacy FBA custom-band text grammar without UI imports."""
    edges = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
        else:
            parts = [
                part.strip()
                for part in line.replace("\t", " ").split(" ")
                if part.strip()
            ]

        try:
            label = None
            if len(parts) == 1 and "-" in parts[0]:
                lower, upper = [
                    part.strip() for part in parts[0].split("-", 1)
                ]
                f_low, f_high = float(lower), float(upper)
            elif len(parts) >= 2:
                f_low, f_high = float(parts[0]), float(parts[1])
                if len(parts) >= 3:
                    label = " ".join(parts[2:]).strip() or None
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"格式错误: {raw!r}") from exc

        if not math.isfinite(f_low) or not math.isfinite(f_high):
            raise ValueError(f"频率必须为有限数: {raw!r}")
        if f_low <= 0 or f_high <= 0:
            raise ValueError(f"频率必须为正数: {raw!r}")
        if f_high <= f_low:
            raise ValueError(f"频段上限必须大于下限: {raw!r}")
        edges.append((f_low, f_high, label))

    edges.sort(key=lambda item: item[0])
    for index in range(1, len(edges)):
        if edges[index][0] < edges[index - 1][1]:
            raise ValueError("自定义频段不允许重叠，请检查相邻频段边界。")
    return edges
