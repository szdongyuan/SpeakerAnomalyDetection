"""Pure helpers for manual upper/lower limit line segments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import LIMIT_VALUE_SEMANTICS_BOUNDS


class ManualLimitValidationError(ValueError):
    pass


_FIELD_LABELS = {
    "start_x": "起始X",
    "start_y": "起始Y",
    "end_x": "截止X",
    "end_y": "截止Y",
}


def normalize_segments(raw_segments) -> list[dict[str, float]]:
    if raw_segments is None:
        return []
    if isinstance(raw_segments, (str, bytes)) or not isinstance(raw_segments, Sequence):
        raise ManualLimitValidationError("手动上下限段配置必须是列表")

    normalized = []
    for index, raw_segment in enumerate(raw_segments, start=1):
        if not isinstance(raw_segment, Mapping):
            raise ManualLimitValidationError(f"第{index}段必须是配置对象")

        segment = {}
        for key in ("start_x", "start_y", "end_x", "end_y"):
            field_label = _FIELD_LABELS[key]
            try:
                raw_value = raw_segment[key]
            except KeyError as exc:
                raise ManualLimitValidationError(f"第{index}段缺少{field_label}") from exc
            if isinstance(raw_value, (bool, np.bool_)):
                raise ManualLimitValidationError(f"第{index}段{field_label}必须是数字")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ManualLimitValidationError(f"第{index}段{field_label}必须是数字") from exc
            if not math.isfinite(value):
                raise ManualLimitValidationError(f"第{index}段{field_label}必须是有限数字")
            segment[key] = value
        normalized.append(segment)
    return normalized


def validate_manual_segments(segments: list[dict[str, float]], *, label: str) -> None:
    normalized = normalize_segments(segments)
    _validate_normalized_segments(normalized, label=label)


def validate_manual_limit_config(
    config: dict,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> None:
    _normalized_enabled_segments(config, value_semantics=value_semantics)


def limits_from_manual_segments(
    config: dict,
    target_x,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> tuple[list[float], list[float], list[float]]:
    upper_enabled, lower_enabled, upper_segments, lower_segments = _normalized_enabled_segments(
        config,
        value_semantics=value_semantics,
    )
    x_values = np.asarray(target_x, dtype=float)
    upper_values = np.full(x_values.shape, np.nan, dtype=float)
    lower_values = np.full(x_values.shape, np.nan, dtype=float)

    if upper_enabled:
        _apply_segments(upper_values, x_values, upper_segments)
    if lower_enabled:
        _apply_segments(lower_values, x_values, lower_segments)

    return x_values.tolist(), upper_values.tolist(), lower_values.tolist()


def assemble_manual_plot_data(
    upper_segments: Sequence[Mapping[str, float]],
    lower_segments: Sequence[Mapping[str, float]],
    *,
    positive_x_support=None,
) -> tuple[list[float], list[float], list[float]]:
    """Assemble plot-only geometry from ordered, finite manual segments."""
    upper_x, upper_y = _assemble_side_plot_data(
        upper_segments,
        positive_x_support=positive_x_support,
    )
    lower_x, lower_y = _assemble_side_plot_data(
        lower_segments,
        positive_x_support=positive_x_support,
    )

    if upper_x and lower_x:
        separator = [np.nan]
        x_values = upper_x + separator + lower_x
        upper_values = upper_y + separator + [np.nan] * len(lower_x)
        lower_values = [np.nan] * len(upper_x) + separator + lower_y
        return x_values, upper_values, lower_values
    if upper_x:
        return upper_x, upper_y, [np.nan] * len(upper_x)
    if lower_x:
        return lower_x, [np.nan] * len(lower_x), lower_y
    return [], [], []


def manual_limit_plot_data(
    config: dict,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
    positive_x_support=None,
) -> tuple[list[float], list[float], list[float]]:
    """Build validated runtime plot geometry from exact configured endpoints."""
    upper_enabled, lower_enabled, upper_segments, lower_segments = _normalized_enabled_segments(
        config,
        value_semantics=value_semantics,
    )
    return assemble_manual_plot_data(
        upper_segments if upper_enabled else [],
        lower_segments if lower_enabled else [],
        positive_x_support=positive_x_support,
    )


def _assemble_side_plot_data(
    segments: Sequence[Mapping[str, float]],
    *,
    positive_x_support=None,
) -> tuple[list[float], list[float]]:
    support = None
    if positive_x_support is not None:
        support = np.asarray(positive_x_support, dtype=float).reshape(-1)
        support = support[np.isfinite(support) & (support > 0)]

    x_values: list[float] = []
    y_values: list[float] = []
    previous_segment_rendered = False

    for segment in segments:
        start_x = float(segment["start_x"])
        start_y = float(segment["start_y"])
        end_x = float(segment["end_x"])
        end_y = float(segment["end_y"])

        if support is not None and start_x <= 0:
            supported_x = support[(support > start_x) & (support <= end_x)]
            if supported_x.size == 0:
                previous_segment_rendered = False
                continue
            start_x = float(np.min(supported_x))
            start_y = float(_segment_value_at(segment, start_x))

        if x_values and not previous_segment_rendered:
            x_values.append(np.nan)
            y_values.append(np.nan)
        x_values.extend([start_x, end_x])
        y_values.extend([start_y, end_y])
        previous_segment_rendered = True

    return x_values, y_values


def _normalized_enabled_segments(
    config: dict | None,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
):
    config = config or {}
    upper_enabled = bool(config.get("manual_upper_enabled", True))
    lower_enabled = bool(config.get("manual_lower_enabled", False))
    if not upper_enabled and not lower_enabled:
        raise ManualLimitValidationError("手动上下限至少需要启用一个")

    upper_segments = []
    lower_segments = []
    if upper_enabled:
        upper_segments = normalize_segments(config.get("manual_upper_segments", []))
        _validate_normalized_segments(upper_segments, label="上限")
    if lower_enabled:
        lower_segments = normalize_segments(config.get("manual_lower_segments", []))
        _validate_normalized_segments(lower_segments, label="下限")
    if upper_enabled and lower_enabled:
        _validate_lower_not_above_upper(upper_segments, lower_segments)

    return upper_enabled, lower_enabled, upper_segments, lower_segments


def _validate_normalized_segments(segments: list[dict[str, float]], *, label: str) -> None:
    if not segments:
        raise ManualLimitValidationError(f"{label}至少需要包含一段配置")
    if segments[0]["start_x"] < 0:
        raise ManualLimitValidationError(f"{label}第1段起始X必须大于或等于0")

    previous_end_x = None
    for index, segment in enumerate(segments, start=1):
        start_x = segment["start_x"]
        end_x = segment["end_x"]
        if end_x <= start_x:
            raise ManualLimitValidationError(f"{label}第{index}段截止X必须大于起始X")
        if previous_end_x is not None and start_x < previous_end_x:
            raise ManualLimitValidationError(
                f"{label}第{index}段起始X必须大于或等于上一段截止X"
            )
        previous_end_x = end_x


def _validate_lower_not_above_upper(
    upper_segments: list[dict[str, float]],
    lower_segments: list[dict[str, float]],
) -> None:
    effective_upper = _effective_segments_with_connectors(upper_segments)
    effective_lower = _effective_segments_with_connectors(lower_segments)

    for upper_piece in effective_upper:
        for lower_piece in effective_lower:
            left = max(upper_piece["start_x"], lower_piece["start_x"])
            right = min(upper_piece["end_x"], lower_piece["end_x"])
            if right <= left:
                continue

            for x_value in (left, right):
                upper_y = _segment_value_at(upper_piece, x_value)
                lower_y = _segment_value_at(lower_piece, x_value)
                if lower_y > upper_y:
                    raise ManualLimitValidationError(
                        "下限不能大于上限：有效阈值线在重叠区间内不合法"
                    )


def _effective_segments_with_connectors(
    segments: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    effective = []
    previous = None
    for segment in segments:
        current = {
            "start_x": float(segment["start_x"]),
            "start_y": float(segment["start_y"]),
            "end_x": float(segment["end_x"]),
            "end_y": float(segment["end_y"]),
        }
        if previous is not None and current["start_x"] > previous["end_x"]:
            effective.append(
                {
                    "start_x": previous["end_x"],
                    "start_y": previous["end_y"],
                    "end_x": current["start_x"],
                    "end_y": current["start_y"],
                }
            )
        effective.append(current)
        previous = current
    return effective


def _apply_segments(
    output_values: np.ndarray,
    x_values: np.ndarray,
    segments: list[dict[str, float]],
) -> None:
    for segment in _effective_segments_with_connectors(segments):
        mask = (x_values > segment["start_x"]) & (x_values <= segment["end_x"])
        output_values[mask] = _segment_value_at(segment, x_values[mask])


def _segment_value_at(segment: Mapping[str, float], x_value: Any):
    ratio = (x_value - segment["start_x"]) / (segment["end_x"] - segment["start_x"])
    return segment["start_y"] + ratio * (segment["end_y"] - segment["start_y"])
