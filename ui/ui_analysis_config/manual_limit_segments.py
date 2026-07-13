"""Pure helpers for manual upper/lower limit line segments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SEGMENT_KEYS = ("start_x", "start_y", "end_x", "end_y")


class ManualLimitValidationError(ValueError):
    pass


def normalize_segments(raw_segments) -> list[dict[str, float]]:
    if raw_segments is None:
        return []
    if isinstance(raw_segments, (str, bytes)) or not isinstance(raw_segments, Sequence):
        raise ManualLimitValidationError("manual limit segments must be a sequence")

    normalized = []
    for index, raw_segment in enumerate(raw_segments, start=1):
        if not isinstance(raw_segment, Mapping):
            raise ManualLimitValidationError(f"segment {index} must be a mapping")

        segment = {}
        for key in SEGMENT_KEYS:
            try:
                raw_value = raw_segment[key]
            except KeyError as exc:
                raise ManualLimitValidationError(f"segment {index} is missing {key}") from exc
            if isinstance(raw_value, (bool, np.bool_)):
                raise ManualLimitValidationError(f"segment {index} {key} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ManualLimitValidationError(f"segment {index} {key} must be numeric") from exc
            if not math.isfinite(value):
                raise ManualLimitValidationError(f"segment {index} {key} must be finite")
            segment[key] = value
        normalized.append(segment)
    return normalized


def validate_manual_segments(segments: list[dict[str, float]], *, label: str) -> None:
    normalized = normalize_segments(segments)
    _validate_normalized_segments(normalized, label=label)


def validate_manual_limit_config(config: dict) -> None:
    _normalized_enabled_segments(config)


def limits_from_manual_segments(config: dict, target_x) -> tuple[list[float], list[float], list[float]]:
    upper_enabled, lower_enabled, upper_segments, lower_segments = _normalized_enabled_segments(config)
    x_values = np.asarray(target_x, dtype=float)
    upper_values = np.full(x_values.shape, np.nan, dtype=float)
    lower_values = np.full(x_values.shape, np.nan, dtype=float)

    if upper_enabled:
        _apply_segments(upper_values, x_values, upper_segments)
    if lower_enabled:
        _apply_segments(lower_values, x_values, lower_segments)

    return x_values.tolist(), upper_values.tolist(), lower_values.tolist()


def _normalized_enabled_segments(config: dict | None):
    config = config or {}
    upper_enabled = bool(config.get("manual_upper_enabled", True))
    lower_enabled = bool(config.get("manual_lower_enabled", False))
    if not upper_enabled and not lower_enabled:
        raise ManualLimitValidationError("at least one manual limit must be enabled")

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
        raise ManualLimitValidationError(f"{label} must include at least one segment")
    if segments[0]["start_x"] < 0:
        raise ManualLimitValidationError(f"{label} segment 1 start_x must be non-negative")

    previous_end_x = None
    for index, segment in enumerate(segments, start=1):
        start_x = segment["start_x"]
        end_x = segment["end_x"]
        if end_x <= start_x:
            raise ManualLimitValidationError(f"{label} segment {index} end_x must be greater than start_x")
        if previous_end_x is not None and start_x < previous_end_x:
            raise ManualLimitValidationError(
                f"{label} segment {index} start_x must be greater than or equal to the previous end_x"
            )
        previous_end_x = end_x


def _validate_lower_not_above_upper(
    upper_segments: list[dict[str, float]],
    lower_segments: list[dict[str, float]],
) -> None:
    for upper_index, upper_segment in enumerate(upper_segments, start=1):
        for lower_index, lower_segment in enumerate(lower_segments, start=1):
            left = max(upper_segment["start_x"], lower_segment["start_x"])
            right = min(upper_segment["end_x"], lower_segment["end_x"])
            if right <= left:
                continue

            checks = (
                (left, "immediately right of the overlap start"),
                (right, "at the overlap end"),
            )
            for x_value, position in checks:
                upper_y = _segment_value_at(upper_segment, x_value)
                lower_y = _segment_value_at(lower_segment, x_value)
                if lower_y > upper_y:
                    raise ManualLimitValidationError(
                        "下限 cannot be greater than 上限 "
                        f"{position} for upper segment {upper_index} and lower segment {lower_index}"
                    )


def _apply_segments(
    output_values: np.ndarray,
    x_values: np.ndarray,
    segments: list[dict[str, float]],
) -> None:
    for segment in segments:
        mask = (x_values > segment["start_x"]) & (x_values <= segment["end_x"])
        output_values[mask] = _segment_value_at(segment, x_values[mask])


def _segment_value_at(segment: Mapping[str, float], x_value: Any):
    ratio = (x_value - segment["start_x"]) / (segment["end_x"] - segment["start_x"])
    return segment["start_y"] + ratio * (segment["end_y"] - segment["start_y"])
