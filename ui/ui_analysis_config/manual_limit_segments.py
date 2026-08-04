"""Pure helpers for manual upper/lower limit values and line segments."""

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


def validate_constant_limit_config(config: dict) -> None:
    _normalized_constant_limits(config)


def limits_from_constant_values(
    config: dict,
    target_x,
) -> tuple[list[float], list[float], list[float]]:
    upper_enabled, lower_enabled, upper_value, lower_value = _normalized_constant_limits(config)
    x_values = np.asarray(target_x, dtype=float)
    upper_values = np.full(
        x_values.shape,
        upper_value if upper_enabled else np.nan,
        dtype=float,
    )
    lower_values = np.full(
        x_values.shape,
        lower_value if lower_enabled else np.nan,
        dtype=float,
    )
    return x_values.tolist(), upper_values.tolist(), lower_values.tolist()


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


def limits_from_manual_config(
    config: dict,
    target_x,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> tuple[list[float], list[float], list[float]]:
    input_mode = str(
        config.get("manual_input_mode", "segments") or "segments"
    ).lower()
    if input_mode == "constant":
        return limits_from_constant_values(config, target_x)
    if input_mode == "segments":
        return limits_from_manual_segments(
            config,
            target_x,
            value_semantics=value_semantics,
        )
    raise ManualLimitValidationError(
        f"不支持的手动阈值输入方式: {input_mode}"
    )


def _normalized_constant_limits(config: dict | None):
    config = config or {}
    upper_enabled = bool(config.get("constant_upper_enabled", True))
    lower_enabled = bool(config.get("constant_lower_enabled", False))
    if not upper_enabled and not lower_enabled:
        raise ManualLimitValidationError("固定上下限至少需要启用一个")

    upper_value = None
    lower_value = None
    if upper_enabled:
        upper_value = _finite_constant_value(
            config.get("constant_upper_value"),
            label="上限",
        )
    if lower_enabled:
        lower_value = _finite_constant_value(
            config.get("constant_lower_value"),
            label="下限",
        )
    if upper_enabled and lower_enabled and lower_value > upper_value:
        raise ManualLimitValidationError("固定上下限配置错误：下限不能大于上限")
    return upper_enabled, lower_enabled, upper_value, lower_value


def _finite_constant_value(raw_value, *, label: str) -> float:
    if isinstance(raw_value, (bool, np.bool_)):
        raise ManualLimitValidationError(f"{label}必须是有限数字")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ManualLimitValidationError(f"{label}必须是有限数字") from exc
    if not math.isfinite(value):
        raise ManualLimitValidationError(f"{label}必须是有限数字")
    return value


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
    for upper_index, upper_segment in enumerate(upper_segments, start=1):
        for lower_index, lower_segment in enumerate(lower_segments, start=1):
            left = max(upper_segment["start_x"], lower_segment["start_x"])
            right = min(upper_segment["end_x"], lower_segment["end_x"])
            if right <= left:
                continue

            for x_value in (left, right):
                upper_y = _segment_value_at(upper_segment, x_value)
                lower_y = _segment_value_at(lower_segment, x_value)
                if lower_y > upper_y:
                    raise ManualLimitValidationError(
                        f"下限不能大于上限：上限第{upper_index}段与下限第{lower_index}段在重叠区间内不合法"
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
