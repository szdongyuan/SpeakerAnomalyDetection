"""Validate, resolve, interpolate, and compare analysis limits."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import (
    LIMIT_VALUE_SEMANTICS_BOUNDS,
    LIMIT_VALUE_SEMANTICS_OFFSET,
)


class ManualLimitValidationError(ValueError):
    pass


class ThresholdCsvManualError(ValueError):
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
    if isinstance(raw_segments, (str, bytes)) or not isinstance(
        raw_segments,
        Sequence,
    ):
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
                raise ManualLimitValidationError(
                    f"第{index}段缺少{field_label}"
                ) from exc
            if isinstance(raw_value, (bool, np.bool_)):
                raise ManualLimitValidationError(
                    f"第{index}段{field_label}必须是数字"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ManualLimitValidationError(
                    f"第{index}段{field_label}必须是数字"
                ) from exc
            if not math.isfinite(value):
                raise ManualLimitValidationError(
                    f"第{index}段{field_label}必须是有限数字"
                )
            segment[key] = value
        normalized.append(segment)
    return normalized


def validate_manual_segments(
    segments: list[dict[str, float]],
    *,
    label: str,
) -> None:
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
    upper_enabled, lower_enabled, upper_value, lower_value = (
        _normalized_constant_limits(config)
    )
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
    upper_enabled, lower_enabled, upper_segments, lower_segments = (
        _normalized_enabled_segments(config, value_semantics=value_semantics)
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


def _validate_normalized_segments(
    segments: list[dict[str, float]],
    *,
    label: str,
) -> None:
    if not segments:
        raise ManualLimitValidationError(f"{label}至少需要包含一段配置")
    if segments[0]["start_x"] < 0:
        raise ManualLimitValidationError(f"{label}第1段起始X必须大于或等于0")

    previous_end_x = None
    for index, segment in enumerate(segments, start=1):
        start_x = segment["start_x"]
        end_x = segment["end_x"]
        if end_x <= start_x:
            raise ManualLimitValidationError(
                f"{label}第{index}段截止X必须大于起始X"
            )
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
                        "下限不能大于上限："
                        f"上限第{upper_index}段与下限第{lower_index}段"
                        "在重叠区间内不合法"
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
    ratio = (x_value - segment["start_x"]) / (
        segment["end_x"] - segment["start_x"]
    )
    return segment["start_y"] + ratio * (
        segment["end_y"] - segment["start_y"]
    )


def validate_limit_data_values(
    limit_data,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
    source_path: str | None = None,
) -> None:
    """Validate threshold values using final-bound or signed-offset semantics."""
    x_values, upper_values, lower_values = _limit_data_lists(limit_data)
    if not x_values:
        raise ThresholdCsvManualError("CSV阈值数据为空")
    duplicate_counts = Counter(x_values)
    source_text = f"\n文件: {source_path}" if source_path else ""

    for line_number, (x_value, upper_value, lower_value) in enumerate(
        zip(x_values, upper_values, lower_values),
        start=2,
    ):
        if _is_missing_number(upper_value) and _is_missing_number(lower_value):
            raise ThresholdCsvManualError(
                f"CSV 数据错误:第 {line_number} 行至少需要一个上下限值{source_text}"
            )
        if duplicate_counts[x_value] != 1:
            continue
        if (
            not _is_missing_number(upper_value)
            and not _is_missing_number(lower_value)
            and lower_value > upper_value
        ):
            error_prefix = (
                "黄金样本上下框线偏移量配置错误"
                if value_semantics == LIMIT_VALUE_SEMANTICS_OFFSET
                else "CSV 上下限配置错误"
            )
            raise ThresholdCsvManualError(
                f"{error_prefix}：下限不能大于上限。\n"
                f"位置: 第{line_number}条数据, X={x_value}\n"
                f"lower={lower_value}, upper={upper_value}{source_text}"
            )


def _limit_data_lists(limit_data):
    if limit_data is None:
        raise ThresholdCsvManualError("CSV阈值数据为空，无法转换为手动分段")
    try:
        x_values, upper_values, lower_values = limit_data
    except (TypeError, ValueError) as exc:
        raise ThresholdCsvManualError(
            "CSV阈值数据格式不正确，无法转换为手动分段"
        ) from exc

    try:
        x_list = list(x_values)
        upper_list = list(upper_values)
        lower_list = list(lower_values)
    except TypeError as exc:
        raise ThresholdCsvManualError(
            "CSV阈值数据字段格式不正确，无法转换为手动分段"
        ) from exc
    if not (len(x_list) == len(upper_list) == len(lower_list)):
        raise ThresholdCsvManualError("CSV阈值数据长度不一致，无法转换为手动分段")
    return x_list, upper_list, lower_list


def _is_missing_number(value: float) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def resolve_spl_overall_limits(config):
    scalar = {
        "constant_upper_enabled": bool(config.get("scalar_upper_enabled", True)),
        "constant_upper_value": config.get("scalar_upper_value", 100.0),
        "constant_lower_enabled": bool(config.get("scalar_lower_enabled", False)),
        "constant_lower_value": config.get("scalar_lower_value", 0.0),
    }
    _, upper, lower = limits_from_constant_values(scalar, [0.0])
    return np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)


def resolve_limit_source(config, target_x):
    mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if mode == "manual":
        return limits_from_manual_config(config, target_x)
    if mode != "csv":
        raise ValueError(f"不支持的阈值模式: {mode}")
    limit_data = config.get("limit_data")
    if not limit_data:
        raise ValueError("已启用阈值，但未加载 CSV 配置文件")
    try:
        x_values, upper, lower = limit_data
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV 阈值数据格式不正确") from exc
    validate_limit_data_values(limit_data)
    return x_values, upper, lower


def resolve_limits(config, target_x):
    mode = str(config.get("limit_mode", "csv") or "csv").lower()
    if mode == "manual":
        _, upper, lower = limits_from_manual_config(config, target_x)
        upper_values = np.asarray(upper, dtype=np.float64)
        lower_values = np.asarray(lower, dtype=np.float64)
    elif mode == "csv":
        limit_data = config.get("limit_data")
        if not limit_data:
            raise ValueError("已启用阈值，但未加载 CSV 配置文件")
        try:
            csv_x, csv_upper, csv_lower = limit_data
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV 阈值数据格式不正确") from exc
        validate_limit_data_values(limit_data)
        upper_values = interpolate_limit_side(target_x, csv_x, csv_upper)
        lower_values = interpolate_limit_side(target_x, csv_x, csv_lower)
    else:
        raise ValueError(f"不支持的阈值模式: {mode}")
    overlap = np.isfinite(upper_values) & np.isfinite(lower_values)
    if np.any(lower_values[overlap] > upper_values[overlap]):
        raise ValueError("下限不能大于上限")
    return upper_values, lower_values


def interpolate_limit_side(target_x, raw_x, raw_values):
    x_values = np.asarray(list(raw_x), dtype=np.float64)
    side_values = np.asarray(list(raw_values), dtype=np.float64)
    if x_values.size != side_values.size:
        raise ValueError("CSV 阈值数据长度不一致")
    finite = np.isfinite(x_values) & np.isfinite(side_values)
    target = np.asarray(target_x, dtype=np.float64)
    output = np.full(target.shape, np.nan, dtype=np.float64)
    if not np.any(finite):
        return output
    points = {
        float(x_value): float(side_value)
        for x_value, side_value in zip(x_values[finite], side_values[finite])
    }
    sorted_x = np.asarray(sorted(points), dtype=np.float64)
    sorted_y = np.asarray([points[value] for value in sorted_x], dtype=np.float64)
    inside = (target >= sorted_x[0]) & (target <= sorted_x[-1])
    if np.any(inside):
        output[inside] = np.interp(target[inside], sorted_x, sorted_y)
    return output


def interpolate_spl_limit_curves(target_x, raw_x, raw_upper, raw_lower):
    target = np.asarray(target_x, dtype=np.float64)
    x_values = np.asarray(raw_x, dtype=np.float64)
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    return (
        _interpolate_spl_limit_side(
            target,
            x_values,
            np.asarray(raw_upper, dtype=np.float64)[order],
        ),
        _interpolate_spl_limit_side(
            target,
            x_values,
            np.asarray(raw_lower, dtype=np.float64)[order],
        ),
    )


def _interpolate_spl_limit_side(target, x_values, y_values):
    output = np.full(target.shape, np.nan, dtype=np.float64)
    if x_values.size == 0 or y_values.size != x_values.size:
        return output
    finite_target = np.isfinite(target)
    right = np.searchsorted(x_values, target, side="right")
    left = right - 1
    between = finite_target & (left >= 0) & (right < x_values.size)
    safe_left = np.clip(left, 0, x_values.size - 1)
    safe_right = np.clip(right, 0, x_values.size - 1)
    left_x = x_values[safe_left]
    right_x = x_values[safe_right]
    left_y = y_values[safe_left]
    right_y = y_values[safe_right]
    segments = (
        between
        & (right_x > left_x)
        & np.isfinite(left_y)
        & np.isfinite(right_y)
    )
    ratio = (target[segments] - left_x[segments]) / (
        right_x[segments] - left_x[segments]
    )
    output[segments] = left_y[segments] + ratio * (
        right_y[segments] - left_y[segments]
    )
    finite_rows = np.isfinite(y_values)
    if np.any(finite_rows):
        exact_x, first = np.unique(x_values[finite_rows], return_index=True)
        exact_y = y_values[finite_rows][first]
        positions = np.searchsorted(exact_x, target, side="left")
        safe = np.clip(positions, 0, exact_x.size - 1)
        exact = finite_target & np.isclose(
            target,
            exact_x[safe],
            rtol=1e-12,
            atol=1e-12,
        )
        output[exact] = exact_y[safe[exact]]
    return output


def compare_with_limits(plot_y, upper_limits, lower_limits, valid_mask=None):
    values = np.asarray(plot_y, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    if values.shape != upper.shape or values.shape != lower.shape:
        raise ValueError("分析曲线与上下限长度不一致")
    valid = (
        np.ones(values.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    upper_ok = np.isfinite(upper)
    lower_ok = np.isfinite(lower)
    out_mask = valid & (
        (upper_ok & (values > upper)) | (lower_ok & (values < lower))
    )
    deviation = 0.0
    is_ok = not np.any(out_mask)
    if not is_ok:
        above = np.where(out_mask & upper_ok, values - upper, 0.0)
        below = np.where(out_mask & lower_ok, lower - values, 0.0)
        deviation = float(np.nanmax(np.maximum(above, below)))
    else:
        inside = valid & np.isfinite(values)
        if np.any(inside):
            margin_upper = np.where(
                upper_ok[inside],
                upper[inside] - values[inside],
                np.inf,
            )
            margin_lower = np.where(
                lower_ok[inside],
                values[inside] - lower[inside],
                np.inf,
            )
            margins = np.minimum(margin_upper, margin_lower)
            margins = margins[np.isfinite(margins)]
            if margins.size:
                deviation = float(np.min(margins))
    return out_mask, round(deviation, 2), bool(is_ok)


__all__ = [
    "ManualLimitValidationError",
    "ThresholdCsvManualError",
    "compare_with_limits",
    "interpolate_limit_side",
    "interpolate_spl_limit_curves",
    "limits_from_constant_values",
    "limits_from_manual_config",
    "limits_from_manual_segments",
    "normalize_segments",
    "resolve_limit_source",
    "resolve_limits",
    "resolve_spl_overall_limits",
    "validate_constant_limit_config",
    "validate_limit_data_values",
    "validate_manual_limit_config",
    "validate_manual_segments",
]
