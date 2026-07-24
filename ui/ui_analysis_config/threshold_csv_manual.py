"""Pure CSV/manual conversion helpers for threshold limits."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import (
    LIMIT_VALUE_SEMANTICS_BOUNDS,
    LIMIT_VALUE_SEMANTICS_TOLERANCE,
)
from ui.ui_analysis_config.manual_limit_segments import (
    ManualLimitValidationError,
    normalize_segments,
    validate_manual_limit_config,
)


class ThresholdCsvManualError(ValueError):
    pass


_X_HEADER = "x"
_UPPER_HEADER = "upperbound"
_LOWER_HEADER = "lowerbound"
_SIDE_CONFIG = {
    "upper": ("manual_upper_enabled", "manual_upper_segments", "上限", _UPPER_HEADER),
    "lower": ("manual_lower_enabled", "manual_lower_segments", "下限", _LOWER_HEADER),
}


@dataclass(frozen=True)
class _SideEndpoint:
    start: float | None = None
    end: float | None = None


def load_threshold_csv(
    csv_path: str,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> tuple[list[float], list[float], list[float]]:
    """Parse two-column or three-column threshold CSV into limit_data."""
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    if len(rows) < 2:
        raise ThresholdCsvManualError(f"CSV文件为空或格式不正确:\n{csv_path}")

    headers = [cell.strip().lower() for cell in rows[0]]
    upper_index, lower_index = _csv_bound_indexes(headers)

    x_values: list[float] = []
    upper_values: list[float] = []
    lower_values: list[float] = []
    expected_columns = len(headers)
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行列数不符合表头")

        x_value = _parse_required_finite_float(row[0], line_number, "X")
        upper_value = _parse_optional_bound(row[upper_index], line_number, "上限") if upper_index is not None else math.nan
        lower_value = _parse_optional_bound(row[lower_index], line_number, "下限") if lower_index is not None else math.nan
        if _is_missing_number(upper_value) and _is_missing_number(lower_value):
            raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行至少需要一个上下限值")

        x_values.append(x_value)
        upper_values.append(upper_value)
        lower_values.append(lower_value)

    validate_limit_data_values(
        (x_values, upper_values, lower_values),
        value_semantics=value_semantics,
        source_path=csv_path,
    )

    return x_values, upper_values, lower_values


def validate_limit_data_values(
    limit_data,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
    source_path: str | None = None,
) -> None:
    """Validate parsed threshold values using final-bound or tolerance semantics."""
    x_values, upper_values, lower_values = _limit_data_lists(limit_data)
    duplicate_counts = Counter(x_values)
    source_text = f"\n文件: {source_path}" if source_path else ""

    for line_number, (x_value, upper_value, lower_value) in enumerate(
        zip(x_values, upper_values, lower_values),
        start=2,
    ):
        if value_semantics == LIMIT_VALUE_SEMANTICS_TOLERANCE:
            if not _is_missing_number(upper_value) and float(upper_value) < 0:
                raise ThresholdCsvManualError(
                    f"黄金样本上下框线配置错误：向上容差不能小于0。\n"
                    f"位置: 第{line_number}条数据, X={x_value}, upper={upper_value}{source_text}"
                )
            if not _is_missing_number(lower_value) and float(lower_value) < 0:
                raise ThresholdCsvManualError(
                    f"黄金样本上下框线配置错误：向下容差不能小于0。\n"
                    f"位置: 第{line_number}条数据, X={x_value}, lower={lower_value}{source_text}"
                )
            continue

        if duplicate_counts[x_value] != 1:
            continue
        if not _is_missing_number(upper_value) and not _is_missing_number(lower_value) and lower_value > upper_value:
            raise ThresholdCsvManualError(
                f"CSV 上下限配置错误：下限不能大于上限。\n"
                f"位置: 第{line_number}条数据, X={x_value}\n"
                f"lower={lower_value}, upper={upper_value}{source_text}"
            )


def manual_config_from_limit_data(
    limit_data,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> dict:
    """Convert limit_data to manual_upper/lower segment config."""
    x_values, upper_values, lower_values = _limit_data_lists(limit_data)
    upper_segments = _segments_from_side_limit_data(x_values, upper_values, label="上限")
    lower_segments = _segments_from_side_limit_data(x_values, lower_values, label="下限")
    config = {
        "manual_upper_enabled": bool(upper_segments),
        "manual_lower_enabled": bool(lower_segments),
        "manual_upper_segments": upper_segments,
        "manual_lower_segments": lower_segments,
    }
    try:
        validate_manual_limit_config(config, value_semantics=value_semantics)
    except ManualLimitValidationError as exc:
        raise ThresholdCsvManualError(str(exc)) from exc
    return config


def manual_config_has_complete_segments(config: dict) -> bool:
    """Return True when saved manual config contains at least one complete segment on an enabled side."""
    config = config or {}
    for side in ("upper", "lower"):
        enabled_key, segments_key, _label, _header = _SIDE_CONFIG[side]
        if not bool(config.get(enabled_key, side == "upper")):
            continue
        try:
            if normalize_segments(config.get(segments_key, [])):
                return True
        except ManualLimitValidationError:
            continue
    return False


def csv_rows_from_manual_config(
    config: dict,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> list[list[str]]:
    """Return CSV rows for the current valid manual config, including header."""
    config = config or {}
    try:
        validate_manual_limit_config(config, value_semantics=value_semantics)
    except ManualLimitValidationError as exc:
        raise ThresholdCsvManualError(str(exc)) from exc

    upper_enabled = bool(config.get("manual_upper_enabled", True))
    lower_enabled = bool(config.get("manual_lower_enabled", False))
    upper_segments = normalize_segments(config.get("manual_upper_segments", [])) if upper_enabled else []
    lower_segments = normalize_segments(config.get("manual_lower_segments", [])) if lower_enabled else []
    upper_endpoints = _endpoint_map_from_segments(upper_segments, label="上限")
    lower_endpoints = _endpoint_map_from_segments(lower_segments, label="下限")

    if upper_enabled and not lower_enabled:
        return _one_sided_rows(_UPPER_HEADER, upper_endpoints)
    if lower_enabled and not upper_enabled:
        return _one_sided_rows(_LOWER_HEADER, lower_endpoints)
    return _two_sided_rows(upper_endpoints, lower_endpoints)


def write_manual_config_csv(
    config: dict,
    csv_path: str,
    *,
    value_semantics: str = LIMIT_VALUE_SEMANTICS_BOUNDS,
) -> None:
    """Write csv_rows_from_manual_config(config) to disk with UTF-8 newline-safe csv.writer."""
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(
            csv_rows_from_manual_config(config, value_semantics=value_semantics)
        )


def _csv_bound_indexes(headers: list[str]) -> tuple[int | None, int | None]:
    if len(headers) not in (2, 3) or not headers or headers[0] == "":
        raise ThresholdCsvManualError("Excel/CSV 格式不符合要求!")
    if len(headers) == 2:
        if headers[1] == _UPPER_HEADER:
            return 1, None
        if headers[1] == _LOWER_HEADER:
            return None, 1
        raise ThresholdCsvManualError("Excel/CSV 格式不符合要求!")

    bound_headers = headers[1:]
    if sorted(bound_headers) != [_LOWER_HEADER, _UPPER_HEADER]:
        raise ThresholdCsvManualError("Excel/CSV 格式不符合要求!")
    return 1 + bound_headers.index(_UPPER_HEADER), 1 + bound_headers.index(_LOWER_HEADER)


def _parse_required_finite_float(raw_value: str, line_number: int, label: str) -> float:
    text = raw_value.strip()
    if text == "":
        raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行{label}为空,无法解析")
    try:
        value = float(text)
    except ValueError as exc:
        raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行{label}不是数字,无法解析") from exc
    if not math.isfinite(value):
        raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行{label}必须是有限数字")
    return value


def _parse_optional_bound(raw_value: str, line_number: int, label: str) -> float:
    text = raw_value.strip()
    if text == "":
        return math.nan
    try:
        value = float(text)
    except ValueError as exc:
        raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行{label}不是数字,无法解析") from exc
    if math.isnan(value):
        return math.nan
    if not math.isfinite(value):
        raise ThresholdCsvManualError(f"CSV 数据错误:第 {line_number} 行{label}必须是有限数字")
    return value


def _limit_data_lists(limit_data) -> tuple[list[Any], list[Any], list[Any]]:
    if limit_data is None:
        raise ThresholdCsvManualError("CSV阈值数据为空，无法转换为手动分段")
    try:
        x_values, upper_values, lower_values = limit_data
    except (TypeError, ValueError) as exc:
        raise ThresholdCsvManualError("CSV阈值数据格式不正确，无法转换为手动分段") from exc

    try:
        x_list = list(x_values)
        upper_list = list(upper_values)
        lower_list = list(lower_values)
    except TypeError as exc:
        raise ThresholdCsvManualError("CSV阈值数据字段格式不正确，无法转换为手动分段") from exc
    if not (len(x_list) == len(upper_list) == len(lower_list)):
        raise ThresholdCsvManualError("CSV阈值数据长度不一致，无法转换为手动分段")
    return x_list, upper_list, lower_list


def _segments_from_side_limit_data(x_values: list[Any], side_values: list[Any], *, label: str) -> list[dict[str, float]]:
    grouped_values: dict[float, list[float]] = defaultdict(list)
    for row_index, (raw_x, raw_value) in enumerate(zip(x_values, side_values), start=1):
        value = _coerce_optional_limit_number(raw_value, label=label, row_index=row_index)
        if _is_missing_number(value):
            continue
        x_value = _coerce_required_limit_number(raw_x, label=label, row_index=row_index, field="X")
        grouped_values[x_value].append(value)

    if not grouped_values:
        return []

    sorted_x_values = sorted(grouped_values)
    collapsed_by_x: dict[float, list[float]] = {}
    for x_value in sorted_x_values:
        collapsed = _collapse_consecutive_equal_values(grouped_values[x_value])
        if len(collapsed) > 2:
            raise ThresholdCsvManualError(f"{label}重复X={x_value}存在多次跳变，无法自动转换为手动分段")
        if len(collapsed) == 2 and (x_value == sorted_x_values[0] or x_value == sorted_x_values[-1]):
            raise ThresholdCsvManualError(f"{label}重复X={x_value}缺少前后相邻点，无法自动转换为手动分段")
        collapsed_by_x[x_value] = collapsed

    if len(sorted_x_values) < 2:
        return []

    segments: list[dict[str, float]] = []
    for start_x, end_x in zip(sorted_x_values, sorted_x_values[1:]):
        start_values = collapsed_by_x[start_x]
        end_values = collapsed_by_x[end_x]
        start_y = start_values[1] if len(start_values) == 2 else start_values[0]
        end_y = end_values[0]
        segments.append(
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }
        )
    return segments


def _coerce_optional_limit_number(raw_value: Any, *, label: str, row_index: int) -> float:
    if raw_value is None:
        return math.nan
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return math.nan
    if isinstance(raw_value, (bool, np.bool_)):
        raise ThresholdCsvManualError(f"{label}第{row_index}行阈值必须是数字")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ThresholdCsvManualError(f"{label}第{row_index}行阈值必须是数字") from exc
    if math.isnan(value):
        return math.nan
    if not math.isfinite(value):
        raise ThresholdCsvManualError(f"{label}第{row_index}行阈值必须是有限数字")
    return value


def _coerce_required_limit_number(raw_value: Any, *, label: str, row_index: int, field: str) -> float:
    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        raise ThresholdCsvManualError(f"{label}第{row_index}行{field}不能为空")
    if isinstance(raw_value, (bool, np.bool_)):
        raise ThresholdCsvManualError(f"{label}第{row_index}行{field}必须是数字")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ThresholdCsvManualError(f"{label}第{row_index}行{field}必须是数字") from exc
    if not math.isfinite(value):
        raise ThresholdCsvManualError(f"{label}第{row_index}行{field}必须是有限数字")
    return value


def _collapse_consecutive_equal_values(values: list[float]) -> list[float]:
    collapsed: list[float] = []
    for value in values:
        if not collapsed or value != collapsed[-1]:
            collapsed.append(value)
    return collapsed


def _endpoint_map_from_segments(segments: list[dict[str, float]], *, label: str) -> dict[float, _SideEndpoint]:
    raw_endpoints: dict[float, dict[str, list[float]]] = defaultdict(lambda: {"start": [], "end": []})
    for segment in segments:
        raw_endpoints[segment["start_x"]]["start"].append(segment["start_y"])
        raw_endpoints[segment["end_x"]]["end"].append(segment["end_y"])

    endpoints: dict[float, _SideEndpoint] = {}
    for x_value, values_by_orientation in raw_endpoints.items():
        start = _single_endpoint_value(values_by_orientation["start"], label=label, x_value=x_value, orientation="起始")
        end = _single_endpoint_value(values_by_orientation["end"], label=label, x_value=x_value, orientation="截止")
        endpoints[x_value] = _SideEndpoint(start=start, end=end)
    return endpoints


def _single_endpoint_value(values: list[float], *, label: str, x_value: float, orientation: str) -> float | None:
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ThresholdCsvManualError(f"{label}X={x_value}存在多个{orientation}端点，无法导出CSV")
    return first


def _one_sided_rows(header: str, endpoints: dict[float, _SideEndpoint]) -> list[list[str]]:
    rows = [[_X_HEADER, header]]
    for x_value in sorted(endpoints):
        endpoint = endpoints[x_value]
        if endpoint.end is not None and endpoint.start is not None and endpoint.end != endpoint.start:
            rows.append([_format_float(x_value), _format_float(endpoint.end)])
            rows.append([_format_float(x_value), _format_float(endpoint.start)])
        elif endpoint.end is not None:
            rows.append([_format_float(x_value), _format_float(endpoint.end)])
        elif endpoint.start is not None:
            rows.append([_format_float(x_value), _format_float(endpoint.start)])
    return rows


def _two_sided_rows(
    upper_endpoints: dict[float, _SideEndpoint],
    lower_endpoints: dict[float, _SideEndpoint],
) -> list[list[str]]:
    rows = [[_X_HEADER, _UPPER_HEADER, _LOWER_HEADER]]
    for x_value in sorted(set(upper_endpoints) | set(lower_endpoints)):
        upper_endpoint = upper_endpoints.get(x_value)
        lower_endpoint = lower_endpoints.get(x_value)
        orientations = _required_row_orientations(upper_endpoint) | _required_row_orientations(lower_endpoint)
        if not orientations:
            orientations = {"end"} if _endpoint_has_orientation(upper_endpoint, "end") or _endpoint_has_orientation(lower_endpoint, "end") else {"start"}
        for orientation in ("end", "start"):
            if orientation not in orientations:
                continue
            rows.append(
                [
                    _format_float(x_value),
                    _format_optional_endpoint(_endpoint_value_for_orientation(upper_endpoint, orientation)),
                    _format_optional_endpoint(_endpoint_value_for_orientation(lower_endpoint, orientation)),
                ]
            )
    return rows


def _required_row_orientations(endpoint: _SideEndpoint | None) -> set[str]:
    if endpoint is None:
        return set()
    if endpoint.end is not None and endpoint.start is not None:
        if endpoint.end == endpoint.start:
            return set()
        return {"end", "start"}
    if endpoint.end is not None:
        return {"end"}
    if endpoint.start is not None:
        return {"start"}
    return set()


def _endpoint_has_orientation(endpoint: _SideEndpoint | None, orientation: str) -> bool:
    if endpoint is None:
        return False
    return getattr(endpoint, orientation) is not None


def _endpoint_value_for_orientation(endpoint: _SideEndpoint | None, orientation: str) -> float | None:
    if endpoint is None:
        return None
    return getattr(endpoint, orientation)


def _format_optional_endpoint(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return str(float(value))


def _is_missing_number(value: float) -> bool:
    return math.isnan(value)
