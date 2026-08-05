from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CURVE_EXPORTS_KEY,
    GOLDEN_SAMPLE_CURVE_EXPORTS_SCHEMA_VERSION,
    GOLDEN_SAMPLE_DISPLAY_MODES,
)


@dataclass(frozen=True)
class GoldenCurveSeries:
    mode: str
    available: bool
    x: list[Any] | None = None
    y: list[Any] | None = None


def _as_plain_list(values: Any) -> list[Any]:
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def build_golden_sample_curve_exports(
    selected_modes: Iterable[str],
    series_by_mode: Mapping[str, tuple[Any, Any] | None],
) -> dict[str, Any]:
    selected_set = set(selected_modes)
    selected = [
        mode
        for mode in GOLDEN_SAMPLE_DISPLAY_MODES
        if mode in selected_set
    ]
    series: dict[str, dict[str, Any]] = {}
    for mode in selected:
        xy = series_by_mode.get(mode)
        if xy is None:
            series[mode] = {"available": False}
            continue
        x, y = xy
        series[mode] = {
            "available": True,
            "x": _as_plain_list(x),
            "y": _as_plain_list(y),
        }
    return {
        "schema_version": GOLDEN_SAMPLE_CURVE_EXPORTS_SCHEMA_VERSION,
        "selected_modes": selected,
        "series": series,
    }


def _valid_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _current_error(message: str) -> tuple[bool, list[GoldenCurveSeries], str]:
    return False, [], message


def _parse_payload_structure(
    result: Mapping[str, Any],
) -> tuple[bool, list[str], Mapping[str, Any] | None, str | None]:
    if GOLDEN_SAMPLE_CURVE_EXPORTS_KEY not in result:
        return True, [], None, None

    payload = result[GOLDEN_SAMPLE_CURVE_EXPORTS_KEY]
    if not isinstance(payload, Mapping):
        return False, [], None, "载荷必须是对象"

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != GOLDEN_SAMPLE_CURVE_EXPORTS_SCHEMA_VERSION
    ):
        return False, [], None, f"不支持的 schema_version: {schema_version!r}"

    selected_modes = payload.get("selected_modes")
    if not isinstance(selected_modes, list) or not selected_modes:
        return False, [], None, "selected_modes 必须是非空列表"
    if (
        any(not isinstance(mode, str) or mode not in GOLDEN_SAMPLE_DISPLAY_MODES for mode in selected_modes)
        or len(set(selected_modes)) != len(selected_modes)
        or selected_modes
        != [mode for mode in GOLDEN_SAMPLE_DISPLAY_MODES if mode in selected_modes]
    ):
        return False, [], None, "selected_modes 必须是规范化且无重复的支持模式列表"

    series = payload.get("series")
    if not isinstance(series, Mapping):
        return False, [], None, "series 必须是对象"
    extra_modes = set(series) - set(selected_modes)
    if extra_modes:
        extra_labels = [
            repr(mode)
            for mode in sorted(extra_modes, key=lambda mode: str(mode))
        ]
        return False, [], None, f"series 包含未选择模式: {extra_labels!r}"

    return False, selected_modes, series, None


def _parse_series_entry(
    mode: str,
    entry: Any,
) -> tuple[GoldenCurveSeries | None, str | None]:
    if not isinstance(entry, Mapping):
        return None, f"模式 {mode!r} 的 series 必须是对象"
    available = entry.get("available")
    if not isinstance(available, bool):
        return None, f"模式 {mode!r} 的 available 必须是布尔值"
    if not available:
        if "x" in entry or "y" in entry:
            return None, f"模式 {mode!r} available=false 时不能包含 x/y"
        return GoldenCurveSeries(mode=mode, available=False), None

    x = entry.get("x")
    y = entry.get("y")
    if not isinstance(x, list):
        return None, f"模式 {mode!r} 的 x 必须是列表"
    if not isinstance(y, list):
        return None, f"模式 {mode!r} 的 y 必须是列表"
    if not x or not y:
        return None, f"模式 {mode!r} 的 x/y 不能为空"
    if len(x) != len(y):
        return None, f"模式 {mode!r} 的 x/y 长度不一致"
    if not all(_valid_finite_number(value) for value in x + y):
        return None, f"模式 {mode!r} 的 x/y 必须是有限数值"
    return (
        GoldenCurveSeries(
            mode=mode,
            available=True,
            x=list(x),
            y=list(y),
        ),
        None,
    )


def parse_selected_golden_sample_curve_exports(
    result: Mapping[str, Any],
    requested_modes: Iterable[str],
) -> tuple[bool, list[GoldenCurveSeries], str | None]:
    is_legacy, selected_modes, series, error = _parse_payload_structure(result)
    if error or is_legacy:
        return is_legacy, [], error
    assert series is not None

    requested_set = {
        mode for mode in requested_modes if mode in GOLDEN_SAMPLE_DISPLAY_MODES
    }
    parsed: list[GoldenCurveSeries] = []
    for mode in GOLDEN_SAMPLE_DISPLAY_MODES:
        if mode not in requested_set:
            continue
        if mode not in selected_modes:
            parsed.append(GoldenCurveSeries(mode=mode, available=False))
            continue
        if mode not in series:
            return _current_error(f"series 缺少已选择模式 {mode!r}")
        parsed_series, entry_error = _parse_series_entry(mode, series[mode])
        if entry_error:
            return _current_error(entry_error)
        assert parsed_series is not None
        parsed.append(parsed_series)

    return False, parsed, None


def parse_golden_sample_curve_exports(
    result: Mapping[str, Any],
) -> tuple[bool, list[GoldenCurveSeries], str | None]:
    is_legacy, selected_modes, series, error = _parse_payload_structure(result)
    if error or is_legacy:
        return is_legacy, [], error
    assert series is not None

    parsed: list[GoldenCurveSeries] = []
    for mode in selected_modes:
        if mode not in series:
            return _current_error(f"series 缺少已选择模式 {mode!r}")
        parsed_series, entry_error = _parse_series_entry(mode, series[mode])
        if entry_error:
            return _current_error(entry_error)
        assert parsed_series is not None
        parsed.append(parsed_series)

    return False, parsed, None
