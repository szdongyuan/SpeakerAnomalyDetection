"""Pure normalization and availability rules for Excel export selections."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from base.golden_sample_comparison import normalize_golden_sample_display_modes
from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
)
from consts.excel_export_consts import (
    EXCEL_OUTPUT_DEVIATION,
    EXCEL_OUTPUT_MARGIN,
    EXCEL_OUTPUT_ORDER,
    EXCEL_OUTPUT_TEST_CURVE,
    SAVE_ITEM_OUTPUTS_KEY,
)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric_value)


def _iterable_values(value: object) -> tuple[Any, ...] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        return tuple(iter(value))
    except TypeError:
        return None


def _has_aligned_finite_pair(x_values: object, side_values: object) -> bool:
    x_sequence = _iterable_values(x_values)
    side_sequence = _iterable_values(side_values)
    if (
        x_sequence is None
        or side_sequence is None
        or not x_sequence
        or len(x_sequence) != len(side_sequence)
    ):
        return False
    return any(
        _is_finite_number(x_value) and _is_finite_number(side_value)
        for x_value, side_value in zip(x_sequence, side_sequence)
    )


def _csv_limit_available(config: Mapping[str, Any]) -> bool:
    limit_data = config.get("limit_data")
    if isinstance(limit_data, (str, bytes, bytearray, Mapping)):
        return False
    try:
        x_values, upper_values, lower_values = limit_data
    except (TypeError, ValueError):
        return False
    x_values = _iterable_values(x_values)
    upper_values = _iterable_values(upper_values)
    lower_values = _iterable_values(lower_values)
    return _has_aligned_finite_pair(
        x_values,
        upper_values,
    ) or _has_aligned_finite_pair(x_values, lower_values)


def _valid_manual_segment(segment: object) -> bool:
    if not isinstance(segment, Mapping):
        return False
    coordinates = tuple(
        segment.get(key) for key in ("start_x", "start_y", "end_x", "end_y")
    )
    return all(_is_finite_number(value) for value in coordinates) and float(
        coordinates[2]
    ) > float(coordinates[0])


def _manual_segment_side_available(
    config: Mapping[str, Any],
    *,
    side: str,
) -> bool:
    if not bool(config.get(f"manual_{side}_enabled", False)):
        return False
    segments = _iterable_values(config.get(f"manual_{side}_segments"))
    return segments is not None and any(
        _valid_manual_segment(segment) for segment in segments
    )


def _manual_limit_available(config: Mapping[str, Any]) -> bool:
    segment_keys = {"manual_upper_segments", "manual_lower_segments"}
    if segment_keys.intersection(config):
        return _manual_segment_side_available(
            config,
            side="upper",
        ) or _manual_segment_side_available(config, side="lower")

    return any(
        bool(config.get(f"manual_{side}_enabled", False))
        and _is_finite_number(config.get(f"manual_{side}"))
        for side in ("upper", "lower")
    )


def _current_loudness_limit_available(config: Mapping[str, Any]) -> bool | None:
    switch_keys = {"curve_upper_enabled", "curve_lower_enabled"}
    if not switch_keys.intersection(config):
        return None
    return any(
        bool(config.get(f"curve_{side}_enabled", False))
        and _is_finite_number(config.get(f"curve_{side}_value"))
        for side in ("upper", "lower")
    )


def _legacy_loudness_limit_available(config: Mapping[str, Any]) -> bool:
    metric = str(config.get("limit_metric", "") or "").lower()
    if metric in {"steady_state_average", "mean"}:
        prefix = "mean"
    elif metric in {"max_transient", "nmax"}:
        prefix = "nmax"
    else:
        return False
    return any(
        bool(config.get(f"{prefix}_{side}_enabled", False))
        and _is_finite_number(config.get(f"{prefix}_{side}_sone"))
        for side in ("upper", "lower")
    )


def is_margin_output_available(config: object) -> bool:
    """Return whether a known enabled limit structure has a finite side."""
    if not isinstance(config, Mapping):
        return False

    if "enable_threshold_judgment" in config:
        return bool(config.get("enable_threshold_judgment")) and any(
            _is_finite_number(config.get(key))
            for key in ("upper_offset_db", "lower_offset_db")
        )

    if not bool(config.get("limit_checked", False)):
        return False

    if config.get("limit_mode") == "manual":
        return _manual_limit_available(config)

    current_loudness_available = _current_loudness_limit_available(config)
    if current_loudness_available is not None:
        return current_loudness_available
    if _legacy_loudness_limit_available(config):
        return True
    return _csv_limit_available(config)


def is_deviation_output_available(config: object) -> bool:
    """Return whether golden-sample deviation display is currently enabled."""
    if not isinstance(config, Mapping):
        return False
    return bool(config.get(GOLDEN_SAMPLE_CHECKED_KEY, False)) and (
        GOLDEN_SAMPLE_DISPLAY_DEVIATION
        in normalize_golden_sample_display_modes(config)
    )


def available_excel_outputs(config: object) -> tuple[str, ...]:
    """Return available output identifiers in stable serialized/UI order."""
    available = {EXCEL_OUTPUT_TEST_CURVE}
    if is_margin_output_available(config):
        available.add(EXCEL_OUTPUT_MARGIN)
    if is_deviation_output_available(config):
        available.add(EXCEL_OUTPUT_DEVIATION)
    return tuple(output for output in EXCEL_OUTPUT_ORDER if output in available)


def _canonical_outputs(outputs: object) -> tuple[str, ...]:
    if isinstance(outputs, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        selected = {
            output for output in outputs if isinstance(output, str)
        }
    except TypeError:
        return ()
    return tuple(output for output in EXCEL_OUTPUT_ORDER if output in selected)


def _available_name_filter(available_items: Iterable[str]) -> set[str]:
    if isinstance(available_items, (str, bytes, bytearray, Mapping)):
        return set()
    try:
        return {name for name in available_items if isinstance(name, str)}
    except TypeError:
        return set()


def normalize_save_item_outputs(
    excel_cfg: object,
    analysis_config: object,
    *,
    available_items: Iterable[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Normalize authoritative new selections or migrate legacy item choices."""
    if not isinstance(excel_cfg, Mapping):
        return {}
    configs = analysis_config if isinstance(analysis_config, Mapping) else {}
    allowed_names = (
        None if available_items is None else _available_name_filter(available_items)
    )

    if SAVE_ITEM_OUTPUTS_KEY in excel_cfg:
        raw_mapping = excel_cfg.get(SAVE_ITEM_OUTPUTS_KEY)
        if not isinstance(raw_mapping, Mapping):
            return {}
        normalized: dict[str, tuple[str, ...]] = {}
        for name, raw_outputs in raw_mapping.items():
            if not isinstance(name, str) or (
                allowed_names is not None and name not in allowed_names
            ):
                continue
            available = set(available_excel_outputs(configs.get(name)))
            outputs = tuple(
                output
                for output in _canonical_outputs(raw_outputs)
                if output in available
            )
            if outputs:
                normalized[name] = outputs
        return normalized

    raw_items = excel_cfg.get("save_items", ())
    if isinstance(raw_items, (str, bytes, bytearray, Mapping)):
        return {}
    try:
        item_names = iter(raw_items)
    except TypeError:
        return {}

    migrated: dict[str, tuple[str, ...]] = {}
    for name in item_names:
        if not isinstance(name, str) or (
            allowed_names is not None and name not in allowed_names
        ):
            continue
        migrated[name] = available_excel_outputs(configs.get(name))
    return migrated


def serialize_save_item_outputs(
    selections: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Return JSON-ready, canonical selections without empty or unknown rows."""
    if not isinstance(selections, Mapping):
        return {}
    serialized: dict[str, list[str]] = {}
    for name, raw_outputs in selections.items():
        if not isinstance(name, str):
            continue
        outputs = _canonical_outputs(raw_outputs)
        if outputs:
            serialized[name] = list(outputs)
    return serialized
