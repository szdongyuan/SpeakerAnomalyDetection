"""Pure channel availability checks for sequence analysis batches."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ui.ui_analysis_config.config_normalization import (
    normalize_analysis_channel,
    normalize_analysis_channels,
)


REQUIRED_CHANNEL_ANALYSIS_TYPES = frozenset({"SPL", "Spec", "FBA"})
PASSIVE_CHANNEL_ANALYSIS_TYPES = frozenset({"AI", "FFT", "LOUD"})
MULTI_CHANNEL_ANALYSIS_TYPES = (
    REQUIRED_CHANNEL_ANALYSIS_TYPES | PASSIVE_CHANNEL_ANALYSIS_TYPES | {"LP"}
)


@dataclass(frozen=True)
class AnalysisChannelSkip:
    item_key: str
    item_type: str
    requested_channel: int
    available_channels: tuple[int, ...]
    reason: str
    config_key: str = ""


@dataclass(frozen=True)
class AnalysisChannelPreflight:
    local_channels: Mapping[str, int]
    skipped: tuple[AnalysisChannelSkip, ...]
    fully_skipped_items: tuple[str, ...] = ()


def _available_live_channels(channels: Any) -> tuple[int, ...]:
    available = []
    for channel in channels or ():
        if isinstance(channel, bool):
            continue
        try:
            value = int(channel)
        except (TypeError, ValueError, OverflowError):
            continue
        if value not in available:
            available.append(value)
    return tuple(available)


def _missing_channel_reason(
    requested_channel: int,
    available_channels: tuple[int, ...],
) -> str:
    available_text = "、".join(f"In{channel + 1}" for channel in available_channels)
    if not available_text:
        available_text = "无"
    return (
        f"请求通道 In{requested_channel + 1} 不存在；"
        f"可用通道：{available_text}"
    )


def preflight_analysis_channels(
    analysis_config: Mapping[str, Any] | None,
    *,
    active_input_channels: Any,
    imported_channel_count: int | None = None,
) -> AnalysisChannelPreflight:
    """Map configured channels to array columns and return invalid items as values."""
    config = analysis_config if isinstance(analysis_config, Mapping) else {}
    display_sequence = config.get("display_sequence", ())
    if not isinstance(display_sequence, (list, tuple)):
        display_sequence = ()

    if imported_channel_count is None:
        available_channels = _available_live_channels(active_input_channels)
    else:
        try:
            count = max(0, int(imported_channel_count))
        except (TypeError, ValueError, OverflowError):
            count = 0
        available_channels = tuple(range(count))

    local_channels: dict[str, int] = {}
    skipped = []
    fully_skipped_items = []
    for raw_item_key in display_sequence:
        item_key = str(raw_item_key)
        item_config = config.get(raw_item_key)
        if not isinstance(item_config, Mapping):
            continue
        item_type = str(item_config.get("type") or "")
        uses_channel_list = (
            imported_channel_count is None
            and item_type in MULTI_CHANNEL_ANALYSIS_TYPES
            and "analysis_channels" in item_config
        )
        if not uses_channel_list and item_type not in REQUIRED_CHANNEL_ANALYSIS_TYPES:
            continue

        requested_channels = (
            normalize_analysis_channels(item_config)
            if uses_channel_list
            else [normalize_analysis_channel(item_config)]
        )
        valid_count = 0
        for requested_channel in requested_channels:
            runtime_key = (
                f"{item_key}--通道{requested_channel + 1}"
                if len(requested_channels) > 1
                else item_key
            )
            if requested_channel in available_channels:
                local_channels[runtime_key] = available_channels.index(requested_channel)
                valid_count += 1
                continue
            skipped.append(
                AnalysisChannelSkip(
                    item_key=runtime_key,
                    config_key=item_key,
                    item_type=item_type,
                    requested_channel=requested_channel,
                    available_channels=available_channels,
                    reason=_missing_channel_reason(
                        requested_channel,
                        available_channels,
                    ),
                )
            )
        if not valid_count:
            fully_skipped_items.append(item_key)

    return AnalysisChannelPreflight(
        local_channels=MappingProxyType(local_channels),
        skipped=tuple(skipped),
        fully_skipped_items=tuple(fully_skipped_items),
    )
