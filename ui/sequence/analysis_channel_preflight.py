"""Pure channel availability checks for sequence analysis batches."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ui.ui_analysis_config.config_normalization import normalize_analysis_channel


REQUIRED_CHANNEL_ANALYSIS_TYPES = frozenset({"SPL", "Spec", "FBA"})
PASSIVE_CHANNEL_ANALYSIS_TYPES = frozenset({"AI", "FFT", "LOUD"})


@dataclass(frozen=True)
class AnalysisChannelSkip:
    item_key: str
    item_type: str
    requested_channel: int
    available_channels: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class AnalysisChannelPreflight:
    local_channels: Mapping[str, int]
    skipped: tuple[AnalysisChannelSkip, ...]


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
    for raw_item_key in display_sequence:
        item_key = str(raw_item_key)
        item_config = config.get(raw_item_key)
        if not isinstance(item_config, Mapping):
            continue
        item_type = str(item_config.get("type") or "")
        if item_type not in REQUIRED_CHANNEL_ANALYSIS_TYPES:
            continue

        requested_channel = normalize_analysis_channel(item_config)
        if requested_channel in available_channels:
            local_channels[item_key] = available_channels.index(requested_channel)
            continue
        skipped.append(
            AnalysisChannelSkip(
                item_key=item_key,
                item_type=item_type,
                requested_channel=requested_channel,
                available_channels=available_channels,
                reason=_missing_channel_reason(
                    requested_channel,
                    available_channels,
                ),
            )
        )

    return AnalysisChannelPreflight(
        local_channels=MappingProxyType(local_channels),
        skipped=tuple(skipped),
    )
