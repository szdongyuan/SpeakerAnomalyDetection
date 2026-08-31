"""Persist display names without changing physical input-channel identities."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from base.load_config import LoadUiConfig
from consts import error_code, model_consts


_DEFAULT_DIRECTION_LABELS = ("前", "后", "左", "右", "上")
DEFAULT_CHANNEL_LAYOUT = dict(
    zip((f"CH{index}" for index in range(1, 6)), _DEFAULT_DIRECTION_LABELS)
)
_CHANNEL_LAYOUT_PATH = model_consts.DEFAULT_DIR + "ui/ui_config/channel_layout.json"


def _normalize_channel_layout(raw: Any) -> Dict[str, str]:
    layout = dict(DEFAULT_CHANNEL_LAYOUT)
    if not isinstance(raw, Mapping):
        return layout

    for channel_label, direction_label in raw.items():
        match = re.fullmatch(r"CH([1-9][0-9]{0,2})", str(channel_label))
        if match is None or int(match.group(1)) > 128:
            continue
        if isinstance(direction_label, str) and direction_label.strip():
            layout[channel_label] = direction_label.strip()
    return layout


def load_channel_layout(path: Optional[str] = None) -> Dict[str, str]:
    """Load display aliases, including legacy CH1-CH5 defaults."""
    load_code, payload = LoadUiConfig.load_data_from_json(
        path or _CHANNEL_LAYOUT_PATH
    )
    if load_code != error_code.OK:
        return dict(DEFAULT_CHANNEL_LAYOUT)
    return _normalize_channel_layout(payload)


def save_channel_layout(
    channel_layout: Mapping[str, str],
    path: Optional[str] = None,
) -> bool:
    """Persist input-channel display aliases in the existing UTF-8 JSON format."""
    return LoadUiConfig.save_data_to_json(
        _normalize_channel_layout(channel_layout),
        path or _CHANNEL_LAYOUT_PATH,
        indent=2,
    )
