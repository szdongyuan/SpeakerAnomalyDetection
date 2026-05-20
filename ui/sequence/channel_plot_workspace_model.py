from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


class ChannelPlotWorkspaceModel(QObject):
    channels_reset = pyqtSignal()
    visibility_changed = pyqtSignal(int, bool)

    def __init__(self, channel_indices: Optional[Iterable[int]] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._channel_indices: List[int] = []
        self._visibility_by_channel: Dict[int, bool] = {}

        if channel_indices is not None:
            self.reset_channels(channel_indices)

    def reset_channels(self, channel_indices: Optional[Iterable[int]]) -> None:
        normalized_channels = self._normalize_channels(channel_indices)
        self._channel_indices = normalized_channels
        self._visibility_by_channel = {channel_index: True for channel_index in normalized_channels}
        self.channels_reset.emit()

    def channel_indices(self) -> List[int]:
        return list(self._channel_indices)

    def visible_channels(self) -> List[int]:
        return [channel_index for channel_index in self._channel_indices if self._visibility_by_channel[channel_index]]

    def hidden_channels(self) -> List[int]:
        return [channel_index for channel_index in self._channel_indices if not self._visibility_by_channel[channel_index]]

    def is_visible(self, channel_index: int) -> bool:
        channel_index = int(channel_index)
        self._require_channel(channel_index)
        return self._visibility_by_channel[channel_index]

    def set_visible(self, channel_index: int, visible: bool) -> bool:
        channel_index = int(channel_index)
        self._require_channel(channel_index)
        visible = bool(visible)

        if self._visibility_by_channel[channel_index] == visible:
            return False

        self._visibility_by_channel[channel_index] = visible
        self.visibility_changed.emit(channel_index, visible)
        return True

    @staticmethod
    def _normalize_channels(channel_indices: Optional[Iterable[int]]) -> List[int]:
        normalized_channels: List[int] = []
        seen_channels = set()

        for channel_index in channel_indices or []:
            normalized_channel = int(channel_index)
            if normalized_channel in seen_channels:
                continue
            seen_channels.add(normalized_channel)
            normalized_channels.append(normalized_channel)

        if not normalized_channels:
            normalized_channels.append(0)

        return normalized_channels

    def _require_channel(self, channel_index: int) -> None:
        if channel_index not in self._visibility_by_channel:
            raise KeyError(f"Unknown channel index: {channel_index}")
