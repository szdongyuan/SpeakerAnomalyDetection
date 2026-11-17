"""Immutable identity and recorded channel provenance for one saved WAV."""

from dataclasses import dataclass
from pathlib import Path

from base.workbench_recording_protocol import StationWavChannel


@dataclass(frozen=True)
class RecordAnalysisReference:
    request_id: str
    station_id: str
    condition_id: str
    wav_path: str
    channels: tuple[StationWavChannel, ...]

    def __post_init__(self):
        for field in ("request_id", "station_id", "condition_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if (
            type(self.wav_path) is not str or "\0" in self.wav_path
            or not Path(self.wav_path).is_absolute()
        ):
            raise ValueError("wav_path must be an absolute path")
        if not isinstance(self.channels, (list, tuple)) or not self.channels:
            raise ValueError("channels must be a nonempty ordered list or tuple")
        channels = tuple(self.channels)
        for index, channel in enumerate(channels):
            if not isinstance(channel, StationWavChannel):
                raise ValueError(f"channels[{index}] must be a StationWavChannel")
            if channel.wav_channel_index != index:
                raise ValueError(f"channels[{index}].wav_channel_index must follow channel order")
        object.__setattr__(self, "channels", channels)

    @property
    def record_key(self):
        return self.request_id, self.station_id
