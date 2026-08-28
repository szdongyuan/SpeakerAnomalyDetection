"""Serializable value contracts shared by the recording process and its owner."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os

import numpy as np

from base.streaming_waveform_accumulator import StreamingWaveformSnapshot


@dataclass(frozen=True)
class FrozenConfig(Mapping):
    """Picklable deep snapshot without mutable references to caller configuration."""
    entries: tuple

    def __getitem__(self, key):
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self.entries)

    def __len__(self):
        return len(self.entries)

    @classmethod
    def snapshot(cls, value):
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("configuration keys must be strings")
            return cls(tuple((key, cls.snapshot(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(cls.snapshot(item) for item in value)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("configuration must contain finite, serializable scalar values")

    def to_dict(self):
        def thaw(value):
            if isinstance(value, FrozenConfig):
                return value.to_dict()
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value
        return {key: thaw(value) for key, value in self.entries}


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _channels(value):
    channels = tuple(value)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must be nonempty and unique")
    for channel in channels:
        _integer("channel", channel)
    return channels


def _device(device, channels, direction):
    if not isinstance(device, Mapping):
        raise ValueError("device must be an explicit device snapshot")
    _integer("device index", device.get("index"))
    _integer("device hostapi", device.get("hostapi"))
    if not isinstance(device.get("name"), str) or not device["name"]:
        raise ValueError("device name is required")
    maximum = device.get(f"max_{direction}_channels")
    _integer(f"max_{direction}_channels", maximum, 1)
    if max(channels) >= maximum:
        raise ValueError(f"selected {direction} channels exceed device capacity")


@dataclass(frozen=True)
class RecordingRequest:
    request_id: str
    purpose: str
    sample_rate: int
    target_samples: int
    channels: tuple[int, ...]
    device: Mapping
    path: str
    streaming: bool
    trim_samples: int
    monitor: Mapping
    calibration_metadata: Mapping | None
    validation_thresholds: Mapping

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id is required")
        if self.purpose not in ("main", "calibration"):
            raise ValueError("purpose must be main or calibration")
        _integer("sample_rate", self.sample_rate, 1)
        _integer("target_samples", self.target_samples, 1)
        _integer("trim_samples", self.trim_samples)
        channels = _channels(self.channels)
        if self.purpose == "calibration" and len(channels) != 1:
            raise ValueError("calibration requires one physical input channel")
        _device(self.device, channels, "input")
        if not isinstance(self.path, str) or not os.path.isabs(self.path):
            raise ValueError("recording path must be absolute")
        if not isinstance(self.streaming, bool):
            raise ValueError("streaming must be a boolean")
        for name in ("device", "monitor", "validation_thresholds"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, FrozenConfig.snapshot(value))
        if self.calibration_metadata is not None:
            if not isinstance(self.calibration_metadata, Mapping):
                raise ValueError("calibration_metadata must be a mapping or None")
            object.__setattr__(self, "calibration_metadata", FrozenConfig.snapshot(self.calibration_metadata))
        object.__setattr__(self, "channels", channels)
        if self.monitor.get("enabled", False):
            output_channels = _channels(self.monitor.get("channels", ()))
            _device(self.monitor.get("device"), output_channels, "output")
            gain = self.monitor.get("gain_db", 0.0)
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain):
                raise ValueError("monitor gain_db must be finite")
            for key in ("mute_leading_samples", "fade_in_samples"):
                _integer(key, self.monitor.get(key, 0))

    @property
    def effective_streaming(self):
        return self.purpose == "main" and bool(self.streaming or self.monitor.get("enabled", False))


@dataclass(frozen=True)
class RecordingResult:
    request_id: str
    purpose: str
    path: str
    sample_rate: int
    channels: tuple[int, ...]
    raw_frames: int
    final_frames: int
    metadata_appended: bool
    warnings: tuple[str, ...] = ()
    handles_released: bool = True
    cleanup_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingFailure:
    request_id: str
    stage: str
    path: str
    message: str
    raw_frames: int = 0
    written_frames: int = 0
    handles_released: bool = True
    cleanup_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingCancelled:
    request_id: str
    path: str
    raw_frames: int
    final_frames: int
    handles_released: bool = True
    cleanup_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingPreview:
    request_id: str
    generation: int
    sequence: int
    sample_stop: int
    channels: tuple[int, ...]
    waveforms: tuple[StreamingWaveformSnapshot, ...]

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("preview request_id is required")
        _integer("generation", self.generation, 1)
        _integer("sequence", self.sequence, 1)
        _integer("sample_stop", self.sample_stop)
        channels = _channels(self.channels)
        if len(channels) != len(self.waveforms):
            raise ValueError("preview channel and waveform counts differ")
        owned = []
        for waveform in self.waveforms:
            if (not isinstance(waveform, StreamingWaveformSnapshot)
                    or waveform.sample_stop != self.sample_stop
                    or not isinstance(waveform.time, np.ndarray)
                    or not isinstance(waveform.amplitude, np.ndarray)
                    or waveform.time.ndim != 1 or waveform.amplitude.ndim != 1
                    or waveform.time.dtype != np.float64 or waveform.amplitude.dtype != np.float32
                    or len(waveform.time) != len(waveform.amplitude) or len(waveform.time) > 4000
                    or not np.all(np.isfinite(waveform.time))
                    or np.any(np.diff(waveform.time) <= 0)):
                raise ValueError("invalid cumulative preview waveform")
            time_axis = waveform.time.copy()
            amplitude = waveform.amplitude.copy()
            time_axis.setflags(write=False)
            amplitude.setflags(write=False)
            owned.append(StreamingWaveformSnapshot(time_axis, amplitude, self.sample_stop))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "waveforms", tuple(owned))

    def __reduce__(self):
        # NumPy pickle does not preserve write protection; reconstruct via validation.
        return (type(self), (self.request_id, self.generation, self.sequence,
                             self.sample_stop, self.channels, self.waveforms))


@dataclass(frozen=True)
class RecordingEvent:
    generation: int
    request_id: str
    kind: str
    payload: object = None
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("unsupported recording protocol version")
        _integer("generation", self.generation, 1)
        if not isinstance(self.request_id, str):
            raise ValueError("request_id must be a string")
        if self.kind not in ("start", "cancel", "preview_ack", "result_ack", "shutdown",
                             "ready", "started", "finalizing", "preview", "completed", "failed", "cancelled"):
            raise ValueError("unknown recording event kind")
        typed_payloads = {"start": RecordingRequest, "preview": RecordingPreview,
                          "completed": RecordingResult, "failed": RecordingFailure,
                          "cancelled": RecordingCancelled}
        expected = typed_payloads.get(self.kind)
        if expected is not None:
            if not isinstance(self.payload, expected) or self.payload.request_id != self.request_id:
                raise ValueError(f"{self.kind} payload must match its type and session")
        elif self.kind == "preview_ack":
            _integer("preview acknowledgement sequence", self.payload, 1)
        elif self.kind == "result_ack":
            if self.payload not in ("accepted", "rejected"):
                raise ValueError("result acknowledgement must be accepted or rejected")
        elif self.payload is not None:
            raise ValueError(f"{self.kind} does not carry a payload")
