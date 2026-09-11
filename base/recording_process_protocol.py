"""Serializable value contracts shared by the recording process and its owner."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os

import numpy as np

from base.recording_preview_config import validate_recording_preview_time_mode
from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
from base.ve3668n_input import (
    normalize_machine_id,
    validate_device_snapshot,
    validate_input_config,
    validate_physical_channels,
    validate_sample_rate,
    ve_acquisition_signature,
)
from base.ve3668n_wav_metadata import validate_ve_wav_metadata
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_MAX_POINTS,
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_LOWER_BOUND_TOLERANCE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from consts.ve3668n_consts import VE_BACKEND


VE_PREWARM_COMMAND = "prewarm_ve"
VE_PREWARM_STARTED = "ve_prewarm_started"
VE_PREWARM_PROGRESS = "ve_prewarm_progress"
VE_PREWARM_DETACHING = "ve_prewarm_detaching"
VE_PREWARM_TERMINAL = "ve_prewarm_terminal"


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


def _monotonic_time(name, value):
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite nonnegative monotonic time")


def _device(device, channels, direction):
    if not isinstance(device, Mapping):
        raise ValueError("device must be an explicit device snapshot")
    backend = device.get("backend", "sounddevice")
    if backend == VE_BACKEND:
        if direction != "input":
            raise ValueError("vkinging backend supports input only")
        snapshot = validate_device_snapshot(device)
        validate_physical_channels(channels)
        if not snapshot["available"]:
            raise ValueError("VE device is unavailable")
        if not set(channels).issubset(snapshot["physical_channels"]):
            raise ValueError("selected input channels are unavailable")
        return snapshot
    if backend != "sounddevice":
        raise ValueError(f"unknown backend: {backend!r}")
    _integer("device index", device.get("index"))
    _integer("device hostapi", device.get("hostapi"))
    if not isinstance(device.get("name"), str) or not device["name"]:
        raise ValueError("device name is required")
    maximum = device.get(f"max_{direction}_channels")
    _integer(f"max_{direction}_channels", maximum, 1)
    if max(channels) >= maximum:
        raise ValueError(f"selected {direction} channels exceed device capacity")
    return device


def _ve_request(request):
    profile = request.device["input_config"]
    if request.sample_rate != profile["sample_rate"]:
        raise ValueError("sample_rate must match device.input_config.sample_rate")
    if request.purpose == "calibration":
        if request.target_samples != 10 * request.sample_rate or request.trim_samples != 0:
            raise ValueError("VE calibration requires ten seconds and trim_samples=0")
    if request.purpose == "main" or request.calibration_metadata is not None:
        metadata = validate_ve_wav_metadata(request.calibration_metadata)
        expected = {"model": request.device["model"],
                    "machine_id": request.device["machine_id"], **profile}
        if metadata["acquisition"] != expected:
            raise ValueError("VE metadata acquisition must match the frozen request")
        entries = metadata["recorded_channels"]
        if (len(entries) != len(request.channels)
                or any(entry["wav_channel_index"] != index
                       or entry["physical_input_channel"] != physical
                       for index, (entry, physical) in enumerate(zip(entries, request.channels)))):
            raise ValueError("VE metadata channel order must match the frozen request")


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
    preview_time_mode: str = PREVIEW_TIME_MODE_RELATIVE_LATEST

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id is required")
        if self.purpose not in ("main", "calibration"):
            raise ValueError("purpose must be main or calibration")
        _integer("sample_rate", self.sample_rate, 1)
        _integer("target_samples", self.target_samples, 1)
        _integer("trim_samples", self.trim_samples)
        if isinstance(self.device, Mapping) and self.device.get("backend") == VE_BACKEND:
            channels = validate_physical_channels(self.channels)
        else:
            channels = _channels(self.channels)
        if self.purpose == "calibration" and len(channels) != 1:
            raise ValueError("calibration requires one physical input channel")
        object.__setattr__(self, "device", _device(self.device, channels, "input"))
        if not isinstance(self.path, str) or not os.path.isabs(self.path):
            raise ValueError("recording path must be absolute")
        if not isinstance(self.streaming, bool):
            raise ValueError("streaming must be a boolean")
        object.__setattr__(
            self,
            "preview_time_mode",
            validate_recording_preview_time_mode(self.preview_time_mode),
        )
        # Historical monitor fields are an inert compatibility shell.
        if not isinstance(self.monitor, Mapping):
            raise ValueError("monitor must be a mapping")
        object.__setattr__(self, "monitor", FrozenConfig.snapshot({}))
        for name in ("device", "validation_thresholds"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, FrozenConfig.snapshot(value))
        if self.calibration_metadata is not None:
            if not isinstance(self.calibration_metadata, Mapping):
                raise ValueError("calibration_metadata must be a mapping or None")
            object.__setattr__(self, "calibration_metadata", FrozenConfig.snapshot(self.calibration_metadata))
        object.__setattr__(self, "channels", channels)
        if self.device.get("backend") == VE_BACKEND:
            _ve_request(self)
    @property
    def effective_streaming(self):
        return self.purpose == "main" and self.streaming


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
    time_mode: str = PREVIEW_TIME_MODE_CUMULATIVE

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("preview request_id is required")
        _integer("generation", self.generation, 1)
        _integer("sequence", self.sequence, 1)
        _integer("sample_stop", self.sample_stop)
        if self.time_mode not in (
            PREVIEW_TIME_MODE_CUMULATIVE,
            PREVIEW_TIME_MODE_RELATIVE_LATEST,
        ):
            raise ValueError("preview time mode is invalid")
        channels = _channels(self.channels)
        if len(channels) != len(self.waveforms):
            raise ValueError("preview channel and waveform counts differ")
        owned = []
        for waveform in self.waveforms:
            structurally_invalid = (
                    not isinstance(waveform, StreamingWaveformSnapshot)
                    or waveform.sample_stop != self.sample_stop
                    or not isinstance(waveform.time, np.ndarray)
                    or not isinstance(waveform.amplitude, np.ndarray)
                    or waveform.time.ndim != 1 or waveform.amplitude.ndim != 1
                    or waveform.time.dtype != np.float64 or waveform.amplitude.dtype != np.float32
                    or len(waveform.time) != len(waveform.amplitude)
                    or len(waveform.time) > MAIN_RECORDING_LIVE_MAX_POINTS
                    or not np.all(np.isfinite(waveform.time))
                    or np.any(np.diff(waveform.time) <= 0)
            )
            if self.time_mode == PREVIEW_TIME_MODE_CUMULATIVE:
                invalid = structurally_invalid or np.any(waveform.time < 0.0)
            else:
                invalid = structurally_invalid or (
                    len(waveform.time) > 0
                    and (
                        waveform.time[-1] != 0.0
                        or np.any(waveform.time > 0.0)
                        or waveform.time[0] < (
                            -MAIN_RECORDING_LIVE_WINDOW_SECONDS
                            - PREVIEW_TIME_LOWER_BOUND_TOLERANCE
                        )
                    )
                )
            if invalid:
                raise ValueError(f"invalid {self.time_mode} preview waveform")
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
                             self.sample_stop, self.channels, self.waveforms,
                             self.time_mode))


@dataclass(frozen=True)
class RecordingProgress:
    """Cumulative valid raw frames, timestamped by the native owner, not IPC."""
    request_id: str
    generation: int
    frames: int
    last_frame_at: float

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("progress request_id is required")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("progress generation must be a positive integer")
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("progress frames must be a nonnegative integer")
        _monotonic_time("last_frame_at", self.last_frame_at)


@dataclass(frozen=True)
class VeLifecycleCounts:
    sdk_open: int
    task_create: int
    task_start: int
    task_stop: int
    task_clear: int
    sdk_close: int

    def __post_init__(self):
        for name in ("sdk_open", "task_create", "task_start", "task_stop",
                     "task_clear", "sdk_close"):
            _integer(name, getattr(self, name))


@dataclass(frozen=True)
class CaptureSlotReleased:
    request_id: str
    generation: int
    target_reached_at: float
    raw_frames: int
    adapter_released: bool
    writer_released: bool
    lifecycle_counts: VeLifecycleCounts

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("slot request_id is required")
        _integer("generation", self.generation, 1)
        _monotonic_time("target_reached_at", self.target_reached_at)
        _integer("raw_frames", self.raw_frames, 1)
        if type(self.adapter_released) is not bool:
            raise ValueError("adapter_released must be a boolean")
        if type(self.writer_released) is not bool:
            raise ValueError("writer_released must be a boolean")
        if not isinstance(self.lifecycle_counts, VeLifecycleCounts):
            raise ValueError("lifecycle_counts must be a VeLifecycleCounts snapshot")
        VeLifecycleCounts.__post_init__(self.lifecycle_counts)


def _acquisition_signature(value):
    if type(value) is not tuple or len(value) != 8:
        raise ValueError("released_signature must be an acquisition signature or None")
    backend, machine_id, channels, sample_rate, mode, unit, minimum, maximum = value
    if backend != VE_BACKEND:
        raise ValueError(f"signature backend must be {VE_BACKEND}")
    normalized_machine_id = normalize_machine_id(machine_id)
    if normalized_machine_id != machine_id:
        raise ValueError("signature machine_id must already be normalized")
    selected = validate_physical_channels(channels)
    if type(channels) is not tuple or selected != channels:
        raise ValueError("signature channels must be an ordered tuple")
    config = validate_input_config({
        "sample_rate": sample_rate, "input_mode": mode, "unit": unit,
        "range_min": minimum, "range_max": maximum,
    })
    return (backend, normalized_machine_id, selected, config["sample_rate"],
            config["input_mode"], config["unit"], config["range_min"], config["range_max"])


@dataclass(frozen=True)
class VePrewarmRequest:
    """Frozen, no-file request for one half-second VE acquisition attempt."""
    warmup_id: str
    device: Mapping
    channels: tuple[int, ...]
    sample_rate: int
    frames_per_channel: int
    attempt: int

    @classmethod
    def create(cls, warmup_id, device, channels, sample_rate, *, attempt):
        rate = validate_sample_rate(sample_rate)
        selected = validate_physical_channels(channels)
        return cls(warmup_id, device, selected, rate,
                   max(1, math.ceil(rate * 0.5)), attempt)

    def __post_init__(self):
        if type(self.warmup_id) is not str or not self.warmup_id:
            raise ValueError("warmup_id is required")
        if self.attempt not in (1, 2) or type(self.attempt) is not int:
            raise ValueError("attempt must be 1 or 2")
        rate = validate_sample_rate(self.sample_rate)
        channels = validate_physical_channels(self.channels)
        snapshot = _device(self.device, channels, "input")
        if snapshot.get("backend") != VE_BACKEND:
            raise ValueError("VE prewarm requires the vkinging backend")
        if snapshot["input_config"]["sample_rate"] != rate:
            raise ValueError("sample_rate must match device.input_config.sample_rate")
        expected_frames = max(1, math.ceil(rate * 0.5))
        if (type(self.frames_per_channel) is not int
                or self.frames_per_channel != expected_frames):
            raise ValueError("frames_per_channel must equal half the sample rate")
        object.__setattr__(self, "device", FrozenConfig.snapshot(snapshot))
        object.__setattr__(self, "channels", channels)

    @property
    def target_samples(self):
        return self.frames_per_channel

    @property
    def signature(self):
        return ve_acquisition_signature(self.device, self.channels, self.sample_rate)


@dataclass(frozen=True)
class VePrewarmStarted:
    """Authenticated native adapter start boundary for one prewarm attempt."""
    warmup_id: str
    generation: int
    attempt: int
    signature: tuple
    started_at: float

    def __post_init__(self):
        if type(self.warmup_id) is not str or not self.warmup_id:
            raise ValueError("warmup_id is required")
        _integer("generation", self.generation, 1)
        if self.attempt not in (1, 2) or type(self.attempt) is not int:
            raise ValueError("attempt must be 1 or 2")
        object.__setattr__(self, "signature", _acquisition_signature(self.signature))
        _monotonic_time("started_at", self.started_at)


@dataclass(frozen=True)
class VePrewarmProgress:
    """Authenticated cumulative per-channel progress for one prewarm attempt."""
    warmup_id: str
    generation: int
    attempt: int
    signature: tuple
    started_at: float
    frames_per_channel: int
    observed_at: float

    def __post_init__(self):
        if type(self.warmup_id) is not str or not self.warmup_id:
            raise ValueError("warmup_id is required")
        _integer("generation", self.generation, 1)
        if self.attempt not in (1, 2) or type(self.attempt) is not int:
            raise ValueError("attempt must be 1 or 2")
        signature = _acquisition_signature(self.signature)
        object.__setattr__(self, "signature", signature)
        _monotonic_time("started_at", self.started_at)
        _integer("frames_per_channel", self.frames_per_channel)
        target = max(1, math.ceil(signature[3] * 0.5))
        if self.frames_per_channel > target:
            raise ValueError("prewarm progress exceeds its per-channel target")
        _monotonic_time("observed_at", self.observed_at)
        if self.observed_at < self.started_at:
            raise ValueError("prewarm observation precedes native start")


@dataclass(frozen=True)
class VePrewarmResult:
    """Structured worker terminal preserving a native first cause verbatim."""
    warmup_id: str
    generation: int
    attempt: int
    signature: tuple
    success: bool
    stage: str
    code: int | None
    detail: str
    frames_per_channel: int
    handles_released: bool
    diagnostics: tuple[str, ...]
    lifecycle_counts: VeLifecycleCounts

    def __post_init__(self):
        if type(self.warmup_id) is not str or not self.warmup_id:
            raise ValueError("warmup_id is required")
        _integer("generation", self.generation, 1)
        if self.attempt not in (1, 2) or type(self.attempt) is not int:
            raise ValueError("attempt must be 1 or 2")
        signature = _acquisition_signature(self.signature)
        object.__setattr__(self, "signature", signature)
        if type(self.success) is not bool:
            raise ValueError("success must be a boolean")
        if type(self.stage) is not str or not self.stage:
            raise ValueError("stage is required")
        if self.code is not None and type(self.code) is not int:
            raise ValueError("code must be an integer or None")
        if type(self.detail) is not str:
            raise ValueError("detail must be a string")
        _integer("frames_per_channel", self.frames_per_channel)
        if type(self.handles_released) is not bool:
            raise ValueError("handles_released must be a boolean")
        if (type(self.diagnostics) is not tuple
                or any(type(item) is not str for item in self.diagnostics)):
            raise ValueError("diagnostics must be a tuple of strings")
        if not isinstance(self.lifecycle_counts, VeLifecycleCounts):
            raise ValueError("lifecycle_counts must be a VeLifecycleCounts snapshot")
        VeLifecycleCounts.__post_init__(self.lifecycle_counts)
        target_frames = max(1, math.ceil(signature[3] * 0.5))
        if self.success:
            if self.stage != "completed":
                raise ValueError("successful prewarm stage must be completed")
            if self.code is not None or self.detail:
                raise ValueError("successful prewarm cannot carry a native error")
            if not self.handles_released:
                raise ValueError("successful prewarm must release its adapter handles")
            if self.frames_per_channel < target_frames:
                raise ValueError("successful prewarm must meet its frame target")
        else:
            if self.stage == "completed":
                raise ValueError("failed prewarm stage cannot be completed")
            if not self.detail:
                raise ValueError("failed prewarm detail is required")


@dataclass(frozen=True)
class VeReleaseOutcome:
    generation: int
    released_signature: tuple | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        _integer("generation", self.generation, 1)
        if self.released_signature is not None:
            object.__setattr__(self, "released_signature",
                               _acquisition_signature(self.released_signature))
        if (type(self.diagnostics) is not tuple
                or any(type(item) is not str for item in self.diagnostics)):
            raise ValueError("diagnostics must be a tuple of strings")


@dataclass(frozen=True)
class WorkerFatal:
    generation: int
    stage: str
    message: str

    def __post_init__(self):
        _integer("generation", self.generation, 1)
        if type(self.stage) is not str or not self.stage:
            raise ValueError("fatal stage is required")
        if type(self.message) is not str or not self.message:
            raise ValueError("fatal message is required")


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
                             "ready", "started", "finalizing", "preview", "progress",
                             "completed", "failed", "cancelled", "capture_slot_released",
                             "release_ve", "ve_released", "ve_release_failed", "worker_fatal",
                             VE_PREWARM_COMMAND, VE_PREWARM_STARTED, VE_PREWARM_PROGRESS,
                             VE_PREWARM_DETACHING, VE_PREWARM_TERMINAL):
            raise ValueError("unknown recording event kind")
        if (self.kind in ("release_ve", "ve_released", "ve_release_failed", "worker_fatal")
                and self.request_id != ""):
            raise ValueError(f"{self.kind} request_id must be empty")
        typed_payloads = {"start": RecordingRequest, "preview": RecordingPreview, "progress": RecordingProgress,
                          "completed": RecordingResult, "failed": RecordingFailure,
                          "cancelled": RecordingCancelled,
                          "capture_slot_released": CaptureSlotReleased,
                          "ve_released": VeReleaseOutcome,
                          "ve_release_failed": VeReleaseOutcome,
                          "worker_fatal": WorkerFatal,
                          VE_PREWARM_COMMAND: VePrewarmRequest,
                          VE_PREWARM_STARTED: VePrewarmStarted,
                          VE_PREWARM_PROGRESS: VePrewarmProgress,
                          VE_PREWARM_DETACHING: VePrewarmProgress,
                          VE_PREWARM_TERMINAL: VePrewarmResult}
        expected = typed_payloads.get(self.kind)
        if expected is not None:
            if not isinstance(self.payload, expected):
                raise ValueError(f"{self.kind} payload must match its type and session")
            if expected in (CaptureSlotReleased, VeReleaseOutcome, WorkerFatal,
                            VePrewarmRequest, VePrewarmStarted, VePrewarmProgress,
                            VePrewarmResult):
                expected.__post_init__(self.payload)
            if expected in (VePrewarmRequest, VePrewarmStarted,
                            VePrewarmProgress, VePrewarmResult):
                if self.request_id != self.payload.warmup_id:
                    raise ValueError(f"{self.kind} warmup ID must match its event")
            if (self.kind == VE_PREWARM_DETACHING
                    and self.payload.frames_per_channel != max(
                        1, math.ceil(self.payload.signature[3] * 0.5))):
                raise ValueError("VE prewarm detaching requires its full frame target")
            payload_request_id = getattr(self.payload, "request_id", self.request_id)
            if payload_request_id != self.request_id:
                raise ValueError(f"{self.kind} payload must match its type and session")
            payload_generation = getattr(self.payload, "generation", self.generation)
            if payload_generation != self.generation:
                raise ValueError(f"{self.kind} generation must match its event")
        elif self.kind == "started" and self.payload is not None:
            _monotonic_time("started", self.payload)
        elif self.kind == "preview_ack":
            _integer("preview acknowledgement sequence", self.payload, 1)
        elif self.kind == "result_ack":
            if self.payload not in ("accepted", "rejected"):
                raise ValueError("result acknowledgement must be accepted or rejected")
        elif self.payload is not None:
            raise ValueError(f"{self.kind} does not carry a payload")
