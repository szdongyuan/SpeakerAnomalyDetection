"""Immutable plans and file descriptors for station recording."""
from collections.abc import Mapping
from dataclasses import dataclass
import ntpath
import os
import re
from pathlib import Path, PureWindowsPath
from string import ascii_letters


def _identity(name, value):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_workbench_request_fields(request_id, sample_rate):
    """Validate the request values copied into the completion descriptor."""
    _identity("request_id", request_id)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


def _has_invalid_windows_path_char(name):
    return any(char in '<>:"/\\|?*' or ord(char) < 32 for char in name)


def _safe_windows_component(name):
    return (
        bool(name)
        and not name.endswith((".", " "))
        and not _has_invalid_windows_path_char(name)
        and not PureWindowsPath(name).is_reserved()
    )


def _validate_windows_anchor(path, drive, label):
    extended_prefix = "\\\\?\\"
    if drive.startswith(extended_prefix):
        # The extended namespace passes the original path to Windows unchanged.
        tail = path.removeprefix(extended_prefix).removesuffix("\\")
        if "/" in path or any(part in ("", ".", "..") for part in tail.split("\\")):
            raise ValueError(f"{label} has an invalid extended Windows path component")
        drive = drive.removeprefix(extended_prefix)
        if re.fullmatch(
            r"Volume\{[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}",
            drive, re.IGNORECASE | re.ASCII,
        ):
            return
        if drive.lower().startswith("unc\\"):
            drive = "\\\\" + drive[4:]
    if len(drive) == 2 and drive[0] in ascii_letters and drive[1] == ":":
        return
    if drive.startswith("\\\\"):
        server, _, share = drive[2:].partition("\\")
        if (
            server and share and server not in (".", "..")
            and not server.endswith(" ") and not _has_invalid_windows_path_char(server)
        ):
            # A UNC share is part of the anchor and cannot be a dot component.
            if not _safe_windows_component(share):
                raise ValueError(f"{label} has unsafe Windows path component {share!r}")
            if share.lower() in ("pipe", "mailslot", "ipc$"):
                raise ValueError(f"{label} has a non-filesystem UNC share {share!r}")
            return
    raise ValueError(f"{label} requires a Windows drive or UNC share, or a volume GUID")


def _validate_filesystem_path(path, label):
    """Check syntax without resolving short names, links, or other filesystem aliases."""
    parsed = Path(path) if type(path) is str and "\0" not in path else None
    if parsed is None or not parsed.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if os.name == "nt":
        _validate_windows_anchor(path, parsed.drive, label)
    for name in parsed.parts[1:]:
        if name != ".." and not _safe_windows_component(name):
            raise ValueError(f"{label} has unsafe Windows path component {name!r}")


def _directory_key(path):
    _validate_filesystem_path(path, "station recording directory")
    key = os.path.normcase(os.path.normpath(path))
    if os.path.dirname(key) == key:
        raise ValueError("station recording directory cannot be a filesystem root")
    return key


def _validate_wav_filename(name, station_id):
    if not _safe_windows_component(name) or PureWindowsPath(name).suffix.lower() != ".wav":
        raise ValueError(f"station {station_id!r} requires a plain WAV filename")


@dataclass(frozen=True)
class StationWavChannel:
    channel_id: str
    channel_name: str
    signal_type: str
    device_ref: str
    physical_channel_index: int
    wav_channel_index: int

    def __post_init__(self):
        for name in ("channel_id", "channel_name", "signal_type", "device_ref"):
            _identity(name, getattr(self, name))
        for name, index in (("physical channel", self.physical_channel_index),
                            ("WAV channel index", self.wav_channel_index)):
            if type(index) is not int or index < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def _station_channels(channels, station_id):
    if not isinstance(channels, (list, tuple)):
        raise ValueError(f"station {station_id!r} requires ordered channels")
    channels = tuple(channels)
    if not channels or any(not isinstance(channel, StationWavChannel) for channel in channels):
        raise ValueError(f"station {station_id!r} requires mapped channels")
    if len({channel.physical_channel_index for channel in channels}) != len(channels):
        raise ValueError(f"station {station_id!r} has duplicate physical channels")
    if any(channel.wav_channel_index != index for index, channel in enumerate(channels)):
        raise ValueError(f"station {station_id!r} WAV channel index must follow channel order")
    return channels


@dataclass(frozen=True)
class StationRecordingPlan:
    station_id: str
    filename: str
    channels: tuple[StationWavChannel, ...]
    calibration_metadata: Mapping | None = None

    def __post_init__(self):
        # FrozenConfig is also the snapshot type of the enclosing recording request.
        from base.recording_process_protocol import FrozenConfig

        _identity("station_id", self.station_id)
        _identity("filename", self.filename)
        _validate_wav_filename(self.filename, self.station_id)
        object.__setattr__(self, "channels", _station_channels(self.channels, self.station_id))

        if self.calibration_metadata is not None:
            if not isinstance(self.calibration_metadata, Mapping):
                raise ValueError(f"station {self.station_id!r} calibration_metadata must be a mapping")
            object.__setattr__(
                self, "calibration_metadata", FrozenConfig.snapshot(self.calibration_metadata),
            )

    @property
    def physical_channels(self):
        return tuple(channel.physical_channel_index for channel in self.channels)


@dataclass(frozen=True)
class StationRecordingGroup:
    project_id: str
    condition_id: str
    final_path: str
    stations: tuple[StationRecordingPlan, ...]
    prepared_generation: int | None = None
    allocate_final_path: bool = False

    def __post_init__(self):
        generation = self.prepared_generation
        if generation is not None and (type(generation) is not int or generation < 1):
            raise ValueError("prepared_generation must be a positive integer")
        _identity("project_id", self.project_id)
        _identity("condition_id", self.condition_id)
        _directory_key(self.final_path)
        if not isinstance(self.stations, (list, tuple)):
            raise ValueError("station group requires ordered stations")
        stations = tuple(self.stations)
        if not stations or any(not isinstance(station, StationRecordingPlan) for station in stations):
            raise ValueError("station group requires at least one station plan in stations")
        if len({station.station_id for station in stations}) != len(stations):
            raise ValueError("station_id must be unique within the station group")
        if len({ntpath.normcase(station.filename) for station in stations}) != len(stations):
            raise ValueError("station filename must be unique within the station group")
        if len({channel.device_ref for station in stations for channel in station.channels}) != 1:
            raise ValueError("all station channels must share one device_ref")
        object.__setattr__(self, "stations", stations)

    def validate_request(self, staging_path, channels):
        staging = _directory_key(staging_path)
        final = _directory_key(self.final_path)
        if staging == final or os.path.dirname(staging) != os.path.dirname(final):
            raise ValueError("staging and final directories must be distinct siblings")
        available = set(channels)
        for station in self.stations:
            missing = set(station.physical_channels) - available
            if missing:
                raise ValueError(
                    f"station {station.station_id!r} uses unavailable physical channels {sorted(missing)}"
                )


@dataclass(frozen=True)
class StationWav:
    station_id: str
    path: str
    channels: tuple[StationWavChannel, ...]

    def __post_init__(self):
        _identity("station_id", self.station_id)
        _validate_filesystem_path(self.path, "station WAV path")
        _validate_wav_filename(os.path.basename(self.path), self.station_id)
        object.__setattr__(self, "channels", _station_channels(self.channels, self.station_id))


def freeze_station_wavs(stations):
    """Own an ordered collection of typed, immutable WAV descriptors."""
    if not isinstance(stations, (tuple, list)):
        raise ValueError("station WAVs must be an ordered collection")
    stations = tuple(stations)
    if any(not isinstance(station, StationWav) for station in stations):
        raise ValueError("station WAVs must contain StationWav descriptors")
    return stations


@dataclass(frozen=True)
class WorkbenchRecordingCompletion:
    request_id: str
    project_id: str
    condition_id: str
    sample_rate: int
    final_frames: int
    stations: tuple[StationWav, ...]

    def __post_init__(self):
        validate_workbench_request_fields(self.request_id, self.sample_rate)
        _identity("project_id", self.project_id)
        _identity("condition_id", self.condition_id)
        if type(self.final_frames) is not int or self.final_frames < 0:
            raise ValueError("final_frames must be a nonnegative integer")
        stations = freeze_station_wavs(self.stations)
        if not stations:
            raise ValueError("completion requires at least one station WAV")
        object.__setattr__(self, "stations", stations)
