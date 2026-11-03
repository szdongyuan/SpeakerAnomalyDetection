"""Pure vkinging input validation and immutable requests; no hardware I/O."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import fspath
import re
from types import MappingProxyType

from consts.audio_consts import VALID_BIT_DEPTHS
from consts.vkinging_consts import (
    VKINGING_HOSTAPI_NAME,
    VKINGING_INPUT_MODE,
    VKINGING_RANGE_INDEX_CONFIG_KEY,
    VKINGING_RANGE_LIMITS,
    VKINGING_SAMPLE_RATE_MAX,
    VKINGING_SAMPLE_RATE_MIN,
    VKINGING_UNIT,
)


def is_vkinging_device(device):
    """Recognize only the reserved Host API value, including frozen mappings."""
    return isinstance(device, Mapping) and device.get("hostapi_name") == VKINGING_HOSTAPI_NAME


def validate_sample_rate(value):
    if type(value) is not int or not VKINGING_SAMPLE_RATE_MIN <= value <= VKINGING_SAMPLE_RATE_MAX:
        raise ValueError(
            f"sample_rate must be an integer from {VKINGING_SAMPLE_RATE_MIN} through "
            f"{VKINGING_SAMPLE_RATE_MAX} Hz")
    return value


def validate_bit_depth(value):
    if type(value) is not int or value not in VALID_BIT_DEPTHS:
        raise ValueError("bit_depth must be 32 or 64 for floating point voltage")
    return value


def validate_range_index(value):
    if type(value) is not int or not 0 <= value < len(VKINGING_RANGE_LIMITS):
        raise ValueError(f"{VKINGING_RANGE_INDEX_CONFIG_KEY} must be an integer from 0 through 6")
    return value


def voltage_limits_for_range(range_index):
    limit = VKINGING_RANGE_LIMITS[validate_range_index(range_index)]
    return -limit, limit


def _range_index(detail):
    if not isinstance(detail, Mapping):
        raise ValueError("detail must be a mapping")
    return validate_range_index(detail.get(VKINGING_RANGE_INDEX_CONFIG_KEY, 0))


def resolve_vkinging_input_config(detail, device):
    """Read rate/format from hardware and range from this queue item only."""
    if not is_vkinging_device(device):
        raise ValueError(f"device.hostapi_name must be {VKINGING_HOSTAPI_NAME!r}")
    minimum, maximum = voltage_limits_for_range(_range_index(detail))
    return {
        "sample_rate": validate_sample_rate(device.get("samplerate")),
        "bit_depth": validate_bit_depth(device.get("bit_depth")),
        "input_mode": VKINGING_INPUT_MODE,
        "unit": VKINGING_UNIT,
        "range_min": minimum,
        "range_max": maximum,
    }


def resolve_calibration_range_index(acq_config, defaults):
    """Resolve a current acquisition item or the full saved mode defaults.

    Both serialized items and the existing mode/detail objects are accepted.
    A historical RECORD_ONLY item missing its range always means built-in 0.
    """
    if isinstance(acq_config, Mapping):
        mode = acq_config.get("mode")
        detail = acq_config.get("detail", {})
    else:
        mode = getattr(acq_config, "mode", None)
        detail = getattr(acq_config, "detail", {})
    if mode == "RECORD_ONLY":
        return _range_index(detail)
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults must be a mapping of acquisition modes")
    return _range_index(defaults.get("RECORD_ONLY", {}))


def parse_vkinging_ain_route(route):
    """Convert a complete SDK device/AIN positive route to its physical index."""
    match = re.fullmatch(r"([^/]+)/AIN([1-9][0-9]*)", route) if isinstance(route, str) else None
    if match is None or not match.group(1).strip():
        raise ValueError("route must have the form device/AIN followed by a positive integer")
    return int(match.group(2)) - 1


def validate_physical_channel(value):
    if type(value) is not int or value < 0:
        raise ValueError("physical_channel must be a nonnegative integer")
    return value


def validate_physical_channels(channels):
    """Own an ordered selection without inventing a device channel limit."""
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes, bytearray)):
        raise ValueError("physical_channels must be an ordered sequence")
    result = tuple(validate_physical_channel(value) for value in channels)
    if not result or len(set(result)) != len(result):
        raise ValueError("physical_channels must be nonempty and unique")
    return result


def _nonempty_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _freeze_snapshot(value):
    """Recursively own JSON-like state and reject unknown mutable objects."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("snapshot mapping keys must be strings")
        return MappingProxyType({key: _freeze_snapshot(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot(child) for child in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"snapshot contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class VkingingCaptureRequest:
    """Validated capture values detached from all mutable UI/configuration state.

    Counts are per channel. target_samples excludes discard_initial_samples.
    Output may be absent for an in-memory calibration; filesystem work belongs
    to the worker. Snapshots are mappings/tuples of JSON scalar values.
    """

    request_id: str
    purpose: str
    device: Mapping
    channels: tuple[int, ...]
    sample_rate: int
    bit_depth: int
    range_index: int
    target_samples: int
    discard_initial_samples: int = 0
    output_path: str | None = None
    calibration_snapshot: Mapping | None = None

    calibration_target_spl: int | None = None

    def __post_init__(self):
        _nonempty_text(self.request_id, "request_id")
        if self.purpose not in ("main", "calibration"):
            raise ValueError("purpose must be 'main' or 'calibration'")
        if not is_vkinging_device(self.device):
            raise ValueError(f"device.hostapi_name must be {VKINGING_HOSTAPI_NAME!r}")
        _nonempty_text(self.device.get("hardware_id"), "device.hardware_id")
        _nonempty_text(self.device.get("name"), "device.name")
        physical_channels = validate_physical_channels(self.device.get("physical_channels"))
        channels = validate_physical_channels(self.channels)
        if not set(channels).issubset(physical_channels):
            raise ValueError("channels must belong to the device physical_channels")
        if validate_sample_rate(self.sample_rate) != validate_sample_rate(self.device.get("samplerate")):
            raise ValueError("sample_rate must equal the hardware samplerate")
        if validate_bit_depth(self.bit_depth) != validate_bit_depth(self.device.get("bit_depth")):
            raise ValueError("bit_depth must equal the hardware bit_depth")
        validate_range_index(self.range_index)
        for field, value, minimum in (
            ("target_samples", self.target_samples, 1),
            ("discard_initial_samples", self.discard_initial_samples, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if self.purpose == "calibration" and (
            len(channels) != 1 or self.target_samples != self.sample_rate * 10 or self.discard_initial_samples != 0
        ):
            raise ValueError("calibration requires one channel, exactly 10 seconds and no discarded samples")
        if self.calibration_target_spl is not None and (
            self.purpose != "calibration" or type(self.calibration_target_spl) is not int
            or self.calibration_target_spl not in (94, 114)
        ):
            raise ValueError("calibration_target_spl must be 94 or 114 for calibration")
        output_path = self.output_path
        if output_path is not None:
            try:
                output_path = fspath(output_path)
            except TypeError as exc:
                raise ValueError("output_path must be a string path or None") from exc
            _nonempty_text(output_path, "output_path")
        if self.calibration_snapshot is not None and not isinstance(self.calibration_snapshot, Mapping):
            raise ValueError("calibration_snapshot must be a mapping or None")
        device_snapshot = dict(self.device)
        device_snapshot["physical_channels"] = physical_channels
        object.__setattr__(self, "device", _freeze_snapshot(device_snapshot))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "calibration_snapshot", _freeze_snapshot(self.calibration_snapshot))
