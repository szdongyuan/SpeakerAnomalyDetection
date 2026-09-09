"""Pure, closed-schema VE3668N input contracts; no device or storage I/O.

Validation raises ValueError and accepts Mapping snapshots (including frozen
ones). Returned dictionaries own their data; channel sequences become tuples.
These values can be frozen or serialized without importing the recording layer.
"""
from collections.abc import Mapping, Sequence

from consts.ve3668n_consts import (
    VE_BACKEND,
    VE_DEFAULT_SAMPLE_RATE,
    VE_DEVICE_SNAPSHOT_FIELDS,
    VE_INPUT_CONFIG_FIELDS,
    VE_INPUT_MODE,
    VE_MAX_INPUT_CHANNELS,
    VE_MODEL,
    VE_RANGE_MAX,
    VE_RANGE_MIN,
    VE_SAMPLE_RATES,
    VE_UNIT,
)


def _require_fields(value, fields, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = [key for key in value if key not in fields]
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown!r}")
    missing = [key for key in fields if key not in value]
    if missing:
        raise ValueError(f"{name} is missing fields: {missing!r}")


def _nonempty_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def validate_sample_rate(value):
    """Return a whitelisted Python int; never coerce or substitute a rate."""
    if type(value) is not int or value not in VE_SAMPLE_RATES:
        raise ValueError("sample_rate must be one of 44100, 48000, 51200 Hz")
    return value


def validate_input_config(config):
    """Validate an existing acquisition config without supplying defaults."""
    _require_fields(config, VE_INPUT_CONFIG_FIELDS, "input_config")
    result = {"sample_rate": validate_sample_rate(config["sample_rate"])}
    for field, expected in (
        ("input_mode", VE_INPUT_MODE), ("unit", VE_UNIT),
        ("range_min", VE_RANGE_MIN), ("range_max", VE_RANGE_MAX),
    ):
        value = config[field]
        allowed_types = (str,) if isinstance(expected, str) else (int, float)
        if type(value) not in allowed_types or value != expected:
            raise ValueError(f"{field} must be {expected!r}")
        result[field] = expected
    return result


def create_input_config(sample_rate=VE_DEFAULT_SAMPLE_RATE):
    """Create a new config; only this creation boundary supplies the default."""
    return validate_input_config({
        "sample_rate": sample_rate,
        "input_mode": VE_INPUT_MODE,
        "unit": VE_UNIT,
        "range_min": VE_RANGE_MIN,
        "range_max": VE_RANGE_MAX,
    })


def normalize_model(value):
    """Match the entire trimmed model, case-insensitively, and canonicalize it."""
    if _nonempty_text(value, "model").casefold() != VE_MODEL.casefold():
        raise ValueError(f"model must be {VE_MODEL}")
    return VE_MODEL


def normalize_machine_id(value):
    """Require a stable nonempty ID; trim padding without changing its case."""
    return _nonempty_text(value, "machine_id")


def validate_physical_channel(value):
    """Return one strict Python integer AIN index (0 through 7)."""
    if type(value) is not int or not 0 <= value < VE_MAX_INPUT_CHANNELS:
        raise ValueError("physical_channel must be an integer from 0 through 7")
    return value


def validate_physical_channels(channels):
    """Return a nonempty, unique ordered tuple; never sort or coerce indices."""
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes, bytearray)):
        raise ValueError("physical_channels must be an ordered sequence")
    result = tuple(validate_physical_channel(value) for value in channels)
    if not result or len(set(result)) != len(result):
        raise ValueError("physical_channels must be nonempty and unique")
    return result


def _validate_device_fields(device):
    """Validate device fields; callers separately choose the config boundary."""
    _require_fields(device, VE_DEVICE_SNAPSHOT_FIELDS, "device")
    if device["backend"] != VE_BACKEND:
        raise ValueError(f"backend must be {VE_BACKEND}")
    model = normalize_model(device["model"])
    machine_id = normalize_machine_id(device["machine_id"])
    name = _nonempty_text(device["name"], "name")
    address = device["address"]
    if address is not None and not isinstance(address, str):
        raise ValueError("address must be a string or None")
    channels = validate_physical_channels(device["physical_channels"])
    maximum = device["max_input_channels"]
    if type(maximum) is not int or not 1 <= maximum <= VE_MAX_INPUT_CHANNELS:
        raise ValueError("max_input_channels must be an integer from 1 through 8")
    if max(channels) >= maximum:
        raise ValueError("max_input_channels must cover every physical_channel")
    if type(device["available"]) is not bool:
        raise ValueError("available must be a boolean")
    return {
        "backend": VE_BACKEND, "model": model, "machine_id": machine_id,
        "name": name, "address": address.strip() if address is not None else None,
        "physical_channels": channels, "max_input_channels": maximum,
        "available": device["available"],
    }


def validate_device_snapshot(device):
    """Return an independent snapshot with a current, valid acquisition config.

    Unavailable devices remain representable: this does not authorize capture
    or replace the acquisition boundary's availability and channel checks.
    """
    result = _validate_device_fields(device)
    result["input_config"] = validate_input_config(device["input_config"])
    return result


def ve_acquisition_signature(device, channels, sample_rate):
    """Return the exact immutable identity of reusable native VE resources."""
    snapshot = validate_device_snapshot(device)
    selected = validate_physical_channels(channels)
    rate = validate_sample_rate(sample_rate)
    if not set(selected).issubset(snapshot["physical_channels"]):
        raise ValueError("selected input channels are unavailable")
    return VE_BACKEND, snapshot["machine_id"], selected, rate


def resolve_effective_input_rate(device, product_rate):
    """Use the VE config's rate, or return the legacy product value unchanged.

    Missing devices/backend markers retain sounddevice behavior. An explicit
    unknown backend or invalid VE snapshot is an error, never a fallback.
    """
    if device is None:
        return product_rate
    if not isinstance(device, Mapping):
        raise ValueError("device must be a mapping or None")
    backend = device.get("backend", "sounddevice")
    if backend == "sounddevice":
        return product_rate
    if backend != VE_BACKEND:
        raise ValueError(f"unknown input backend: {backend!r}")
    return validate_device_snapshot(device)["input_config"]["sample_rate"]
