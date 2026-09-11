"""Pure, closed-schema VE3668N input contracts; no device or storage I/O.

Validation raises ValueError and accepts Mapping snapshots (including frozen
ones). Returned dictionaries own their data; channel sequences become tuples.
These values can be frozen or serialized without importing the recording layer.
"""
from collections.abc import Mapping, Sequence
import math

from consts.ve3668n_consts import (
    VE_BACKEND,
    VE_DEFAULT_SAMPLE_RATE,
    VE_DEVICE_SNAPSHOT_FIELDS,
    VE_INPUT_CONFIG_FIELDS,
    VE_INPUT_MODE,
    VE_MAX_INPUT_CHANNELS,
    VE_MODEL,
    VE_RANGE_INDEX_CONFIG_KEY,
    VE_RANGE_LIMITS,
    VE_SAMPLE_RATE_MAX,
    VE_SAMPLE_RATE_MIN,
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


def _finite_range(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_sample_rate(value):
    """Return an in-range Python int; never coerce or substitute a rate."""
    if type(value) is not int or not VE_SAMPLE_RATE_MIN <= value <= VE_SAMPLE_RATE_MAX:
        raise ValueError(
            f"sample_rate must be an integer from {VE_SAMPLE_RATE_MIN} through {VE_SAMPLE_RATE_MAX} Hz")
    return value


def validate_range_index(value):
    """Return a strict queue range enumeration, never a voltage value."""
    if type(value) is not int or not 0 <= value < len(VE_RANGE_LIMITS):
        raise ValueError(f"{VE_RANGE_INDEX_CONFIG_KEY} must be an integer from 0 through 6")
    return value


def voltage_limits_for_range(range_index):
    """Map a queue enumeration to the SDK's minimum and maximum volts."""
    limit = VE_RANGE_LIMITS[validate_range_index(range_index)]
    return -limit, limit


def validate_voltage_range(range_min, range_max):
    """Validate one of the seven symmetric voltage ranges, independently of rate."""
    minimum = _finite_range(range_min, "range_min")
    maximum = _finite_range(range_max, "range_max")
    if maximum not in VE_RANGE_LIMITS or minimum != -maximum:
        raise ValueError(f"range_min and range_max must be a symmetric range from {VE_RANGE_LIMITS!r} V")
    return minimum, maximum


def validate_input_config(config):
    """Validate an existing acquisition config without supplying defaults."""
    _require_fields(config, VE_INPUT_CONFIG_FIELDS, "input_config")
    result = {"sample_rate": validate_sample_rate(config["sample_rate"])}
    for field, expected in (
        ("input_mode", VE_INPUT_MODE), ("unit", VE_UNIT),
    ):
        value = config[field]
        if type(value) is not str or value != expected:
            raise ValueError(f"{field} must be {expected!r}")
        result[field] = expected
    minimum, maximum = validate_voltage_range(config["range_min"], config["range_max"])
    result.update(range_min=minimum, range_max=maximum)
    return result


def create_input_config(sample_rate=VE_DEFAULT_SAMPLE_RATE, *, range_index=0):
    """Create a new config; only this creation boundary supplies the default."""
    minimum, maximum = voltage_limits_for_range(range_index)
    return validate_input_config({
        "sample_rate": sample_rate,
        "input_mode": VE_INPUT_MODE,
        "unit": VE_UNIT,
        "range_min": minimum,
        "range_max": maximum,
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
    config = snapshot["input_config"]
    if rate != config["sample_rate"]:
        raise ValueError("sample_rate must match device.input_config.sample_rate")
    if not set(selected).issubset(snapshot["physical_channels"]):
        raise ValueError("selected input channels are unavailable")
    return (VE_BACKEND, snapshot["machine_id"], selected, rate,
            config["input_mode"], config["unit"], config["range_min"], config["range_max"])


def validate_calibration_conditions(config):
    """Extract structural conditions, NOT permission to acquire.

    The full closed input_config schema is required, but sample_rate is ignored.
    Future nonempty modes/units and finite ordered ranges must remain observable
    for sticky invalidation, even when the saved rate is invalid. Acquisition
    callers must instead use validate_input_config or validate_device_snapshot.
    """
    _require_fields(config, VE_INPUT_CONFIG_FIELDS, "input_config")
    result = {
        "input_mode": _nonempty_text(config["input_mode"], "input_mode"),
        "unit": _nonempty_text(config["unit"], "unit"),
        "range_min": _finite_range(config["range_min"], "range_min"),
        "range_max": _finite_range(config["range_max"], "range_max"),
    }
    if result["range_min"] >= result["range_max"]:
        raise ValueError("range_min must be less than range_max")
    return result


def calibration_fingerprint(device, physical_channel, config):
    """Return the full historical channel conditions, not a capture configuration.

    Routing/display names, selected-channel order, availability and sample_rate
    do not identify a calibration. Config conditions are structurally validated
    without enabling unsupported acquisitions.
    """
    identity = _validate_device_fields(device)
    validate_calibration_conditions(device["input_config"])
    conditions = validate_calibration_conditions(config)
    return {
        "backend": VE_BACKEND, "model": VE_MODEL,
        "machine_id": identity["machine_id"],
        "physical_channel": validate_physical_channel(physical_channel),
        **conditions,
    }


def calibration_applicability_key(fingerprint):
    """Project an already validated fingerprint onto completed-factor identity.

    Historical ranges remain in the full fingerprint for provenance and active
    calibration ownership; legal acquisition ranges do not change a Pa/V factor.
    """
    return tuple(fingerprint[name] for name in (
        "backend", "model", "machine_id", "physical_channel", "input_mode", "unit",
    ))


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
