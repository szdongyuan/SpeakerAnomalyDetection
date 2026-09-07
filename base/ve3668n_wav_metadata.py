"""Closed VE v1 voltage/provenance schema, independent of current registries.

Validation accepts frozen Mapping snapshots and returns owned JSON-ready data.
Historical file rates are positive integers, not the new-acquisition whitelist.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math

from base.ve3668n_input import (
    validate_device_snapshot, validate_input_config, validate_physical_channel,
    validate_physical_channels,
)
from consts.ve3668n_consts import (
    VE_BACKEND, VE_INPUT_MODE, VE_MODEL, VE_RANGE_MAX, VE_RANGE_MIN, VE_UNIT,
)


def _fields(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly {tuple(expected)!r}")


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _calibration(value):
    _fields(value, ("standard_spl", "calibrated_at", "sample_rate", "duration_seconds"),
            "calibration")
    timestamp = value["calibrated_at"]
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("calibrated_at must be an ISO-8601 timestamp with a UTC offset")
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("calibrated_at must include a UTC offset")
    return {
        "standard_spl": _number(value["standard_spl"], "standard_spl"),
        "calibrated_at": timestamp,
        "sample_rate": _positive_integer(value["sample_rate"], "calibration.sample_rate"),
        "duration_seconds": _number(value["duration_seconds"], "duration_seconds", positive=True),
    }


def validate_ve_wav_metadata(payload):
    """Return an independent v1 snapshot, or raise ValueError; never repair it."""
    _fields(payload, ("schema_version", "backend", "acquisition", "recorded_channels"),
            "VE metadata")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported VE schema_version")
    if payload["backend"] != VE_BACKEND:
        raise ValueError(f"backend must be {VE_BACKEND}")
    acquisition = payload["acquisition"]
    _fields(acquisition, ("model", "machine_id", "input_mode", "unit", "range_min",
                          "range_max", "sample_rate"), "acquisition")
    for name, expected in (("model", VE_MODEL), ("input_mode", VE_INPUT_MODE),
                           ("unit", VE_UNIT), ("range_min", VE_RANGE_MIN),
                           ("range_max", VE_RANGE_MAX)):
        value = acquisition[name]
        types = (str,) if isinstance(expected, str) else (int, float)
        if type(value) not in types or value != expected:
            raise ValueError(f"{name} must be {expected!r}")
    machine_id = acquisition["machine_id"]
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise ValueError("machine_id must be a nonempty string")
    _positive_integer(acquisition["sample_rate"], "acquisition.sample_rate")
    channels = payload["recorded_channels"]
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ValueError("recorded_channels must be a nonempty list")
    normalized_channels = []
    indices, physical_channels = set(), set()
    for channel in channels:
        _fields(channel, ("wav_channel_index", "physical_input_channel", "factor_source",
                          "calibrated", "v2pa_factor", "calibration"), "channel")
        index = channel["wav_channel_index"]
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("wav_channel_index must be a unique nonnegative integer")
        physical = validate_physical_channel(channel["physical_input_channel"])
        if physical in physical_channels:
            raise ValueError("physical_input_channel must be unique")
        indices.add(index)
        physical_channels.add(physical)
        source = channel["factor_source"]
        if source == "measured":
            if channel["calibrated"] is not True:
                raise ValueError("measured requires calibrated=true")
            factor = _number(channel["v2pa_factor"], "v2pa_factor", positive=True)
            calibration = _calibration(channel["calibration"])
        elif source == "none":
            if (channel["calibrated"] is not False or channel["v2pa_factor"] is not None
                    or channel["calibration"] is not None):
                raise ValueError("none requires calibrated=false and null factor/calibration")
            factor, calibration = None, None
        else:
            raise ValueError("factor_source must be measured or none")
        normalized_channels.append({
            "wav_channel_index": index, "physical_input_channel": physical,
            "factor_source": source, "calibrated": channel["calibrated"],
            "v2pa_factor": factor, "calibration": calibration,
        })
    if indices != set(range(len(channels))):
        raise ValueError("wav_channel_index must be contiguous from zero")
    return {
        "schema_version": 1, "backend": VE_BACKEND,
        "acquisition": dict(acquisition), "recorded_channels": normalized_channels,
    }


def build_ve_recording_metadata(device, channels, profile, calibration_store):
    """Observe effective records once before recording; retain no store references.

    ``profile`` is the current input_config, not original calibration conditions.
    The injected VECalibrationStore owns applicability/invalidation and raises on
    storage failure. Invalidated diagnostic records become explicit none entries.
    """
    if calibration_store is None:
        raise ValueError("a VE calibration store is required")
    profile = validate_input_config(profile)
    current = validate_device_snapshot({**device, "input_config": profile})
    channels = validate_physical_channels(channels)
    if any(channel not in current["physical_channels"] for channel in channels):
        raise ValueError("selected physical channels are not present in the device")
    records = deepcopy(calibration_store.observe(current))
    recorded_channels = []
    for index, physical in enumerate(channels):
        record = records.get(physical)
        if record is not None and record["status"] not in ("valid", "invalidated"):
            raise ValueError("calibration record status must be valid or invalidated")
        measured = record is not None and record["status"] == "valid"
        recorded_channels.append({
            "wav_channel_index": index, "physical_input_channel": physical,
            "factor_source": "measured" if measured else "none", "calibrated": measured,
            "v2pa_factor": record["v2pa_factor"] if measured else None,
            "calibration": {
                "standard_spl": record["standard_spl"],
                "calibrated_at": record["calibrated_at"],
                "sample_rate": record["calibration_sample_rate"],
                "duration_seconds": record["calibration_duration_seconds"],
            } if measured else None,
        })
    return validate_ve_wav_metadata({
        "schema_version": 1, "backend": VE_BACKEND,
        "acquisition": {"model": current["model"], "machine_id": current["machine_id"], **profile},
        "recorded_channels": recorded_channels,
    })


@dataclass(frozen=True)
class VEWavCalibrationResolution:
    factor: float | None
    state: str  # measured | none | invalid
    diagnostic: str = ""


def resolve_ve_wav_channel_v2pa_factor(metadata, wav_channel_index):
    """Resolve only file-local provenance; missing/invalid factors are never 1."""
    try:
        normalized = validate_ve_wav_metadata(metadata)
    except ValueError as exc:
        return VEWavCalibrationResolution(None, "invalid", str(exc))
    if type(wav_channel_index) is not int or not 0 <= wav_channel_index < len(
        normalized["recorded_channels"]
    ):
        return VEWavCalibrationResolution(None, "invalid", "invalid WAV channel index")
    channel = next(item for item in normalized["recorded_channels"]
                   if item["wav_channel_index"] == wav_channel_index)
    if channel["factor_source"] == "none":
        return VEWavCalibrationResolution(None, "none", "uncalibrated voltage data; no measured Pa/V")
    return VEWavCalibrationResolution(channel["v2pa_factor"], "measured")
