"""Independent, versioned VE profiles and measured channel calibrations.

Both documents have ``schema_version=1`` and ``devices`` keyed by MachineId.
Profile entries are closed input_config objects; calibration entries contain
``channels`` keyed by decimal physical indices, never by sampling rate.

The UI's persisted-profile entry point is ``profiles.load(device, calibrations)``.
It observes saved conditions before rejecting unsupported acquisition settings.
Use its returned config in the current device snapshot before calibration lookup::

    config = profiles.load(device, calibrations)
    current = {**device, "input_config": config}
    factor = calibrations.get_factor(current, physical_channel)

``set_sample_rate`` returns the new config only after a successful profile write.
``observe``/``get_record`` retain invalidated provenance for diagnostics;
``get_factor`` alone resolves a usable coefficient (None for absent/invalidated).
``save`` takes an already-verified measurement; capture/result integrity belongs
to the calibration workflow, not the storage layer.

Keep one owner instance per file. Locks and failed-invalidation retry latches
are instance-scoped; reads reload disk to observe external changes. No published
state is cached: callers keep their old snapshots if an operation raises.
"""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock

from base.ve3668n_input import (
    calibration_fingerprint,
    create_input_config,
    normalize_machine_id,
    normalize_model,
    validate_calibration_conditions,
    validate_input_config,
    validate_physical_channel,
    validate_sample_rate,
)
from consts import model_consts
from consts.ve3668n_consts import VE_BACKEND, VE_INPUT_SCHEMA_VERSION


class VEStoreFormatError(ValueError):
    """A saved VE document is untrusted; preserve it for explicit recovery."""


class VEStoreIOError(OSError):
    """A VE read/write failed; no successful state may be published."""


def _fields(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly {tuple(expected)!r}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_non_json_number(value):
    raise ValueError(f"non-finite JSON literal: {value}")


def _device_fingerprint(device, physical_channel=0):
    if not isinstance(device, Mapping):
        raise ValueError("device must be a mapping")
    return calibration_fingerprint(device, physical_channel, device.get("input_config"))


def _finite_number(value, field, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{field} must be finite" + (" and positive" if positive else ""))
    return number


def _validated_record(record, machine_id, physical_channel):
    _fields(record, (
        "status", "fingerprint", "v2pa_factor", "standard_spl",
        "calibration_sample_rate", "calibration_duration_seconds", "calibrated_at",
    ), "calibration record")
    if record["status"] not in ("valid", "invalidated"):
        raise ValueError("status must be valid or invalidated")
    fingerprint = record["fingerprint"]
    _fields(fingerprint, (
        "backend", "model", "machine_id", "physical_channel",
        "input_mode", "unit", "range_min", "range_max",
    ), "fingerprint")
    if fingerprint["backend"] != VE_BACKEND:
        raise ValueError(f"fingerprint backend must be {VE_BACKEND}")
    model = normalize_model(fingerprint["model"])
    if normalize_machine_id(fingerprint["machine_id"]) != machine_id:
        raise ValueError("fingerprint machine_id does not match its device key")
    if validate_physical_channel(fingerprint["physical_channel"]) != physical_channel:
        raise ValueError("fingerprint physical_channel does not match its channel key")
    conditions = validate_calibration_conditions({
        "sample_rate": None,
        **{key: fingerprint[key] for key in ("input_mode", "unit", "range_min", "range_max")},
    })
    factor = _finite_number(record["v2pa_factor"], "v2pa_factor", positive=True)
    standard_spl = _finite_number(record["standard_spl"], "standard_spl")
    if standard_spl not in (94.0, 114.0):
        raise ValueError("standard_spl must be exactly 94 or 114")
    try:
        sample_rate = validate_sample_rate(record["calibration_sample_rate"])
    except ValueError as exc:
        raise ValueError(f"calibration_sample_rate: {exc}") from exc
    duration = _finite_number(
        record["calibration_duration_seconds"], "calibration_duration_seconds", positive=True,
    )
    timestamp = record["calibrated_at"]
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("calibrated_at must be an ISO-8601 timestamp with a UTC offset")
    timestamp = timestamp.strip()
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("calibrated_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("calibrated_at must include a UTC offset")
    return {
        "status": record["status"],
        "fingerprint": {
            "backend": VE_BACKEND, "model": model, "machine_id": machine_id,
            "physical_channel": physical_channel, **conditions,
        },
        "v2pa_factor": factor, "standard_spl": standard_spl,
        "calibration_sample_rate": sample_rate,
        "calibration_duration_seconds": duration, "calibrated_at": timestamp,
    }


def _atomic_write_json(path, payload):
    descriptor = None
    temporary = None
    stream = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None  # Ownership transferred to stream.
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        # File/JSON serialization boundary. Cleanup is best-effort but retains
        # diagnostics, and must never replace the original persistence error.
        error = VEStoreIOError(f"Could not update VE store {path}: {exc}")
        if stream is not None:
            try:
                stream.close()
            except OSError as cleanup:
                error.add_note(f"Could not close temporary {temporary}: {cleanup}")
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as cleanup:
                error.add_note(f"Could not close temporary descriptor: {cleanup}")
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError as cleanup:
                error.add_note(f"Could not remove owned temporary {temporary}: {cleanup}")
        raise error from exc


class _JsonStore:
    """Shared file boundary; callers hold their instance lock across mutations."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = RLock()

    def _read(self):
        try:
            stream = self.path.open(encoding="utf-8")
        except FileNotFoundError:
            return {"schema_version": VE_INPUT_SCHEMA_VERSION, "devices": {}}
        except OSError as exc:
            raise VEStoreIOError(f"Could not read VE store {self.path}: {exc}") from exc
        try:
            with stream:
                payload = json.load(
                    stream, object_pairs_hook=_unique_object,
                    parse_constant=_reject_non_json_number,
                )
            _fields(payload, ("schema_version", "devices"), "root")
            version = payload["schema_version"]
            if type(version) is not int or version != VE_INPUT_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version: {version!r}")
            if not isinstance(payload["devices"], Mapping):
                raise ValueError("devices must be an object keyed by machine_id")
            devices = {}
            for key, entry in payload["devices"].items():
                machine_id = normalize_machine_id(key)
                if machine_id in devices:
                    raise ValueError(f"duplicate machine_id: {machine_id!r}")
                devices[machine_id] = self._validate_entry(machine_id, entry)
            return {"schema_version": VE_INPUT_SCHEMA_VERSION, "devices": devices}
        except (ValueError, RecursionError) as exc:
            raise VEStoreFormatError(f"Invalid VE store {self.path}: {exc}") from exc
        except OSError as exc:
            raise VEStoreIOError(f"Could not read VE store {self.path}: {exc}") from exc


class VEInputProfileStore(_JsonStore):
    """Acquisition configuration only; calibration is always a separate file."""

    def __init__(self, path=None):
        if path is None:
            path = (Path(model_consts.DEFAULT_DIR) / "configs/ve3668n_input_profiles.json").resolve()
        super().__init__(path)

    def _validate_entry(self, machine_id, entry):
        # A structurally valid saved rate may be wrong and need user repair;
        # future conditions must be observable before acquisition validation.
        validate_calibration_conditions(entry)
        return deepcopy(entry)

    def load(self, device, calibrations):
        """Observe the persisted profile, then return a supported input_config.

        Only a new device/file gets defaults. Corrupt/unreadable documents and
        invalid saved rates raise; this method never writes profile defaults.
        """
        with self._lock:
            machine_id = _device_fingerprint(device)["machine_id"]
            registry = self._read()
            config = registry["devices"].get(machine_id, create_input_config())
            calibrations.observe({**device, "input_config": config})
            return validate_input_config(config)

    def set_sample_rate(self, device, sample_rate, calibrations):
        """Commit a legal rate (including explicit repair of an invalid rate).

        Unchanged conditions never write calibration. An external condition
        mismatch is observed and persisted, then rejected without a profile
        write. An invalid requested rate fails before either store is changed.
        """
        validate_sample_rate(sample_rate)
        with self._lock:
            machine_id = _device_fingerprint(device)["machine_id"]
            registry = self._read()
            config = registry["devices"].get(machine_id, create_input_config())
            config = {**config, "sample_rate": sample_rate}
            calibrations.observe({**device, "input_config": config})
            config = validate_input_config(config)
            registry["devices"][machine_id] = config
            _atomic_write_json(self.path, registry)
            return deepcopy(config)


class VECalibrationStore(_JsonStore):
    """One current measured record per MachineId and physical channel."""

    def __init__(self, path=None):
        if path is None:
            path = (Path(model_consts.DEFAULT_DIR) / "configs/ve3668n_input_calibration.json").resolve()
        super().__init__(path)
        # An I/O failure must not forget a mismatch already observed by this
        # owner. This is a retry latch, not a published calibration-status cache.
        self._pending_invalidations = set()

    def _validate_entry(self, machine_id, entry):
        _fields(entry, ("channels",), "device calibration")
        if not isinstance(entry["channels"], Mapping):
            raise ValueError("channels must be an object")
        channels = {}
        for key, record in entry["channels"].items():
            if not isinstance(key, str) or len(key) != 1 or key not in "01234567":
                raise ValueError("channel keys must be canonical physical indices 0 through 7")
            channels[key] = _validated_record(record, machine_id, int(key))
        return {"channels": channels}

    def _write(self, registry, *, recalibrated_channel=None):
        """Commit all observed invalidations with this registry mutation.

        Only a new measurement can supersede its channel's pending mismatch.
        Failed writes leave the retry latch intact and publish no new state.
        """
        for machine_id, channel in self._pending_invalidations:
            if (machine_id, channel) == recalibrated_channel:
                continue
            channels = registry["devices"].get(machine_id, {"channels": {}})["channels"]
            record = channels.get(str(channel))
            if record is not None:
                record["status"] = "invalidated"
        _atomic_write_json(self.path, registry)
        self._pending_invalidations.clear()

    def observe(self, device):
        """Persist sticky mismatches for *all* this device's saved channels.

        Identity/conditions are structurally validated first. Acquisition
        validation happens only after invalidation is durable. On persistence
        failure raise without returning records; the same owner retries pending
        invalidations even if the external profile has since been restored.
        Returns independent records keyed by physical integer channel.
        """
        identity = _device_fingerprint(device)
        with self._lock:
            registry = self._read()
            channels = registry["devices"].get(identity["machine_id"], {"channels": {}})["channels"]
            changed = False
            for channel, record in channels.items():
                expected = {**identity, "physical_channel": int(channel)}
                key = (identity["machine_id"], int(channel))
                if record["status"] == "valid" and record["fingerprint"] != expected:
                    self._pending_invalidations.add(key)
                if record["status"] == "valid" and key in self._pending_invalidations:
                    record["status"] = "invalidated"
                    changed = True
            if changed:
                self._write(registry)
            self._pending_invalidations.difference_update(
                (identity["machine_id"], int(channel)) for channel in channels
            )
            validate_input_config(device["input_config"])
            return {int(channel): deepcopy(record) for channel, record in channels.items()}

    def get_record(self, device, physical_channel):
        """Observe first; return diagnostic provenance or None if absent."""
        channel = validate_physical_channel(physical_channel)
        return self.observe(device).get(channel)

    def get_factor(self, device, physical_channel):
        """Observe first; return only a valid measured Pa/V factor, else None."""
        record = self.get_record(device, physical_channel)
        if record is None or record["status"] != "valid":
            return None
        return record["v2pa_factor"]

    def save(
        self, device, physical_channel, *, v2pa_factor, standard_spl,
        calibration_sample_rate, calibration_duration_seconds, calibrated_at,
    ):
        """Atomically replace one channel with a verified successful measurement.

        Measurement rate/duration/timestamp are required provenance. The rate
        need not equal the current device rate and is never part of identity.
        """
        fingerprint = _device_fingerprint(device, physical_channel)
        validate_input_config(device["input_config"])
        record = _validated_record({
            "status": "valid", "fingerprint": fingerprint,
            "v2pa_factor": v2pa_factor, "standard_spl": standard_spl,
            "calibration_sample_rate": calibration_sample_rate,
            "calibration_duration_seconds": calibration_duration_seconds,
            "calibrated_at": calibrated_at,
        }, fingerprint["machine_id"], physical_channel)
        with self._lock:
            registry = self._read()
            channels = registry["devices"].setdefault(fingerprint["machine_id"], {"channels": {}})["channels"]
            channels[str(physical_channel)] = record
            self._write(registry, recalibrated_channel=(fingerprint["machine_id"], physical_channel))
            return deepcopy(record)

    def reset(self, device, physical_channel):
        """Clear only this channel; return whether a record was removed.

        Does not edit the profile or authorize acquisition, so a structurally
        identifiable channel can be reset even with unsupported conditions.
        """
        identity = _device_fingerprint(device, physical_channel)
        with self._lock:
            registry = self._read()
            channels = registry["devices"].get(identity["machine_id"], {"channels": {}})["channels"]
            if channels.pop(str(physical_channel), None) is None:
                return False
            if not channels:
                del registry["devices"][identity["machine_id"]]
            self._write(registry)
            return True
