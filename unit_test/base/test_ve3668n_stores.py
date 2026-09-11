"""VE persistence contracts; every file is isolated under pytest's tmp_path."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from base import ve3668n_stores as stores
from consts.ve3668n_consts import VE_RANGE_LIMITS
from unit_test.base.ve3668n_fakes import device_info, input_config


def make_stores(tmp_path):
    return (
        stores.VEInputProfileStore(tmp_path / "configs/ve3668n_input_profiles.json"),
        stores.VECalibrationStore(tmp_path / "configs/ve3668n_input_calibration.json"),
    )


def measured_record(device, channel, factor=10.0, rate=51200):
    from base.ve3668n_input import calibration_fingerprint

    return {
        "status": "valid",
        "fingerprint": calibration_fingerprint(device, channel, device["input_config"]),
        "v2pa_factor": factor,
        "standard_spl": 94.0,
        "calibration_sample_rate": rate,
        "calibration_duration_seconds": 10.0,
        "calibrated_at": "2026-08-28T10:00:00+08:00",
    }


def save_measurement(calibrations, device, channel, **overrides):
    record = measured_record(device, channel)
    record.update(overrides)
    return calibrations.save(
        device, channel,
        **{key: value for key, value in record.items() if key not in ("status", "fingerprint")},
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("initial_rate", [44100, 48000, 51200])
@pytest.mark.parametrize("new_rate", [44100, 48000, 51200])
def test_rate_change_does_not_write_or_mutate_calibration(
    tmp_path, monkeypatch, initial_rate, new_rate,
):
    profiles, calibrations = make_stores(tmp_path)
    devices = [device_info(machine_id=f"machine-{index}") for index in (1, 2)]
    for index, device in enumerate(devices):
        profiles.set_sample_rate(device, initial_rate, calibrations)
        for channel in (7, 1):
            save_measurement(
                calibrations, device, channel, v2pa_factor=10.0 + index + channel,
                standard_spl=114.0 if channel == 1 else 94.0,
                calibration_sample_rate=44100 if index else 51200,
                calibration_duration_seconds=10.0 + index,
                calibrated_at=f"2026-08-28T1{index}:00:00+08:00",
            )

    # Seed an already-invalidated record without requiring observation yet.
    saved = read_json(calibrations.path)
    saved["devices"]["machine-2"]["channels"]["1"]["status"] = "invalidated"
    write_json(calibrations.path, saved)
    original_bytes = calibrations.path.read_bytes()
    original = deepcopy(saved)
    replace = stores.os.replace
    profile_writes = []

    def only_replace_profiles(source, target):
        assert Path(target) == profiles.path, "rate changes must never write calibration"
        profile_writes.append(target)
        return replace(source, target)

    monkeypatch.setattr(stores.os, "replace", only_replace_profiles)
    for rate in (new_rate, initial_rate):
        for device in devices:
            assert profiles.set_sample_rate(device, rate, calibrations) == input_config(rate)
        profiles, calibrations = make_stores(tmp_path)
        for device in devices:
            config = profiles.load(device, calibrations)
            assert config == input_config(rate)
            current = {**device, "input_config": config}
            for channel in (7, 1):
                expected = original["devices"][device["machine_id"]]["channels"][str(channel)]
                assert calibrations.get_record(current, channel) == expected
                expected_factor = expected["v2pa_factor"] if expected["status"] == "valid" else None
                assert calibrations.get_factor(current, channel) == expected_factor
            assert calibrations.get_record(current, 0) is None
            assert calibrations.get_factor(current, 0) is None
        assert calibrations.path.read_bytes() == original_bytes
        assert read_json(calibrations.path) == original
    assert profile_writes


@pytest.mark.parametrize("initial_limit", VE_RANGE_LIMITS)
def test_all_legal_ranges_and_rates_retain_calibration_without_writes(tmp_path, monkeypatch, initial_limit):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info(input_config=input_config(range_min=-initial_limit, range_max=initial_limit))
    original = save_measurement(calibrations, device, 7)
    save_measurement(calibrations, device, 1)
    saved = read_json(calibrations.path)
    saved["devices"][device["machine_id"]]["channels"]["1"]["status"] = "invalidated"
    write_json(calibrations.path, saved)
    before = calibrations.path.read_bytes()

    def no_write(*args):
        pytest.fail("legal range/rate changes must not write calibration")

    monkeypatch.setattr(stores.os, "replace", no_write)
    for limit in VE_RANGE_LIMITS:
        for rate in (8000, 32000, 44100, 48000, 51200, 96000, 102400):
            config = input_config(rate, range_min=-limit, range_max=limit)
            current = {**device, "input_config": config}
            write_json(profiles.path, {"schema_version": 1, "devices": {device["machine_id"]: config}})
            assert profiles.load(current, calibrations) == config
            assert calibrations.observe(current)[7] == original
            assert calibrations.get_factor(current, 7) == 10.0
            assert calibrations.get_record(current, 7)["status"] == "valid"
            assert calibrations.get_factor(current, 1) is None
            assert calibrations.get_record(current, 1)["status"] == "invalidated"
            assert calibrations.get_factor(current, 0) is None
            assert calibrations.path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"input_mode": "VOLTAGE"}, {"unit": "g"},
    {"range_min": -5.0}, {"range_max": 5.0},
])
@pytest.mark.parametrize("saved_rate", [51200, 102400])
@pytest.mark.parametrize("action", ["load", "set_sample_rate"])
def test_invalidation_is_sticky_but_reset_is_channel_local(tmp_path, change, saved_rate, action):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    other = device_info(machine_id="other")
    for current in (device, other):
        profiles.set_sample_rate(current, 48000, calibrations)
        for channel in (7, 1):
            save_measurement(calibrations, current, channel)
    profile_bytes = profiles.path.read_bytes()
    original = read_json(calibrations.path)
    changed = read_json(profiles.path)
    changed["devices"][device["machine_id"]].update(change, sample_rate=saved_rate)
    write_json(profiles.path, changed)

    # The public load path must observe *all* this device's records before
    # rejecting unsupported acquisition, even with only channel 7 selected.
    selection = {**device, "physical_channels": [7]}
    with pytest.raises(ValueError, match="sample_rate|input_mode|unit|range_"):
        if action == "load":
            profiles.load(selection, calibrations)
        else:
            profiles.set_sample_rate(selection, 44100, calibrations)
    invalidated = deepcopy(original)
    for record in invalidated["devices"][device["machine_id"]]["channels"].values():
        record["status"] = "invalidated"
    assert read_json(calibrations.path) == invalidated
    invalidated_bytes = calibrations.path.read_bytes()

    profiles.path.write_bytes(profile_bytes)
    profiles, calibrations = make_stores(tmp_path)
    assert profiles.load(device, calibrations) == input_config(48000)
    assert calibrations.get_factor(device, 7) is None
    assert calibrations.get_factor(device, 1) is None
    assert calibrations.get_factor(other, 7) == 10.0
    assert calibrations.path.read_bytes() == invalidated_bytes

    assert calibrations.reset(device, 7) is True
    assert calibrations.get_record(device, 7) is None
    assert calibrations.get_factor(device, 7) is None
    invalidated["devices"][device["machine_id"]]["channels"].pop("7")
    assert read_json(calibrations.path) == invalidated
    reset_bytes = calibrations.path.read_bytes()
    assert calibrations.reset(device, 7) is False
    assert calibrations.path.read_bytes() == reset_bytes
    replacement = save_measurement(
        calibrations, device, 1, v2pa_factor=2.5, standard_spl=114.0,
        calibration_sample_rate=44100, calibrated_at="2026-08-28T12:00:00+08:00",
    )
    assert replacement["status"] == "valid"
    assert calibrations.get_factor(device, 1) == 2.5
    invalidated["devices"][device["machine_id"]]["channels"]["1"] = replacement
    assert read_json(calibrations.path) == invalidated
    assert profiles.path.read_bytes() == profile_bytes


def test_routing_name_channel_selection_and_normalized_identity_do_not_invalidate(tmp_path, monkeypatch):
    _, calibrations = make_stores(tmp_path)
    device = device_info()
    original = save_measurement(calibrations, device, 7)
    before = calibrations.path.read_bytes()

    def no_write(*args):
        pytest.fail("routing/selection changes cannot write calibration")

    monkeypatch.setattr(stores.os, "replace", no_write)
    for channels in ([1], [1, 7], [7], [0, 7, 1]):
        renamed = device_info(
            model=" ve3668n ", machine_id=" test-machine-1 ", name="Dev42",
            address=None, available=False, physical_channels=channels,
        )
        assert calibrations.get_record(renamed, 7) == original
    assert calibrations.path.read_bytes() == before


@pytest.mark.parametrize("invalid_rate", [
    None, True, False, 44100.0, "48000", 1, 7999, 102401,
])
def test_pure_rate_errors_never_invalidate_and_explicit_selection_repairs_saved_rate(tmp_path, invalid_rate):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    profiles.set_sample_rate(device, 51200, calibrations)
    save_measurement(calibrations, device, 7)
    original_calibration = calibrations.path.read_bytes()
    original_profile = profiles.path.read_bytes()
    with pytest.raises(ValueError, match="sample_rate"):
        profiles.set_sample_rate(device, invalid_rate, calibrations)
    assert profiles.path.read_bytes() == original_profile

    saved = read_json(profiles.path)
    saved["devices"][device["machine_id"]]["sample_rate"] = invalid_rate
    write_json(profiles.path, saved)
    broken_profile = profiles.path.read_bytes()
    with pytest.raises(ValueError, match="sample_rate"):
        profiles.load(device, calibrations)
    with pytest.raises(ValueError, match="sample_rate"):
        calibrations.get_factor(device_info(input_config=input_config(invalid_rate)), 7)
    assert calibrations.path.read_bytes() == original_calibration
    assert profiles.path.read_bytes() == broken_profile
    assert profiles.set_sample_rate(device, 44100, calibrations) == input_config(44100)
    assert profiles.load(device, calibrations) == input_config(44100)
    assert calibrations.path.read_bytes() == original_calibration


def document(kind):
    device = device_info()
    data = input_config() if kind == "profile" else {"channels": {"7": measured_record(device, 7)}}
    return {"schema_version": 1, "devices": {device["machine_id"]: data}}


def operations(kind, profiles, calibrations):
    device = device_info()
    if kind == "profile":
        return (
            lambda: profiles.load(device, calibrations),
            lambda: profiles.set_sample_rate(device, 48000, calibrations),
        )
    return (
        lambda: calibrations.get_record(device, 7),
        lambda: calibrations.get_factor(device, 7),
        lambda: save_measurement(calibrations, device, 7),
        lambda: calibrations.reset(device, 7),
    )


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("bad_root", [
    None, [], {}, {"devices": {}}, {"schema_version": 1},
    {"schema_version": 2, "devices": {}},
    {"schema_version": True, "devices": {}},
    {"schema_version": 1.0, "devices": {}},
    {"schema_version": "1", "devices": {}},
    {"schema_version": 1, "devices": []},
    {"schema_version": 1, "devices": None},
    {"schema_version": 1, "devices": {}, "sensitivity": 1000},
    {"schema_version": 1, "devices": {}, "unknown": 0},
])
def test_strict_root_is_rejected_without_defaulting_or_overwriting(tmp_path, kind, bad_root):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    write_json(target.path, bad_root)
    original = target.path.read_bytes()
    for operation in operations(kind, profiles, calibrations):
        with pytest.raises(ValueError, match="VE store") as error:
            operation()
        assert str(target.path) in str(error.value)
        assert target.path.read_bytes() == original


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("raw", [
    b"{", b"", b"\xff", b"[] trailing",
    b'{"schema_version":1,"schema_version":1,"devices":{}}',
    b'{"schema_version":1,"devices":{"duplicate":{},"duplicate":{}}}',
    pytest.param(b"[" * 1100 + b"]" * 1100, id="excessive-depth"),
])
def test_corrupt_json_has_path_diagnostic_and_is_preserved(tmp_path, kind, raw):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(raw)
    for operation in operations(kind, profiles, calibrations):
        with pytest.raises(ValueError, match="VE store") as error:
            operation()
        assert str(target.path) in str(error.value)
        assert target.path.read_bytes() == raw


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("mutation", ["empty-id", "duplicate-id", "unknown", "sensitivity", "missing", "not-object"])
def test_strict_device_and_profile_shapes_are_not_repaired(tmp_path, kind, mutation):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    payload = document(kind)
    devices = payload["devices"]
    entry = devices["test-machine-1"]
    if mutation == "empty-id":
        devices[" "] = devices.pop("test-machine-1")
    elif mutation == "duplicate-id":
        devices[" test-machine-1 "] = deepcopy(entry)
    elif mutation in ("unknown", "sensitivity"):
        entry[mutation] = 1000
    elif mutation == "missing":
        entry.pop("sample_rate" if kind == "profile" else "channels")
    else:
        devices["test-machine-1"] = []
    write_json(target.path, payload)
    original = target.path.read_bytes()
    for operation in operations(kind, profiles, calibrations):
        with pytest.raises(ValueError, match="VE store"):
            operation()
        assert target.path.read_bytes() == original


@pytest.mark.parametrize("level", ["record", "fingerprint"])
@pytest.mark.parametrize("mutation", ["unknown", "sensitivity", "sample_rate", "missing", "not-object"])
def test_record_and_fingerprint_are_closed_schemas(tmp_path, level, mutation):
    profiles, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    channels = payload["devices"]["test-machine-1"]["channels"]
    record = channels["7"]
    target = record if level == "record" else record["fingerprint"]
    if mutation == "missing":
        target.pop("calibrated_at" if level == "record" else "unit")
    elif mutation == "not-object":
        if level == "record":
            channels["7"] = []
        else:
            record["fingerprint"] = []
    else:
        target[mutation] = 1000
    write_json(calibrations.path, payload)
    original = calibrations.path.read_bytes()
    for operation in operations("calibration", profiles, calibrations):
        with pytest.raises(stores.VEStoreFormatError):
            operation()
        assert calibrations.path.read_bytes() == original


@pytest.mark.parametrize("field,bad_value", [
    ("v2pa_factor", value)
    for value in (None, True, "10", 0, -1, float("nan"), float("inf"), -float("inf"), 10 ** 400)
] + [
    ("standard_spl", value) for value in (None, True, "94", 0, 95, float("nan"), float("inf"))
] + [
    ("calibration_sample_rate", value) for value in (None, True, "44100", 44100.0, 0, -1, 102401)
] + [
    ("calibration_duration_seconds", value) for value in (None, True, "10", 0, -1, float("nan"), float("inf"))
] + [
    ("calibrated_at", value) for value in (
        None, 123, "", "yesterday", "2026-08-28", "2026-08-28T10:00:00", "2026-02-30T10:00:00Z",
    )
])
def test_provenance_is_strict_on_load_and_failed_recalibration_preserves_valid_record(tmp_path, field, bad_value):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    original_record = save_measurement(calibrations, device, 7)
    original_bytes = calibrations.path.read_bytes()
    with pytest.raises(ValueError, match=field):
        save_measurement(calibrations, device, 7, **{field: bad_value})
    assert calibrations.path.read_bytes() == original_bytes
    assert calibrations.get_record(device, 7) == original_record

    # Invalidated records must retain *valid provenance*, not bypass validation.
    for status in ("valid", "invalidated"):
        payload = document("calibration")
        record = payload["devices"]["test-machine-1"]["channels"]["7"]
        record.update({field: bad_value, "status": status})
        write_json(calibrations.path, payload)
        corrupted_bytes = calibrations.path.read_bytes()
        for operation in operations("calibration", profiles, calibrations):
            with pytest.raises(stores.VEStoreFormatError, match=f"{field}|non-finite JSON"):
                operation()
            assert calibrations.path.read_bytes() == corrupted_bytes


@pytest.mark.parametrize("status", [None, True, "none", "measured", "VALID", "", 1])
def test_record_status_has_only_two_stored_values(tmp_path, status):
    _, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    payload["devices"]["test-machine-1"]["channels"]["7"]["status"] = status
    write_json(calibrations.path, payload)
    with pytest.raises(stores.VEStoreFormatError, match="status"):
        calibrations.get_factor(device_info(), 7)


@pytest.mark.parametrize("channel_key", ["", "07", "7.0", "-1", "8", "true", "AIN8", "51200", "٧"])
def test_calibration_channel_keys_are_physical_indices_not_rates(tmp_path, channel_key):
    _, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    channels = payload["devices"]["test-machine-1"]["channels"]
    channels[channel_key] = channels.pop("7")
    write_json(calibrations.path, payload)
    with pytest.raises(stores.VEStoreFormatError, match="channel"):
        calibrations.observe(device_info())


@pytest.mark.parametrize("field,bad_value", [
    ("backend", "sounddevice"), ("model", "VE3668N-other"),
    ("machine_id", "other"), ("machine_id", ""),
    ("physical_channel", 1), ("physical_channel", True), ("physical_channel", 7.0),
    ("input_mode", None), ("input_mode", " "), ("unit", False), ("unit", ""),
    ("range_min", True), ("range_min", "-10"), ("range_min", float("nan")),
    ("range_min", 10), ("range_max", -10), ("range_max", float("inf")),
])
def test_fingerprint_identity_channel_and_conditions_are_checked(tmp_path, field, bad_value):
    _, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    payload["devices"]["test-machine-1"]["channels"]["7"]["fingerprint"][field] = bad_value
    write_json(calibrations.path, payload)
    original = calibrations.path.read_bytes()
    with pytest.raises(stores.VEStoreFormatError):
        calibrations.get_factor(device_info(), 7)
    assert calibrations.path.read_bytes() == original


@pytest.mark.parametrize("kind", ["profile", "calibration"])
def test_atomic_write_flushes_fsyncs_and_closes_owned_same_directory_temp_before_replace(tmp_path, monkeypatch, kind):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    events = []
    opened = []
    mkstemp, fdopen = stores.tempfile.mkstemp, stores.os.fdopen
    fsync, replace = stores.os.fsync, stores.os.replace

    def track_temp(*args, **kwargs):
        descriptor, name = mkstemp(*args, **kwargs)
        assert Path(name).parent == target.path.parent
        assert Path(name) != target.path
        opened.append({"name": Path(name), "descriptor": descriptor})
        return descriptor, name

    class TrackedFile:
        def __init__(self, stream):
            self.stream = stream

        def write(self, value):
            return self.stream.write(value)

        def fileno(self):
            return self.stream.fileno()

        def flush(self):
            events.append("flush")
            self.stream.flush()

        def close(self):
            events.append("close")
            self.stream.close()

    def track_open(descriptor, *args, **kwargs):
        stream = fdopen(descriptor, *args, **kwargs)
        opened[-1]["stream"] = stream
        return TrackedFile(stream)

    def track_fsync(descriptor):
        assert events == ["flush"]
        assert descriptor == opened[-1]["descriptor"]
        assert read_json(opened[-1]["name"])["schema_version"] == 1
        events.append("fsync")
        return fsync(descriptor)

    def track_replace(source, destination):
        assert events == ["flush", "fsync", "close"]
        assert opened[-1]["stream"].closed
        assert Path(source) == opened[-1]["name"]
        assert Path(destination) == target.path
        events.append("replace")
        return replace(source, destination)

    monkeypatch.setattr(stores.tempfile, "mkstemp", track_temp)
    monkeypatch.setattr(stores.os, "fdopen", track_open)
    monkeypatch.setattr(stores.os, "fsync", track_fsync)
    monkeypatch.setattr(stores.os, "replace", track_replace)
    operation = operations(kind, profiles, calibrations)[1 if kind == "profile" else 2]
    operation()
    assert events == ["flush", "fsync", "close", "replace"]
    assert set(target.path.parent.iterdir()) == {target.path}


@pytest.mark.parametrize("failure", ["fsync", "replace"])
@pytest.mark.parametrize("operation_name", ["profile", "recalibrate", "reset", "invalidate"])
def test_atomic_failure_preserves_previous_records_and_retries(tmp_path, monkeypatch, failure, operation_name):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    other = device_info(machine_id="other")
    profiles.set_sample_rate(device, 51200, calibrations)
    held_record = save_measurement(calibrations, device, 7)
    save_measurement(calibrations, device, 1)
    save_measurement(calibrations, other, 7)
    held_copy = deepcopy(held_record)
    old_profile, old_calibration = profiles.path.read_bytes(), calibrations.path.read_bytes()
    target = profiles if operation_name == "profile" else calibrations
    changed_device = device_info(input_config=input_config(unit="g"))
    actions = {
        "profile": lambda: profiles.set_sample_rate(device, 44100, calibrations),
        "recalibrate": lambda: save_measurement(calibrations, device, 7, v2pa_factor=6.25),
        "reset": lambda: calibrations.reset(device, 7),
        "invalidate": lambda: calibrations.get_factor(changed_device, 7),
    }

    def fail(*args):
        raise OSError(f"injected {failure} failure")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, failure, fail)
        with pytest.raises(OSError, match="VE store") as error:
            actions[operation_name]()
        assert str(target.path) in str(error.value)
        assert isinstance(error.value.__cause__, OSError)
        assert profiles.path.read_bytes() == old_profile
        assert calibrations.path.read_bytes() == old_calibration
        assert held_record == held_copy
        assert set(target.path.parent.iterdir()) == {profiles.path, calibrations.path}
    if operation_name == "invalidate":
        with pytest.raises(ValueError, match="unit"):
            actions[operation_name]()
        assert calibrations.get_factor(device, 7) is None
        assert calibrations.get_factor(device, 1) is None
    else:
        actions[operation_name]()
        assert calibrations.get_factor(device, 1) == 10.0
        if operation_name == "profile":
            assert profiles.load(device, calibrations) == input_config(44100)
            assert calibrations.path.read_bytes() == old_calibration
        elif operation_name == "recalibrate":
            assert calibrations.get_factor(device, 7) == 6.25
        else:
            assert calibrations.get_factor(device, 7) is None
    assert calibrations.get_factor(other, 7) == 10.0


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_first_write_leaves_no_file_or_temporary_and_can_retry(tmp_path, monkeypatch, kind, failure):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    operation = operations(kind, profiles, calibrations)[1 if kind == "profile" else 2]

    def fail(*args):
        raise OSError(f"injected {failure}")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, failure, fail)
        with pytest.raises(OSError, match="VE store"):
            operation()
    assert not target.path.exists()
    assert list(target.path.parent.iterdir()) == []
    operation()
    assert target.path.is_file()


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("boundary", ["open", "read", "missing-during-read"])
def test_read_failure_is_not_a_missing_file_or_a_default_and_can_retry(tmp_path, monkeypatch, kind, boundary):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    write_json(target.path, document(kind))
    original = target.path.read_bytes()
    open_path = Path.open

    def fail_open(path, *args, **kwargs):
        if path == target.path:
            raise PermissionError("injected denied read")
        return open_path(path, *args, **kwargs)

    def fail_read(*args, **kwargs):
        if boundary == "missing-during-read":
            raise FileNotFoundError("already opened; not a new store")
        raise OSError("injected read error")

    with monkeypatch.context() as patch:
        if boundary == "open":
            patch.setattr(Path, "open", fail_open)
        else:
            patch.setattr(stores.json, "load", fail_read)
        for operation in operations(kind, profiles, calibrations):
            with pytest.raises(OSError, match="VE store") as error:
                operation()
            assert str(target.path) in str(error.value)
    assert target.path.read_bytes() == original
    assert operations(kind, profiles, calibrations)[0]() is not None


def test_failed_invalidation_stays_fail_closed_when_external_profile_is_restored(tmp_path, monkeypatch):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    profiles.set_sample_rate(device, 48000, calibrations)
    held = save_measurement(calibrations, device, 7)
    save_measurement(calibrations, device, 1)
    original_profile = profiles.path.read_bytes()
    original_calibration = calibrations.path.read_bytes()
    external = read_json(profiles.path)
    external["devices"][device["machine_id"]]["unit"] = "g"
    write_json(profiles.path, external)

    def fail(*args):
        raise OSError("invalidation could not be persisted")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, "replace", fail)
        with pytest.raises(stores.VEStoreIOError):
            profiles.load(device, calibrations)
        profiles.path.write_bytes(original_profile)
        for operation in (
            lambda: profiles.load(device, calibrations),
            lambda: calibrations.get_factor(device, 7),
            lambda: calibrations.get_factor(device, 1),
        ):
            with pytest.raises(stores.VEStoreIOError):
                operation()
        assert calibrations.path.read_bytes() == original_calibration
        assert held == measured_record(device, 7)

    # Retrying the same instance must persist the observed mismatch, even
    # though the supported original conditions have since been restored.
    assert profiles.load(device, calibrations) == input_config(48000)
    profiles, calibrations = make_stores(tmp_path)
    assert calibrations.get_factor(device, 7) is None
    assert calibrations.get_factor(device, 1) is None
    assert calibrations.get_record(device, 7) == {**held, "status": "invalidated"}


@pytest.mark.parametrize("action", ["reset", "recalibrate"])
@pytest.mark.parametrize("observed_machine_ids", [
    ("test-machine-1",), ("other",), ("test-machine-1", "other"), (),
])
def test_successful_channel_recovery_persists_pending_invalidations_before_restart(
    tmp_path, monkeypatch, action, observed_machine_ids,
):
    _, calibrations = make_stores(tmp_path)
    device = device_info()
    devices = (device, device_info(machine_id="other"))
    for current in devices:
        for channel in (7, 1):
            save_measurement(calibrations, current, channel)
    original_bytes = calibrations.path.read_bytes()
    expected = read_json(calibrations.path)

    def fail(*args):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, "replace", fail)
        for current in devices:
            if current["machine_id"] in observed_machine_ids:
                with pytest.raises(stores.VEStoreIOError):
                    calibrations.observe({**current, "input_config": input_config(unit="g")})
                for record in expected["devices"][current["machine_id"]]["channels"].values():
                    record["status"] = "invalidated"
        assert calibrations.path.read_bytes() == original_bytes

    if action == "reset":
        assert calibrations.reset(device, 7) is True
        expected["devices"][device["machine_id"]]["channels"].pop("7")
    else:
        replacement = save_measurement(
            calibrations, device, 7, v2pa_factor=2.5, standard_spl=114.0,
            calibration_sample_rate=44100, calibrated_at="2026-08-28T12:00:00+08:00",
        )
        expected["devices"][device["machine_id"]]["channels"]["7"] = replacement

    # Recreate before any lookup: the previous owner's retry latch must not
    # conceal a recovery write that left sibling/other-device records valid.
    _, calibrations = make_stores(tmp_path)
    for current in devices:
        for channel in (1, 7):
            record = expected["devices"][current["machine_id"]]["channels"].get(str(channel))
            factor = record["v2pa_factor"] if record and record["status"] == "valid" else None
            assert calibrations.get_factor(current, channel) == factor
            assert calibrations.get_record(current, channel) == record
    assert read_json(calibrations.path) == expected


@pytest.mark.parametrize("action", ["reset", "recalibrate"])
@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_channel_recovery_retains_pending_invalidations_for_retry(
    tmp_path, monkeypatch, action, failure,
):
    _, calibrations = make_stores(tmp_path)
    device = device_info()
    held = save_measurement(calibrations, device, 7)
    save_measurement(calibrations, device, 1)
    original_bytes = calibrations.path.read_bytes()
    expected = read_json(calibrations.path)

    def fail(*args):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, "replace", fail)
        with pytest.raises(stores.VEStoreIOError):
            calibrations.observe(device_info(input_config=input_config(unit="g")))

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, failure, fail)
        with pytest.raises(stores.VEStoreIOError):
            if action == "reset":
                calibrations.reset(device, 7)
            else:
                save_measurement(calibrations, device, 7, v2pa_factor=2.5)
        assert calibrations.path.read_bytes() == original_bytes
        assert held == measured_record(device, 7)
        assert set(calibrations.path.parent.iterdir()) == {calibrations.path}

    # A failed recovery cannot consume the latch or publish the replacement.
    # Retry observation with the original conditions and old measurements.
    calibrations.observe(device)
    _, calibrations = make_stores(tmp_path)
    for channel, record in expected["devices"][device["machine_id"]]["channels"].items():
        record["status"] = "invalidated"
        assert calibrations.get_factor(device, int(channel)) is None
        assert calibrations.get_record(device, int(channel)) == record
    assert read_json(calibrations.path) == expected


def test_successful_observation_persists_other_devices_pending_invalidations_before_restart(
    tmp_path, monkeypatch,
):
    _, calibrations = make_stores(tmp_path)
    devices = (device_info(), device_info(machine_id="other"))
    for current in devices:
        for channel in (7, 1):
            save_measurement(calibrations, current, channel)
    original_bytes = calibrations.path.read_bytes()
    expected = read_json(calibrations.path)

    def fail(*args):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, "replace", fail)
        with pytest.raises(stores.VEStoreIOError):
            calibrations.observe({**devices[1], "input_config": input_config(unit="g")})
        assert calibrations.path.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="unit"):
        calibrations.observe({**devices[0], "input_config": input_config(unit="g")})
    _, calibrations = make_stores(tmp_path)
    for current in reversed(devices):
        for channel, record in expected["devices"][current["machine_id"]]["channels"].items():
            record["status"] = "invalidated"
            assert calibrations.get_factor(current, int(channel)) is None
            assert calibrations.get_record(current, int(channel)) == record
    assert read_json(calibrations.path) == expected


def test_default_paths_are_separate_and_old_soundcard_registry_is_untouched(tmp_path, monkeypatch):
    from base import soundcard_calibration_manager as soundcard
    from consts import model_consts

    legacy_path = tmp_path / "ui/ui_config/mic_input_calibration.json"
    write_json(legacy_path, {"version": 2, "devices": []})
    old_bytes = legacy_path.read_bytes()
    monkeypatch.setattr(model_consts, "DEFAULT_DIR", str(tmp_path))
    monkeypatch.setattr(soundcard, "MIC_INPUT_CALIBRATION_PATH", str(legacy_path))

    def no_legacy_writes(*args):
        pytest.fail("VE stores must not use the old soundcard registry")

    monkeypatch.setattr(soundcard, "_atomic_write_json", no_legacy_writes)
    profiles, calibrations = stores.VEInputProfileStore(), stores.VECalibrationStore()
    assert profiles.path == tmp_path / "configs/ve3668n_input_profiles.json"
    assert calibrations.path == tmp_path / "configs/ve3668n_input_calibration.json"
    assert not profiles.path.exists()
    assert not calibrations.path.exists()
    device = device_info()
    assert profiles.load(device, calibrations) == input_config(51200)
    assert calibrations.get_record(device, 7) is None
    assert calibrations.get_factor(device, 7) is None
    assert calibrations.reset(device, 7) is False
    assert not calibrations.path.exists()
    profiles.set_sample_rate(device, 44100, calibrations)
    save_measurement(calibrations, device, 7)
    calibrations.reset(device, 7)
    assert read_json(profiles.path) == {
        "schema_version": 1, "devices": {"test-machine-1": input_config(44100)},
    }
    assert read_json(calibrations.path) == {"schema_version": 1, "devices": {}}
    assert legacy_path.read_bytes() == old_bytes
    assert soundcard.load_mic_input_calibration() == {"version": 2, "devices": []}


def test_frozen_input_and_return_values_do_not_share_mutable_state(tmp_path):
    from base.recording_process_protocol import FrozenConfig

    profiles, calibrations = make_stores(tmp_path)
    other_profiles, other_calibrations = make_stores(tmp_path / "independent")
    device = device_info()
    frozen = FrozenConfig.snapshot(device)
    config = profiles.set_sample_rate(frozen, 44100, calibrations)
    config["sample_rate"] = 48000
    result = save_measurement(calibrations, frozen, 7)
    result["fingerprint"]["unit"] = "g"
    result["status"] = "invalidated"
    result["v2pa_factor"] = 100
    returned = calibrations.observe(frozen)
    returned[7]["fingerprint"]["machine_id"] = "wrong"
    returned.clear()
    assert profiles.load(frozen, calibrations) == input_config(44100)
    assert calibrations.get_record(frozen, 7) == measured_record(device, 7)
    assert other_profiles.load(frozen, other_calibrations) == input_config()
    assert other_calibrations.get_factor(frozen, 7) is None
    assert profiles._lock is not other_profiles._lock
    assert calibrations._lock is not other_calibrations._lock
    assert frozen == FrozenConfig.snapshot(device)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_rate_literals_are_corruption_not_repairable_saved_rates(tmp_path, constant):
    profiles, calibrations = make_stores(tmp_path)
    payload = json.dumps(document("profile")).replace('"sample_rate": 51200', f'"sample_rate": {constant}')
    profiles.path.parent.mkdir(parents=True)
    profiles.path.write_text(payload, encoding="utf-8")
    before = profiles.path.read_bytes()
    for operation in operations("profile", profiles, calibrations):
        with pytest.raises(stores.VEStoreFormatError, match="non-finite JSON"):
            operation()
        assert profiles.path.read_bytes() == before


@pytest.mark.parametrize("bad_device", [None, [], {}, {"input_config": input_config()}])
def test_public_device_boundaries_report_validation_errors_without_writes(tmp_path, bad_device):
    profiles, calibrations = make_stores(tmp_path)
    for operation in (
        lambda: profiles.load(bad_device, calibrations),
        lambda: profiles.set_sample_rate(bad_device, 48000, calibrations),
        lambda: calibrations.observe(bad_device),
        lambda: calibrations.get_factor(bad_device, 7),
        lambda: calibrations.reset(bad_device, 7),
    ):
        with pytest.raises(ValueError, match="device"):
            operation()
    assert not profiles.path.exists()
    assert not calibrations.path.exists()


@pytest.mark.parametrize("kind", ["profile", "calibration"])
@pytest.mark.parametrize("boundary", ["mkdir", "mkstemp", "fdopen", "serialize", "flush", "close"])
def test_other_write_failures_preserve_bytes_and_clean_only_owned_temp(tmp_path, monkeypatch, kind, boundary):
    profiles, calibrations = make_stores(tmp_path)
    target = profiles if kind == "profile" else calibrations
    write_json(target.path, document(kind))
    original = target.path.read_bytes()
    unrelated = target.path.parent / "unrelated.tmp"
    unrelated.write_bytes(b"do not delete")
    fdopen = stores.os.fdopen
    opened = []

    def fail(*args, **kwargs):
        raise OSError(f"injected {boundary} failure")

    class FailingFile:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def flush(self):
            if boundary == "flush":
                fail()
            self.stream.flush()

        def close(self):
            was_closed = self.stream.closed
            self.stream.close()
            if boundary == "close" and not was_closed:
                fail()

    def wrap_open(descriptor, *args, **kwargs):
        stream = fdopen(descriptor, *args, **kwargs)
        opened.append(stream)
        return FailingFile(stream)

    with monkeypatch.context() as patch:
        if boundary in ("flush", "close"):
            patch.setattr(stores.os, "fdopen", wrap_open)
        else:
            owner, name = {
                "mkdir": (Path, "mkdir"), "mkstemp": (stores.tempfile, "mkstemp"),
                "fdopen": (stores.os, "fdopen"), "serialize": (stores.json, "dump"),
            }[boundary]
            patch.setattr(owner, name, fail)
        operation = operations(kind, profiles, calibrations)[1 if kind == "profile" else 2]
        with pytest.raises(stores.VEStoreIOError, match="update VE store") as error:
            operation()
        assert boundary in str(error.value.__cause__)
    assert all(stream.closed for stream in opened)
    assert target.path.read_bytes() == original
    assert unrelated.read_bytes() == b"do not delete"
    assert set(target.path.parent.iterdir()) == {target.path, unrelated}


def test_cleanup_failure_is_diagnostic_and_preserves_primary_persistence_error(tmp_path, monkeypatch):
    _, calibrations = make_stores(tmp_path)
    save_measurement(calibrations, device_info(), 7)
    original = calibrations.path.read_bytes()

    def fail_replace(*args):
        raise OSError("primary replace failure")

    def fail_cleanup(*args):
        raise PermissionError("secondary cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(stores.os, "replace", fail_replace)
        patch.setattr(stores.os, "unlink", fail_cleanup)
        with pytest.raises(stores.VEStoreIOError) as error:
            calibrations.reset(device_info(), 7)
    assert str(error.value.__cause__) == "primary replace failure"
    assert any("secondary cleanup failure" in note for note in error.value.__notes__)
    assert calibrations.path.read_bytes() == original
    leftovers = [path for path in calibrations.path.parent.iterdir() if path != calibrations.path]
    assert len(leftovers) == 1
    assert any(str(leftovers[0]) in note for note in error.value.__notes__)
    leftovers[0].unlink()


@pytest.mark.parametrize("level,field", [
    ("record", key) for key in measured_record(device_info(), 7)
] + [
    ("fingerprint", key) for key in measured_record(device_info(), 7)["fingerprint"]
])
def test_every_record_and_fingerprint_field_is_required(tmp_path, level, field):
    _, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    record = payload["devices"]["test-machine-1"]["channels"]["7"]
    (record if level == "record" else record["fingerprint"]).pop(field)
    write_json(calibrations.path, payload)
    with pytest.raises(stores.VEStoreFormatError):
        calibrations.observe(device_info())


def test_saved_normalized_identity_is_compared_without_rewriting_calibration(tmp_path):
    _, calibrations = make_stores(tmp_path)
    payload = document("calibration")
    entry = payload["devices"].pop("test-machine-1")
    payload["devices"][" test-machine-1 "] = entry
    entry["channels"]["7"]["fingerprint"].update(
        model=" ve3668n ", machine_id=" test-machine-1 ", input_mode=" IEPE ", unit=" V ",
    )
    write_json(calibrations.path, payload)
    original = calibrations.path.read_bytes()
    assert calibrations.get_record(device_info(), 7) == measured_record(device_info(), 7)
    assert calibrations.path.read_bytes() == original


def test_no_calibration_file_is_created_for_rate_changes_and_recalibration_never_buckets_by_rate(tmp_path):
    profiles, calibrations = make_stores(tmp_path)
    device = device_info()
    for rate in (44100, 48000, 51200):
        profiles.set_sample_rate(device, rate, calibrations)
        assert calibrations.get_factor(device, 7) is None
        assert not calibrations.path.exists()
    original_profile = profiles.path.read_bytes()
    for rate, factor in ((44100, 2.0), (48000, 3.0), (51200, 4.0)):
        result = save_measurement(calibrations, device, 7, calibration_sample_rate=rate, v2pa_factor=factor)
        assert result == measured_record(device, 7, factor=factor, rate=rate)
        assert read_json(calibrations.path) == {
            "schema_version": 1, "devices": {"test-machine-1": {"channels": {"7": result}}},
        }
        assert profiles.path.read_bytes() == original_profile
