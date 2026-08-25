import inspect
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import sounddevice as sd

import base.soundcard_calibration_manager as manager
from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
    build_mic_input_identity,
    clear_mic_channel_calibrations,
    get_mic_v2pa_factor,
    load_mic_channel_calibrations,
    load_mic_channel_v2pa_factors,
    load_mic_input_calibration,
    resolve_mic_channel_v2pa_factor,
    save_mic_channel_calibration,
)


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 4,
}
OTHER_DEVICE = {**DEVICE, "index": 8, "name": "Other Microphone"}


@pytest.fixture(autouse=True)
def host_api_name():
    with mock.patch(
        "base.soundcard_calibration_manager.SoundDeviceManager.get_api_info",
        return_value={"name": "Test API"},
    ):
        yield


def save_channel(path, device=DEVICE, channel=1, factor=2.5, spl=94.0, **overrides):
    kwargs = {
        "v2pa_factor": factor,
        "input_device": device,
        "input_channel": channel,
        "standard_spl_db": spl,
        "sample_rate_hz": 44100,
        "duration_seconds": 10.0,
        "calibration_path": str(path),
        "calibrated_at": "2026-08-24T10:00:00+08:00",
    }
    kwargs.update(overrides)
    return save_mic_channel_calibration(**kwargs)


def valid_registry():
    return {
        "version": 2,
        "devices": [{
            "input": {"api_name": "Test API", "device_name": "Test Microphone"},
            "channels": {"1": {
                "v2pa_factor": 2.5,
                "standard_spl_db": 94.0,
                "sample_rate_hz": 44100,
                "duration_seconds": 10.0,
                "calibrated_at": "2026-08-24T10:00:00+08:00",
            }},
        }],
    }


def write_registry(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def beyond_digit_limit_channel():
    configured_limit = sys.get_int_max_str_digits()
    digit_count = max(configured_limit + 100, 5000)
    return 10 ** digit_count, "1" + ("0" * digit_count)


def test_v2_registry_preserves_devices_and_channels(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=0, factor=1.25)
    save_channel(path, channel=2, factor=2.5, spl=114.0)
    save_channel(path, device=OTHER_DEVICE, channel=1, factor=3.0)

    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {0: 1.25, 2: 2.5}
    assert load_mic_channel_v2pa_factors(OTHER_DEVICE, str(path)) == {1: 3.0}
    assert resolve_mic_channel_v2pa_factor(DEVICE, 1, str(path)) is None


def test_load_supports_channel_key_beyond_interpreter_digit_limit(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    channel, channel_key = beyond_digit_limit_channel()
    payload = valid_registry()
    payload["devices"][0]["channels"] = {
        channel_key: payload["devices"][0]["channels"]["1"]
    }
    write_registry(path, payload)

    records = load_mic_channel_calibrations(DEVICE, str(path))

    assert list(records) == [channel]
    assert records[channel]["v2pa_factor"] == 2.5


def test_resolve_supports_channel_beyond_interpreter_digit_limit(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    channel, channel_key = beyond_digit_limit_channel()
    payload = valid_registry()
    payload["devices"][0]["channels"] = {
        channel_key: payload["devices"][0]["channels"]["1"]
    }
    write_registry(path, payload)

    assert resolve_mic_channel_v2pa_factor(DEVICE, channel, str(path)) == 2.5


def test_save_supports_channel_beyond_interpreter_digit_limit(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    channel, channel_key = beyond_digit_limit_channel()

    save_channel(path, channel=channel, factor=3.5)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert list(persisted["devices"][0]["channels"]) == [channel_key]
    assert resolve_mic_channel_v2pa_factor(DEVICE, channel, str(path)) == 3.5


def test_clear_supports_channel_beyond_interpreter_digit_limit(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    channel, channel_key = beyond_digit_limit_channel()
    payload = valid_registry()
    payload["devices"][0]["channels"] = {
        channel_key: payload["devices"][0]["channels"]["1"]
    }
    write_registry(path, payload)

    assert clear_mic_channel_calibrations(DEVICE, [channel], str(path)) is True
    assert load_mic_input_calibration(str(path)) == {"version": 2, "devices": []}


def test_updating_channel_preserves_every_other_record(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=0, factor=1.25)
    save_channel(path, channel=2, factor=2.5)
    save_channel(path, device=OTHER_DEVICE, channel=1, factor=3.0)

    save_channel(path, channel=0, factor=4.0, spl=114.0)

    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {0: 4.0, 2: 2.5}
    assert load_mic_channel_v2pa_factors(OTHER_DEVICE, str(path)) == {1: 3.0}


def test_device_index_change_does_not_change_stable_identity(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=1)

    assert load_mic_channel_v2pa_factors(
        {**DEVICE, "index": 99},
        str(path),
    ) == {1: 2.5}


def test_clear_only_selected_channels_for_current_device(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=0, factor=1.25)
    save_channel(path, channel=2, factor=2.5)
    save_channel(path, device=OTHER_DEVICE, channel=0, factor=3.0)

    changed = clear_mic_channel_calibrations(DEVICE, [2], str(path))

    assert changed is True
    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {0: 1.25}
    assert load_mic_channel_v2pa_factors(OTHER_DEVICE, str(path)) == {0: 3.0}


@pytest.mark.parametrize("initial", [None, {"version": 1}])
def test_missing_file_and_exact_v1_load_as_empty_registry(tmp_path, initial):
    path = tmp_path / "mic_input_calibration.json"
    if initial is not None:
        write_registry(path, initial)

    assert load_mic_input_calibration(str(path)) == {"version": 2, "devices": []}
    assert load_mic_channel_calibrations(DEVICE, str(path)) == {}
    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {}


def test_first_save_replaces_version_1_with_version_2(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    write_registry(path, {"version": 1, "legacy": "ignored"})

    save_channel(path)

    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2
    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {1: 2.5}


@pytest.mark.parametrize(
    "payload",
    [{}, {"devices": []}, {"version": 1.0}, {"version": 3, "devices": []},
     {"version": True, "devices": []}, {"version": 2, "devices": {}}, [], None],
)
def test_invalid_root_or_version_raises_format_error(tmp_path, payload):
    path = tmp_path / "mic_input_calibration.json"
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


def test_malformed_json_raises_format_error(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


def test_deeply_nested_json_raises_chained_format_error(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    path.write_text(("[" * 5000) + ("0") + ("]" * 5000), encoding="utf-8")

    with pytest.raises(MicCalibrationFormatError) as caught:
        load_mic_input_calibration(str(path))

    assert isinstance(caught.value.__cause__, RecursionError)


def test_integer_literal_beyond_digit_limit_raises_chained_format_error(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    _, huge_decimal = beyond_digit_limit_channel()
    serialized = json.dumps(valid_registry())
    serialized = serialized.replace(
        '"sample_rate_hz": 44100',
        f'"sample_rate_hz": {huge_decimal}',
    )
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(MicCalibrationFormatError) as caught:
        load_mic_input_calibration(str(path))

    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "field,value",
    [("api_name", ""), ("api_name", "  "), ("device_name", ""), ("device_name", "\t")],
)
def test_empty_device_identity_raises_format_error(tmp_path, field, value):
    path = tmp_path / "mic_input_calibration.json"
    payload = valid_registry()
    payload["devices"][0]["input"][field] = value
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


def test_duplicate_device_identity_raises_format_error(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    payload = valid_registry()
    payload["devices"].append(json.loads(json.dumps(payload["devices"][0])))
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


@pytest.mark.parametrize("channel_key", ["01", "+1", "-1", "True", "false", "1.0", ""])
def test_noncanonical_channel_key_raises_format_error(tmp_path, channel_key):
    path = tmp_path / "mic_input_calibration.json"
    payload = valid_registry()
    payload["devices"][0]["channels"] = {
        channel_key: payload["devices"][0]["channels"]["1"]
    }
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("v2pa_factor", True), ("v2pa_factor", 0), ("v2pa_factor", -1),
        ("v2pa_factor", float("nan")), ("v2pa_factor", float("inf")),
        ("standard_spl_db", True), ("standard_spl_db", 93), ("standard_spl_db", 115),
        ("standard_spl_db", float("nan")), ("sample_rate_hz", True),
        ("sample_rate_hz", 44100.0), ("sample_rate_hz", 0),
        ("duration_seconds", True), ("duration_seconds", 0),
        ("duration_seconds", -1), ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")), ("calibrated_at", ""),
        ("calibrated_at", "not-a-time"), ("calibrated_at", "2026-08-24T10:00:00"),
    ],
)
def test_invalid_record_field_raises_format_error(tmp_path, field, value):
    path = tmp_path / "mic_input_calibration.json"
    payload = valid_registry()
    payload["devices"][0]["channels"]["1"][field] = value
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError):
        load_mic_input_calibration(str(path))


@pytest.mark.parametrize(
    "field",
    ["v2pa_factor", "standard_spl_db", "duration_seconds"],
)
def test_oversized_persisted_numeric_raises_format_error(tmp_path, field):
    path = tmp_path / "mic_input_calibration.json"
    payload = valid_registry()
    payload["devices"][0]["channels"]["1"][field] = 10 ** 309
    write_registry(path, payload)

    with pytest.raises(MicCalibrationFormatError) as caught:
        load_mic_input_calibration(str(path))

    assert isinstance(caught.value.__cause__, OverflowError)


def test_read_oserror_is_chained_as_io_error(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    original = PermissionError("denied")
    with mock.patch("builtins.open", side_effect=original):
        with pytest.raises(MicCalibrationIOError) as caught:
            load_mic_input_calibration(str(path))
    assert caught.value.__cause__ is original


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_device": None}, {"input_channel": True}, {"input_channel": -1},
        {"input_channel": 1.0}, {"v2pa_factor": True}, {"v2pa_factor": 0},
        {"standard_spl_db": 95}, {"sample_rate_hz": 44100.0},
        {"duration_seconds": float("inf")},
        {"calibrated_at": "2026-08-24T10:00:00"},
    ],
)
def test_invalid_save_arguments_raise_before_touching_disk(tmp_path, overrides):
    path = tmp_path / "mic_input_calibration.json"
    with pytest.raises(ValueError):
        save_channel(path, **overrides)
    assert not path.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"v2pa_factor": 10 ** 309},
        {"standard_spl_db": 10 ** 309},
        {"duration_seconds": 10 ** 309},
    ],
)
def test_oversized_save_numeric_raises_value_error_without_disk_mutation(
    tmp_path,
    overrides,
):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()

    with pytest.raises(ValueError) as caught:
        save_channel(path, factor=3.0, **overrides)

    assert isinstance(caught.value.__cause__, OverflowError)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_default_timestamp_has_local_offset_seconds_and_no_microseconds(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, calibrated_at=None)

    timestamp = load_mic_channel_calibrations(DEVICE, str(path))[1]["calibrated_at"]
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.utcoffset() is not None
    assert parsed.microsecond == 0
    assert "." not in timestamp


@pytest.mark.parametrize("spl", [94, 114.0])
def test_saved_record_keeps_exact_selected_spl(tmp_path, spl):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, spl=spl)

    record = load_mic_channel_calibrations(DEVICE, str(path))[1]
    assert record["standard_spl_db"] == spl
    assert type(record["standard_spl_db"]) is type(spl)


@pytest.mark.parametrize("spl", [np.int64(94), np.float64(114.0)])
def test_accepted_numpy_scalars_are_normalized_to_json_native_values(tmp_path, spl):
    path = tmp_path / "mic_input_calibration.json"

    save_channel(
        path,
        channel=np.int64(2),
        factor=np.float64(2.5),
        spl=spl,
        sample_rate_hz=np.int64(48000),
        duration_seconds=np.float64(10.0),
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    record = persisted["devices"][0]["channels"]["2"]
    assert record == {
        "v2pa_factor": 2.5,
        "standard_spl_db": 94 if spl == 94 else 114.0,
        "sample_rate_hz": 48000,
        "duration_seconds": 10.0,
        "calibrated_at": "2026-08-24T10:00:00+08:00",
    }
    assert all(type(value) in (int, float, str) for value in record.values())


def test_atomic_write_uses_target_directory_and_fsync_before_replace(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    events = []
    real_mkstemp = manager.tempfile.mkstemp
    real_fsync = manager.os.fsync
    real_replace = manager.os.replace

    def tracked_mkstemp(*args, **kwargs):
        events.append(("mkstemp", kwargs["dir"]))
        return real_mkstemp(*args, **kwargs)

    def tracked_fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    def tracked_replace(source, destination):
        events.append(("replace", Path(source).parent, destination))
        return real_replace(source, destination)

    with mock.patch.object(manager.tempfile, "mkstemp", tracked_mkstemp), mock.patch.object(
        manager.os, "fsync", tracked_fsync
    ), mock.patch.object(manager.os, "replace", tracked_replace):
        save_channel(path)

    assert events[0] == ("mkstemp", str(tmp_path))
    assert events[1][0] == "fsync"
    assert events[2] == ("replace", tmp_path, str(path))


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "replace"])
def test_atomic_stage_failure_preserves_original_and_removes_temporary(tmp_path, stage):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()
    failure = OSError(f"{stage} failed")

    if stage in {"write", "flush"}:
        real_fdopen = manager.os.fdopen

        class FileProxy:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __enter__(self):
                self.wrapped.__enter__()
                return self

            def __exit__(self, *args):
                return self.wrapped.__exit__(*args)

            def write(self, value):
                if stage == "write":
                    raise failure
                return self.wrapped.write(value)

            def flush(self):
                if stage == "flush":
                    raise failure
                return self.wrapped.flush()

            def fileno(self):
                return self.wrapped.fileno()

        patcher = mock.patch.object(
            manager.os, "fdopen",
            side_effect=lambda *args, **kwargs: FileProxy(real_fdopen(*args, **kwargs)),
        )
    elif stage == "fsync":
        patcher = mock.patch.object(manager.os, "fsync", side_effect=failure)
    else:
        patcher = mock.patch.object(manager.os, "replace", side_effect=failure)

    with patcher:
        with pytest.raises(MicCalibrationIOError) as caught:
            save_channel(path, factor=3.0)

    assert caught.value.__cause__ is failure
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_mkstemp_failure_preserves_original_and_never_replaces(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()
    failure = PermissionError("no temp file")

    with mock.patch.object(manager.tempfile, "mkstemp", side_effect=failure), mock.patch.object(
        manager.os, "replace"
    ) as replace:
        with pytest.raises(MicCalibrationIOError) as caught:
            save_channel(path, factor=3.0)

    assert caught.value.__cause__ is failure
    assert path.read_bytes() == original
    replace.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("not JSON serializable"),
        UnicodeEncodeError("utf-8", "x", 0, 1, "encoding failed"),
    ],
)
def test_serialization_failure_is_io_error_and_cleans_temporary(tmp_path, failure):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()

    with mock.patch.object(manager.json, "dump", side_effect=failure):
        with pytest.raises(MicCalibrationIOError) as caught:
            save_channel(path, factor=3.0)

    assert caught.value.__cause__ is failure
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_oversized_integer_dump_failure_preserves_destination_and_cleans_temp(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()
    huge_sample_rate, _ = beyond_digit_limit_channel()

    with pytest.raises(MicCalibrationIOError) as caught:
        save_channel(path, factor=3.0, sample_rate_hz=huge_sample_rate)

    assert isinstance(caught.value.__cause__, ValueError)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_portaudio_identity_failure_has_documented_public_results(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, factor=1.25)
    original = path.read_bytes()

    with mock.patch.object(
        manager.SoundDeviceManager,
        "get_api_info",
        side_effect=sd.PortAudioError("stale host API"),
    ):
        assert build_mic_input_identity(DEVICE) is None
        assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {}
        assert resolve_mic_channel_v2pa_factor(DEVICE, 1, str(path)) is None
        assert get_mic_v2pa_factor(DEVICE, [1], str(path)) == 0.0
        with pytest.raises(ValueError):
            save_channel(path, factor=3.0)

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


class TrackingLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        assert not self.held
        self.held = True

    def __exit__(self, *_args):
        self.held = False


@pytest.mark.parametrize("operation", ["save", "clear"])
def test_mutations_hold_lock_across_read_modify_write(tmp_path, operation):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path)
    tracking_lock = TrackingLock()
    real_load = manager._load_mic_input_calibration_unlocked
    real_write = manager._atomic_write_json

    def tracked_load(*args, **kwargs):
        assert tracking_lock.held
        return real_load(*args, **kwargs)

    def tracked_write(*args, **kwargs):
        assert tracking_lock.held
        return real_write(*args, **kwargs)

    with mock.patch.object(manager, "_mic_input_calibration_io_lock", tracking_lock), mock.patch.object(
        manager, "_load_mic_input_calibration_unlocked", tracked_load
    ), mock.patch.object(manager, "_atomic_write_json", tracked_write):
        if operation == "save":
            save_channel(path, channel=2)
        else:
            clear_mic_channel_calibrations(DEVICE, [1], str(path))

    assert tracking_lock.held is False


def test_concurrent_channel_saves_preserve_both_records(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    barrier = threading.Barrier(2)

    def concurrent_save(channel):
        barrier.wait()
        save_channel(path, channel=channel, factor=channel + 1.25)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(concurrent_save, [0, 2]))

    assert load_mic_channel_v2pa_factors(DEVICE, str(path)) == {0: 1.25, 2: 3.25}


def test_clear_nonexistent_records_is_idempotent_without_write(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=0)
    original = path.read_bytes()

    assert clear_mic_channel_calibrations(DEVICE, [2], str(path)) is False
    assert clear_mic_channel_calibrations(OTHER_DEVICE, [0], str(path)) is False
    assert path.read_bytes() == original


def test_clear_final_channel_removes_empty_device(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=0)

    assert clear_mic_channel_calibrations(DEVICE, [0], str(path)) is True
    assert load_mic_input_calibration(str(path)) == {"version": 2, "devices": []}


def test_exact_device_and_channel_resolution_never_falls_back(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=2, factor=3.5)

    assert resolve_mic_channel_v2pa_factor(DEVICE, 2, str(path)) == 3.5
    assert resolve_mic_channel_v2pa_factor(DEVICE, 1, str(path)) is None
    assert resolve_mic_channel_v2pa_factor(OTHER_DEVICE, 2, str(path)) is None


@pytest.mark.parametrize("channels,expected", [([1], 2.5), ([], 0.0), ([0, 1], 0.0), ([0], 0.0)])
def test_legacy_adapter_only_accepts_one_exact_channel(tmp_path, channels, expected):
    path = tmp_path / "mic_input_calibration.json"
    save_channel(path, channel=1)
    assert get_mic_v2pa_factor(DEVICE, channels, str(path)) == expected


def test_legacy_adapter_propagates_format_errors(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    path.write_text("broken", encoding="utf-8")

    with pytest.raises(MicCalibrationFormatError):
        get_mic_v2pa_factor(DEVICE, [1], str(path))


def test_public_api_has_no_database_path_parameter():
    for function in (
        load_mic_input_calibration, load_mic_channel_calibrations,
        load_mic_channel_v2pa_factors, resolve_mic_channel_v2pa_factor,
        save_mic_channel_calibration, clear_mic_channel_calibrations,
        get_mic_v2pa_factor,
    ):
        assert "db_path" not in inspect.signature(function).parameters
