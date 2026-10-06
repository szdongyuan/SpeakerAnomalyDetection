import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import pytest

from base.recording_defaults import (
    RecordingDefaultsStore,
    recording_profile_key,
    validated_recording_profile,
)
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from consts.running_consts import DEFAULT_DIR
from consts.ve3668n_consts import VE_BACKEND, VE_RANGE_INDEX_CONFIG_KEY


def test_profiles_survive_new_store_instances(tmp_path):
    path = tmp_path / "defaults.json"
    store = RecordingDefaultsStore(path)
    assert store.save("soundcard", {"total_time": 2.5, "sample_rate": 48000}) is None
    assert store.save("vkinging", {"sample_rate": 4000, VE_RANGE_INDEX_CONFIG_KEY: 5}) is None
    reopened = RecordingDefaultsStore(path)
    assert reopened.load("soundcard") == {"total_time": 2.5, "sample_rate": 48000}
    assert reopened.load("vkinging") == {"sample_rate": 4000, VE_RANGE_INDEX_CONFIG_KEY: 5}


def test_missing_profiles_and_returned_data_are_independent(tmp_path):
    store = RecordingDefaultsStore(tmp_path / "defaults.json")
    missing = store.load("soundcard")
    missing["total_time"] = 9
    assert store.load("soundcard") == {}
    detail = {"total_time": 2}
    store.save("soundcard", detail)
    detail["total_time"] = 10
    loaded = store.load("soundcard")
    loaded["total_time"] = 15
    assert store.load("soundcard") == {"total_time": 2}
    assert store.load("vkinging") == {}


@pytest.mark.parametrize("mic, expected", [
    (None, None), ({}, None), ({"name": "Mic"}, "soundcard"),
    ({"backend": VE_BACKEND}, "vkinging"),
    ({"backend": "other"}, "soundcard"),
])
def test_device_profile_selection(mic, expected):
    assert recording_profile_key(mic) == expected


@pytest.mark.parametrize("key", [None, "", "unknown", True, [], {}])
def test_invalid_profile_keys_are_rejected(tmp_path, key):
    store = RecordingDefaultsStore(tmp_path / "defaults.json")
    with pytest.raises(ValueError, match="profile_key"):
        store.load(key)
    with pytest.raises(ValueError, match="profile_key"):
        store.save(key, {})
    with pytest.raises(ValueError, match="profile_key"):
        validated_recording_profile(key, {})


def test_default_path_is_the_application_configuration_path():
    assert RecordingDefaultsStore.default_path() == Path(DEFAULT_DIR) / "ui/ui_config/recording_default_config.json"


def test_validation_projects_mapping_without_modifying_it():
    detail = MappingProxyType({
        "total_time": 100000, "sample_rate": 44100, "startup_trim_ms": 900001,
        "use_streaming_recording": False,
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE,
        VE_RANGE_INDEX_CONFIG_KEY: "irrelevant", "monitor_extra": {"x": 1},
    })
    expected = dict(detail)
    del expected[VE_RANGE_INDEX_CONFIG_KEY]
    del expected["monitor_extra"]
    assert validated_recording_profile("soundcard", detail) == expected
    assert detail[VE_RANGE_INDEX_CONFIG_KEY] == "irrelevant"
    assert validated_recording_profile("vkinging", {"startup_trim_ms": 0}) == {"startup_trim_ms": 0}
    assert validated_recording_profile("soundcard", {}) == {}


def write_envelope(path, profiles):
    path.write_text(json.dumps({"schema_version": 1, "profiles": profiles}), encoding="utf-8")


@pytest.mark.parametrize("envelope, diagnostic", [
    (None, "object"), ([], "object"), (True, "object"),
    ({}, "schema_version"),
    ({"schema_version": True, "profiles": {}}, "schema_version"),
    ({"schema_version": 1.0, "profiles": {}}, "schema_version"),
    ({"schema_version": 2, "profiles": {}}, "schema_version"),
    ({"schema_version": 1}, "profiles"),
    ({"schema_version": 1, "profiles": None}, "profiles"),
    ({"schema_version": 1, "profiles": []}, "profiles"),
])
def test_invalid_envelope_is_rejected(tmp_path, envelope, diagnostic):
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match=diagnostic):
        RecordingDefaultsStore(path).load("soundcard")


@pytest.mark.parametrize("detail", [None, [], True, 1, "invalid"])
def test_explicit_non_mapping_template_is_rejected(tmp_path, detail):
    path = tmp_path / "defaults.json"
    write_envelope(path, {"soundcard": detail})
    with pytest.raises(ValueError, match="mapping"):
        RecordingDefaultsStore(path).load("soundcard")
    with pytest.raises(ValueError, match="mapping"):
        validated_recording_profile("soundcard", detail)


@pytest.mark.parametrize("key, field, value", [
    *[("soundcard", "total_time", v) for v in
      (None, True, "2", 0, -1, float("nan"), float("inf"), float("-inf"), 10 ** 400)],
    *[("soundcard", "sample_rate", v) for v in (None, True, "48000", 48000.0, 4000)],
    *[("vkinging", "sample_rate", v) for v in (None, True, "4000", 4000.0, 0, 999999999)],
    *[("soundcard", "startup_trim_ms", v) for v in (None, True, "0", 0.0, -1)],
    *[("soundcard", "use_streaming_recording", v) for v in (None, 0, 1, "false")],
    *[("soundcard", RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY, v) for v in (None, True, 0, "invalid")],
    *[("vkinging", VE_RANGE_INDEX_CONFIG_KEY, v) for v in (None, True, "5", 5.0, -1, 7)],
])
def test_invalid_known_fields_are_rejected_on_load_and_validation(tmp_path, key, field, value):
    path = tmp_path / "defaults.json"
    write_envelope(path, {key: {field: value}})
    with pytest.raises(ValueError, match=field):
        RecordingDefaultsStore(path).load(key)
    with pytest.raises(ValueError, match=field):
        validated_recording_profile(key, {field: value})


@pytest.mark.parametrize("content, error", [(b"{broken", ValueError), (b"\xff", UnicodeError)])
def test_file_parse_errors_propagate(tmp_path, content, error):
    path = tmp_path / "defaults.json"
    path.write_bytes(content)
    with pytest.raises(error):
        RecordingDefaultsStore(path).load("soundcard")


def test_read_permission_error_propagates(tmp_path, monkeypatch):
    path = tmp_path / "defaults.json"
    failure = PermissionError("read denied")

    def deny_open(*args, **kwargs):
        raise failure

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(PermissionError) as caught:
        RecordingDefaultsStore(path).load("soundcard")
    assert caught.value is failure


def test_load_ignores_other_invalid_template_and_unknown_fields(tmp_path):
    path = tmp_path / "defaults.json"
    write_envelope(path, {
        "soundcard": {"total_time": 2, "device": {"name": "私人设备"}, VE_RANGE_INDEX_CONFIG_KEY: None},
        "vkinging": None,
    })
    store = RecordingDefaultsStore(path)
    assert store.load("soundcard") == {"total_time": 2}
    with pytest.raises(ValueError, match="mapping"):
        store.load("vkinging")


@pytest.mark.parametrize("stage", ["create", "write", "replace"])
def test_failed_atomic_save_preserves_file_and_cleans_temp(tmp_path, monkeypatch, stage):
    path = tmp_path / "defaults.json"
    write_envelope(path, {"soundcard": {"total_time": 2}})
    before = path.read_bytes()
    failure = OSError(f"{stage} failed")

    def fail(*args, **kwargs):
        if stage == "write":
            args[1].write('{"partial":')
        raise failure

    if stage == "create":
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail)
    elif stage == "write":
        monkeypatch.setattr(json, "dump", fail)
    else:
        monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError) as caught:
        RecordingDefaultsStore(path).save("soundcard", {"total_time": 3})
    assert caught.value is failure
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_save_creates_parents_and_replaces_whole_current_template(tmp_path):
    path = tmp_path / "nested" / "config" / "defaults.json"
    store = RecordingDefaultsStore(path)
    store.save("soundcard", {"total_time": 2, "sample_rate": 48000})
    store.save("soundcard", {"startup_trim_ms": 0, VE_RANGE_INDEX_CONFIG_KEY: 5, "directory": "private"})
    assert store.load("soundcard") == {"startup_trim_ms": 0}
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1, "profiles": {"soundcard": {"startup_trim_ms": 0}},
    }
    assert set(path.parent.iterdir()) == {path}


@pytest.mark.parametrize("content", [
    b"{broken", b"\xff", b"[]",
    b'{"schema_version": 2, "profiles": {}}',
    b'{"schema_version": true, "profiles": {}}',
    b'{"schema_version": 1, "profiles": []}',
])
def test_save_does_not_overwrite_bad_existing_file(tmp_path, content):
    path = tmp_path / "defaults.json"
    path.write_bytes(content)
    with pytest.raises((ValueError, UnicodeError)):
        RecordingDefaultsStore(path).save("soundcard", {"total_time": 2})
    assert path.read_bytes() == content
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("other", [None, [], {"sample_rate": "invalid", "private": {"name": "私人设备"}}])
def test_save_preserves_other_template_without_validating_it(tmp_path, other):
    path = tmp_path / "defaults.json"
    write_envelope(path, {"soundcard": None, "vkinging": other})
    RecordingDefaultsStore(path).save("soundcard", {"total_time": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"] == {
        "soundcard": {"total_time": 2}, "vkinging": other,
    }


@pytest.mark.parametrize("literal", ["1e999", "NaN"])
def test_save_preserves_other_nonfinite_template_and_load_still_rejects_it(tmp_path, literal):
    path = tmp_path / "defaults.json"
    path.write_text(
        '{"schema_version": 1, "profiles": {"vkinging": {"total_time": ' + literal + '}}}',
        encoding="utf-8",
    )
    store = RecordingDefaultsStore(path)
    assert store.save("soundcard", {"total_time": 2}) is None
    preserved = json.loads(path.read_text(encoding="utf-8"))["profiles"]["vkinging"]["total_time"]
    assert math.isinf(preserved) if literal == "1e999" else math.isnan(preserved)
    assert store.load("soundcard") == {"total_time": 2}
    with pytest.raises(ValueError, match="total_time"):
        store.load("vkinging")


def test_invalid_new_profile_is_rejected_before_file_access(tmp_path, monkeypatch):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid profile must be rejected before reading the file")

    monkeypatch.setattr(Path, "open", unexpected_open)
    with pytest.raises(ValueError, match="total_time"):
        RecordingDefaultsStore(tmp_path / "defaults.json").save("soundcard", {"total_time": float("nan")})


def test_cleanup_error_is_logged_without_hiding_primary_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "defaults.json"
    write_envelope(path, {"soundcard": {"total_time": 2}})
    before = path.read_bytes()
    failure = OSError("replace failed")

    def fail_replace(*args):
        raise failure

    def fail_cleanup(*args, **kwargs):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as caught:
        RecordingDefaultsStore(path).save("soundcard", {"total_time": 3})
    assert caught.value is failure
    assert path.read_bytes() == before
    assert "cleanup denied" in caplog.text
