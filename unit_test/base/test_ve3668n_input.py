"""Pure VE device/configuration contracts, independent of native acquisition."""
from collections import UserDict
from copy import deepcopy
import json
import pickle
from types import MappingProxyType

import pytest

from unit_test.base.ve3668n_fakes import device_info, input_config


def test_ve_acquisition_signature_is_exact_hashable_and_ordered():
    from base.ve3668n_input import ve_acquisition_signature
    from consts.ve3668n_consts import VE_BACKEND

    device = device_info(machine_id=" test-machine-1 ", physical_channels=[0, 2, 7])
    first = ve_acquisition_signature(device, (0, 2), 48000)

    assert first == (VE_BACKEND, "test-machine-1", (0, 2), 48000)
    assert hash(first) == hash(tuple(first))
    assert ve_acquisition_signature(device, (2, 0), 48000) != first
    assert ve_acquisition_signature(device_info(machine_id="other", physical_channels=[0, 2, 7]),
                                    (0, 2), 48000) != first
    assert ve_acquisition_signature(device, (0, 7), 48000) != first
    assert ve_acquisition_signature(device, (0, 2), 51200) != first


def test_ve_acquisition_signature_ignores_request_and_software_metadata(tmp_path):
    from base.ve3668n_input import ve_acquisition_signature
    from unit_test.base.ve3668n_fakes import capture_request

    first = capture_request(tmp_path / "first.wav", target_samples=512,
                            calibration_metadata={"annotation": "first"})
    second = capture_request(tmp_path / "second.wav", target_samples=1024,
                             calibration_metadata={"annotation": "second"})

    assert ve_acquisition_signature(first.device, first.channels, first.sample_rate) == (
        ve_acquisition_signature(second.device, second.channels, second.sample_rate)
    )


@pytest.mark.parametrize("device,channels,rate", [
    (device_info(physical_channels=[0, 2]), (0, 7), 48000),
    (device_info(), (), 48000),
    (device_info(), (7, 1), 96000),
])
def test_ve_acquisition_signature_rejects_incomplete_or_invalid_acquisition(
        device, channels, rate):
    from base.ve3668n_input import ve_acquisition_signature

    with pytest.raises(ValueError):
        ve_acquisition_signature(device, channels, rate)

    incomplete = device_info()
    del incomplete["machine_id"]
    with pytest.raises(ValueError, match="machine_id"):
        ve_acquisition_signature(incomplete, (7, 1), 51200)


@pytest.mark.parametrize(
    "value, accepted",
    [(44100, True), (48000, True), (51200, True)]
    + [(value, False) for value in (
        None, True, False, 44100.0, "48000", 1, 44099, 44101,
        47999, 48001, 51199, 51201, 96000, 102400,
    )],
)
def test_rate_validation_is_strict(value, accepted):
    from base.ve3668n_input import validate_sample_rate

    if accepted:
        assert validate_sample_rate(value) == value
        assert type(validate_sample_rate(value)) is int
    else:
        with pytest.raises(ValueError, match="sample_rate"):
            validate_sample_rate(value)


@pytest.mark.parametrize("level", ["device", "input_config"])
@pytest.mark.parametrize("field", ["sensitivity", "unknown", "index", "hostapi"])
def test_profile_rejects_sensitivity_and_unknown_fields(level, field):
    from base.ve3668n_input import validate_device_snapshot, validate_input_config

    device = device_info()
    target = device if level == "device" else device["input_config"]
    target[field] = 1000.0

    with pytest.raises(ValueError, match=field):
        validate_device_snapshot(device)
    if level == "input_config":
        with pytest.raises(ValueError, match=field):
            validate_input_config(target)


def test_new_config_defaults_only_at_creation():
    from base.ve3668n_input import create_input_config
    from consts.ve3668n_consts import VE_DEFAULT_SAMPLE_RATE, VE_SAMPLE_RATES

    assert VE_SAMPLE_RATES == (44100, 48000, 51200)
    assert VE_DEFAULT_SAMPLE_RATE == 51200
    first = create_input_config()
    assert first == input_config()
    first["sample_rate"] = 44100
    assert create_input_config() == input_config(51200)


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_existing_configs_restore_exact_rate(rate):
    from base.ve3668n_input import create_input_config, validate_input_config

    saved = input_config(rate)
    assert validate_input_config(saved) == saved
    assert create_input_config(rate) == saved


@pytest.mark.parametrize("rate", [
    None, True, False, 44100.0, "48000", 1, 44099, 44101,
    47999, 48001, 51199, 51201, 96000, 102400,
])
def test_invalid_rate_never_falls_back_to_default(rate):
    from base.ve3668n_input import (
        create_input_config, validate_device_snapshot, validate_input_config,
    )

    with pytest.raises(ValueError, match="sample_rate"):
        create_input_config(rate)
    with pytest.raises(ValueError, match="sample_rate"):
        validate_input_config(input_config(rate))
    with pytest.raises(ValueError, match="sample_rate"):
        validate_device_snapshot(device_info(input_config=input_config(rate)))


@pytest.mark.parametrize("field", [
    "sample_rate", "input_mode", "unit", "range_min", "range_max",
])
def test_config_requires_every_saved_field(field):
    from base.ve3668n_input import validate_input_config

    saved = input_config()
    del saved[field]
    with pytest.raises(ValueError, match=field):
        validate_input_config(saved)


@pytest.mark.parametrize("field, value", [
    ("input_mode", "voltage"), ("input_mode", "iepe"), ("input_mode", None),
    ("unit", "Pa"), ("unit", "g"), ("unit", None),
    ("range_min", -5), ("range_max", 5), ("range_min", "-10"),
    ("range_max", "10"), ("range_min", False), ("range_max", True),
    ("range_min", float("nan")), ("range_max", float("inf")),
])
def test_config_rejects_unsupported_acquisition_conditions(field, value):
    from base.ve3668n_input import validate_device_snapshot, validate_input_config

    config = input_config(**{field: value})
    with pytest.raises(ValueError, match=field):
        validate_input_config(config)
    with pytest.raises(ValueError, match=field):
        validate_device_snapshot(device_info(input_config=config))


@pytest.mark.parametrize("value", [None, [], (), "", 51200])
def test_config_requires_a_mapping(value):
    from base.ve3668n_input import validate_input_config

    with pytest.raises(ValueError, match="input_config"):
        validate_input_config(value)


@pytest.mark.parametrize("model", [
    None, True, 3668, "", " \t", "VE3668", "VE3668N-plus", "prefix-VE3668N",
    "VE3668N VE3668N", "VE 3668N", "VE3668N\nother",
])
def test_model_matching_is_full_not_substring(model):
    from base.ve3668n_input import normalize_model, validate_device_snapshot

    device = device_info(model=model)
    with pytest.raises(ValueError, match="model"):
        normalize_model(model)
    with pytest.raises(ValueError, match="model"):
        validate_device_snapshot(device)


@pytest.mark.parametrize("machine_id", [None, True, 1, "", " \t\r\n"])
def test_missing_stable_identity_is_not_replaced_with_name_or_address(machine_id):
    from base.ve3668n_input import normalize_machine_id, validate_device_snapshot

    device = device_info(machine_id=machine_id)
    with pytest.raises(ValueError, match="machine_id"):
        normalize_machine_id(machine_id)
    with pytest.raises(ValueError, match="machine_id"):
        validate_device_snapshot(device)


@pytest.mark.parametrize("channels", [(7, 1), [7, 1], (0,), list(range(8)), range(8)])
def test_physical_channels_keep_order_and_physical_capacity(channels):
    from base.ve3668n_input import validate_device_snapshot, validate_physical_channels

    assert validate_physical_channels(channels) == tuple(channels)
    snapshot = validate_device_snapshot(device_info(physical_channels=channels))
    assert snapshot["physical_channels"] == tuple(channels)
    assert snapshot["max_input_channels"] == 8


@pytest.mark.parametrize("channels", [
    None, [], (), [7, 7], [0, 1, 0], [-1], [8], [True], [False],
    [7, True], [1.0], ["1"], [[1]], "7,1", {7, 1}, {7: 1}, 7,
])
def test_physical_channels_reject_duplicates_invalid_indices_and_unordered_data(channels):
    from base.ve3668n_input import validate_device_snapshot, validate_physical_channels

    with pytest.raises(ValueError, match="physical_channel"):
        validate_physical_channels(channels)
    device = device_info(physical_channels=channels)
    with pytest.raises(ValueError, match="physical_channel"):
        validate_device_snapshot(device)


@pytest.mark.parametrize("channel", [None, True, False, -1, 8, 1.0, "1", [1]])
def test_fingerprint_rejects_invalid_single_physical_channel(channel):
    from base.ve3668n_input import validate_physical_channel

    with pytest.raises(ValueError, match="physical_channel"):
        validate_physical_channel(channel)


@pytest.mark.parametrize("field, value", [
    ("backend", None), ("backend", "sounddevice"), ("backend", "unknown"),
    ("backend", " vkinging "), ("name", None), ("name", ""), ("name", " \t"),
    ("name", 1), ("address", 1), ("address", []),
    ("max_input_channels", None), ("max_input_channels", True),
    ("max_input_channels", 0), ("max_input_channels", 9),
    ("max_input_channels", 8.0), ("max_input_channels", 2),
    ("available", None), ("available", 1), ("available", "true"),
])
def test_device_snapshot_fields_are_validated(field, value):
    from base.ve3668n_input import validate_device_snapshot

    device = device_info(**{field: value})
    with pytest.raises(ValueError, match=field):
        validate_device_snapshot(device)


@pytest.mark.parametrize("address", [None, "", "192.0.2.1"])
def test_unavailable_snapshot_and_optional_address_are_preserved(address):
    from base.ve3668n_input import validate_device_snapshot

    device = validate_device_snapshot(device_info(address=address, available=False))
    assert device["available"] is False
    assert device["address"] == address


@pytest.mark.parametrize("field", [
    "backend", "model", "machine_id", "name", "address", "physical_channels",
    "max_input_channels", "available", "input_config",
])
def test_device_snapshot_requires_all_fields(field):
    from base.ve3668n_input import validate_device_snapshot

    device = device_info()
    del device[field]
    with pytest.raises(ValueError, match=field):
        validate_device_snapshot(device)


@pytest.mark.parametrize("value", [None, [], (), "", 1])
def test_device_snapshot_requires_a_mapping(value):
    from base.ve3668n_input import validate_device_snapshot

    with pytest.raises(ValueError, match="device"):
        validate_device_snapshot(value)


@pytest.mark.parametrize("rate", [22050, 44100, 96000, 102400, 44100.0, "96000", None])
@pytest.mark.parametrize("device", [
    None, {}, {"index": 7, "hostapi": 0, "name": "legacy mic"},
    {"backend": "sounddevice", "index": 7, "hostapi": 0},
])
def test_soundcard_effective_rate_is_returned_untouched(device, rate):
    from base.ve3668n_input import resolve_effective_input_rate

    assert resolve_effective_input_rate(device, rate) is rate


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("product_rate", [22050, 96000, None])
def test_ve_effective_rate_uses_only_its_input_config(rate, product_rate):
    from base.ve3668n_input import resolve_effective_input_rate

    device = device_info(input_config=input_config(rate))
    assert resolve_effective_input_rate(device, product_rate) == rate


@pytest.mark.parametrize("rate", [
    None, True, False, 44100.0, "48000", 1, 44099, 44101,
    47999, 48001, 51199, 51201, 96000, 102400,
])
def test_effective_rate_never_substitutes_a_product_or_default_rate(rate):
    from base.ve3668n_input import resolve_effective_input_rate

    with pytest.raises(ValueError, match="sample_rate"):
        resolve_effective_input_rate(device_info(input_config=input_config(rate)), 44100)


@pytest.mark.parametrize("backend", [None, "", "unknown", "VKINGING", True, 0])
def test_unknown_backend_is_not_a_soundcard_fallback(backend):
    from base.ve3668n_input import resolve_effective_input_rate

    with pytest.raises(ValueError, match="backend"):
        resolve_effective_input_rate({"backend": backend}, 96000)


@pytest.mark.parametrize("device", [[], (), "", 7])
def test_rate_resolution_rejects_nonmapping_device_data(device):
    from base.ve3668n_input import resolve_effective_input_rate

    with pytest.raises(ValueError, match="device"):
        resolve_effective_input_rate(device, 96000)


@pytest.mark.parametrize("overrides, field", [
    ({"sample_rate": 44100}, "sample_rate"),
    ({"sensitivity": 1000.0}, "sensitivity"),
    ({"input_config": input_config(sensitivity=1000.0)}, "sensitivity"),
    ({"input_config": input_config(input_mode="VOLTAGE")}, "input_mode"),
    ({"input_config": input_config(unit="g")}, "unit"),
    ({"input_config": input_config(range_min=-5.0)}, "range_min"),
    ({"machine_id": ""}, "machine_id"),
])
def test_rate_resolution_cannot_bypass_ve_acquisition_validation(overrides, field):
    from base.ve3668n_input import resolve_effective_input_rate

    with pytest.raises(ValueError, match=field):
        resolve_effective_input_rate(device_info(**overrides), 44100)


@pytest.mark.parametrize("mapping_kind", ["dict", "user_dict", "proxy", "frozen"])
def test_all_contract_boundaries_accept_existing_mapping_snapshots(mapping_kind):
    from base.recording_process_protocol import FrozenConfig
    from base.ve3668n_input import (
        resolve_effective_input_rate,
        validate_device_snapshot, validate_input_config,
    )

    wrap = {"dict": dict, "user_dict": UserDict, "proxy": MappingProxyType,
            "frozen": FrozenConfig.snapshot}[mapping_kind]
    raw = device_info(model=" ve3668n ", machine_id=" test-machine-1 ")
    expected = validate_device_snapshot(raw)
    raw["input_config"] = wrap(raw["input_config"])
    source = wrap(raw)

    assert validate_device_snapshot(source) == expected
    assert validate_input_config(source["input_config"]) == input_config()
    assert resolve_effective_input_rate(source, 96000) == 51200
    assert resolve_effective_input_rate(wrap({"backend": "sounddevice"}), 96000) == 96000
    assert resolve_effective_input_rate(wrap({"index": 7}), 102400) == 102400


def test_normalized_snapshots_are_independent_without_mutating_the_caller():
    from base.ve3668n_input import (
        resolve_effective_input_rate,
        validate_device_snapshot, validate_input_config,
    )

    source = device_info(model=" ve3668n ", machine_id=" test-machine-1 ",
                         name=" Dev1 ", address=" 192.0.2.1 ")
    before = deepcopy(source)
    snapshot = validate_device_snapshot(source)
    config = validate_input_config(source["input_config"])
    assert resolve_effective_input_rate(source, 96000) == 51200
    assert source == before
    assert snapshot["name"] == "Dev1"
    assert snapshot["address"] == "192.0.2.1"
    assert snapshot["physical_channels"] == (7, 1)

    snapshot["input_config"]["sample_rate"] = 44100
    config["input_mode"] = "changed"
    assert source == before

    source["input_config"]["unit"] = "Pa"
    source["physical_channels"].reverse()
    source["machine_id"] = "other"
    assert snapshot["input_config"] == input_config(44100)
    assert snapshot["physical_channels"] == (7, 1)
    assert snapshot["machine_id"] == "test-machine-1"
    assert config["unit"] == "V"


def test_snapshots_freeze_and_round_trip_through_json_and_pickle():
    from base.recording_process_protocol import FrozenConfig
    from base.ve3668n_input import (
        validate_device_snapshot, validate_input_config,
    )

    snapshot = validate_device_snapshot(device_info())
    values = (
        snapshot, validate_input_config(snapshot["input_config"]),
    )
    for value in values:
        frozen = FrozenConfig.snapshot(value)
        thawed = json.loads(json.dumps(value, allow_nan=False))
        assert frozen.to_dict() == thawed
        assert pickle.loads(pickle.dumps(value)) == value
        assert pickle.loads(pickle.dumps(frozen)) == frozen
        with pytest.raises(TypeError):
            frozen["new_field"] = "not allowed"

    frozen_device = FrozenConfig.snapshot(snapshot)
    snapshot["input_config"]["sample_rate"] = 48000
    assert frozen_device["input_config"]["sample_rate"] == 51200
    with pytest.raises(TypeError):
        frozen_device["input_config"]["sample_rate"] = 44100
    with pytest.raises(TypeError):
        frozen_device["physical_channels"][0] = 1
    assert validate_device_snapshot(json.loads(json.dumps(snapshot))) == snapshot
    assert validate_device_snapshot(frozen_device)["input_config"]["sample_rate"] == 51200


@pytest.mark.parametrize("overrides", [
    {"input_config": input_config(sensitivity=1000.0)},
    {"input_config": input_config(102400)}, {"physical_channels": [7, 7]},
    {"machine_id": " "}, {"model": "prefix VE3668N"},
])
def test_failed_validation_also_leaves_caller_data_unchanged(overrides):
    from base.ve3668n_input import resolve_effective_input_rate, validate_device_snapshot

    source = device_info(**overrides)
    before = deepcopy(source)
    for validate in (validate_device_snapshot, lambda value: resolve_effective_input_rate(value, 44100)):
        with pytest.raises(ValueError):
            validate(source)
        assert source == before


def test_rates_and_channels_require_python_int_not_numeric_lookalikes():
    import numpy as np
    from base.ve3668n_input import validate_physical_channel, validate_sample_rate

    class IntSubclass(int):
        pass

    for rate in (IntSubclass(44100), np.int64(48000), float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sample_rate"):
            validate_sample_rate(rate)
    for channel in (IntSubclass(1), np.int64(7)):
        with pytest.raises(ValueError, match="physical_channel"):
            validate_physical_channel(channel)


def test_shared_schema_constants_and_fake_data_do_not_share_mutable_state():
    from consts import ve3668n_consts

    assert ve3668n_consts.VE_INPUT_SCHEMA_VERSION == 1
    assert ve3668n_consts.VE_MAX_INPUT_CHANNELS == 8
    assert set(ve3668n_consts.VE_DEVICE_SNAPSHOT_FIELDS) == set(device_info())
    assert set(ve3668n_consts.VE_INPUT_CONFIG_FIELDS) == set(input_config())
    assert all(not isinstance(value, (dict, list, set))
               for name, value in vars(ve3668n_consts).items() if name.startswith("VE_"))
    first, second = device_info(), device_info()
    first["input_config"]["sample_rate"] = 44100
    first["physical_channels"].reverse()
    assert second == device_info()
