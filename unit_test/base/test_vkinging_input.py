"""Pure vkinging configuration and request contracts: no native SDK required."""
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from types import SimpleNamespace

import pytest


def device(**changes):
    result = {
        "hostapi_name": "vkinging", "hardware_id": "machine-A", "name": "input-A",
        "samplerate": 51200, "bit_depth": 32, "physical_channels": (0, 8, 15, 31),
    }
    result.update(changes)
    return result


def test_queue_range_and_hardware_rate_have_separate_authority():
    from base.vkinging_input import resolve_vkinging_input_config

    config = resolve_vkinging_input_config(
        {"sample_rate": 44100, "bit_depth": 64, "vkinging_range_index": 2}, device())
    assert config == {"sample_rate": 51200, "bit_depth": 32,
                      "input_mode": "IEPE", "unit": "V",
                      "range_min": -2.5, "range_max": 2.5}


@pytest.mark.parametrize("value, expected", [
    (None, False), ("vkinging", False), ({"backend": "vkinging"}, False),
    ({"hostapi_name": "Vkinging"}, False), ({"hostapi_name": "WASAPI"}, False),
    (MappingProxyType({"hostapi_name": "vkinging"}), True),
])
def test_backend_identity_uses_only_reserved_hostapi_name(value, expected):
    from base.vkinging_input import is_vkinging_device

    assert is_vkinging_device(value) is expected


@pytest.mark.parametrize("index, limit", enumerate((10, 5, 2.5, 1, .5, .1, .02)))
def test_range_index_resolves_all_seven_voltage_ranges(index, limit):
    from base.vkinging_input import resolve_vkinging_input_config, validate_range_index

    assert validate_range_index(index) == index
    config = resolve_vkinging_input_config({"vkinging_range_index": index}, device())
    assert (config["range_min"], config["range_max"]) == (-limit, limit)


@pytest.mark.parametrize("value", [True, False, 2.0, "2", None, -1, 7])
def test_explicit_invalid_queue_range_is_rejected(value):
    from base.vkinging_input import resolve_vkinging_input_config

    with pytest.raises(ValueError, match="vkinging_range_index"):
        resolve_vkinging_input_config({"vkinging_range_index": value}, device())


@pytest.mark.parametrize("value", [4000, 44100, 48000, 51200, 100001, 102400])
def test_hardware_integer_rate_range(value):
    from base.vkinging_input import resolve_vkinging_input_config

    assert resolve_vkinging_input_config({}, device(samplerate=value))["sample_rate"] == value


@pytest.mark.parametrize("value", [True, 51200.0, "51200", None, 3999, 102401])
def test_invalid_hardware_rate_has_no_queue_fallback(value):
    from base.vkinging_input import resolve_vkinging_input_config

    with pytest.raises(ValueError, match="sample_rate"):
        resolve_vkinging_input_config({"sample_rate": 48000}, device(samplerate=value))


@pytest.mark.parametrize("value", [True, 32.0, "32", None, 24])
def test_bit_depth_is_strictly_hardware_float_format(value):
    from base.vkinging_input import resolve_vkinging_input_config

    with pytest.raises(ValueError, match="bit_depth"):
        resolve_vkinging_input_config({"bit_depth": 32}, device(bit_depth=value))


def test_input_resolution_requires_vkinging_device_and_mapping_detail():
    from base.vkinging_input import resolve_vkinging_input_config

    with pytest.raises(ValueError, match="hostapi_name"):
        resolve_vkinging_input_config({}, {"samplerate": 48000})
    with pytest.raises(ValueError, match="detail"):
        resolve_vkinging_input_config(None, device())


def test_calibration_range_uses_current_record_item_before_defaults():
    from base.vkinging_input import resolve_calibration_range_index

    defaults = {"RECORD_ONLY": {"vkinging_range_index": 6}}
    assert resolve_calibration_range_index(
        {"mode": "RECORD_ONLY", "detail": {"vkinging_range_index": 2}}, defaults) == 2
    assert resolve_calibration_range_index({"mode": "RECORD_ONLY", "detail": {}}, defaults) == 0
    assert resolve_calibration_range_index(None, defaults) == 6
    assert resolve_calibration_range_index({"mode": "PLAY_AND_RECORD", "detail": {}}, defaults) == 6
    assert resolve_calibration_range_index(None, {}) == 0
    assert resolve_calibration_range_index(
        SimpleNamespace(mode="RECORD_ONLY", detail={"vkinging_range_index": 3}), defaults) == 3


@pytest.mark.parametrize("acq, defaults", [
    ({"mode": "RECORD_ONLY", "detail": {"vkinging_range_index": True}}, {}),
    (None, {"RECORD_ONLY": {"vkinging_range_index": 7}}),
])
def test_calibration_range_rejects_explicit_invalid_values(acq, defaults):
    from base.vkinging_input import resolve_calibration_range_index

    with pytest.raises(ValueError, match="vkinging_range_index"):
        resolve_calibration_range_index(acq, defaults)


@pytest.mark.parametrize("number", [1, 9, 16, 32, 1000])
def test_ain_route_preserves_real_zero_based_physical_index(number):
    from base.vkinging_input import parse_vkinging_ain_route

    assert parse_vkinging_ain_route(f"input-A/AIN{number}") == number - 1


def test_ain_route_allows_spaces_in_the_real_device_name():
    from base.vkinging_input import parse_vkinging_ain_route

    assert parse_vkinging_ain_route("Input device A/AIN32") == 31


@pytest.mark.parametrize("route", ["input-A/AIN0", "AIN1", "input-A/AIN-1", "input-A/AIN1x",
                                   "input-A/DIN1", "input-A/AIN1.0", "input-A/AIN", None])
def test_invalid_analog_routes_are_rejected(route):
    from base.vkinging_input import parse_vkinging_ain_route

    with pytest.raises(ValueError, match="route"):
        parse_vkinging_ain_route(route)


@pytest.mark.parametrize("channels", [tuple(range(16)), tuple(range(32)), (31, 0, 15, 8)])
def test_dynamic_channels_preserve_order_and_sparse_indices(channels):
    from base.vkinging_input import validate_physical_channels

    assert validate_physical_channels(list(channels)) == channels


@pytest.mark.parametrize("channels", [(), (0, 0), (-1,), (True,), (1.0,), "0,1", {1, 2}])
def test_invalid_channel_selection_is_rejected(channels):
    from base.vkinging_input import validate_physical_channels

    with pytest.raises(ValueError, match="physical_channel"):
        validate_physical_channels(channels)


def request_values(**changes):
    result = dict(request_id="request-A", purpose="main", device=device(), channels=[15, 0, 8],
                  sample_rate=51200, bit_depth=32, range_index=2, target_samples=204800,
                  discard_initial_samples=5120, output_path="record.partial.wav",
                  calibration_snapshot={"recorded_channels": [{"physical_channel": 15, "factor": 2.5}]})
    result.update(changes)
    return result


def test_capture_request_defensively_freezes_all_nested_state():
    from base.vkinging_input import VkingingCaptureRequest

    values = request_values()
    values["device"]["extra"] = {"routing": [0, 8, 15]}
    request = VkingingCaptureRequest(**values)
    values["device"]["extra"]["routing"].append(31)
    values["channels"].clear()
    values["calibration_snapshot"]["recorded_channels"][0]["factor"] = 99
    assert request.channels == (15, 0, 8)
    assert request.device["extra"]["routing"] == (0, 8, 15)
    assert request.calibration_snapshot["recorded_channels"][0]["factor"] == 2.5
    assert isinstance(request.device, Mapping)
    with pytest.raises(TypeError):
        request.device["extra"]["routing"] = ()
    with pytest.raises(TypeError):
        request.calibration_snapshot["recorded_channels"][0]["factor"] = 3
    with pytest.raises(FrozenInstanceError):
        request.range_index = 1


@pytest.mark.parametrize("count", [16, 32])
def test_capture_request_allows_full_dynamic_device_channel_count(count):
    from base.vkinging_input import VkingingCaptureRequest

    channels = tuple(range(count))
    request = VkingingCaptureRequest(**request_values(device=device(physical_channels=channels), channels=channels))
    assert request.channels == channels


@pytest.mark.parametrize("changes, field", [
    ({"request_id": ""}, "request_id"), ({"purpose": "listen"}, "purpose"),
    ({"channels": [1]}, "channels"), ({"sample_rate": 48000}, "sample_rate"),
    ({"bit_depth": 64}, "bit_depth"), ({"range_index": True}, "vkinging_range_index"),
    ({"target_samples": 0}, "target_samples"), ({"target_samples": 1.5}, "target_samples"),
    ({"discard_initial_samples": -1}, "discard_initial_samples"),
    ({"discard_initial_samples": True}, "discard_initial_samples"),
    ({"device": device(hardware_id="")}, "hardware_id"),
    ({"device": device(physical_channels=())}, "physical_channel"),
    ({"calibration_snapshot": {"bad": bytearray(b"mutable")}}, "snapshot"),
])
def test_capture_request_rejects_invalid_contract(changes, field):
    from base.vkinging_input import VkingingCaptureRequest

    with pytest.raises(ValueError, match=field):
        VkingingCaptureRequest(**request_values(**changes))


def test_calibration_request_is_exactly_ten_seconds_single_channel_without_discard():
    from base.vkinging_input import VkingingCaptureRequest

    values = request_values(purpose="calibration", channels=[31], target_samples=512000,
                            discard_initial_samples=0, output_path=None, calibration_snapshot=None)
    assert VkingingCaptureRequest(**values).channels == (31,)
    for changes in ({"channels": [0, 8]}, {"target_samples": 511999}, {"discard_initial_samples": 1}):
        with pytest.raises(ValueError, match="calibration"):
            VkingingCaptureRequest(**(values | changes))
