import os
import threading

import pytest

from base.vkinging_discovery import discover_vkinging_devices
from base.vkinging_sdk import VkDaqError
from unit_test.base.vkinging_fakes import FakeSDK


def device_sdk(name="dev", machine_id="Exact-ID", routes=None):
    return FakeSDK(devices=(("address", name),),
                   attributes={(name, "MachineId"): machine_id, (name, "Model"): "native model"},
                   channels={name: tuple(routes) if routes is not None else (f"{name}/AIN1", f"{name}/AIN32")})


def test_discovery_returns_runtime_snapshot_and_preserves_route_order():
    sdk = device_sdk(routes=("dev/AIN32", "dev/DIN1", "dev/AIN1", "dev/MIN1", "dev/AIN9", "dev/RAWAIN1", "dev/AIN16"))
    result = discover_vkinging_devices(sdk)
    assert not result.diagnostics
    assert len(result.devices) == 1
    device = result.devices[0]
    assert device["hardware_id"] == "Exact-ID"
    assert device["hostapi_name"] == "vkinging"
    assert device["name"] == "dev"
    assert device["physical_channels"] == (31, 0, 8, 15)
    assert device["max_input_channels"] == 32
    assert device["model"] == "native model"
    assert not sdk.closed
    assert {call[0] for call in sdk.calls} == {"get_devices", "get_device_attribute", "get_channels"}
    assert {call[2] for call in sdk.calls} == {threading.get_ident()}
    assert {call[3] for call in sdk.calls} == {os.getpid()}


@pytest.mark.parametrize("count", [16, 32])
def test_dynamic_channel_counts(count):
    sdk = device_sdk(routes=tuple(f"dev/AIN{i}" for i in range(1, count + 1)))
    assert discover_vkinging_devices(sdk).devices[0]["physical_channels"] == tuple(range(count))


def test_runtime_snapshots_are_immutable():
    result = discover_vkinging_devices(device_sdk())
    with pytest.raises(TypeError):
        result.devices[0]["name"] = "changed"


def test_optional_model_failure_diagnoses_but_keeps_verified_device():
    sdk = device_sdk()
    sdk.attributes[("dev", "Model")] = None
    result = discover_vkinging_devices(sdk)
    assert len(result.devices) == 1
    assert result.devices[0]["model"] is None
    assert "optional Model" in result.diagnostics[0]


@pytest.mark.parametrize("entries", [None, "bad", (("one",),), (("a", "bad/name"),)])
def test_malformed_enumeration_diagnosed(entries):
    sdk = FakeSDK()
    sdk.devices = entries
    result = discover_vkinging_devices(sdk)
    assert not result.devices
    assert result.diagnostics


def test_name_change_keeps_exact_stable_identity():
    old = discover_vkinging_devices(device_sdk("before", " ID-Case ")).devices[0]
    new = discover_vkinging_devices(device_sdk("after", " ID-Case ")).devices[0]
    assert old["hardware_id"] == new["hardware_id"] == " ID-Case "
    assert old["name"] != new["name"]


def test_duplicate_native_names_are_not_queried():
    sdk = FakeSDK(devices=(("a", "same"), ("b", "same")))
    result = discover_vkinging_devices(sdk)
    assert not result.devices
    assert "name" in " ".join(result.diagnostics)
    assert [call[0] for call in sdk.calls] == ["get_devices"]


def test_duplicate_id_rejects_even_candidate_with_bad_channels():
    sdk = device_sdk()
    sdk.devices += (("other", "dev2"),)
    sdk.attributes.update({("dev2", "MachineId"): "Exact-ID", ("dev2", "Model"): "model"})
    sdk.channels["dev2"] = ("dev2/AIN0",)
    result = discover_vkinging_devices(sdk)
    assert not result.devices
    assert "identity" in " ".join(result.diagnostics)


@pytest.mark.parametrize("routes", [(), ("dev/AIN0",), ("dev/AIN01",), ("dev/AINx",),
    ("other/AIN1",), ("dev/AIN1", "dev/AIN1"), ("dev/UNKNOWN1",), ("dev/DIN1",), (None,)])
def test_malformed_or_unverifiable_analog_channels_diagnosed(routes):
    result = discover_vkinging_devices(device_sdk(routes=routes))
    assert not result.devices
    assert result.diagnostics


@pytest.mark.parametrize("machine_id", ["", " ", None, "id\x00bad", "id\nbad"])
def test_missing_or_malformed_identity_rejected(machine_id):
    result = discover_vkinging_devices(device_sdk(machine_id=machine_id))
    assert not result.devices
    assert result.diagnostics


def test_sdk_error_becomes_discovery_diagnostic_without_closing_caller_sdk():
    sdk = device_sdk()
    sdk.failures["get_devices"] = VkDaqError("get_devices", -777, "offline")
    result = discover_vkinging_devices(sdk)
    assert not result.devices
    assert "-777" in result.diagnostics[0]
    assert not sdk.closed


def test_unexpected_sdk_exception_propagates_to_owner_boundary():
    sdk = device_sdk()
    sdk.failures["get_devices"] = RuntimeError("unexpected fake failure")
    with pytest.raises(RuntimeError, match="unexpected fake"):
        discover_vkinging_devices(sdk)
