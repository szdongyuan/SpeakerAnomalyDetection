import ctypes
import runpy
import math
import os
from pathlib import Path
import threading

import pytest

from base.vkinging_sdk import VkDaqClient, VkDaqError
from unit_test.base.vkinging_fakes import FakeDLL


def test_import_and_construction_are_lazy(monkeypatch):
    def forbidden(*args):
        pytest.fail("native loading during import/construction")
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    runpy.run_path("base/vkinging_sdk.py")
    runpy.run_path("base/vkinging_discovery.py")
    VkDaqClient(dll_path="does-not-exist.dll")


def test_explicit_path_has_no_fallback(tmp_path, monkeypatch):
    (tmp_path / "libvkdaq.dll").touch()
    monkeypatch.setenv("VKDAQ_HOME", str(tmp_path))
    client = VkDaqClient(dll_path=tmp_path / "absent.dll")
    with pytest.raises(RuntimeError, match="not found") as failure:
        client.get_devices()
    assert failure.value.operation == "load_sdk"


@pytest.mark.parametrize("code,reason", [(193, "architecture"), (126, "dependencies")])
def test_loader_failure_preserves_code_and_closes_directory(tmp_path, monkeypatch, code, reason):
    path = tmp_path / "libvkdaq.dll"
    path.touch()
    closed = []
    class Directory:
        def close(self):
            closed.append(True)
    monkeypatch.setattr(os, "add_dll_directory", lambda parent: Directory())
    def fail(path):
        error = OSError("injected loader failure")
        error.winerror = code
        raise error
    monkeypatch.setattr(ctypes, "CDLL", fail)
    with pytest.raises(RuntimeError, match=reason) as failure:
        VkDaqClient(dll_path=path).get_devices()
    assert failure.value.code == code
    assert closed == [True]


def test_environment_precedes_install_search_and_uses_cdecl(tmp_path, monkeypatch):
    arch = "x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
    home = tmp_path / arch
    home.mkdir()
    path = home / "libvkdaq.dll"
    path.touch()
    monkeypatch.setenv("VKDAQ_HOME", str(tmp_path))
    installed = tmp_path / "installed" / "VkDaqAssistant" / arch / "libvkdaq.dll"
    installed.parent.mkdir(parents=True)
    installed.touch()
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.setenv(variable, str(tmp_path / "installed"))
    loaded = []
    dll = FakeDLL()
    dll.VkDaqCreateTask.handler = lambda task: 0
    monkeypatch.setattr(ctypes, "CDLL", lambda value: loaded.append(value) or dll)
    class Directory:
        def close(self):
            pass
    monkeypatch.setattr(os, "add_dll_directory", lambda parent: Directory())
    with VkDaqClient() as sdk:
        sdk.create_task("task")
    assert Path(loaded[0]) == path


def test_install_search_matches_python_architecture(tmp_path, monkeypatch):
    bits = ctypes.sizeof(ctypes.c_void_p) * 8
    arch = "x64" if bits == 64 else "x86"
    path = tmp_path / "VkDaqAssistant" / arch / "libvkdaq.dll"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.delenv("VKDAQ_HOME", raising=False)
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.setenv(variable, str(tmp_path))
    loaded = []
    dll = FakeDLL()
    monkeypatch.setattr(ctypes, "CDLL", lambda value: loaded.append(value) or dll)
    class Directory:
        def close(self):
            pass
    monkeypatch.setattr(os, "add_dll_directory", lambda parent: Directory())
    with VkDaqClient() as sdk:
        sdk.create_task("task")
    assert Path(loaded[0]) == path


@pytest.mark.parametrize("root_variable", ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"])
def test_32_bit_search_finds_x86_library_under_each_install_root(tmp_path, monkeypatch, root_variable):
    monkeypatch.setattr(ctypes, "sizeof", lambda value: 4)
    monkeypatch.delenv("VKDAQ_HOME", raising=False)
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    selected_root = tmp_path / root_variable
    expected = selected_root / "VkDaqAssistant" / "x86" / "libvkdaq.dll"
    expected.parent.mkdir(parents=True)
    expected.touch()
    # A native x64 DLL in an earlier root must never become a 32-bit candidate.
    wrong_arch = tmp_path / "ProgramW6432" / "VkDaqAssistant" / "x64" / "libvkdaq.dll"
    wrong_arch.parent.mkdir(parents=True)
    wrong_arch.touch()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: pytest.fail("selection must not load a DLL"))
    assert VkDaqClient()._select_dll() == expected


def test_missing_symbol_is_diagnostic():
    dll = FakeDLL()
    del dll.VkDaqGetTaskData
    with pytest.raises(RuntimeError, match="VkDaqGetTaskData") as failure:
        VkDaqClient(dll=dll).create_task("task")
    assert failure.value.operation == "bind_sdk"


def test_native_signatures_and_iepe_clock_lifecycle_parameters():
    dll = FakeDLL()
    sdk = VkDaqClient(dll=dll)
    sdk.create_task("采集")
    sdk.create_iepe_voltage_channel("采集", "dev/AIN32,dev/AIN1", range_min=-2.5, range_max=2.5)
    sdk.configure_sample_clock("采集", 51200)
    sdk.start_task("采集")
    sdk.stop_task("采集")
    sdk.clear_task("采集")
    c, i, d = ctypes.c_char_p, ctypes.c_int32, ctypes.c_double
    out = ctypes.POINTER(ctypes.c_char)
    signatures = {
        "VkDaqGetLastErrorInfo": ([], c),
        "VkDaqGetDevices": ([out, out, i], i),
        "VkDaqGetChannels": ([c, out, i], i),
        "VkDaqGetDeviceAttribute": ([c, c, out, i], i),
        "VkDaqCreateAIAccelChan": ([c, c, c, i, d, d, i, d, i, c], i),
        "VkDaqCfgSampClkTiming": ([c, i, d, i, i, i], i),
        "VkDaqGetTaskData": ([c, ctypes.POINTER(d), i, i, d], i),
    }
    for name in ("VkDaqCreateTask", "VkDaqStartTask", "VkDaqStopTask", "VkDaqClearTask"):
        signatures[name] = ([c], i)
    for name, (args, result) in signatures.items():
        assert getattr(dll, name).argtypes == args
        assert getattr(dll, name).restype is result
    assert dll.calls[1][1] == ("采集".encode(), b"dev/AIN32,dev/AIN1", b"", 0,
                              -2.5, 2.5, 4, 1000.0, 0, b"")
    assert dll.calls[2][1] == ("采集".encode(), 0, 51200.0, 1, 0, 0)
    assert {entry[2] for entry in dll.calls} == {threading.get_ident()}
    assert {entry[3] for entry in dll.calls} == {os.getpid()}


def test_text_growth_and_utf8_enumeration():
    dll = FakeDLL()
    sizes = []
    def devices(addresses, names, size):
        sizes.append(size)
        if len(sizes) == 1:
            return -10028
        dll.write_text(addresses, "地址")
        dll.write_text(names, "设备")
        return 0
    dll.VkDaqGetDevices.handler = devices
    dll.VkDaqGetChannels.handler = lambda name, buf, size: dll.write_text(buf, "设备/AIN32") or 0
    dll.VkDaqGetDeviceAttribute.handler = lambda dev, attr, buf, size: dll.write_text(buf, "ID甲") or 0
    sdk = VkDaqClient(dll=dll)
    assert sdk.get_devices() == (("地址", "设备"),)
    assert sizes == [4096, 8192]
    assert sdk.get_channels("设备") == ("设备/AIN32",)
    assert sdk.get_device_attribute("设备", "MachineId") == "ID甲"


@pytest.mark.parametrize("raw,reason", [(None, "NUL"), (b"\xff", "UTF-8")])
def test_invalid_native_text_rejected(raw, reason):
    dll = FakeDLL()
    if raw is not None:
        dll.VkDaqGetChannels.handler = lambda name, buf, size: dll.write_text(buf, raw) or 0
    with pytest.raises(RuntimeError, match=reason):
        VkDaqClient(dll=dll).get_channels("dev")


def test_text_growth_is_bounded_and_preserves_sdk_code():
    dll = FakeDLL()
    dll.VkDaqGetChannels.handler = lambda *args: -10028
    with pytest.raises(VkDaqError) as failure:
        VkDaqClient(dll=dll).get_channels("dev")
    assert failure.value.code == -10028
    sizes = [call[1][-1] for call in dll.calls if call[0] == "VkDaqGetChannels"]
    assert sizes == [4096 * 2**power for power in range(9)]


@pytest.mark.parametrize("addresses,names", [("a,b", "name"), ("a", "name,"), ("a", "bad\nname")])
def test_malformed_device_lists_rejected(addresses, names):
    dll = FakeDLL()
    def devices(out_addresses, out_names, size):
        dll.write_text(out_addresses, addresses)
        dll.write_text(out_names, names)
        return 0
    dll.VkDaqGetDevices.handler = devices
    with pytest.raises(VkDaqError):
        VkDaqClient(dll=dll).get_devices()


def test_secondary_error_info_failure_keeps_primary_code():
    dll = FakeDLL()
    dll.VkDaqStartTask.handler = lambda task: -123
    def fail():
        raise OSError("error-info unavailable")
    dll.VkDaqGetLastErrorInfo.handler = fail
    with pytest.raises(VkDaqError) as failure:
        VkDaqClient(dll=dll).start_task("t")
    assert failure.value.code == -123
    assert "error-info unavailable" in failure.value.detail


def test_close_is_idempotent_and_disables_native_use():
    dll = FakeDLL()
    sdk = VkDaqClient(dll=dll)
    sdk.close()
    sdk.close()
    with pytest.raises(VkDaqError, match="closed"):
        sdk.create_task("t")
    assert not dll.calls


@pytest.mark.parametrize("minimum,maximum", [(-3, 3), (True, 1), (-10, float("inf"))])
def test_invalid_voltage_range_is_rejected_before_native_use(minimum, maximum):
    dll = FakeDLL()
    with pytest.raises(ValueError, match="range"):
        VkDaqClient(dll=dll).create_iepe_voltage_channel(
            "t", "dev/AIN1", range_min=minimum, range_max=maximum)
    assert not dll.calls


def test_native_error_keeps_code_operation_and_text():
    dll = FakeDLL()
    dll.error_text = "原始故障".encode()
    dll.VkDaqStartTask.handler = lambda task: -12345
    with pytest.raises(RuntimeError) as failure:
        VkDaqClient(dll=dll).start_task("task")
    assert (failure.value.operation, failure.value.code, failure.value.detail) == (
        "VkDaqStartTask", -12345, "原始故障")


@pytest.mark.parametrize("actual", ["51200", "51200.000", "5.12e4"])
def test_exact_sample_rate_after_start(actual):
    dll = FakeDLL()
    dll.VkDaqGetDeviceAttribute.handler = lambda dev, attr, buf, size: dll.write_text(buf, actual) or 0
    sdk = VkDaqClient(dll=dll)
    sdk.start_task("task")
    assert sdk.verify_actual_sample_rate("dev", 51200) == 51200
    assert [entry[0] for entry in dll.calls] == ["VkDaqStartTask", "VkDaqGetDeviceAttribute"]
    assert dll.calls[-1][1][1] == b"SamplingFrequency"


@pytest.mark.parametrize("actual", ["51199.999999999999999", "51200.000000000000001", "NaN", "inf", "", "51_200"])
def test_sample_rate_mismatch_is_never_rounded(actual):
    dll = FakeDLL()
    dll.VkDaqGetDeviceAttribute.handler = lambda dev, attr, buf, size: dll.write_text(buf, actual) or 0
    with pytest.raises(RuntimeError, match="exactly 51200"):
        VkDaqClient(dll=dll).verify_actual_sample_rate("dev", 51200)


@pytest.mark.parametrize("channels,requested,count", [(3, 5, 2), (32, 5, 2), (16, 4, 0)])
def test_owned_nan_buffer_scan_interleaved_prefix(channels, requested, count):
    dll = FakeDLL()
    expected = [float(frame * 100 + channel) for frame in range(count) for channel in range(channels)]
    def read(task, buffer, frames, mode, timeout):
        assert len(buffer) == requested * channels
        assert all(math.isnan(value) for value in buffer)
        assert (frames, mode, timeout) == (requested, 0, .2)
        buffer[:len(expected)] = expected
        return count
    dll.VkDaqGetTaskData.handler = read
    sdk = VkDaqClient(dll=dll)
    buffer, actual = sdk.read_task_data("t", channel_count=channels,
                                      samples_per_channel=requested, timeout_seconds=.2)
    sdk.close()
    assert actual == count
    assert isinstance(buffer, ctypes.Array) and buffer._type_ is ctypes.c_double
    assert list(buffer[:count * channels]) == expected
    assert all(math.isnan(value) for value in buffer[count * channels:])


@pytest.mark.parametrize("count", [-7, 6, 2**31, True])
def test_invalid_native_read_count(count):
    dll = FakeDLL()
    dll.VkDaqGetTaskData.handler = lambda *args: count
    with pytest.raises(RuntimeError):
        VkDaqClient(dll=dll).read_task_data("t", channel_count=3, samples_per_channel=5, timeout_seconds=.1)


@pytest.mark.parametrize("channel_count,samples,timeout", [
    (True, 5, .1), (0, 5, .1), (2**31, 5, .1), (3, True, .1),
    (3, 0, .1), (3, 2**31, .1), (2**30, 4, .1),
    (3, 5, float("nan")), (3, 5, float("inf")), (3, 5, .21), (3, 5, True),
    (3, 5, 2**2000),
])
def test_read_validates_counts_capacity_and_timeout_before_native_calls(channel_count, samples, timeout):
    dll = FakeDLL()
    with pytest.raises(ValueError):
        VkDaqClient(dll=dll).read_task_data("t", channel_count=channel_count,
                                          samples_per_channel=samples, timeout_seconds=timeout)
    assert not dll.calls


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_unwritten_or_nonfinite_valid_prefix_is_rejected(value):
    dll = FakeDLL()
    def read(task, buffer, *args):
        if value is not None:
            buffer[0] = value
        return 1
    dll.VkDaqGetTaskData.handler = read
    with pytest.raises(RuntimeError, match="finite"):
        VkDaqClient(dll=dll).read_task_data("t", channel_count=1, samples_per_channel=5, timeout_seconds=.1)
