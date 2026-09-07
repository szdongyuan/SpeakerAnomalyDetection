"""Native ABI contracts tested only with in-memory fakes, never hardware."""
import ctypes
import importlib
import importlib.util
import os
from pathlib import Path
import threading

import pytest

from unit_test.base.ve3668n_fakes import FakeVkDaqDLL


def forbidden_native_io(*args, **kwargs):
    pytest.fail("unexpected disk access or native DLL loading")


@pytest.fixture(autouse=True)
def block_real_sdk(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", forbidden_native_io)
    monkeypatch.setattr(ctypes, "WinDLL", forbidden_native_io)
    monkeypatch.setattr(os, "add_dll_directory", forbidden_native_io)


def test_import_does_not_load_sdk(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", forbidden_native_io)
    monkeypatch.setattr(ctypes, "WinDLL", forbidden_native_io)
    monkeypatch.setattr(os, "add_dll_directory", forbidden_native_io)
    monkeypatch.setattr(Path, "is_file", forbidden_native_io)
    assert importlib.util.find_spec("base.vkinging_sdk") is not None
    sdk = importlib.reload(importlib.import_module("base.vkinging_sdk"))
    sdk.VkDaqClient()
    sdk.VkDaqClient(dll_path="missing/libvkdaq.dll")
    assert "vkinging_daq" not in sdk.__dict__


def test_iepe_voltage_call_uses_fixed_native_arguments(monkeypatch):
    from base.vkinging_sdk import VkDaqClient

    monkeypatch.setattr(ctypes, "CDLL", forbidden_native_io)
    monkeypatch.setattr(os, "add_dll_directory", forbidden_native_io)
    monkeypatch.setattr(Path, "is_file", forbidden_native_io)
    fake = FakeVkDaqDLL()
    client = VkDaqClient(dll=fake)
    assert fake.trace == []
    assert client.create_iepe_voltage_channel("task", "Dev1/AIN8,Dev1/AIN2") == 0
    assert fake.trace == [{
        "operation": "VkDaqCreateAIAccelChan",
        "args": (b"task", b"Dev1/AIN8,Dev1/AIN2", b"", 0, -10.0,
                 10.0, 4, 1000.0, 0, b""),
        "pid": os.getpid(), "thread_id": threading.get_ident(),
    }]
    with pytest.raises(TypeError, match="sensitivity"):
        client.create_iepe_voltage_channel("task", "Dev1/AIN1", sensitivity=1)
    assert len(fake.trace) == 1


def abi_signatures():
    char = ctypes.c_char_p
    output = ctypes.POINTER(ctypes.c_char)
    integer = ctypes.c_int32
    double = ctypes.c_double
    return {
        "VkDaqGetLastErrorInfo": ([], char),
        "VkDaqGetDevices": ([output, output, integer], integer),
        "VkDaqGetChannels": ([char, output, integer], integer),
        "VkDaqGetDeviceAttribute": ([char, char, output, integer], integer),
        "VkDaqCreateTask": ([char], integer),
        "VkDaqCreateAIAccelChan": (
            [char, char, char, integer, double, double, integer, double, integer, char],
            integer,
        ),
        "VkDaqCfgSampClkTiming": (
            [char, integer, double, integer, integer, integer], integer,
        ),
        "VkDaqStartTask": ([char], integer),
        "VkDaqStopTask": ([char], integer),
        "VkDaqClearTask": ([char], integer),
        "VkDaqGetTaskData": (
            [char, ctypes.POINTER(double), integer, integer, double], integer,
        ),
    }


@pytest.mark.parametrize("operation,signature", abi_signatures().items())
def test_every_abi_argument_and_result_matches_header(operation, signature):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    VkDaqClient(dll=fake).create_iepe_voltage_channel("task", "Dev1/AIN1")
    function = getattr(fake, operation)
    assert (function.argtypes, function.restype) == signature


@pytest.mark.parametrize("method,operation,args", [
    ("create_task", "VkDaqCreateTask", ("task-测试",)),
    ("start_task", "VkDaqStartTask", ("task-测试",)),
    ("stop_task", "VkDaqStopTask", ("task-测试",)),
    ("clear_task", "VkDaqClearTask", ("task-测试",)),
])
def test_task_calls_preserve_names_and_nonnegative_status(method, operation, args):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    getattr(fake, operation).return_value = 3
    assert getattr(VkDaqClient(dll=fake), method)(*args) == 3
    assert fake.trace[0]["args"] == ("task-测试".encode("utf-8"),)
    assert fake.trace[0]["operation"] == operation


@pytest.mark.parametrize("method,operation,args", [
    ("create_task", "VkDaqCreateTask", ("task",)),
    ("start_task", "VkDaqStartTask", ("task",)),
    ("stop_task", "VkDaqStopTask", ("task",)),
    ("clear_task", "VkDaqClearTask", ("task",)),
    ("create_iepe_voltage_channel", "VkDaqCreateAIAccelChan", ("task", "Dev1/AIN1")),
])
def test_negative_task_codes_preserve_operation_code_detail(method, operation, args):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    getattr(fake, operation).return_value = -42
    fake.VkDaqGetLastErrorInfo.return_value = "设备忙".encode("utf-8")
    with pytest.raises(VkDaqError) as caught:
        getattr(VkDaqClient(dll=fake), method)(*args)
    assert (caught.value.operation, caught.value.code, caught.value.detail) == (
        operation, -42, "设备忙",
    )
    assert operation in str(caught.value) and "-42" in str(caught.value)
    assert [entry["operation"] for entry in fake.trace] == [
        operation, "VkDaqGetLastErrorInfo",
    ]


@pytest.mark.parametrize("raw,diagnostic", [
    (None, "no error text"), (b"", "no error text"),
    (b"\xff", "invalid UTF-8"),
])
def test_unavailable_error_text_does_not_erase_native_code(raw, diagnostic):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.VkDaqStartTask.return_value = -7
    fake.VkDaqGetLastErrorInfo.return_value = raw
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll=fake).start_task("task")
    assert caught.value.operation == "VkDaqStartTask"
    assert caught.value.code == -7
    assert diagnostic in caught.value.detail


@pytest.mark.parametrize("value", [None, "", " ", "task\0other", b"task", 3])
def test_native_input_strings_reject_invalid_values_without_io(value):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    with pytest.raises(ValueError, match="task"):
        VkDaqClient(dll=fake).create_task(value)
    assert fake.trace == []


class DirectoryHandle:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def install_loader_fake(monkeypatch, existing):
    """Fake filesystem existence, DLL loading and dependency-directory handles."""
    fake = FakeVkDaqDLL()
    loaded = []
    handles = []
    monkeypatch.setattr(Path, "is_file", lambda path: path in existing)

    def add_directory(path):
        handle = DirectoryHandle()
        handles.append((Path(path), handle))
        return handle

    def load(path):
        loaded.append(Path(path))
        assert handles and handles[-1][1].close_count == 0
        return fake

    monkeypatch.setattr(os, "add_dll_directory", add_directory)
    monkeypatch.setattr(ctypes, "CDLL", load)
    return fake, loaded, handles


@pytest.mark.parametrize("choice", ["explicit", "home_arch", "home_direct", "vendor"])
def test_lazy_dll_path_selection_and_dependency_lifetime(monkeypatch, choice):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    arch = "x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
    home = Path("C:/fake-sdk-home")
    vendor = Path("C:/fake-program-files/VkDaqAssistant")
    paths = {
        "explicit": Path("C:/explicit-sdk/libvkdaq.dll"),
        "home_arch": home / arch / "libvkdaq.dll",
        "home_direct": home / "libvkdaq.dll",
        "vendor": vendor / arch / "libvkdaq.dll",
    }
    monkeypatch.setenv("VKDAQ_HOME", str(home))
    monkeypatch.setenv("ProgramFiles", str(vendor.parent))
    monkeypatch.setenv("ProgramFiles(x86)", str(vendor.parent))
    # A missing HOME is allowed to fall through to the vendor installation.
    existing = {paths[choice], paths["vendor"]}
    fake, loaded, handles = install_loader_fake(monkeypatch, existing)
    original_path = os.environ.get("PATH")
    client = VkDaqClient(dll_path=paths["explicit"] if choice == "explicit" else None)
    assert loaded == [] and handles == [] and fake.trace == []
    client.create_task("task")
    client.start_task("task")
    assert loaded == [paths[choice]]
    assert handles[0][0] == paths[choice].parent
    assert handles[0][1].close_count == 0
    client.close()
    client.close()
    assert handles[0][1].close_count == 1
    assert os.environ.get("PATH") == original_path
    with pytest.raises(VkDaqError, match="closed"):
        client.create_task("again")


@pytest.mark.parametrize("explicit", [True, False])
def test_missing_dll_has_actionable_diagnostics_and_no_load(monkeypatch, explicit):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    _, loaded, handles = install_loader_fake(monkeypatch, set())
    client = VkDaqClient(dll_path="C:/missing-sdk/libvkdaq.dll" if explicit else None)
    with pytest.raises(VkDaqError) as caught:
        client.create_task("task")
    assert caught.value.operation == "load_sdk"
    assert "libvkdaq.dll" in caught.value.detail
    assert "VKDAQ_HOME" in caught.value.detail
    assert f"{ctypes.sizeof(ctypes.c_void_p) * 8}-bit" in caught.value.detail
    assert loaded == [] and handles == []


@pytest.mark.parametrize("code,diagnostic", [(193, "bitness"), (126, "dependencies"), (5, "failed")])
def test_loader_errors_close_directory_and_preserve_cause(monkeypatch, code, diagnostic):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    path = Path("C:/fake-sdk/libvkdaq.dll")
    _, _, handles = install_loader_fake(monkeypatch, {path})
    error = OSError("loader failure detail")
    error.winerror = code

    def load_failure(path):
        raise error

    monkeypatch.setattr(ctypes, "CDLL", load_failure)
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll_path=path).create_task("task")
    assert caught.value.operation == "load_sdk"
    assert caught.value.code == code
    assert diagnostic in caught.value.detail
    assert "loader failure detail" in caught.value.detail
    assert caught.value.__cause__ is error
    assert handles[0][1].close_count == 1


@pytest.mark.parametrize("symbol", abi_signatures())
def test_missing_symbols_fail_before_native_use_and_close_directory(monkeypatch, symbol):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    path = Path("C:/fake-sdk/libvkdaq.dll")
    fake, _, handles = install_loader_fake(monkeypatch, {path})
    delattr(fake, symbol)
    client = VkDaqClient(dll_path=path)
    with pytest.raises(VkDaqError) as caught:
        client.create_task("task")
    assert caught.value.operation == "bind_sdk"
    assert symbol in caught.value.detail
    assert fake.trace == []
    assert handles[0][1].close_count == 1
    client.close()
    assert handles[0][1].close_count == 1


def test_clients_and_context_cleanup_are_instance_local(monkeypatch):
    from base.vkinging_sdk import VkDaqClient

    first = FakeVkDaqDLL()
    second = FakeVkDaqDLL()
    monkeypatch.setattr(Path, "is_file", forbidden_native_io)
    with VkDaqClient(dll=first, dll_path="missing/libvkdaq.dll") as one:
        one.create_task("one")
        with VkDaqClient(dll=second) as two:
            two.create_task("two")
        one.start_task("one")
    assert [row["args"] for row in first.trace] == [(b"one",), (b"one",)]
    assert [row["args"] for row in second.trace] == [(b"two",)]


def text_operation(client, operation):
    if operation == "VkDaqGetDevices":
        return client.get_devices()
    if operation == "VkDaqGetChannels":
        return client.get_channels("Dev1")
    return client.get_device_attribute("Dev1", "Model")


@pytest.mark.parametrize("operation", [
    "VkDaqGetDevices", "VkDaqGetChannels", "VkDaqGetDeviceAttribute",
])
def test_text_calls_have_writable_buffers_and_exact_arguments(operation):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    value = text_operation(VkDaqClient(dll=fake), operation)
    args = fake.trace[0]["args"]
    assert args[-1] == 4096
    if operation == "VkDaqGetDevices":
        assert value == (("192.0.2.1", "Dev1"),)
        buffers = args[:2]
    elif operation == "VkDaqGetChannels":
        assert value == ("Dev1/AIN1", "Dev1/AIN8")
        assert args[0] == b"Dev1"
        buffers = args[1:2]
    else:
        assert value == "VE3668N"
        assert args[:2] == (b"Dev1", b"Model")
        buffers = args[2:3]
    assert all(isinstance(buffer, ctypes.Array) and buffer._type_ is ctypes.c_char
               for buffer in buffers)


@pytest.mark.parametrize("operation", [
    "VkDaqGetDevices", "VkDaqGetChannels", "VkDaqGetDeviceAttribute",
])
def test_text_grows_only_on_native_buffer_too_small(operation):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    long_text = b"x" * 6000
    fake.devices = (long_text, b"Dev1")
    fake.channels = long_text
    fake.attributes[b"Model"] = long_text
    result = text_operation(VkDaqClient(dll=fake), operation)
    assert long_text.decode() in str(result)
    assert [row["args"][-1] for row in fake.trace] == [4096, 8192]


@pytest.mark.parametrize("operation", [
    "VkDaqGetDevices", "VkDaqGetChannels", "VkDaqGetDeviceAttribute",
])
@pytest.mark.parametrize("code", [-1, -10001, -2147483648, -10028])
def test_text_negative_returns_are_errors_with_bounded_growth(operation, code):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    function = getattr(fake, operation)
    function.side_effect = None
    function.return_value = code
    with pytest.raises(VkDaqError) as caught:
        text_operation(VkDaqClient(dll=fake), operation)
    assert caught.value.operation == operation
    assert caught.value.code == code
    assert "fake native failure" in caught.value.detail
    sizes = [row["args"][-1] for row in fake.trace if row["operation"] == operation]
    assert sizes == ([4096 * 2 ** exponent for exponent in range(9)]
                     if code == -10028 else [4096])
    assert fake.trace[-1]["operation"] == "VkDaqGetLastErrorInfo"


@pytest.mark.parametrize("output_index", [0, 1])
@pytest.mark.parametrize("payload", [None, b"unterminated", b"\xff\0"])
def test_device_text_rejects_unwritten_unterminated_or_invalid_utf8(output_index, payload):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()

    def write_bad_text(*args):
        for index, buffer in enumerate(args[:-1]):
            # An unwritten buffer must not be mistaken for an empty string.
            assert b"\0" not in bytes(buffer)
            if index != output_index:
                fake.write_text(buffer, args[-1], b"valid")
            elif payload is not None:
                ctypes.memmove(buffer, payload, len(payload))
        return 0

    fake.VkDaqGetDevices.side_effect = write_bad_text
    with pytest.raises(VkDaqError, match="UTF-8" if payload == b"\xff\0" else "NUL"):
        VkDaqClient(dll=fake).get_devices()
    assert len(fake.trace) == 1


@pytest.mark.parametrize("operation", ["VkDaqGetChannels", "VkDaqGetDeviceAttribute"])
@pytest.mark.parametrize("payload", [None, b"unterminated", b"\xff\0"])
def test_single_text_rejects_unwritten_unterminated_or_invalid_utf8(operation, payload):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()

    def write_bad_text(*args):
        buffer = args[-2]
        assert b"\0" not in bytes(buffer)
        if payload is not None:
            ctypes.memmove(buffer, payload, len(payload))
        return 0

    getattr(fake, operation).side_effect = write_bad_text
    with pytest.raises(VkDaqError, match="UTF-8" if payload == b"\xff\0" else "NUL"):
        text_operation(VkDaqClient(dll=fake), operation)
    assert len(fake.trace) == 1


@pytest.mark.parametrize("addresses,names", [
    (b"a,b", b"one"), (b"a", b"one,two"), (b"", b"one"),
    (b"a,", b"one,two"), (b",b", b"one,two"), (b"a, ,b", b"one,two,three"),
    (b"a,b", b"one,"), (b"a,b", b",two"), (b"a,b", b"one, \t"),
    (b"a\nb", b"one"), (b"a", b"one\x01two"),
])
def test_malformed_device_lists_fail_without_zip_truncation(addresses, names):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.devices = (addresses, names)
    with pytest.raises(VkDaqError, match="list"):
        VkDaqClient(dll=fake).get_devices()


@pytest.mark.parametrize("channels", [b",", b"AIN1,", b",AIN1", b"AIN1,,AIN2", b"AIN1, ", b"AIN1\nAIN2"])
def test_malformed_channel_lists_fail(channels):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.channels = channels
    with pytest.raises(VkDaqError, match="list"):
        VkDaqClient(dll=fake).get_channels("Dev1")


def test_empty_lists_and_trimmed_utf8_entries_are_unambiguous():
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    client = VkDaqClient(dll=fake)
    fake.devices = (b"", b"")
    fake.channels = b""
    assert client.get_devices() == ()
    assert client.get_channels("Dev1") == ()
    fake.devices = (b" addr1 , addr2 ", " 设备一 , 设备二 ".encode("utf-8"))
    fake.channels = b" Dev1/AIN8, Dev1/AIN2 "
    assert client.get_devices() == (("addr1", "设备一"), ("addr2", "设备二"))
    assert client.get_channels("Dev1") == ("Dev1/AIN8", "Dev1/AIN2")


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_clock_configuration_uses_continuous_vendor_arguments_without_readback(rate):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    assert VkDaqClient(dll=fake).configure_sample_clock("task", rate) == 0
    assert len(fake.trace) == 1
    assert fake.trace[0]["operation"] == "VkDaqCfgSampClkTiming"
    assert fake.trace[0]["args"] == (b"task", 0, float(rate), 1, 0, 0)
    assert type(fake.trace[0]["args"][2]) is float


@pytest.mark.parametrize("method,args", [
    ("configure_sample_clock", ("task",)),
    ("verify_actual_sample_rate", ("Dev1",)),
])
@pytest.mark.parametrize("rate", [
    None, True, False, 44100.0, "48000", 1, 44099, 44101, 47999, 48001,
    51199, 51201, 96000, 102400,
])
def test_rate_inputs_use_strict_existing_whitelist_before_io(method, args, rate):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    with pytest.raises(ValueError, match="sample_rate"):
        getattr(VkDaqClient(dll=fake), method)(*args, rate)
    assert fake.trace == []


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_actual_sampling_frequency_must_be_verified(rate):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    fake.attributes[b"SamplingFrequency"] = b"100000.000"

    def started(task):
        fake.attributes[b"SamplingFrequency"] = f"{rate}.000".encode()
        return 0

    fake.VkDaqStartTask.side_effect = started
    client = VkDaqClient(dll=fake)
    client.configure_sample_clock("task", rate)
    client.start_task("task")
    actual = client.verify_actual_sample_rate("Dev1", rate)
    assert actual == rate and type(actual) is int
    assert [row["operation"] for row in fake.trace] == [
        "VkDaqCfgSampClkTiming", "VkDaqStartTask", "VkDaqGetDeviceAttribute",
    ]
    assert fake.trace[-1]["args"][:2] == (b"Dev1", b"SamplingFrequency")


@pytest.mark.parametrize("raw", [
    b"", b" ", b"NaN", b"Infinity", b"-Infinity", b"51200Hz", b"0", b"-51200",
    b"51200.1", b"51199.999", b"51200.000000000000001", b"51199.999999999999999",
    b"51_200", b"48000.000", b"not a number",
])
def test_actual_rate_rejects_missing_nonfinite_rounding_and_mismatch(raw):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.attributes[b"SamplingFrequency"] = raw
    with pytest.raises(VkDaqError, match="SamplingFrequency") as caught:
        VkDaqClient(dll=fake).verify_actual_sample_rate("Dev1", 51200)
    assert "51200" in caught.value.detail
    assert len(fake.trace) == 1


@pytest.mark.parametrize("method,operation,first_arg", [
    ("configure_sample_clock", "VkDaqCfgSampClkTiming", "task"),
    ("verify_actual_sample_rate", "VkDaqGetDeviceAttribute", "Dev1"),
])
def test_configuration_or_rate_readback_negative_code_never_changes_rate(method, operation, first_arg):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    getattr(fake, operation).side_effect = None
    getattr(fake, operation).return_value = -55
    with pytest.raises(VkDaqError) as caught:
        getattr(VkDaqClient(dll=fake), method)(first_arg, 44100)
    assert (caught.value.operation, caught.value.code) == (operation, -55)
    assert caught.value.detail == "fake native failure"
    assert len(fake.trace) == 2


@pytest.mark.parametrize("actual", [0, 1, 2, 4])
@pytest.mark.parametrize("timeout", [0, 0.025, 0.2])
def test_read_returns_raw_owned_double_buffer_and_actual_count(actual, timeout):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    raw = [1.25, -2.5, 3.125, -4.75, 5.5, -6.0, 7.25, -8.5]

    def read(task, buffer, requested, mode, seconds):
        assert (task, requested, mode, seconds) == (b"task", 4, 1, timeout)
        assert type(seconds) is float
        assert isinstance(buffer, ctypes.Array)
        assert buffer._type_ is ctypes.c_double
        assert len(buffer) == 8
        buffer[:] = raw
        return actual

    fake.VkDaqGetTaskData.side_effect = read
    client = VkDaqClient(dll=fake)
    buffer, count = client.read_task_data(
        "task", channel_count=2, samples_per_channel=4, timeout_seconds=timeout,
    )
    assert count == actual and type(count) is int
    assert list(buffer) == raw
    assert len(fake.trace) == 1
    fake.VkDaqGetTaskData.side_effect = None
    second_buffer, second_count = client.read_task_data(
        "task", channel_count=2, samples_per_channel=4, timeout_seconds=timeout,
    )
    assert second_count == 0
    assert buffer is not second_buffer
    second_buffer[0] = 99
    client.close()
    assert list(buffer) == raw


@pytest.mark.parametrize("returned", [0, 1, 4])
def test_native_read_prefills_nan_before_call_and_preserves_owned_unwritten_tail(returned):
    import math
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    def read(task, buffer, requested, mode, seconds):
        assert all(math.isnan(value) for value in buffer)
        for index in range(returned * 2):
            buffer[index] = 8.25 if index < returned else 2.5
        return returned
    fake.VkDaqGetTaskData.side_effect = read
    client = VkDaqClient(dll=fake)
    buffer, count = client.read_task_data("task", channel_count=2, samples_per_channel=4,
                                          timeout_seconds=.2)
    client.close()
    assert count == returned
    assert all(math.isnan(value) for value in buffer[returned * 2:])
    assert list(buffer[:returned]) == [8.25] * returned
    assert list(buffer[returned:returned * 2]) == [2.5] * returned


@pytest.mark.parametrize("code", [-1, -10001, -10028, -2147483648])
def test_all_negative_reads_are_errors_without_retry_or_tail_access(code):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.VkDaqGetTaskData.return_value = code
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll=fake).read_task_data(
            "task", channel_count=2, samples_per_channel=4, timeout_seconds=0.1,
        )
    assert (caught.value.operation, caught.value.code, caught.value.detail) == (
        "VkDaqGetTaskData", code, "fake native failure",
    )
    assert [row["operation"] for row in fake.trace] == [
        "VkDaqGetTaskData", "VkDaqGetLastErrorInfo",
    ]


@pytest.mark.parametrize("count", [5, 2147483647, None, True, 1.5])
def test_invalid_actual_count_is_rejected_before_interpreting_samples(count):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.VkDaqGetTaskData.return_value = count
    with pytest.raises(VkDaqError, match="count|int32"):
        VkDaqClient(dll=fake).read_task_data(
            "task", channel_count=2, samples_per_channel=4, timeout_seconds=0.1,
        )
    assert len(fake.trace) == 1


@pytest.mark.parametrize("field,value", [
    *[("channel_count", value) for value in (None, 0, -1, True, 2.0, "2", 9)],
    *[("samples_per_channel", value) for value in (None, 0, -1, True, 4.0, "4", 2049, 2**31)],
    *[("timeout_seconds", value) for value in (None, -1, True, 0.201, float("nan"), float("inf"), "0.1")],
])
def test_read_arguments_are_bounded_and_validated_before_io(field, value):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    args = {"channel_count": 2, "samples_per_channel": 4, "timeout_seconds": 0.1}
    args[field] = value
    with pytest.raises(ValueError, match=field):
        VkDaqClient(dll=fake).read_task_data("task", **args)
    assert fake.trace == []


def test_read_does_not_interpret_nonfinite_values_or_assume_short_read_stride():
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()

    def read(task, buffer, requested, mode, seconds):
        buffer[:] = [2.5, float("nan"), float("inf"), -3.25]
        return 1

    fake.VkDaqGetTaskData.side_effect = read
    buffer, count = VkDaqClient(dll=fake).read_task_data(
        "task", channel_count=2, samples_per_channel=2, timeout_seconds=0.1,
    )
    assert count == 1 and len(buffer) == 4
    assert buffer[0] == 2.5 and buffer[1] != buffer[1]
    assert buffer[2] == float("inf") and buffer[3] == -3.25


def test_maximum_ve_read_buffer_has_space_for_every_channel():
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    fake.VkDaqGetTaskData.return_value = 2048
    buffer, count = VkDaqClient(dll=fake).read_task_data(
        "task", channel_count=8, samples_per_channel=2048, timeout_seconds=0.2,
    )
    assert len(buffer) == 8 * 2048 and count == 2048


@pytest.mark.parametrize("failure", [OSError("native boundary failure"), ctypes.ArgumentError("bad ABI")])
def test_native_call_exceptions_have_operation_context(failure):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()

    def fail(task):
        raise failure

    fake.VkDaqStartTask.side_effect = fail
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll=fake).start_task("task")
    assert caught.value.operation == "VkDaqStartTask"
    assert str(failure) in caught.value.detail
    assert caught.value.__cause__ is failure


def test_last_error_failure_preserves_original_native_failure():
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    fake = FakeVkDaqDLL()
    fake.VkDaqStartTask.return_value = -91

    def fail_error_info():
        raise OSError("error info unavailable")

    fake.VkDaqGetLastErrorInfo.side_effect = fail_error_info
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll=fake).start_task("task")
    assert (caught.value.operation, caught.value.code) == ("VkDaqStartTask", -91)
    assert "error info unavailable" in caught.value.detail


@pytest.mark.parametrize("during", ["close", "body", "load", "bind"])
def test_cleanup_failures_are_diagnosed_without_masking_first_failure(monkeypatch, during):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    path = Path("C:/fake-sdk/libvkdaq.dll")
    fake, _, handles = install_loader_fake(monkeypatch, {path})

    def fail_close():
        raise OSError("directory cleanup failed")

    original_loader = ctypes.CDLL

    def load(dll_path):
        result = original_loader(dll_path)
        handles[-1][1].close = fail_close
        if during == "load":
            raise OSError("first loader failure")
        if during == "bind":
            del result.VkDaqStartTask
        return result

    monkeypatch.setattr(ctypes, "CDLL", load)
    client = VkDaqClient(dll_path=path)
    if during == "body":
        with pytest.raises(ValueError, match="first body failure") as caught:
            with client:
                client.create_task("task")
                raise ValueError("first body failure")
        assert any("directory cleanup failed" in note for note in caught.value.__notes__)
    else:
        with pytest.raises(VkDaqError) as caught:
            client.create_task("task")
            client.close()
        if during == "close":
            assert caught.value.operation == "close_sdk"
            assert "directory cleanup failed" in caught.value.detail
        else:
            assert caught.value.operation == f"{during}_sdk"
            assert any("directory cleanup failed" in note for note in caught.value.__notes__)
    with pytest.raises(VkDaqError, match="closed"):
        client.create_task("after failure")
    # Retain ownership after a failed removal so callers can retry close.
    handles[-1][1].close = lambda: None
    client.close()


@pytest.mark.parametrize("boundary", ["existence", "directory"])
def test_filesystem_and_dependency_directory_errors_are_load_errors(monkeypatch, boundary):
    from base.vkinging_sdk import VkDaqClient, VkDaqError

    path = Path("C:/fake-sdk/libvkdaq.dll")
    _, loaded, _ = install_loader_fake(monkeypatch, {path})
    failure = PermissionError("permission denied for SDK path")

    def fail(*args):
        raise failure

    if boundary == "existence":
        monkeypatch.setattr(Path, "is_file", fail)
    else:
        monkeypatch.setattr(os, "add_dll_directory", fail)
    with pytest.raises(VkDaqError) as caught:
        VkDaqClient(dll_path=path).create_task("task")
    assert caught.value.operation == "load_sdk"
    assert "permission denied" in caught.value.detail
    assert caught.value.__cause__ is failure
    assert loaded == []


@pytest.mark.parametrize("bits,env_name,folder", [
    (64, "ProgramFiles", "x64"), (32, "ProgramFiles(x86)", "x86"),
])
def test_vendor_search_matches_python_bitness(monkeypatch, bits, env_name, folder):
    from base.vkinging_sdk import VkDaqClient

    path = Path("C:/fake-vendor/VkDaqAssistant") / folder / "libvkdaq.dll"
    monkeypatch.delenv("VKDAQ_HOME", raising=False)
    monkeypatch.setenv(env_name, "C:/fake-vendor")
    monkeypatch.setattr(ctypes, "sizeof", lambda kind: bits // 8)
    _, loaded, _ = install_loader_fake(monkeypatch, {path})
    with VkDaqClient() as client:
        client.create_task("task")
    assert loaded == [path]


@pytest.mark.parametrize("size", [4095, 4096, 1048575])
def test_text_terminator_fits_exact_capacity_including_one_mib(size):
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    fake.attributes[b"Model"] = b"x" * size
    assert VkDaqClient(dll=fake).get_device_attribute("Dev1", "Model") == "x" * size
    assert fake.trace[-1]["args"][-1] <= 1048576


def test_fake_trace_records_actual_caller_thread_and_pid():
    from base.vkinging_sdk import VkDaqClient

    fake = FakeVkDaqDLL()
    client = VkDaqClient(dll=fake)
    thread = threading.Thread(target=client.create_task, args=("worker-task",))
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert fake.trace[0]["thread_id"] == thread.ident
    assert fake.trace[0]["thread_id"] != threading.get_ident()
    assert fake.trace[0]["pid"] == os.getpid()
