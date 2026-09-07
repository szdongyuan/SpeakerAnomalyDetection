from __future__ import annotations

import ctypes

import pytest

import vkinging_daq


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


def _write_text(buffer, payload: bytes) -> None:
    ctypes.memmove(buffer, payload, len(payload))


class FakeNativeLibrary:
    def __init__(self) -> None:
        self.last_error = b"fake native diagnostic"
        self.attribute_values = {
            (b"Dev1", b"MachineId"): "MID-001".encode(),
            (b"Dev1", b"Model"): "VE3668N".encode(),
        }
        self.attribute_error: int | None = None
        self.attribute_sizes: list[int] = []
        self.task_values = [0.25, -0.5]

        self.VkDaqGetLastErrorInfo = FakeFunction(lambda: self.last_error)
        self.VkDaqGetDevices = FakeFunction(self._get_devices)
        self.VkDaqGetChannels = FakeFunction(self._get_channels)
        self.VkDaqGetDeviceAttribute = FakeFunction(self._get_device_attribute)
        self.VkDaqCreateTask = FakeFunction(lambda task: 0)
        self.VkDaqCreateAIVoltageChan = FakeFunction(lambda *args: 0)
        self.VkDaqCreateAIAccelChan = FakeFunction(lambda *args: 0)
        self.VkDaqCfgSampClkTiming = FakeFunction(lambda *args: 0)
        self.VkDaqStartTask = FakeFunction(lambda task: 0)
        self.VkDaqGetTaskData = FakeFunction(self._get_task_data)
        self.VkDaqStopTask = FakeFunction(lambda task: 0)
        self.VkDaqClearTask = FakeFunction(lambda task: 0)

    @staticmethod
    def _get_devices(addresses, names, size):
        _write_text(addresses, b"USB0\0")
        _write_text(names, b"Dev1\0")
        return 0

    @staticmethod
    def _get_channels(device, channels, size):
        _write_text(channels, b"AIN1\0")
        return 0

    def _get_device_attribute(self, device, attribute, output, size):
        self.attribute_sizes.append(size)
        if self.attribute_error is not None:
            return self.attribute_error
        payload = self.attribute_values[(device, attribute)] + b"\0"
        if len(payload) > size:
            _write_text(output, b"stale\0")
            return vkinging_daq.BUFFER_TOO_SMALL
        _write_text(output, payload)
        return 0

    def _get_task_data(self, task, output, requested, grouping, timeout):
        actual = min(len(self.task_values), requested)
        for index, value in enumerate(self.task_values[:actual]):
            output[index] = value
        return actual


def test_device_attribute_accepts_current_device_info_and_name() -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        device = client.list_devices()[0]
        assert client.get_device_attribute(device, "MachineId") == "MID-001"
        assert client.get_device_attribute(device.name, "Model") == "VE3668N"

    call = fake.VkDaqGetDeviceAttribute.calls[0]
    assert call[:2] == (b"Dev1", b"MachineId")
    assert isinstance(call[2], ctypes.Array)
    assert call[2]._type_ is ctypes.c_char


def test_device_attribute_decodes_utf8_and_grows_only_for_buffer_too_small() -> None:
    fake = FakeNativeLibrary()
    value = "型号-" + "x" * 5000
    fake.attribute_values[(b"Dev1", b"Model")] = value.encode("utf-8")

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        assert client.get_device_attribute(0, "Model") == value

    assert fake.attribute_sizes == [4096, 8192]


def test_device_attribute_native_error_never_returns_stale_buffer_data() -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        assert client.get_device_attribute("Dev1", "Model") == "VE3668N"
        fake.attribute_error = -123
        with pytest.raises(vkinging_daq.VkDaqError) as caught:
            client.get_device_attribute("Dev1", "Model")

    assert caught.value.operation == "VkDaqGetDeviceAttribute"
    assert caught.value.code == -123
    assert "fake native diagnostic" in str(caught.value)
    assert fake.attribute_sizes == [4096, 4096]


def test_device_attribute_rejects_stale_or_unknown_device_info() -> None:
    fake = FakeNativeLibrary()
    stale = vkinging_daq.DeviceInfo(index=0, address="old", name="Dev1")

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        with pytest.raises(ValueError, match="stale|unavailable"):
            client.get_device_attribute(stale, "Model")
        with pytest.raises(ValueError, match="No SDK device"):
            client.get_device_attribute("missing", "Model")

    assert fake.VkDaqGetDeviceAttribute.calls == []


@pytest.mark.parametrize("attribute", ["", "Machine\0Id"])
def test_device_attribute_rejects_empty_or_embedded_nul_attribute(attribute) -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        with pytest.raises(ValueError, match="attribute"):
            client.get_device_attribute(0, attribute)

    assert fake.VkDaqGetDeviceAttribute.calls == []


def test_device_attribute_rejects_calls_after_close() -> None:
    fake = FakeNativeLibrary()
    client = vkinging_daq.VkDaqClient(native_library=fake)
    client.close()

    with pytest.raises(vkinging_daq.VkDaqStateError, match="closed"):
        client.get_device_attribute(0, "Model")

    assert fake.VkDaqGetDeviceAttribute.calls == []


def _acquire(
    client,
    *,
    mode="iepe_voltage",
    terminal="single_ended",
    sensitivity=None,
    min_value=-10.0,
    max_value=10.0,
):
    return client.acquire(
        channels=["AIN1"],
        mode=mode,
        sample_rate=51_200,
        samples_per_channel=2,
        min_value=min_value,
        max_value=max_value,
        terminal=terminal,
        timeout=1.0,
        sensitivity=sensitivity,
    )


def _stream(
    client,
    *,
    mode="iepe_voltage",
    terminal="single_ended",
    sensitivity=None,
    min_value=-10.0,
    max_value=10.0,
):
    return client.stream(
        channels=["AIN1"],
        mode=mode,
        sample_rate=51_200,
        samples_per_chunk=2,
        min_value=min_value,
        max_value=max_value,
        terminal=terminal,
        timeout=1.0,
        sensitivity=sensitivity,
    )


def test_iepe_voltage_acquire_uses_voltage_return_abi_and_preserves_raw_volts() -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        result = _acquire(client)

    call = fake.VkDaqCreateAIAccelChan.calls[0]
    assert call[1:] == (
        b"Dev1/AIN1",
        b"",
        vkinging_daq.SINGLE_ENDED,
        -10.0,
        10.0,
        4,
        1000.0,
        0,
        b"",
    )
    assert result.mode == "iepe_voltage"
    assert result.samples == ((0.25, -0.5),)


def test_iepe_voltage_stream_uses_voltage_return_abi_and_preserves_raw_volts() -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        with _stream(client) as stream:
            chunk = next(stream)

    call = fake.VkDaqCreateAIAccelChan.calls[0]
    assert call[1:] == (
        b"Dev1/AIN1",
        b"",
        vkinging_daq.SINGLE_ENDED,
        -10.0,
        10.0,
        4,
        1000.0,
        0,
        b"",
    )
    assert chunk.mode == "iepe_voltage"
    assert chunk.samples == ((0.25, -0.5),)


@pytest.mark.parametrize("operation", ["acquire", "stream"])
def test_iepe_voltage_normalizes_caller_range_to_fixed_ten_volts(operation) -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        if operation == "acquire":
            _acquire(client, min_value=-2.5, max_value=3.5)
        else:
            with _stream(client, min_value=-2.5, max_value=3.5) as stream:
                next(stream)

    call = fake.VkDaqCreateAIAccelChan.calls[0]
    assert call[4:6] == (-10.0, 10.0)


@pytest.mark.parametrize("mode", ["voltage", "iepe"])
def test_existing_modes_preserve_caller_voltage_range(mode) -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        _acquire(
            client,
            mode=mode,
            sensitivity=123.5 if mode == "iepe" else None,
            min_value=-2.5,
            max_value=3.5,
        )

    if mode == "voltage":
        call = fake.VkDaqCreateAIVoltageChan.calls[0]
    else:
        call = fake.VkDaqCreateAIAccelChan.calls[0]
    assert call[4:6] == (-2.5, 3.5)


@pytest.mark.parametrize("operation", ["acquire", "stream"])
def test_iepe_voltage_rejects_differential_before_task_creation(operation) -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        with pytest.raises(ValueError, match="single-ended"):
            if operation == "acquire":
                _acquire(client, terminal="differential")
            else:
                _stream(client, terminal="differential")

    assert fake.VkDaqCreateTask.calls == []


def test_existing_iepe_keeps_acceleration_unit_and_caller_sensitivity() -> None:
    fake = FakeNativeLibrary()

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        result = _acquire(client, mode="iepe", sensitivity=123.5)

    call = fake.VkDaqCreateAIAccelChan.calls[0]
    assert call[6:9] == (vkinging_daq.ACCELERATION_G, 123.5, 0)
    assert result.mode == "iepe"
    assert result.samples == ((0.25, -0.5),)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("operation", ["acquire", "stream"])
def test_iepe_voltage_rejects_non_finite_raw_voltage_and_cleans_up(
    operation, bad_value
) -> None:
    fake = FakeNativeLibrary()
    fake.task_values = [0.25, bad_value]

    with vkinging_daq.VkDaqClient(native_library=fake) as client:
        client.select_device(0)
        with pytest.raises(vkinging_daq.VkDaqError, match="non-finite raw voltage"):
            if operation == "acquire":
                _acquire(client)
            else:
                with _stream(client) as stream:
                    next(stream)

    assert len(fake.VkDaqStopTask.calls) == 1
    assert len(fake.VkDaqClearTask.calls) == 1
