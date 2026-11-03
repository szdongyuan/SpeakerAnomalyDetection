"""Injectable VkDaq doubles; no vendor DLL or hardware access."""
from collections import deque
import os
import threading


class FakeFunction:
    def __init__(self, owner, name, handler):
        self.owner, self.name, self.handler = owner, name, handler
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.owner.calls.append((self.name, args, threading.get_ident(), os.getpid()))
        return self.handler(*args)


class FakeDLL:
    """Python callable symbols with inspectable ctypes signatures and calls.

    Replace a symbol's ``handler`` to script return codes or buffer writes.
    Call records are (operation, arguments, thread_id, process_id).
    """
    def __init__(self):
        self.calls = []
        self.error_text = b"fake SDK error"
        for name in (
            "VkDaqGetDevices", "VkDaqGetChannels", "VkDaqGetDeviceAttribute",
            "VkDaqCreateTask", "VkDaqCreateAIAccelChan", "VkDaqCfgSampClkTiming",
            "VkDaqStartTask", "VkDaqStopTask", "VkDaqClearTask", "VkDaqGetTaskData",
        ):
            setattr(self, name, FakeFunction(self, name, lambda *args: 0))
        self.VkDaqGetLastErrorInfo = FakeFunction(
            self, "VkDaqGetLastErrorInfo", lambda: self.error_text)

    @staticmethod
    def write_text(buffer, text):
        buffer.value = text.encode("utf-8") if isinstance(text, str) else text


class FakeSDK:
    """Caller-owned SDK fake shared by discovery and later capture tasks.

    Devices are (address, current native name) pairs. Attributes are keyed by
    (name, attribute). Reads are (flat scan-interleaved values, frame_count).
    ``failures`` maps operations to exceptions, raised after recording the call.
    """
    def __init__(self, *, devices=(), attributes=None, channels=None, reads=()):
        self.devices = tuple(devices)
        self.attributes = dict(attributes or {})
        self.channels = dict(channels or {})
        self.reads = deque(reads)
        self.calls = []
        self.failures = {}
        self.closed = False

    def _record(self, name, *args):
        self.calls.append((name, args, threading.get_ident(), os.getpid()))
        if name in self.failures:
            raise self.failures[name]

    def get_devices(self):
        self._record("get_devices")
        return self.devices

    def get_device_attribute(self, name, attribute):
        self._record("get_device_attribute", name, attribute)
        return self.attributes[(name, attribute)]

    def get_channels(self, name):
        self._record("get_channels", name)
        return self.channels[name]

    def create_task(self, task):
        self._record("create_task", task)

    def start_task(self, task):
        self._record("start_task", task)

    def stop_task(self, task):
        self._record("stop_task", task)

    def clear_task(self, task):
        self._record("clear_task", task)

    def create_iepe_voltage_channel(self, task, physical_channels, **kwargs):
        self._record("create_iepe_voltage_channel", task, physical_channels, kwargs)

    def configure_sample_clock(self, task, sample_rate):
        self._record("configure_sample_clock", task, sample_rate)

    def verify_actual_sample_rate(self, device, requested_rate):
        self._record("verify_actual_sample_rate", device, requested_rate)
        return requested_rate

    def read_task_data(self, task, *, channel_count, samples_per_channel, timeout_seconds):
        import ctypes
        self._record("read_task_data", task, channel_count, samples_per_channel, timeout_seconds)
        values, count = self.reads.popleft()
        buffer = (ctypes.c_double * (channel_count * samples_per_channel))()
        buffer[:] = [float("nan")] * len(buffer)
        buffer[:len(values)] = values
        return buffer, count

    def close(self):
        self._record("close")
        self.closed = True
