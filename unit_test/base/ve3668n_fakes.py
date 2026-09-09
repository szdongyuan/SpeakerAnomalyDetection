"""Independent, importable VE test data; no SDK, hardware, or shared state."""
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import threading
import time


def input_config(sample_rate=51200, **overrides):
    config = {
        "sample_rate": sample_rate,
        "input_mode": "IEPE",
        "unit": "V",
        "range_min": -10.0,
        "range_max": 10.0,
    }
    config.update(overrides)
    return deepcopy(config)


def device_info(**overrides):
    device = {
        "backend": "vkinging",
        "model": "VE3668N",
        "machine_id": "test-machine-1",
        "name": "Dev1",
        "address": "192.0.2.1",
        "physical_channels": [7, 1],
        "max_input_channels": 8,
        "available": True,
        "input_config": input_config(),
    }
    device.update(overrides)
    return deepcopy(device)


def capture_request(path, **overrides):
    """Frozen request with file-local provenance; never accesses real stores."""
    from base.recording_process_protocol import RecordingRequest

    rate = overrides.get("sample_rate", 51200)
    device = device_info(input_config=input_config(rate))
    values = dict(request_id="ve-capture", purpose="main", sample_rate=rate,
                  target_samples=9, channels=(7, 1), device=device,
                  path=str(path), streaming=False, trim_samples=2,
                  monitor={}, calibration_metadata=None,
                  validation_thresholds={"enabled": False})
    values.update(overrides)
    return RecordingRequest(**values)


class NativeFunction:
    """Writable ctypes signature plus deterministic calls; never loads a DLL."""

    def __init__(self, name, trace, return_value=0, side_effect=None):
        self.name = name
        self.trace = trace
        self.argtypes = None
        self.restype = None
        self.return_value = return_value
        self.side_effect = side_effect

    def __call__(self, *args):
        self.trace.append({
            "operation": self.name, "args": args,
            "pid": os.getpid(), "thread_id": threading.get_ident(),
        })
        if self.side_effect is not None:
            return self.side_effect(*args)
        return self.return_value


class FakeVkDaqDLL:
    """Instance-local native-function fake shared by binding/capture tests."""

    def __init__(self):
        self.trace = []
        for name in (
            "VkDaqGetLastErrorInfo", "VkDaqGetDevices", "VkDaqGetChannels",
            "VkDaqGetDeviceAttribute", "VkDaqCreateTask",
            "VkDaqCreateAIAccelChan", "VkDaqCfgSampClkTiming",
            "VkDaqStartTask", "VkDaqStopTask", "VkDaqClearTask",
            "VkDaqGetTaskData",
        ):
            setattr(self, name, NativeFunction(name, self.trace))
        self.VkDaqGetLastErrorInfo.return_value = b"fake native failure"
        self.devices = (b"192.0.2.1", b"Dev1")
        self.channels = b"Dev1/AIN1,Dev1/AIN8"
        self.attributes = {
            b"SamplingFrequency": b"51200.000", b"Model": b"VE3668N",
            b"MachineId": b"test-machine-1",
        }
        self.VkDaqGetDevices.side_effect = self._get_devices
        self.VkDaqGetChannels.side_effect = self._get_channels
        self.VkDaqGetDeviceAttribute.side_effect = self._get_attribute

    @staticmethod
    def write_text(buffer, size, payload):
        if len(payload) + 1 > size:
            return -10028
        ctypes.memmove(buffer, payload + b"\0", len(payload) + 1)
        return 0

    def _get_devices(self, addresses, names, size):
        if any(len(value) + 1 > size for value in self.devices):
            return -10028
        self.write_text(addresses, size, self.devices[0])
        self.write_text(names, size, self.devices[1])
        return 0

    def _get_channels(self, device, buffer, size):
        return self.write_text(buffer, size, self.channels)

    def _get_attribute(self, device, attribute, buffer, size):
        return self.write_text(buffer, size, self.attributes[attribute])


def discovery_record(name="Dev1", machine_id="machine-1", model="VE3668N",
                     address="192.0.2.1", channels=None, status="ready"):
    return {
        "name": name, "address": address, "Model": model, "MachineId": machine_id,
        "DeviceStatus": status,
        "channels": tuple(f"{name}/AIN{i}" for i in range(1, 9))
        if channels is None else channels,
    }


class DiscoverySDK:
    """High-level read-only SDK fake; aliases deliberately cannot use addresses."""

    def __init__(self, records=None):
        self.records = [discovery_record()] if records is None else records
        self.trace = []
        self.closed = False

    def _value(self, name, field):
        self.trace.append((field, name))
        matches = [record for record in self.records if record["name"] == name]
        if len(matches) != 1:
            raise AssertionError("ambiguous alias or undocumented address query")
        value = matches[0][field]
        if isinstance(value, Exception):
            raise value
        return value

    def get_devices(self):
        self.trace.append(("devices",))
        return tuple((record["address"], record["name"]) for record in self.records)

    def get_device_attribute(self, name, attribute):
        assert attribute != "SamplingStatus"
        return self._value(name, attribute)

    def get_channels(self, name):
        return self._value(name, "channels")

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def discovery_trace(path, stage):
    if path is not None:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "pid": os.getpid()}) + "\n")


class DiscoveryClock:
    """Parent-only clock: trigger deadlines only after the intended fake phase."""

    def __init__(self):
        self._value = 100.0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._value

    def advance(self, seconds):
        with self._lock:
            self._value += seconds


class CaptureSDK:
    """High-level injected SDK, compact GROUP_BY_CHANNEL short reads only.

    This models the chosen contract, not evidence of the vendor's short stride.
    Hooks run on the real caller thread; events can pause any native boundary.
    """

    def __init__(self, *, counts=(), failures=(), hooks=None, clock=None):
        self.trace = []
        self._trace_lock = threading.Lock()
        self.counts = iter(counts)
        self.failures = set(failures)
        self.hooks = {} if hooks is None else hooks
        self.clock = clock
        self.rate = None
        self.record = discovery_record(name="FreshDev", machine_id="test-machine-1")
        self.closed = False
        self._values_override = None
        self.values_by_physical_channel = {1: 2.5, 3: -4.75, 7: 8.25}
        self._selected_physical_channels = (7, 1)
        self.layout = "compact"

    @property
    def values(self):
        """Legacy positional values, overridden only when a test assigns them."""
        return ((8.25, 2.5) if self._values_override is None
                else self._values_override)

    @values.setter
    def values(self, values):
        self._values_override = tuple(values)

    def _call(self, operation, *args, **kwargs):
        with self._trace_lock:
            self.trace.append(dict(operation=operation, args=args, kwargs=kwargs,
                                   pid=os.getpid(), thread_id=threading.get_ident()))
        if operation in self.hooks:
            self.hooks[operation](*args, **kwargs)
        if operation in self.failures:
            from base.vkinging_sdk import VkDaqError
            raise VkDaqError(operation, -17, "injected " + operation)

    def get_devices(self):
        self._call("get_devices")
        return ((self.record["address"], self.record["name"]),)

    def get_device_attribute(self, name, attribute):
        self._call("get_device_attribute", name, attribute)
        assert name == self.record["name"]
        assert attribute in ("MachineId", "Model", "DeviceStatus")
        return self.record[attribute]

    def get_channels(self, name):
        self._call("get_channels", name)
        assert name == self.record["name"]
        return self.record["channels"]

    def create_task(self, task):
        self._call("create_task", task)

    def create_iepe_voltage_channel(self, task, channels):
        self._call("create_iepe_voltage_channel", task, channels)
        self._selected_physical_channels = tuple(
            int(route.rsplit("AIN", 1)[1]) - 1 for route in channels.split(",")
        )

    def configure_sample_clock(self, task, rate):
        self._call("configure_sample_clock", task, rate)
        self.rate = rate

    def start_task(self, task):
        self._call("start_task", task)

    def verify_actual_sample_rate(self, name, requested):
        self._call("verify_actual_sample_rate", name, requested)
        assert name == self.record["name"]
        assert requested == self.rate
        return self.rate

    def read_task_data(self, task, *, channel_count, samples_per_channel, timeout_seconds):
        self._call("read_task_data", task, channel_count=channel_count,
                   samples_per_channel=samples_per_channel, timeout_seconds=timeout_seconds)
        count = next(self.counts, samples_per_channel)
        buffer = (ctypes.c_double * (channel_count * samples_per_channel))()
        buffer[:] = [float("nan")] * len(buffer)
        if type(count) is int and 0 < count <= samples_per_channel:
            stride = samples_per_channel if self.layout == "capacity" else count
            for channel in range(channel_count):
                physical = self._selected_physical_channels[channel]
                value = (self._values_override[channel]
                         if self._values_override is not None
                         else self.values_by_physical_channel.get(
                             physical, float(physical) + .25))
                for frame in range(count):
                    buffer[channel * stride + frame] = value
        return buffer, count

    def stop_task(self, task):
        self._call("stop_task", task)

    def clear_task(self, task):
        self._call("clear_task", task)

    def close(self):
        self._call("close")
        self.closed = True

    def calls(self, operation):
        """Return a thread-safe operation count without touching native state."""
        with self._trace_lock:
            return sum(item["operation"] == operation for item in self.trace)

    @property
    def owner_thread_ids(self):
        """Snapshot all thread identities that crossed the fake SDK boundary."""
        with self._trace_lock:
            return {item["thread_id"] for item in self.trace}

    def operations(self):
        """Return an immutable, ordered operation-name snapshot."""
        with self._trace_lock:
            return tuple(item["operation"] for item in self.trace)


def capture_dependencies(*, trace_path, read_delay=.06, block_operation=None,
                         crash_operation=None, fail_operation=None, first_only=False,
                         broken_preview=False, preview_fault=None, zero_reads=False):
    """Spawn uses the real capture/owner/writer, replacing only the SDK boundary.

    Native and parent clocks are both time.monotonic, never DiscoveryClock.
    Options crossing spawn are frozen scalars and paths, not native objects.
    """
    first = not Path(trace_path).exists()
    lock = threading.Lock()

    def record(operation, **details):
        with lock, open(trace_path, "a", encoding="utf-8") as trace:
            trace.write(json.dumps(dict(
                operation=operation,
                pid=os.getpid(),
                thread_id=threading.get_ident(),
                thread_name=threading.current_thread().name,
                at=time.monotonic(),
                **details,
            )) + "\n")

    preview_fault = preview_fault or ("append" if broken_preview else None)
    if preview_fault is not None:
        if preview_fault not in ("construct", "begin", "append", "snapshot"):
            raise ValueError("unknown injected preview fault")
        from base import recording_capture as recording_capture_module

        real_session = recording_capture_module.MultichannelWaveformSession

        class FaultyPreviewSession(real_session):
            def __init__(self, **kwargs):
                record(
                    "preview_session_construct",
                )
                if preview_fault == "construct":
                    record("preview_fault", phase=preview_fault)
                    raise RuntimeError("injected preview construction failure")
                super().__init__(**kwargs)

            def begin(self, **kwargs):
                if preview_fault == "begin":
                    record("preview_fault", phase=preview_fault)
                    raise RuntimeError("injected preview begin failure")
                return super().begin(**kwargs)

            def append(self, block):
                if preview_fault == "append":
                    record("preview_fault", phase=preview_fault)
                    raise RuntimeError("injected preview append failure")
                return super().append(block)

            def snapshots(self):
                if preview_fault == "snapshot":
                    record("preview_fault", phase=preview_fault)
                    raise RuntimeError("injected preview snapshot failure")
                return super().snapshots()

        recording_capture_module.MultichannelWaveformSession = FaultyPreviewSession

    class TracedSDK(CaptureSDK):
        def _call(self, operation, *args, **kwargs):
            record(operation)
            if first or not first_only:
                if operation == crash_operation:
                    os._exit(23)
                if operation == block_operation:
                    while not Path(trace_path + ".release").exists():
                        threading.Event().wait(.005)
                if operation == fail_operation:
                    raise RuntimeError("injected native " + operation)
            if operation == "read_task_data":
                threading.Event().wait(read_delay)
            super()._call(operation, *args, **kwargs)

        def read_task_data(self, *args, **kwargs):
            buffer, count = super().read_task_data(*args, **kwargs)
            return buffer, 0 if zero_reads else count

    return {"ve_sdk_factory": TracedSDK}


def blocked_control_worker(control, preview, generation, backend_factory, backend_options,
                           cancel_timeout, preview_interval):
    """Hold the real worker's send boundary as a full pipe would; not its owner."""
    from base.recording_worker import recording_worker

    class Connection:
        def __getattr__(self, name):
            return getattr(control, name)

        def send(self, event):
            if event.kind == "started":
                gate = Path(backend_options["trace_path"] + ".ipc-release")
                while not gate.exists():
                    threading.Event().wait(.005)
            control.send(event)

    recording_worker(Connection(), preview, generation, backend_factory, backend_options,
                     cancel_timeout, preview_interval)


def capture_orphan_parent(connection, options):
    """Disposable parent of the actual recording worker with a hung fake SDK."""
    from base.recording_service import RecordingService

    service = RecordingService(backend_factory="unit_test.base.ve3668n_fakes:capture_dependencies",
                               backend_options=options, cancel_timeout=.3)
    service.start(capture_request(Path(options["trace_path"]).with_suffix(".wav")))
    path = Path(options["trace_path"])
    awaited = (options["block_operation"] if options["block_operation"] == "read_task_data"
               else "start_task")
    while True:
        if path.exists():
            lines = path.read_text().rpartition("\n")[0].splitlines()
            if any(json.loads(line)["operation"] == awaited for line in lines):
                connection.send(service.worker_pid)
                connection.close()
                break
        threading.Event().wait(.01)
    threading.Event().wait()


def discovery_factory(*, trace_path=None, mode="normal"):
    """Importable spawn factory; no SDK/DLL even in the unpatched child."""
    first = trace_path is None or not Path(trace_path).exists()
    discovery_trace(trace_path, "factory")
    if mode == "block_factory":
        threading.Event().wait()
    if mode == "huge_error":
        raise RuntimeError("fake-error-" * 20000)
    if mode.startswith("wire_"):
        from base import ve3668n_discovery
        payloads = {
            "wire_json": b"not-json",
            "wire_shape": b"{}",
            "wire_diagnostic": b'{"devices":[],"diagnostics":[null]}',
            "wire_audio": b'{"devices":[],"diagnostics":[],"audio":[1]}',
            "wire_duplicate": b'{"devices":[],"devices":[],"diagnostics":[]}',
            "wire_oversize": b"x" * 131072,
            "wire_nested": b"[" * 1500 + b"]" * 1500,
        }
        ve3668n_discovery._encode_result = lambda result, limit: payloads[mode]

    class TracedSDK(DiscoverySDK):
        def get_devices(self):
            discovery_trace(trace_path, "probe")
            if mode == "crash":
                os._exit(23)
            if mode == "block_probe" or (mode == "block_once" and first):
                threading.Event().wait()
            if mode == "probe_and_close_error":
                from base.vkinging_sdk import VkDaqError
                raise VkDaqError("get_devices", -17, "fake primary probe failure")
            return super().get_devices()

        def close(self):
            if mode == "block_close":
                discovery_trace(trace_path, "closing")
                threading.Event().wait()
            if mode in ("close_error", "probe_and_close_error"):
                raise RuntimeError("fake SDK close failure")
            super().close()
            discovery_trace(trace_path, "closed")

    records = None
    if mode == "oversize":
        records = [discovery_record(machine_id="x" * 131072)]
    elif mode == "empty":
        records = []
    return TracedSDK(records)


def discovery_orphan_parent(connection, options):
    """Test process whose OS death must reap its blocked discovery helper."""
    from base.ve3668n_discovery import DiscoveryService

    service = DiscoveryService(sdk_factory="unit_test.base.ve3668n_fakes.discovery_factory",
                               sdk_options=options, deadline=30)
    service.start()
    trace = Path(options["trace_path"])
    stage = {"block_factory": "factory", "block_probe": "probe", "block_close": "closing"}[options["mode"]]
    while True:
        if trace.exists():
            lines = trace.read_text().rpartition("\n")[0].splitlines()
            matching = [json.loads(line) for line in lines if f'"stage": "{stage}"' in line]
            if matching:
                connection.send(matching[0]["pid"])
                break
        threading.Event().wait(.01)
    connection.close()
    threading.Event().wait()
