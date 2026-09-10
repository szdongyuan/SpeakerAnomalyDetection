import ctypes
import itertools
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import (
    RecordingResult,
    VeLifecycleCounts,
    VePrewarmRequest,
)
from base.ve3668n_discovery import discover_devices
from base.ve3668n_resource import (
    VeResourceConfigurationError,
    VeResourceController,
    VeResourceFault,
    VeResourceReleaseOutcome,
)
from base.vkinging_sdk import VkDaqError
from unit_test.base.ve3668n_fakes import (
    CaptureSDK,
    DiscoveryClock,
    capture_request,
    device_info,
    discovery_record,
    wav_metadata,
)


FAKE_PHYSICAL_CHANNELS = (1, 3, 7)
ALL_ORDERED_CHANNEL_CASES = tuple(
    order
    for size in range(1, len(FAKE_PHYSICAL_CHANNELS) + 1)
    for order in itertools.permutations(FAKE_PHYSICAL_CHANNELS, size)
)


def controller_for(sdk_factory, fatals=None, **options):
    fatals = [] if fatals is None else fatals
    bind_timeout = options.pop("bind_timeout", .25)
    detach_timeout = options.pop("detach_timeout", .25)
    controller = VeResourceController(
        sdk_factory=sdk_factory,
        fatal=lambda stage, message: fatals.append((stage, message)),
        bind_timeout=bind_timeout,
        detach_timeout=detach_timeout,
        **options,
    )
    return controller, fatals


def stream(controller, tmp_path, name="A", *, target=4, callback=None, **request_options):
    channels = request_options.get("channels", (7, 1))
    device = request_options.get("device")
    metadata = wav_metadata(("none", "none"), request_options.get("sample_rate", 51200))
    metadata["acquisition"]["machine_id"] = (
        device["machine_id"] if device is not None else "test-machine-1")
    by_physical = {entry["physical_input_channel"]: entry
                   for entry in metadata["recorded_channels"]}
    metadata["recorded_channels"] = []
    for index, physical in enumerate(channels):
        entry = by_physical[physical]
        entry["wav_channel_index"] = index
        metadata["recorded_channels"].append(entry)
    request = capture_request(
        tmp_path / f"{name}.wav", request_id=name, target_samples=target,
        trim_samples=0, calibration_metadata=metadata, **request_options,
    )
    blocks = []
    failures = []
    adapter = controller.stream(
        request=request,
        callback=(lambda block, frames, *_: blocks.append(block.copy()))
        if callback is None else callback,
        fail=lambda stage, message: failures.append((stage, message)),
        stop_event=threading.Event(),
    )
    return adapter, blocks, failures


def wait_released(adapter, timeout=2):
    deadline = time.monotonic() + timeout
    while not adapter.handles_released and time.monotonic() < deadline:
        time.sleep(.002)
    assert adapter.handles_released


def prewarm_request(warmup_id="warm-1", *, channels=(7, 1), sample_rate=44100):
    return VePrewarmRequest.create(
        warmup_id,
        device_info(input_config={
            "sample_rate": sample_rate,
            "input_mode": "IEPE",
            "unit": "V",
            "range_min": -10.0,
            "range_max": 10.0,
        }),
        channels,
        sample_rate,
        attempt=1,
    )


def wait_failed(controller, timeout=2):
    deadline = time.monotonic() + timeout
    while not controller.failed and time.monotonic() < deadline:
        time.sleep(.002)
    assert controller.failed


def test_prewarm_discards_partial_multichannel_reads_and_retains_task(tmp_path):
    sdk = CaptureSDK(
        counts=(1000, 2048, 17),
        hooks={"read_task_data": lambda *args, **kwargs: time.sleep(.001)},
    )
    controller, fatals = controller_for(lambda: sdk)
    request = prewarm_request()
    failures = []

    adapter = controller.prewarm(request=request, fail=lambda *item: failures.append(item))

    assert adapter.start()
    assert adapter.completed.wait(2)
    assert adapter.progress_snapshot.frames == request.frames_per_channel
    assert adapter.handles_released
    assert adapter.failure_snapshot is None
    assert failures == [] and fatals == []
    assert controller.signature == request.signature
    assert sdk.calls("start_task") == 1
    assert not hasattr(request, "path")
    assert list(tmp_path.iterdir()) == []

    recording, blocks, recording_failures = stream(
        controller,
        tmp_path,
        device=dict(request.device),
        channels=request.channels,
        sample_rate=request.sample_rate,
    )
    assert recording.start()
    wait_released(recording)
    assert blocks and recording_failures == []
    assert sdk.calls("start_task") == 1
    assert controller.release(.5).success


@pytest.mark.parametrize(
    ("buffer_factory", "returned", "expected_code"),
    [
        (lambda requested, channels: (ctypes.c_float * (requested * channels))(),
         1, None),
        (lambda requested, channels: (ctypes.c_double * (requested * channels - 1))(),
         1, None),
        (lambda requested, channels: (ctypes.c_double * (requested * channels))(),
         -1, -1),
        (lambda requested, channels: (ctypes.c_double * (requested * channels))(),
         lambda requested: requested + 1, 2049),
    ],
    ids=("wrong_scalar_type", "wrong_capacity", "negative_count", "oversized_count"),
)
def test_prewarm_rejects_malformed_native_buffer_before_progress(
        buffer_factory, returned, expected_code):
    class MalformedSDK(CaptureSDK):
        def read_task_data(self, task, *, channel_count, samples_per_channel,
                           timeout_seconds):
            self._call(
                "read_task_data", task, channel_count=channel_count,
                samples_per_channel=samples_per_channel,
                timeout_seconds=timeout_seconds,
            )
            count = returned(samples_per_channel) if callable(returned) else returned
            return buffer_factory(samples_per_channel, channel_count), count

    controller, _ = controller_for(lambda: MalformedSDK())
    failures = []
    adapter = controller.prewarm(
        request=prewarm_request(), fail=lambda *item: failures.append(item))

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert adapter.progress_snapshot.frames == 0
    assert len(failures) == 1
    assert adapter.failure_snapshot.stage == "read_task_data"
    assert adapter.failure_snapshot.code == expected_code
    assert not adapter.handles_released or controller.failed


def test_prewarm_rejects_short_scalar_payload_before_progress():
    class ShortPayloadSDK(CaptureSDK):
        def read_task_data(self, task, *, channel_count, samples_per_channel,
                           timeout_seconds):
            self._call(
                "read_task_data", task, channel_count=channel_count,
                samples_per_channel=samples_per_channel,
                timeout_seconds=timeout_seconds,
            )
            buffer = (ctypes.c_double * (samples_per_channel * channel_count))()
            buffer[:] = [float("nan")] * len(buffer)
            buffer[0] = 1.0
            return buffer, 1

    controller, _ = controller_for(lambda: ShortPayloadSDK())
    failures = []
    adapter = controller.prewarm(
        request=prewarm_request(), fail=lambda *item: failures.append(item))

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert adapter.progress_snapshot.frames == 0
    assert adapter.failure_snapshot == VeResourceFault(
        "read_task_data", None, "VkDaq returned non-finite voltage samples")
    assert len(failures) == 1


def test_prewarm_malformed_later_block_preserves_last_valid_frame_progress():
    class ValidThenMalformedSDK(CaptureSDK):
        def __init__(self):
            super().__init__(counts=(1,))
            self.reads = 0

        def read_task_data(self, task, *, channel_count, samples_per_channel,
                           timeout_seconds):
            self.reads += 1
            if self.reads == 1:
                return super().read_task_data(
                    task,
                    channel_count=channel_count,
                    samples_per_channel=samples_per_channel,
                    timeout_seconds=timeout_seconds,
                )
            self._call(
                "read_task_data", task, channel_count=channel_count,
                samples_per_channel=samples_per_channel,
                timeout_seconds=timeout_seconds,
            )
            return (ctypes.c_double * 1)(), 1

    controller, _ = controller_for(lambda: ValidThenMalformedSDK())
    adapter = controller.prewarm(request=prewarm_request(), fail=lambda *_: None)

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert adapter.progress_snapshot.frames == 1
    assert adapter.failure_snapshot.stage == "read_task_data"


def test_prewarm_preserves_native_read_code_and_detail():
    controller, _ = controller_for(
        lambda: CaptureSDK(failures=("read_task_data",)))
    adapter = controller.prewarm(request=prewarm_request(), fail=lambda *_: None)

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert adapter.failure_snapshot == VeResourceFault(
        "read_task_data", -17, "injected read_task_data")


@pytest.mark.parametrize("returned", [True, "1", 1.0, None])
def test_prewarm_noninteger_returned_frames_are_protocol_faults_without_code(
        returned):
    class NonIntegerCountSDK(CaptureSDK):
        def read_task_data(self, task, *, channel_count, samples_per_channel,
                           timeout_seconds):
            self._call(
                "read_task_data", task, channel_count=channel_count,
                samples_per_channel=samples_per_channel,
                timeout_seconds=timeout_seconds,
            )
            buffer = (ctypes.c_double * (samples_per_channel * channel_count))()
            return buffer, returned

    controller, _ = controller_for(lambda: NonIntegerCountSDK())
    adapter = controller.prewarm(request=prewarm_request(), fail=lambda *_: None)

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert adapter.progress_snapshot.frames == 0
    assert adapter.failure_snapshot.code is None
    assert adapter.failure_snapshot.stage == "read_task_data"
    assert "integer" in adapter.failure_snapshot.detail


@pytest.mark.parametrize(
    ("boundary", "offset", "expected_detail"),
    [
        ("no_progress", 0.0, "VE capture made no progress for 5 seconds"),
        ("no_progress", .001, "VE capture made no progress for 5 seconds"),
        ("total", 0.0, "VE capture total deadline exceeded before target frames"),
        ("total", .001, "VE capture total deadline exceeded before target frames"),
    ],
    ids=(
        "no_progress_exact", "no_progress_just_after",
        "total_exact", "total_just_after",
    ),
)
def test_prewarm_rejects_final_valid_block_at_or_after_capture_deadline(
        boundary, offset, expected_detail):
    clock = DiscoveryClock()
    calls = 0

    def advance_for_boundary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if boundary == "total" and calls == 10:
            clock.advance(1.0)
        if calls == 11:
            clock.advance((5.0 if boundary == "no_progress" else 4.5) + offset)

    sdk = CaptureSDK(
        counts=(2048,) * 10 + (1570,),
        hooks={"read_task_data": advance_for_boundary},
    )
    controller, _ = controller_for(lambda: sdk, clock=clock)
    request = prewarm_request()
    adapter = controller.prewarm(request=request, fail=lambda *_: None)

    assert adapter.start()
    assert adapter.completed.wait(2)
    wait_failed(controller)

    assert calls == 11
    assert adapter.progress_snapshot.frames == 20480
    assert adapter.progress_snapshot.frames < request.frames_per_channel
    assert adapter.failure_snapshot == VeResourceFault(
        "read_task_data", None, expected_detail)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", None, "detail"), "stage"),
        (("   ", None, "detail"), "stage"),
        (("read_task_data", True, "detail"), "code"),
        (("read_task_data", "-17", "detail"), "code"),
        (("read_task_data", -17, ""), "detail"),
        (("read_task_data", -17, "   "), "detail"),
        (("read_task_data", -17, None), "detail"),
    ],
)
def test_ve_resource_fault_rejects_worker_incompatible_shape(args, message):
    with pytest.raises(ValueError, match=message):
        VeResourceFault(*args)


def test_ve_resource_fault_accepts_native_integer_code_and_no_code():
    assert VeResourceFault("start_task", -12001, "native failure").code == -12001
    assert VeResourceFault("read_task_data", None, "protocol failure").code is None


@pytest.mark.parametrize("native", [False, True], ids=("runtime", "native"))
def test_prewarm_normalizes_empty_initialization_exception_once(native):
    class EmptyStartFailureSDK(CaptureSDK):
        def start_task(self, task):
            self._call("start_task", task)
            if native:
                raise VkDaqError("VkDaqStartTask", -12001, "   ")
            raise RuntimeError()

    controller, fatals = controller_for(lambda: EmptyStartFailureSDK())
    failures = []
    adapter = controller.prewarm(
        request=prewarm_request(), fail=lambda *item: failures.append(item))

    assert adapter.start() is False
    assert adapter.completed.wait(2)
    assert controller.failed

    expected = VeResourceFault(
        "start_task", -12001 if native else None,
        "VkDaqError" if native else "RuntimeError",
    )
    assert adapter.failure_snapshot == expected
    assert controller.failure_snapshot == expected
    assert len(failures) == len(fatals) == 1
    assert failures[0][0] == fatals[0][0] == "start_task"
    assert expected.detail in failures[0][1]
    assert expected.detail in fatals[0][1]
    assert any(item.startswith("bind:") for item in adapter.diagnostics)
    assert adapter.failure_snapshot == expected
    assert adapter.progress_snapshot.frames == 0


@pytest.mark.parametrize("native", [False, True], ids=("runtime", "native"))
def test_prewarm_normalizes_empty_read_exception_once(native):
    def fail_read(*args, **kwargs):
        if native:
            raise VkDaqError("VkDaqGetTaskData", -17, "")
        raise RuntimeError()

    controller, fatals = controller_for(
        lambda: CaptureSDK(hooks={"read_task_data": fail_read}))
    failures = []
    adapter = controller.prewarm(
        request=prewarm_request(), fail=lambda *item: failures.append(item))

    assert adapter.start()
    assert adapter.completed.wait(2)
    assert controller.failed

    expected = VeResourceFault(
        "read_task_data", -17 if native else None,
        "VkDaqError" if native else "RuntimeError",
    )
    assert adapter.failure_snapshot == expected
    assert controller.failure_snapshot == expected
    assert len(failures) == len(fatals) == 1
    assert failures[0][0] == fatals[0][0] == "read_task_data"
    assert expected.detail in failures[0][1]
    assert expected.detail in fatals[0][1]
    assert adapter.progress_snapshot.frames == 0
    assert adapter.progress_snapshot.frames < adapter.request.frames_per_channel


def test_prewarm_preserves_first_native_fault_over_cleanup_and_bind_diagnostics():
    class StartFailureSDK(CaptureSDK):
        def start_task(self, task):
            self._call("start_task", task)
            raise VkDaqError(
                "VkDaqStartTask", -12001,
                "iio_device_create_multi_buffer: invalid argument",
            )

    sdk = StartFailureSDK(failures=("clear_task", "close"))
    controller, _ = controller_for(lambda: sdk)
    failures = []
    adapter = controller.prewarm(
        request=prewarm_request(), fail=lambda *item: failures.append(item))

    assert adapter.start() is False
    assert adapter.completed.wait(2)

    expected = VeResourceFault(
        "start_task", -12001,
        "iio_device_create_multi_buffer: invalid argument",
    )
    assert adapter.failure_snapshot == expected
    assert controller.failure_snapshot == expected
    assert failures == [("start_task", str(VkDaqError(
        "VkDaqStartTask", -12001,
        "iio_device_create_multi_buffer: invalid argument")))]
    assert any(item.startswith("bind:") for item in adapter.diagnostics)
    assert controller.release(.2).success is False
    assert adapter.failure_snapshot == expected
    assert controller.failure_snapshot == expected


def test_prewarm_detach_timeout_has_structured_first_fault():
    entered = threading.Event()
    allow_read = threading.Event()

    def block(*args, **kwargs):
        entered.set()
        assert allow_read.wait(2)

    controller, _ = controller_for(
        lambda: CaptureSDK(hooks={"read_task_data": block}),
        detach_timeout=.03,
    )
    adapter = controller.prewarm(request=prewarm_request(), fail=lambda *_: None)
    assert adapter.start() and entered.wait(1)

    assert adapter.stop() is False
    assert adapter.failure_snapshot == VeResourceFault(
        "detach", None, "VE adapter detach confirmation timed out")
    allow_read.set()


@pytest.mark.parametrize("channel_order", ALL_ORDERED_CHANNEL_CASES)
def test_all_nonempty_ordered_channel_permutations_share_the_same_slot_release_path(
        tmp_path, channel_order):
    """Every legal fake identity/order takes the same compatible fast path."""
    sdk = CaptureSDK(
        counts=(4, 0, 4),
        hooks={"read_task_data": lambda *args, **kwargs: time.sleep(.001)},
    )
    sdk.record = discovery_record(
        name="FreshDev",
        machine_id="test-machine-1",
        channels=tuple(
            f"FreshDev/AIN{physical_channel + 1}"
            for physical_channel in FAKE_PHYSICAL_CHANNELS
        ),
    )
    discovered = discover_devices(sdk)
    assert discovered.diagnostics == ()
    assert discovered.devices[0]["physical_channels"] == FAKE_PHYSICAL_CHANNELS
    controller, fatals = controller_for(lambda: sdk)
    device = device_info(
        physical_channels=list(FAKE_PHYSICAL_CHANNELS),
        max_input_channels=max(FAKE_PHYSICAL_CHANNELS) + 1,
    )
    expected_values = {1: 2.5, 3: -4.75, 7: 8.25}
    transitions = []

    for identity in ("A", "B"):
        metadata = wav_metadata(
            ("none",) * len(channel_order),
            sample_rate=51200,
            physical_channels=channel_order,
        )
        metadata["acquisition"]["machine_id"] = device["machine_id"]
        request = capture_request(
            tmp_path / f"{identity}.wav",
            request_id=identity,
            device=device,
            channels=channel_order,
            target_samples=4,
            trim_samples=0,
            calibration_metadata=metadata,
        )
        capture = RecordingCapture(
            request,
            ve_stream_factory=controller.stream,
            blocksize=4,
        )
        capture.start()
        assert capture.started.wait(2) or capture.done.is_set()
        assert capture.capture_slot_released.wait(2)
        slot = capture.capture_slot
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingResult), outcome
        audio, rate = sf.read(outcome.path, dtype="float32", always_2d=True)
        np.testing.assert_array_equal(
            audio,
            np.tile([expected_values[channel] for channel in channel_order], (4, 1)),
        )
        assert rate == request.sample_rate
        assert outcome.channels == channel_order
        transitions.append((
            slot.adapter_released,
            slot.writer_released,
            slot.raw_frames,
            outcome.raw_frames,
            outcome.handles_released,
        ))

    assert transitions == [(True, True, 4, 4, True)] * 2
    assert not fatals
    assert controller.lifecycle_counts == VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    assert sdk.calls("start_task") == 1
    assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 0
    routes = [item["args"][1] for item in sdk.trace
              if item["operation"] == "create_iepe_voltage_channel"]
    assert routes == [",".join(f"FreshDev/AIN{channel + 1}" for channel in channel_order)]
    assert controller.release(.5).success


def test_first_start_is_lazy_and_all_native_calls_have_one_owner(tmp_path):
    sdk = CaptureSDK(counts=(4,), hooks={"read_task_data": lambda *a, **k: time.sleep(.001)})
    controller, fatals = controller_for(lambda: sdk)
    adapter, blocks, failures = stream(controller, tmp_path)

    assert controller.signature is None
    assert controller.lifecycle_counts == VeLifecycleCounts(0, 0, 0, 0, 0, 0)
    assert adapter.start()
    wait_released(adapter)
    outcome = controller.release(.5)

    assert outcome.success and not fatals and not failures and len(blocks) == 1
    assert sdk.calls("create_task") == sdk.calls("start_task") == 1
    assert len(sdk.owner_thread_ids) == 1
    assert threading.get_ident() not in sdk.owner_thread_ids
    assert controller.lifecycle_counts == VeLifecycleCounts(1, 1, 1, 1, 1, 1)


def test_compatible_adapters_reuse_started_task_without_inter_request_cleanup(tmp_path):
    sdk = CaptureSDK(counts=(4, 0, 4), hooks={"read_task_data": lambda *a, **k: time.sleep(.001)})
    controller, _ = controller_for(lambda: sdk)
    first, first_blocks, _ = stream(controller, tmp_path, "A")
    assert first.start()
    wait_released(first)
    counts_after_first = controller.lifecycle_counts

    second, second_blocks, _ = stream(controller, tmp_path, "B")
    assert second.start()
    wait_released(second)

    assert len(first_blocks) == len(second_blocks) == 1
    assert controller.lifecycle_counts == counts_after_first == VeLifecycleCounts(1, 1, 1, 0, 0, 0)
    assert sdk.calls("start_task") == 1
    assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 0
    assert controller.release(.5).success


def test_initial_adapter_preserves_native_start_time_across_slow_rate_verification(
        tmp_path):
    clock = DiscoveryClock()
    observed = []
    holder = {}

    def slow_verify(*args, **kwargs):
        adapter = holder["adapter"]
        observed.append((clock(), adapter.started.is_set(), adapter._bound.is_set()))
        clock.advance(1)

    sdk = CaptureSDK(
        counts=(4,),
        hooks={
            "verify_actual_sample_rate": slow_verify,
            "read_task_data": lambda *args, **kwargs: time.sleep(.001),
        },
    )
    controller, _ = controller_for(lambda: sdk, clock=clock)
    adapter, _, failures = stream(controller, tmp_path, target=4)
    holder["adapter"] = adapter

    assert adapter.start()
    wait_released(adapter)

    assert observed == [(100.0, False, False)]
    assert adapter.started_at == 100.0
    assert not failures
    assert controller.release(.5).success


def test_initial_adapter_never_publishes_latched_time_when_rate_verification_fails(
        tmp_path):
    clock = DiscoveryClock()
    sdk = CaptureSDK(
        failures=("verify_actual_sample_rate",),
        hooks={"verify_actual_sample_rate": lambda *args, **kwargs: clock.advance(1)},
    )
    controller, _ = controller_for(lambda: sdk, clock=clock)
    adapter, _, failures = stream(controller, tmp_path)

    assert adapter.start() is False

    assert not adapter.started.is_set()
    assert adapter.started_at is None
    assert [stage for stage, _ in failures] == ["verify_actual_sample_rate"]
    assert controller.release(.5).success is False


def test_reused_adapter_started_at_remains_its_actual_bind_time(tmp_path):
    clock = DiscoveryClock()
    sdk = CaptureSDK(
        counts=(4, 0, 4),
        hooks={"read_task_data": lambda *args, **kwargs: time.sleep(.001)},
    )
    controller, _ = controller_for(lambda: sdk, clock=clock)
    first, _, _ = stream(controller, tmp_path, "A")
    assert first.start()
    wait_released(first)

    clock.advance(2)
    second, _, failures = stream(controller, tmp_path, "B")
    assert second.start()
    wait_released(second)

    assert first.started_at == 100.0
    assert second.started_at == 102.0
    assert not failures
    assert controller.release(.5).success


def test_idle_read_is_discarded_and_bind_starts_with_the_following_read(tmp_path):
    idle_entered = threading.Event()
    allow_idle = threading.Event()
    reads = []

    def gate(*args, **kwargs):
        reads.append(len(reads) + 1)
        if len(reads) == 2:
            idle_entered.set()
            assert allow_idle.wait(2)

    sdk = CaptureSDK(counts=(4, 4, 4), hooks={"read_task_data": gate})
    controller, _ = controller_for(lambda: sdk)
    first, _, _ = stream(controller, tmp_path, "A")
    assert first.start()
    wait_released(first)
    assert idle_entered.wait(1)

    callback_reads = []
    second, blocks, _ = stream(
        controller, tmp_path, "B", callback=lambda block, frames, *_:
        (callback_reads.append(len(reads)), blocks.append(block.copy())),
    )
    result = []
    starter = threading.Thread(target=lambda: result.append(second.start()))
    starter.start()
    allow_idle.set()
    starter.join(1)
    wait_released(second)

    assert result == [True]
    assert callback_reads == [3]
    assert len(blocks) == 1
    assert controller.release(.5).success


def test_target_detaches_automatically_and_discards_overread(tmp_path):
    sdk = CaptureSDK(counts=(7,), hooks={"read_task_data": lambda *a, **k: time.sleep(.001)})
    controller, _ = controller_for(lambda: sdk)
    adapter, blocks, failures = stream(controller, tmp_path, target=3)

    assert adapter.start()
    wait_released(adapter)

    assert not failures
    assert sum(len(block) for block in blocks) == 3
    assert adapter.progress_snapshot.frames == 3
    assert adapter.started_at == adapter.progress_snapshot.started_at
    assert controller.lifecycle_counts.task_stop == 0
    assert controller.release(.5).success


@pytest.mark.parametrize("change", [
    {"channels": (1, 7)},
    {"sample_rate": 48000},
    {"device": "different"},
])
def test_incompatible_signature_requires_successful_release(tmp_path, change):
    delay = {"read_task_data": lambda *a, **k: time.sleep(.001)}
    sdks = [CaptureSDK(counts=(4,), hooks=delay), CaptureSDK(counts=(4,), hooks=delay)]
    controller, _ = controller_for(lambda: sdks.pop(0))
    first, _, _ = stream(controller, tmp_path, "A")
    assert first.start()
    wait_released(first)

    options = dict(change)
    if options.pop("device", None):
        request = capture_request(tmp_path / "B.wav", request_id="B", target_samples=4)
        device = dict(request.device)
        device["machine_id"] = "different-machine"
        options["device"] = device
        sdks[-1].record["MachineId"] = "different-machine"
    with pytest.raises(VeResourceConfigurationError):
        stream(controller, tmp_path, "B", **options)

    assert controller.release(.5).success
    second, _, _ = stream(controller, tmp_path, "B", **options)
    assert second.start()
    wait_released(second)
    assert controller.close(.5).success


def test_stop_and_close_share_one_bounded_detach_with_no_late_callback(tmp_path):
    entered = threading.Event()
    allow_read = threading.Event()
    callbacks = []

    def block(*args, **kwargs):
        entered.set()
        assert allow_read.wait(2)

    sdk = CaptureSDK(counts=(4,), hooks={"read_task_data": block})
    controller, _ = controller_for(lambda: sdk)
    adapter, _, _ = stream(
        controller, tmp_path, target=100,
        callback=lambda *args: callbacks.append("called"),
    )
    assert adapter.start()
    assert entered.wait(1)
    stopped = []
    stopper = threading.Thread(target=lambda: stopped.append(adapter.stop()))
    stopper.start()
    allow_read.set()
    stopper.join(1)

    assert stopped == [True]
    assert adapter.close() is True
    assert adapter.stop() is True
    assert adapter.handles_released and callbacks == []
    assert controller.release(.5).success


def test_detach_timeout_is_permanently_fatal_and_never_restores_release_claim(tmp_path):
    entered = threading.Event()
    allow_read = threading.Event()
    sdk = CaptureSDK(hooks={"read_task_data": lambda *a, **k:
                            (entered.set(), allow_read.wait(2))})
    controller, fatals = controller_for(lambda: sdk, detach_timeout=.03)
    adapter, _, _ = stream(controller, tmp_path, target=100)
    assert adapter.start() and entered.wait(1)

    assert adapter.stop() is False
    assert not adapter.handles_released and controller.failed
    allow_read.set()
    time.sleep(.05)

    assert not adapter.handles_released
    assert len(fatals) == 1
    assert controller.release(.1).success is False


def test_bind_timeout_never_claims_attachment_and_permanently_fails(tmp_path):
    entered = threading.Event()
    allow_factory = threading.Event()

    def factory():
        entered.set()
        assert allow_factory.wait(2)
        return CaptureSDK()

    controller, fatals = controller_for(factory, bind_timeout=.03)
    adapter, _, failures = stream(controller, tmp_path)

    assert adapter.start() is False and entered.is_set()
    assert adapter.handles_released and not adapter.started.is_set()
    assert controller.failed and len(fatals) == 1
    allow_factory.set()
    time.sleep(.05)
    assert len(failures) == 1
    assert controller.release(.1).success is False


def test_sdk_initialization_failure_reports_exact_stage_to_request_and_generation(tmp_path):
    sdk = CaptureSDK(failures=("create_task",))
    controller, fatals = controller_for(lambda: sdk)
    adapter, _, failures = stream(controller, tmp_path)

    assert adapter.start() is False
    assert [stage for stage, _ in failures] == ["create_task"]
    assert [stage for stage, _ in fatals] == ["create_task"]
    assert failures[0][1] == fatals[0][1]
    assert "injected create_task" in failures[0][1]


def test_release_during_blocked_initialization_is_busy_and_cannot_mask_init_failure(tmp_path):
    entered = threading.Event()
    allow_failure = threading.Event()

    def factory():
        entered.set()
        assert allow_failure.wait(2)
        raise RuntimeError("INIT-BOOM")

    controller, fatals = controller_for(factory, bind_timeout=.5)
    adapter, _, failures = stream(controller, tmp_path)
    started = []
    starter = threading.Thread(target=lambda: started.append(adapter.start()))
    starter.start()
    assert entered.wait(1)

    outcome = controller.release(.1)
    assert not outcome.success
    assert outcome.diagnostics == ("release: initialization in progress",)
    assert not controller.failed
    allow_failure.set()
    starter.join(1)

    assert started == [False]
    assert failures == [("sdk_open", "INIT-BOOM")]
    assert fatals == [("sdk_open", "INIT-BOOM")]
    final = controller.release(.1)
    assert not final.success
    assert final.diagnostics[0] == "sdk_open: INIT-BOOM"
    assert all(not item.startswith("release:") for item in final.diagnostics)


def test_close_during_blocked_initialization_remains_bounded(tmp_path):
    entered = threading.Event()
    allow_factory = threading.Event()

    def factory():
        entered.set()
        assert allow_factory.wait(2)
        return CaptureSDK()

    controller, _ = controller_for(factory, bind_timeout=.5)
    adapter, _, _ = stream(controller, tmp_path)
    starter = threading.Thread(target=adapter.start)
    starter.start()
    assert entered.wait(1)

    before = time.monotonic()
    outcome = controller.close(.03)
    elapsed = time.monotonic() - before

    assert not outcome.success and elapsed < .2 and controller.failed
    allow_factory.set()
    starter.join(1)


def test_release_rejects_an_active_adapter_without_native_cleanup(tmp_path):
    entered = threading.Event()
    allow_read = threading.Event()

    def block(*args, **kwargs):
        entered.set()
        assert allow_read.wait(2)

    sdk = CaptureSDK(hooks={"read_task_data": block})
    controller, _ = controller_for(lambda: sdk)
    adapter, _, _ = stream(controller, tmp_path, target=100)
    assert adapter.start() and entered.wait(1)

    outcome = controller.release(.1)
    assert not outcome.success and outcome.diagnostics == ("release: active adapter",)
    assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 0
    allow_read.set()
    assert adapter.stop()
    assert controller.release(.5).success


def test_release_timeout_remains_failed_after_late_owner_cleanup(tmp_path):
    idle_entered = threading.Event()
    allow_idle = threading.Event()
    reads = []

    def block_idle(*args, **kwargs):
        reads.append(1)
        if len(reads) == 2:
            idle_entered.set()
            assert allow_idle.wait(2)

    sdk = CaptureSDK(counts=(4, 4), hooks={"read_task_data": block_idle})
    controller, fatals = controller_for(lambda: sdk)
    adapter, _, _ = stream(controller, tmp_path)
    assert adapter.start()
    wait_released(adapter)
    assert idle_entered.wait(1)

    outcome = controller.release(.03)
    assert not outcome.success and controller.failed and len(fatals) == 1
    allow_idle.set()
    deadline = time.monotonic() + 1
    while sdk.calls("close") == 0 and time.monotonic() < deadline:
        time.sleep(.002)

    assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 1
    assert controller.failed and controller.signature is not None
    assert controller.release(.1).success is False


def test_unstarted_adapter_close_is_released_without_initializing_sdk(tmp_path):
    sdk = CaptureSDK()
    controller, _ = controller_for(lambda: sdk)
    adapter, _, _ = stream(controller, tmp_path)

    assert adapter.close() and adapter.stop() and adapter.handles_released
    assert controller.signature is None and not sdk.trace


@pytest.mark.parametrize("idle", [False, True])
def test_read_fault_reports_one_fatal_and_permanently_disables_controller(tmp_path, idle):
    reads = []

    def fault(*args, **kwargs):
        reads.append(1)
        if not idle or len(reads) == 2:
            raise RuntimeError("injected read fault")

    sdk = CaptureSDK(counts=(4,), hooks={"read_task_data": fault})
    controller, fatals = controller_for(lambda: sdk)
    adapter, _, failures = stream(controller, tmp_path, target=4)
    assert adapter.start()
    deadline = time.monotonic() + 1
    while not controller.failed and time.monotonic() < deadline:
        time.sleep(.002)

    assert controller.failed and len(fatals) == 1
    assert "read fault" in fatals[0][1]
    if not idle:
        assert len(failures) == 1
    else:
        assert failures == []  # a detached request cannot receive an idle fatal
    with pytest.raises(RuntimeError, match="failed"):
        stream(controller, tmp_path, "B").start()
    assert controller.release(.2).success is False


def test_release_orders_cleanup_on_owner_and_retains_every_error(tmp_path):
    sdk = CaptureSDK(
        counts=(4,), failures=("stop_task", "clear_task", "close"),
        hooks={"read_task_data": lambda *a, **k: time.sleep(.001)},
    )
    controller, fatals = controller_for(lambda: sdk)
    adapter, _, _ = stream(controller, tmp_path)
    assert adapter.start()
    wait_released(adapter)

    outcome = controller.release(.5)

    assert isinstance(outcome, VeResourceReleaseOutcome) and not outcome.success
    assert [next(op for op in ("stop_task", "clear_task", "close") if op in item)
            for item in outcome.diagnostics] == ["stop_task", "clear_task", "close"]
    assert sdk.operations()[-3:] == ("stop_task", "clear_task", "close")
    assert len(sdk.owner_thread_ids) == 1 and len(fatals) == 1
    assert controller.failed
    assert controller.release(.2) == outcome
    assert controller.close(.2) == outcome
    assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 1


def test_successful_release_returns_to_uninitialized_and_cleanup_is_idempotent(tmp_path):
    delay = {"read_task_data": lambda *a, **k: time.sleep(.001)}
    sdks = [CaptureSDK(counts=(4,), hooks=delay), CaptureSDK(counts=(4,), hooks=delay)]
    controller, _ = controller_for(lambda: sdks.pop(0))
    first, _, _ = stream(controller, tmp_path, "A")
    assert first.start()
    wait_released(first)
    released = controller.release(.5)

    assert released.success and controller.signature is None and not controller.failed
    assert controller.release(.5).success
    second, _, _ = stream(controller, tmp_path, "B", sample_rate=48000)
    assert second.start()
    wait_released(second)
    assert controller.close(.5).success
    assert controller.lifecycle_counts == VeLifecycleCounts(2, 2, 2, 2, 2, 2)


def _main_metadata(device, channels, rate):
    metadata = wav_metadata(tuple("none" for _ in channels), rate, physical_channels=channels)
    metadata["acquisition"]["machine_id"] = device["machine_id"]
    return metadata
