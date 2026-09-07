"""Fake-only discovery, stable routing, and real spawn lifecycle regressions."""
import ctypes
import json
import multiprocessing
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from base import ve3668n_discovery as discovery
from base import vkinging_sdk
from base.ve3668n_input import validate_device_snapshot
from consts.ve3668n_consts import VE_DEVICE_SNAPSHOT_FIELDS
from unit_test.base.ve3668n_fakes import DiscoveryClock, DiscoverySDK, device_info, discovery_record


@pytest.fixture(autouse=True)
def no_native_or_portaudio(monkeypatch):
    original_children = {child.pid for child in multiprocessing.active_children()}
    original_threads = set(threading.enumerate())
    def forbidden(*args, **kwargs):
        pytest.fail("real SDK or PortAudio fallback called")

    class NoDefault:
        def __getattribute__(self, name):
            forbidden()

        def __setattr__(self, name, value):
            forbidden()

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(ctypes, "WinDLL", forbidden)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        default=NoDefault(), query_devices=forbidden, InputStream=forbidden))
    yield
    assert {child.pid for child in multiprocessing.active_children()} <= original_children
    wait_until(lambda: not any(thread not in original_threads and thread.name.startswith("ve-discovery-")
                               for thread in threading.enumerate()), timeout=1)


@pytest.mark.parametrize("model,accepted", [
    ("VE3668N", True), ("  ve3668n\t", True), ("Ve3668N", True),
    ("VE3668", False), ("VE3668N-Pro", False), ("XVE3668N", False),
    ("", False), (None, False), (3668, False), ("VE3668N\0", False),
])
def test_full_model_match_and_closed_snapshot(model, accepted):
    sdk = DiscoverySDK([discovery_record(model=model)])
    result = discovery.discover_devices(sdk)
    assert len(result.devices) == int(accepted)
    if accepted:
        device = result.devices[0]
        assert device == validate_device_snapshot(device)
        assert set(device) == set(VE_DEVICE_SNAPSHOT_FIELDS)
        assert device["model"] == "VE3668N"
        assert device["physical_channels"] == tuple(range(8))
        assert device["input_config"]["sample_rate"] == 51200
        assert result.diagnostics == ()
    else:
        assert result.diagnostics
    assert not sdk.closed  # caller owns the pure policy's SDK


@pytest.mark.parametrize("machine_id", [None, "", " \t", 42, "bad\0id"])
def test_missing_or_malformed_stable_identity_is_never_recordable(machine_id):
    result = discovery.discover_devices(DiscoverySDK([discovery_record(machine_id=machine_id)]))
    assert result.devices == ()
    assert "machine" in " ".join(result.diagnostics).lower()


@pytest.mark.parametrize("field", ["Model", "MachineId", "channels", "DeviceStatus"])
def test_mandatory_failures_exclude_but_optional_status_only_diagnoses(field):
    record = discovery_record()
    record[field] = vkinging_sdk.VkDaqError(field, -17, "fake read failure")
    result = discovery.discover_devices(DiscoverySDK([record]))
    assert len(result.devices) == int(field == "DeviceStatus")
    assert field in " ".join(result.diagnostics)
    assert "-17" in " ".join(result.diagnostics)


def test_analog_only_preserves_inventory_order_and_multiple_devices():
    sdk = DiscoverySDK([
        discovery_record(channels=("Dev1/MIN1", "Dev1/AIN8", "Dev1/DIN1",
                                   "Dev1/RAWAIN2", "Dev1/AIN2")),
        discovery_record("Dev2", "machine-2", address="192.0.2.2"),
    ])
    result = discovery.discover_devices(sdk)
    assert [item["machine_id"] for item in result.devices] == ["machine-1", "machine-2"]
    assert result.devices[0]["physical_channels"] == (7, 1)
    assert result.devices[1]["physical_channels"] == tuple(range(8))
    assert result.devices[0]["max_input_channels"] == 8
    assert result.diagnostics == ()


@pytest.mark.parametrize("channels", [
    (), ("Dev1/MIN1", "Dev1/DIN1", "Dev1/RAWAIN1"),
    ("Dev1/AIN0",), ("Dev1/AIN9",), ("Dev1/AIN01",),
    ("Dev1/ain1",), ("Dev1/AIN1x",), ("Dev1/AIN1\0",),
    ("Dev1/AIN1", "Dev1/AIN1"), ("Dev1/AIN1", "Dev2/AIN2"),
    ("Dev1/AIN1", "Dev2/MIN1"), ("AIN1",), ("Dev1/AIN1/extra",),
    (None,), "Dev1/AIN1",
])
def test_malformed_or_crosswired_channel_routes_fail_closed(channels):
    result = discovery.discover_devices(DiscoverySDK([discovery_record(channels=channels)]))
    assert result.devices == ()
    assert result.diagnostics


def test_simultaneous_duplicate_aliases_are_not_queried_by_address():
    sdk = DiscoverySDK([discovery_record(), discovery_record(machine_id="other", address="192.0.2.2")])
    result = discovery.discover_devices(sdk)
    assert result.devices == ()
    assert "ambiguous" in " ".join(result.diagnostics)
    assert sdk.trace == [("devices",)]


def test_duplicate_machine_ids_reject_all_candidates_not_first_match():
    sdk = DiscoverySDK([discovery_record(), discovery_record("Dev2")])
    result = discovery.discover_devices(sdk)
    assert result.devices == ()
    assert "ambiguous" in " ".join(result.diagnostics)


def test_resolve_tracks_identity_not_alias_and_preserves_selection_order():
    sdk = DiscoverySDK()
    first = discovery.resolve_device(sdk, " machine-1 ", (7, 1))
    assert first["name"] == "Dev1"
    sdk.records = [discovery_record(machine_id="replacement"),
                   discovery_record("Dev9", "machine-1", address="192.0.2.9")]
    resolved = discovery.resolve_device(sdk, "machine-1", (7, 1))
    assert resolved["name"] == "Dev9"
    assert resolved["address"] == "192.0.2.9"
    assert resolved["physical_channels"] == (7, 1)
    assert first["name"] == "Dev1"
    assert sdk.trace.count(("devices",)) == 2
    sdk.records = [discovery_record(machine_id="replacement")]
    with pytest.raises(ValueError, match="unavailable"):
        discovery.resolve_device(sdk, "machine-1", (7, 1))


@pytest.mark.parametrize("channels", [(), (1, 1), (True,), (8,), (-1,), "1", (1.0,)])
def test_resolve_rejects_bad_selection_before_sdk_io(channels):
    sdk = DiscoverySDK()
    with pytest.raises(ValueError, match="physical_channel"):
        discovery.resolve_device(sdk, "machine-1", channels)
    assert sdk.trace == []


@pytest.mark.parametrize("record", [
    discovery_record(model="VE3668"),
    discovery_record(channels=("Dev1/AIN2",)),
    discovery_record(machine_id="other"),
])
def test_resolve_rejects_missing_identity_model_or_selected_channels(record):
    with pytest.raises(ValueError, match="unavailable"):
        discovery.resolve_device(DiscoverySDK([record]), "machine-1", (7, 1))


def test_resolve_does_not_choose_first_duplicate_stable_id():
    sdk = DiscoverySDK([discovery_record(), discovery_record("Dev2", model="wrong")])
    with pytest.raises(ValueError, match="ambiguous"):
        discovery.resolve_device(sdk, "machine-1", (0,))


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(.01)
    assert predicate(), "condition did not complete within test deadline"


def close_service(service):
    began = time.monotonic()
    service.close()
    assert time.monotonic() - began < .2
    assert service.wait_closed(3)


def wait_for_stage(tmp_path, stage):
    trace = tmp_path / "trace.jsonl"
    wait_until(lambda: trace.exists() and f'"stage": "{stage}"' in trace.read_text())


def track_resources(service, monkeypatch):
    processes, pipes = [], []
    original_process, original_pipe = service._context.Process, service._context.Pipe

    def process(*args, **kwargs):
        child = original_process(*args, **kwargs)
        processes.append(child)
        return child

    def pipe(*args, **kwargs):
        endpoints = original_pipe(*args, **kwargs)
        pipes.extend(endpoints)
        return endpoints

    monkeypatch.setattr(service._context, "Process", process)
    monkeypatch.setattr(service._context, "Pipe", pipe)
    return processes, pipes


def test_spawn_success_is_async_and_closes_process_and_pipe_handles(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    callbacks = []
    service = discovery.DiscoveryService(
        sdk_factory="unit_test.base.ve3668n_fakes.discovery_factory",
        sdk_options={"trace_path": str(trace)},
        on_result=lambda result: callbacks.append((result, threading.get_ident())),
    )
    processes, pipes = track_resources(service, monkeypatch)
    try:
        began = time.monotonic()
        generation = service.start()
        assert time.monotonic() - began < .2
        wait_until(lambda: callbacks)
        event, callback_thread = callbacks[0]
        assert callback_thread != threading.get_ident()
        assert event.generation == generation
        assert event.status == "completed"
        assert event.result.devices[0]["machine_id"] == "machine-1"
        assert event.child_pid != os.getpid()
        assert 0 < event.message_bytes <= service.max_result_bytes
        assert event.handles_released
        assert service.poll_result() == event
        assert service.poll_result() is None
        assert service.wait_idle(1)
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        assert [call["stage"] for call in calls] == ["factory", "probe", "closed"]
        assert {call["pid"] for call in calls} == {event.child_pid}
        assert event.child_pid not in {child.pid for child in multiprocessing.active_children()}
    finally:
        close_service(service)


def fake_service(tmp_path, mode, **kwargs):
    return discovery.DiscoveryService(
        sdk_factory="unit_test.base.ve3668n_fakes.discovery_factory",
        sdk_options={"trace_path": str(tmp_path / "trace.jsonl"), "mode": mode}, **kwargs)


@pytest.mark.parametrize("mode", ["block_factory", "block_probe", "block_close"])
def test_deadline_retires_blocked_native_factory_probe_or_close(tmp_path, monkeypatch, mode):
    service = fake_service(tmp_path, mode, deadline=.5, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    processes, pipes = track_resources(service, monkeypatch)
    try:
        service.start()
        wait_for_stage(tmp_path, {"block_factory": "factory", "block_probe": "probe",
                                  "block_close": "closing"}[mode])
        began = time.monotonic()
        clock.advance(1)
        assert service.wait_idle(2)
        assert time.monotonic() - began < 2
        event = service.poll_result()
        assert event.status == "unavailable"
        assert "deadline" in " ".join(event.result.diagnostics)
        assert event.handles_released
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        close_service(service)


def test_crashed_native_probe_reports_exitcode_and_releases_handles(tmp_path, monkeypatch):
    service = fake_service(tmp_path, "crash")
    processes, pipes = track_resources(service, monkeypatch)
    try:
        service.start()
        assert service.wait_idle(3)
        event = service.poll_result()
        assert event.status == "unavailable"
        assert "23" in " ".join(event.result.diagnostics)
        assert event.handles_released
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        close_service(service)


def test_deadline_includes_spawn_startup_outside_calling_thread(tmp_path, monkeypatch):
    service = fake_service(tmp_path, "normal", deadline=.1, retire_timeout=.05)
    original = service._context.Process

    def slow_start(*args, **kwargs):
        child = original(*args, **kwargs)
        start = child.start

        def delayed():
            threading.Event().wait(.2)
            del child.start  # do not pickle this parent-only instrumentation
            start()

        child.start = delayed
        return child

    monkeypatch.setattr(service._context, "Process", slow_start)
    try:
        began = time.monotonic()
        service.start()
        assert time.monotonic() - began < .1
        assert service.wait_idle(2)
        assert "deadline" in " ".join(service.poll_result().result.diagnostics)
    finally:
        close_service(service)


def gate_first_process_start(service, monkeypatch, *, outcome="success", after_start=False,
                            error_type=OSError):
    processes, pipes = track_resources(service, monkeypatch)
    original_process = service._context.Process
    gate = SimpleNamespace(entered=threading.Event(), release=threading.Event(),
                           returned=threading.Event(), processes=processes, pipes=pipes,
                           threads=[], unsafe_access=[])

    def guard(operation):
        original = getattr(multiprocessing.process.BaseProcess, operation)

        def checked(child, *args, **kwargs):
            if (child in processes and gate.entered.is_set() and not gate.returned.is_set()
                    and threading.current_thread() is not gate.threads[0]):
                gate.unsafe_access.append(operation)
                raise AssertionError(f"Process.{operation} raced with start")
            return original(child, *args, **kwargs)

        monkeypatch.setattr(multiprocessing.process.BaseProcess, operation, checked)

    guard("_check_closed")  # includes pid/exitcode/is_alive/join/terminate/kill
    guard("close")

    def process(*args, **kwargs):
        # A replacement may not even be allocated until the prior owner is
        # retired; the last two endpoints belong to this new probe's Pipe().
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes[:-2])
        child = original_process(*args, **kwargs)
        if len(processes) == 1:
            original_start = child.start

            def start():
                del child.start  # parent-only instrumentation must not be pickled
                gate.threads.append(threading.current_thread())
                try:
                    if after_start:
                        original_start()
                    gate.entered.set()
                    gate.release.wait()  # only the test's finally may release a failed assertion
                    if outcome == "failure":
                        raise error_type("fake late launch failure")
                    if not after_start:
                        original_start()
                    if outcome == "started_failure":
                        raise error_type("fake late launch failure after child started")
                finally:
                    gate.returned.set()

            child.start = start
        return child

    monkeypatch.setattr(service._context, "Process", process)
    return gate


@pytest.mark.parametrize("outcome,after_start", [
    ("success", False), ("failure", False), ("started_failure", False),
    ("success", True), ("started_failure", True),
])
def test_held_process_start_reports_deadline_before_release_and_keeps_ownership(
        tmp_path, monkeypatch, caplog, outcome, after_start):
    events = []
    service = fake_service(tmp_path, "normal", on_result=events.append,
                           deadline=.05, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    gate = gate_first_process_start(service, monkeypatch, outcome=outcome, after_start=after_start)
    try:
        generation = service.start()
        assert gate.entered.wait(1)
        clock.advance(.1)
        wait_until(lambda: events, timeout=.3)
        event = events[0]
        assert not gate.release.is_set()
        assert event.generation == generation
        assert event.status == "unavailable"
        assert "deadline" in " ".join(event.result.diagnostics)
        assert event.result.devices == ()
        assert event.child_pid is None
        assert event.message_bytes == 0
        assert not event.handles_released
        assert not service.wait_idle(.06)
        assert not service.wait_closed(0)
        assert sum(thread.name == "ve-discovery-launcher" for thread in threading.enumerate()) == 1
        assert not any(thread.name == "ve-discovery-receiver" for thread in threading.enumerate())
        assert not gate.processes[0]._closed
        assert all(not endpoint.closed for endpoint in gate.pipes)
        assert service.poll_result() == event
        gate.release.set()
        assert service.wait_idle(3)
        assert all(child._closed for child in gate.processes)
        assert all(endpoint.closed for endpoint in gate.pipes)
        assert gate.unsafe_access == []
        assert gate.threads[0].name == "ve-discovery-launcher"
        assert not gate.threads[0].is_alive()
        if outcome != "success":
            assert "fake late launch failure" in caplog.text
        assert events == [event]
        assert service.poll_result() is None
    finally:
        gate.release.set()
        close_service(service)


@pytest.mark.parametrize("outcome", ["success", "failure", "started_failure"])
def test_refresh_deadline_is_supervised_while_prior_launch_is_held(tmp_path, monkeypatch, outcome):
    events = []
    service = fake_service(tmp_path, "normal", on_result=events.append,
                           deadline=.05, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    gate = gate_first_process_start(service, monkeypatch, outcome=outcome)
    try:
        first = service.start()
        assert gate.entered.wait(1)
        began = time.monotonic()
        for _ in range(20):
            expired = service.refresh()
        assert time.monotonic() - began < .2
        clock.advance(.1)
        wait_until(lambda: events, timeout=.3)
        event = events[0]
        assert event.generation == expired > first
        assert event.status == "unavailable"
        assert "deadline" in " ".join(event.result.diagnostics)
        assert not event.handles_released
        assert len(gate.processes) == 1
        assert len(gate.pipes) == 2
        assert not service.wait_idle(.06)
        assert not gate.release.is_set()
        for _ in range(20):
            latest = service.refresh()
        assert service.poll_result() is None
        assert len(gate.processes) == 1
        gate.release.set()
        wait_until(lambda: len(events) == 2, timeout=3)
        assert service.wait_idle(1)
        assert events[1].generation == latest
        assert events[1].status == "completed"
        assert events[1].handles_released
        assert [item.generation for item in events] == [expired, latest]
        assert len(gate.processes) == 2
        assert all(child._closed for child in gate.processes)
        assert all(endpoint.closed for endpoint in gate.pipes)
        assert gate.unsafe_access == []
    finally:
        gate.release.set()
        close_service(service)


@pytest.mark.parametrize("outcome", ["success", "failure", "started_failure"])
@pytest.mark.parametrize("action", ["cancel", "close"])
def test_held_launch_lifecycle_invalidates_results_and_reaps_after_release(
        tmp_path, monkeypatch, outcome, action):
    events = []
    service = fake_service(tmp_path, "block_probe", on_result=events.append, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    gate = gate_first_process_start(service, monkeypatch, outcome=outcome)
    try:
        service.start()
        assert gate.entered.wait(1)
        began = time.monotonic()
        for _ in range(20):
            service.refresh()
        getattr(service, action)()
        assert time.monotonic() - began < .2
        clock.advance(10)
        assert not service.wait_idle(.06)
        assert not service.wait_closed(.03)
        assert events == []
        assert service.poll_result() is None
        assert len(gate.processes) == 1
        assert all(not endpoint.closed for endpoint in gate.pipes)
        gate.release.set()
        assert service.wait_idle(3)
        if action == "close":
            assert service.wait_closed(1)
        assert events == []
        assert service.poll_result() is None
        assert all(child._closed for child in gate.processes)
        assert all(endpoint.closed for endpoint in gate.pipes)
        assert gate.unsafe_access == []
        assert not gate.threads[0].is_alive()
    finally:
        gate.release.set()
        close_service(service)


@pytest.mark.parametrize("outcome", ["failure", "started_failure"])
@pytest.mark.parametrize("expired", [False, True])
def test_spawn_serialization_failure_is_diagnosed_and_never_delivers_late_devices(
        tmp_path, monkeypatch, caplog, outcome, expired):
    events = []
    service = fake_service(tmp_path, "normal", on_result=events.append,
                           deadline=.05, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    gate = gate_first_process_start(service, monkeypatch, outcome=outcome, error_type=TypeError)
    try:
        service.start()
        assert gate.entered.wait(1)
        if expired:
            clock.advance(.1)
            wait_until(lambda: events, timeout=.3)
            assert not events[0].handles_released
        gate.release.set()
        wait_until(lambda: events, timeout=3)
        assert service.wait_idle(3)
        assert len(events) == 1
        event = events[0]
        assert event.status == "unavailable"
        assert event.result.devices == ()
        diagnostic = caplog.text if expired else " ".join(event.result.diagnostics)
        assert "TypeError" in diagnostic
        assert "fake late launch failure" in diagnostic
        assert all(child._closed for child in gate.processes)
        assert all(endpoint.closed for endpoint in gate.pipes)
        assert gate.unsafe_access == []
    finally:
        gate.release.set()
        close_service(service)


def test_held_launch_timeout_callback_cannot_block_public_lifecycle(tmp_path, monkeypatch):
    entered, release_callback = threading.Event(), threading.Event()
    events = []

    def callback(event):
        events.append(event)
        entered.set()
        release_callback.wait()

    service = fake_service(tmp_path, "normal", on_result=callback,
                           deadline=.05, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    gate = gate_first_process_start(service, monkeypatch)
    try:
        service.start()
        assert gate.entered.wait(1)
        clock.advance(.1)
        assert entered.wait(.3)
        assert not events[0].handles_released
        began = time.monotonic()
        service.refresh()
        service.cancel()
        service.close()
        assert time.monotonic() - began < .2
        assert not service.wait_closed(.03)
        assert not service.wait_idle(0)
        assert service.poll_result() is None
        assert len(gate.processes) == 1
        gate.release.set()
        release_callback.set()
        assert service.wait_closed(3)
        assert len(events) == 1
        assert all(child._closed for child in gate.processes)
        assert all(endpoint.closed for endpoint in gate.pipes)
        assert gate.unsafe_access == []
    finally:
        gate.release.set()
        release_callback.set()
        close_service(service)


def test_terminate_escalates_to_kill_and_confirms_death(tmp_path, monkeypatch):
    service = fake_service(tmp_path, "block_probe", deadline=.4, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    processes, pipes = track_resources(service, monkeypatch)
    try:
        service.start()
        wait_for_stage(tmp_path, "probe")
        child = processes[0]
        monkeypatch.setattr(child, "terminate", lambda: None)
        clock.advance(1)
        assert service.wait_idle(2)
        assert service.poll_result().handles_released
        assert child._closed
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        for child in processes:
            if not child._closed:
                child.kill()
                child.join(2)
                child.close()
        close_service(service)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OS process-death observation")
@pytest.mark.parametrize("mode", ["block_factory", "block_probe", "block_close"])
def test_parent_death_exits_helper_even_while_native_is_blocked(tmp_path, mode):
    import _winapi
    from unit_test.base.ve3668n_fakes import discovery_orphan_parent

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    parent = context.Process(target=discovery_orphan_parent, args=(send, {
        "trace_path": str(tmp_path / "trace.jsonl"), "mode": mode}))
    handle = None
    try:
        parent.start()
        send.close()
        assert receive.poll(5)
        pid = receive.recv()
        handle = _winapi.OpenProcess(0x00100000 | 0x0001, False, pid)
        assert _winapi.WaitForSingleObject(handle, 0) == 258
        parent.terminate()
        parent.join(2)
        assert not parent.is_alive()
        wait_until(lambda: _winapi.WaitForSingleObject(handle, 0) == 0, timeout=2)
    finally:
        if parent.is_alive():
            parent.kill()
        parent.join(2)
        parent.close()
        receive.close()
        send.close()
        if handle is not None:
            if _winapi.WaitForSingleObject(handle, 0) != 0:
                _winapi.TerminateProcess(handle, 1)
                assert _winapi.WaitForSingleObject(handle, 2000) == 0
            _winapi.CloseHandle(handle)


@pytest.mark.parametrize("mode", [
    "wire_json", "wire_shape", "wire_diagnostic", "wire_audio", "wire_duplicate",
    "wire_oversize", "wire_nested", "oversize", "huge_error", "close_error",
])
def test_untrusted_or_oversized_result_is_bounded_unavailable_and_reaped(tmp_path, mode):
    service = fake_service(tmp_path, mode, max_result_bytes=1024)
    try:
        service.start()
        assert service.wait_idle(3)
        event = service.poll_result()
        assert event.status == "unavailable"
        assert event.result.devices == ()
        assert event.result.diagnostics
        assert 0 <= event.message_bytes <= 1024
        assert sum(len(text.encode("utf-8")) for text in event.result.diagnostics) < 1024
        assert event.handles_released
        if mode in ("oversize", "huge_error", "close_error"):
            assert event.message_bytes > 0  # child normalized, not a crash/empty pipe
        if mode == "oversize":
            assert "limit" in " ".join(event.result.diagnostics)
    finally:
        close_service(service)


@pytest.mark.parametrize("extra", ["sensitivity", "index", "audio"])
def test_sender_rejects_unknown_fields_without_serializing_arbitrary_objects(extra):
    class NotSerializable:
        def __reduce__(self):
            pytest.fail("result object was pickled")

    result = discovery.DiscoveryResult(devices=(device_info() | {extra: NotSerializable()},))
    with pytest.raises(ValueError, match="unknown"):
        discovery._encode_result(result, 1024)


def test_wire_roundtrip_retains_only_safe_closed_fields():
    result = discovery.discover_devices(DiscoverySDK())
    encoded = discovery._encode_result(result, 1024)
    assert len(encoded) <= 1024
    assert discovery._decode_result(encoded) == result
    assert set(json.loads(encoded)) == {"devices", "diagnostics"}


@pytest.mark.parametrize("kwargs", [
    {"deadline": value} for value in [0, -1, True, float("inf"), float("nan"), "5"]
] + [
    {"retire_timeout": value} for value in [0, -1, False, float("inf"), "1"]
] + [
    {"max_result_bytes": value} for value in [0, 1023, 65537, True, 2048.0]
] + [
    {"sdk_factory": value} for value in [None, "", "factory", "x:factory", "a.<locals>.f", object()]
] + [
    {"sdk_options": value} for value in [[], {1: "value"}, {"x": object()}, {"x": "x" * 5000}]
] + [{"on_result": 17}])
def test_invalid_service_configuration_is_rejected_before_start(kwargs):
    with pytest.raises(ValueError):
        service = discovery.DiscoveryService(**kwargs)
        close_service(service)


def test_production_default_deadline_is_five_seconds_and_close_is_idempotent():
    service = discovery.DiscoveryService()
    assert service.deadline == 5.0
    close_service(service)
    close_service(service)
    with pytest.raises(RuntimeError, match="closed"):
        service.refresh()


@pytest.mark.parametrize("action", ["cancel", "close"])
@pytest.mark.parametrize("phase", ["startup", "native", "result"])
def test_cancel_or_close_at_each_phase_suppresses_delivery_and_reaps(tmp_path, monkeypatch, phase, action):
    callbacks = []
    service = fake_service(tmp_path, "normal" if phase == "result" else "block_probe",
                           on_result=callbacks.append, retire_timeout=.05)
    processes, pipes = track_resources(service, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original_process = service._context.Process

    def delayed_process(*args, **kwargs):
        child = original_process(*args, **kwargs)
        entered.set()
        assert release.wait(2)
        return child

    if phase == "startup":
        monkeypatch.setattr(service._context, "Process", delayed_process)
        launched = []
        original_start = multiprocessing.process.BaseProcess.start

        def start(child):
            launched.append(child)
            original_start(child)

        monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", start)
    try:
        service.start()
        if phase == "startup":
            assert entered.wait(2)
        elif phase == "native":
            wait_until(lambda: (tmp_path / "trace.jsonl").exists())
        else:
            wait_until(lambda: callbacks)
        delivered = list(callbacks)
        began = time.monotonic()
        getattr(service, action)()
        assert time.monotonic() - began < .2
        release.set()
        assert service.wait_idle(2)
        assert callbacks == delivered
        assert service.poll_result() is None
        if phase == "startup":
            assert launched == []
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        release.set()
        close_service(service)


def test_repeated_refresh_drops_stale_generations_and_can_recover_after_block(tmp_path, monkeypatch):
    callbacks = []
    service = fake_service(tmp_path, "block_once", on_result=callbacks.append,
                           retire_timeout=.05)
    processes, pipes = track_resources(service, monkeypatch)
    try:
        first = service.start()
        wait_until(lambda: (tmp_path / "trace.jsonl").exists())
        began = time.monotonic()
        for _ in range(20):
            latest = service.refresh()
        assert time.monotonic() - began < .2
        wait_until(lambda: callbacks)
        assert [event.generation for event in callbacks] == [latest]
        assert latest > first
        assert callbacks[0].status == "completed"
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
        assert service.poll_result().generation == latest
        assert service.poll_result() is None
    finally:
        close_service(service)


@pytest.mark.parametrize("where", ["pipe", "spawn"])
def test_process_setup_failure_reports_async_unavailable_and_closes_handles(tmp_path, monkeypatch, where):
    service = fake_service(tmp_path, "normal")
    processes, pipes = track_resources(service, monkeypatch)
    if where == "pipe":
        def failure(*args, **kwargs):
            raise OSError("fake pipe allocation failure")

        monkeypatch.setattr(service._context, "Pipe", failure)
    else:
        def failure(*args, **kwargs):
            raise OSError("fake spawn failure")

        monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", failure)
    try:
        service.start()
        assert service.wait_idle(2)
        event = service.poll_result()
        assert event.status == "unavailable"
        assert "fake" in " ".join(event.result.diagnostics)
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        close_service(service)


def test_terminate_error_still_kills_and_keeps_original_deadline_diagnostic(tmp_path, monkeypatch):
    service = fake_service(tmp_path, "block_probe", deadline=.5, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    processes, pipes = track_resources(service, monkeypatch)
    try:
        service.start()
        wait_for_stage(tmp_path, "probe")

        def failed_terminate():
            raise OSError("fake terminate failure")

        monkeypatch.setattr(processes[0], "terminate", failed_terminate)
        clock.advance(1)
        assert service.wait_idle(2)
        event = service.poll_result()
        assert event.handles_released
        assert "deadline" in " ".join(event.result.diagnostics)
        assert "fake terminate failure" in " ".join(event.result.diagnostics)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        for child in processes:
            if not child._closed:
                child.kill()
                child.join(2)
                child.close()
        close_service(service)


def test_stalled_result_receive_cannot_block_deadline_or_retirement(tmp_path, monkeypatch):
    service = fake_service(tmp_path, "normal", deadline=1, retire_timeout=.2)
    service._clock = clock = DiscoveryClock()
    processes, pipes = track_resources(service, monkeypatch)
    original_pipe = service._context.Pipe
    entered, released = threading.Event(), threading.Event()

    def stalled_pipe(*args, **kwargs):
        receive, send = original_pipe(*args, **kwargs)
        original_receive = receive.recv_bytes
        original_close = receive.close

        def stalled_receive(*args, **kwargs):
            payload = original_receive(*args, **kwargs)
            entered.set()
            assert released.wait(3)
            return payload

        def close_receive():
            released.set()
            original_close()

        monkeypatch.setattr(receive, "recv_bytes", stalled_receive)
        monkeypatch.setattr(receive, "close", close_receive)
        return receive, send

    monkeypatch.setattr(service._context, "Pipe", stalled_pipe)
    try:
        service.start()
        assert entered.wait(2)
        clock.advance(2)
        assert service.wait_idle(2)
        event = service.poll_result()
        assert "deadline" in " ".join(event.result.diagnostics)
        assert event.handles_released
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
        assert not any(thread.name == "ve-discovery-receiver" for thread in threading.enumerate())
    finally:
        released.set()
        close_service(service)
        for child in processes:
            if not child._closed:
                child.kill()
                child.join(2)
                child.close()


def test_invalid_address_cannot_hide_a_simultaneous_ambiguous_alias():
    sdk = DiscoverySDK([discovery_record(), discovery_record(address=42, machine_id="other")])
    result = discovery.discover_devices(sdk)
    assert result.devices == ()
    assert "ambiguous" in " ".join(result.diagnostics)
    assert sdk.trace == [("devices",)]


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_unconfirmed_death_retains_ownership_and_does_not_claim_idle(tmp_path, monkeypatch, action):
    events = []
    service = fake_service(tmp_path, "block_probe", on_result=events.append,
                           deadline=.7, retire_timeout=.05)
    service._clock = clock = DiscoveryClock()
    processes, pipes = track_resources(service, monkeypatch)
    real_kill = None
    try:
        service.start()
        wait_for_stage(tmp_path, "probe")
        child = processes[0]
        real_kill = child.kill
        monkeypatch.setattr(child, "terminate", lambda: None)
        monkeypatch.setattr(child, "kill", lambda: None)
        clock.advance(1)
        wait_until(lambda: events)
        assert not events[0].handles_released
        assert "unconfirmed" in " ".join(events[0].result.diagnostics)
        began = time.monotonic()
        getattr(service, action)()
        assert time.monotonic() - began < .2
        assert not service.wait_idle(.05)
        if action == "close":
            assert not service.wait_closed(.05)
        assert child.is_alive()
        monkeypatch.setattr(child, "kill", real_kill)
        assert service.wait_idle(2)
        assert child._closed
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        if real_kill is not None:
            monkeypatch.setattr(processes[0], "kill", real_kill)
        close_service(service)


def test_empty_inventory_returns_actionable_unavailable_diagnostic(tmp_path):
    service = fake_service(tmp_path, "empty")
    try:
        service.start()
        assert service.wait_idle(3)
        event = service.poll_result()
        assert event.status == "unavailable"
        assert "no verified" in " ".join(event.result.diagnostics)
    finally:
        close_service(service)


def test_discovery_uses_task3_binding_with_injected_native_dll_only():
    from unit_test.base.ve3668n_fakes import FakeVkDaqDLL

    dll = FakeVkDaqDLL()
    dll.channels = b"Dev1/AIN8,Dev1/AIN2,Dev1/MIN1,Dev1/DIN1,Dev1/RAWAIN1"
    dll.attributes[b"DeviceStatus"] = b"ready"
    with vkinging_sdk.VkDaqClient(dll=dll) as sdk:
        device = discovery.resolve_device(sdk, "test-machine-1", (7, 1))
    assert device["physical_channels"] == (7, 1)
    assert device["name"] == "Dev1"
    assert {call["operation"] for call in dll.trace} == {
        "VkDaqGetDevices", "VkDaqGetDeviceAttribute", "VkDaqGetChannels"}
    assert all(call["args"][0] == b"Dev1" for call in dll.trace
               if call["operation"] == "VkDaqGetDeviceAttribute")


def test_active_callback_does_not_block_public_lifecycle_and_no_later_callback(tmp_path):
    entered, release = threading.Event(), threading.Event()
    events = []

    def callback(event):
        events.append(event)
        entered.set()
        assert release.wait(3)

    service = fake_service(tmp_path, "normal", on_result=callback)
    try:
        service.start()
        assert entered.wait(3)
        began = time.monotonic()
        service.refresh()
        service.cancel()
        service.close()
        assert time.monotonic() - began < .2
        release.set()
        assert service.wait_closed(2)
        assert len(events) == 1
        assert service.poll_result() is None
    finally:
        release.set()
        close_service(service)


def test_duplicate_identity_is_rejected_even_when_other_address_is_malformed():
    result = discovery.discover_devices(DiscoverySDK([
        discovery_record(), discovery_record("Dev2", address=42)]))
    assert result.devices == ()
    assert "ambiguous" in " ".join(result.diagnostics)


def test_sdk_close_failure_does_not_erase_primary_discovery_failure(tmp_path):
    service = fake_service(tmp_path, "probe_and_close_error")
    try:
        service.start()
        assert service.wait_idle(3)
        event = service.poll_result()
        assert event.status == "unavailable"
        diagnostic = " ".join(event.result.diagnostics)
        assert "fake primary probe failure" in diagnostic
        assert "fake SDK close failure" in diagnostic
        assert event.handles_released
    finally:
        close_service(service)


@pytest.mark.parametrize("thread_name", ["ve-discovery-receiver", "ve-discovery-launcher"])
def test_service_thread_start_failure_still_closes_child_and_ipc(tmp_path, monkeypatch, thread_name):
    service = fake_service(tmp_path, "normal")
    processes, pipes = track_resources(service, monkeypatch)
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == thread_name:
            raise RuntimeError("fake service thread startup failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        service.start()
        assert service.wait_idle(2)
        event = service.poll_result()
        assert event.status == "unavailable"
        assert "fake service thread startup failure" in " ".join(event.result.diagnostics)
        assert event.handles_released
        assert all(child._closed for child in processes)
        assert all(endpoint.closed for endpoint in pipes)
    finally:
        close_service(service)
        for child in processes:
            if not child._closed:
                if child.is_alive():
                    child.kill()
                child.join(2)
                child.close()
