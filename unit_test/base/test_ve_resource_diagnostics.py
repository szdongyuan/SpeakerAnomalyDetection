"""Fake SDK evidence for observational resource logging only."""
import json
import logging
import os
import threading
import time

import pytest

from base.ve3668n_discovery import discover_devices
from base.ve3668n_resource import VeResourceController
from base.ve_startup_policy import VeStartupBudget
from base.recording_process_protocol import VePrewarmRequest
from base.vkinging_sdk import VkDaqError
from unit_test.base.ve3668n_fakes import (
    CaptureSDK, DiscoveryClock, DiscoverySDK, capture_request, device_info, discovery_record,
)


def sink():
    records = []
    logger = logging.Logger("ve-test", logging.INFO)

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger.addHandler(Handler())
    return logger, records


def events(records, event=None, phase=None):
    result = []
    for record in list(records):
        message = record.getMessage()
        assert message.startswith("VE resource diagnostic ")
        assert len(message.encode("utf-8")) <= 8192
        value = json.loads(message[len("VE resource diagnostic "):])
        assert record.levelno == (logging.WARNING if value["event"] in
                                  ("ERROR", "TIMEOUT") else logging.INFO)
        if (event is None or value["event"] == event) and (phase is None or value["phase"] == phase):
            result.append(value)
    return result


def adapter(controller, tmp_path, request_id="A"):
    return controller.stream(
        request=capture_request(tmp_path / "unused.wav", request_id=request_id),
        callback=lambda *args: None, fail=lambda *args: None,
        stop_event=threading.Event())


def test_success_reuse_correlation_and_native_order(tmp_path):
    logger, records = sink()
    sdk = CaptureSDK()
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger, generation=17)
    for request_id in ("A", "B"):
        stream = adapter(controller, tmp_path, request_id)
        assert stream.start()
        assert stream.completed.wait(2)
    assert controller.release(1).success
    values = events(records)
    phases = [value["phase"] for value in values if value["event"] == "BEGIN"]
    assert phases == ["startup_attempt", "initialize", "sdk_open", "device", "get_devices",
                      "get_device_attribute", "get_device_attribute", "get_channels",
                      "get_device_attribute", "create_task", "create_iepe_voltage_channel",
                      "configure_sample_clock", "start_task", "verify_actual_sample_rate",
                      "stop_task", "clear_task", "close"]
    bound = events(records, "END", "bind")
    assert [value["request_id"] for value in bound] == ["A", "B"]
    assert all(value["elapsed_ms"] > 0 for value in bound)
    assert len({value["task"] for value in values}) == 1
    assert all(value["generation"] == 17 and value["pid"] == os.getpid() for value in values)
    assert all(value["machine_id"] == "test-machine-1" and value["channels"] == [7, 1]
               and value["sample_rate"] == 51200 and value["range_min"] == -10.0
               and value["range_max"] == 10.0 for value in values)
    status = next(value for value in events(records, "END", "get_device_attribute")
                  if value["attribute"] == "DeviceStatus")
    assert status["result"] == "ready"
    start = events(records, "BEGIN", "start_task")[0]
    assert start["device_status"]["value"] == "ready"
    assert start["device_status"]["source"] == "discovery"
    assert start["device_status"]["age_ms"] >= 0
    assert all(value["elapsed_ms"] >= 0 for value in values if value["event"] == "END")
    release = events(records, "END", "release")[0]
    assert release["request_id"] == "B" and release["request_association"] == "last_bound"
    assert all(value["phase"] != "read_task_data" for value in values)
    assert tuple(op for op in sdk.operations() if op != "read_task_data") == (
        "get_devices", "get_device_attribute", "get_device_attribute", "get_channels",
        "get_device_attribute", "create_task", "create_iepe_voltage_channel",
        "configure_sample_clock", "start_task", "verify_actual_sample_rate",
        "stop_task", "clear_task", "close")


@pytest.mark.parametrize("blocked", ["start_task", "DeviceStatus"])
def test_startup_revoke_identifies_blocked_operation_before_return(tmp_path, blocked):
    logger, records = sink()
    entered, resume = threading.Event(), threading.Event()

    def block(*args):
        if blocked != "DeviceStatus" or args[-1] == "DeviceStatus":
            entered.set()
            assert resume.wait(3)

    sdk = CaptureSDK(hooks={"get_device_attribute" if blocked == "DeviceStatus" else blocked: block})
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger,
                                      generation=3, bind_timeout=.1)
    stream = adapter(controller, tmp_path)
    stream.startup_budget = VeStartupBudget.create(time.monotonic(), .3, .05, .05)
    try:
        assert not stream.start()
        assert entered.is_set()
        timeout, = events(records, "TIMEOUT", "startup_revoke")
        assert timeout["attempt"] == 1
        current = timeout["current_phase"]
        assert current["phase"] == ("get_device_attribute" if blocked == "DeviceStatus" else blocked)
        if blocked == "DeviceStatus":
            assert current["attribute"] == "DeviceStatus"
            assert timeout["device_status"] is None
        else:
            assert timeout["device_status"]["value"] == "ready"
        assert current["elapsed_ms"] > 0
        assert timeout["request_id"] == "A" and timeout["generation"] == 3
        assert timeout["owner_alive"] and timeout["owner_thread_id"] in sdk.owner_thread_ids
        assert 1 <= len(timeout["owner_stack"]) <= 12
        assert all(set(frame) == {"file", "function", "line"} for frame in timeout["owner_stack"])
        assert controller.failed and stream.failure_snapshot.stage == "startup_cleanup_timeout"
        if blocked == "start_task":
            assert not events(records, "END", "start_task")
    finally:
        resume.set()
        controller._owner.join(2)
    assert not controller._owner.is_alive()
    if blocked == "start_task":
        assert sdk.operations()[-3:] == ("stop_task", "clear_task", "close")
    else:
        assert sdk.operations()[-1:] == ("close",)
        assert sdk.calls("create_task") == 0
    assert controller.failed
    assert any(value["after_timeout"] for value in events(records, "END"))


@pytest.mark.parametrize("operation", ["stop_task", "clear_task"])
def test_release_timeout_identifies_pending_cleanup(tmp_path, operation):
    logger, records = sink()
    resume = threading.Event()
    sdk = CaptureSDK(hooks={operation: lambda *args: resume.wait(3)})
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger)
    stream = adapter(controller, tmp_path)
    assert stream.start() and stream.completed.wait(2)
    owner = controller._owner
    try:
        assert not controller.release(.1).success
        timeout, = events(records, "TIMEOUT", "release")
        assert timeout["resource_state"] == "RELEASING"
        assert timeout["current_phase"]["phase"] == operation
        assert timeout["request_association"] == "last_bound"
        assert timeout["owner_alive"] and timeout["owner_stack"]
    finally:
        resume.set()
        owner.join(2)
    assert sdk.operations()[-3:] == ("stop_task", "clear_task", "close")
    assert controller.failed


def test_cleanup_error_preserves_result_and_native_code(tmp_path):
    logger, records = sink()
    sdk = CaptureSDK(failures={"clear_task"})
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger)
    stream = adapter(controller, tmp_path)
    assert stream.start() and stream.completed.wait(2)
    outcome = controller.release(1)
    assert not outcome.success and "clear_task" in outcome.diagnostics[0]
    error, = events(records, "ERROR", "clear_task")
    assert error["error_code"] == -17 and "injected clear_task" in error["error_detail"]
    assert sdk.closed


def test_start_timestamp_precedes_diagnostic_delivery(tmp_path, monkeypatch):
    logger, records = sink()
    clock = DiscoveryClock()
    original = logger.log

    def deliver(level, message, payload):
        value = json.loads(payload)
        if value["event"] == "END" and value["phase"] == "start_task":
            clock.advance(.001)
        original(level, message, payload)

    monkeypatch.setattr(logger, "log", deliver)
    controller = VeResourceController(sdk_factory=CaptureSDK, clock=clock, logger=logger)
    stream = adapter(controller, tmp_path)
    try:
        assert stream.start() and stream.completed.wait(2)
        assert stream.started_at == 100.0
    finally:
        controller.release(1)


@pytest.mark.parametrize("status", ["ready", "", "unknown-value", VkDaqError("status", -17, "failed")])
def test_optional_status_observer_preserves_raw_result_and_query_trace(status):
    sdk = DiscoverySDK([discovery_record(status=status)])
    observations = []
    result = discover_devices(sdk, observer=lambda **event: observations.append(event))
    assert len(result.devices) == 1
    assert sdk.trace == [("devices",), ("MachineId", "Dev1"), ("Model", "Dev1"),
                         ("channels", "Dev1"), ("DeviceStatus", "Dev1")]
    observed = [value for value in observations if value.get("attribute") == "DeviceStatus"]
    assert observed[0]["event"] == "BEGIN"
    assert observed[1]["at"] >= observed[0]["at"]
    if isinstance(status, Exception):
        assert observed[1]["event"] == "ERROR" and observed[1]["error"] is status
        assert "-17" in result.diagnostics[0]
    else:
        assert observed[1]["event"] == "END" and observed[1]["result"] == status


def test_raising_observer_preserves_primary_error():
    def broken(**event):
        raise RuntimeError("sink failure")

    sdk = DiscoverySDK([discovery_record(status=VkDaqError("status", -17, "original"))])
    observed = discover_devices(sdk, observer=broken)
    baseline = DiscoverySDK(sdk.records)
    assert observed == discover_devices(baseline)
    assert sdk.trace == baseline.trace


@pytest.mark.parametrize("status", ["", "unknown", VkDaqError("DeviceStatus", -23, "native detail" * 1000)])
def test_prewarm_retains_optional_status_evidence_without_readiness_gate(status):
    logger, records = sink()
    sdk = CaptureSDK()
    sdk.record["DeviceStatus"] = status
    if isinstance(status, Exception):
        def fail_status(name, attribute):
            if attribute == "DeviceStatus":
                raise status
        sdk.hooks["get_device_attribute"] = fail_status
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger, generation=11)
    request = VePrewarmRequest.create("warm-3", device_info(), (7, 1), 51200, attempt=1)
    stream = controller.prewarm(request=request, fail=lambda *args: None)
    assert stream.start() and stream.completed.wait(2)
    assert controller.release(1).success
    start, = events(records, "BEGIN", "start_task")
    assert start["warmup_id"] == "warm-3" and start["request_id"] is None
    observed = start["device_status"]
    assert observed["observed_at"] <= start["at"] and observed["age_ms"] >= 0
    if isinstance(status, Exception):
        assert observed["error_code"] == -23 and len(observed["error_detail"]) <= 256
    else:
        assert observed["value"] == status
    assert sdk.calls("get_device_attribute") == 3


def test_cached_status_belongs_to_selected_machine():
    from base.ve_resource_diagnostics import VeResourceDiagnostics

    logger, records = sink()
    diag = VeResourceDiagnostics(logger)
    diag.context(machine_id="selected")
    sdk = DiscoverySDK([discovery_record(machine_id="selected", status="selected-value"),
                        discovery_record("Dev2", machine_id="other", status="other-value")])
    assert len(discover_devices(sdk, observer=diag.observe).devices) == 2
    diag.emit("BEGIN", "start_task")
    start, = events(records, "BEGIN", "start_task")
    assert start["device_status"]["value"] == "selected-value"


@pytest.mark.parametrize("failure", [None, "start_task"])
def test_raising_logger_cannot_change_binding_cleanup_or_error(tmp_path, failure):
    class BrokenLogger:
        def log(self, *args, **kwargs):
            raise RuntimeError("logger failure")

    sdk = CaptureSDK(failures=() if failure is None else {failure})
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=BrokenLogger())
    stream = adapter(controller, tmp_path)
    assert stream.start() is (failure is None)
    assert stream.completed.wait(2)
    if failure:
        controller._owner.join(2)
        assert controller.failure_snapshot.code == -17
        assert controller.failure_snapshot.stage == failure
    else:
        assert controller.release(1).success
    assert sdk.operations()[-3:] == ("stop_task", "clear_task", "close")


def test_payload_bounds_stack_failure_and_long_status(tmp_path, monkeypatch):
    from base import ve_resource_diagnostics as module

    logger, records = sink()
    diag = module.VeResourceDiagnostics(logger, generation=9)
    diag.context(request_id="长" * 20000, task="t", machine_id="m")
    diag.emit("ERROR", "example", error_detail="长" * 20000,
              owner_stack=[{"file": "f" * 20000, "function": "g" * 20000, "line": 4}] * 100,
              **{f"extra_{i}": "长" * 20000 for i in range(100)})
    assert events(records, "ERROR")[0]["truncated"]
    monkeypatch.setattr(module.sys, "_current_frames", lambda: (_ for _ in ()).throw(RuntimeError("frames failed")))
    diag.timeout("bind", threading.current_thread())
    value, = events(records, "TIMEOUT")
    assert "frames failed" in value["stack_error"]
    sdk = CaptureSDK()
    sdk.record["DeviceStatus"] = "长" * 20000
    controller = VeResourceController(sdk_factory=lambda: sdk, logger=logger)
    stream = adapter(controller, tmp_path)
    assert stream.start() and stream.completed.wait(2)
    assert controller.release(1).success
    events(records)


def test_worker_passes_borrowed_logger_and_generation(monkeypatch):
    from base import recording_worker as module

    logger, _ = sink()
    seen = []
    original = module.VeResourceController

    def construct(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    class Connection:
        def poll(self, timeout):
            return True

        def recv(self):
            raise EOFError

        def send(self, event):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module.LogManager, "set_log_handler", lambda route: logger)
    monkeypatch.setattr(module.multiprocessing, "parent_process", lambda: None)
    monkeypatch.setattr(module, "VeResourceController", construct)
    module.recording_worker(Connection(), Connection(), 42, None, {}, .5)
    assert seen[0]["logger"] is logger and seen[0]["generation"] == 42
