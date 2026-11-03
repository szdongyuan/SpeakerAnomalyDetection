"""Coordinator tests use controlled blocking, never a real native timeout."""
import ast
from dataclasses import FrozenInstanceError
import logging
import os
from pathlib import Path
import threading

import pytest

from base.vkinging_input import VkingingCaptureRequest
from base.vkinging_service import CaptureResult, VkingingService
from unit_test.base.vkinging_fakes import FakeSDK


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def request(request_id="capture", **kwargs):
    values = dict(request_id=request_id, purpose="main", device={
        "hostapi_name": "vkinging", "hardware_id": "id", "name": "dev",
        "physical_channels": (0, 15), "samplerate": 4000, "bit_depth": 32},
        channels=(15, 0), sample_rate=4000, bit_depth=32, range_index=0,
        target_samples=4000)
    values.update(kwargs)
    return VkingingCaptureRequest(**values)


def wait_state(service, state):
    with service._condition:
        assert service._condition.wait_for(lambda: service.state == state, timeout=2)


def finish(service, release=None):
    if release is not None:
        release.set()
    service.close()
    service._worker.join(timeout=2)
    assert not service._worker.is_alive()


def terminals(service):
    return [event for event in service.poll_events() if event.kind in (
        "succeeded", "failed", "cancelled")]


def test_discovery_and_capture_use_one_daemon_owner_in_this_process():
    owner_calls, sdks = [], []

    def factory():
        owner_calls.append((threading.get_ident(), os.getpid()))
        sdk = FakeSDK()
        sdks.append(sdk)
        return sdk

    def capture(sdk, req, control):
        sdk.create_task(req.request_id)
        control.set_phase("capturing")
        control.report_progress(4000)
        sdk.stop_task(req.request_id)
        sdk.clear_task(req.request_id)
        return CaptureResult({"frames": 4000}, resources_released=True)

    service = VkingingService(sdk_factory=factory, capture_handler=capture)
    try:
        assert service._worker.daemon
        assert service.submit_discovery("discovery")
        wait_state(service, "idle")
        first = terminals(service)
        assert [(e.request_id, e.kind) for e in first] == [("discovery", "succeeded")]
        assert first[0].value.devices == ()
        assert service.start_capture(request())
        wait_state(service, "idle")
        event, = terminals(service)
        assert (event.request_id, event.value["frames"]) == ("capture", 4000)
        calls = [call for sdk in sdks for call in sdk.calls]
        assert len({call[2] for call in calls}) == 1
        assert {call[3] for call in calls} == {os.getpid()}
        assert owner_calls == [(calls[0][2], os.getpid())] * 2
        assert calls[0][2] != threading.get_ident()
        assert all(sdk.closed for sdk in sdks)
        with pytest.raises(TypeError):
            event.value["frames"] = 0
        with pytest.raises(FrozenInstanceError):
            event.kind = "failed"
    finally:
        finish(service)


@pytest.mark.parametrize("operation,phase,deadline,reason", [
    ("discovery", None, 10, "discovery_timeout"),
    ("capture", None, 10, "start_timeout"),
    ("capture", "capturing", 3, "read_stall"),
    ("capture", "stopping", 3, "release_timeout"),
    ("capture", "releasing", 3, "release_timeout"),
])
def test_timeouts_revoke_results_without_touching_sdk_on_caller(
        operation, phase, deadline, reason):
    entered, release = threading.Event(), threading.Event()
    clock, sdks, controls = Clock(), [], []

    def factory():
        sdk = FakeSDK()
        sdks.append(sdk)
        if operation == "discovery":
            original = sdk.get_devices

            def blocked():
                original()
                entered.set()
                assert release.wait(2)
                return ()

            sdk.get_devices = blocked
        return sdk

    def capture(sdk, req, control):
        controls.append(control)
        if phase:
            control.set_phase(phase)
        entered.set()
        assert release.wait(2)
        return CaptureResult({"late": True}, resources_released=True)

    service = VkingingService(sdk_factory=factory, capture_handler=capture, clock=clock)
    try:
        assert (service.submit_discovery("old") if operation == "discovery"
                else service.start_capture(request("old")))
        assert entered.wait(2)
        assert not service.submit_discovery("overlap")
        clock.now = deadline
        service.tick(clock.now)
        assert service.state == "unavailable"
        event, = terminals(service)
        assert (event.request_id, event.kind, event.reason) == ("old", "failed", reason)
        assert not service.start_capture(request("new"))
        assert not sdks[0].closed
        assert not any(call[0] in ("stop_task", "clear_task") for call in sdks[0].calls)
        if controls:
            assert controls[0].cancelled
        release.set()
        wait_state(service, "idle")
        assert terminals(service) == []
        assert service.submit_discovery("retry")
        wait_state(service, "idle")
        assert [e.request_id for e in terminals(service)] == ["retry"]
    finally:
        finish(service, release)


def test_raw_progress_resets_stall_clock_but_not_total_capture_budget():
    entered, progress, progressed, release = (threading.Event() for _ in range(4))
    clock = Clock()

    def capture(sdk, req, control):
        control.set_phase("capturing")
        entered.set()
        assert progress.wait(2)
        control.report_progress(1)
        progressed.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, clock=clock)
    try:
        assert service.start_capture(request(discard_initial_samples=4000))
        assert entered.wait(2)
        clock.now = 2.5
        progress.set()
        assert progressed.wait(2)
        service.tick(3)
        assert service.state == "capturing"
        # Total budget is (target + discard) / rate + 5 = 7 seconds.
        clock.now = 7
        service.tick(7)
        event, = terminals(service)
        assert event.reason == "capture_timeout"
        assert event.raw_frames == 1
    finally:
        progress.set()
        finish(service, release)


def test_repeated_frame_count_does_not_hide_sampling_stall():
    entered, repeat, repeated, release = (threading.Event() for _ in range(4))
    clock = Clock()

    def capture(sdk, req, control):
        control.set_phase("capturing")
        control.report_progress(2)
        entered.set()
        assert repeat.wait(2)
        control.report_progress(2)
        repeated.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, clock=clock)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        clock.now = 2
        repeat.set()
        assert repeated.wait(2)
        clock.now = 3
        service.tick(3)
        assert terminals(service)[0].reason == "read_stall"
    finally:
        repeat.set()
        finish(service, release)


def test_cancel_is_one_terminal_and_release_timeout_does_not_allow_overlap():
    entered, release = threading.Event(), threading.Event()
    clock, controls = Clock(), []

    def capture(sdk, req, control):
        controls.append(control)
        control.set_phase("capturing")
        entered.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, clock=clock)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        assert not service.cancel("wrong")
        assert service.cancel("capture")
        assert not service.cancel("capture")
        assert controls[0].cancelled
        assert service.state == "stopping"
        assert [e.kind for e in terminals(service)] == ["cancelled"]
        clock.now = 3
        service.tick(3)
        assert service.state == "unavailable"
        assert not service.submit_discovery("new")
        release.set()
        wait_state(service, "idle")
        assert terminals(service) == []
    finally:
        finish(service, release)


def test_release_acknowledgement_survives_new_work_until_consumer_drains_it():
    entered, release = threading.Event(), threading.Event()

    def capture(*args):
        entered.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, terminal_capacity=1)
    try:
        service.start_capture(request("old"))
        assert entered.wait(2)
        service.cancel("old")
        # The terminal was consumed before native/file cleanup returned.
        cancelled, = terminals(service)
        assert not cancelled.resources_released
        release.set()
        wait_state(service, "idle")
        accepted = service.submit_discovery("next")
        if accepted:
            wait_state(service, "idle")
        events = service.poll_events()
        proof, = [event for event in events if event.request_id == "old" and event.kind == "released"]
        assert proof.resources_released
        assert len(events) <= 3  # One terminal, one reliable release ack, one transient.
        if not accepted:
            assert service.submit_discovery("after-ack")
            wait_state(service, "idle")
    finally:
        finish(service, release)


@pytest.mark.parametrize("failure", ["uncertain", "task_cleanup", "sdk_close"])
def test_uncertain_or_failed_release_permanently_disables_backend(failure):
    sdk = FakeSDK()
    if failure == "sdk_close":
        sdk.failures["close"] = OSError("close broke")

    def capture(sdk, req, control):
        return CaptureResult(resources_released=failure == "sdk_close",
                             cleanup_error="Clear failed" if failure == "task_cleanup" else None)

    service = VkingingService(sdk_factory=lambda: sdk, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "unavailable")
        event, = terminals(service)
        assert event.kind == "failed"
        assert not event.resources_released
        assert event.cleanup_error
        assert not service.submit_discovery("retry")
        service.tick(100000)
        assert service.state == "unavailable"
    finally:
        finish(service)


def test_worker_logs_primary_and_cleanup_failure_without_losing_either(caplog):
    sdk = FakeSDK()
    sdk.failures["close"] = OSError("cleanup failure")

    def capture(sdk, req, control):
        raise RuntimeError("primary failure")

    service = VkingingService(sdk_factory=lambda: sdk, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "unavailable")
        event, = terminals(service)
        assert "primary failure" in event.reason
        assert "cleanup failure" in event.cleanup_error
        assert "primary failure" in caplog.text and "cleanup failure" in caplog.text
        assert "capture" in caplog.text and "starting" in caplog.text
    finally:
        finish(service)


def test_returned_cleanup_failure_is_diagnosed_without_an_exception(caplog):
    service = VkingingService(sdk_factory=FakeSDK, capture_handler=lambda *args:
                             CaptureResult(cleanup_error="Clear rejected with code=-9"))
    try:
        service.start_capture(request())
        wait_state(service, "unavailable")
        assert "Clear rejected with code=-9" in caplog.text
        assert "capture" in caplog.text and "raw_frames=0" in caplog.text
    finally:
        finish(service)


def test_handler_can_report_cleanup_failure_while_preserving_primary_exception():
    def capture(sdk, req, control):
        try:
            raise OSError("primary read failed")
        finally:
            control.report_cleanup_failure("first Clear failure")
            control.report_cleanup_failure("later file close failure")

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "unavailable")
        event, = terminals(service)
        assert event.reason == "primary read failed"
        assert event.cleanup_error == "first Clear failure; later file close failure"
    finally:
        finish(service)


def test_exception_with_explicit_cleanup_confirmation_allows_retry():
    def capture(sdk, req, control):
        control.confirm_released()
        raise OSError("write failed after closing")

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "idle")
        event, = terminals(service)
        assert event.kind == "failed" and event.resources_released
        assert service.submit_discovery("retry")
        wait_state(service, "idle")
    finally:
        finish(service)


def test_finalization_runs_after_sdk_close_and_is_not_a_sampling_stall():
    entered, release = threading.Event(), threading.Event()
    clock, sdk, committed = Clock(), FakeSDK(), []

    def finalize(control):
        assert sdk.closed
        entered.set()
        assert release.wait(2)
        return CaptureResult({"path": "validated.wav"}, resources_released=True,
                             commit=lambda control: committed.append("done") or {"path": "saved.wav"})

    def capture(sdk, req, control):
        control.set_phase("capturing")
        return CaptureResult(resources_released=True, finalize=finalize)

    service = VkingingService(sdk_factory=lambda: sdk, capture_handler=capture, clock=clock)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        clock.now = 100
        service.tick(100)
        assert service.state == "finalizing"
        assert not service.submit_discovery("overlap")
        assert terminals(service) == []
        release.set()
        wait_state(service, "idle")
        event, = terminals(service)
        assert event.value["path"] == "saved.wav"
        assert committed == ["done"]
    finally:
        finish(service, release)


@pytest.mark.parametrize("invalidate", ["cancel", "close"])
def test_invalidation_during_finalization_never_commits(invalidate):
    entered, release = threading.Event(), threading.Event()
    commits = []

    def finalize(control):
        entered.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True,
                             commit=lambda control: commits.append("forbidden"))

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=lambda *args:
                             CaptureResult(resources_released=True, finalize=finalize))
    try:
        service.start_capture(request())
        assert entered.wait(2)
        if invalidate == "cancel":
            service.cancel("capture")
        else:
            service.close()
        release.set()
        wait_state(service, "idle" if invalidate == "cancel" else "closed")
        assert commits == []
        assert all(e.kind != "succeeded" for e in terminals(service))
    finally:
        finish(service, release)


def test_late_finalization_cannot_reset_cancel_release_deadline():
    entered, proceed, finalizing, release = (threading.Event() for _ in range(4))
    clock = Clock()

    def finalize(control):
        finalizing.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    def capture(*args):
        entered.set()
        assert proceed.wait(2)
        return CaptureResult(resources_released=True, finalize=finalize)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, clock=clock)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        service.cancel("capture")
        clock.now = 2
        proceed.set()
        assert finalizing.wait(2)
        clock.now = 3
        service.tick(3)
        assert service.state == "unavailable"
        assert [event.kind for event in terminals(service)] == ["cancelled"]
        release.set()
        wait_state(service, "idle")
    finally:
        proceed.set()
        finish(service, release)


@pytest.mark.parametrize("stage", ["finalize", "commit"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_postprocessing_failure_requires_new_cleanup_proof(stage, confirmed):
    def fail(control):
        if confirmed:
            control.confirm_released()
        raise OSError("file operation failed")

    def capture(*args):
        return CaptureResult(resources_released=True, **{stage: fail})

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "idle" if confirmed else "unavailable")
        event, = terminals(service)
        assert event.kind == "failed"
        assert event.resources_released is confirmed
        assert "file operation failed" in event.reason
    finally:
        finish(service)


def test_terminal_capacity_backpressures_new_work_and_progress_is_coalesced():
    entered, release = threading.Event(), threading.Event()

    def capture(sdk, req, control):
        control.set_phase("capturing")
        for frames in range(10000):
            control.report_progress(frames)
        entered.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture, terminal_capacity=2)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        events = service.poll_events()
        assert len(events) == 1 and events[0].raw_frames == 9999
        release.set()
        wait_state(service, "idle")
        assert service.submit_discovery("second")
        wait_state(service, "idle")
        assert not service.submit_discovery("full")
        events = service.poll_events()
        assert len(events) <= 3
        assert [(e.request_id, e.kind) for e in events if e.kind == "succeeded"] == [
            ("capture", "succeeded"), ("second", "succeeded")]
        assert service.submit_discovery("after-drain")
        wait_state(service, "idle")
    finally:
        finish(service, release)


def test_close_is_nonblocking_suppresses_events_and_never_replaces_owner():
    entered, release = threading.Event(), threading.Event()
    clock, sdks = Clock(), []

    def factory():
        sdk = FakeSDK()
        sdks.append(sdk)
        return sdk

    def capture(sdk, req, control):
        entered.set()
        assert release.wait(2)
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=factory, capture_handler=capture, clock=clock)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        owner = service._worker
        service.close()
        assert service.state == "closing"
        assert not sdks[0].closed
        assert not service.submit_discovery("after-close")
        clock.now = 3
        service.tick(3)
        assert service.state == "closed"
        assert owner.is_alive() and owner.daemon and service._worker is owner
        assert service.poll_events() == []
        release.set()
        owner.join(2)
        assert not owner.is_alive()
        assert sdks[0].closed
        assert service.poll_events() == []
    finally:
        finish(service, release)


@pytest.mark.parametrize("action", ["close", "cancel", "tick"])
def test_blocked_commit_does_not_block_caller_paths(action):
    entered, release, returned = (threading.Event() for _ in range(3))
    clock, results = Clock(), []

    def commit(control):
        entered.set()
        assert release.wait(2)
        return {"path": "committed.wav"}

    service = VkingingService(sdk_factory=FakeSDK, clock=clock, capture_handler=lambda *args:
                             CaptureResult(resources_released=True, commit=commit))

    def call():
        try:
            if action == "close":
                service.close()
            elif action == "cancel":
                results.append(service.cancel("capture"))
            else:
                clock.now = 100
                service.tick(100)
        finally:
            returned.set()

    caller = threading.Thread(target=call, daemon=True)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        caller.start()
        assert returned.wait(0.5), "caller waited for blocked filesystem commit"
        assert not service.submit_discovery("overlap")
        if action == "cancel":
            assert results == [False]  # Commit already claimed publication.
        if action == "close":
            clock.now = 3
            service.tick(3)
            assert service.state == "closed"
        else:
            assert service.state == "committing"
        release.set()
        wait_state(service, "closed" if action == "close" else "idle")
        events = terminals(service)
        assert [event.kind for event in events] == ([] if action == "close" else ["succeeded"])
    finally:
        release.set()
        if caller.ident is not None:
            caller.join(2)
            assert not caller.is_alive()
        finish(service)


def test_blocked_diagnostic_logging_does_not_block_close():
    entered, release, returned = (threading.Event() for _ in range(3))

    class BlockingLogger:
        def error(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=lambda *args:
                             CaptureResult(cleanup_error="cannot clear"))
    service._logger = BlockingLogger()

    def close():
        service.close()
        returned.set()

    caller = threading.Thread(target=close, daemon=True)
    try:
        service.start_capture(request())
        assert entered.wait(2)
        caller.start()
        assert returned.wait(0.5), "close waited for blocked logging I/O"
        assert service.state == "closing"
        assert service.poll_events() == []
    finally:
        release.set()
        if caller.ident is not None:
            caller.join(2)
            assert not caller.is_alive()
        finish(service)


class BrokenTextError(Exception):
    def __str__(self):
        raise RuntimeError("exception text is broken")

    def __repr__(self):
        raise RuntimeError("exception repr must not be used as a fallback")


def test_raising_log_handler_does_not_strand_confirmed_release():
    class RaisingHandler(logging.Handler):
        def emit(self, record):
            raise OSError("diagnostic sink failed")

    logger = logging.Logger("vkinging-test-raising-handler")
    logger.addHandler(RaisingHandler())

    def capture(sdk, req, control):
        control.confirm_released()
        raise OSError("primary write failure")

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    service._logger = logger
    try:
        service.start_capture(request())
        wait_state(service, "idle")
        event, = terminals(service)
        assert event.kind == "failed" and event.resources_released
        assert event.reason == "primary write failure"
        assert event.cleanup_error is None
        assert event.diagnostic_error == "diagnostic sink failed"
        assert service._worker.is_alive()
        assert service.submit_discovery("retry")
        wait_state(service, "idle")
        assert [event.kind for event in terminals(service)] == ["succeeded"]
    finally:
        finish(service)


def test_broken_primary_exception_text_does_not_strand_confirmed_release():
    def capture(sdk, req, control):
        control.confirm_released()
        raise BrokenTextError()

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "idle")
        event, = terminals(service)
        assert event.kind == "failed" and event.resources_released
        assert "BrokenTextError" in event.reason
        assert "unavailable" in event.reason
        assert event.cleanup_error is None
        assert service._worker.is_alive()
        assert service.submit_discovery("retry")
        wait_state(service, "idle")
        assert [event.kind for event in terminals(service)] == ["succeeded"]
    finally:
        finish(service)


def test_broken_logging_exception_cannot_bypass_reliable_release_acknowledgement():
    entered, release = threading.Event(), threading.Event()

    class RaisingLogger:
        def error(self, *args, **kwargs):
            raise BrokenTextError()

    def capture(sdk, req, control):
        entered.set()
        assert release.wait(2)
        control.confirm_released()
        raise OSError("late read error after cancellation")

    service = VkingingService(sdk_factory=FakeSDK, capture_handler=capture)
    service._logger = RaisingLogger()
    try:
        service.start_capture(request())
        assert entered.wait(2)
        assert service.cancel("capture")
        cancelled, = terminals(service)
        assert cancelled.kind == "cancelled"
        release.set()
        wait_state(service, "idle")
        assert service._worker.is_alive()
        assert not service.submit_discovery("before-ack")
        ack, = [event for event in service.poll_events() if event.kind == "released"]
        assert ack.resources_released and ack.cleanup_error is None
        assert "BrokenTextError" in ack.diagnostic_error
        assert "unavailable" in ack.diagnostic_error
        assert service.submit_discovery("after-ack")
        wait_state(service, "idle")
        assert [event.kind for event in terminals(service)] == ["succeeded"]
    finally:
        finish(service, release)


@pytest.mark.parametrize("primary_error", [False, True])
def test_broken_sdk_close_exception_text_preserves_real_cleanup_failure(primary_error):
    sdk = FakeSDK()
    sdk.failures["close"] = BrokenTextError()

    def capture(sdk, req, control):
        if primary_error:
            control.confirm_released()
            raise OSError("primary read failure")
        return CaptureResult(resources_released=True)

    service = VkingingService(sdk_factory=lambda: sdk, capture_handler=capture)
    try:
        service.start_capture(request())
        wait_state(service, "unavailable")
        event, = terminals(service)
        assert event.kind == "failed" and not event.resources_released
        assert "BrokenTextError" in event.cleanup_error
        assert "unavailable" in event.cleanup_error
        if primary_error:
            assert event.reason == "primary read failure"
        else:
            assert "BrokenTextError" in event.reason
        assert service._worker.is_alive()
        assert not service.submit_discovery("unsafe-retry")
    finally:
        finish(service)


def test_default_capture_handler_owns_native_calls_and_publishes_after_sdk_close(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from base.vkinging_wav import VkingingVoltageWriter
    from unit_test.base.test_vkinging_capture import request as capture_request, sdk_for
    req = capture_request(tmp_path, target=2, discard=0)
    values = np.array([[2.5, -3., 8.], [9., 10., 11.]])
    sdk = sdk_for(req, [(values.ravel(), 2)])
    factories, file_calls = [], []
    def factory():
        factories.append(threading.get_ident())
        return sdk
    original_close, original_finalize = VkingingVoltageWriter.close, VkingingVoltageWriter.finalize
    def close(writer):
        file_calls.append(('close', threading.get_ident(), sdk.closed))
        return original_close(writer)
    def finalize(writer, **kwargs):
        file_calls.append(('finalize', threading.get_ident(), sdk.closed))
        assert sdk.closed
        return original_finalize(writer, **kwargs)
    monkeypatch.setattr(VkingingVoltageWriter, 'close', close)
    monkeypatch.setattr(VkingingVoltageWriter, 'finalize', finalize)
    service = VkingingService(sdk_factory=factory)
    try:
        assert service.start_capture(req)
        wait_state(service, "idle")
        event, = terminals(service)
        assert event.kind == 'succeeded' and event.resources_released
        assert event.value['frames'] == 2
        actual, _ = sf.read(event.value['path'], dtype='float64', always_2d=True)
        np.testing.assert_array_equal(actual, values)
        owner, = factories
        assert owner != threading.get_ident()
        assert {call[2] for call in sdk.calls} == {owner}
        assert {call[1] for call in file_calls} == {owner}
        assert file_calls[0] == ('close', owner, False)
        assert file_calls[1] == ('finalize', owner, True)
        assert [c[0] for c in sdk.calls][-3:] == ['stop_task', 'clear_task', 'close']
    finally:
        finish(service)


def test_default_capture_uses_injected_service_clock_without_gui_tick(tmp_path):
    import ctypes
    from unit_test.base.test_vkinging_capture import request as capture_request, sdk_for
    req = capture_request(tmp_path)
    sdk, clock = sdk_for(req), Clock()
    def read(*args, **kwargs):
        clock.now += 3
        return (ctypes.c_double * (3 * kwargs['samples_per_channel']))(), 0
    sdk.read_task_data = read
    service = VkingingService(sdk_factory=lambda: sdk, clock=clock)
    try:
        assert service.start_capture(req)
        wait_state(service, 'idle')
        event, = terminals(service)
        assert event.kind == 'failed' and 'read_stall' in event.reason
        assert event.resources_released and sdk.closed
    finally:
        finish(service)


def test_no_process_or_qt_worker_mechanism_is_imported():
    import base.vkinging_service as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names]
    imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("multiprocessing", "subprocess", "PyQt")) for name in imports)
