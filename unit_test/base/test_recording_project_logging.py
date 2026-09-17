"""Real file-routing regressions for recording diagnostics."""
import logging
import multiprocessing
import queue
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from concurrent_log_handler import ConcurrentRotatingFileHandler
import numpy as np
import pytest
import soundfile as sf

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingResult
from base.recording_service import RecordingCallbacks, RecordingService, RecordingSession
from unit_test.base.recording_process_fakes import ControlledWriter, FakeBackend, known_audio
from unit_test.base.test_recording_capture import request
from unit_test.base.test_ve3668n_capture import make_stream
from unit_test.base.ve3668n_fakes import CaptureSDK
from unit_test.logging_test_support import isolated_project_logger


def rotating_handlers(logger):
    return [handler for handler in logger.handlers
            if isinstance(handler, ConcurrentRotatingFileHandler)]


def read_project_log(state, source):
    for handler in state.logger.handlers:
        handler.flush()
    assert state.path.exists(), "business diagnostic did not reach the project file"
    data = state.path.read_bytes()
    assert re.search(rb"\[" + re.escape(source.encode("ascii")) + rb":[1-9][0-9]*\]", data)
    return data


def stop_service(service):
    service.shutdown()
    assert service.closed.wait(5)
    for thread in service.threads:
        thread.join(3)
        assert not thread.is_alive()


def test_recording_capture_acquires_project_handler(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        capture = RecordingCapture(request(tmp_path), backend=FakeBackend())
        assert capture._logger is state.logger
        assert len(rotating_handlers(state.logger)) == 1


def test_recording_service_acquires_handler_before_supervisor(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        original = RecordingService._start_thread
        starts = []

        def start(service, thread, **kwargs):
            assert service._logger is state.logger
            assert len(rotating_handlers(state.logger)) == 1
            starts.append(thread)
            return original(service, thread, **kwargs)

        monkeypatch.setattr(RecordingService, "_start_thread", start)
        service = RecordingService()
        try:
            assert starts == [service._supervisor]
            assert service._supervisor.is_alive()
        finally:
            stop_service(service)


def test_finalizing_record_preserves_project_source_time_and_once_only_callback(
        tmp_path, monkeypatch, caplog):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = RecordingService()
        callbacks = []
        session = RecordingSession(service, request(tmp_path), RecordingCallbacks(
            finalizing=lambda session: callbacks.append(time.monotonic())))
        before = time.monotonic()
        try:
            service._notify_finalizing(session)
            service._notify_finalizing(session)
        finally:
            stop_service(service)
        after = time.monotonic()
        assert len(callbacks) == 1
        record, = [r for r in caplog.records if "stage=finalizing_observed" in r.msg]
        assert record.name == "core" and record.levelno == logging.INFO
        assert record.filename == "recording_service.py"
        assert record.funcName == "_notify_finalizing"
        assert record.args[0] == session.request.request_id
        assert before <= record.args[1] <= callbacks[0] <= after
        logged = read_project_log(state, "recording_service.py")
        assert logged.count(b"stage=finalizing_observed") == 1
        assert len(rotating_handlers(state.logger)) == 1


@pytest.mark.parametrize("guard", ["cancel_requested", "_terminal"])
def test_finalizing_guards_do_not_start_logging_consumer(tmp_path, guard):
    service = RecordingService()
    calls = []
    session = RecordingSession(service, request(tmp_path), RecordingCallbacks(
        finalizing=lambda session: calls.append(session)))
    setattr(session, guard, True)
    try:
        service._notify_finalizing(session)
        assert calls == []
        assert service._timing_logger.thread is None
    finally:
        stop_service(service)


def test_ve_stream_acquires_project_handler(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        stream, *_ = make_stream(tmp_path)
        try:
            assert stream._logger is state.logger
            assert len(rotating_handlers(state.logger)) == 1
        finally:
            stream.close()


def test_instances_share_handler_that_survives_service_and_stream_shutdown(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = RecordingService()
        stream = None
        try:
            handler, = rotating_handlers(state.logger)
            stream, _, _, _, failures = make_stream(tmp_path)
            capture = RecordingCapture(request(tmp_path), backend=FakeBackend())
            assert stream._logger is capture._logger is service._logger is state.logger
            assert rotating_handlers(state.logger) == [handler]
            assert stream.start()
            assert stream.done.wait(3)
        finally:
            if stream is not None:
                stream.close()
                if stream._owner is not None:
                    stream._owner.join(3)
                    assert not stream._owner.is_alive()
            stop_service(service)
        assert failures == [] and stream.handles_released
        assert rotating_handlers(state.logger) == [handler]
        service._diagnose("shared-handler-after-shutdown-probe")
        logged = read_project_log(state, "recording_service.py")
        assert logged.count(b"core ERROR shared-handler-after-shutdown-probe") == 1


def test_capture_preview_failure_reaches_project_file_without_invalidating_audio(
        tmp_path, monkeypatch, caplog):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        backend = FakeBackend()
        capture = RecordingCapture(request(tmp_path), backend=backend,
                                   writer_factory=ControlledWriter())
        real_append = capture._waveforms.append
        calls = 0

        def fail_second_append(block):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("capture-preview-probe")
            return real_append(block)

        capture._waveforms.append = fail_second_append
        capture.start()
        try:
            assert capture.started.wait(3)
            data = known_audio()
            for block in (data[:2], data[2:5], data[5:]):
                backend.stream.feed(block)
            outcome = capture.wait(3)
            assert isinstance(outcome, RecordingResult)
            assert outcome.handles_released and backend.stream.closed
            assert calls == 2
            assert sum("preview disabled" in item for item in outcome.warnings) == 1
            assert capture.snapshot(generation=1, sequence=1) is None
            saved, _ = sf.read(outcome.path, dtype="float32", always_2d=True)
            np.testing.assert_array_equal(saved, data[:9, (0, 2)][2:])
        finally:
            capture.cancel()
            capture.wait(5)
            capture._thread.join(3)
            assert not capture._thread.is_alive()
        logged = read_project_log(state, "recording_capture.py")
        assert b"core ERROR Preview failed for capture-1" in logged
        assert b"RuntimeError: capture-preview-probe" in logged
        assert b"Traceback" in logged
        records = [record for record in caplog.records
                   if record.name == "core" and record.msg == "Preview failed for %s"]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info[0] is RuntimeError


def test_service_started_callback_failure_reaches_project_file(tmp_path, monkeypatch, caplog):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = RecordingService()
        calls = []

        def fail_started(session):
            calls.append(session)
            raise RuntimeError("service-started-probe")

        session = RecordingSession(service, request(tmp_path),
                                   RecordingCallbacks(started=fail_started))
        try:
            service._notify(session, "started")
            assert calls == [session]
            assert session.failure is None and session.state == "starting"
            assert service._supervisor.is_alive()
            assert service.diagnostics == [
                "capture-1: started callback failed: service-started-probe"]
            logged = read_project_log(state, "recording_service.py")
            assert b"core ERROR Recording started callback failed" in logged
            assert b"RuntimeError: service-started-probe" in logged
            assert b"capture-1: started callback failed: service-started-probe" in logged
            assert b"Traceback" in logged
            records = [record for record in caplog.records
                       if record.name == "core" and record.msg == "Recording %s callback failed"]
            assert len(records) == 1
            assert records[0].levelno == logging.ERROR
            assert records[0].exc_info[0] is RuntimeError
        finally:
            stop_service(service)


@pytest.mark.parametrize("operation,stage", [("stop_task", "stop_task"), ("close", "close_sdk")])
def test_ve_cleanup_failure_reaches_project_file(tmp_path, monkeypatch, caplog, operation, stage):
    error = RuntimeError("ve-cleanup-probe")

    def fail_cleanup(*args):
        raise error

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        sdk = CaptureSDK(hooks={operation: fail_cleanup})
        stream, _, _, _, failures = make_stream(tmp_path, sdk)
        try:
            assert stream.start()
            assert stream.done.wait(3)
        finally:
            stream.close()
            if stream._owner is not None:
                stream._owner.join(3)
                assert not stream._owner.is_alive()
        assert sdk.operations()[-3:] == ("stop_task", "clear_task", "close")
        assert failures == [(stage, "ve-cleanup-probe")]
        assert not stream.handles_released
        logged = read_project_log(state, "ve3668n_capture.py")
        assert b"core ERROR [VE cleanup]" in logged
        assert f"event=failed operation={stage}".encode("ascii") in logged
        assert b"RuntimeError: ve-cleanup-probe" in logged and b"Traceback" in logged
        records = [record for record in caplog.records
                   if record.name == "core" and "event=failed" in record.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info == (RuntimeError, error, error.__traceback__)


def test_sender_fault_reaches_project_file(tmp_path, monkeypatch, caplog):
    from base.recording_worker import _send_loop

    class BrokenConnection:
        def send(self, value):
            raise ValueError("sender-serialization-probe")

    outgoing = queue.Queue()
    outgoing.put(object())
    broken = threading.Event()
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        _send_loop(BrokenConnection(), outgoing, broken)
        assert broken.is_set()
        assert outgoing.unfinished_tasks == 0
        data = read_project_log(state, "recording_worker.py")
        assert b"core ERROR Recording sender failed" in data
        assert b"ValueError: sender-serialization-probe" in data
        assert b"Traceback" in data
        record, = [r for r in caplog.records if r.name == "core"]
        assert record.msg == "Recording sender failed"
        assert record.levelno == logging.ERROR and record.exc_info[0] is ValueError


def test_sender_fault_in_real_spawn_reaches_child_project_file(tmp_path):
    from unit_test.logging_test_support import spawn_sender_fault

    child = multiprocessing.get_context("spawn").Process(
        target=spawn_sender_fault, args=(str(tmp_path),))
    try:
        child.start()
        child.join(10)
        assert not child.is_alive(), "sender probe exceeded its spawn deadline"
        assert child.exitcode == 0
        path = tmp_path / "logs" / "main.log"
        assert path.exists(), "child business diagnostic did not reach its project file"
        data = path.read_bytes()
        assert b"core ERROR Recording sender failed" in data
        assert b"ValueError: sender-serialization-probe" in data
        assert b"Traceback" in data and b"recording_worker.py:" in data
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        assert not child.is_alive()
        child.close()


def test_worker_logger_setup_failure_closes_endpoints_before_threads(monkeypatch):
    from base import log_manager, recording_worker as worker_module

    control, preview = Mock(), Mock()
    error = PermissionError("logger-setup-probe")
    monkeypatch.setattr(log_manager.LogManager, "set_log_handler", Mock(side_effect=error))
    start = Mock(side_effect=AssertionError("thread started before logger setup completed"))
    monkeypatch.setattr(worker_module.threading.Thread, "start", start)
    with pytest.raises(PermissionError) as raised:
        worker_module.recording_worker(control, preview, 1, None, {})
    assert raised.value is error
    control.close.assert_called_once_with()
    preview.close.assert_called_once_with()
    start.assert_not_called()


@pytest.mark.parametrize("setup_failure", [False, True])
def test_discovery_pipe_failure_preserves_cleanup(tmp_path, monkeypatch, caplog, setup_failure):
    from base import log_manager, ve3668n_discovery as discovery
    from unit_test.base import ve3668n_fakes

    sdk = ve3668n_fakes.DiscoverySDK()
    monkeypatch.setattr(ve3668n_fakes, "DiscoverySDK", lambda: sdk)
    connection = Mock()
    connection.send_bytes.side_effect = OSError("discovery-pipe-probe")
    original_thread = threading.Thread
    watchers = []

    def track_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        watchers.append(thread)
        return thread

    monkeypatch.setattr(discovery.threading, "Thread", track_thread)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        if setup_failure:
            error = PermissionError("discovery-logger-setup-probe")
            monkeypatch.setattr(log_manager.LogManager, "set_log_handler", Mock(side_effect=error))
            with pytest.raises(PermissionError) as raised:
                discovery._discovery_worker(
                    connection, "unit_test.base.ve3668n_fakes.DiscoverySDK", "{}", 65536)
            assert raised.value is error
        else:
            discovery._discovery_worker(
                connection, "unit_test.base.ve3668n_fakes.DiscoverySDK", "{}", 65536)
        assert sdk.closed and sdk.trace[0] == ("devices",)
        connection.close.assert_called_once_with()
        connection.send_bytes.assert_called_once()
        assert len(watchers) == 1 and not watchers[0].is_alive()
        if not setup_failure:
            data = read_project_log(state, "ve3668n_discovery.py")
            assert b"core WARNING Discovery result pipe closed" in data
            assert b"OSError: discovery-pipe-probe" in data and b"Traceback" in data
            record, = [r for r in caplog.records if r.name == "core"]
            assert record.levelno == logging.WARNING and record.exc_info[0] is OSError


def test_discovery_logger_exists_before_supervisor(tmp_path, monkeypatch):
    from base.ve3668n_discovery import DiscoveryService

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        original_start = threading.Thread.start
        starts = []

        def start(thread):
            service = thread._target.__self__
            assert service._logger is state.logger
            assert len(rotating_handlers(state.logger)) == 1
            starts.append(thread)
            return original_start(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        service = DiscoveryService()
        try:
            assert starts == [service._supervisor]
        finally:
            service.close()
            assert service.wait_closed(3)
            service._supervisor.join(3)
            assert not service._supervisor.is_alive()


def test_discovery_callback_failure_reaches_project_file(tmp_path, monkeypatch, caplog):
    from base.ve3668n_discovery import DiscoveryService

    called = threading.Event()
    received = []

    def fail_callback(event):
        received.append(event)
        called.set()
        raise RuntimeError("discovery-callback-probe")

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = DiscoveryService(
            sdk_factory="unit_test.base.ve3668n_fakes.discovery_factory",
            sdk_options={"trace_path": str(tmp_path / "discovery-trace.jsonl")},
            on_result=fail_callback)
        try:
            generation = service.start()
            assert called.wait(5)
            assert service.wait_idle(3)
            event, = received
            assert event.generation == generation and event.status == "completed"
            assert event.handles_released and service.poll_result() == event
        finally:
            service.close()
            assert service.wait_closed(3)
            service._supervisor.join(3)
            assert not service._supervisor.is_alive()
        data = read_project_log(state, "ve3668n_discovery.py")
        assert b"core ERROR Discovery callback failed" in data
        assert b"RuntimeError: discovery-callback-probe" in data and b"Traceback" in data
        record, = [r for r in caplog.records if r.name == "core"]
        assert record.msg == "Discovery callback failed"
        assert record.levelno == logging.ERROR and record.exc_info[0] is RuntimeError


@pytest.mark.parametrize("blocked_preview", [False, True])
def test_worker_fatal_and_sender_exit_cleanup_reach_project_file(
        tmp_path, monkeypatch, caplog, blocked_preview):
    from base import recording_worker as worker_module

    done = threading.Event()
    done.set()
    capture = SimpleNamespace(done=done, cancel=Mock(), join=lambda timeout=0: True)
    pipeline = SimpleNamespace(shutdown_snapshot=lambda: [SimpleNamespace(capture=capture)])
    monkeypatch.setattr(worker_module, "WorkerCapturePipeline", lambda: pipeline)
    controller = SimpleNamespace(close=Mock(return_value=SimpleNamespace(success=True)))
    monkeypatch.setattr(worker_module, "VeResourceController", lambda **kwargs: controller)
    control, preview = Mock(), Mock()
    control.poll.side_effect = ValueError("worker-fatal-probe")
    threads = []

    class ControlledThread:
        def __init__(self, *, target, name, daemon, args=()):
            self.target, self.name, self.args = target, name, args
            threads.append(self)

        def start(self):
            if blocked_preview and self.name == "recording-preview-sender":
                self.args[1].put_nowait(object())

        def join(self, timeout):
            if self.name == "recording-control-sender":
                self.target(*self.args)
            elif not blocked_preview and self.name == "recording-preview-sender":
                self.target(*self.args)

    monkeypatch.setattr(worker_module.threading, "Thread", ControlledThread)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        worker_module.recording_worker(control, preview, 7, None, {})
        control.close.assert_called_once_with()
        preview.close.assert_called_once_with()
        capture.cancel.assert_called_once_with()
        controller.close.assert_called_once()
        assert all(thread.args[2].is_set() for thread in threads if thread.args)
        events = [call.args[0] for call in control.send.call_args_list]
        assert [event.kind for event in events] == ["worker_fatal", "ready"]
        assert events[0].generation == 7
        assert events[0].payload.stage == "worker"
        assert events[0].payload.message == "worker-fatal-probe"
        data = read_project_log(state, "recording_worker.py")
        assert b"core ERROR Recording worker failed" in data
        assert b"ValueError: worker-fatal-probe" in data and b"Traceback" in data
        records = [r for r in caplog.records if r.name == "core"]
        fatal, = [r for r in records if r.msg == "Recording worker failed"]
        assert fatal.levelno == logging.ERROR and fatal.exc_info[0] is ValueError
        warnings = [r for r in records if r.msg == "Discarding blocked recording sender at exit"]
        assert len(warnings) == int(blocked_preview)
        if blocked_preview:
            assert warnings[0].levelno == logging.WARNING and warnings[0].exc_info is None
            assert b"core WARNING Discarding blocked recording sender at exit" in data


def test_discovery_retirement_diagnostics_reach_project_file(tmp_path, monkeypatch, caplog):
    from base import ve3668n_discovery as discovery

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = discovery.DiscoveryService()
        service.close()
        assert service.wait_closed(3)
        service._supervisor.join(3)
        launcher = Mock(ident=1)
        launcher.is_alive.return_value = False
        child = Mock(pid=123, exitcode=None)
        child.is_alive.return_value = True
        child.terminate.side_effect = OSError("retire-terminate-probe")
        child.kill.side_effect = OSError("retire-kill-probe")
        resources = discovery._ProbeResources(
            launcher=launcher, child=child, receive=Mock(), send=Mock(),
            launch_failure=discovery.DiscoveryResult(diagnostics=("launch-probe",)))
        receive, send = resources.receive, resources.send
        service._unreaped = resources
        service._idle.clear()
        service._reap_unconfirmed()
        assert service._unreaped is resources and not service.wait_idle(0)
        assert resources.launcher is resources.receive is resources.send is None
        receive.close.assert_called_once_with()
        send.close.assert_called_once_with()
        assert child.kill.call_count == 2
        assert child.join.call_count == 4
        child.close.assert_not_called()
        child.is_alive.return_value = False
        service._reap_unconfirmed()
        assert resources.released and service._unreaped is None and service.wait_idle(0)
        child.close.assert_called_once_with()
        data = read_project_log(state, "ve3668n_discovery.py")
        assert b"core WARNING Late launch-probe" in data
        assert b"core WARNING Late discovery retirement: helper terminate: OSError: retire-terminate-probe; helper kill: OSError: retire-kill-probe; helper death unconfirmed after kill; service quarantined until reaped" in data
        assert b"core ERROR Discovery helper kill retry failed" in data
        assert b"OSError: retire-kill-probe" in data and b"Traceback" in data
        records = [r for r in caplog.records if r.name == "core"]
        assert [(r.msg, r.levelno) for r in records] == [
            ("Late %s", logging.WARNING),
            ("Late discovery retirement: %s", logging.WARNING),
            ("Discovery helper kill retry failed", logging.ERROR),
        ]
        assert records[0].args == ("launch-probe",) and records[0].exc_info is None
        assert records[1].args == (
            "helper terminate: OSError: retire-terminate-probe; helper kill: OSError: retire-kill-probe; helper death unconfirmed after kill; service quarantined until reaped",)
        assert records[1].exc_info is None
        assert records[2].exc_info[0] is OSError
