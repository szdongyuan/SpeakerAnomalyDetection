import multiprocessing
import os
import time

import pytest

from base.video.service import VideoService
from base.video.models import CommandKind, Event, EventKind
from base.video.worker import SimulationOptions, simulated_video_worker


def wait_until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("video condition did not become true before timeout")


@pytest.fixture
def service_factory():
    services = []

    def create(options=None, **kwargs):
        service = VideoService(
            worker_target=simulated_video_worker,
            worker_options=options or SimulationOptions(), **kwargs,
        )
        services.append(service)
        assert service.start()
        wait_until(lambda: service.status.connection == "ready")
        return service

    yield create
    for service in services:
        service.shutdown()
        assert service.wait_closed()
        assert not any(child.pid == service.process_id for child in multiprocessing.active_children())


def test_separate_process_manual_sessions_and_preview_continues(service_factory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = service_factory()
    assert service.process_id != os.getpid()
    wait_until(lambda: service.latest_preview() is not None)
    for _ in range(3):
        assert service.start_recording()
        assert not service.start_recording()
        wait_until(lambda: service.status.recording == "recording")
        assert service.stop_recording()
        assert not service.stop_recording()
        wait_until(lambda: service.status.recording == "completed")
    frame = service.latest_preview()
    wait_until(lambda: service.latest_preview(frame.sequence) is not None)
    assert "未生成视频文件" in service.status.detail
    assert list(tmp_path.iterdir()) == []


def test_stop_while_starting_is_not_late_restarted(service_factory):
    service = service_factory(SimulationOptions(start_delay=0.5))
    service.start_recording()
    service.stop_recording()
    wait_until(lambda: service.status.recording == "completed")
    time.sleep(0.6)
    assert service.status.recording == "completed"
    assert service.status.started_at is None


def test_reconnect_resumes_same_session(service_factory):
    service = service_factory(SimulationOptions(disconnect_after=0.1, reconnect_after=0.25))
    service.start_recording()
    wait_until(lambda: service.status.recording == "recovering")
    session = service.status.session_id
    started = service.status.started_at
    assert service.latest_preview() is None
    wait_until(lambda: service.status.recording == "recording")
    assert service.status.session_id == session
    assert service.status.started_at == started


def test_stop_while_recovering_does_not_resume_but_preview_recovers(service_factory):
    service = service_factory(SimulationOptions(disconnect_after=0.1, reconnect_after=0.5))
    service.start_recording()
    wait_until(lambda: service.status.recording == "recovering")
    assert service.stop_recording()
    assert not service.status.record_intent
    wait_until(lambda: service.status.recording == "interrupted")
    wait_until(lambda: service.status.connection == "ready")
    assert service.status.recording == "interrupted"
    wait_until(lambda: service.latest_preview() is not None)


def test_start_failure_does_not_report_recording(service_factory):
    service = service_factory(SimulationOptions(fail_start=True))
    service.start_recording()
    wait_until(lambda: service.status.recording == "failed")
    assert service.status.started_at is None
    assert not service.status.record_intent


@pytest.mark.parametrize("options", [SimulationOptions(crash_after=0.1), SimulationOptions(stall_after=0.1)])
def test_child_crash_and_stall_are_bounded_failures(service_factory, options):
    service = service_factory(options, heartbeat_timeout=1)
    service.start_recording()
    assert service.wait_closed(timeout=5)
    assert service.status.recording == "failed"
    assert service.status.connection == "unavailable"
    assert not service.start_recording()
    if options.stall_after is not None:
        assert service.forced_termination


def test_shutdown_finishes_active_simulation_asynchronously(service_factory):
    service = service_factory()
    service.start_recording()
    wait_until(lambda: service.status.recording == "recording")
    started = time.monotonic()
    service.shutdown()
    assert time.monotonic() - started < 0.2
    assert service.wait_closed()
    assert service.status.connection == "closed"
    assert service.status.recording == "completed"
    assert not service.forced_termination


def test_shutdown_without_start_is_idempotent():
    service = VideoService(worker_target=simulated_video_worker, worker_options=SimulationOptions())
    service.shutdown()
    service.shutdown()
    assert service.is_closed
    assert service.status.connection == "closed"
    assert not service.start()


def close_without_exiting_worker(channel, mailbox, generation, options):
    channel.send(Event(EventKind.READY, generation, 1, time.monotonic()))
    channel.recv()
    channel.send(Event(EventKind.CLOSED, generation, 2, time.monotonic()))
    time.sleep(20)


def test_closed_message_is_not_proof_of_process_exit():
    service = VideoService(
        worker_target=close_without_exiting_worker, worker_options=None, shutdown_timeout=0.15,
    )
    service.start()
    try:
        wait_until(lambda: service.status.connection == "ready")
        service.shutdown()
        assert service.wait_closed(4)
        assert service.forced_termination
        assert service.status.connection == "unavailable"
        assert "未退出" in service.status.detail
    finally:
        service.shutdown()
        assert service.wait_closed()


def test_stop_timeout_is_failed_not_completed(service_factory):
    service = service_factory(SimulationOptions(stop_delay=2), command_timeout=0.25)
    service.start_recording()
    wait_until(lambda: service.status.recording == "recording")
    service.stop_recording()
    assert service.wait_closed(4)
    assert service.status.recording == "failed"
    assert "启停响应超时" in service.status.detail


def progressing_drain_worker(channel, mailbox, generation, options):
    progress_enabled, delay = options
    sequence = progress = 0
    identity = ""
    drain_at = None
    closing = False

    def emit(kind, session_id=""):
        nonlocal sequence
        sequence += 1
        channel.send(Event(kind, generation, sequence, time.monotonic(), session_id,
                           progress=progress))

    emit(EventKind.READY)
    while True:
        if channel.poll(.04):
            command = channel.recv()
            if command.kind == CommandKind.START:
                identity = command.session_id
                emit(EventKind.STARTED, identity)
            else:
                closing |= command.kind == CommandKind.SHUTDOWN
                drain_at = time.monotonic() + delay
        if drain_at is not None:
            if progress_enabled:
                progress += 1
            if time.monotonic() >= drain_at:
                emit(EventKind.COMPLETED, identity)
                identity, drain_at = "", None
                if closing:
                    emit(EventKind.CLOSED)
                    channel.close()
                    return
        emit(EventKind.HEARTBEAT)


@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("mode", ["progress", "heartbeat-only", "total-timeout"])
def test_drain_deadlines_use_media_progress_not_heartbeats(shutdown, mode):
    service = VideoService(
        worker_target=progressing_drain_worker,
        worker_options=(mode != "heartbeat-only", .6 if mode != "total-timeout" else 10),
        heartbeat_timeout=4, command_timeout=.2, drain_timeout=.3,
        shutdown_timeout=.5 if mode == "total-timeout" else 2,
    )
    service.start()
    try:
        wait_until(lambda: service.status.connection == "ready")
        assert service.start_recording()
        wait_until(lambda: service.status.recording == "recording")
        if shutdown:
            service.shutdown()
        else:
            assert service.stop_recording()
        if mode == "progress":
            wait_until(lambda: service.status.recording == "completed")
            assert not service.forced_termination
        else:
            assert service.wait_closed(5)
            assert service.forced_termination
            assert service.status.recording == "failed"
            assert "超时" in service.status.detail or "时限" in service.status.detail
    finally:
        service.shutdown()
        assert service.wait_closed(5)


@pytest.mark.parametrize("status", ["no-runtime", "timeout", "drained-with-errors", "unconfirmed-death"])
def test_supervisor_log_drain_precedes_terminate_and_kill(monkeypatch, caplog, status):
    import threading
    from types import SimpleNamespace
    import base.video.service as module
    from base.log_exit import DrainResult, run_with_log_drain
    entered, release = threading.Event(), threading.Event()
    calls, processes = [], []
    endpoint = object()
    class Drain:
        child_endpoint = endpoint
        reason = None
        def begin(self, reason):
            self.reason = reason
            calls.append("begin")
        def wait(self):
            assert threading.current_thread().name == "VideoSupervisor"
            entered.set()
            assert release.wait(3)
            calls.append("ack")
            return DrainResult("no-runtime" if status == "unconfirmed-death" else status)
        def poll(self, **kwargs):
            return DrainResult("no-runtime" if status == "unconfirmed-death" else status)
        def close(self):
            calls.append("drain-close")
    class Process:
        pid = None
        alive = True
        exitcode = None
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            processes.append(self)
        def start(self):
            self.pid = 123
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append("terminate")
        def kill(self):
            calls.append("kill")
            self.alive = status == "unconfirmed-death"
        def join(self, timeout):
            pass
        def close(self):
            calls.append("process-close")
    channel = SimpleNamespace(close=lambda: None, poll=lambda: False)
    context = SimpleNamespace(Process=Process, Pipe=lambda **kwargs: (channel, channel))
    monkeypatch.setattr(module, "ProcessLogDrain", SimpleNamespace(create=lambda _: Drain()), raising=False)
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(module, "PreviewMailbox", lambda _: object())
    service = VideoService(worker_target=close_without_exiting_worker, worker_options="custom", heartbeat_timeout=-1)
    service.start()
    try:
        assert entered.wait(1), "supervisor terminated without requesting log drain"
        assert calls == ["begin"]
        started = time.monotonic()
        service.shutdown()
        assert time.monotonic() - started < .2
        assert not service.is_closed
    finally:
        release.set()
        assert service.wait_closed(4)
    if status == "unconfirmed-death":
        assert calls == ["begin", "ack", "terminate", "kill"]
        assert service._unreaped[0] is processes[0]
        processes[0].alive = False
        assert service.wait_closed(0)
        assert service._unreaped is None
    assert calls == ["begin", "ack", "terminate", "kill", "process-close", "drain-close"]
    assert processes[0].kwargs["target"] is run_with_log_drain
    target, args, bound_endpoint = processes[0].kwargs["args"]
    assert target is close_without_exiting_worker
    assert args[2:] == (1, "custom") and bound_endpoint is endpoint
    if status in ("timeout", "drained-with-errors"):
        assert status in caplog.text and "pending=unknown" in caplog.text


def logged_blocked_video_worker(channel, mailbox, generation, options):
    import threading
    from pathlib import Path
    from unit_test.base.log_manager_process_fakes import configure
    from base import log_manager
    directory, blocked_sink = options
    manager = configure(directory)
    entered = threading.Event()
    if blocked_sink:
        def blocked_write(sink, message):
            assert sink.is_locked
            entered.set()
            threading.Event().wait()
        log_manager._BatchFileHandler.do_write = blocked_write
    logger = manager.set_log_handler("core")
    if blocked_sink:
        logger.error("parent-blocked-sink")
        assert entered.wait(3)
    else:
        for index in range(3):
            logger.info("parent-video-tail=%d", index)
    channel.send(Event(EventKind.READY, generation, 1, time.monotonic()))
    threading.Event().wait()
    Path(directory, "returned").touch()


@pytest.mark.parametrize("blocked_sink", [False, True])
def test_real_video_parent_log_drain_flushes_or_times_out_off_gui(tmp_path, monkeypatch, caplog, blocked_sink):
    from types import SimpleNamespace
    from base.log_exit import ProcessLogDrain
    import base.video.service as module
    drains = []
    def create(context):
        drain = ProcessLogDrain.create(context)
        drains.append(drain)
        return drain
    monkeypatch.setattr(module, "ProcessLogDrain", SimpleNamespace(create=create))
    service = VideoService(worker_target=logged_blocked_video_worker,
                           worker_options=(str(tmp_path), blocked_sink), shutdown_timeout=.1)
    try:
        service.start()
        wait_until(lambda: service.status.connection == "ready")
        started = time.monotonic()
        service.shutdown()
        assert time.monotonic() - started < .2
        assert service.wait_closed(4)
        assert time.monotonic() - started < 3.5
        assert service.forced_termination
        result = drains[0].poll()
        assert result.status == ("timeout" if blocked_sink else "drained")
        assert drains[0].child_endpoint is None
        assert not (tmp_path / "returned").exists()
        assert not any(child.pid == service.process_id for child in multiprocessing.active_children())
        if blocked_sink:
            assert "status=timeout" in caplog.text
        else:
            assert result.stats["pending"] == 0
            text = (tmp_path / "main.log").read_text(encoding="utf-8")
            assert all(f"parent-video-tail={index}" in text for index in range(3))
    finally:
        service.shutdown()
        assert service.wait_closed(6)


@pytest.mark.parametrize("start_failure", [False, True])
def test_video_log_drain_cleans_start_failure_and_reports_self_exit(monkeypatch, caplog, start_failure):
    from types import SimpleNamespace
    from base.log_exit import ProcessLogDrain, DrainResult
    import base.video.service as module
    context = multiprocessing.get_context("spawn")
    drain = ProcessLogDrain.create(context)
    drain._result = DrainResult("timeout", detail="self-exit test")
    closed = []
    class Process:
        pid = None
        exitcode = 24
        def __init__(self, **kwargs):
            pass
        def start(self):
            if start_failure:
                raise OSError("video start denied")
            self.pid = 123
        def is_alive(self):
            return False
        def join(self, timeout):
            pass
        def close(self):
            closed.append(True)
    channel = SimpleNamespace(close=lambda: None, poll=lambda: False)
    context = SimpleNamespace(Process=Process, Pipe=lambda **kwargs: (channel, channel))
    monkeypatch.setattr(module, "ProcessLogDrain", SimpleNamespace(create=lambda _: drain))
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(module, "PreviewMailbox", lambda _: object())
    service = VideoService(worker_target=close_without_exiting_worker, worker_options=None)
    service.start()
    assert service.wait_closed(3)
    assert drain.child_endpoint is None
    assert service.status.connection == "unavailable"
    if start_failure:
        assert "video start denied" in service.status.detail and not closed
    else:
        assert closed
        assert "reason=self-exit" in caplog.text and "status=timeout" in caplog.text
        assert "pending=unknown" in caplog.text
