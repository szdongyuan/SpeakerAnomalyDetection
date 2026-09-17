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
