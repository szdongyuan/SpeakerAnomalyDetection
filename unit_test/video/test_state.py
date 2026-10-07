import pytest

from base.video.models import Event, EventKind, VideoState


def ready_state():
    state = VideoState(1)
    state.apply(Event(EventKind.READY, 1, 1, 1))
    return state


def test_camera_error_and_recovery_do_not_replace_first_recording_failure():
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.RECORDING_FAILED, 1, 2, 2, "s1", detail="MP4写入拒绝访问"))
    state.apply(Event(EventKind.OFFLINE, 1, 3, 3, detail="MJPEG解码失败"))
    assert state.status.recording_detail == "MP4写入拒绝访问"
    assert state.status.connection_detail == "MJPEG解码失败"
    assert not state.apply(Event(EventKind.RECORDING_FAILED, 1, 4, 4, "s1", detail="后续收尾失败"))
    state.apply(Event(EventKind.READY, 1, 5, 5))
    assert state.status.detail == "MP4写入拒绝访问"
    assert state.status.connection_detail == ""
    assert state.status.ended_at == 2
    state.fail("视频进程退出", 6)
    assert state.status.recording_detail == "MP4写入拒绝访问"
    assert state.status.ended_at == 2
    state.apply(Event(EventKind.READY, 1, 6, 7))
    assert state.request_start("s2")
    assert state.status.recording_detail == state.status.connection_detail == ""
    assert not state.apply(Event(EventKind.RECORDING_FAILED, 1, 7, 8, "s1", detail="旧错误"))
    assert state.status.recording == "starting"


def test_start_requires_ready_and_actual_started_ack():
    state = VideoState(1)
    assert not state.request_start("s1")
    state = ready_state()
    assert state.request_start("s1")
    assert not state.request_start("s2")
    state.apply(Event(EventKind.ACCEPTED, 1, 2, 2, command_id="c1"))
    assert state.status.recording == "starting"
    assert state.status.elapsed(20) == 0
    state.apply(Event(EventKind.STARTED, 1, 3, 3, "s1"))
    assert state.status.recording == "recording"
    assert state.status.elapsed(13) == 10


def test_stop_during_start_ignores_late_started():
    state = ready_state()
    state.request_start("s1")
    assert state.request_stop()
    assert not state.request_stop()
    assert not state.apply(Event(EventKind.STARTED, 1, 2, 2, "s1"))
    assert state.status.recording == "stopping"
    state.apply(Event(EventKind.COMPLETED, 1, 3, 3, "s1"))
    assert state.status.recording == "completed"
    assert not state.status.record_intent


def test_generation_sequence_and_session_fences():
    state = ready_state()
    state.request_start("s1")
    assert not state.apply(Event(EventKind.STARTED, 2, 100, 2, "s1"))
    assert not state.apply(Event(EventKind.STARTED, 1, 1, 2, "s1"))
    assert not state.apply(Event(EventKind.STARTED, 1, 2, 2, "old-session"))
    assert state.apply(Event(EventKind.STARTED, 1, 3, 3, "s1"))
    assert not state.apply(Event(EventKind.COMPLETED, 1, 2, 4, "s1"))
    assert state.status.recording == "recording"


def test_disconnect_stops_session_and_rejects_late_resume():
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.STARTED, 1, 2, 10, "s1"))
    state.apply(Event(EventKind.RECOVERING, 1, 3, 20, "s1"))
    assert state.status.recording == "stopping"
    assert not state.status.record_intent
    assert state.status.elapsed(29) == 10
    assert not state.request_stop()
    state.apply(Event(EventKind.READY, 1, 4, 30))
    assert not state.request_start("s2")  # Wait for file finalization.
    assert not state.apply(Event(EventKind.RECOVERING, 1, 5, 40, "s1"))
    assert not state.apply(Event(EventKind.STARTED, 1, 6, 41, "s1"))
    state.apply(Event(EventKind.COMPLETED, 1, 7, 42, "s1"))
    assert state.status.recording == "interrupted"
    assert state.status.had_gap
    assert state.status.elapsed(10000) == 10
    assert not state.apply(Event(EventKind.STARTED, 1, 8, 43, "s1"))


def test_manual_restart_after_disconnect_creates_fresh_timer():
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.STARTED, 1, 2, 10.25, "s1"))
    state.apply(Event(EventKind.RECOVERING, 1, 3, 20.75, "s1"))
    state.apply(Event(EventKind.COMPLETED, 1, 4, 22, "s1"))
    assert not state.request_start("s2")  # Camera is still offline.
    state.apply(Event(EventKind.READY, 1, 5, 25))
    assert state.status.elapsed(1000) == 10
    assert state.status.recording == "interrupted"
    assert state.request_start("s2")
    state.apply(Event(EventKind.STARTED, 1, 6, 100, "s2"))
    assert state.status.elapsed(110) == 10


@pytest.mark.parametrize("kind", [EventKind.FAILED, EventKind.RECORDING_FAILED, EventKind.CLOSED])
def test_failure_or_shutdown_during_gap_keeps_timer_frozen(kind):
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.STARTED, 1, 2, 10, "s1"))
    state.apply(Event(EventKind.RECOVERING, 1, 3, 20, "s1"))
    state.apply(Event(kind, 1, 4, 100, "s1", detail="test failure"))
    assert state.status.elapsed(1000) == 10
    assert not state.status.record_intent


def test_disconnect_before_first_written_frame_cancels_start():
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.RECOVERING, 1, 2, 10, "s1"))
    assert state.status.elapsed(100) == 0
    assert not state.status.record_intent
    assert not state.apply(Event(EventKind.STARTED, 1, 3, 100, "s1"))
    state.apply(Event(EventKind.COMPLETED, 1, 4, 101, "s1"))
    assert state.status.recording == "interrupted"
    assert state.status.elapsed(110) == 0


def test_failed_or_closed_active_session_is_not_success():
    for kind, expected in [(EventKind.FAILED, "failed"), (EventKind.CLOSED, "interrupted")]:
        state = ready_state()
        state.request_start("s1")
        state.apply(Event(EventKind.STARTED, 1, 2, 10, "s1"))
        state.apply(Event(kind, 1, 3, 20, detail="test failure"))
        assert state.status.recording == expected
        assert not state.status.record_intent
        assert state.status.elapsed(50) == 10


def test_new_manual_session_resets_timer_and_accepts_no_old_completion():
    state = ready_state()
    state.request_start("s1")
    state.apply(Event(EventKind.STARTED, 1, 2, 10, "s1"))
    state.request_stop()
    state.apply(Event(EventKind.COMPLETED, 1, 3, 20, "s1"))
    assert state.request_start("s2")
    assert state.status.elapsed(100) == 0
    assert not state.apply(Event(EventKind.COMPLETED, 1, 4, 21, "s1"))
    assert state.status.recording == "starting"
