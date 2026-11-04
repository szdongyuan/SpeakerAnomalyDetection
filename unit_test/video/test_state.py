from base.video.models import Event, EventKind, VideoState


def test_state_ignores_stale_events_and_keeps_camera_failure_visible_after_close():
    state = VideoState(2)
    assert not state.apply(Event(EventKind.READY, 1, 1, 0))
    assert state.apply(Event(EventKind.OFFLINE, 2, 1, 0, detail="disconnected"))
    assert state.status.connection == "reconnecting"
    assert state.apply(Event(EventKind.READY, 2, 2, 0))
    assert not state.status.connection_detail
    assert not state.apply(Event(EventKind.OFFLINE, 2, 1, 0))
    state.apply(Event(EventKind.FAILED, 2, 3, 0, detail="camera error"))
    state.apply(Event(EventKind.CLOSED, 2, 4, 0))
    assert state.status.connection == "unavailable"
    assert state.status.connection_detail == "camera error"
