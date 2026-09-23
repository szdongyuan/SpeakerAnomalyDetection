"""Production startup observations, independent of benchmark instrumentation."""
from dataclasses import FrozenInstanceError, replace
import os
import time

import pytest

from base.recording_process_protocol import RecordingEvent
from base.recording_service import RecordingCallbacks, RecordingSession
from unit_test.base.test_recording_service import services, request, eventually
from unit_test.ui.conftest import ui_qapp
from ui.recording_service_bridge import RecordingServiceBridge


def test_real_start_snapshot_separates_delayed_qt_delivery(tmp_path, services, ui_qapp):
    service = services({'manual': True})
    bridge = RecordingServiceBridge(service)
    delivered = []
    before = time.perf_counter()
    session = bridge.start(request(tmp_path), RecordingCallbacks(started=lambda s: delivered.append(s.startup_timing)))
    after = time.perf_counter()
    try:
        eventually(lambda: session.state == 'recording')
        capture = session.startup_timing
        assert before <= capture.accepted_seconds <= after
        assert capture.accepted_seconds <= capture.capture_observed_seconds
        assert capture.qt_started_seconds is None
        assert capture.request_to_qt_seconds is None
        assert capture.worker_pid == session.worker_pid != os.getpid()
        assert capture.generation == session.generation
        assert capture.request_id == session.request.request_id
        with pytest.raises(FrozenInstanceError):
            capture.accepted_seconds = 0
        time.sleep(.06)  # No Qt dispatch: supervisor observation must stay fixed.
        ui_qapp.processEvents()
        assert len(delivered) == 1
        timing = delivered[0]
        assert timing.capture_observed_seconds == capture.capture_observed_seconds
        assert timing.request_to_capture_seconds == capture.request_to_capture_seconds
        assert timing.qt_delivery_seconds >= .06
        assert timing.request_to_qt_seconds >= timing.request_to_capture_seconds
        bridge._deliver(('started', session, None))
        assert session.startup_timing == timing
        assert len(delivered) == 1
    finally:
        session.cancel()
        assert session.released.wait(10)
        ui_qapp.processEvents()


def test_only_valid_first_worker_started_observation_counts(tmp_path, services, monkeypatch):
    service = services({'manual': True})
    original = service._event
    held, processed = [], []
    def defer_first_started(worker, event):
        if isinstance(event, RecordingEvent) and event.kind == 'started' and not held:
            held.append((worker, event))
            return
        original(worker, event)
        processed.append(event)
    monkeypatch.setattr(service, '_event', defer_first_started)
    session = service.start(request(tmp_path))
    try:
        eventually(lambda: bool(held))
        assert session.startup_timing.capture_observed_seconds is None
        worker, event = held[0]
        stale = replace(event, generation=event.generation + 1)
        other = replace(event, request_id='other-recording')
        service._inbox.put(('event', worker, stale))
        service._inbox.put(('event', worker, other))
        service._inbox.put(('event', object(), event))
        eventually(lambda: stale in processed and other in processed and event in processed)
        assert session.startup_timing.capture_observed_seconds is None
        service._inbox.put(('event', worker, event))
        eventually(lambda: session.state == 'recording')
        first = session.startup_timing
        service._inbox.put(('event', worker, event))
        eventually(lambda: processed.count(event) == 3)
        assert session.startup_timing == first
    finally:
        session.cancel()
        assert session.released.wait(10)


def test_unaccepted_or_missing_started_never_reports_successful_start(tmp_path, services):
    service = services({'manual': True})
    unaccepted = RecordingSession(service, request(tmp_path), RecordingCallbacks())
    assert unaccepted.startup_timing.accepted_seconds is None
    assert unaccepted.startup_timing.request_to_capture_seconds is None
    with pytest.raises(TypeError):
        service.start(None)
    assert not service._sessions
    session = service.start(request(tmp_path))
    try:
        original = session.startup_timing.accepted_seconds
        with pytest.raises(RuntimeError, match='busy'):
            service.start(request(tmp_path, request_id='rejected', path=str(tmp_path / 'other.wav')))
        assert set(service._sessions) == {'one'}
        assert session.startup_timing.accepted_seconds == original
    finally:
        session.cancel()
        assert session.released.wait(10)
    # Directly constructed/rejected sessions cannot invent a Qt capture point.
    unaccepted._observe_qt_started()
    assert unaccepted.startup_timing.qt_started_seconds is None
