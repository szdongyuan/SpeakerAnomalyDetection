"""Soundcard reader-start failure preserves baseline ownership and future admission."""
import pytest
from pathlib import Path
from base.recording_result_reader import ResultReader
from unit_test.base.test_recording_service import services, Events, request, eventually


@pytest.mark.parametrize("failure_at", ["construct", "start"])
def test_reader_prestart_failure_releases_lease_and_allows_next_capture(tmp_path, services, failure_at):
    calls = 0
    def reader_factory(descriptor, completed):
        nonlocal calls
        calls += 1
        if calls == 1 and failure_at == "construct":
            raise RuntimeError("injected reader construction failure")
        reader = ResultReader(descriptor, completed)
        if calls == 1:
            def fail_start():
                raise RuntimeError("injected reader start failure")
            reader.start = fail_start
        return reader
    service = services(reader_factory=reader_factory, cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(0,)), events.callbacks)
    assert events.failed.get(timeout=10).stage == "service"
    assert session.released.wait(3), "a reader that never started cannot retain the result lease"
    eventually(lambda: service.worker_pid is None, timeout=3)
    assert not service.is_path_leased(session.request.path)
    assert not Path(session.request.path).parent.exists()
    assert service.can_start_recording
    next_events = Events()
    next_session = service.start(request(tmp_path, request_id="next", path=str(tmp_path / "next.wav")), next_events.callbacks)
    next_events.results.get(timeout=10)
    next_session.accept_result()
    assert next_session.released.wait(5)
    assert next_events.failed.empty()
