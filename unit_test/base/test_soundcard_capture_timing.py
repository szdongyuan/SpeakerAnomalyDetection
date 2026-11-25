"""Software sample completion and soundcard capture/finalization overlap."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingCaptureTiming, RecordingEvent, RecordingResult
from base.recording_service import RecordingCallbacks, RecordingService
from unit_test.base.recording_process_fakes import ControlledMetadataAppender, FakeBackend, FakeStream
from unit_test.base.test_recording_capture import request, captures


@pytest.mark.parametrize("time_info", [None, SimpleNamespace(currentTime=0, inputBufferAdcTime=0)])
def test_full_cycle_uses_software_sample_completion(tmp_path, monkeypatch, time_info):
    capture = RecordingCapture(request(tmp_path, report_capture_timing=True), backend=FakeBackend())
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("base.recording_capture.time.monotonic", lambda: clock.now)
    capture._input_callback(np.zeros((5, 3)), 5, time_info, None)
    assert capture.sample_timing is None
    clock.now += .05
    capture._input_callback(np.zeros((5, 3)), 5, time_info, None)
    assert capture.raw_frames == 9
    assert capture.sample_timing == (100, 100.05)


def test_soundcard_releases_capture_before_metadata(captures, tmp_path):
    finalizer = ControlledMetadataAppender()
    capture, backend = captures(request(tmp_path, report_capture_timing=True), metadata_appender=finalizer)
    callback = backend.stream.config["callback"]
    callback(np.zeros((9, 3)), 9,
             SimpleNamespace(currentTime=20.1, inputBufferAdcTime=20), None)
    try:
        assert finalizer.entered.wait(3)
        assert capture.capture_slot_released.is_set()
        assert capture.capture_slot.adapter_released and capture.capture_slot.writer_released
        assert not capture.done.is_set()
    finally:
        finalizer.release.set()
    assert isinstance(capture.wait(3), RecordingResult)


@pytest.mark.parametrize("changes", [dict(started_at=float("nan")), dict(ended_at=9),
                                   dict(frames=0), dict(generation=0)])
def test_timing_protocol_rejects_invalid_payload(changes):
    fields = dict(request_id="one", generation=1, started_at=10, ended_at=11, frames=100)
    fields.update(changes)
    with pytest.raises(ValueError):
        RecordingCaptureTiming(**fields)


def test_timing_event_rejects_mismatched_identity():
    with pytest.raises(ValueError):
        RecordingEvent(1, "other", "capture_timing", RecordingCaptureTiming("one", 1, 10, 11, 100))


def timed_soundcard_dependencies(**options):
    """Importable backend for a real worker process; no physical device involved."""
    class Stream(FakeStream):
        def start(self):
            super().start()
            callback = self.config["callback"]
            capture = callback.__self__

            def feed():
                time.sleep(.03)
                frames = capture.request.target_samples
                callback(np.zeros((frames, 3), dtype=np.float32), frames,
                         None, None)

            self.thread = threading.Thread(target=feed)
            self.thread.start()

        def stop(self):
            self.thread.join(3)
            super().stop()

    backend = FakeBackend()
    backend.InputStream = lambda **config: Stream(backend, **config)

    def append_metadata(path, metadata, **kwargs):
        if Path(path).name == "first.wav" or "first.stage" in Path(path).parts:
            deadline = time.monotonic() + 10
            while not Path(options["release_file"]).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test metadata gate not released")
                time.sleep(.01)
        return False

    return {"backend": backend, "metadata_appender": append_metadata}


def test_worker_starts_second_capture_while_first_is_finalizing(tmp_path):
    release_file = tmp_path / "release"
    service = RecordingService(
        backend_factory="unit_test.base.test_soundcard_capture_timing:timed_soundcard_dependencies",
        backend_options={"release_file": str(release_file)})
    slot, second_done, first_done = threading.Event(), threading.Event(), threading.Event()
    timings, failures = [], []
    callbacks = dict(
        result_ready=lambda session, audio: session.accept_result(),
        failed=lambda session, failure: failures.append(failure),
        capture_timing=lambda session, timing: timings.append(timing),
        capture_available=lambda session, timing: slot.set())
    try:
        first = service.start(request(tmp_path, path=str(tmp_path / "first.wav"), report_capture_timing=True),
                              RecordingCallbacks(**callbacks, accepted=lambda *args: first_done.set()))
        assert slot.wait(10), failures
        assert service.can_start_recording and not first_done.is_set()
        second = service.start(request(tmp_path, request_id="second", path=str(tmp_path / "second.wav"), report_capture_timing=True),
                               RecordingCallbacks(**callbacks, accepted=lambda *args: second_done.set()))
        assert second_done.wait(5), failures
        assert not first_done.is_set()
        release_file.touch()
        assert first_done.wait(5), failures
        assert first.released.wait(5) and second.released.wait(5)
        assert len(timings) == 2 and not failures
        assert service.can_start_recording
    finally:
        release_file.touch()
        service.shutdown()
        assert service.closed.wait(10)


def test_grouped_wav_publication_preserves_overlapping_condition_ownership(tmp_path):
    from unit_test.base.test_workbench_recording_service import grouped_request

    release_file = tmp_path / "release"
    original = replace(grouped_request(tmp_path), report_capture_timing=True)
    first_request = replace(original, request_id="first", path=str(tmp_path / "first.stage"),
                            station_group=replace(original.station_group, final_path=str(tmp_path / "first")))
    second_request = replace(original, request_id="second", path=str(tmp_path / "second.stage"),
                             station_group=replace(original.station_group, condition_id="next",
                                                   final_path=str(tmp_path / "second")))
    service = RecordingService(
        backend_factory="unit_test.base.test_soundcard_capture_timing:timed_soundcard_dependencies",
        backend_options={"release_file": str(release_file)})
    available, second_done, first_done = threading.Event(), threading.Event(), threading.Event()
    results, failures = [], []

    def completed(session, result):
        results.append(result)
        (first_done if result.request_id == "first" else second_done).set()

    callbacks = RecordingCallbacks(
        capture_available=lambda *args: available.set(), workbench_completed=completed,
        failed=lambda session, failure: failures.append(failure))
    try:
        first = service.start(first_request, callbacks)
        assert available.wait(10), failures
        assert service.can_start_recording and not first_done.is_set()
        second = service.start(second_request, callbacks)
        assert second_done.wait(5), failures
        assert not first_done.is_set()
        release_file.touch()
        assert first_done.wait(5), failures
        assert first.released.wait(5) and second.released.wait(5)
        assert [(r.request_id, r.condition_id) for r in results] == [("second", "next"), ("first", "condition")]
        assert all(Path(station.path).is_file() for r in results for station in r.stations)
        assert not failures
    finally:
        release_file.touch()
        service.shutdown()
        assert service.closed.wait(10)


def test_vking_reports_the_same_capture_and_release_events_for_two_conditions(tmp_path):
    from unit_test.base.test_workbench_recording_service import grouped_request

    service = RecordingService(
        backend_factory="unit_test.base.vkinging_fakes:capture_dependencies",
        backend_options={"trace_path": str(tmp_path / "ve.jsonl")})
    original = grouped_request(tmp_path, "ve")
    available = threading.Event()
    observed, failures = [], []

    def released(session, timing):
        observed.append(("available", timing.request_id))
        available.set()

    callbacks = RecordingCallbacks(
        capture_timing=lambda session, timing: observed.append(("samples", timing.request_id)),
        capture_available=released, failed=lambda session, failure: failures.append(failure))
    try:
        for condition in ("forward", "reverse"):
            available.clear()
            request = replace(original, request_id=condition, report_capture_timing=True,
                              path=str(tmp_path / (condition + ".stage")), station_group=replace(
                                  original.station_group, condition_id=condition, final_path=str(tmp_path / condition)))
            session = service.start(request, callbacks)
            assert available.wait(10), failures
            assert session.released.wait(10), failures
            assert (tmp_path / condition / "A.wav").is_file()
        assert observed == [(kind, c) for c in ("forward", "reverse") for kind in ("samples", "available")]
        assert not failures
    finally:
        service.shutdown()
        assert service.closed.wait(10)
