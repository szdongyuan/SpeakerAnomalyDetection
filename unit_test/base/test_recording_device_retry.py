"""Native pre-start recovery through real worker/capture paths, without hardware."""
import numpy as np
import pytest
import soundfile as sf
import threading

from unit_test.base.test_recording_audio_lifetime import harness
from unit_test.base.test_recording_capture import request
from unit_test.base.recording_process_fakes import known_audio
from base.wav_calibration_metadata import read_wav_calibration_metadata


@pytest.mark.parametrize("failure_at", ["open", "start"])
def test_native_prestart_retry_preserves_request_and_samples(tmp_path, harness, monkeypatch, failure_at):
    worker = harness()
    original_open = worker.backend.InputStream
    attempts = []

    def open_stream(**config):
        attempts.append(config)
        stream = None
        if failure_at == "start":
            stream = original_open(**config)
        if "library_initialized" not in worker.names():
            def fail():
                raise OSError("MME error 2: device ID out of range")
            if stream is None:
                fail()
            stream.start = fail
        return stream or original_open(**config)

    monkeypatch.setattr(worker.backend, "InputStream", open_stream)
    selected = {**worker.backend.device, "hostapi_name": "Selected API"}
    worker.backend.apis = [{"name": "Selected API"}]
    worker.backend.inventory = [worker.backend.device.copy()]
    worker.backend.fresh_inventory = [{**worker.backend.device, "index": 8, "hostapi": 1}]
    worker.backend.fresh_apis = [{"name": "Other API"}, {"name": "Selected API"}]
    metadata = {"recorded_channels": [
        {"wav_channel_index": 0, "physical_input_channel": 0, "calibrated": False},
        {"wav_channel_index": 1, "physical_input_channel": 2, "calibrated": False},
    ]}
    req = request(tmp_path, device=selected, calibration_metadata=metadata)
    worker.start(req)
    capture = worker.pipeline.active.capture
    assert capture.request is req
    assert capture.request.calibration_metadata is req.calibration_metadata
    assert capture.request.preview_time_mode == req.preview_time_mode
    worker.backend.stream.feed(known_audio())
    result = worker.receive("completed").payload
    assert (result.request_id, result.path, result.channels) == (req.request_id, req.path, req.channels)
    assert result.raw_frames == 9 and result.final_frames == 7
    saved, rate = sf.read(req.path, dtype="float32", always_2d=True)
    np.testing.assert_allclose(saved, known_audio()[2:9, (0, 2)], atol=1 / 8388608)
    assert rate == req.sample_rate
    assert result.metadata_appended
    saved_metadata = read_wav_calibration_metadata(req.path)["recorded_channels"]
    for saved_channel, expected in zip(saved_metadata, metadata["recorded_channels"], strict=True):
        assert all(saved_channel[key] == value for key, value in expected.items())
    assert len(attempts) == 2
    assert [attempt["device"] for attempt in attempts] == [7, 8]
    assert req.device["index"] == 7 and req.device["hostapi"] == 0
    assert worker.names().count("library_initialized") == 1
    kinds = [event.kind for event in worker.control.sent]
    assert kinds.count("started") == kinds.count("completed") == 1
    assert "failed" not in kinds
    worker.command("result_ack", req, "accepted")
    worker.shutdown()
    assert not worker.errors


def fail_open(worker, monkeypatch):
    attempts = []

    def fail(**config):
        attempts.append(config)
        raise OSError("native open failed")

    monkeypatch.setattr(worker.backend, "InputStream", fail)
    return attempts


@pytest.mark.parametrize("preflight", [False, True])
def test_persistent_native_failure_uses_one_budget(tmp_path, harness, monkeypatch, preflight):
    worker = harness()
    attempts = fail_open(worker, monkeypatch)
    if preflight:
        worker.backend.device["name"] = "stale"
    req = request(tmp_path)
    worker.command("start", req)
    failed = worker.receive("failed").payload
    assert failed.stage == "device" and failed.handles_released
    assert req.device["name"] in failed.message and "HostAPI" in failed.message
    assert "native open failed" in failed.message and "reselect" in failed.message
    assert len(attempts) == (1 if preflight else 2)
    assert worker.names().count("library_initialized") == 1
    assert sum(event.kind == "failed" for event in worker.control.sent) == 1
    worker.command("result_ack", req, "rejected")
    worker.shutdown()
    assert not worker.errors and worker.pipeline.empty


@pytest.mark.parametrize("duplicate", [False, True])
def test_retry_rebind_missing_or_ambiguous_identity_never_opens_again(tmp_path, harness, monkeypatch, duplicate):
    worker = harness()
    attempts = fail_open(worker, monkeypatch)
    worker.backend.apis = [{"name": "Selected API"}]
    worker.backend.inventory = [worker.backend.device.copy()]
    worker.backend.fresh_apis = worker.backend.apis
    worker.backend.fresh_inventory = (
        [worker.backend.device.copy(), {**worker.backend.device, "index": 8}]
        if duplicate else [{**worker.backend.device, "name": "different microphone"}])
    req = request(tmp_path, device={**worker.backend.device, "hostapi_name": "Selected API"})
    worker.command("start", req)
    failure = worker.receive("failed").payload
    assert ("ambiguous" if duplicate else "missing") in failure.message
    assert "Selected API" in failure.message and "reselect" in failure.message
    assert len(attempts) == 1 and worker.names().count("library_initialized") == 1
    worker.command("result_ack", req, "rejected")
    worker.shutdown()
    assert not worker.errors


@pytest.mark.parametrize("command", ["cancel", "shutdown"])
@pytest.mark.parametrize("when", ["before_join", "during_refresh"])
def test_cancel_or_shutdown_prevents_replacement(tmp_path, harness, monkeypatch, command, when):
    worker = harness(hold_thread=when == "before_join")
    attempts = fail_open(worker, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    initialize = worker.backend._initialize

    def blocked_initialize():
        entered.set()
        assert release.wait(3)
        initialize()

    monkeypatch.setattr(worker.backend, "_initialize", blocked_initialize)
    req = request(tmp_path)
    try:
        worker.command("start", req)
        assert (worker.thread_done if when == "before_join" else entered).wait(3)
        worker.command(command, req if command == "cancel" else None)
        release.set()
        worker.release_thread.set()
        worker.receive("failed")
        if command == "cancel":
            worker.command("result_ack", req, "rejected")
            worker.shutdown()
        else:
            worker.worker.join(3)
        assert not worker.worker.is_alive()
        assert len(attempts) == 1
        assert worker.names().count("library_initialized") == (when == "during_refresh")
        assert not worker.errors
    finally:
        release.set()
        worker.release_thread.set()


@pytest.mark.parametrize("failure", ["terminate", "initialize"])
def test_native_retry_reset_failure_orders_terminal_before_fatal(tmp_path, harness, monkeypatch, failure):
    from unit_test.base.test_recording_audio_lifetime import HardExit

    worker = harness(**{f"{failure}_fails": True})
    attempts = fail_open(worker, monkeypatch)
    worker.command("start", request(tmp_path))
    worker.worker.join(3)
    kinds = [event.kind for event in worker.control.sent]
    assert kinds == ["ready", "failed", "worker_fatal"]
    assert len(attempts) == 1 and worker.pipeline.empty
    assert worker.names().count("library_terminated") == 1
    assert worker.names().count("library_initialized") == (failure == "initialize")
    assert len(worker.errors) == 1 and isinstance(worker.errors[0], HardExit)


@pytest.mark.parametrize("scenario", ["injected", "stream_cleanup", "writer_cleanup", "writer_open",
                                      "metadata", "post_start_zero", "post_start_samples", "identity",
                                      "prestart_samples"])
def test_nonretryable_failures_never_refresh(tmp_path, harness, monkeypatch, scenario):
    from base import recording_worker
    from unit_test.base.recording_process_fakes import ControlledWriter, FakeStatus

    worker = harness(injected=scenario == "injected", close_fails=scenario == "stream_cleanup")
    req = request(tmp_path)
    options = {}
    retained_writer = None
    if scenario == "writer_cleanup":
        retained_writer = ControlledWriter(fail_at="close")
        options["_writer_factory"] = retained_writer
    if scenario == "writer_open":
        def bad_writer(*args, **kwargs):
            raise OSError("disk open failed")
        options["_writer_factory"] = bad_writer
    if scenario == "metadata":
        def bad_metadata(*args, **kwargs):
            raise OSError("metadata failed")
        options["_metadata_appender"] = bad_metadata
    init = recording_worker.RecordingCapture.__init__

    def capture_init(capture, *args, **kwargs):
        init(capture, *args, **kwargs)
        for key, value in options.items():
            setattr(capture, key, value)

    monkeypatch.setattr(recording_worker.RecordingCapture, "__init__", capture_init)
    if scenario in ("injected", "writer_cleanup", "writer_open"):
        fail_open(worker, monkeypatch)
    elif scenario in ("stream_cleanup", "prestart_samples"):
        original_open = worker.backend.InputStream

        def bad_start(**config):
            stream = original_open(**config)
            def fail():
                if scenario == "prestart_samples":
                    stream.feed(known_audio()[:1])
                raise OSError("native start failed")
            stream.start = fail
            return stream
        monkeypatch.setattr(worker.backend, "InputStream", bad_start)
    elif scenario == "identity":
        query = worker.backend.query_devices
        def changed_in_capture(index=None):
            if threading.get_ident() != worker.worker.ident:
                raise ValueError("identity changed between preflight and capture")
            return query(index)
        monkeypatch.setattr(worker.backend, "query_devices", changed_in_capture)
    worker.command("start", req)
    if scenario in ("metadata", "post_start_zero", "post_start_samples"):
        worker.receive("started")
        if scenario == "metadata":
            worker.backend.stream.feed(known_audio())
        else:
            if scenario == "post_start_samples":
                worker.backend.stream.feed(known_audio()[:2])
            worker.backend.stream.feed(known_audio()[:1], status=FakeStatus(input_overflow=True))
    try:
        failure = worker.receive("failed").payload
        assert "library_initialized" not in worker.names()
        assert len(worker.captures) == 1
        if scenario in ("writer_cleanup", "stream_cleanup"):
            assert not failure.handles_released
    finally:
        if retained_writer is not None:
            retained_writer.writer.finalize()
    worker.command("result_ack", req, "rejected")
    worker.shutdown()
