"""Real spawn integration tests. No device or Process replacements."""
from dataclasses import fields, is_dataclass, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from base.recording_service import RecordingCallbacks, RecordingService
from base.recording_result_reader import ResultReader
from unit_test.base.recording_process_fakes import device_info, generated_audio, known_audio
from base.recording_process_protocol import RecordingEvent, RecordingRequest
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_MAX_POINTS,
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_LOWER_BOUND_TOLERANCE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)


def request(tmp_path, **changes):
    values = dict(request_id="one", purpose="main", sample_rate=100,
                  target_samples=9, channels=(0, 2), device=device_info(),
                  path=str(tmp_path / "recording.wav"), streaming=True, trim_samples=2,
                  monitor={}, calibration_metadata=None, validation_thresholds={"enabled": False})
    values.update(changes)
    return RecordingRequest(**values)


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        threading.Event().wait(.01)
    raise AssertionError("condition did not become true before deadline")


def read_trace(tmp_path):
    try:
        return json.loads((tmp_path / "trace.json").read_text())
    except (FileNotFoundError, PermissionError):
        # Windows replacement can temporarily deny a concurrent observer's open.
        # The eventual assertion retries; this trace is not the recording file.
        return {}


def test_fake_trace_io_is_off_audio_producer_and_terminal_flushes_latest_state(
        tmp_path, monkeypatch):
    from unit_test.base import recording_process_fakes as fakes

    publish_entered = threading.Event()
    release_publish = threading.Event()
    all_chunks_fed = threading.Event()
    fed = []
    replace_attempts = []
    atomic_publish = fakes._atomic_publish_trace
    real_replace = fakes.os.replace

    def contended_replace(*args, **kwargs):
        replace_attempts.append(args)
        if len(replace_attempts) < 3:
            raise PermissionError("simulated Windows trace reader contention")
        return real_replace(*args, **kwargs)

    def blocked_publish(*args, **kwargs):
        publish_entered.set()
        assert release_publish.wait(3), "test did not release trace publication"
        return atomic_publish(*args, **kwargs)

    monkeypatch.setattr(fakes, "_atomic_publish_trace", blocked_publish)
    monkeypatch.setattr(fakes.os, "replace", contended_replace)
    dependencies = fakes.process_dependencies(
        trace_dir=str(tmp_path), frames=8, chunk_frames=2, pace_writer=False)

    def consume(*_args):
        fed.append(1)
        if len(fed) == 4:
            all_chunks_fed.set()

    stream = dependencies["backend"].InputStream(channels=3, callback=consume)
    try:
        stream.start()
        assert publish_entered.wait(1)
        assert all_chunks_fed.wait(1), "trace I/O blocked the simulated audio producer"
    finally:
        release_publish.set()
        stream.stop()

    dependencies["backend"].close_trace()
    trace = read_trace(tmp_path)
    assert trace["open_round"] == 1
    assert trace["fed_chunks"] == 4
    assert trace["capture_pid"] == os.getpid()
    assert trace["stop_entered"] is True
    assert len(replace_attempts) >= 3


def test_fake_trace_close_is_terminal_after_publication_error_and_idempotent(
        tmp_path, monkeypatch):
    from unit_test.base import recording_process_fakes as fakes

    publisher = fakes._CoalescedTracePublisher(tmp_path)
    monkeypatch.setattr(
        fakes, "_atomic_publish_trace",
        lambda *_args: (_ for _ in ()).throw(OSError("trace disk failed")))
    publisher.update(terminal=True)

    with pytest.raises(OSError, match="trace disk failed"):
        publisher.close()

    assert publisher._closed
    assert not publisher._thread.is_alive()
    publisher.close()
    with pytest.raises(RuntimeError, match="closed"):
        publisher.update(late=True)


def test_fake_trace_close_is_bounded_after_publish_timeout(tmp_path, monkeypatch):
    from unit_test.base import recording_process_fakes as fakes

    entered = threading.Event()
    release_stall = threading.Event()

    def stalled_publish(trace_dir, trace, cancelled):
        # The third argument is the supported cooperative cancellation signal.
        entered.set()
        while not cancelled.is_set():
            if release_stall.wait(.005):
                (trace_dir / "trace.json").write_text(json.dumps(trace))
                return
        raise fakes._TracePublicationCancelled("test publication cancelled")

    monkeypatch.setattr(fakes, "_atomic_publish_trace", stalled_publish)
    publisher = fakes._CoalescedTracePublisher(tmp_path)
    publisher.update(terminal=True)
    assert entered.wait(1)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="did not flush"):
            publisher.close(timeout=.12)
        assert time.monotonic() - started <= .2
        assert publisher._closed
        assert not publisher._thread.is_alive()
        assert not (tmp_path / "trace.json").exists()
        release_stall.set()
        threading.Event().wait(.03)
        assert not (tmp_path / "trace.json").exists()
        publisher.close(timeout=.12)
    finally:
        release_stall.set()
        cancel = getattr(publisher, "_cancel", None)
        if cancel is not None:
            cancel.set()
        publisher._wake.set()
        publisher._thread.join(1)


def test_fake_trace_cancelled_permission_retry_cleans_temporary_file(
        tmp_path, monkeypatch):
    from unit_test.base import recording_process_fakes as fakes

    attempts = []

    def denied_replace(*_args):
        attempts.append(1)
        raise PermissionError("reader held trace")

    monkeypatch.setattr(fakes.os, "replace", denied_replace)
    publisher = fakes._CoalescedTracePublisher(tmp_path)
    publisher.update(terminal=True)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="did not flush"):
        publisher.close(timeout=.12)

    assert time.monotonic() - started <= .2
    assert attempts
    assert publisher._closed
    assert not publisher._thread.is_alive()
    assert not (tmp_path / "trace.tmp").exists()
    assert not (tmp_path / "trace.json").exists()


def test_fake_trace_terminal_close_publishes_exact_latest_state_and_stops_worker(tmp_path):
    from unit_test.base import recording_process_fakes as fakes

    publisher = fakes._CoalescedTracePublisher(tmp_path)
    publisher.update(round=1, stale="old")
    publisher.flush()
    publisher.update(round=2, terminal=True)
    publisher.close()

    assert read_trace(tmp_path) == {"round": 2, "stale": "old", "terminal": True}
    assert not publisher._thread.is_alive()


def test_fake_process_writer_preserves_close_failure_when_trace_flush_also_fails(
        tmp_path, monkeypatch):
    from unit_test.base import recording_process_fakes as fakes

    dependencies = fakes.process_dependencies(
        trace_dir=str(tmp_path), fail_close=True)
    writer = dependencies["writer_factory"](
        str(tmp_path / "dual-failure.wav"), 8000, 1)
    monkeypatch.setattr(
        fakes, "_atomic_publish_trace",
        lambda *_args: (_ for _ in ()).throw(OSError("trace cleanup failed")))

    try:
        with pytest.raises(OSError, match="injected writer close failure") as captured:
            writer.finalize()
        assert any("trace cleanup failed" in note
                   for note in getattr(captured.value, "__notes__", ()))
    finally:
        writer.sf_file.close()
        try:
            dependencies["backend"].close_trace()
        except OSError:
            pass


class Events:
    def __init__(self):
        self.started = queue.Queue()
        self.preview = queue.Queue()
        self.results = queue.Queue()
        self.accepted = queue.Queue()
        self.failed = queue.Queue()
        self.cancelled = queue.Queue()
        self.callbacks = RecordingCallbacks(
            started=self.started.put,
            preview=lambda session, preview: self.preview.put(preview),
            result_ready=lambda session, result: self.results.put(result),
            accepted=lambda session, result: self.accepted.put(result),
            failed=lambda session, error: self.failed.put(error),
            cancelled=lambda session, result: self.cancelled.put(result))


@pytest.fixture
def services(tmp_path):
    owned = []
    def make(options=None, **kwargs):
        service = RecordingService(
            backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
            backend_options=dict(trace_dir=str(tmp_path), **(options or {})), **kwargs)
        owned.append(service)
        return service
    yield make
    for service in owned:
        service.shutdown()
        assert service.closed.wait(12), service.diagnostics
        assert service.worker_pid is None
        assert not any(thread.is_alive() for thread in service.threads)


@pytest.mark.parametrize("streaming", [False, True])
def test_spawn_pid_complete_arrays_and_healthy_reuse(tmp_path, services, streaming):
    service = services()
    for index in range(2):
        events = Events()
        session = service.start(request(tmp_path, request_id=str(index), streaming=streaming), events.callbacks)
        events.started.get(timeout=10)
        result = events.results.get(timeout=10)
        assert session.state == "delivering" and not session.released.is_set()
        expected = known_audio()[:9, [0, 2]][2:]
        np.testing.assert_array_equal(result.multi, expected)
        np.testing.assert_array_equal(result.mono, expected.mean(axis=1))
        assert result.multi.dtype == result.mono.dtype == np.float32
        assert result.multi.flags.owndata and result.mono.flags.owndata
        trace = eventually(lambda: read_trace(tmp_path))
        assert trace["capture_pid"] == trace["writer_pid"] == session.worker_pid != os.getpid()
        if index:
            assert session.worker_pid == first_pid and session.generation == first_generation
        first_pid, first_generation = session.worker_pid, session.generation
        session.accept_result()
        assert session.released.wait(5)
        assert session.state == "completed"


@pytest.mark.parametrize("pause_seconds", [1.0, 30.0])
def test_paused_preview_credit_keeps_disk_progress_and_next_snapshot_progress_cumulative(
        tmp_path, services, pause_seconds):
    service = services(dict(manual=True))
    events = Events()
    session = service.start(request(tmp_path, target_samples=100, trim_samples=0), events.callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    first = events.preview.get(timeout=5)
    eventually(lambda: read_trace(tmp_path).get("written_frames", 0) >= 3)
    (tmp_path / "feed-1").touch()
    eventually(lambda: read_trace(tmp_path).get("written_frames", 0) >= 7)
    # This is the specified preview-consumer pause, not a timing guess for work.
    assert not threading.Event().wait(pause_seconds)
    assert events.preview.empty()
    assert (tmp_path / "recording.wav").stat().st_size >= 7 * 2 * 4
    session.release_preview(first.sequence)
    second = events.preview.get(timeout=5)
    assert second.sequence == first.sequence + 1 and second.sample_stop >= 7
    assert second.time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
    assert second.waveforms[0].amplitude.max() == np.float32(.95)
    assert second.waveforms[0].time[-1] == 0.0
    assert second.waveforms[0].time[0] >= -MAIN_RECORDING_LIVE_WINDOW_SECONDS
    (tmp_path / "feed-2").touch()
    result = events.results.get(timeout=5)  # No acknowledgement for second preview.
    expected = (known_audio(110) % .5).astype(np.float32)
    expected[5, 0] = .95
    expected = expected[:100, (0, 2)]
    np.testing.assert_array_equal(result.multi, expected)
    np.testing.assert_array_equal(result.mono, expected.mean(axis=1))
    saved, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
    assert rate == 100
    np.testing.assert_array_equal(saved, expected)
    session.accept_result()
    assert session.released.wait(5)


def test_spawn_cumulative_request_publishes_cumulative_preview_and_exact_completion(
        tmp_path, services):
    service = services(dict(manual=True))
    events = Events()
    session = service.start(request(
        tmp_path, target_samples=9, trim_samples=0,
        preview_time_mode=PREVIEW_TIME_MODE_CUMULATIVE,
    ), events.callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    preview = events.preview.get(timeout=5)
    assert preview.time_mode == PREVIEW_TIME_MODE_CUMULATIVE
    assert preview.sample_stop == 3
    for waveform in preview.waveforms:
        assert waveform.time[0] == 0.0
        assert waveform.time[-1] > 0.0
        assert np.all(np.diff(waveform.time) > 0.0)
    session.release_preview(preview.sequence)
    (tmp_path / "feed-1").touch()
    (tmp_path / "feed-2").touch()
    result = events.results.get(timeout=5)
    expected = (known_audio(110) % .5).astype(np.float32)
    expected[5, 0] = .95
    expected = expected[:9, (0, 2)]
    np.testing.assert_array_equal(result.multi, expected)
    np.testing.assert_array_equal(result.mono, expected.mean(axis=1))
    saved, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
    assert rate == 100
    np.testing.assert_array_equal(saved, expected)
    session.accept_result()
    assert events.accepted.get(timeout=5) is result
    assert session.released.wait(5)
    assert session.state == "completed" and events.failed.empty()


@pytest.mark.parametrize("preview_fault", ["construct", "begin", "append", "snapshot"])
def test_spawn_preview_fault_does_not_change_progress_completion_or_success(
        tmp_path, services, preview_fault):
    service = services(dict(manual=True, preview_fault=preview_fault))
    events = Events()
    session = service.start(request(
        tmp_path, target_samples=9, trim_samples=0,
        preview_time_mode=PREVIEW_TIME_MODE_CUMULATIVE,
    ), events.callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    eventually(lambda: read_trace(tmp_path).get("preview_fault_observed") == preview_fault)
    (tmp_path / "feed-1").touch()
    (tmp_path / "feed-2").touch()
    result = events.results.get(timeout=5)

    expected = (known_audio(110) % .5).astype(np.float32)
    expected[5, 0] = .95
    expected = expected[:9, (0, 2)]
    trace = eventually(lambda: read_trace(tmp_path).get("written_frames") == 9 and read_trace(tmp_path))
    assert trace["preview_session_constructions"] == 1
    assert trace["preview_rolling_window_seconds"] is None
    assert (result.descriptor.raw_frames, result.descriptor.final_frames) == (9, 9)
    np.testing.assert_array_equal(result.multi, expected)
    np.testing.assert_array_equal(result.mono, expected.mean(axis=1))
    saved, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
    assert rate == 100
    np.testing.assert_array_equal(saved, expected)
    assert events.preview.empty() and events.failed.empty() and events.cancelled.empty()
    session.accept_result()
    assert events.accepted.get(timeout=5) is result
    assert session.released.wait(5)
    assert session.state == "completed"


@pytest.mark.parametrize("signal", ["ack", "stop", "stop_at_deadline", "ack_at_deadline", "timeout"])
def test_synthetic_writer_wait_is_ack_woken_cancelable_and_bounded(tmp_path, monkeypatch, signal):
    """Drive the actual fake feeder with deterministic event/clock scheduling.

    The writer notifies while the feeder is waiting. Only waiting on that
    event can wake at notification time; a stop wait must run to its timeout.
    No wall-clock sleep or thread scheduling threshold decides this regression.
    """
    from unit_test.base import recording_process_fakes as fakes

    state = dict(now=100.0, pending=None, ack=False, stop=False, waits=[])

    class ScheduledEvent:
        def __init__(self, name):
            self.name = name

        def is_set(self):
            return state[self.name]

        def clear(self):
            state[self.name] = False

        def set(self):
            state[self.name] = True

        def wait(self, timeout):
            assert timeout == .005
            state["waits"].append(self.name)
            if self.is_set():
                return True
            end = state["now"] + timeout
            pending = state["pending"]
            if pending is not None and pending[1] <= end:
                name, at = pending
                state[name] = True
                state["pending"] = None
                if name == self.name:
                    state["now"] = at
                    return True
            state["now"] = end
            return self.is_set()

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    events = iter((ScheduledEvent("ack"), ScheduledEvent("stop")))
    monkeypatch.setattr(fakes, "threading", SimpleNamespace(
        Event=lambda: next(events), Lock=threading.Lock, Thread=InlineThread))
    monkeypatch.setattr(fakes, "time", SimpleNamespace(monotonic=lambda: state["now"]))
    dependencies = fakes.process_dependencies(trace_dir=str(tmp_path), frames=4, chunk_frames=2)
    delivered = []

    def consume(chunk, *args):
        delivered.append(chunk.copy())
        if signal != "timeout":
            name = "ack" if signal.startswith("ack") else "stop"
            delay = 10 if signal.endswith("at_deadline") else .001
            state["pending"] = (name, state["now"] + delay)

    stream = dependencies["backend"].InputStream(channels=3, callback=consume)
    stream.start()
    dependencies["backend"].flush_trace()
    trace = read_trace(tmp_path)
    dependencies["backend"].close_trace()
    if signal == "ack":
        assert state["now"] == pytest.approx(100.002), "writer acknowledgement did not wake the feeder"
        assert state["waits"] == ["ack", "ack"]
        assert trace["fed_chunks"] == 2
        assert "feeder_error" not in trace
        np.testing.assert_array_equal(np.concatenate(delivered), generated_audio(0, 4))
    else:
        assert trace["fed_chunks"] == 1
        np.testing.assert_array_equal(delivered[0], generated_audio(0, 2))
        if signal in ("timeout", "ack_at_deadline"):
            assert 110 <= state["now"] <= 110.005 + 1e-9
            assert trace["feeder_error"] == "synthetic writer acknowledgement timed out"
        else:
            stopped_at = 110 if signal == "stop_at_deadline" else 100.001
            assert stopped_at <= state["now"] <= stopped_at + .005 + 1e-9
            assert "feeder_error" not in trace


@pytest.mark.parametrize("pace_writer", [False, True])
def test_synthetic_producer_writer_handshake_prevents_scheduling_overflow(
        tmp_path, services, pace_writer):
    """A held real writer reproduces the old faster-than-hardware fixture race."""
    service = services(dict(frames=441017, chunk_frames=16000,
                            pause_first_write=True, pace_writer=pace_writer))
    events = Events()
    session = service.start(request(tmp_path, sample_rate=44100, target_samples=441000,
                                    trim_samples=0), events.callbacks)
    try:
        eventually(lambda: read_trace(tmp_path).get("writer_entered"))
        if pace_writer:
            eventually(lambda: read_trace(tmp_path).get("producer_waiting_for_writer"), timeout=2)
            trace = read_trace(tmp_path)
            assert trace["fed_chunks"] == 1
            assert trace.get("written_frames", 0) == 0
        else:
            # Seven 16000-frame callbacks exceed the 88200-frame queue while
            # the first consumer-owned block is held at the writer boundary.
            eventually(lambda: read_trace(tmp_path).get("fed_chunks", 0) >= 7)
    finally:
        (tmp_path / "release-writer").touch()
    if pace_writer:
        audio = events.results.get(timeout=10)
        np.testing.assert_array_equal(audio.multi, generated_audio(0, 441000)[:, (0, 2)])
        session.accept_result()
    else:
        failure = events.failed.get(timeout=10)
        assert failure.stage == "capture" and "queue capacity exceeded" in failure.message
        assert events.results.empty() and events.accepted.empty()
    assert session.released.wait(5)


def test_one_active_session_is_registered_before_async_start(tmp_path, services):
    service = services(dict(manual=True))
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    with pytest.raises(RuntimeError, match="busy"):
        service.start(request(tmp_path, request_id="two", path=str(tmp_path / "other.wav")))
    events.started.get(timeout=10)
    session.cancel()
    events.cancelled.get(timeout=5)
    assert session.released.wait(5)


@pytest.mark.parametrize("channels", [(2,), (2, 0)])
def test_spawn_accepts_600_seconds_with_bounded_rolling_previews_and_exact_wav(
        tmp_path, services, monkeypatch, channels):
    """600 simulated audio seconds at 1000 Hz, not a hardware throughput test."""
    sample_rate, trim, final_frames = 1000, 250, 600000
    target = final_frames + trim
    service = services(dict(frames=target + 17, chunk_frames=1999, pause_finalize=True),
                       preview_interval=.01)
    received = []
    dispatch = service._event
    def inspect_ipc(worker, event):
        received.append(event)
        dispatch(worker, event)
    monkeypatch.setattr(service, "_event", inspect_ipc)
    events = Events()
    session = service.start(request(tmp_path, target_samples=target, trim_samples=trim,
                                    sample_rate=sample_rate, channels=channels), events.callbacks)
    try:
        first = events.preview.get(timeout=10)
        # Withhold the one preview credit until every raw frame is on disk.
        # Wait on supervisor state while the 300 paced writes and rolling reduction
        # run; repeatedly opening trace.json can starve the child's atomic replace
        # on Windows. Once finalizing, poll only the final writer gate at low cadence.
        eventually(lambda: session.state == "finalizing", timeout=120)
        trace_deadline = time.monotonic() + 10
        while True:
            trace = read_trace(tmp_path)
            if trace.get("finalize_waiting_for_release"):
                break
            if time.monotonic() >= trace_deadline:
                raise AssertionError("finalization trace did not reach its release gate")
            threading.Event().wait(.1)
        assert trace["written_frames"] == target
        assert trace["capture_pid"] == trace["writer_pid"] == session.worker_pid != os.getpid()
        assert events.preview.empty()
        assert events.results.empty(), "finalization gate must still own the open WAV"
        session.release_preview(first.sequence)
        rolling = events.preview.get(timeout=5)
        assert rolling.sequence == first.sequence + 1
        assert rolling.channels == channels
        assert rolling.sample_stop == final_frames
        for preview in (first, rolling):
            assert preview.time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            for waveform in preview.waveforms:
                assert 0 < len(waveform.time) <= MAIN_RECORDING_LIVE_MAX_POINTS
                assert waveform.time.shape == waveform.amplitude.shape
                assert waveform.time[-1] == 0.0
                assert np.all(waveform.time <= 0.0)
                assert waveform.time[0] >= (
                    -MAIN_RECORDING_LIVE_WINDOW_SECONDS - PREVIEW_TIME_LOWER_BOUND_TOLERANCE
                )
        assert rolling.waveforms[0].time[-1] == 0.0
        assert rolling.waveforms[0].time[0] < -9.9
    finally:
        (tmp_path / "release-finalize").touch()
    # Completion must bypass the still-withheld second preview credit.
    audio = events.results.get(timeout=10)
    expected = generated_audio(0, target)[trim:, channels]
    np.testing.assert_array_equal(audio.multi, expected)
    np.testing.assert_array_equal(audio.mono, expected.mean(axis=1))
    assert (audio.descriptor.raw_frames, audio.descriptor.final_frames) == (target, final_frames)
    assert audio.descriptor.final_frames / audio.descriptor.sample_rate == 600
    saved, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
    assert rate == sample_rate
    np.testing.assert_array_equal(saved, expected)
    session.accept_result()
    assert events.accepted.get(timeout=5) is audio
    assert session.released.wait(5)
    assert session.state == "completed" and events.failed.empty() and events.preview.empty()

    def arrays(value):
        if isinstance(value, np.ndarray):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from arrays(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from arrays(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from arrays(item)
    assert sum(event.kind == "completed" for event in received) == 1
    for event in received:
        payload_arrays = list(arrays(event.payload))
        if event.kind == "preview":
            assert len(payload_arrays) == 2 * len(channels)
            assert all(array.ndim == 1 and array.size <= 4000 for array in payload_arrays)
        else:
            assert not payload_arrays, "raw audio history must never cross control IPC"


class PausedReader:
    """Pause a real open SoundFile read, retaining its handle until released."""
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def __call__(self, descriptor, completed):
        owner = self
        class Source:
            def __init__(self, *args, **kwargs):
                self.source = sf.SoundFile(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.source, name)

            def __len__(self):
                return len(self.source)

            def read(self, *args, **kwargs):
                owner.entered.set()
                assert owner.release.wait(15), "test must release its paused reader"
                return self.source.read(*args, **kwargs)

            def close(self):
                self.source.close()
                owner.closed.set()
        return ResultReader(descriptor, completed, opener=Source, block_frames=2)


@pytest.mark.parametrize("action", ["cancel", "reject_result", "shutdown"])
def test_reader_lease_defers_ack_cleanup_and_reuse(tmp_path, services, action):
    paused = PausedReader()
    service = services(reader_factory=paused)
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(2,)), events.callbacks)
    try:
        assert paused.entered.wait(10)
        path = Path(session.request.path)
        assert path.exists() and path.parent != tmp_path
        if action == "shutdown":
            service.shutdown()
        else:
            getattr(session, action)()
        eventually(lambda: session.state in ("cancelled", "failed"))
        assert not session.acknowledged and not session.released.is_set()
        assert not paused.closed.is_set() and path.exists()
        assert service.is_path_leased(str(path))
        with pytest.raises(RuntimeError):
            service.start(request(tmp_path, request_id="retry", path=str(path)))
    finally:
        paused.release.set()
    assert session.released.wait(5)
    assert paused.closed.is_set() and not path.parent.exists()
    assert events.results.empty() and events.accepted.empty()


def test_reader_timeout_retires_worker_and_isolates_only_old_path(tmp_path, services):
    paused = PausedReader()
    service = services(reader_factory=paused, cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    try:
        assert paused.entered.wait(10)
        old_pid = session.worker_pid
        session.cancel()
        events.cancelled.get(timeout=3)
        eventually(lambda: service.worker_pid is None)
        assert not session.acknowledged and not session.released.is_set()
        assert Path(session.request.path).exists()
        with pytest.raises(RuntimeError, match="leased"):
            service.start(request(tmp_path, request_id="same"))
        fresh_events = Events()
        fresh = service.start(request(tmp_path, request_id="fresh", path=str(tmp_path / "fresh.wav")),
                              fresh_events.callbacks)
        fresh_events.started.get(timeout=10)
        assert fresh.worker_pid != old_pid and fresh.generation > session.generation
        paused.release.set()
        assert session.released.wait(5)
        offered = fresh_events.results.get(timeout=5)
        fresh.accept_result()
        fresh_events.accepted.get(timeout=5)
        assert fresh.released.wait(5) and Path(fresh.request.path).exists()
        assert events.results.empty() and events.failed.empty()
        assert offered.multi.shape == (7, 2)
    finally:
        paused.release.set()


def test_trusted_old_reader_preview_ack_cannot_fault_replacement_worker(tmp_path, services):
    paused = PausedReader()

    def reader_factory(descriptor, completed):
        if descriptor.request_id == "old":
            return paused(descriptor, completed)
        return ResultReader(descriptor, completed)

    service = services(reader_factory=reader_factory, terminate_timeout=.2)
    old_events = Events()
    old = service.start(request(tmp_path, request_id="old", path=str(tmp_path / "old.wav")),
                        old_events.callbacks)
    try:
        assert paused.entered.wait(10)
        old_worker = service._worker
        old_worker.process.terminate()
        eventually(lambda: service.worker_pid is None)
        assert old._trusted_terminal and old.state == "delivering"

        fresh_events = Events()
        fresh = service.start(request(
            tmp_path, request_id="fresh", path=str(tmp_path / "fresh.wav"),
            target_samples=100, trim_samples=0), fresh_events.callbacks)
        fresh_events.started.get(timeout=10)
        replacement = service._worker
        assert replacement.generation > old.generation

        old._preview_pending = 17
        old.release_preview(17)
        eventually(lambda: old._preview_pending is None)
        assert not threading.Event().wait(.3)
        assert service._worker is replacement and not replacement.retiring
        assert fresh_events.failed.empty()

        fresh.cancel()
        fresh_events.cancelled.get(timeout=5)
        assert fresh.released.wait(5)
    finally:
        paused.release.set()
    old_events.results.get(timeout=5)
    old.accept_result()
    assert old.released.wait(5)


@pytest.mark.parametrize("stage", ["recording", "finalizing", "delivering"])
def test_worker_death_before_acceptance_fails_once_and_new_generation(tmp_path, services, stage):
    options = dict(manual=True) if stage == "recording" else dict(hang_finalize=True) if stage == "finalizing" else {}
    service = services(options)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.started.get(timeout=10)
    if stage == "finalizing":
        eventually(lambda: read_trace(tmp_path).get("finalize_entered"))
    elif stage == "delivering":
        events.results.get(timeout=10)
    worker = service._worker
    worker.process.terminate()
    eventually(lambda: service.worker_pid is None)
    if stage == "delivering":
        session.accept_result()
        events.accepted.get(timeout=5)
        assert session.released.wait(5)
        assert session.state == "completed" and Path(session.request.path).exists()
        assert events.failed.empty()
    else:
        events.failed.get(timeout=5)
        assert session.released.wait(5)
        session.accept_result()
        assert events.failed.empty() and events.accepted.empty()
        assert session.state == "failed" and not Path(session.request.path).exists()
    next_events = Events()
    fresh = service.start(request(tmp_path, request_id="next"), next_events.callbacks)
    next_events.started.get(timeout=10)
    assert fresh.worker_pid != session.worker_pid and fresh.generation > session.generation
    fresh.cancel()


@pytest.mark.parametrize("scenario,stage", [("ready", "ready_timeout"), ("start", "start_timeout"),
                                          ("cancel", "cancel_timeout")])
def test_startup_and_cancel_deadlines_retire_real_worker(tmp_path, services, scenario, stage):
    options = {"hang_ready": True} if scenario == "ready" else (
        {"hang_start_round": 1} if scenario == "start" else {"manual": True, "hang_close": True})
    service = services(options, ready_timeout=.5 if scenario == "ready" else 5,
                       start_timeout=.3, cancel_timeout=.3, terminate_timeout=.2)
    events = Events()
    before = time.monotonic()
    session = service.start(request(tmp_path), events.callbacks)
    assert time.monotonic() - before < .1
    if scenario == "cancel":
        events.started.get(timeout=10)
        session.cancel()
    failure = events.failed.get(timeout=10)
    assert failure.stage == stage
    assert session.released.wait(5)
    eventually(lambda: service.worker_pid is None)
    assert events.results.empty() and events.accepted.empty() and events.failed.empty()


def test_healthy_ready_does_not_remove_next_round_start_deadline(tmp_path, services):
    service = services(dict(hang_start_round=2), start_timeout=.3, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.results.get(timeout=10)
    session.accept_result()
    assert session.released.wait(5)
    next_events = Events()
    fresh = service.start(request(tmp_path, request_id="second"), next_events.callbacks)
    assert next_events.failed.get(timeout=5).stage == "start_timeout"
    assert fresh.worker_pid == session.worker_pid
    assert fresh.released.wait(5)
    eventually(lambda: service.worker_pid is None)


def test_unreleased_writer_handles_force_retirement_before_cleanup(tmp_path, services):
    service = services(dict(fail_close=True), terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    failure = events.failed.get(timeout=10)
    assert not failure.handles_released and "close failure" in failure.message
    assert session.released.wait(5)
    assert service.worker_pid is None and not Path(session.request.path).exists()


def test_result_offer_requires_acceptance_and_reject_is_failure(tmp_path, services):
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    offered = events.results.get(timeout=10)
    assert events.accepted.empty() and not session.released.is_set()
    session.reject_result("workspace channels differ")
    failure = events.failed.get(timeout=5)
    assert failure.stage == "semantic" and "channels differ" in failure.message
    assert session.released.wait(5)
    assert not Path(offered.descriptor.path).exists()


def test_result_read_failure_never_offers_arrays(tmp_path, services):
    def unreadable(descriptor, completed):
        def fail_open(*args, **kwargs):
            raise OSError("injected reader permission failure")
        return ResultReader(descriptor, completed, opener=fail_open)
    service = services(reader_factory=unreadable)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    assert events.failed.get(timeout=10).stage == "read"
    assert session.released.wait(5) and events.results.empty()


def test_duplicate_and_stale_events_cannot_redeliver_or_overwrite(tmp_path, services):
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    result = events.results.get(timeout=10)
    worker = service._worker
    duplicate = RecordingEvent(session.generation, session.request.request_id, "completed", result.descriptor)
    service._inbox.put(("event", worker, duplicate))
    service._inbox.put(("event", worker, replace(duplicate, generation=session.generation + 1)))
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5)
    assert events.results.empty() and events.accepted.empty() and events.failed.empty()
    service._inbox.put(("event", worker, duplicate))
    session.cancel()
    assert session.state == "completed"


def test_contradictory_duplicate_terminal_retires_generation_without_rewriting_trusted_session(
        tmp_path, services):
    service = services(terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    result = events.results.get(timeout=10)
    worker = service._worker
    contradictory = replace(result.descriptor, warnings=("contradictory duplicate",))
    service._inbox.put(("event", worker, RecordingEvent(
        session.generation, session.request.request_id, "completed", contradictory)))
    eventually(lambda: service.worker_pid is None)
    assert session.state == "delivering" and session.failure is None
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5)
    assert events.failed.empty()


def test_malformed_duplicate_terminal_preserves_trusted_reader_through_worker_death(
        tmp_path, services):
    service = services(terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    result = events.results.get(timeout=10)
    worker = service._worker
    malformed = RecordingEvent.__new__(RecordingEvent)
    object.__setattr__(malformed, "generation", session.generation)
    object.__setattr__(malformed, "request_id", session.request.request_id)
    object.__setattr__(malformed, "kind", "completed")
    object.__setattr__(malformed, "payload", replace(result.descriptor, request_id="other"))
    object.__setattr__(malformed, "version", 1)

    service._inbox.put(("event", worker, malformed))
    eventually(lambda: service.worker_pid is None)

    assert session._trusted_terminal and session._child_released
    assert session.state == "delivering" and session.failure is None
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5)
    assert events.failed.empty()


def test_shutdown_deadline_handles_native_close_hang(tmp_path, services):
    service = services(dict(manual=True, hang_close=True), cancel_timeout=.2,
                       shutdown_timeout=.3, terminate_timeout=.2)
    events = Events()
    service.start(request(tmp_path), events.callbacks)
    events.started.get(timeout=10)
    callback = threading.Event()
    before = time.monotonic()
    service.shutdown(callback.set)
    assert time.monotonic() - before < .1
    assert callback.wait(5) and service.closed.wait(5)
    assert service.worker_pid is None


def test_invalid_backend_payload_rejected_without_process_or_device():
    with pytest.raises(ValueError, match="identifier"):
        RecordingService(backend_factory=lambda: None)
    with pytest.raises(ValueError, match="serializable"):
        RecordingService(backend_factory="test:backend", backend_options={"lock": threading.Event()})


def test_request_ids_cannot_be_reused_within_worker_generation(tmp_path, services):
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.results.get(timeout=10)
    session.accept_result()
    assert session.released.wait(5)
    with pytest.raises(ValueError, match="request_id"):
        service.start(request(tmp_path))


def test_shutdown_with_stuck_reader_reports_boundedly_but_retains_lease(tmp_path, services):
    paused = PausedReader()
    service = services(reader_factory=paused, cancel_timeout=.2, shutdown_timeout=.25, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(0,)), events.callbacks)
    callback = threading.Event()
    try:
        assert paused.entered.wait(10)
        service.shutdown(callback.set)
        assert callback.wait(3), "shutdown must report pending file release without waiting for stuck I/O"
        assert service.worker_pid is None
        assert service.is_path_leased(session.request.path) and Path(session.request.path).exists()
        assert any("leased" in message for message in service.diagnostics)
        assert not session.released.is_set() and not session.acknowledged
    finally:
        paused.release.set()
    assert session.released.wait(5)
    assert not Path(session.request.path).parent.exists()


def test_exact_owned_metadata_temporary_is_cleaned_only_after_worker_death(tmp_path, services):
    unrelated = tmp_path / "unrelated.wav"
    unrelated.write_bytes(b"keep")
    service = services(dict(metadata_retained=True), terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    failure = events.failed.get(timeout=10)
    assert failure.stage == "metadata" and not failure.handles_released
    assert failure.cleanup_paths == (str(tmp_path / "exact-owned-temporary.wav"),)
    assert session.released.wait(5)
    assert service.worker_pid is None
    assert not Path(failure.cleanup_paths[0]).exists()
    assert unrelated.read_bytes() == b"keep"


@pytest.mark.skipif(os.name != "nt", reason="Windows process-handle liveness acceptance")
@pytest.mark.parametrize("hang_close", [False, True])
def test_os_parent_death_ends_worker_even_if_native_close_hangs(tmp_path, hang_close):
    from unit_test.base.recording_process_fakes import orphan_service_parent, open_process_observer
    context = mp.get_context("spawn")
    incoming, outgoing = context.Pipe(duplex=False)
    parent = context.Process(target=orphan_service_parent,
                             args=(outgoing, dict(trace_dir=str(tmp_path), manual=True, hang_close=hang_close)))
    kernel = handle = None
    try:
        parent.start()
        outgoing.close()
        assert incoming.poll(15)
        worker_pid = incoming.recv()
        kernel, handle = open_process_observer(worker_pid)
        assert kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT: worker alive.
        parent.terminate()
        parent.join(5)
        assert not parent.is_alive()
        eventually(lambda: kernel.WaitForSingleObject(handle, 0) == 0, timeout=5)
    finally:
        if parent.is_alive():
            parent.terminate()
        parent.join(5)
        parent.close()
        incoming.close()
        outgoing.close()
        if handle:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                assert kernel.WaitForSingleObject(handle, 5000) == 0
            kernel.CloseHandle(handle)


def test_backend_import_failure_preserves_application_error(tmp_path):
    service = RecordingService(backend_factory="nonexistent_recording_test_backend:factory", terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    try:
        error = events.failed.get(timeout=10)
        assert "nonexistent_recording_test_backend" in error.message
        assert session.released.wait(5)
    finally:
        service.shutdown()
        assert service.closed.wait(5)


def test_spawn_os_failure_is_async_releases_lease_and_shuts_down(tmp_path, services, monkeypatch):
    import multiprocessing.spawn
    monkeypatch.setattr(multiprocessing.spawn, "get_executable", lambda: str(tmp_path / "missing-python.exe"))
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    failure = events.failed.get(timeout=5)
    assert failure.stage == "service"
    assert session.released.wait(3)
    assert service.worker_pid is None and not service.is_path_leased(session.request.path)


def test_calibration_success_uses_owned_temp_without_touching_requested_path(tmp_path, services):
    requested = tmp_path / "recording.wav"
    requested.write_bytes(b"unrelated product recording")
    service = services()
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(2,), target_samples=9), events.callbacks)
    audio = events.results.get(timeout=10)
    np.testing.assert_array_equal(audio.multi, known_audio()[:9, [2]])
    assert audio.descriptor.final_frames == 9 and not audio.descriptor.warnings
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5)
    assert not Path(session.request.path).parent.exists()
    assert requested.read_bytes() == b"unrelated product recording"


def test_public_methods_never_perform_pipe_process_or_file_io(tmp_path, services, monkeypatch):
    from multiprocessing.connection import _ConnectionBase
    from multiprocessing.process import BaseProcess
    caller = threading.get_ident()
    observed = []
    for cls, name in ((_ConnectionBase, "send"), (_ConnectionBase, "recv"),
                      (BaseProcess, "start"), (BaseProcess, "join")):
        original = getattr(cls, name)
        def checked(self, *args, _original=original, _name=name, **kwargs):
            observed.append((_name, threading.get_ident()))
            assert threading.get_ident() != caller
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(cls, name, checked)
    def reader(descriptor, completed):
        def opener(*args, **kwargs):
            observed.append(("read", threading.get_ident()))
            assert threading.get_ident() != caller
            return sf.SoundFile(*args, **kwargs)
        return ResultReader(descriptor, completed, opener=opener)
    service = services(reader_factory=reader)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.results.get(timeout=10)
    session.accept_result()
    assert session.released.wait(5)
    service.shutdown()
    assert service.closed.wait(5)
    assert {name for name, _ in observed} >= {"send", "recv", "start", "join", "read"}


def test_acceptance_and_release_continuations_have_explicit_order(tmp_path, services):
    service = services()
    order = []
    offered = threading.Event()
    def accepted(session, audio):
        order.append("accepted")
        assert not session.released.is_set()
        followup.append(service.start(
            request(tmp_path, request_id="overlap", path=str(tmp_path / "other.wav"))))
    followup = []
    callbacks = RecordingCallbacks(result_ready=lambda session, audio: offered.set(), accepted=accepted,
                                   released=lambda session: order.append("released"))
    session = service.start(request(tmp_path), callbacks)
    assert offered.wait(10)
    session.accept_result()
    assert session.released.wait(5)
    eventually(lambda: order == ["accepted", "released"])
    assert followup and followup[0].request.request_id == "overlap"
    followup[0].cancel()
    assert followup[0].released.wait(5)
    assert not service.diagnostics


def test_cancelled_product_file_preserved_after_partial_audio_flush(tmp_path, services):
    service = services(dict(manual=True))
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    eventually(lambda: read_trace(tmp_path).get("written_frames", 0) >= 3)
    session.cancel()
    events.cancelled.get(timeout=5)
    assert session.released.wait(5)
    audio, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(audio, (known_audio(110) % .5)[:3, [0, 2]])
    assert rate == 100 and events.results.empty()


def test_duplicate_pending_preview_does_not_grant_extra_credit(tmp_path, services):
    service = services(dict(manual=True))
    events = Events()
    session = service.start(request(tmp_path, target_samples=100, trim_samples=0), events.callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    first = events.preview.get(timeout=5)
    service._inbox.put(("event", service._worker,
                       RecordingEvent(session.generation, "one", "preview", first)))
    (tmp_path / "feed-1").touch()
    eventually(lambda: read_trace(tmp_path).get("written_frames", 0) >= 7)
    # Several preview intervals without consuming/acknowledging the first one.
    with pytest.raises(queue.Empty):
        events.preview.get(timeout=.25)
    session.release_preview(first.sequence)
    second = events.preview.get(timeout=3)
    assert second.sequence == first.sequence + 1 and second.sample_stop == 7
    session.cancel()
    assert session.released.wait(5)


def test_repeated_healthy_and_retired_workers_do_not_accumulate_threads(tmp_path, services):
    service = services()
    generations = []
    for index in range(6):
        events = Events()
        session = service.start(request(tmp_path, request_id=str(index)), events.callbacks)
        events.results.get(timeout=10)
        generations.append(session.generation)
        if index in (1, 3):
            service._worker.process.terminate()
            eventually(lambda: service.worker_pid is None)
        session.accept_result()
        events.accepted.get(timeout=5)
        assert session.released.wait(5)
        eventually(lambda: len(service.threads) <= 4)
    assert generations == [1, 1, 2, 2, 3, 3]


def test_shutdown_callback_is_reentrant_and_runs_when_already_closed(tmp_path, services):
    service = services()
    first = threading.Event()
    service.shutdown(first.set)
    assert first.wait(5) and service.closed.wait(5)
    again = threading.Event()
    service.shutdown(again.set)
    assert again.wait(2)


def test_finalizing_state_is_observable_while_writer_is_closing(tmp_path, services):
    service = services(dict(hang_finalize=True), cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.started.get(timeout=10)
    eventually(lambda: read_trace(tmp_path).get("finalize_entered"))
    eventually(lambda: session.state == "finalizing", timeout=1)
    session.cancel()
    assert events.failed.get(timeout=5).stage == "cancel_timeout"
    assert session.released.wait(5)


def test_worker_death_after_acceptance_does_not_invalidate_completed_audio(tmp_path, services):
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.results.get(timeout=10)
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5)
    service._worker.process.terminate()
    eventually(lambda: service.worker_pid is None)
    assert session.state == "completed" and events.failed.empty()
    assert Path(session.request.path).exists()


def test_dead_worker_between_offer_and_accept_command_cannot_publish_success(tmp_path, services):
    service = services()
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    events.results.get(timeout=10)
    worker = service._worker
    worker.process.terminate()
    eventually(lambda: not worker.process.is_alive() if service._worker is worker else True)
    session.accept_result()
    events.accepted.get(timeout=5)
    assert session.released.wait(5) and events.failed.empty()


def test_worker_and_service_imports_do_not_load_qt_or_hardware():
    import subprocess
    import sys
    command = "import sys; import base.recording_worker; import base.recording_service; " \
              "assert not any(n == 'ui' or n.startswith(('ui.', 'PyQt', 'PySide')) for n in sys.modules); " \
              "assert 'sounddevice' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_exact_cleanup_paths_remain_leased_until_old_reader_releases(tmp_path, services):
    paused = PausedReader()
    service = services(dict(metadata_leftover=True), reader_factory=paused, cancel_timeout=.2,
                       terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path), events.callbacks)
    owned = tmp_path / "exact-owned-temporary.wav"
    try:
        assert paused.entered.wait(10) and owned.exists()
        session.cancel()
        events.cancelled.get(timeout=5)
        eventually(lambda: service.worker_pid is None)
        with pytest.raises(RuntimeError, match="leased"):
            service.start(request(tmp_path, request_id="unsafe-temp-reuse", path=str(owned)))
    finally:
        paused.release.set()
    assert session.released.wait(5)
    assert not owned.exists() and not service.is_path_leased(str(owned))


def test_preview_callback_failure_disables_preview_once_but_keeps_audio(tmp_path, services):
    service = services(dict(manual=True))
    events = Events()
    failures = []
    def broken_preview(session, preview):
        failures.append(preview.sequence)
        raise RuntimeError("injected preview delivery failure")
    callbacks = replace(events.callbacks, preview=broken_preview)
    session = service.start(request(tmp_path, target_samples=100, trim_samples=0), callbacks)
    events.started.get(timeout=10)
    (tmp_path / "feed-0").touch()
    eventually(lambda: failures)
    (tmp_path / "feed-1").touch()
    eventually(lambda: read_trace(tmp_path).get("written_frames", 0) >= 7)
    # Permit the next preview interval to run while capture is still active.
    assert not threading.Event().wait(.15)
    (tmp_path / "feed-2").touch()
    audio = events.results.get(timeout=5)
    session.accept_result()
    assert session.released.wait(5)
    assert len(failures) == 1
    assert sum("preview callback failed" in text for text in service.diagnostics) == 1
    assert audio.multi.shape == (100, 2) and events.failed.empty()


@pytest.mark.parametrize("failure_at", ["reader_construct", "reader_start", "ipc_control", "ipc_preview"])
def test_thread_start_ownership_failure_classifies_worker_and_reader_ownership(
    tmp_path, services, monkeypatch, failure_at,
):
    message = f"injected {failure_at}: can't start new thread"
    reader_attempts = 0
    if failure_at == "reader_construct":
        def reader_factory(descriptor, completed):
            nonlocal reader_attempts
            reader_attempts += 1
            if reader_attempts == 1:
                raise RuntimeError(message)
            return ResultReader(descriptor, completed)
    else:
        reader_factory = ResultReader
    original_reader_start = ResultReader.start
    if failure_at == "reader_start":
        def fail_reader_start(self):
            raise RuntimeError(message)
        monkeypatch.setattr(ResultReader, "start", fail_reader_start)
    elif failure_at.startswith("ipc_"):
        original_start = threading.Thread.start
        name = "recording-parent-" + failure_at.removeprefix("ipc_") + "-1"
        def fail_ipc_start(thread):
            if thread.name == name:
                raise RuntimeError(message)
            original_start(thread)
        monkeypatch.setattr(threading.Thread, "start", fail_ipc_start)
    service = services(reader_factory=reader_factory, cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(0,)), events.callbacks)
    failure = events.failed.get(timeout=10)
    assert failure.stage == "service" and failure.message == message
    assert session.generation == 1 and session.worker_pid is not None
    assert session.released.wait(3), "no reader started, so no parent file ownership can remain"
    assert not service.is_path_leased(session.request.path)
    assert not Path(session.request.path).parent.exists()
    if failure_at.startswith("reader_"):
        assert service.worker_pid == session.worker_pid
        assert not service._worker.retiring and not service._ownership_uncertain
        assert service.can_start_recording
        if failure_at == "reader_start":
            monkeypatch.setattr(ResultReader, "start", original_reader_start)
        retry_events = Events()
        retry = service.start(request(
            tmp_path, request_id="reader-retry", purpose="calibration", channels=(0,)),
            retry_events.callbacks)
        retry_events.results.get(timeout=10)
        retry.accept_result()
        assert retry.released.wait(3) and retry.worker_pid == session.worker_pid
        assert retry_events.failed.empty()
    else:
        eventually(lambda: service.worker_pid is None, timeout=3)
    callback = threading.Event()
    service.shutdown(callback.set)
    assert callback.wait(3) and service.closed.wait(3)
    assert events.failed.empty() and events.results.empty() and events.accepted.empty()
    assert service.diagnostics == [message]
    assert all(thread.ident is not None for thread in service.threads)


def test_reader_start_that_begins_then_raises_retains_lease_until_reader_exits(tmp_path, services):
    paused = PausedReader()
    def reader_factory(descriptor, completed):
        reader = paused(descriptor, completed)
        original_start = reader.start
        def start_then_fail():
            original_start()
            assert paused.entered.wait(5)
            raise RuntimeError("injected failure after reader really started")
        reader.start = start_then_fail
        return reader
    service = services(reader_factory=reader_factory, cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    session = service.start(request(tmp_path, purpose="calibration", channels=(0,)), events.callbacks)
    try:
        assert "really started" in events.failed.get(timeout=10).message
        assert session.reader.thread.is_alive()
        eventually(lambda: service.worker_pid is None, timeout=3)
        assert service.is_path_leased(session.request.path)
        assert Path(session.request.path).exists()
        assert not session.acknowledged and not session.released.is_set()
    finally:
        paused.release.set()
    assert session.released.wait(5)
    assert paused.closed.is_set() and not Path(session.request.path).parent.exists()
    assert events.failed.empty() and events.results.empty() and events.accepted.empty()


@pytest.mark.parametrize("failure_at,purpose,expected_endpoints", [
    ("mkdtemp", "calibration", 0),
    ("pipe_first", "main", 0),
    ("pipe_second", "main", 2),
    ("pipe_second", "calibration", 2),
    ("process_construct", "main", 4),
    ("worker_construct", "calibration", 4),
])
def test_setup_allocation_failure_rolls_back_and_allows_retry(
    tmp_path, services, monkeypatch, failure_at, purpose, expected_endpoints,
):
    from base import recording_service as module
    context = mp.get_context("spawn")
    original_pipe, original_process = context.Pipe, context.Process
    original_mkdtemp = module.tempfile.mkdtemp
    endpoints, processes, temporary_dirs = [], [], []
    pipe_calls = 0
    message = f"injected setup failure: {failure_at}"
    def allocate_pipe(*args, **kwargs):
        nonlocal pipe_calls
        pipe_calls += 1
        if (failure_at == "pipe_first" and pipe_calls == 1
                or failure_at == "pipe_second" and pipe_calls == 2):
            raise OSError(message)
        pair = original_pipe(*args, **kwargs)
        endpoints.extend(pair)
        return pair
    def allocate_process(*args, **kwargs):
        if failure_at == "process_construct":
            raise OSError(message)
        process = original_process(*args, **kwargs)
        processes.append(process)
        return process
    def allocate_temp(*args, **kwargs):
        if failure_at == "mkdtemp":
            raise OSError(message)
        path = original_mkdtemp(*args, **kwargs)
        temporary_dirs.append(Path(path))
        return path
    def fail_worker(*args, **kwargs):
        raise RuntimeError(message)
    previous = tmp_path / "recording.wav"
    previous.write_bytes(b"previous recording must survive setup failure")
    service = services(cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    with monkeypatch.context() as fault:
        fault.setattr(context, "Pipe", allocate_pipe)
        fault.setattr(context, "Process", allocate_process)
        fault.setattr(module.tempfile, "mkdtemp", allocate_temp)
        if failure_at == "worker_construct":
            fault.setattr(module, "_Worker", fail_worker)
        session = service.start(request(tmp_path, purpose=purpose, channels=(0,)), events.callbacks)
        failure = events.failed.get(timeout=5)
        assert failure.stage == "service" and failure.message == message
        assert session.released.wait(2), "setup failed before any child could own the path"
        assert not service.busy and service.worker_pid is None
        assert not service.is_path_leased(session.request.path)
        assert len(endpoints) == expected_endpoints and all(endpoint.closed for endpoint in endpoints)
        for process in processes:
            with pytest.raises(ValueError, match="closed"):
                process.is_alive()
        assert all(not path.exists() for path in temporary_dirs)
        assert previous.read_bytes() == b"previous recording must survive setup failure"
    next_events = Events()
    fresh = service.start(request(tmp_path, request_id="retry", purpose=purpose, channels=(0,)),
                          next_events.callbacks)
    next_events.results.get(timeout=10)
    fresh.accept_result()
    next_events.accepted.get(timeout=5)
    assert fresh.released.wait(5)
    callback = threading.Event()
    service.shutdown(callback.set)
    assert callback.wait(5) and service.closed.wait(5)
    assert events.failed.empty() and events.results.empty() and events.accepted.empty()
    assert next_events.failed.empty() and service.diagnostics == [message]


@pytest.mark.parametrize("failure_at", ["after_spawn", "after_start_command"])
def test_setup_failure_with_actual_child_waits_for_retirement_before_release(
    tmp_path, services, monkeypatch, failure_at,
):
    from multiprocessing.process import BaseProcess
    service = services(cancel_timeout=.2, terminate_timeout=.2)
    events = Events()
    released_pids = []
    callbacks = replace(events.callbacks, released=lambda session: released_pids.append(service.worker_pid))
    message = f"injected setup failure {failure_at}"
    with monkeypatch.context() as fault:
        if failure_at == "after_spawn":
            original_start = BaseProcess.start
            def launch_then_fail(process):
                original_start(process)
                raise RuntimeError(message)
            fault.setattr(BaseProcess, "start", launch_then_fail)
        else:
            original_command = service._command
            def send_then_fail(kind, session=None, payload=None):
                original_command(kind, session, payload)
                if kind == "start":
                    raise RuntimeError(message)
            fault.setattr(service, "_command", send_then_fail)
        session = service.start(request(tmp_path), callbacks)
        failure = events.failed.get(timeout=5)
        assert failure.message == message
        assert session.worker_pid is not None and session.generation == 1
        assert session.released.wait(5)
        eventually(lambda: released_pids == [None])
        assert not service.is_path_leased(session.request.path)
        assert service.worker_pid is None
    next_events = Events()
    fresh = service.start(request(tmp_path, request_id="retry"), next_events.callbacks)
    next_events.results.get(timeout=10)
    assert fresh.generation > session.generation and fresh.worker_pid != session.worker_pid
    fresh.accept_result()
    assert fresh.released.wait(5)
    assert events.failed.empty() and events.results.empty() and events.accepted.empty()
