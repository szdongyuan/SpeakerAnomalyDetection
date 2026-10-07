"""Real worker/capture lifecycle with controlled native boundaries, never hardware."""
import gc
import queue
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from base import recording_capture, recording_worker
from base.recording_process_protocol import RecordingEvent, RecordingFailure
from base.recording_worker_pipeline import WorkerCaptureState
from unit_test.base.recording_process_fakes import FakeBackend, FakeStream, known_audio
from unit_test.base.test_recording_capture import request
from unit_test.base.ve3668n_fakes import CaptureSDK, capture_request
from unit_test.base.test_ve3668n_resource import prewarm_request


class HardExit(BaseException):
    pass


class Connection:
    def __init__(self):
        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.pending = None
        self.sent = []

    def poll(self, timeout):
        try:
            self.pending = self.commands.get(timeout=timeout)
            return True
        except queue.Empty:
            return False

    def recv(self):
        return self.pending

    def send(self, event):
        self.sent.append(event)
        self.events.put(event)

    def close(self):
        pass


class Harness:
    def __init__(self, monkeypatch, *, injected=False, load_fails=False,
                 terminate_fails=False, close_fails=False, hold_thread=False, ve=False,
                 initialize_fails=False, retained_audio=None):
        self.trace = []
        self.control = Connection()
        self.preview = Connection()
        self.errors = []
        self.captures = []
        self.release_thread = threading.Event()
        self.thread_done = threading.Event()
        harness = self

        pipeline_type = recording_worker.WorkerCapturePipeline

        class Pipeline(pipeline_type):
            def __init__(self):
                super().__init__()
                harness.pipeline = self
                if retained_audio is not None:
                    self.finalizers["previous"] = WorkerCaptureState(
                        "previous", retained_audio, finalizing=True)

        monkeypatch.setattr(recording_worker, "WorkerCapturePipeline", Pipeline)

        def trace(kind):
            harness.trace.append((kind, threading.get_ident()))

        class Stream(FakeStream):
            def start(self):
                trace("stream_started")
                super().start()

            def stop(self):
                trace("stream_stopped")
                super().stop()

            def close(self):
                if close_fails:
                    raise OSError("native close failed")
                super().close()
                trace("stream_closed")

        class Backend(FakeBackend):
            _initialized = 0

            def query_devices(self, index=None):
                trace("device_queried")
                if self.inventory is not None:
                    if index is None:
                        return self.inventory
                    return next(device for device in self.inventory if device["index"] == index)
                return super().query_devices(index)

            def query_hostapis(self, index=None):
                return self.apis if index is None else self.apis[index]

            def _initialize(self):
                trace("library_initialized")
                if initialize_fails:
                    raise RuntimeError("native reinitialize failed")
                self._initialized += 1
                self.device = self.current_device.copy()
                if self.fresh_inventory is not None:
                    self.inventory = self.fresh_inventory
                    self.apis = self.fresh_apis

            def InputStream(self, **config):
                trace("stream_opened")
                if self.fail_open:
                    self.fail_open -= 1
                    raise OSError("device open failed")
                self.stream = Stream(self, **config)
                return self.stream

            def _terminate(self):
                assert retained_audio is None or retained_audio.stream_closed
                for ref in harness.captures:
                    capture = ref()
                    assert capture is None or capture._thread is None or not capture._thread.is_alive()
                trace("library_terminated")
                if terminate_fails:
                    raise RuntimeError("native terminate failed")
                self._initialized -= 1

            def exit_handler(self):
                while self._initialized:
                    self._terminate()

        self.backend = Backend()
        self.backend.current_device = self.backend.device.copy()
        self.backend.fail_open = False
        self.backend.inventory = self.backend.fresh_inventory = None
        self.backend.apis = [{"name": "unused"}, {"name": "Selected API"}]
        self.backend.fresh_apis = None

        def load():
            trace("library_loaded")
            if load_fails:
                raise RuntimeError("native initialize failed")
            harness.backend._initialized += 1
            return harness.backend

        class Capture(recording_capture.RecordingCapture):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                harness.captures.append(weakref.ref(self))

            def _run(self):
                super()._run()
                harness.thread_done.set()
                if hold_thread:
                    assert harness.release_thread.wait(5)
                trace("capture_thread_exited")

        monkeypatch.setattr(recording_capture, "sounddevice_backend", load)
        monkeypatch.setattr(recording_worker, "sounddevice_backend", load, raising=False)
        monkeypatch.setattr(recording_worker, "RecordingCapture", Capture)
        monkeypatch.setattr(recording_worker.multiprocessing, "parent_process", lambda: None)
        if ve:
            self.sdk = CaptureSDK()
            monkeypatch.setattr(recording_worker, "VkDaqClient", lambda: self.sdk)

        def hard_exit(code):
            raise HardExit(code)

        monkeypatch.setattr(recording_worker, "exit_with_log_drain", hard_exit)
        factory = None
        if injected:
            factory = "lifetime_test:dependencies"
            monkeypatch.setattr(recording_worker.importlib, "import_module",
                                lambda name: SimpleNamespace(
                                    dependencies=lambda **kw: {"backend": self.backend}))

        def run():
            try:
                recording_worker.recording_worker(
                    self.control, self.preview, 1, factory, {}, .2, .05)
            except BaseException as exc:
                self.errors.append(exc)

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        self.receive("ready")

    def command(self, kind, req=None, payload=None):
        self.control.commands.put(RecordingEvent(
            1, "" if req is None else req.request_id, kind,
            req if kind == "start" else payload))

    def receive(self, kind, timeout=3):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            try:
                event = self.control.events.get(timeout=.05)
            except queue.Empty:
                continue
            seen.append(event.kind)
            if event.kind == kind:
                return event
        pytest.fail(f"missing {kind}; received {seen}; worker errors {self.errors}")

    def start(self, req):
        self.command("start", req)
        self.receive("started")

    def complete(self, req):
        self.start(req)
        self.backend.stream.feed(known_audio())
        self.receive("completed")
        self.command("result_ack", req, "accepted")

    def shutdown(self):
        self.command("shutdown")
        self.worker.join(3)
        assert not self.worker.is_alive()

    def names(self):
        return [kind for kind, _thread in self.trace]


@pytest.fixture
def harness(monkeypatch):
    active = []

    def make(**kwargs):
        item = Harness(monkeypatch, **kwargs)
        active.append(item)
        return item

    yield make
    for item in active:
        item.release_thread.set()
        if item.worker.is_alive():
            item.shutdown()
        for ref in item.captures:
            capture = ref()
            if capture is not None and capture._thread is not None:
                capture._thread.join(3)
                assert not capture._thread.is_alive()


def test_worker_owns_one_library_across_collected_recordings(tmp_path, harness):
    worker = harness()
    for number in range(2):
        req = request(tmp_path, request_id=f"record-{number}", purpose="calibration", channels=(0,),
                      streaming=False, path=str(tmp_path / f"{number}.wav"))
        worker.complete(req)
        gc.collect()
        assert worker.names().count("library_terminated") == 0
        if number:
            assert worker.captures[0]() is None
    worker.shutdown()
    assert not worker.errors
    assert worker.names().count("library_loaded") == 1
    assert worker.names().count("library_terminated") == 1
    for kind, thread in worker.trace:
        if kind != "device_queried":
            assert (thread == worker.worker.ident) == kind.startswith("library_")
    names = worker.names()
    assert names.index("stream_closed") < names.index("capture_thread_exited")
    assert max(i for i, name in enumerate(names) if name == "capture_thread_exited") < names.index("library_terminated")
    worker.backend.exit_handler()
    assert worker.names().count("library_terminated") == 1


@pytest.mark.parametrize("old_name", ["different microphone", "fake recording devic",
                                     "fake recording device"])
def test_rebind_selected_identity_after_indices_change(tmp_path, harness, old_name):
    worker = harness()
    selected = {**worker.backend.device, "hostapi": 1, "hostapi_name": "Selected API"}
    original = selected.copy()
    old_slot = {**worker.backend.device, "hostapi": 1, "name": old_name}
    moved = {**worker.backend.device, "index": 8, "hostapi": 0}
    worker.backend.inventory = [old_slot]
    worker.backend.apis = [{"name": "Selected API"}, {"name": "Other API"}]
    worker.backend.fresh_inventory = [old_slot, moved]
    worker.backend.fresh_apis = [{"name": "Selected API"}, {"name": "Other API"}]
    req = request(tmp_path, device=selected)
    worker.start(req)
    assert worker.backend.stream.config["device"] == 8
    assert dict(req.device) == original
    assert selected == original
    worker.backend.stream.feed(known_audio())
    result = worker.receive("completed").payload
    assert (result.request_id, result.path, result.channels) == (req.request_id, req.path, req.channels)
    assert worker.names().count("library_initialized") == 1
    worker.command("result_ack", req, "accepted")
    worker.shutdown()
    assert not worker.errors


def test_matching_stable_input_uses_fast_path_without_refresh(tmp_path, harness):
    worker = harness()
    worker.backend.apis = [{"name": "Selected API"}]
    selected = {**worker.backend.device, "hostapi_name": "Selected API"}
    worker.complete(request(tmp_path, device=selected))
    assert "library_initialized" not in worker.names()
    assert "library_terminated" not in worker.names()
    worker.shutdown()
    assert not worker.errors


@pytest.mark.parametrize("problem,reason", [
    ("missing", "missing"), ("prefix", "missing"), ("wrong_api", "missing"),
    ("duplicate", "ambiguous"), ("capacity", "channels"),
])
def test_rebind_fails_closed_with_selected_identity_diagnostics(tmp_path, harness, problem, reason):
    worker = harness()
    selected = {**worker.backend.device, "hostapi_name": "Selected API"}
    old_slot = {**worker.backend.device, "name": "other input"}
    exact = {**worker.backend.device, "index": 8, "hostapi": 1}
    candidates = [exact]
    if problem == "missing":
        candidates = []
    elif problem == "prefix":
        exact["name"] = selected["name"][:-1]
    elif problem == "wrong_api":
        exact["hostapi"] = 0
    elif problem == "duplicate":
        # Even an exact old numeric slot must not hide another duplicate.
        candidates.append({**exact, "index": selected["index"]})
        old_slot = None
    elif problem == "capacity":
        exact["max_input_channels"] = 1
    worker.backend.inventory = [{**worker.backend.device, "name": "stale input"}]
    worker.backend.fresh_inventory = ([] if old_slot is None else [old_slot]) + candidates
    worker.backend.fresh_apis = [{"name": "Other API"}, {"name": "Selected API"}]
    worker.command("start", request(tmp_path, device=selected))
    failure = worker.receive("failed").payload
    assert failure.stage == "device"
    assert selected["name"] in failure.message and "Selected API" in failure.message
    assert reason in failure.message and "reselect" in failure.message
    assert worker.names().count("library_initialized") == 1
    assert "stream_opened" not in worker.names()
    assert not (tmp_path / "recording.wav").exists()
    worker.shutdown()
    assert not worker.errors


@pytest.mark.parametrize("stale", [dict(name="previous microphone"), dict(hostapi=4),
                                   dict(index=8), dict(max_input_channels=0)])
def test_stale_inventory_refreshes_once_before_selected_device_opens(tmp_path, harness, stale):
    worker = harness()
    worker.backend.device.update(stale)
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    worker.start(req)
    assert worker.names().count("library_terminated") == 1
    assert worker.names().count("library_initialized") == 1
    assert worker.backend.stream.config["device"] == req.device["index"]
    assert worker.backend.stream.config["channels"] == 1
    worker.backend.stream.feed(known_audio())
    worker.receive("completed")
    worker.command("result_ack", req, "accepted")
    worker.complete(request(tmp_path, request_id="again", purpose="calibration",
                            channels=(0,), streaming=False))
    assert worker.names().count("library_initialized") == 1
    worker.shutdown()
    assert not worker.errors
    assert all(thread == worker.worker.ident for name, thread in worker.trace
               if name.startswith("library_"))


@pytest.mark.parametrize("failure", ["terminate", "initialize"])
@pytest.mark.parametrize("pending_result", [False, True])
def test_refresh_failure_retires_and_releases_request_once(tmp_path, harness, failure, pending_result):
    worker = harness(**{f"{failure}_fails": True})
    if pending_result:
        previous = request(tmp_path, request_id="previous", purpose="calibration",
                           channels=(0,), streaming=False, path=str(tmp_path / "previous.wav"))
        worker.start(previous)
        worker.backend.stream.feed(known_audio())
        worker.receive("completed")
    worker.backend.device["name"] = "old microphone"
    req = request(tmp_path)
    worker.command("start", req)
    worker.worker.join(3)
    assert not worker.worker.is_alive()
    failures = [event for event in worker.control.sent if event.kind == "failed"]
    assert len(failures) == 1
    assert failures[0].payload.stage == "device"
    assert failures[0].payload.handles_released
    assert sum(event.kind == "worker_fatal" for event in worker.control.sent) == 1
    kinds = [event.kind for event in worker.control.sent]
    assert kinds.index("failed") < kinds.index("worker_fatal")
    assert worker.pipeline.empty
    assert worker.names().count("library_terminated") == 1
    assert worker.names().count("library_initialized") == (failure == "initialize")
    assert worker.names().count("stream_opened") == int(pending_result)
    assert not (tmp_path / "recording.wav").exists()
    assert len(worker.errors) == 1 and isinstance(worker.errors[0], HardExit)


def test_prior_cleanup_failure_blocks_refresh_after_result_ack(tmp_path, harness):
    worker = harness(close_fails=True)
    first = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    worker.start(first)
    worker.backend.stream.feed(known_audio())
    assert not worker.receive("failed").payload.handles_released
    worker.command("result_ack", first, "rejected")
    worker.backend.device["name"] = "old microphone"
    second = request(tmp_path, request_id="second", path=str(tmp_path / "second.wav"))
    worker.command("start", second)
    failure = worker.receive("failed").payload
    assert failure.stage == "device"
    assert "ownership" in failure.message
    assert "library_initialized" not in worker.names()
    assert "library_terminated" not in worker.names()
    assert not (tmp_path / "second.wav").exists()


@pytest.mark.parametrize("changed,message", [
    (dict(name="other microphone"), "identity changed"),
    (dict(hostapi=2), "identity changed"),
    (dict(index=8), "unknown fake device"),
    (dict(max_input_channels=1), "no longer supports selected channels"),
])
def test_persistent_inventory_problem_fails_closed(tmp_path, harness, changed, message):
    worker = harness()
    worker.backend.device.update(changed)
    worker.backend.current_device.update(changed)
    worker.command("start", request(tmp_path))
    failure = worker.receive("failed").payload
    assert failure.stage == "device" and message in failure.message
    assert worker.names().count("library_initialized") == 1
    assert "stream_opened" not in worker.names()
    assert not (tmp_path / "recording.wav").exists()
    worker.shutdown()
    assert not worker.errors


@pytest.mark.parametrize("duplicate,completed", [(False, False), (True, False), (True, True)])
def test_invalid_start_never_queries_or_refreshes(tmp_path, harness, duplicate, completed):
    worker = harness()
    first = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    worker.start(first)
    if completed:
        worker.backend.stream.feed(known_audio())
        worker.receive("completed")
    before_queries = worker.names().count("device_queried")
    worker.backend.device["name"] = "stale microphone"
    second = first if duplicate else request(tmp_path, request_id="overlap")
    worker.command("start", second)
    assert worker.receive("worker_fatal").payload.stage == "protocol/start"
    worker.worker.join(3)
    assert not worker.worker.is_alive()
    assert worker.names().count("device_queried") == before_queries
    assert "library_initialized" not in worker.names()
    assert worker.names().count("library_terminated") == 1  # shutdown only
    assert not worker.errors


def test_ordinary_start_during_ve_prewarm_never_touches_audio(tmp_path, harness):
    import itertools

    worker = harness(ve=True)
    worker.sdk.counts = itertools.repeat(0)
    warmup = prewarm_request()
    worker.control.commands.put(RecordingEvent(1, warmup.warmup_id, "prewarm_ve", warmup))
    worker.receive("ve_prewarm_started")
    worker.command("start", request(tmp_path))
    assert worker.receive("worker_fatal").payload.stage == "protocol/start"
    worker.worker.join(3)
    assert not worker.worker.is_alive()
    assert not any(name.startswith("library_") or name == "device_queried"
                   for name in worker.names())
    assert not worker.errors


@pytest.mark.parametrize("ownership", ["closed", "open", "uncertain"])
def test_retained_audio_finalizer_controls_refresh_safety(tmp_path, harness, ownership):
    # Ordinary captures currently retain their slot through finalization. Model
    # the pipeline's retained-finalizer contract without creating a native handle.
    class RetainedAudio:
        request = SimpleNamespace(device={})
        done = threading.Event()
        stream_closed = ownership == "closed"
        outcome = RecordingFailure("previous", "cancelled", "previous.wav", "test cleanup")

        def cancel(self):
            self.done.set()

        def join(self, timeout=0):
            return self.done.is_set()

    worker = harness(retained_audio=RetainedAudio())
    worker.backend.device["name"] = "stale microphone"
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    if ownership == "closed":
        worker.complete(req)
        assert worker.names().count("library_initialized") == 1
    else:
        worker.command("start", req)
        assert "ownership" in worker.receive("failed").payload.message
        assert not any(name in ("device_queried", "library_initialized", "library_terminated")
                       for name in worker.names())
        assert not (tmp_path / "recording.wav").exists()
    worker.shutdown()
    assert bool(worker.errors) == (ownership != "closed")


@pytest.mark.parametrize("first", ["cancel", "open_failure"])
def test_library_survives_failed_or_cancelled_request(tmp_path, harness, first):
    worker = harness()
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    if first == "cancel":
        worker.start(req)
        worker.command("cancel", req)
        worker.receive("cancelled")
    else:
        worker.backend.fail_open = 2
        worker.command("start", req)
        assert worker.receive("failed").payload.stage == "device"
    worker.command("result_ack", req, "rejected")
    assert worker.names().count("library_terminated") == int(first == "open_failure")
    worker.complete(request(tmp_path, request_id="retry", purpose="calibration", channels=(0,), streaming=False))
    worker.shutdown()
    assert not worker.errors
    assert worker.names().count("library_loaded") == 1
    assert worker.names().count("library_terminated") == 1 + int(first == "open_failure")


def test_initialization_failure_keeps_identity_and_creates_no_wav(tmp_path, harness):
    worker = harness(load_fails=True)
    req = request(tmp_path)
    worker.command("start", req)
    failure = worker.receive("failed").payload
    assert (failure.request_id, failure.path, failure.stage) == (req.request_id, req.path, "device")
    assert failure.message == "native initialize failed"
    assert not (tmp_path / "recording.wav").exists()
    worker.shutdown()
    assert worker.trace == [("library_loaded", worker.worker.ident)]
    assert not worker.errors


def test_termination_exception_is_logged_and_exits_nonzero(tmp_path, harness, caplog):
    worker = harness(terminate_fails=True)
    worker.complete(request(tmp_path, purpose="calibration", channels=(0,), streaming=False))
    worker.shutdown()
    assert len(worker.errors) == 1 and isinstance(worker.errors[0], HardExit)
    assert worker.errors[0].args == (1,)
    assert any(record.exc_info and "terminate" in record.getMessage().lower()
               for record in caplog.records)


@pytest.mark.parametrize("release", [False, True])
def test_done_does_not_allow_termination_beneath_live_thread(tmp_path, harness, release):
    worker = harness(hold_thread=True)
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    worker.start(req)
    worker.backend.stream.feed(known_audio())
    assert worker.thread_done.wait(3)
    if release:
        worker.release_thread.set()
        worker.receive("completed")
    worker.shutdown()
    if release:
        assert not worker.errors
        assert worker.names().index("capture_thread_exited") < worker.names().index("library_terminated")
    else:
        assert len(worker.errors) == 1 and isinstance(worker.errors[0], HardExit)
        assert "library_terminated" not in worker.names()


def test_uncertain_stream_close_forbids_library_termination(tmp_path, harness):
    worker = harness(close_fails=True)
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    worker.start(req)
    worker.backend.stream.feed(known_audio())
    assert not worker.receive("failed").payload.handles_released
    worker.shutdown()
    assert len(worker.errors) == 1 and isinstance(worker.errors[0], HardExit)
    assert "library_terminated" not in worker.names()


@pytest.mark.parametrize("injected", [False, True])
def test_idle_and_injected_backend_never_load_real_library(tmp_path, harness, injected):
    worker = harness(injected=injected)
    if injected:
        worker.complete(request(tmp_path, purpose="calibration", channels=(0,), streaming=False))
    worker.shutdown()
    assert not worker.errors
    assert not any(name.startswith("library_") for name in worker.names())
    assert not any(name == "device_queried" and thread == worker.worker.ident
                   for name, thread in worker.trace)


def test_ve_prewarm_and_recording_never_load_real_library(tmp_path, harness):
    worker = harness(ve=True)
    warmup = prewarm_request()
    worker.control.commands.put(RecordingEvent(1, warmup.warmup_id, "prewarm_ve", warmup))
    assert worker.receive("ve_prewarm_terminal").payload.success
    req = capture_request(tmp_path / "ve.wav", sample_rate=44100)
    worker.command("start", req)
    worker.receive("completed")
    worker.shutdown()
    assert not worker.errors
    assert worker.sdk.calls("create_task") == 1
    assert worker.sdk.closed
    assert not any(name.startswith("library_") for name in worker.names())
    assert "device_queried" not in worker.names()
