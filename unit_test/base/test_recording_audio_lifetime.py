"""Real worker/capture lifecycle with controlled native boundaries, never hardware."""
import gc
import queue
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from base import recording_capture, recording_worker
from base.recording_process_protocol import RecordingEvent
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

    def poll(self, timeout):
        try:
            self.pending = self.commands.get(timeout=timeout)
            return True
        except queue.Empty:
            return False

    def recv(self):
        return self.pending

    def send(self, event):
        self.events.put(event)

    def close(self):
        pass


class Harness:
    def __init__(self, monkeypatch, *, injected=False, load_fails=False,
                 terminate_fails=False, close_fails=False, hold_thread=False, ve=False):
        self.trace = []
        self.control = Connection()
        self.preview = Connection()
        self.errors = []
        self.captures = []
        self.release_thread = threading.Event()
        self.thread_done = threading.Event()
        harness = self

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

            def InputStream(self, **config):
                trace("stream_opened")
                if self.fail_open:
                    self.fail_open = False
                    raise OSError("device open failed")
                self.stream = Stream(self, **config)
                return self.stream

            def _terminate(self):
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
        self.backend.fail_open = False

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

        monkeypatch.setattr(recording_worker.os, "_exit", hard_exit)
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
        assert (thread == worker.worker.ident) == kind.startswith("library_")
    names = worker.names()
    assert names.index("stream_closed") < names.index("capture_thread_exited")
    assert max(i for i, name in enumerate(names) if name == "capture_thread_exited") < names.index("library_terminated")
    worker.backend.exit_handler()
    assert worker.names().count("library_terminated") == 1


@pytest.mark.parametrize("first", ["cancel", "open_failure"])
def test_library_survives_failed_or_cancelled_request(tmp_path, harness, first):
    worker = harness()
    req = request(tmp_path, purpose="calibration", channels=(0,), streaming=False)
    if first == "cancel":
        worker.start(req)
        worker.command("cancel", req)
        worker.receive("cancelled")
    else:
        worker.backend.fail_open = True
        worker.command("start", req)
        assert worker.receive("failed").payload.stage == "device"
    worker.command("result_ack", req, "rejected")
    assert "library_terminated" not in worker.names()
    worker.complete(request(tmp_path, request_id="retry", purpose="calibration", channels=(0,), streaming=False))
    worker.shutdown()
    assert not worker.errors
    assert worker.names().count("library_loaded") == 1
    assert worker.names().count("library_terminated") == 1


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
