import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import av
import numpy as np
import pytest

from base.video.config import VideoConfig
from base.video.runtime import VideoRuntime, usb_video_worker
from base.video.recording import RecordingSession
from base.video.models import Command, CommandKind, EventKind
from base.video.service import VideoService
from unit_test.video.test_recording import config_for, decoded
from unit_test.video.test_threaded_encoding_poc import synthetic_frame, verify_files
from base.video.recording import FrameEncoder, TimedVideoFile, MediaQueue


class SyntheticCapture:
    """Only the camera is synthetic; encoder, files, IPC and Qt paths are real."""

    def __init__(self, config, on_frame, on_connection):
        self.config, self.on_frame, self.on_connection = config, on_frame, on_connection
        self.last_progress = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        self.on_connection(True, "")
        index = 0
        while not self.stop_event.wait(0.05):
            index += 1
            self.last_progress = time.monotonic()
            if self.config.device_name == "disconnect" and index == 30:
                self.on_connection(False, "test unplug")
                self.stop_event.wait(.5)
                self.on_connection(True, "")
            pixels = np.full((90, 160, 3), index % 200, dtype=np.uint8)
            pixels[:, :] = (65, 115, 160)
            pixels[:, index % 155:index % 155 + 5] = (230, 230, 230)
            import cv2
            cv2.putText(pixels, f"TEST {index}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
            self.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), time.monotonic())


def synthetic_video_worker(channel, mailbox, generation, config):
    try:
        VideoRuntime(channel, mailbox, generation, config, capture_factory=SyntheticCapture).run()
    finally:
        channel.close()


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError("real video pipeline timed out")


@pytest.mark.parametrize("name", ["normal", "disconnect"])
def test_real_encoding_in_independent_process_and_manual_stop(tmp_path, name):
    service = VideoService(
        worker_target=synthetic_video_worker, worker_options=config_for(tmp_path, device_name=name),
        heartbeat_timeout=8, command_timeout=10, shutdown_timeout=10,
    )
    try:
        service.start()
        wait_for(lambda: service.status.connection == "ready")
        assert service.start_recording()
        wait_for(lambda: service.status.recording == "recording")
        assert decoded(next(tmp_path.glob("*/*/*.recording.mp4")))
        if name == "disconnect":
            wait_for(lambda: service.status.recording == "recovering")
            assert service.stop_recording()
            wait_for(lambda: service.status.recording == "interrupted")
        else:
            assert service.stop_recording()
            wait_for(lambda: service.status.recording == "completed")
        files = list(tmp_path.glob("*/*/*.mp4"))
        assert files
        assert all(decoded(path) for path in files)
        assert not list(tmp_path.rglob("*.json"))
        wait_for(lambda: service.latest_preview() is not None)
        time.sleep(.7)
        assert not service.status.record_intent
    finally:
        service.shutdown()
        assert service.wait_closed(12)
        assert not service.forced_termination


def test_storage_failure_does_not_stop_preview(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")
    service = VideoService(
        worker_target=synthetic_video_worker, worker_options=config_for(root), heartbeat_timeout=8,
    )
    try:
        service.start()
        wait_for(lambda: service.status.connection == "ready")
        service.start_recording()
        wait_for(lambda: service.status.recording == "failed")
        assert service.status.connection == "ready"
        preview = wait_for(service.latest_preview)
        wait_for(lambda: service.latest_preview(preview.sequence) is not None)
        assert root.read_text(encoding="utf-8") == "occupied"
    finally:
        service.shutdown()
        assert service.wait_closed(10)


@pytest.mark.parametrize(
    "width,height,capacity",
    [(1280, 720, 60), (1920, 1080, 60), (2560, 1440, 60), (3840, 2160, 26), (7680, 4320, 6)],
)
def test_frame_capacity_respects_60_frames_and_640_mib_budget(width, height, capacity):
    runtime = VideoRuntime(None, None, 1, VideoConfig(width=width, height=height), capture_factory=SyntheticCapture)
    assert runtime.frame_capacity == capacity
    assert runtime.frame_capacity * width * height * 3 <= 640 * 1024**2


def test_overflow_is_bounded_and_keeps_recording_with_log_only(tmp_path, caplog):
    class Mailbox:
        width, height = 320, 180

        def publish(self, rgb):
            return True

    runtime = VideoRuntime(None, Mailbox(), 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.on_connection(True, "")
    runtime._command(Command(CommandKind.START, 1, "c1", "s1"))
    runtime.accept_frames = True  # Model an already prepared writer that stops consuming.
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    assert runtime.frame_capacity == 60
    for index in range(60):
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), index)
    assert runtime.accept_frames
    assert runtime.work.qsize() <= 61
    for index in range(60, 100):
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), index)
    assert runtime.accept_frames
    assert not runtime.session_error
    assert runtime.work.qsize() == runtime.frame_capacity + 1
    assert runtime.raw_frames == 60
    assert runtime.admission.candidates == 100
    assert runtime.admission.dropped == 40
    events = []
    while not runtime.events.empty():
        events.append(runtime.events.get_nowait())
    assert not any(kind in {EventKind.STOPPING, EventKind.RECORDING_FAILED} for kind, _, _ in events)
    # Producer only takes an in-memory snapshot; the control thread emits it later.
    assert not any("Video diagnostic:" in record.message for record in caplog.records)
    runtime._report_diagnostics()
    reports = [record for record in caplog.records if "Video frame admission:" in record.message]
    assert len(reports) == 1
    assert "'dropped': 40" in reports[0].message
    assert "'accepted': 60" in reports[0].message
    runtime._report_diagnostics()
    assert len([r for r in caplog.records if "Video frame admission:" in r.message]) == 1


def test_burst_drains_all_accepted_frames_without_compressing_timestamps(tmp_path):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.preview_due = float("inf")  # No preview transport in this writer-only check.
    runtime.on_connection(True, "")
    runtime._command(Command(CommandKind.START, 1, "start", "burst"))
    runtime.accept_frames = True  # Deterministically queue a burst before the writer runs.
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    accepted_times = []
    for index in range(60):
        before = runtime.admission.dropped
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + index / 30)
        if runtime.admission.dropped == before:
            accepted_times.append(index / 30)
    assert runtime.accept_frames
    assert 0 < runtime.admission.dropped < 6
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy and not runtime.writer_busy and runtime.encoded.snapshot()["items"] == 0)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "burst"))
        wait_for(lambda: not runtime.session_id)
        assert not list(tmp_path.rglob("*.json"))
        frames = decoded(next(tmp_path.glob("video/*/*.mp4")))
        assert [t for t, _ in frames] == pytest.approx(accepted_times, abs=1 / 90000)
        assert runtime.raw_frames == runtime.raw_bytes == 0
        assert not any(kind == EventKind.RECORDING_FAILED for kind, _, _ in runtime.events.queue)
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        assert not runtime.encoder_thread.is_alive()
        runtime.writer.join(5)
        assert not runtime.writer.is_alive()


def test_blocked_writer_does_not_block_encoder(tmp_path, caplog):
    entered, release = threading.Event(), threading.Event()

    class BlockedSession(RecordingSession):
        def write(self, frame, stamp):
            if not self.summary["frames"]:
                with self.diagnostics.stage("file_write"):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test writer was not released")
            return super().write(frame, stamp)

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path),
                           capture_factory=SyntheticCapture, session_factory=BlockedSession)
    runtime.preview_due = float("inf")
    runtime.online = True
    runtime.writer.start()
    runtime.encoder_thread.start()
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "blocked"))
        wait_for(lambda: runtime.accept_frames)
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100)
        assert entered.wait(2)
        for index in range(60):
            runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + (index + 1) / 30)
            wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy)
        assert runtime.accept_frames
        assert runtime.encoded.snapshot()["items"] == 61
        time.sleep(.21)
        runtime._report_diagnostics()
        reports = [r.args for r in caplog.records if r.msg == "Video diagnostic: %s"]
        assert len(reports) == 1
        assert reports[0]["session"] == "blocked"
        assert reports[0]["active_stage"] == "file_write"
        assert reports[0]["compressed"]["items"] == 61
        runtime._command(Command(CommandKind.STOP, 1, "stop", "blocked"))
        assert runtime.session_id == "blocked"  # Not completed while the file owner is blocked.
        release.set()
        wait_for(lambda: not runtime.session_id)
        assert len(decoded(next(tmp_path.glob("video/*/*.mp4")))) == 61
        assert not any(e[0] == EventKind.RECORDING_FAILED for e in runtime.events.queue)
    finally:
        release.set()
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("outcome", ["record", "stop", "shutdown", "failure"])
def test_slow_prepare_does_not_queue_frames_or_restart_after_cancel(tmp_path, outcome):
    entered, release = threading.Event(), threading.Event()

    class SlowSession(RecordingSession):
        def prepare(self, parameters=None):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test preparation was not released")
            if outcome == "failure":
                raise OSError("编码器初始化失败")
            super().prepare(parameters)

    class Mailbox:
        width, height = 320, 180
        published = 0

        def publish(self, rgb):
            self.published += 1

    mailbox = Mailbox()
    runtime = VideoRuntime(
        None, mailbox, 1, config_for(tmp_path),
        capture_factory=SyntheticCapture, session_factory=SlowSession,
    )
    runtime.on_connection(True, "")
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "slow"))
        assert entered.wait(2)
        pixels = np.zeros((90, 160, 3), dtype=np.uint8)
        for i in range(runtime.frame_capacity * 3):
            runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + i / 30)
        assert mailbox.published > 0
        assert not runtime.accept_frames
        assert runtime.work.empty()
        assert not any(kind == EventKind.STOPPING for kind, _, _ in runtime.events.queue)
        if outcome in {"stop", "shutdown"}:
            kind = CommandKind.STOP if outcome == "stop" else CommandKind.SHUTDOWN
            runtime._command(Command(kind, 1, "cancel", "slow"))
        release.set()
        if outcome == "record":
            wait_for(lambda: runtime.accept_frames)
            for i in range(15):
                runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 200 + i / 10)
                wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy and not runtime.writer_busy and runtime.encoded.snapshot()["items"] == 0)
            runtime._command(Command(CommandKind.STOP, 1, "stop", "slow"))
        wait_for(lambda: not runtime.session_id)
        assert not runtime.accept_frames
        events = list(runtime.events.queue)
        assert any(kind == EventKind.STARTED for kind, _, _ in events) == (outcome == "record")
        assert any(kind == EventKind.RECORDING_FAILED for kind, _, _ in events) == (outcome == "failure")
        assert not list(tmp_path.rglob("*.json"))
        if outcome == "record":
            assert len(decoded(next(tmp_path.glob("*/*/*.mp4")))) == 15
        else:
            assert not list(tmp_path.glob("*/*/*.mp4"))
    finally:
        release.set()
        if not runtime.stopping:
            runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        assert not runtime.encoder_thread.is_alive()
        runtime.writer.join(5)
        assert not runtime.writer.is_alive()


def test_disabled_real_backend_never_creates_recording_root(tmp_path):
    root = tmp_path / "must-not-exist"
    service = VideoService(worker_target=usb_video_worker, worker_options=VideoConfig(recording_root=str(root)))
    service.start()
    assert service.wait_closed(8)
    assert service.status.connection == "closed"
    assert not root.exists()


def test_failed_session_queue_cannot_cancel_next_recording(tmp_path):
    finished = []
    sessions = {}

    class Session:
        directory = tmp_path

        def __init__(self, config, identity):
            self.identity = identity
            self.summary = {"frames": 0, "gaps": 0}
            sessions[identity] = self

        def prepare(self, parameters=None):
            pass

        def maintain(self):
            pass

        def write(self, frame, stamp):
            if self.identity == "old":
                raise PermissionError("原始写盘错误")
            self.summary["frames"] += 1
            return True

        def finish(self, error=""):
            finished.append((self.identity, error))
            if error:
                raise OSError("后续收尾错误")

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path),
                           capture_factory=SyntheticCapture, session_factory=Session)
    runtime.preview_due = float("inf")
    runtime.online = True
    runtime._command(Command(CommandKind.START, 1, "c1", "old"))
    runtime.accept_frames = True
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    # Queue a burst, then fail on the writer; later stale errors must not replace it.
    for i in range(runtime.frame_capacity):
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), i)
    original_post = runtime.post
    restart_requested = []

    def post(kind, identity="", detail=""):
        original_post(kind, identity, detail)
        if kind == EventKind.RECORDING_FAILED and not restart_requested:
            restart_requested.append(True)
            runtime._command(Command(CommandKind.START, 1, "c2", "new"))

    runtime.post = post
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        wait_for(lambda: "new" in sessions and not runtime.writer_busy and runtime.work.empty())
        assert runtime.session_id == "new"
        assert runtime.accept_frames
        failures = [e for e in runtime.events.queue if e[0] == EventKind.RECORDING_FAILED]
        assert failures == [(EventKind.RECORDING_FAILED, "old", "原始写盘错误")]
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100)
        wait_for(lambda: sessions["new"].summary["frames"] == 1)
        runtime.frame_slots.acquire(False)
        runtime.raw_frames += 1
        runtime.work.put_nowait(("frame", "old", (None, 101, 0)))
        runtime.work.put_nowait(("gap", "old", ("旧掉线", "")))
        runtime.work.put_nowait(("stop", "old", None))
        runtime.work.put_nowait(("fail", "old", "迟到旧错误"))
        wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy and not runtime.writer_busy and runtime.encoded.snapshot()["items"] == 0)
        assert runtime.session_id == "new" and runtime.accept_frames
        assert len([e for e in runtime.events.queue if e[0] == EventKind.RECORDING_FAILED]) == 1
        runtime._command(Command(CommandKind.STOP, 1, "c3", "new"))
        wait_for(lambda: not runtime.session_id)
        assert len([s for s, _ in finished if s == "old"]) == 1
        assert sessions["new"].summary["frames"] == 1
        # Discarded old frames must release every slot.
        for _ in range(runtime.frame_capacity):
            assert runtime.frame_slots.acquire(False)
        assert not runtime.frame_slots.acquire(False)
        for _ in range(runtime.frame_capacity):
            runtime.frame_slots.release()
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        assert not runtime.encoder_thread.is_alive()
        runtime.writer.join(5)
        assert not runtime.writer.is_alive()


def encoder_init_failure_worker(channel, mailbox, generation, config):
    import base.video.runtime as runtime_module

    original = runtime_module.FrameEncoder

    def fail_once(*args, **kwargs):
        runtime_module.FrameEncoder = original
        raise OSError("模拟编码器初始化失败")

    runtime_module.FrameEncoder = fail_once
    try:
        synthetic_video_worker(channel, mailbox, generation, config)
    finally:
        runtime_module.FrameEncoder = original


def test_encoder_init_failure_keeps_preview_and_allows_retry(tmp_path):
    service = VideoService(
        worker_target=encoder_init_failure_worker, worker_options=config_for(tmp_path),
        heartbeat_timeout=8, command_timeout=10, shutdown_timeout=10,
    )
    try:
        service.start()
        wait_for(lambda: service.status.connection == "ready")
        process_id = service.process_id
        assert service.start_recording()
        wait_for(lambda: service.status.recording == "failed", timeout=2)
        assert service.status.recording_detail == "模拟编码器初始化失败"
        assert service.status.connection == "ready"
        assert not service.status.record_intent
        assert not service.is_closed
        assert not list(tmp_path.rglob("*.mp4"))
        preview = wait_for(service.latest_preview)
        wait_for(lambda: service.latest_preview(preview.sequence) is not None)

        assert service.start_recording()
        wait_for(lambda: service.status.recording == "recording")
        assert service.stop_recording()
        wait_for(lambda: service.status.recording == "completed")
        assert service.process_id == process_id
        assert decoded(next(tmp_path.glob("*/*/*.mp4")))
    finally:
        service.shutdown()
        assert service.wait_closed(12)
        assert not service.forced_termination


def test_stale_init_failure_cannot_cancel_preparing_or_active_session(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def delayed_encoder(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release encoder initialization")
        return FrameEncoder(*args, **kwargs)

    monkeypatch.setattr("base.video.runtime.FrameEncoder", delayed_encoder)
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.preview_due, runtime.online = float("inf"), True
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "new"))
        assert entered.wait(2)
        # The current request exists, but its writer session has not been created.
        runtime.encoded.put_nowait(("fail", "old", "迟到的初始化错误"))
        wait_for(lambda: runtime.encoded.snapshot()["items"] == 0)
        assert runtime.session_id == "new" and runtime.start_pending
        assert not runtime.session_error

        release.set()
        wait_for(lambda: runtime.accept_frames)
        runtime.encoded.put_nowait(("fail", "old", "重复的旧初始化错误"))
        wait_for(lambda: runtime.encoded.snapshot()["items"] == 0)
        assert runtime.session_id == "new" and runtime.accept_frames
        assert not any(e[0] == EventKind.RECORDING_FAILED for e in runtime.events.queue)
        pixels = np.zeros((90, 160, 3), dtype=np.uint8)
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "new"))
        wait_for(lambda: any(e[:2] == (EventKind.COMPLETED, "new") for e in runtime.events.queue))
        assert decoded(next(tmp_path.glob("*/*/*.mp4")))
    finally:
        release.set()
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


def test_worker_startup_ignores_legacy_manifest_without_opening_camera(tmp_path, monkeypatch):
    directory = tmp_path / "video" / "2026-09-09_10-23-30"
    directory.mkdir(parents=True)
    manifest = directory / "session.json"
    manifest.write_bytes(b"unreadable legacy manifest")
    partial = directory / "2026-09-09_10-23-30_001.recording.mp4"
    partial.write_bytes(b"preserve incomplete media")
    calls = []

    class FakeRuntime:
        writer = SimpleNamespace(is_alive=lambda: False)
        encoder_thread = SimpleNamespace(is_alive=lambda: False)

        def __init__(self, *args):
            pass

        def run(self):
            calls.append("run")

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".json":
            raise PermissionError("No manifest access allowed")
        return original_open(path, *args, **kwargs)

    # No production log handler, application instance, or CapturePump in this test.
    monkeypatch.setitem(sys.modules, "base.log_manager", SimpleNamespace(
        LogManager=SimpleNamespace(set_log_handler=lambda _: None),
    ))
    monkeypatch.setattr("base.video.runtime.VideoRuntime", FakeRuntime)
    channel = SimpleNamespace(send=lambda event: calls.append(event.kind), close=lambda: calls.append("close"))
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        usb_video_worker(channel, None, 1, config_for(tmp_path))
    assert calls == ["run", "close"]
    assert manifest.read_bytes() == b"unreadable legacy manifest"
    assert partial.read_bytes() == b"preserve incomplete media"


@pytest.mark.parametrize("pause", [2, 5, 10])
def test_production_runtime_recovers_from_real_file_write_pause(tmp_path, monkeypatch, pause):
    config = replace(config_for(tmp_path), width=1280, height=720, fps_num=30,
                     target_bitrate_bps=4_000_000)
    runtime = VideoRuntime(None, None, 1, config, capture_factory=SyntheticCapture)
    runtime.preview_due, runtime.online = float("inf"), True
    stats = {"encoded": 0, "paused": False}
    encode, write = FrameEncoder.write, TimedVideoFile.write
    background = np.random.default_rng(17).integers(16, 236, (720, 1280, 3), dtype=np.uint8)

    def observed_encode(owner, frame, stamp):
        assert threading.current_thread() is runtime.encoder_thread
        result = encode(owner, frame, stamp)
        stats["encoded"] += 1
        return result

    def paused_write(owner, data):
        assert threading.current_thread() is runtime.writer
        if stats["encoded"] >= 15 and not stats["paused"]:
            stats["paused"] = True
            before = stats["encoded"]
            time.sleep(pause)
            stats["encoded_during_pause"] = stats["encoded"] - before
        return write(owner, data)

    monkeypatch.setattr(FrameEncoder, "write", observed_encode)
    monkeypatch.setattr(TimedVideoFile, "write", paused_write)
    runtime.writer.start()
    runtime.encoder_thread.start()
    frames = (pause + 3) * 30
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "write-pause"))
        wait_for(lambda: runtime.accept_frames)
        started = time.monotonic()
        for index in range(frames):
            time.sleep(max(0, started + index / 30 - time.monotonic()))
            runtime.on_frame(synthetic_frame(index, 1280, 720, background), 100 + index / 30)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "write-pause"))
        wait_for(lambda: not runtime.session_id, timeout=20)
        verify_files(sorted(tmp_path.glob("video/*/*.mp4")), frames)
        assert not any(e[0] == EventKind.RECORDING_FAILED for e in runtime.events.queue)
        assert stats["encoded"] == frames
        assert stats["encoded_during_pause"] >= pause * 30 * .7
        snapshot = runtime.encoded.snapshot()
        assert snapshot["peak_bytes"] <= 64 * 1024**2
        assert snapshot["bytes"] == snapshot["packets"] == runtime.raw_bytes == 0
        print(f"Production write pause={pause}s frames={frames}: {stats}, {snapshot}")
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_production_segments_and_reconnect_preserve_all_frames(tmp_path, codec):
    config = replace(config_for(tmp_path), width=320, height=180, fps_num=30, codec=codec)
    # Test-only shortened segment boundary; saved settings still require two hours.
    config = SimpleNamespace(**(vars(config) | {"segment_duration_seconds": 1}))
    runtime = VideoRuntime(None, None, 1, config, capture_factory=SyntheticCapture)
    runtime.preview_due, runtime.online = float("inf"), True
    runtime.writer.start()
    runtime.encoder_thread.start()
    background = np.random.default_rng(17).integers(16, 236, (180, 320, 3), dtype=np.uint8)
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "segments"))
        wait_for(lambda: runtime.accept_frames)
        for index in range(90):
            if index == 45:
                runtime.on_connection(False, "合成断线")
                runtime.on_connection(True, "")
            runtime.on_frame(synthetic_frame(index, 320, 180, background), 100 + index / 30)
            wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "segments"))
        wait_for(lambda: not runtime.session_id)
        files = sorted(tmp_path.glob("video/*/*.mp4"))
        assert len(files) == 4
        verify_files(files, 90)
        assert runtime.encoded.snapshot()["bytes"] == runtime.raw_bytes == 0
        assert not list(tmp_path.rglob("*.recording.mp4"))
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_runtime_uses_production_two_hour_boundary(tmp_path, codec):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path, codec=codec), capture_factory=SyntheticCapture)
    runtime.preview_due, runtime.online = float("inf"), True
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "two-hours"))
        wait_for(lambda: runtime.accept_frames)
        for stamp in (0, 7_199.9, 7_200, 7_200.1):
            pixels = np.zeros((90, 160, 3), dtype=np.uint8)
            runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), stamp)
            wait_for(lambda: runtime.raw_frames == 0 and runtime.encoded.snapshot()["items"] == 0)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "two-hours"))
        wait_for(lambda: any(e[:2] == (EventKind.COMPLETED, "two-hours") for e in runtime.events.queue))
        files = sorted(tmp_path.glob("*/*/*.mp4"))
        segments = [decoded(path) for path in files]
        assert [len(frames) for frames in segments] == [2, 2]
        assert [frames[0][0] for frames in segments] == [0, 0]
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


def test_impossible_single_batch_fails_and_releases_all_credits(tmp_path):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.encoded = MediaQueue(byte_limit=1)
    runtime.preview_due, runtime.online = float("inf"), True
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "too-small"))
        wait_for(lambda: runtime.accept_frames)
        pixels = np.zeros((90, 160, 3), dtype=np.uint8)
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100)
        wait_for(lambda: not runtime.session_id)
        failures = [e for e in runtime.events.queue if e[0] == EventKind.RECORDING_FAILED]
        assert len(failures) == 1 and "单批压缩数据超过缓冲上限" in failures[0][2]
        assert not runtime.accept_frames
        assert runtime.encoded.snapshot()["bytes"] == runtime.raw_bytes == 0
        assert not list(tmp_path.rglob("*.mp4"))
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


def test_progress_watchdog_distinguishes_native_stall_and_total_drain(tmp_path):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    now = time.monotonic()
    runtime.writer_busy = True
    runtime.writer_busy_since = now
    runtime._check_progress(now + 29)
    with pytest.raises(TimeoutError, match="封装或写盘"):
        runtime._check_progress(now + 31)
    runtime.writer_busy = False
    runtime.encoder_busy = True
    runtime.encoder_busy_since = now
    with pytest.raises(TimeoutError, match="编码"):
        runtime._check_progress(now + 31)
    runtime.encoder_busy = False
    runtime.session_id, runtime.drain_started = "draining", now
    with pytest.raises(TimeoutError, match="收尾超时"):
        runtime._check_progress(now + 121)


def test_compressed_budget_warning_does_not_require_raw_backlog(tmp_path, caplog):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.encoded = MediaQueue(byte_limit=100)
    runtime.encoded.put_nowait("test-packet", size=80, packets=1)
    runtime._report_diagnostics()
    reports = [r.args for r in caplog.records if r.msg == "Video diagnostic: %s"]
    assert len(reports) == 1
    assert reports[0]["queue_items"] == 0
    assert reports[0]["compressed"]["bytes"] == 80
    runtime._report_diagnostics()
    assert len([r for r in caplog.records if r.msg == "Video diagnostic: %s"]) == 1


@pytest.mark.parametrize("stage", ["encode", "write", "sync", "close", "rename"])
def test_production_failure_is_not_completed_and_next_session_works(tmp_path, monkeypatch, stage):
    from pathlib import Path
    import os

    write_hook = [None]
    original_write = TimedVideoFile.write

    def hooked_write(owner, data):
        callback = write_hook[0] or original_write
        return callback(owner, data)

    # PyAV caches this bound callback when opening the container.
    if stage == "write":
        monkeypatch.setattr(TimedVideoFile, "write", hooked_write)
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.preview_due, runtime.online = float("inf"), True
    runtime.writer.start()
    runtime.encoder_thread.start()
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    entered = threading.Event()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "bad"))
        wait_for(lambda: runtime.accept_frames)
        # First get valid data into the production file, then inject a real boundary failure.
        for i in range(15):
            runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + i / 10)
            wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy
                     and runtime.encoded.snapshot()["items"] == 0)
        with monkeypatch.context() as guard:
            owner, name = {
                "encode": (FrameEncoder, "write"), "write": (TimedVideoFile, "write"),
                "sync": (os, "fsync"), "close": (TimedVideoFile, "close"),
                "rename": (Path, "rename"),
            }[stage]
            original = original_write if stage == "write" else getattr(owner, name)

            def failed(*args, **kwargs):
                if not entered.is_set():
                    entered.set()
                    if stage == "close":
                        original(*args, **kwargs)  # Release the real test handle, still report failure.
                    raise OSError(f"模拟{stage}失败")
                return original(*args, **kwargs)

            if stage == "write":
                write_hook[0] = failed
            else:
                guard.setattr(owner, name, failed)
            if stage in {"encode", "write"}:
                for i in range(15, 30):
                    runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + i / 10)
                    wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy)
            runtime._command(Command(CommandKind.STOP, 1, "stop", "bad"))
            wait_for(lambda: not runtime.session_id)
            assert entered.is_set()
            failures = [e for e in runtime.events.queue if e[0] == EventKind.RECORDING_FAILED]
            assert len(failures) == 1 and f"模拟{stage}失败" in failures[0][2]
            assert not any(e[0] == EventKind.COMPLETED for e in runtime.events.queue)
        write_hook[0] = None
        # No stale compressed packets/errors can complete or cancel this new recording.
        runtime._command(Command(CommandKind.START, 1, "restart", "good"))
        wait_for(lambda: runtime.accept_frames)
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 200)
        runtime._command(Command(CommandKind.STOP, 1, "stop-good", "good"))
        wait_for(lambda: not runtime.session_id)
        assert any(e[:2] == (EventKind.COMPLETED, "good") for e in runtime.events.queue)
        assert runtime.encoded.snapshot()["items"] == runtime.raw_bytes == 0
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
        assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


def permanently_blocked_video_worker(channel, mailbox, generation, config):
    """Exercise the real child teardown, using no device or production logging."""
    import os
    import base.video.runtime as runtime_module
    from base.video.recording import RecordingRootLease

    class BlockedSession(RecordingSession):
        def write(self, frame, stamp):
            threading.Event().wait()  # Only this disposable test child is blocked.

    class BlockedRuntime(VideoRuntime):
        def __init__(self, *args):
            super().__init__(*args, capture_factory=SyntheticCapture, session_factory=BlockedSession)
            self.no_progress_timeout = .3

    real_exit = os._exit

    def exit_with_lease_check(code):
        try:
            with RecordingRootLease(config.recording_root):
                real_exit(99)  # Bug: directory ownership was released with a live writer.
        except OSError:
            real_exit(code)

    runtime_module.VideoRuntime = BlockedRuntime
    os._exit = exit_with_lease_check
    sys.modules["base.log_manager"] = SimpleNamespace(
        LogManager=SimpleNamespace(set_log_handler=lambda _: None),
    )
    usb_video_worker(channel, mailbox, generation, config)


def test_stuck_file_owner_exits_before_releasing_directory_lease(tmp_path):
    from base.video.recording import RecordingRootLease

    service = VideoService(
        worker_target=permanently_blocked_video_worker, worker_options=config_for(tmp_path),
        heartbeat_timeout=8, command_timeout=10, shutdown_timeout=10, drain_timeout=5,
    )
    try:
        service.start()
        wait_for(lambda: service.status.connection == "ready")
        assert service.start_recording()
        assert service.wait_closed(10)
        assert service.status.recording == "failed"
        assert "exitcode=24" in service.status.connection_detail
        assert not service.forced_termination  # Child self-terminates while still holding its lease.
        assert list(tmp_path.rglob("*.recording.mp4"))
        assert not [p for p in tmp_path.rglob("*.mp4") if ".recording." not in p.name]
        with RecordingRootLease(tmp_path):
            pass  # OS releases the lease only after the worker has exited.
    finally:
        service.shutdown()
        assert service.wait_closed(10)
