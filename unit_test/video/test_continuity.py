"""Continuous recording protection; no camera or production output is used."""

import threading
import time
import logging
from dataclasses import replace
from types import SimpleNamespace

import av
import numpy as np
import pytest

from base.video.models import Command, CommandKind, EventKind
from base.video.recording import (
    MediaQueue, MediaQueueCancelled, MediaQueueClosed, RecordingSession,
    PENDING_MEDIA_BYTE_LIMIT,
)
from base.video.runtime import FrameAdmission, VideoRuntime
from unit_test.video.test_recording import config_for, decoded
from unit_test.video.test_runtime import SyntheticCapture, wait_for
from unit_test.video.test_threaded_encoding_poc import synthetic_frame


def test_shedding_is_spread_hysteretic_and_recovers():
    admission = FrameAdmission("s1")
    decisions = []
    for i in range(12):
        skip = admission.select(.8, 0, i / 30)
        admission.record(skip, i / 30, i / 30, "raw")
        decisions.append(skip)
    assert decisions == [False] * 5 + [True] + [False] * 5 + [True]
    assert admission.level == 1
    admission.select(.9, 0, 1)
    assert admission.level == 2
    admission.select(.6, 0, 2)
    assert admission.level == 2
    admission.select(.1, .1, 3)
    admission.select(.1, .6, 4)
    admission.select(.1, .1, 5)
    assert admission.level == 2
    admission.select(.1, .1, 7)
    assert admission.level == 0


def test_full_compressed_queue_waits_with_one_bounded_pending_batch():
    channel = MediaQueue(byte_limit=10, packet_limit=2)
    channel.put_nowait("first", size=10, packets=1)
    active = channel.get()
    finished = threading.Event()

    def send():
        channel.put("second", size=4, packets=1)
        finished.set()

    thread = threading.Thread(target=send)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not channel.snapshot()["pending_bytes"]:
            time.sleep(.01)
        snapshot = channel.snapshot()
        assert snapshot["bytes"] == 10
        assert snapshot["pending_bytes"] == 4
        assert snapshot["resident_bytes"] == 14
        assert not finished.is_set()
        channel.release(active)
        assert finished.wait(2)
        second = channel.get()
        assert second.message == "second"
        channel.release(second)
        assert channel.snapshot()["resident_bytes"] == 0
    finally:
        channel.close(discard=True)
        thread.join(2)
    assert not thread.is_alive()


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_producer_wait_is_woken_by_failure_or_close(action):
    channel = MediaQueue(byte_limit=10)
    channel.put_nowait("first", size=10)
    cancelled = threading.Event()
    errors = []

    def send():
        try:
            channel.put("second", size=5, cancelled=cancelled)
        except (MediaQueueClosed, MediaQueueCancelled) as exc:
            errors.append(exc)

    thread = threading.Thread(target=send)
    thread.start()
    try:
        wait_for(lambda: channel.snapshot()["pending_bytes"] == 5)
        if action == "cancel":
            cancelled.set()
            channel.wake()
        else:
            channel.close()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
        expected = MediaQueueCancelled if action == "cancel" else MediaQueueClosed
        assert isinstance(errors[0], expected)
        assert channel.snapshot()["pending_bytes"] == 0
        assert channel.snapshot()["bytes"] == 10
    finally:
        channel.close(discard=True)
        thread.join(2)


@pytest.mark.parametrize("size,packets", [(11, 1), (1, 3), (-1, 1)])
def test_impossible_wait_is_rejected_without_retaining_a_batch(size, packets):
    channel = MediaQueue(byte_limit=10, packet_limit=2)
    with pytest.raises(ValueError):
        channel.put("impossible", size=size, packets=packets)
    assert channel.snapshot()["resident_bytes"] == 0


def test_production_pending_batch_has_a_separate_hard_limit():
    channel = MediaQueue()
    assert channel.pending_limit == PENDING_MEDIA_BYTE_LIMIT
    with pytest.raises(ValueError):
        channel.put("too-large", size=PENDING_MEDIA_BYTE_LIMIT + 1)
    assert channel.snapshot()["resident_bytes"] == 0


def test_packet_and_data_item_credits_block_even_when_bytes_are_available():
    for packets in (0, 2):
        channel = MediaQueue(byte_limit=100, packet_limit=2)
        channel.put_nowait("a", size=1, packets=packets)
        if not packets:
            channel.put_nowait("b", size=1)
        done = threading.Event()
        thread = threading.Thread(target=lambda: (channel.put("last", size=1, packets=1), done.set()))
        thread.start()
        try:
            wait_for(lambda: channel.snapshot()["pending_bytes"] == 1)
            assert not done.is_set()
            item = channel.get()
            assert not done.is_set()  # In-flight credits are still charged.
            channel.release(item)
            assert done.wait(2)
        finally:
            channel.close(discard=True)
            thread.join(2)


def test_logging_coalesces_recovery_and_final_report_is_once():
    admission = FrameAdmission("stats")
    for i in range(6):
        admission.record(admission.select(.8, 0, i / 30), i / 30, i / 30, "raw")
    first = admission.report(1)
    assert first["event"] == "started" and first["window_dropped"] == 1
    admission.select(0, 0, 2)
    admission.select(0, 0, 4)
    assert admission.level == 0
    assert admission.report(4) is None
    recovered = admission.report(31)
    assert recovered["event"] == "recovered"
    assert recovered["recoveries"] == 1 and recovered["window_dropped"] == 0
    assert admission.report(61) is None
    final = admission.report(62, final=True)
    assert final["event"] == "ended" and final["drop_ratio"] == pytest.approx(1 / 6)
    assert admission.report(63, final=True) is None


def test_byte_pressure_and_full_slots_do_not_cancel_recording(tmp_path, monkeypatch):
    import base.video.runtime as runtime_module

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime._command(Command(CommandKind.START, 1, "start", "bytes"))
    runtime.accept_frames = True
    frame = av.VideoFrame.from_ndarray(np.zeros((90, 160, 3), dtype=np.uint8), format="rgb24")
    monkeypatch.setattr(runtime_module, "RAW_FRAME_BYTE_LIMIT", 160 * 90 * 3)
    for i in range(100):
        runtime.on_frame(frame, i / 30)
    assert runtime.accept_frames and not runtime.session_error
    assert runtime.raw_frames == 1 and runtime.raw_bytes == 160 * 90 * 3
    assert runtime.admission.dropped == 99
    assert runtime.admission.candidates == 100
    assert not any(event[0] == EventKind.RECORDING_FAILED for event in runtime.events.queue)


def test_impossible_raw_frame_is_a_real_error_not_endless_shedding(tmp_path, monkeypatch):
    import base.video.runtime as runtime_module

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime._command(Command(CommandKind.START, 1, "start", "oversized"))
    runtime.accept_frames = True
    monkeypatch.setattr(runtime_module, "RAW_FRAME_BYTE_LIMIT", 1)
    frame = av.VideoFrame.from_ndarray(np.zeros((90, 160, 3), dtype=np.uint8), format="rgb24")
    runtime.on_frame(frame, 100)
    assert not runtime.accept_frames
    assert "单帧超过录像缓冲预算" in runtime.session_error
    assert runtime.raw_bytes == runtime.raw_frames == 0


def test_usb_gap_is_not_counted_as_a_long_software_drop_run(tmp_path):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.online = runtime.accept_frames = True
    runtime.admission.record(False, 0, 0, "raw")
    runtime.admission.record(True, .1, .1, "raw")
    runtime.on_connection(False, "unplug")
    runtime.admission.record(True, 100, 100, "raw")
    assert runtime.admission.max_drop_run == 1
    assert runtime.admission.max_drop_span == pytest.approx(.1)


@pytest.mark.parametrize("writer_fails", [False, True])
def test_waiting_writer_stop_or_failure_preserves_order_and_releases_memory(tmp_path, writer_fails):
    entered, release = threading.Event(), threading.Event()

    class PausedSession(RecordingSession):
        def write(self, frame, stamp):
            if not self.summary["frames"]:
                entered.set()
                assert release.wait(5), "test writer was not released"
                if writer_fails:
                    raise OSError("injected disk error")
            return super().write(frame, stamp)

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path),
                           capture_factory=SyntheticCapture, session_factory=PausedSession)
    runtime.encoded = MediaQueue(byte_limit=1024 * 1024, packet_limit=3)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime.writer.start()
    runtime.encoder_thread.start()
    pixels = np.zeros((90, 160, 3), dtype=np.uint8)
    accepted_times = []

    def capture(index):
        before = runtime.admission.dropped
        runtime.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), 100 + index / 30)
        if runtime.admission.dropped == before:
            accepted_times.append(index / 30)

    try:
        runtime._command(Command(CommandKind.START, 1, "start", "paused"))
        wait_for(lambda: runtime.accept_frames)
        capture(0)
        assert entered.wait(2)
        for i in range(1, 4):
            capture(i)
            if i < 3:
                wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy)
        wait_for(lambda: runtime.encoded.snapshot()["pending_bytes"] > 0)
        for i in range(4, 100):
            capture(i)
        assert runtime.admission.dropped > 0
        assert runtime.accept_frames
        runtime._command(Command(CommandKind.STOP, 1, "stop", "paused"))
        time.sleep(.03)
        assert runtime.session_id == "paused"
        assert runtime.encoder_thread.is_alive()
        release.set()
        wait_for(lambda: not runtime.session_id)
        wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy
                 and runtime.encoded.snapshot()["resident_bytes"] == 0)
        assert runtime.raw_frames == runtime.raw_bytes == 0
        errors = [e for e in runtime.events.queue if e[0] == EventKind.RECORDING_FAILED]
        if writer_fails:
            assert len(errors) == 1 and "injected disk error" in errors[0][2]
        else:
            assert not errors
            frames = decoded(next(tmp_path.glob("video/*/*.mp4")))
            assert [t for t, _ in frames] == pytest.approx(accepted_times, abs=1 / 90000)
            assert any(e[0] == EventKind.COMPLETED and e[2] == "已保存" for e in runtime.events.queue)
        for _ in range(runtime.frame_capacity):
            assert runtime.frame_slots.acquire(False)
        for _ in range(runtime.frame_capacity):
            runtime.frame_slots.release()
        snapshot = runtime.encoded.snapshot()
        assert snapshot["resident_peak_bytes"] <= runtime.encoded.byte_limit + runtime.encoded.pending_limit
    finally:
        release.set()
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_pressure_segments_reconnect_and_new_session_preserve_admitted_frames(tmp_path, codec):
    entered, release = threading.Event(), threading.Event()

    class PausedSession(RecordingSession):
        def write(self, frame, stamp):
            if not self.summary["frames"]:
                entered.set()
                assert release.wait(5), "test writer was not released"
            return super().write(frame, stamp)

    config = replace(config_for(tmp_path), width=320, height=180, fps_num=30, codec=codec)
    config = SimpleNamespace(**(vars(config) | {"segment_duration_seconds": 1}))
    runtime = VideoRuntime(None, None, 1, config, capture_factory=SyntheticCapture,
                           session_factory=PausedSession)
    runtime.encoded = MediaQueue(byte_limit=1024 * 1024, packet_limit=3)
    runtime.online, runtime.preview_due = True, float("inf")
    background = np.random.default_rng(19).integers(16, 236, (180, 320, 3), dtype=np.uint8)
    accepted = []

    def capture(index):
        before = runtime.admission.dropped
        runtime.on_frame(synthetic_frame(index, 320, 180, background), 100 + index / 30)
        if runtime.admission.dropped == before:
            accepted.append(index)

    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "pressure-segments"))
        wait_for(lambda: runtime.accept_frames)
        capture(0)
        assert entered.wait(2)
        for index in range(1, 4):
            capture(index)
            if index < 3:
                wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy)
        wait_for(lambda: runtime.encoded.snapshot()["pending_bytes"] > 0)
        for index in range(4, 120):
            if index == 45:
                runtime.on_connection(False, "synthetic reconnect under pressure")
                runtime.on_connection(True, "")
            capture(index)
        assert runtime.admission.dropped > 0 and runtime.accept_frames
        runtime._command(Command(CommandKind.STOP, 1, "stop", "pressure-segments"))
        release.set()
        wait_for(lambda: not runtime.session_id)
        wait_for(lambda: not runtime.encoder_busy and runtime.work.empty()
                 and runtime.encoded.snapshot()["resident_bytes"] == 0)
        files = sorted(tmp_path.glob("video/*/*.mp4"))
        assert len(files) >= 3
        identities = []
        for path in files:
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
            assert frames and frames[0].key_frame
            segment_ids = []
            for frame in frames:
                pixels = np.frombuffer(frame.planes[0], np.uint8).reshape(
                    frame.height, frame.planes[0].line_size)
                segment_ids.append(sum(1 << bit for bit in range(16)
                                       if pixels[12:36, bit * 20 + 5:bit * 20 + 15].mean() > 128))
            assert [float(frame.time) for frame in frames] == pytest.approx(
                [(identity - segment_ids[0]) / 30 for identity in segment_ids], abs=1 / 90000)
            identities.extend(segment_ids)
        assert identities == accepted
        assert runtime.raw_frames == runtime.raw_bytes == 0
        assert not list(tmp_path.rglob("*.recording.mp4"))

        runtime._command(Command(CommandKind.START, 1, "start-again", "fresh"))
        wait_for(lambda: runtime.accept_frames)
        assert runtime.admission.candidates == runtime.admission.dropped == 0
        runtime._log_admission("pressure-segments", final=True)
        assert not runtime.admission.finished
        for index in range(200, 205):
            capture(index)
            wait_for(lambda: not runtime.encoder_busy and runtime.work.empty()
                     and runtime.encoded.snapshot()["resident_bytes"] == 0)
        runtime._command(Command(CommandKind.STOP, 1, "stop-again", "fresh"))
        wait_for(lambda: not runtime.session_id)
        assert runtime.admission.candidates == 5 and runtime.admission.dropped == 0
        assert not any(event[0] == EventKind.RECORDING_FAILED for event in runtime.events.queue)
    finally:
        release.set()
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


def test_writer_progress_is_used_while_encoder_waits(tmp_path):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    now = time.monotonic()
    runtime.encoder_busy = True
    runtime.encoder_busy_since = now - 100
    runtime.encoder_waiting_since = now - 100
    runtime.diagnostics.progress_at = now
    runtime._check_progress(now)
    with pytest.raises(TimeoutError, match="写盘等位"):
        runtime._check_progress(now + 31)


def test_admission_log_reaches_project_file_and_does_not_emit_failure(tmp_path, monkeypatch):
    import base.log_manager as log_module

    target = tmp_path / "logs" / "main.log"
    monkeypatch.setattr(log_module, "LOG_DIR", str(target.parent))
    monkeypatch.setattr(log_module, "LOG_MAPPING", {"core": {
        "log_name": str(target), "log_format": "%(name)s %(message)s",
    }})
    core = logging.getLogger("core")
    old_level = core.level
    monkeypatch.setattr(core, "handlers", [])
    log_module.LogManager.set_log_handler("core")
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.admission = FrameAdmission("file-log")
    runtime.session_id = "file-log"
    try:
        for i in range(6):
            runtime.admission.record(runtime.admission.select(.8, 0, i), i, i, "raw_pressure")
        runtime._log_admission("file-log")
        runtime._log_admission("file-log", final=True)
        runtime._log_admission("file-log", final=True)
        for handler in core.handlers:
            handler.flush()
        output = target.read_text(encoding="utf-8")
        assert output.count("Video frame admission:") == 2
        assert "'session': 'file-log'" in output
        assert "'candidates': 6" in output and "'dropped': 1" in output
        assert "'event': 'ended'" in output
        assert runtime.events.empty()
    finally:
        for handler in core.handlers:
            handler.close()
        core.setLevel(old_level)
