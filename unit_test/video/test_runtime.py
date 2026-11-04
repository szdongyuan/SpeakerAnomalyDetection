import threading
import time

import av
import numpy as np

from base.video.config import VideoConfig
from base.video.runtime import VideoRuntime, usb_video_worker
from base.video.models import EventKind
from base.video.service import VideoService


class SyntheticCapture:
    """Synthetic USB camera for preview, disconnect and shutdown integration tests."""

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
            if self.config.device_name == "disconnect" and index == 10:
                self.on_connection(False, "test unplug")
                self.stop_event.wait(.5)
                self.on_connection(True, "")
            if self.config.device_name in {"stall", "stall8"} and index == 30:
                # No OFFLINE callback: exercise the unchanged 3-second watchdog.
                self.stop_event.wait(8 if self.config.device_name == "stall8" else 4.2)
                self.last_progress = time.monotonic()
            pixels = np.full((90, 160, 3), index % 200, dtype=np.uint8)
            pixels[:, :] = (65, 115, 160)
            pixels[:, index % 155:index % 155 + 5] = (230, 230, 230)
            import cv2
            cv2.putText(pixels, f"TEST {index}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
            self.on_frame(av.VideoFrame.from_ndarray(pixels, format="rgb24"), time.monotonic())


def synthetic_video_worker(channel, mailbox, generation, config):
    try:
        runtime = VideoRuntime(channel, mailbox, generation, config, capture_factory=SyntheticCapture)
        runtime.run()
        runtime.emit(EventKind.CLOSED)
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


def test_camera_disconnect_resumes_preview_without_media_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = VideoConfig(enabled=True, device_id="synthetic", device_name="disconnect", width=160, height=90)
    service = VideoService(worker_target=synthetic_video_worker, worker_options=config, preview_size=config.preview_size)
    service.start()
    try:
        before = wait_for(service.latest_preview)
        wait_for(lambda: service.status.connection == "reconnecting")
        assert service.latest_preview() is None
        after = wait_for(lambda: service.latest_preview(before.sequence))
        assert after.width == 160 and after.height == 90
    finally:
        service.shutdown()
        assert service.wait_closed()
    assert not list(tmp_path.rglob("*.mp4"))
    assert not (tmp_path / "video_diagnostics").exists()
    assert not (tmp_path / ".video-owner.lock").exists()


def test_unconfigured_camera_reports_failure():
    service = VideoService(worker_target=usb_video_worker, worker_options=VideoConfig())
    service.start()
    assert service.wait_closed()
    assert service.status.connection == "unavailable"
    assert "请选择摄像头" in service.status.connection_detail
