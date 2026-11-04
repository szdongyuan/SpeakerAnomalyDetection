"""Live camera capture and latest-frame preview in the video child process."""

import multiprocessing
import queue
import threading
import time

from base.log_manager import LogManager
from base.video.models import CommandKind, Event, EventKind
from base.video.usb_capture import CapturePump


class VideoRuntime:
    def __init__(self, channel, mailbox, generation, config, *, capture_factory=CapturePump):
        self.channel, self.mailbox, self.generation = channel, mailbox, generation
        self.gate = threading.Lock()
        self.events = queue.SimpleQueue()
        self.logger = LogManager.set_log_handler("core.video")
        self.sequence = 0
        self.online = False
        self.preview_due = 0.0
        self.preview_enabled = config.enabled
        self.preview_revision = 0
        self.frame_timeout = max(3.0, 3 * config.fps_den / config.fps_num)
        self.capture = capture_factory(config, self.on_frame, self.on_connection)

    def emit(self, kind, **fields):
        self.sequence += 1
        self.channel.send(Event(kind, self.generation, self.sequence, time.monotonic(), **fields))

    def on_connection(self, ready, detail):
        with self.gate:
            if ready != self.online or detail:
                self.logger.info("Video connection ready=%s: %s", ready, detail)
                self.online = ready
                self.events.put((EventKind.READY if ready else EventKind.OFFLINE, detail))

    def on_frame(self, frame, stamp):
        import cv2
        import numpy as np

        self.on_connection(True, "")
        with self.gate:
            if not self.preview_enabled or stamp < self.preview_due:
                return
            revision = self.preview_revision
        rgb = frame.to_ndarray(format="rgb24")
        height, width = rgb.shape[:2]
        factor = min(self.mailbox.width / width, self.mailbox.height / height)
        resized = cv2.resize(rgb, (max(1, round(width * factor)), max(1, round(height * factor))))
        preview = np.zeros((self.mailbox.height, self.mailbox.width, 3), dtype=np.uint8)
        y = (self.mailbox.height - resized.shape[0]) // 2
        x = (self.mailbox.width - resized.shape[1]) // 2
        preview[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        with self.gate:
            if self.preview_enabled and self.preview_revision == revision:
                self.mailbox.publish(preview.tobytes(), revision)
                self.preview_due = stamp + 1 / 15

    def set_preview(self, command):
        with self.gate:
            if command.generation != self.generation or command.preview_revision <= self.preview_revision:
                return
            self.preview_enabled = command.preview_enabled
            self.preview_revision = command.preview_revision
            self.preview_due = 0.0
            self.emit(EventKind.PREVIEW_CHANGED, command_id=command.command_id,
                      preview_enabled=self.preview_enabled, preview_revision=self.preview_revision)

    def run(self):
        self.capture.start()
        heartbeat_due = 0.0
        parent = multiprocessing.parent_process()
        try:
            while parent is None or parent.is_alive():
                if self.channel.poll(0.01):
                    command = self.channel.recv()
                    if command.generation != self.generation:
                        continue
                    if command.kind == CommandKind.SHUTDOWN:
                        break
                    self.set_preview(command)
                while not self.events.empty():
                    kind, detail = self.events.get_nowait()
                    self.emit(kind, detail=detail)
                now = time.monotonic()
                if self.online and now - self.capture.last_progress > self.frame_timeout:
                    self.on_connection(False, "摄像头暂无新画面")
                if now - self.capture.last_progress > 12:
                    raise TimeoutError("摄像头驱动长时间无响应")
                if now >= heartbeat_due:
                    self.emit(EventKind.HEARTBEAT)
                    heartbeat_due = now + 0.25
        finally:
            self.capture.stop()
            self.capture.thread.join(timeout=5)
            if self.capture.thread.is_alive():
                raise TimeoutError("摄像头退出超时")


def usb_video_worker(channel, mailbox, generation, config):
    runtime = VideoRuntime(channel, mailbox, generation, config)
    try:
        config.validate_capture()
        runtime.run()
    except Exception as exc:
        LogManager.set_log_handler("core.video").exception("Video capture failed")
        runtime.emit(EventKind.FAILED, detail=str(exc)[:2000])
    finally:
        try:
            runtime.emit(EventKind.CLOSED)
        finally:
            channel.close()
