"""Bounded local evidence for capture failures; never changes recovery decisions."""

from collections import deque
import hashlib
import time
from uuid import uuid4

from base.log_manager import LogManager


class CaptureDiagnostics:
    def __init__(self, config):
        self._logger = LogManager.set_log_handler("core.video")
        self.config = config
        self.run = uuid4().hex[:12]
        self.attempt = 0
        self.stage = "open"
        self.outage_since = None
        self.packet = None
        self.packet_number = 0
        self.actual = {}
        self.logs = deque(maxlen=16)

    def begin_attempt(self):
        self.attempt += 1
        self.stage = "open"
        self.packet = None
        self.packet_number = 0
        self.actual = {}
        self.logs.clear()
        self._logger.info(
            "Video capture attempt: run=%s attempt=%s device=%s requested=%sx%s fps=%s/%s input=%s",
            self.run, self.attempt, hashlib.sha256(self.config.device_id.encode()).hexdigest()[:12],
            self.config.width, self.config.height, self.config.fps_num, self.config.fps_den,
            self.config.input_format,
        )

    def opened(self, source):
        stream = source.streams.video[0]
        codec = stream.codec_context
        self.actual = {
            "codec": codec.name, "width": codec.width, "height": codec.height,
            "pixel_format": codec.format.name if codec.format else None,
            "average_rate": str(stream.average_rate), "time_base": str(stream.time_base),
            "thread_count": codec.thread_count, "thread_type": str(codec.thread_type),
        }
        self._logger.info("Video capture opened: run=%s attempt=%s actual=%s", self.run, self.attempt, self.actual)

    def add_logs(self, logs):
        for level, component, message in logs:
            self.logs.append((self.stage, level, component[:80], message[:1000]))

    def remember(self, packet):
        self.packet_number += 1
        self.packet = {
            "number": self.packet_number, "size": packet.size,
            "pts": packet.pts, "dts": packet.dts, "time_base": str(packet.time_base),
            "keyframe": packet.is_keyframe, "corrupt": packet.is_corrupt,
        }

    def first_frame(self):
        elapsed = None if self.outage_since is None else time.monotonic() - self.outage_since
        self._logger.info("Video capture first frame: run=%s attempt=%s outage_seconds=%s",
                    self.run, self.attempt, elapsed)
        self.outage_since = None

    def failed(self, exc, *, retry_delay_seconds=None):
        if self.outage_since is None:
            self.outage_since = time.monotonic()
        # str(FFmpegError) may include a global last-error log from another thread.
        self._logger.warning(
            "Video capture failure: run=%s attempt=%s stage=%s error_type=%s errno=%s "
            "message=%s packet=%s local_ffmpeg_logs=%s retry_delay_seconds=%s",
            self.run, self.attempt, self.stage, type(exc).__name__, getattr(exc, "errno", None),
            str(getattr(exc, "strerror", None) or exc)[:1000], self.packet, list(self.logs),
            retry_delay_seconds,
        )
