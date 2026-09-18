"""Bounded local evidence for capture failures; never changes recovery decisions."""

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from uuid import uuid4


logger = logging.getLogger("core.video")
PACKET_BYTE_LIMIT = 4 * 1024**2
PACKET_COUNT_LIMIT = 16
FAILURE_SAMPLE_LIMIT = 3


class CaptureDiagnostics:
    def __init__(self, config):
        self.config = config
        self.run = uuid4().hex[:12]
        self.attempt = 0
        self.stage = "open"
        self.outage_since = None
        self.history = deque()
        self.history_bytes = 0
        self.packet = None
        self.packet_number = 0
        self.actual = {}
        self.logs = deque(maxlen=16)

    def begin_attempt(self):
        self.attempt += 1
        self.stage = "open"
        self.history.clear()
        self.history_bytes = 0
        self.packet = None
        self.packet_number = 0
        self.actual = {}
        self.logs.clear()
        logger.info(
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
        logger.info("Video capture opened: run=%s attempt=%s actual=%s", self.run, self.attempt, self.actual)

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
        while self.history and (self.history_bytes + packet.size > PACKET_BYTE_LIMIT
                                or len(self.history) >= PACKET_COUNT_LIMIT):
            self.history_bytes -= len(self.history.popleft()[1])
        if packet.size <= PACKET_BYTE_LIMIT:
            data = bytes(packet)
            self.history.append((self.packet, data))
            self.history_bytes += len(data)

    def first_frame(self):
        elapsed = None if self.outage_since is None else time.monotonic() - self.outage_since
        logger.info("Video capture first frame: run=%s attempt=%s outage_seconds=%s",
                    self.run, self.attempt, elapsed)
        self.outage_since = None

    def failed(self, exc):
        if self.outage_since is None:
            self.outage_since = time.monotonic()
        # str(FFmpegError) may include a global last-error log from another thread.
        logger.warning(
            "Video capture failure: run=%s attempt=%s stage=%s error_type=%s errno=%s "
            "message=%s packet=%s local_ffmpeg_logs=%s",
            self.run, self.attempt, self.stage, type(exc).__name__, getattr(exc, "errno", None),
            str(getattr(exc, "strerror", None) or exc)[:1000], self.packet, list(self.logs),
        )
        if self.stage == "decode" and self.config.recording_root:
            try:
                self._save_sample(exc)
            except OSError:
                logger.exception("Video capture evidence write failed: run=%s attempt=%s; retry continues",
                                 self.run, self.attempt)

    def _save_sample(self, exc):
        root = Path(self.config.recording_root) / "video_diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        destination = None
        for number in range(1, FAILURE_SAMPLE_LIMIT + 1):
            candidate = root / f"capture-failure-{number}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            destination = candidate
            break
        if destination is None:
            logger.warning("Video capture evidence limit reached: directory=%s limit=%s; logs continue",
                           root, FAILURE_SAMPLE_LIMIT)
            return
        report = {
            "run": self.run, "attempt": self.attempt, "stage": self.stage,
            "captured_at": datetime.now(timezone.utc).isoformat(), "actual": self.actual,
            "error_type": type(exc).__name__, "errno": getattr(exc, "errno", None),
            "message": str(getattr(exc, "strerror", None) or exc)[:1000],
            "failed_packet": self.packet, "local_ffmpeg_logs": list(self.logs),
            "log_scope": "capture thread only; excludes native decoder threads",
            "saved_packets": [], "byte_limit": PACKET_BYTE_LIMIT,
            "requested": {"width": self.config.width, "height": self.config.height,
                          "fps_num": self.config.fps_num, "fps_den": self.config.fps_den,
                          "input_format": self.config.input_format},
        }
        for metadata, data in self.history:
            filename = f"packet-{metadata['number']:08d}.bin"
            (destination / filename).write_bytes(data)
            report["saved_packets"].append({**metadata, "file": filename,
                                             "sha256": hashlib.sha256(data).hexdigest()})
        (destination / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.warning("Video capture evidence saved: run=%s attempt=%s directory=%s bytes=%s packets=%s",
                       self.run, self.attempt, destination, self.history_bytes, len(self.history))

    def retry(self):
        logger.info("Video capture closed; retry scheduled: run=%s attempt=%s delay_seconds=2",
                    self.run, self.attempt)
