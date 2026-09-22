"""Video encoding, two-hour files and bounded in-memory recording statistics."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import logging
import io
import os
import queue
from pathlib import Path
import re
import shutil
import threading
import time


logger = logging.getLogger("core.video")
PROGRESS_LOG_INTERVAL_SECONDS = 600
PENDING_MEDIA_BYTE_LIMIT = 16 * 1024**2


class VideoDiagnostics:
    """One writer times nested stages; capture/control threads only sample metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = ("idle", time.perf_counter())
        self._stats = {}
        self._arrivals = deque(maxlen=120)
        self._slow = False
        self._last_slow = None
        self._recent_slow = deque(maxlen=16)
        self._recent_slow_truncated = 0
        self._next_report = 0.0
        self._overflow = None
        self.session_id = ""
        self.progress = 0
        self.progress_at = time.monotonic()

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        with self._lock:
            previous = self._active
            self._active = (name, started)
        try:
            yield
            if name in {"encode", "encoder_flush", "file_write", "disk_sync", "file_close", "rename_media", "mux_close"}:
                with self._lock:
                    self.progress += 1
                    self.progress_at = time.monotonic()
        finally:
            ended = time.perf_counter()
            elapsed = ended - started
            with self._lock:
                count, total, maximum, peak_start, peak_end = self._stats.get(name, (0, 0.0, 0.0, 0, 0))
                if not count or elapsed > maximum:
                    maximum, peak_start, peak_end = elapsed, round(started * 1e9), round(ended * 1e9)
                self._stats[name] = (count + 1, total + elapsed, maximum, peak_start, peak_end)
                self._slow |= elapsed >= 0.2
                if elapsed >= 0.2:
                    self._last_slow = {"stage": name, "elapsed_ms": round(elapsed * 1000, 2),
                                       "start_ns": round(started * 1e9), "end_ns": round(ended * 1e9)}
                    if name in {"disk_sync", "file_write", "file_close", "mux_close"}:
                        if len(self._recent_slow) == self._recent_slow.maxlen:
                            self._recent_slow_truncated += 1
                        self._recent_slow.append(self._last_slow)
                self._active = previous

    def begin(self, session_id):
        with self._lock:
            self.session_id = session_id
            self._stats.clear()
            self._arrivals.clear()
            self._slow = False
            self._last_slow = None
            self._recent_slow.clear()
            self._recent_slow_truncated = 0
            self._active = ("idle", time.perf_counter())

    def arrival(self):
        with self._lock:
            self._arrivals.append(time.perf_counter())

    def progress_snapshot(self):
        with self._lock:
            return self.progress, self.progress_at

    def _snapshot(self, reason, queue_items, capacity, now):
        stage, started = self._active
        return {
            "reason": reason, "session": self.session_id, "captured_at": utc_now(),
            "queue_items": queue_items, "frame_capacity": capacity,
            "active_stage": stage,
            "active_start_ns": round(started * 1e9) if stage != "idle" else None,
            "captured_perf_ns": round(now * 1e9),
            "interval_clock": "perf_counter",
            "active_ms": round((now - started) * 1000, 2) if stage != "idle" else 0,
            "arrivals_last_second": sum(now - stamp <= 1 for stamp in self._arrivals),
            "arrival_sample_limit": self._arrivals.maxlen,
            "last_slow": dict(self._last_slow) if self._last_slow else None,
            "recent_slow_scope": "last_16_slow_io_intervals",
            "recent_slow": [dict(interval) for interval in self._recent_slow],
            "recent_slow_truncated": self._recent_slow_truncated,
            "stages_scope": "cumulative",
            "stages": {
                name: {"count": n, "mean_ms": round(total / n * 1000, 2),
                       "max_ms": round(maximum * 1000, 2),
                       "max_start_ns": peak_start, "max_end_ns": peak_end}
                for name, (n, total, maximum, peak_start, peak_end) in self._stats.items()
            },
        }

    def overflow(self, queue_items, capacity):
        with self._lock:
            if self._overflow is None:
                self._overflow = self._snapshot("queue_full", queue_items, capacity, time.perf_counter())

    def take_report(self, queue_items, capacity, *, backlog=False):
        with self._lock:
            if self._overflow is not None:
                report, self._overflow = self._overflow, None
                self._next_report = time.perf_counter() + 30
                self._slow = False
                return report
            now = time.perf_counter()
            stage, started = self._active
            slow_active = stage != "idle" and now - started >= 0.2
            backlog |= queue_items >= capacity * 0.8
            if now < self._next_report or not (self._slow or slow_active or backlog):
                return None
            report = self._snapshot("slow_or_backlog", queue_items, capacity, now)
            self._next_report = now + 30
            self._slow = False
            return report


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_space(root, reserve):
    if shutil.disk_usage(root).free <= reserve:
        raise OSError("录像磁盘空间不足，已停止录像；不会删除已有文件")


class RecordingRootLease:
    """One live video owner per recording root; OS releases the lock on a crash."""

    def __init__(self, root):
        self.root = Path(root)
        self.stream = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = (self.root / ".video-owner.lock").open("a+b")
        try:
            if self.stream.tell() == 0:
                self.stream.write(b"\0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise OSError("录像目录正在被另一视频进程使用，未修改其录像记录") from exc
        return self

    def __exit__(self, *args):
        self.stream.close()


def validate_volume(root):
    """Two-hour segments can exceed 4 GiB at higher bitrates; require a suitable local filesystem."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    volume = ctypes.create_unicode_buffer(261)
    if not kernel.GetVolumePathNameW(str(root), volume, len(volume)):
        raise ctypes.WinError(ctypes.get_last_error())
    if kernel.GetDriveTypeW(volume.value) == 4:
        raise ValueError("首版录像仅支持本地磁盘，不支持网络共享目录")
    filesystem = ctypes.create_unicode_buffer(261)
    serial, length, flags = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    if not kernel.GetVolumeInformationW(
        volume.value, None, 0, ctypes.byref(serial), ctypes.byref(length),
        ctypes.byref(flags), filesystem, len(filesystem),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if filesystem.value.upper() in {"FAT", "FAT32"}:
        raise ValueError("FAT/FAT32不支持所需的大文件，请选择NTFS或exFAT本地磁盘")


@dataclass(frozen=True)
class StreamParameters:
    codec: str
    width: int
    height: int
    extradata: bytes


@dataclass(frozen=True)
class EncodedPacket:
    payload: bytes
    pts: int
    dts: int
    duration: int
    time_base: Fraction
    keyframe: bool
    side_data: tuple

    @classmethod
    def capture(cls, packet):
        return cls(bytes(packet), packet.pts, packet.dts, packet.duration,
                   packet.time_base, packet.is_keyframe,
                   tuple((side.data_type, bytes(side)) for side in packet.iter_sidedata()))

    def restore(self):
        import av

        packet = av.Packet(self.payload)
        packet.pts, packet.dts = self.pts, self.dts
        packet.duration, packet.time_base = self.duration, self.time_base
        packet.is_keyframe = self.keyframe
        for name, data in self.side_data:
            side = av.packet.PacketSideData(
                av.packet.packet_sidedata_type_from_literal(name), len(data),
            )
            side.update(data)
            side.to_packet(packet)
        return packet


@dataclass(frozen=True)
class EncodedFrame:
    packets: tuple
    frames: int = 1

    @property
    def cost(self):
        return 128 + sum(
            len(p.payload) + sum(len(data) for _, data in p.side_data) + 512
            for p in self.packets
        )


@dataclass(frozen=True)
class MediaItem:
    message: object
    size: int
    packets: int
    data: bool
    enqueued_at: float


class MediaQueueClosed(OSError):
    pass


class MediaQueueCancelled(OSError):
    pass


class MediaQueue:
    """FIFO credits cover queued AND in-flight data, with separate control slots."""

    def __init__(self, byte_limit=64 * 1024**2, packet_limit=8192, control_limit=32):
        self.byte_limit, self.packet_limit, self.control_limit = byte_limit, packet_limit, control_limit
        self._condition = threading.Condition()
        self._queue = deque()
        self._inflight = {}
        self._closed = False
        self._bytes = self._packets = self._data = self._controls = 0
        self._peak = 0
        # One producer may hold one bounded batch while the writer releases credits.
        self.pending_limit = min(byte_limit, PENDING_MEDIA_BYTE_LIMIT)
        self._pending = None
        self._resident_peak = 0

    def put(self, message, *, size=0, packets=0, cancelled=None):
        """Wait for media credits, preserving FIFO; controls never wait for media."""
        if not size and not packets:
            return self.put_nowait(message)
        if size < 0 or packets < 0:
            raise ValueError("媒体队列额度不能为负数")
        if size > self.pending_limit or packets > self.packet_limit:
            raise ValueError("单批压缩数据超过缓冲上限，录像不完整")
        with self._condition:
            if self._pending is not None:
                raise RuntimeError("压缩通道只允许一个编码生产者")
            self._pending = (size, packets, time.monotonic())
            self._resident_peak = max(self._resident_peak, self._bytes + size)
            try:
                while True:
                    if self._closed:
                        raise MediaQueueClosed("视频写入通道已关闭")
                    if cancelled is not None and cancelled.is_set():
                        raise MediaQueueCancelled("录像会话已结束")
                    if (self._bytes + size <= self.byte_limit
                            and self._packets + packets <= self.packet_limit
                            and self._data < self.packet_limit):
                        self.put_nowait(message, size=size, packets=packets)
                        return
                    self._condition.wait(.1)
            finally:
                self._pending = None

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def put_nowait(self, message, *, size=0, packets=0):
        if size < 0 or packets < 0:
            raise ValueError("媒体队列额度不能为负数")
        data = size > 0 or packets > 0
        with self._condition:
            if self._closed:
                raise MediaQueueClosed("视频写入通道已关闭")
            if data:
                if (self._bytes + size > self.byte_limit
                        or self._packets + packets > self.packet_limit
                        or self._data >= self.packet_limit):
                    raise queue.Full
            elif self._controls >= self.control_limit:
                raise queue.Full
            item = MediaItem(message, size, packets, data, time.monotonic())
            self._queue.append(item)
            self._bytes += size
            self._packets += packets
            self._data += data
            self._controls += not data
            self._peak = max(self._peak, self._bytes)
            self._resident_peak = max(self._resident_peak, self._bytes)
            self._condition.notify()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._queue:
                if self._closed:
                    raise MediaQueueClosed("视频写入通道已关闭")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            item = self._queue.popleft()
            self._inflight[id(item)] = item
            return item

    def _uncharge(self, item):
        self._bytes -= item.size
        self._packets -= item.packets
        self._data -= item.data
        self._controls -= not item.data

    def release(self, item):
        with self._condition:
            if self._inflight.pop(id(item), None) is not item:
                raise ValueError("媒体数据额度已释放或不属于此队列")
            self._uncharge(item)
            self._condition.notify_all()

    def close(self, *, discard=False):
        with self._condition:
            self._closed = True
            if discard:
                while self._queue:
                    self._uncharge(self._queue.popleft())
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            oldest = self._queue[0].enqueued_at if self._queue else time.monotonic()
            for item in self._inflight.values():
                oldest = min(oldest, item.enqueued_at)
            pending_bytes, pending_packets, pending_since = self._pending or (0, 0, time.monotonic())
            return {"bytes": self._bytes, "packets": self._packets,
                    "data_items": self._data,
                    "items": self._data + self._controls, "waiting": len(self._queue),
                    "peak_bytes": self._peak, "oldest_seconds": time.monotonic() - oldest,
                    "pending_bytes": pending_bytes, "pending_packets": pending_packets,
                    "pending_seconds": time.monotonic() - pending_since,
                    "resident_bytes": self._bytes + pending_bytes,
                    "resident_peak_bytes": self._resident_peak}


class FrameEncoder:
    """Own a single segment's codec on the encoding thread; never perform file I/O."""

    def __init__(self, config, diagnostics=None):
        import av

        self.diagnostics = diagnostics or VideoDiagnostics()
        self.first_time = self.last_time = None
        self.last_pts = -1
        with self.diagnostics.stage("open_encoder"):
            self.codec = av.CodecContext.create("libx264" if config.codec == "h264" else "libx265", "w")
            codec = self.codec
            codec.width, codec.height, codec.pix_fmt = config.width, config.height, "yuv420p"
            codec.time_base = Fraction(1, 90000)
            codec.framerate = Fraction(config.fps_num, config.fps_den)
            codec.bit_rate, codec.thread_count = config.target_bitrate_bps, 2
            codec.max_b_frames, codec.gop_size = 0, max(1, round(codec.framerate))
            codec.flags |= av.codec.context.Flags.global_header
            codec.options = {"preset": "veryfast", "tune": "zerolatency"}
            if config.codec == "h265":
                codec.options["x265-params"] = "pools=1:frame-threads=1:log-level=error"
            codec.open()
            self.parameters = StreamParameters(config.codec, config.width, config.height, bytes(codec.extradata))

    def write(self, frame, timestamp):
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("采集时间戳未递增，已停止录像，避免生成错误的录像时间轴")
        if self.first_time is None:
            self.first_time = timestamp
        pts = max(round((timestamp - self.first_time) * 90000), self.last_pts + 1)
        with self.diagnostics.stage("pixel_convert"):
            frame = frame.reformat(width=self.codec.width, height=self.codec.height, format="yuv420p")
        frame.pts, frame.time_base = pts, self.codec.time_base
        with self.diagnostics.stage("encode"):
            result = EncodedFrame(tuple(EncodedPacket.capture(p) for p in self.codec.encode(frame)))
        self.last_time, self.last_pts = timestamp, pts
        return result

    def finish(self):
        with self.diagnostics.stage("encoder_flush"):
            return EncodedFrame(tuple(EncodedPacket.capture(p) for p in self.codec.encode(None)), frames=0)


class TimedVideoFile:
    """Preserve complete writes and I/O failures while measuring the actual file boundary."""

    def __init__(self, stream, diagnostics):
        self.stream, self.diagnostics = stream, diagnostics

    def write(self, data):
        with self.diagnostics.stage("file_write"):
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = self.stream.write(view[written:])
                if count is None or count <= 0:
                    raise OSError("录像文件写入未取得进展")
                written += count
            return written

    def flush(self):
        with self.diagnostics.stage("file_flush"):
            return self.stream.flush()

    def close(self):
        with self.diagnostics.stage("file_close"):
            return self.stream.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class SegmentWriter:
    """Writer-owned fMP4 muxer; accepts packets from the separate FrameEncoder."""

    def __init__(self, path, *, parameters: StreamParameters, diagnostics=None):
        import av

        self.path = Path(path)
        self.diagnostics = diagnostics or VideoDiagnostics()
        self.frames = self.packet_count = 0
        self.failed = False
        self.first_time = self.last_time = None
        self._file = TimedVideoFile(self.path.open("xb", buffering=0), self.diagnostics)
        self._container = None
        try:
            self._container = av.open(self._file, "w", format="mp4", options={
                "movflags": "empty_moov+default_base_moof+frag_keyframe",
                "frag_duration": "1000000", "flush_packets": "1",
            })
            fmt = "h264" if parameters.codec == "h264" else "hevc"
            with av.open(io.BytesIO(parameters.extradata), format=fmt) as header:
                self.stream = self._container.add_stream_from_template(header.streams.video[0], opaque=True)
            context = self.stream.codec_context
            context.width, context.height = parameters.width, parameters.height
            context.pix_fmt, context.extradata = "yuv420p", parameters.extradata
            self.stream.time_base = Fraction(1, 90000)
            self._container.start_encoding()
            self._header_size = self._file.tell()
        except Exception:
            self.abandon()
            raise

    @property
    def has_written_media(self):
        return self.packet_count > 0 and self._file.tell() > self._header_size

    def write(self, frame: EncodedFrame, timestamp):
        try:
            with self.diagnostics.stage("mux_write"):
                for item in frame.packets:
                    packet = item.restore()
                    packet.stream = self.stream
                    self._container.mux(packet)
                    self.packet_count += 1
        except Exception:
            self.failed = True
            raise
        if frame.frames:
            if self.first_time is None:
                self.first_time = timestamp
            self.last_time = timestamp
            self.frames += frame.frames

    def sync(self):
        try:
            with self.diagnostics.stage("disk_sync"):
                self._file.flush()
                os.fsync(self._file.fileno())
        except OSError:
            self.failed = True
            raise

    def finish(self):
        if self.failed:
            raise OSError("录像文件写入曾失败，已保留未完成尾段")
        try:
            with self.diagnostics.stage("mux_close"):
                container, self._container = self._container, None
                container.close()
            self.sync()
        finally:
            self._file.close()
        if not self.frames:
            raise ValueError("录像没有有效视频帧")
        # Read-back validation belongs to offline tests: large-file probing stalls
        # the writer for seconds and can exhaust even a separate encoded queue.
        final = self.path.with_name(self.path.name.replace(".recording.mp4", ".mp4"))
        if final.exists():
            raise FileExistsError(final)
        with self.diagnostics.stage("rename_media"):
            self.path.rename(final)
        return final

    def abandon(self):
        """Release only on the owner thread; retain failed media for inspection."""
        try:
            if self._container is not None:
                container, self._container = self._container, None
                container.close()
        finally:
            self._file.close()


class RecordingSession:
    """One manual recording; media on disk, current counters in memory, history in log."""

    def __init__(self, config, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            raise ValueError("invalid recording session identity")
        self.config, self.session_id = config, session_id
        self.diagnostics = VideoDiagnostics()
        root = Path(config.recording_root) / "video"
        root.mkdir(parents=True, exist_ok=True)
        validate_volume(root)
        check_space(root, config.min_free_bytes)
        name = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = 1
        while True:
            self.directory = root / (name if suffix == 1 else f"{name}_{suffix:02d}")
            try:
                self.directory.mkdir()
                break
            except FileExistsError:
                # Never open old manifests, including when a directory name collides.
                suffix += 1
        self.segment_writer = None
        self.first_time = None
        self.last_time = None
        self._last_sync = time.monotonic()
        self._last_progress_log = self._last_sync
        self._segment_sequence = 0
        self._last_file_name = ""
        self._last_file_bytes = 0
        self._pending_gap = False
        self.closed = False
        # No historical lists: memory use does not grow with segments or USB reconnects.
        self.summary = {
            "state": "starting", "frames": 0, "segments": 0, "gaps": 0,
            "elapsed_seconds": 0, "current_segment": None, "error": "",
        }
        logger.info(
            "Video session created: session=%s directory=%s requested_size=%sx%s "
            "requested_fps=%s/%s codec=%s target_bitrate_bps=%s",
            session_id, self.directory, config.width, config.height,
            config.fps_num, config.fps_den, config.codec, config.target_bitrate_bps,
        )

    def _log_statistics(self, event, *, level=logging.INFO):
        logger.log(
            level, "Video %s: session=%s directory=%s state=%s elapsed_seconds=%.3f "
            "frames=%d segments=%d gaps=%d file=%s bytes=%d error=%s media_validation=not_performed",
            event, self.session_id, self.directory.name, self.summary["state"],
            self.summary["elapsed_seconds"], self.summary["frames"],
            self.summary["segments"], self.summary["gaps"],
            self._last_file_name, self._last_file_bytes, self.summary["error"],
        )

    def prepare(self, parameters: StreamParameters):
        """Open the output file from encoder parameters before accepting packets."""
        if self.closed:
            raise RuntimeError("recording session is closed")
        if self.segment_writer is None:
            with self.diagnostics.stage("check_space"):
                check_space(self.directory, self.config.min_free_bytes)
            self._segment_sequence += 1
            path = self.directory / f"{self.directory.name}_{self._segment_sequence:03d}.recording.mp4"
            self.summary["current_segment"] = path.name
            self._last_file_name, self._last_file_bytes = path.name, 0
            logger.info(
                "Video segment preparing: session=%s file=%s", self.session_id, path.name,
            )
            with self.diagnostics.stage("open_segment"):
                self.segment_writer = SegmentWriter(path, parameters=parameters, diagnostics=self.diagnostics)

    def write(self, frame: EncodedFrame, timestamp):
        if self.closed:
            raise RuntimeError("recording session is closed")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("采集时间戳未递增，已停止录像，避免生成错误的录像时间轴")
        if self.segment_writer is None:
            raise RuntimeError("压缩帧缺少对应的录像文件头")
        self.segment_writer.write(frame, timestamp)
        if self.first_time is None:
            self.first_time = timestamp
            self._last_progress_log = time.monotonic()
        self.last_time = timestamp
        self.summary["frames"] += 1
        self.summary["elapsed_seconds"] = timestamp - self.first_time
        if self.segment_writer.has_written_media:
            transitioned = self.summary["state"] == "starting" or self._pending_gap
            self.summary["state"] = "recording"
            if transitioned:
                self._last_file_bytes = self.segment_writer.path.stat().st_size
                self._log_statistics("capture resumed" if self._pending_gap else "recording started")
                self._pending_gap = False
        self.maintain()
        return self.segment_writer.has_written_media

    def next_segment(self, parameters):
        self._finish_segment()
        self.prepare(parameters)

    def write_tail(self, frame):
        if self.segment_writer is not None and frame.packets:
            self.segment_writer.write(frame, self.last_time)

    def maintain(self):
        if self.closed:
            return
        now = time.monotonic()
        if now - self._last_sync >= 1:
            # Media sync/space checks stay independent of the ten-minute log interval.
            with self.diagnostics.stage("check_space"):
                check_space(self.directory, self.config.min_free_bytes)
            if self.segment_writer is not None:
                self.segment_writer.sync()
            self._last_sync = now
        if (
            self.segment_writer is not None and self.segment_writer.has_written_media
            and now - self._last_progress_log >= PROGRESS_LOG_INTERVAL_SECONDS
        ):
            self._last_file_bytes = self.segment_writer.path.stat().st_size
            self._log_statistics("recording progress")
            # Do not emit catch-up bursts if the writer was delayed or disconnected.
            self._last_progress_log = now

    def _finish_segment(self):
        segment_writer, self.segment_writer = self.segment_writer, None
        if segment_writer is None:
            return
        try:
            if segment_writer.failed:
                raise OSError("录像文件写入曾失败，已保留未完成尾段")
            if segment_writer.first_time is None:
                segment_writer.abandon()
                segment_writer.path.unlink()
                self.summary["current_segment"] = None
                self._last_file_name, self._last_file_bytes = "", 0
                return
            final = segment_writer.finish()
        except Exception:
            # Take ownership once: later session cleanup must not re-flush/retry a
            # failed trailer or rename. Keep the incomplete file for inspection.
            try:
                segment_writer.abandon()
            except (OSError, ValueError):
                logger.exception("Video resource cleanup failed: session=%s", self.session_id)
            raise
        self.summary["segments"] += 1
        self.summary["current_segment"] = None
        self._last_file_name, self._last_file_bytes = final.name, final.stat().st_size
        logger.info(
            "Video segment completed: session=%s file=%s frames=%d elapsed_seconds=%.3f bytes=%d",
            self.session_id, final.name, segment_writer.frames,
            segment_writer.last_time - segment_writer.first_time, self._last_file_bytes,
        )

    def gap(self, reason, *, disconnected_at=None):
        self.summary["gaps"] += 1
        self.summary["state"] = "recovering"
        self._pending_gap = True
        logger.warning(
            "Video capture interrupted: session=%s directory=%s detected_at=%s "
            "elapsed_seconds=%.3f frames=%d reason=%s",
            self.session_id, self.directory.name, disconnected_at or utc_now(),
            self.summary["elapsed_seconds"], self.summary["frames"], reason[:1000],
        )
        self._finish_segment()

    def finish(self, *, error=""):
        if self.closed:
            return
        try:
            self._finish_segment()
        except Exception as exc:
            logger.exception(
                "Video finalization failed: session=%s directory=%s", self.session_id, self.directory.name,
            )
            error = f"{error}; 收尾失败：{exc}".strip("; ")
        self.closed = True
        self.summary["state"] = "failed" if error else "interrupted" if self.summary["gaps"] else "completed"
        self.summary["error"] = error[:2000]
        self._log_statistics("session ended", level=logging.ERROR if error else logging.INFO)
        if error:
            raise OSError(error)
