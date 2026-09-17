"""Video child with separate capture, encoding and file-owner threads."""

import multiprocessing
import logging
import os
import queue
import threading
import time

from base.video.models import CommandKind, Event, EventKind
from base.video.recording import (
    RecordingRootLease, RecordingSession, VideoDiagnostics, FrameEncoder,
    MediaQueue, MediaQueueClosed, MediaQueueCancelled, utc_now,
)
from base.video.usb_capture import CapturePump


logger = logging.getLogger("core.video")

RAW_FRAME_BYTE_LIMIT = 640 * 1024**2
SHEDDING_START_PRESSURE = .8
SHEDDING_STRONG_PRESSURE = .9
SHEDDING_RECOVERY_PRESSURE = .5
SHEDDING_RECOVERY_SECONDS = 2.0
SHEDDING_INTERVALS = (0, 6, 3)
SHEDDING_LOG_INTERVAL_SECONDS = 30.0


class FrameAdmission:
    """Bounded per-session admission statistics, owned under VideoRuntime.gate."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.level = self.phase = 0
        self.low_since = None
        self.candidates = self.dropped = 0
        self.episodes = self.recoveries = 0
        self.raw_peak = self.compressed_peak = 0.0
        self.first_drop_at = None
        self.drop_run = self.max_drop_run = 0
        self.last_stamp = self.drop_run_start = None
        self.max_drop_span = 0.0
        self.reason = ""
        self.next_log_at = 0.0
        self.reported_candidates = self.reported_dropped = 0
        self.reported_recoveries = self.reported_episodes = 0
        self.has_reported = self.finished = False

    def select(self, raw_pressure, compressed_pressure, now):
        self.raw_peak = max(self.raw_peak, raw_pressure)
        self.compressed_peak = max(self.compressed_peak, compressed_pressure)
        pressure = max(raw_pressure, compressed_pressure)
        previous = self.level
        if pressure >= SHEDDING_STRONG_PRESSURE:
            self.level = 2
        elif pressure >= SHEDDING_START_PRESSURE:
            self.level = max(self.level, 1)
        if pressure < SHEDDING_RECOVERY_PRESSURE and self.level:
            if self.low_since is None:
                self.low_since = now
            if now - self.low_since >= SHEDDING_RECOVERY_SECONDS:
                self.level = 0
                self.recoveries += 1
        else:
            self.low_since = None
        if self.level != previous:
            self.phase = 0
            if previous == 0:
                self.episodes += 1
        if not self.level:
            return False
        self.phase += 1
        return self.phase % SHEDDING_INTERVALS[self.level] == 0

    def record(self, skipped, stamp, now, reason):
        self.candidates += 1
        if skipped:
            self.dropped += 1
            self.reason = reason
            if self.first_drop_at is None:
                self.first_drop_at = now
            if not self.drop_run:
                self.drop_run_start = self.last_stamp if self.last_stamp is not None else stamp
            self.drop_run += 1
            self.max_drop_run = max(self.max_drop_run, self.drop_run)
            self.max_drop_span = max(self.max_drop_span, stamp - self.drop_run_start)
        else:
            self.drop_run = 0
        self.last_stamp = stamp

    def report(self, now, *, final=False):
        if self.finished:
            return None
        changed = (self.dropped != self.reported_dropped
                   or self.recoveries != self.reported_recoveries
                   or self.episodes != self.reported_episodes)
        if not final:
            if not self.dropped:
                return None
            if self.has_reported and (now < self.next_log_at or (not changed and not self.level)):
                return None
        window_candidates = self.candidates - self.reported_candidates
        window_dropped = self.dropped - self.reported_dropped
        if final:
            event = "ended"
        elif not self.has_reported:
            event = "started"
        else:
            event = "progress" if self.level else "recovered"
        report = {
            "event": event, "session": self.session_id, "level": self.level,
            "reason": self.reason, "candidates": self.candidates,
            "accepted": self.candidates - self.dropped, "dropped": self.dropped,
            "drop_ratio": self.dropped / self.candidates if self.candidates else 0.0,
            "window_candidates": window_candidates, "window_dropped": window_dropped,
            "window_drop_ratio": window_dropped / window_candidates if window_candidates else 0.0,
            "seconds_since_first_drop": now - self.first_drop_at if self.first_drop_at is not None else 0.0,
            "episodes": self.episodes, "recoveries": self.recoveries,
            "max_drop_run": self.max_drop_run, "max_drop_span_seconds": self.max_drop_span,
            "raw_pressure_peak": self.raw_peak, "compressed_pressure_peak": self.compressed_peak,
            "media_validation": "not_performed",
        }
        self.has_reported, self.finished = True, final
        self.next_log_at = now + SHEDDING_LOG_INTERVAL_SECONDS
        self.reported_candidates, self.reported_dropped = self.candidates, self.dropped
        self.reported_episodes, self.reported_recoveries = self.episodes, self.recoveries
        return report


class VideoRuntime:
    def __init__(
        self, channel, mailbox, generation, config, *,
        capture_factory=CapturePump, session_factory=RecordingSession,
    ):
        self.channel, self.mailbox = channel, mailbox
        self.generation, self.config = generation, config
        self.session_factory = session_factory
        self.sequence = 0
        self.events = queue.Queue(maxsize=32)
        # RGB24-sized frame estimate; codec/driver/preview allocations are additional.
        self.frame_capacity = min(60, RAW_FRAME_BYTE_LIMIT // (config.width * config.height * 3))
        if not self.frame_capacity:
            raise ValueError("单帧超过640MiB录像缓冲预算，请降低采集分辨率")
        self.frame_slots = threading.BoundedSemaphore(self.frame_capacity)
        self.work = queue.Queue(maxsize=self.frame_capacity + 32)
        self.encoded = MediaQueue()
        self.raw_bytes = 0
        self.raw_frames = 0
        self.admission = FrameAdmission("")
        self.session_cancelled = threading.Event()
        self.encoder_waiting_since = None
        self.session_error = ""
        self.drain_started = None
        self.encoder_busy = False
        self.encoder_busy_since = self.writer_busy_since = time.monotonic()
        self.no_progress_timeout = 30.0
        self.drain_timeout = 120.0
        self.stop_requested = False
        self.gate = threading.Lock()
        self.session_id = ""
        self.online = False
        self.connection_epoch = 0
        self.accept_frames = False
        self.start_pending = False
        self.stopping = False
        self.fatal_error = ""
        self.writer_busy = False
        self.diagnostics = VideoDiagnostics()
        self.encode_diagnostics = VideoDiagnostics()
        self.preview_due = 0.0
        self.capture = capture_factory(config, self.on_frame, self.on_connection)
        self.encoder_thread = threading.Thread(target=self._encode_loop, name="VideoEncoder", daemon=True)
        self.writer = threading.Thread(target=self._write_loop, name="VideoWriter", daemon=True)

    def post(self, kind, session_id="", detail=""):
        try:
            self.events.put_nowait((kind, session_id, detail))
        except queue.Full:
            self.fatal_error = "视频状态队列已满"

    def on_connection(self, ready, detail):
        with self.gate:
            was_online = self.online
            self.connection_epoch += 1
            self.online = ready
            if ready != was_online or detail:
                logger.info("Video connection ready=%s: %s", ready, detail)
            self.post(EventKind.READY if ready else EventKind.OFFLINE, detail=detail)
            if not ready and was_online and self.accept_frames:
                # USB gaps are not part of a run of software-skipped frames.
                self.admission.drop_run = 0
                self.admission.last_stamp = None
                self.admission.low_since = None
                self._raw_control("gap", self.session_id, (detail, utc_now()))
                self.post(EventKind.RECOVERING, self.session_id, detail)

    def on_frame(self, frame, stamp):
        import cv2

        if not self.online:
            self.on_connection(True, "")
        if stamp >= self.preview_due:
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            factor = min(self.mailbox.width / width, self.mailbox.height / height)
            resized = cv2.resize(rgb, (max(1, round(width * factor)), max(1, round(height * factor))))
            # Letterbox here so the fixed transport shape never distorts the source.
            import numpy as np
            preview = np.zeros((self.mailbox.height, self.mailbox.width, 3), dtype=np.uint8)
            y = (self.mailbox.height - resized.shape[0]) // 2
            x = (self.mailbox.width - resized.shape[1]) // 2
            preview[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            self.mailbox.publish(preview.tobytes())
            self.preview_due = stamp + 1 / 15
        with self.gate:
            if not self.accept_frames:
                return
            self.diagnostics.arrival()
            cost = self._frame_bytes(frame)
            if cost > RAW_FRAME_BYTE_LIMIT:
                self._failure_locked("单帧超过录像缓冲预算，录像已停止")
                self._raw_control("fail", self.session_id, self.session_error)
                return
            now = time.monotonic()
            compressed = self.encoded.snapshot()
            raw_pressure = max(self.raw_frames / self.frame_capacity, self.raw_bytes / RAW_FRAME_BYTE_LIMIT)
            compressed_pressure = max(
                compressed["resident_bytes"] / self.encoded.byte_limit,
                (compressed["packets"] + compressed["pending_packets"]) / self.encoded.packet_limit,
                compressed["data_items"] / self.encoded.packet_limit,
            )
            reason = "raw_pressure" if raw_pressure >= compressed_pressure else "compressed_pressure"
            if self.admission.select(raw_pressure, compressed_pressure, now):
                self.admission.record(True, stamp, now, reason)
                return
            acquired = self.frame_slots.acquire(False)
            if not acquired or self.raw_bytes + cost > RAW_FRAME_BYTE_LIMIT:
                if acquired:
                    self.frame_slots.release()
                self.admission.record(True, stamp, now, "raw_capacity")
                return
            self.raw_bytes += cost
            self.raw_frames += 1
            try:
                self.work.put_nowait(("frame", self.session_id, (frame, stamp, self.connection_epoch)))
            except queue.Full:
                self.raw_bytes -= cost
                self.raw_frames -= 1
                self.frame_slots.release()
                self._failure_locked("视频控制消息积压，录像不完整")
                self.fatal_error = self.session_error
                return
            self.admission.record(False, stamp, now, reason)

    def _frame_bytes(self, frame):
        return max(self.config.width * self.config.height * 3,
                   sum(plane.buffer_size for plane in frame.planes)) if frame is not None else 0

    def _raw_control(self, kind, identity, value=None):
        try:
            self.work.put_nowait((kind, identity, value))
        except queue.Full:
            self.fatal_error = "视频控制消息积压，录像不完整"

    def _failure_locked(self, detail):
        if not self.session_error:
            self.session_error = detail
        self.accept_frames = self.start_pending = False
        self.session_cancelled.set()
        self.encoded.wake()
        if not self.stop_requested:
            self.stop_requested = True
            self.drain_started = time.monotonic()
            self.post(EventKind.STOPPING, self.session_id, "录像异常，正在收尾")

    def _command(self, command):
        if command.generation != self.generation:
            return
        with self.gate:
            if command.kind == CommandKind.START:
                if self.session_id or not self.online or self.stopping:
                    return
                self.session_id = command.session_id
                self.session_error = ""
                self.session_cancelled = threading.Event()
                self.admission = FrameAdmission(self.session_id)
                self.stop_requested = False
                self.drain_started = None
                self.start_pending = True
                self._raw_control("start", self.session_id)
            elif command.kind == CommandKind.STOP and command.session_id == self.session_id:
                if self.stop_requested:
                    return
                self.stop_requested = True
                self.drain_started = time.monotonic()
                self.start_pending = self.accept_frames = False
                self._raw_control("stop", self.session_id)
            elif command.kind == CommandKind.SHUTDOWN and not self.stopping:
                self.start_pending = self.accept_frames = False
                self.stopping = True
                if self.drain_started is None:
                    self.drain_started = time.monotonic()
                self._raw_control("close", "")
                self.capture.stop()

    def _send_encoded(self, kind, identity, value=None, batch=None):
        if batch is None:
            self.encoded.put_nowait((kind, identity, value))
            return
        with self.gate:
            if identity != self.session_id:
                raise MediaQueueCancelled("录像会话已结束")
            cancelled = self.session_cancelled
        self.encoder_waiting_since = time.monotonic()
        try:
            self.encoded.put((kind, identity, value), size=batch.cost,
                             packets=len(batch.packets), cancelled=cancelled)
        finally:
            # Queue waiting is downstream work, not a stuck call into the codec.
            self.encoder_busy_since = time.monotonic()
            self.encoder_waiting_since = None

    def _encode_loop(self):
        encoder = None
        active_id = ""
        last_stamp = None

        def flush():
            if encoder is not None:
                tail = encoder.finish()
                if tail.packets:
                    self._send_encoded("tail", active_id, tail, tail)

        try:
            while True:
                try:
                    kind, identity, value = self.work.get(timeout=.2)
                except queue.Empty:
                    continue
                self.encoder_busy_since = time.monotonic()
                self.encoder_busy = True
                try:
                    if kind not in {"start", "close"} and (
                        identity != active_id or identity != self.session_id
                    ):
                        continue
                    if kind == "start":
                        active_id, last_stamp = identity, None
                        self.encode_diagnostics.begin(identity)
                        encoder = FrameEncoder(self.config, self.encode_diagnostics)
                        self._send_encoded("start", identity, encoder.parameters)
                    elif kind == "frame":
                        frame, stamp, epoch = value
                        if last_stamp is not None and stamp <= last_stamp:
                            raise ValueError("采集时间戳未递增，录像已停止")
                        if (encoder is None or (encoder.first_time is not None
                                and stamp - encoder.first_time >= self.config.segment_duration_seconds)):
                            flush()
                            encoder = FrameEncoder(self.config, self.encode_diagnostics)
                            self._send_encoded("segment", identity, encoder.parameters)
                        batch = encoder.write(frame, stamp)
                        self._send_encoded("frame", identity, (batch, stamp, epoch), batch)
                        last_stamp = stamp
                    elif kind == "gap":
                        flush()
                        encoder = None
                        self._send_encoded("gap", identity, value)
                    elif kind in {"stop", "fail", "close"}:
                        if active_id and active_id == self.session_id:
                            flush()
                            self._send_encoded(
                                "fail" if self.session_error else "stop", active_id,
                                self.session_error or None,
                            )
                        encoder = None
                        active_id = ""
                        if kind == "close":
                            self._send_encoded("close", "")
                            return
                except MediaQueueCancelled:
                    encoder = None
                    active_id = ""
                    if kind == "close":
                        self._send_encoded("close", "")
                        return
                except MediaQueueClosed:
                    return
                except Exception as exc:
                    detail = "压缩数据缓冲已满，录像不完整" if isinstance(exc, queue.Full) else str(exc)
                    logger.exception("Video encoding failed: session=%s operation=%s", active_id, kind)
                    with self.gate:
                        if active_id == self.session_id:
                            self._failure_locked(detail)
                    if isinstance(exc, queue.Full):
                        self.diagnostics.overflow(self.encoded.snapshot()["items"], 8192)
                    try:
                        self._send_encoded("fail", active_id, detail)
                        if kind == "close":
                            self._send_encoded("close", "")
                            return
                    except (queue.Full, MediaQueueClosed):
                        self.fatal_error = detail
                        return
                    encoder = None
                    active_id = ""
                finally:
                    if kind == "frame":
                        with self.gate:
                            self.raw_bytes -= self._frame_bytes(value[0])
                            self.raw_frames -= 1
                        self.frame_slots.release()
                    # Do not retain the previous payload after returning its budget.
                    value = frame = batch = None
                    self.encoder_busy = False
        finally:
            encoder = None
            self.encoded.close()

    def _write_loop(self):
        session = None
        acknowledged = False
        active_id = ""
        while True:
            try:
                item = self.encoded.get(timeout=0.2)
                kind, message_session_id, value = item.message
            except MediaQueueClosed:
                if session is not None:
                    self._finish_failed(session, active_id, "编码通道异常关闭，录像不完整")
                return
            except queue.Empty:
                if session is not None:
                    try:
                        self.writer_busy_since = time.monotonic()
                        self.writer_busy = True
                        with self.diagnostics.stage("maintenance"):
                            session.maintain()
                    except OSError as exc:
                        self._finish_failed(session, active_id, str(exc))
                        session = None
                        active_id = ""
                self.writer_busy = False
                continue
            self.writer_busy_since = time.monotonic()
            self.writer_busy = True
            try:
                # Encoder initialization can fail before the writer receives start.
                if kind == "fail" and session is None:
                    with self.gate:
                        current_session = message_session_id == self.session_id
                    if current_session:
                        self._finish_failed(None, message_session_id, value)
                        continue
                # The writer may still have old frames/control messages after a failure.
                # Discard them without touching the next session; finally returns media credits.
                if kind not in {"start", "close"} and (
                    session is None or message_session_id != active_id
                ):
                    if kind != "frame":
                        logger.info(
                            "Ignored stale video command: kind=%s session=%s detail=%s",
                            kind, message_session_id, value,
                        )
                    continue
                if kind == "start":
                    active_id = message_session_id
                    self.diagnostics.begin(active_id)
                    with self.diagnostics.stage("session_create"):
                        session = self.session_factory(self.config, active_id)
                    session.diagnostics = self.diagnostics
                    acknowledged = False
                    # Do not fill the frame queue while files/codecs are being opened.
                    with self.diagnostics.stage("prepare"):
                        session.prepare(value)
                    with self.gate:
                        if self.session_id == active_id and self.start_pending and not self.stopping:
                            self.start_pending = False
                            self.accept_frames = True
                elif kind == "frame":
                    frame, stamp, epoch = value
                    with self.diagnostics.stage("frame_total"):
                        written = session.write(frame, stamp)
                    with self.gate:
                        if written and not acknowledged and self.accept_frames and epoch == self.connection_epoch:
                            acknowledged = True
                            self.post(EventKind.STARTED, active_id, str(session.directory))
                elif kind == "segment":
                    session.next_segment(value)
                elif kind == "tail":
                    session.write_tail(value)
                elif kind == "gap":
                    with self.diagnostics.stage("gap"):
                        session.gap(value[0], disconnected_at=value[1])
                    acknowledged = False
                elif kind in {"stop", "close", "fail"}:
                    if kind == "fail":
                        raise OSError(value)
                    if session is not None:
                        with self.diagnostics.stage("finalize"):
                            session.finish()
                        detail = "已结束，存在中断" if session.summary["gaps"] else "已保存"
                        if not session.summary["frames"]:
                            detail = "已停止，未保存有效画面"
                        session = None
                        self._log_admission(active_id, final=True)
                        with self.gate:
                            if self.session_id == active_id:
                                self.session_id = ""
                        self.post(EventKind.COMPLETED, active_id, detail)
                        active_id = ""
                    if kind == "close":
                        return
            except Exception as exc:
                # Encoder/filesystem boundary: retain files and deliver a visible failure.
                logger.exception("Video writer failed (%s): %s", kind, active_id)
                self._finish_failed(session, active_id, str(exc))
                session = None
                active_id = ""
                if kind == "close":
                    return
            finally:
                self.encoded.release(item)
                item = value = frame = None
                self.writer_busy = False

    def _finish_failed(self, session, active_id, detail):
        with self.gate:
            if self.session_id == active_id:
                self._failure_locked(detail)
                detail = self.session_error
        logger.error("Video recording failed (%s): %s", active_id, detail)
        self._log_admission(active_id, final=True)
        if session is not None:
            try:
                with self.diagnostics.stage("failure_finalize"):
                    session.finish(error=detail)
            except Exception:
                logger.exception("Video failure finalization: session=%s; first error retained", active_id)
        with self.gate:
            if self.session_id != active_id:
                return
            self.start_pending = False
            self.accept_frames = False
            self.session_id = ""
        self.post(EventKind.RECORDING_FAILED, active_id, detail[:2000])

    def _report_diagnostics(self):
        # Logging stays on the control thread, outside the media ownership locks.
        self._log_admission(self.session_id)
        compressed = self.encoded.snapshot()
        report = self.diagnostics.take_report(
            self.work.qsize(), self.frame_capacity,
            backlog=(compressed["bytes"] >= self.encoded.byte_limit * .8
                     or compressed["oldest_seconds"] >= 5),
        )
        if report is not None:
            report["thread"] = "writer"
            report["compressed"] = compressed
            logger.warning("Video diagnostic: %s", report)
        report = self.encode_diagnostics.take_report(self.work.qsize(), self.frame_capacity)
        if report is not None:
            report["thread"] = "encoder"
            logger.warning("Video encoder diagnostic: %s", report)

    def _log_admission(self, identity, *, final=False):
        with self.gate:
            if not identity or self.admission.session_id != identity:
                return
            report = self.admission.report(time.monotonic(), final=final)
        if report is not None:
            logger.log(logging.WARNING if report["dropped"] else logging.INFO,
                       "Video frame admission: %s", report)

    def _check_progress(self, now):
        with self.gate:
            drain_started, session_id, error = self.drain_started, self.session_id, self.session_error
        for busy, since, diagnostics, name in (
            (self.encoder_busy and self.encoder_waiting_since is None,
             self.encoder_busy_since, self.encode_diagnostics, "编码"),
            (self.writer_busy, self.writer_busy_since, self.diagnostics, "封装或写盘"),
        ):
            if busy and now - max(since, diagnostics.progress_snapshot()[1]) > self.no_progress_timeout:
                raise TimeoutError(error or f"录像{name}长时间无进展，录像不完整")
        waiting = self.encoder_waiting_since
        if waiting is not None and now - max(waiting, self.diagnostics.progress_snapshot()[1]) > self.no_progress_timeout:
            raise TimeoutError(error or "录像写盘等位长时间无进展，录像不完整")
        if drain_started is not None and session_id and now - drain_started > self.drain_timeout:
            raise TimeoutError(error or "录像收尾超时，部分数据未保存")

    def run(self):
        self.writer.start()
        self.encoder_thread.start()
        self.capture.start()
        heartbeat_due = 0.0
        try:
            while True:
                now = time.monotonic()
                if self.channel.poll(0.01):
                    self._command(self.channel.recv())
                for _ in range(32):
                    try:
                        kind, session_id, detail = self.events.get_nowait()
                    except queue.Empty:
                        break
                    self.sequence += 1
                    self.channel.send(Event(kind, self.generation, self.sequence, now, session_id, detail=detail))
                if self.stopping and not self.writer.is_alive() and not self.encoder_thread.is_alive():
                    self.sequence += 1
                    self.channel.send(Event(EventKind.CLOSED, self.generation, self.sequence, now))
                    return
                if self.fatal_error:
                    raise RuntimeError(self.fatal_error)
                if not self.writer.is_alive() and not self.stopping:
                    raise RuntimeError("录像写入线程已异常退出")
                frame_timeout = max(3, 3 * self.config.fps_den / self.config.fps_num)
                if self.online and now - self.capture.last_progress > frame_timeout:
                    self.on_connection(False, "摄像头暂无新画面，已标记录像中断")
                if not self.stopping and now - self.capture.last_progress > 12:
                    raise TimeoutError("摄像头驱动长时间无响应")
                self._check_progress(now)
                if now >= heartbeat_due:
                    self._report_diagnostics()
                    self.sequence += 1
                    self.channel.send(Event(
                        EventKind.HEARTBEAT, self.generation, self.sequence, now,
                        progress=self.diagnostics.progress_snapshot()[0] + self.encode_diagnostics.progress_snapshot()[0],
                    ))
                    heartbeat_due = now + .25
                parent = multiprocessing.parent_process()
                if parent is not None and not parent.is_alive() and not self.stopping:
                    from base.video.models import Command
                    self._command(Command(CommandKind.SHUTDOWN, self.generation, "parent-exit"))
        finally:
            self._report_diagnostics()
            self.capture.stop()
            with self.gate:
                self.start_pending = False
                self.accept_frames = False
                if self.writer.is_alive() and not self.stopping:
                    self._raw_control("close", "")
            self._log_admission(self.session_id, final=True)
            self.encoder_thread.join(timeout=2)
            self.encoded.close()
            self.writer.join(timeout=2)


def usb_video_worker(channel, mailbox, generation, config):
    runtime = None
    try:
        if not config.enabled:
            channel.send(Event(EventKind.CLOSED, generation, 1, time.monotonic()))
            return
        from base.log_manager import LogManager
        LogManager.set_log_handler("core")
        with RecordingRootLease(config.recording_root):
            runtime = VideoRuntime(channel, mailbox, generation, config)
            try:
                runtime.run()
            except Exception as exc:
                # Report the cause before forced exit of a stuck native owner.
                runtime.sequence += 1
                channel.send(Event(EventKind.FAILED, generation, runtime.sequence,
                                   time.monotonic(), detail=str(exc)[:2000]))
                raise
            finally:
                if runtime.writer.is_alive() or runtime.encoder_thread.is_alive():
                    # A stuck native writer must die before the root lease is released.
                    os._exit(24)
    except Exception as exc:
        logger.exception("Video process failed")
        sequence = runtime.sequence if runtime is not None else 0
        channel.send(Event(EventKind.FAILED, generation, sequence + 1, time.monotonic(), detail=str(exc)[:2000]))
        channel.send(Event(EventKind.CLOSED, generation, sequence + 2, time.monotonic()))
    finally:
        channel.close()
