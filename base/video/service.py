"""Nonblocking GUI-facing supervisor for one independently owned video process."""

import logging
import multiprocessing
import queue
import threading
import time
from dataclasses import replace
from uuid import uuid4

from base.log_exit import ProcessLogDrain, run_with_log_drain
from base.video.preview import PreviewMailbox
from base.video.models import Command, CommandKind, Event, EventKind, VideoState


logger = logging.getLogger("core.video")


class VideoService:
    def __init__(
        self, *, worker_target, worker_options, generation=1,
        heartbeat_timeout=3.0, command_timeout=5.0, shutdown_timeout=3.0,
        drain_timeout=None,
        initial_connection="connecting",
    ):
        # No default fake backend: callers must explicitly select a worker.
        self._worker_target = worker_target
        self._worker_options = worker_options
        self._state = VideoState(generation)
        self._state.status = replace(self._state.status, connection=initial_connection)
        self._lock = threading.Lock()
        self._commands = queue.Queue(maxsize=16)
        self._closed = threading.Event()
        self._closing = False
        self._started = False
        self._mailbox = None
        self._thread = None
        self._unreaped = None
        self._heartbeat_timeout = heartbeat_timeout
        self._command_timeout = command_timeout
        self._shutdown_timeout = shutdown_timeout
        self._drain_timeout = command_timeout if drain_timeout is None else drain_timeout
        self.process_id = None
        self.forced_termination = False

    @property
    def status(self):
        with self._lock:
            return self._state.status

    @property
    def is_closed(self):
        return self._closed.is_set()

    def start(self):
        with self._lock:
            if self._started or self._closing:
                return False
            self._started = True
            self._thread = threading.Thread(target=self._run, name="VideoSupervisor", daemon=True)
            self._thread.start()
        return True

    def start_recording(self):
        with self._lock:
            if self._closing or not self._state.request_start(uuid4().hex):
                return False
            return self._enqueue_locked(CommandKind.START)

    def stop_recording(self):
        with self._lock:
            if self._closing or not self._state.request_stop():
                return False
            return self._enqueue_locked(CommandKind.STOP)

    def shutdown(self):
        """Request asynchronous shutdown. Qt callers must NOT wait/join here."""
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._state.request_stop()
            if not self._started:
                self._state.status = replace(self._state.status, connection="closed")
                self._closed.set()
                return
            self._enqueue_locked(CommandKind.SHUTDOWN)

    def wait_closed(self, timeout=8):
        """For tests/non-GUI teardown only."""
        closed = self._closed.wait(timeout)
        if closed:
            with self._lock:
                if self._unreaped is not None:
                    process, drain = self._unreaped
                    if not process.is_alive():
                        process.join(timeout=0)
                        process.close()
                        drain.close()
                        self._unreaped = None
        return closed

    def latest_preview(self, after_sequence=0):
        mailbox = self._mailbox
        if mailbox is None or self.status.connection != "ready":
            return None
        return mailbox.latest(after_sequence)

    def _enqueue_locked(self, kind):
        command = Command(kind, self._state.generation, uuid4().hex, self._state.status.session_id)
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            self._state.fail("视频控制队列已满", time.monotonic())
            return False
        return True

    def _fail(self, detail):
        logger.error("Video supervisor failure: %s", detail)
        with self._lock:
            self._state.fail(detail, time.monotonic())

    def _run(self):
        process = channel = child_channel = log_drain = None
        received_closed = False
        try:
            context = multiprocessing.get_context("spawn")
            self._mailbox = PreviewMailbox(context)
            channel, child_channel = context.Pipe(duplex=True)
            log_drain = ProcessLogDrain.create(context)
            process = context.Process(
                target=run_with_log_drain, name="VideoWorker",
                args=(self._worker_target,
                      (child_channel, self._mailbox, self._state.generation, self._worker_options),
                      log_drain.child_endpoint),
            )
            process.start()
            self.process_id = process.pid
            child_channel.close()
            last_event = time.monotonic()
            progress_token = 0
            last_progress = last_event
            shutdown_deadline = None
            shutdown_since = None
            closed_at = None
            pending_since = last_event
            last_phase = ""
            while True:
                now = time.monotonic()
                # Bounded batches prevent either IPC direction starving the other.
                for _ in range(16):
                    if received_closed:
                        break
                    try:
                        if not channel.poll():
                            break
                        event = channel.recv()
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        break
                    if not isinstance(event, Event):
                        raise ValueError("unexpected video protocol message")
                    with self._lock:
                        accepted = self._state.apply(event)
                    if accepted:
                        last_event = now
                        if event.progress > progress_token:
                            progress_token, last_progress = event.progress, now
                        received_closed |= event.kind == EventKind.CLOSED
                if not process.is_alive():
                    if not received_closed:
                        self._fail(f"视频进程意外退出（exitcode={process.exitcode}）")
                    break
                if received_closed:
                    if closed_at is None:
                        closed_at = now
                    if now - closed_at >= self._shutdown_timeout:
                        self._fail("视频进程报告关闭后未退出")
                        break
                    process.join(timeout=0.1)
                    continue
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    command = None
                if command is not None:
                    channel.send(command)
                    if command.kind == CommandKind.SHUTDOWN:
                        shutdown_since = now
                        shutdown_deadline = now + self._shutdown_timeout
                phase = self.status.recording
                if phase != last_phase:
                    last_phase, pending_since = phase, now
                if now - last_event > self._heartbeat_timeout:
                    self._fail("视频进程响应超时")
                    break
                if phase == "starting" and now - pending_since > self._command_timeout:
                    self._fail("视频启停响应超时")
                    break
                if phase == "stopping":
                    if now - max(pending_since, last_progress) > self._drain_timeout:
                        self._fail("视频启停响应超时：录像收尾未取得进展")
                        break
                    if now - pending_since > self._shutdown_timeout:
                        self._fail("录像收尾超过总时限，部分数据未保存")
                        break
                if shutdown_since is not None and now - max(shutdown_since, last_progress) > self._drain_timeout:
                    self._fail("视频进程退出时长时间无进展")
                    break
                if shutdown_deadline is not None and now >= shutdown_deadline:
                    self._fail("视频进程退出超时")
                    break
                time.sleep(0.01)
        except Exception as exc:
            # External process/IPC boundary: expose failure, don't lose the thread silently.
            logger.exception("Video supervisor failed")
            self._fail(f"视频服务异常：{type(exc).__name__}: {exc}")
        finally:
            if process is not None and process.pid is not None:
                if process.is_alive():
                    log_drain.begin("video supervisor retirement")
                    self._report_log_drain(process.pid, log_drain, log_drain.wait(is_alive=process.is_alive))
                    if process.is_alive():
                        self.forced_termination = True
                        process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                else:
                    self._report_log_drain(process.pid, log_drain, log_drain.poll(already_dead=True))
                    process.join(timeout=1)
                if not process.is_alive():
                    process.close()
                    log_drain.close()
                else:
                    # Keep both owners until a later non-GUI wait confirms death.
                    self._unreaped = (process, log_drain)
                    logger.error("Video process death unconfirmed pid=%s", process.pid)
            elif log_drain is not None:
                log_drain.close()
            if channel is not None:
                channel.close()
            if child_channel is not None:
                child_channel.close()
            self._mailbox = None
            self._closed.set()

    @staticmethod
    def _report_log_drain(pid, drain, result):
        if (result.status in ("timeout", "drained-with-errors")
                or (result.status == "already-dead" and result.detail is not None)):
            logger.error(
                "Video log drain pid=%s reason=%s status=%s pending=%s stats=%s detail=%s",
                pid, drain.reason or "self-exit", result.status,
                "unknown" if result.stats is None else result.stats["pending"],
                result.stats, result.detail)
