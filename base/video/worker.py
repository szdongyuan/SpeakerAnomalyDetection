"""Explicit simulation backend for tests/demo; never opens or saves real video."""

from dataclasses import dataclass
import os
import time

from base.video.models import CommandKind, Event, EventKind


@dataclass(frozen=True)
class SimulationOptions:
    start_delay: float = 0.1
    stop_delay: float = 0.15
    fail_start: bool = False
    disconnect_after: float | None = None
    reconnect_after: float = 0.5
    crash_after: float | None = None
    stall_after: float | None = None


def simulated_video_worker(channel, mailbox, generation, options):
    """Spawn target, importable without Qt. Fault options are NOT saved settings."""
    sequence = 0
    session_id = ""
    start_command = ""
    started_at = None
    start_due = None
    stop_due = None
    recovery_due = None
    disconnected_once = False
    frame_number = 0
    next_frame = 0.0
    next_heartbeat = 0.0

    def emit(kind, session="", command_id="", detail=""):
        nonlocal sequence
        sequence += 1
        channel.send(Event(kind, generation, sequence, time.monotonic(), session, command_id, detail))

    try:
        emit(EventKind.READY)
        while True:
            if channel.poll(0.01):
                command = channel.recv()
                if command.generation != generation:
                    continue
                emit(EventKind.ACCEPTED, command_id=command.command_id)
                if command.kind == CommandKind.START and not session_id:
                    session_id = command.session_id
                    start_command = command.command_id
                    start_due = time.monotonic() + options.start_delay
                    disconnected_once = False
                elif command.kind == CommandKind.STOP and command.session_id == session_id:
                    start_due = None
                    stop_due = time.monotonic() + options.stop_delay
                elif command.kind == CommandKind.SHUTDOWN:
                    if session_id:
                        emit(EventKind.COMPLETED, session_id, detail="模拟会话结束，未生成视频文件")
                    emit(EventKind.CLOSED)
                    return

            now = time.monotonic()
            if start_due is not None and now >= start_due:
                start_due = None
                if options.fail_start:
                    emit(EventKind.FAILED, session_id, start_command, "模拟录像启动失败")
                    session_id = ""
                else:
                    started_at = now
                    emit(EventKind.STARTED, session_id, start_command)
            if stop_due is not None and now >= stop_due:
                emit(EventKind.COMPLETED, session_id, detail="模拟会话结束，未生成视频文件")
                session_id = ""
                started_at = stop_due = None
            if started_at is not None and stop_due is None:
                elapsed = now - started_at
                if options.crash_after is not None and elapsed >= options.crash_after:
                    os._exit(23)  # Fault injection: intentionally bypass normal cleanup.
                if options.stall_after is not None and elapsed >= options.stall_after:
                    while True:
                        time.sleep(1)  # Fault injection: parent must terminate a hung backend.
                if (
                    options.disconnect_after is not None and not disconnected_once
                    and elapsed >= options.disconnect_after
                ):
                    disconnected_once = True
                    recovery_due = now + options.reconnect_after
                    emit(EventKind.RECOVERING, session_id, detail="模拟摄像头断开，等待恢复")
            if recovery_due is not None and now >= recovery_due:
                recovery_due = None
                emit(EventKind.READY)
                if session_id and started_at is not None and stop_due is None:
                    emit(EventKind.STARTED, session_id)
            if now >= next_frame and recovery_due is None:
                frame_number += 1
                # Small RGB test pattern with a moving bar; constant memory footprint.
                column = frame_number % mailbox.width
                row = b"\x28\x40\x58" * column + b"\x48\xbc\xda" + b"\x28\x40\x58" * (mailbox.width - column - 1)
                mailbox.publish(row * mailbox.height)
                next_frame = now + 1 / 15
            if now >= next_heartbeat:
                emit(EventKind.HEARTBEAT)
                next_heartbeat = now + 0.25
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return
    finally:
        channel.close()
