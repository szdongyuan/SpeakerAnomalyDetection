"""Explicit simulation backend for tests/demo; never opens or saves real video."""

from dataclasses import dataclass
import os
import time

from base.video.models import CommandKind, Event, EventKind


@dataclass(frozen=True)
class SimulationOptions:
    disconnect_after: float | None = None
    reconnect_after: float = 0.5
    crash_after: float | None = None
    stall_after: float | None = None
    preview_enabled: bool = True
    ready_delay: float = 0.0


def simulated_video_worker(channel, mailbox, generation, options):
    """Spawn target, importable without Qt. Fault options are NOT saved settings."""
    sequence = 0
    recovery_due = None
    disconnected_once = False
    frame_number = 0
    next_frame = 0.0
    next_heartbeat = 0.0
    preview_enabled, preview_revision = options.preview_enabled, 0
    ready_due = time.monotonic() + options.ready_delay
    ready = False

    def emit(kind, **fields):
        nonlocal sequence
        sequence += 1
        channel.send(Event(kind, generation, sequence, time.monotonic(), **fields))

    try:
        while True:
            if channel.poll(0.01):
                command = channel.recv()
                if command.generation != generation:
                    continue
                if command.kind == CommandKind.SET_PREVIEW:
                    if command.preview_revision > preview_revision:
                        preview_enabled, preview_revision = command.preview_enabled, command.preview_revision
                        emit(EventKind.PREVIEW_CHANGED, command_id=command.command_id,
                             preview_enabled=preview_enabled, preview_revision=preview_revision)
                elif command.kind == CommandKind.SHUTDOWN:
                    emit(EventKind.CLOSED)
                    return

            now = time.monotonic()
            if not ready and recovery_due is None and now >= ready_due:
                ready = True
                emit(EventKind.READY)
            if now >= ready_due:
                elapsed = now - ready_due
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
                    ready = False
                    recovery_due = now + options.reconnect_after
                    emit(EventKind.OFFLINE, detail="模拟摄像头断开")
            if recovery_due is not None and now >= recovery_due:
                recovery_due = None
                ready = True
                emit(EventKind.READY)
            if ready and preview_enabled and now >= next_frame and recovery_due is None:
                frame_number += 1
                # Small RGB test pattern with a moving bar; constant memory footprint.
                column = frame_number % mailbox.width
                row = b"\x28\x40\x58" * column + b"\x48\xbc\xda" + b"\x28\x40\x58" * (mailbox.width - column - 1)
                mailbox.publish(row * mailbox.height, preview_revision)
                next_frame = now + 1 / 15
            if now >= next_heartbeat:
                emit(EventKind.HEARTBEAT)
                next_heartbeat = now + 0.25
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return
    finally:
        channel.close()
