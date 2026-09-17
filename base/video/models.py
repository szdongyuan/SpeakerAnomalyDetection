"""Video control messages and deterministic state; no Qt or camera dependencies."""

from dataclasses import dataclass, replace
from enum import Enum
import math


class CommandKind(str, Enum):
    START = "start_recording"
    STOP = "stop_recording"
    SHUTDOWN = "shutdown"


class EventKind(str, Enum):
    READY = "ready"
    OFFLINE = "offline"
    RECORDING_FAILED = "recording_failed"
    STOPPING = "recording_stopping"
    ACCEPTED = "accepted"
    STARTED = "recording_started"
    RECOVERING = "recording_recovering"
    COMPLETED = "recording_completed"
    FAILED = "failed"
    CLOSED = "closed"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    generation: int
    command_id: str
    session_id: str = ""
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.kind, CommandKind) or self.schema_version != 1:
            raise ValueError("invalid command kind/version")
        if type(self.generation) is not int or self.generation < 1 or not self.command_id:
            raise ValueError("command identity is required")
        if self.kind in {CommandKind.START, CommandKind.STOP} and not self.session_id:
            raise ValueError("recording commands require session_id")


@dataclass(frozen=True)
class Event:
    kind: EventKind
    generation: int
    sequence: int
    monotonic_time: float
    session_id: str = ""
    command_id: str = ""
    detail: str = ""
    schema_version: int = 1
    progress: int = 0

    def __post_init__(self):
        if not isinstance(self.kind, EventKind) or self.schema_version != 1:
            raise ValueError("invalid event kind/version")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid generation")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("invalid sequence")
        if type(self.progress) is not int or self.progress < 0:
            raise ValueError("invalid media progress counter")
        if not math.isfinite(self.monotonic_time) or self.monotonic_time < 0:
            raise ValueError("invalid monotonic timestamp")
        if self.kind in {EventKind.STARTED, EventKind.RECOVERING, EventKind.COMPLETED}:
            if not self.session_id:
                raise ValueError("recording events require session_id")


@dataclass(frozen=True)
class VideoStatus:
    connection: str = "connecting"
    recording: str = "idle"
    session_id: str = ""
    record_intent: bool = False
    had_gap: bool = False
    started_at: float | None = None
    ended_at: float | None = None
    connection_detail: str = ""
    recording_detail: str = ""
    directory: str = ""

    @property
    def detail(self):
        """Read-only summary for existing consumers; each source owns its own detail."""
        if self.recording == "failed":
            return self.recording_detail
        return self.connection_detail if self.connection != "ready" else self.recording_detail

    def elapsed(self, now):
        if self.started_at is None:
            return 0
        end = self.ended_at if self.ended_at is not None else now
        return max(0, int(end - self.started_at))


class VideoState:
    def __init__(self, generation):
        self.generation = generation
        self.sequence = 0
        self.status = VideoStatus()

    def request_start(self, session_id):
        status = self.status
        if status.connection != "ready" or status.record_intent or status.recording == "stopping":
            return False
        self.status = VideoStatus(
            connection="ready", recording="starting", session_id=session_id, record_intent=True,
        )
        return True

    def request_stop(self):
        if not self.status.record_intent:
            return False
        # Revoke recovery intent BEFORE sending stop to the child process.
        self.status = replace(self.status, record_intent=False, recording="stopping")
        return True

    def fail(self, detail, now):
        active = self.status.recording in {"starting", "recording", "recovering", "stopping"}
        self.status = replace(
            self.status, connection="unavailable", record_intent=False, connection_detail=detail,
            recording_detail=detail if active else self.status.recording_detail,
            recording="failed" if active else self.status.recording,
            ended_at=now if active else self.status.ended_at,
        )

    def apply(self, event):
        if event.generation != self.generation or event.sequence <= self.sequence:
            return False
        self.sequence = event.sequence
        status = self.status
        kind = event.kind
        if event.session_id and event.session_id != status.session_id:
            return False
        if kind == EventKind.READY:
            self.status = replace(status, connection="ready", connection_detail="")
        elif kind == EventKind.OFFLINE:
            self.status = replace(status, connection="reconnecting", connection_detail=event.detail)
        elif kind == EventKind.STOPPING:
            if status.recording not in {"starting", "recording", "recovering", "stopping"}:
                return False
            self.status = replace(status, recording="stopping", record_intent=False, recording_detail=event.detail)
        elif kind == EventKind.STARTED:
            if not status.record_intent or status.recording not in {"starting", "recovering"}:
                return False
            self.status = replace(
                status, connection="ready", recording="recording", connection_detail="", recording_detail="",
                started_at=status.started_at if status.started_at is not None else event.monotonic_time,
                directory=event.detail or status.directory,
            )
        elif kind == EventKind.RECOVERING:
            if not status.record_intent or status.recording not in {"starting", "recording", "recovering"}:
                return False
            self.status = replace(
                status, connection="reconnecting", recording="recovering",
                had_gap=True, connection_detail=event.detail,
            )
        elif kind == EventKind.COMPLETED:
            if status.recording not in {"starting", "recording", "recovering", "stopping"}:
                return False
            self.status = replace(
                status, recording="interrupted" if status.had_gap else "completed", record_intent=False,
                ended_at=event.monotonic_time, recording_detail=event.detail,
            )
        elif kind == EventKind.FAILED:
            self.fail(event.detail, event.monotonic_time)
        elif kind == EventKind.RECORDING_FAILED:
            if status.recording not in {"starting", "recording", "recovering", "stopping"}:
                return False
            self.status = replace(
                status, recording="failed", record_intent=False,
                recording_detail=event.detail, ended_at=event.monotonic_time,
            )
        elif kind == EventKind.CLOSED:
            active = status.recording in {"starting", "recording", "recovering", "stopping"}
            self.status = replace(
                status, connection="closed", record_intent=False,
                recording="interrupted" if active else status.recording,
                ended_at=event.monotonic_time if active else status.ended_at,
            )
        return True
