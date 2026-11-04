"""Video control messages and deterministic state; no Qt or camera dependencies."""

from dataclasses import dataclass, replace
from enum import Enum
import math


class CommandKind(str, Enum):
    SHUTDOWN = "shutdown"
    SET_PREVIEW = "set_preview"


class EventKind(str, Enum):
    READY = "ready"
    OFFLINE = "offline"
    FAILED = "failed"
    CLOSED = "closed"
    HEARTBEAT = "heartbeat"
    PREVIEW_CHANGED = "preview_changed"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    generation: int
    command_id: str
    schema_version: int = 1
    preview_enabled: bool | None = None
    preview_revision: int = 0

    def __post_init__(self):
        if not isinstance(self.kind, CommandKind) or self.schema_version != 1:
            raise ValueError("invalid command kind/version")
        if type(self.generation) is not int or self.generation < 1 or not self.command_id:
            raise ValueError("command identity is required")
        if self.kind == CommandKind.SET_PREVIEW:
            if (type(self.preview_enabled) is not bool
                    or type(self.preview_revision) is not int or self.preview_revision < 1):
                raise ValueError("invalid preview command")


@dataclass(frozen=True)
class Event:
    kind: EventKind
    generation: int
    sequence: int
    monotonic_time: float
    command_id: str = ""
    detail: str = ""
    schema_version: int = 1
    preview_enabled: bool | None = None
    preview_revision: int = 0

    def __post_init__(self):
        if not isinstance(self.kind, EventKind) or self.schema_version != 1:
            raise ValueError("invalid event kind/version")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid generation")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("invalid sequence")
        if not math.isfinite(self.monotonic_time) or self.monotonic_time < 0:
            raise ValueError("invalid monotonic timestamp")
        if self.kind == EventKind.PREVIEW_CHANGED:
            if (not self.command_id or type(self.preview_enabled) is not bool
                    or type(self.preview_revision) is not int or self.preview_revision < 1):
                raise ValueError("invalid preview acknowledgement")


@dataclass(frozen=True)
class VideoStatus:
    connection: str = "connecting"
    connection_detail: str = ""
    preview_enabled: bool = True
    preview_revision: int = 0


class VideoState:
    def __init__(self, generation):
        self.generation = generation
        self.sequence = 0
        self.status = VideoStatus()

    def fail(self, detail):
        self.status = replace(self.status, connection="unavailable", connection_detail=detail)

    def apply(self, event):
        if event.generation != self.generation or event.sequence <= self.sequence:
            return False
        self.sequence = event.sequence
        status = self.status
        kind = event.kind
        if kind == EventKind.PREVIEW_CHANGED:
            if event.preview_revision <= status.preview_revision:
                return False
            self.status = replace(status, preview_enabled=event.preview_enabled,
                                  preview_revision=event.preview_revision)
        elif kind == EventKind.READY:
            self.status = replace(status, connection="ready", connection_detail="")
        elif kind == EventKind.OFFLINE:
            self.status = replace(status, connection="reconnecting", connection_detail=event.detail)
        elif kind == EventKind.FAILED:
            self.fail(event.detail)
        elif kind == EventKind.CLOSED and status.connection != "unavailable":
            self.status = replace(status, connection="closed")
        return True
