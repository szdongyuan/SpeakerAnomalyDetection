"""Temporary adapter from canonical recording admission to the legacy recorder."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4
from weakref import ref

from PyQt5.QtCore import QObject, Qt, pyqtSlot

from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    RecordingCompleted,
    RecordingFailed,
    RecordingStarted,
)


def legacy_recording_session_snapshot(command: Any, configuration: Any) -> dict:
    """Build the immutable admission input shared by starts and replays."""
    snapshot = {
        "command_id": command.command_id,
        "source": command.source,
        "record_id": getattr(command, "record_id", command.command_id),
        "label": getattr(command, "label", "not_labeled"),
        "skip_sn_regex_validation": bool(
            getattr(command, "skip_sn_regex_validation", False)
        ),
        "configuration": configuration,
    }
    configuration_generation = getattr(command, "configuration_generation", None)
    if type(configuration_generation) is int:
        snapshot["configuration_generation"] = configuration_generation
    return snapshot


@dataclass(frozen=True, slots=True)
class _LegacyRecordingToken:
    token_id: str
    session_id: str
    workflow_generation: int
    bridge_generation: int
    session_snapshot: Any


class LegacyRecordingTerminalPort:
    """Completion boundary bound to exactly one admitted recording session."""

    __slots__ = ("_bridge_ref", "_token")

    def __init__(
        self, bridge: "LegacyRecordingAdmissionBridge", token: _LegacyRecordingToken
    ) -> None:
        self._bridge_ref = ref(bridge)
        self._token = token

    @property
    def session_id(self) -> str:
        return self._token.session_id

    @property
    def workflow_generation(self) -> int:
        return self._token.workflow_generation

    @property
    def session_snapshot(self) -> Any:
        return self._token.session_snapshot

    def recording_completed(self, *, sample_count: int, result_snapshot: Any) -> bool:
        bridge = self._bridge_ref()
        if bridge is None:
            return False
        return bridge._recording_completed(
            self._token,
            sample_count=sample_count,
            result_snapshot=result_snapshot,
        )

    def recording_failed(self, reason: Any) -> bool:
        bridge = self._bridge_ref()
        if bridge is None:
            return False
        return bridge._recording_failed(self._token, reason)


class _QueuedAdmissionGuard(QObject):
    def __init__(self, owner: "LegacyRecordingAdmissionBridge") -> None:
        super().__init__(owner)
        self._owner_ref = ref(owner)

    @pyqtSlot(object)
    def deliver(self, command: Any) -> None:
        owner = self._owner_ref()
        if owner is not None and owner._accept_queued_delivery:
            owner.handle_begin_recording(command)


class LegacyRecordingAdmissionBridge(QObject):
    """Translate Workflow admission into the current legacy recorder boundary."""

    def __init__(
        self,
        bus: SequenceEventBus,
        start_recording: Callable[
            [BeginRecordingRequested, LegacyRecordingTerminalPort], bool
        ],
        *,
        workflow_generation_provider: Callable[[], int] | None = None,
        recent_identity_limit: int = 256,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if type(recent_identity_limit) is not int or recent_identity_limit < 1:
            raise ValueError("recent_identity_limit must be a positive integer")
        self.bus = bus
        self.start_recording = start_recording
        self.workflow_generation_provider = workflow_generation_provider
        self.logger = logger
        self._lock = RLock()
        self._active = True
        self._bridge_generation = 0
        self._active_admission: BeginRecordingRequested | None = None
        self._active_token: _LegacyRecordingToken | None = None
        self._starting = False
        self._terminal_claimed = False
        self._pending_terminal: RecordingCompleted | RecordingFailed | None = None
        self._recent_identity_limit = recent_identity_limit
        self._recent_identities: set[tuple[str, str]] = set()
        self._recent_identity_order: deque[tuple[str, str]] = deque()
        self._accept_queued_delivery = True
        self._guard = _QueuedAdmissionGuard(self)
        bus.commands.begin_recording_requested.connect(
            self._guard.deliver, Qt.QueuedConnection
        )

    @property
    def active_admission(self) -> BeginRecordingRequested | None:
        with self._lock:
            return self._active_admission

    @property
    def recent_identity_count(self) -> int:
        with self._lock:
            return len(self._recent_identities)

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self.logger, level, None)
        if callable(callback):
            callback(message)

    def disconnect(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._bridge_generation += 1
            self._accept_queued_delivery = False
            self._clear_active_locked()
        try:
            self.bus.commands.begin_recording_requested.disconnect(self._guard.deliver)
        except (RuntimeError, TypeError):
            pass

    def _workflow_generation(self) -> int | None:
        provider = self.workflow_generation_provider
        if provider is None:
            return None
        generation = provider()
        return generation if type(generation) is int else None

    @staticmethod
    def _admission_generation(command: BeginRecordingRequested) -> int | None:
        snapshot = command.session_snapshot
        if not isinstance(snapshot, Mapping):
            return None
        generation = snapshot.get("workflow_generation")
        return generation if type(generation) is int else None

    def _remember_identity_locked(self, identity: tuple[str, str]) -> None:
        if identity in self._recent_identities:
            return
        self._recent_identities.add(identity)
        self._recent_identity_order.append(identity)
        while len(self._recent_identity_order) > self._recent_identity_limit:
            retired = self._recent_identity_order.popleft()
            self._recent_identities.discard(retired)

    def _clear_active_locked(self) -> None:
        self._active_admission = None
        self._active_token = None
        self._starting = False
        self._terminal_claimed = False
        self._pending_terminal = None

    def _token_is_current_locked(self, token: _LegacyRecordingToken) -> bool:
        if not self._active or token != self._active_token:
            return False
        if token.bridge_generation != self._bridge_generation:
            return False
        if self.workflow_generation_provider is None:
            return True
        current_generation = self._workflow_generation()
        return current_generation == token.workflow_generation

    def _finish_terminal_locked(
        self, event: RecordingCompleted | RecordingFailed
    ) -> None:
        admission = self._active_admission
        if admission is not None:
            self._remember_identity_locked((admission.command_id, admission.session_id))
        self._clear_active_locked()
        if type(event) is RecordingCompleted:
            self.bus.events.recording_completed.emit(event)
        else:
            self.bus.events.recording_failed.emit(event)

    @pyqtSlot(object)
    def handle_begin_recording(self, command: BeginRecordingRequested) -> bool:
        if type(command) is not BeginRecordingRequested:
            return False
        identity = (command.command_id, command.session_id)
        with self._lock:
            if not self._active or not self._accept_queued_delivery:
                return False
            admission_generation = self._admission_generation(command)
            current_generation = self._workflow_generation()
            if (
                admission_generation is None
                or (
                    self.workflow_generation_provider is not None
                    and admission_generation != current_generation
                )
            ):
                self._log("debug", f"忽略过期录音准入: {command.session_id}")
                self.bus.events.recording_failed.emit(
                    RecordingFailed(
                        command.session_id, "stale workflow generation"
                    )
                )
                return False
            if identity in self._recent_identities or self._active_admission is not None:
                self._log("debug", f"忽略重复录音准入: {command.session_id}")
                return False
            token = _LegacyRecordingToken(
                uuid4().hex,
                command.session_id,
                admission_generation,
                self._bridge_generation,
                command.session_snapshot,
            )
            terminal = LegacyRecordingTerminalPort(self, token)
            self._active_admission = command
            self._active_token = token
            self._starting = True
            self._terminal_claimed = False
            self._pending_terminal = None

        start_error: Exception | None = None
        try:
            started = bool(self.start_recording(command, terminal))
        except Exception as error:
            self._log("error", f"旧录音入口启动失败: {error}")
            start_error = error
            started = False

        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            pending = self._pending_terminal
            if pending is not None:
                if type(pending) is RecordingCompleted:
                    self.bus.events.recording_started.emit(
                        RecordingStarted(command.session_id, command.session_snapshot)
                    )
                self._finish_terminal_locked(pending)
                return type(pending) is RecordingCompleted
            if not started:
                reason = (
                    str(start_error) or "legacy recording start failed"
                    if start_error is not None
                    else "legacy recording did not start"
                )
                self._terminal_claimed = True
                self._finish_terminal_locked(RecordingFailed(command.session_id, reason))
                return False

            self.bus.events.recording_started.emit(
                RecordingStarted(command.session_id, command.session_snapshot)
            )
            self._starting = False
            pending = self._pending_terminal
            if pending is not None:
                self._finish_terminal_locked(pending)
            return True

    def _recording_completed(
        self,
        token: _LegacyRecordingToken,
        *,
        sample_count: int,
        result_snapshot: Any,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token) or self._terminal_claimed:
                return False
            self._terminal_claimed = True
            event = RecordingCompleted(token.session_id, sample_count, result_snapshot)
            if self._starting:
                self._pending_terminal = event
            else:
                self._finish_terminal_locked(event)
            return True

    def _recording_failed(self, token: _LegacyRecordingToken, reason: Any) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token) or self._terminal_claimed:
                return False
            self._terminal_claimed = True
            event = RecordingFailed(
                token.session_id, str(reason) or "legacy recording failed"
            )
            if self._starting:
                self._pending_terminal = event
            else:
                self._finish_terminal_locked(event)
            return True

    # Compatibility entry points intentionally refuse unbound terminal callbacks.
    # Callers must retain the LegacyRecordingTerminalPort received at admission.
    def recording_completed(self, *, sample_count: int, result_snapshot: Any) -> bool:
        return False

    def recording_failed(self, reason: Any) -> bool:
        return False
