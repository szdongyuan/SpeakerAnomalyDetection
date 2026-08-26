"""Mutable main-thread state for sequence trigger inputs."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from weakref import WeakMethod, ref


_UNCHANGED = object()


@dataclass(frozen=True, slots=True)
class TcpIdentityTransition:
    sequence: int
    previous: Any
    current: Any


class TcpIdentityOutboxCapacityError(RuntimeError):
    pass


class TcpIdentityAdmissionRejected(RuntimeError):
    pass


@dataclass(slots=True)
class SequenceTriggerModel:
    """Own trigger-domain state without owning the workflow phase."""

    debounce_interval_ms: int = 50
    fast_input_max_seconds: float = 0.4
    minimum_auto_commit_length: int = 7
    dedup_window_seconds: float = 0.8
    hid_suppression_seconds: float = 1.0

    barcode_first_char_ts: float | None = None
    barcode_last_char_ts: float | None = None
    barcode_capture_buffer: str = ""
    barcode_capture_first_ts: float | None = None
    barcode_capture_last_ts: float | None = None
    barcode_capture_target_text: str | None = None
    barcode_capture_target_cursor_position: int | None = None
    debounce_pending: bool = False
    sn_textchange_manual_guard: bool = False
    hid_mode_active_until: float = 0.0

    last_committed_barcode: str | None = None
    last_committed_barcode_time: float = 0.0
    pending_start_command_id: str | None = None
    shortcut_processing: bool = False
    external_trigger_available: bool = True

    tcp_enabled: bool = False
    tcp_host: str = "127.0.0.1"
    tcp_port: Any = None
    tcp_server: Any = None
    tcp_running: bool = False
    tcp_connected: bool = False
    tcp_last_request_id: str | None = None
    tcp_server_token: str | None = None
    tcp_lifecycle_generation: int | None = None

    recent_command_id_limit: int = 256
    tcp_identity_outbox_limit: int = 256
    _recent_command_ids: set[str] = field(default_factory=set, repr=False)
    _recent_command_id_order: deque[str] = field(default_factory=deque, repr=False)
    _tcp_duplicate_lock: Any = field(default_factory=RLock, repr=False)
    _tcp_identity_outbox: Any = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _tcp_identity_observers: dict[int, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _tcp_identity_next_sequence: int = field(default=1, init=False, repr=False)
    _tcp_identity_next_observer_token: int = field(
        default=1, init=False, repr=False
    )
    _tcp_identity_admission_guard: Any = field(default=None, init=False, repr=False)
    _tcp_identity_admission_guard_token: int | None = field(
        default=None, init=False, repr=False
    )
    _tcp_server_identity_observer_token: int | None = field(
        default=None, init=False, repr=False
    )

    def __setattr__(self, attribute: str, value: Any) -> None:
        if attribute != "tcp_server":
            object.__setattr__(self, attribute, value)
            return
        try:
            lock = object.__getattribute__(self, "_tcp_duplicate_lock")
        except AttributeError:
            object.__setattr__(self, attribute, value)
            return
        transition = None
        with lock:
            previous = object.__getattribute__(self, "tcp_server")
            if previous is value:
                return
            admitted, transition = self._reserve_tcp_identity_transition_locked(
                previous, value
            )
            if not admitted:
                if len(self._tcp_identity_outbox) >= self.tcp_identity_outbox_limit:
                    raise TcpIdentityOutboxCapacityError(
                        "tcp identity outbox capacity exhausted"
                    )
                raise TcpIdentityAdmissionRejected(
                    "tcp identity admission rejected"
                )
            object.__setattr__(self, attribute, value)
        self._notify_tcp_identity_observers()

    @staticmethod
    def _weak_callable(callback: Any) -> Any:
        if getattr(callback, "__self__", None) is not None:
            return WeakMethod(callback)
        return ref(callback)

    def subscribe_tcp_identity_observer(self, observer: Any) -> int:
        if not callable(observer):
            raise TypeError("tcp identity observer must be callable")
        callback_ref = self._weak_callable(observer)
        with self._tcp_duplicate_lock:
            token = self._tcp_identity_next_observer_token
            self._tcp_identity_next_observer_token += 1
            self._tcp_identity_observers[token] = callback_ref
            return token

    def unsubscribe_tcp_identity_observer(self, token: int) -> bool:
        with self._tcp_duplicate_lock:
            return self._tcp_identity_observers.pop(token, None) is not None

    def has_tcp_identity_observer(self, token: int) -> bool:
        with self._tcp_duplicate_lock:
            callback_ref = self._tcp_identity_observers.get(token)
            if callback_ref is None:
                return False
            if callback_ref() is None:
                self._tcp_identity_observers.pop(token, None)
                return False
            return True

    @property
    def tcp_identity_observer_count(self) -> int:
        with self._tcp_duplicate_lock:
            self._prune_tcp_identity_observers_locked()
            return len(self._tcp_identity_observers)

    def _prune_tcp_identity_observers_locked(self) -> None:
        for token, callback_ref in tuple(self._tcp_identity_observers.items()):
            if callback_ref() is None:
                self._tcp_identity_observers.pop(token, None)

    def _notify_tcp_identity_observers(self) -> None:
        with self._tcp_duplicate_lock:
            self._prune_tcp_identity_observers_locked()
            callbacks = tuple(self._tcp_identity_observers.items())
        dead_tokens = []
        for token, callback_ref in callbacks:
            callback = callback_ref()
            if callback is None:
                dead_tokens.append(token)
                continue
            try:
                callback()
            except BaseException:
                continue
        if dead_tokens:
            with self._tcp_duplicate_lock:
                for token in dead_tokens:
                    callback_ref = self._tcp_identity_observers.get(token)
                    if callback_ref is not None and callback_ref() is None:
                        self._tcp_identity_observers.pop(token, None)

    def subscribe_tcp_identity_admission_guard(self, guard: Any) -> int:
        if not callable(guard):
            raise TypeError("tcp identity admission guard must be callable")
        guard_ref = self._weak_callable(guard)
        with self._tcp_duplicate_lock:
            token = self._tcp_identity_next_observer_token
            self._tcp_identity_next_observer_token += 1
            self._tcp_identity_admission_guard = guard_ref
            self._tcp_identity_admission_guard_token = token
            return token

    def unsubscribe_tcp_identity_admission_guard(self, token: int) -> bool:
        with self._tcp_duplicate_lock:
            if token != self._tcp_identity_admission_guard_token:
                return False
            self._tcp_identity_admission_guard = None
            self._tcp_identity_admission_guard_token = None
            return True

    @property
    def tcp_identity_admission_guard_count(self) -> int:
        with self._tcp_duplicate_lock:
            guard_ref = self._tcp_identity_admission_guard
            if guard_ref is None:
                return 0
            if guard_ref() is None:
                self._tcp_identity_admission_guard = None
                self._tcp_identity_admission_guard_token = None
                return 0
            return 1

    def _reserve_tcp_identity_transition_locked(
        self, previous: Any, current: Any
    ) -> tuple[bool, TcpIdentityTransition | None]:
        if previous is current:
            return True, None
        if len(self._tcp_identity_outbox) >= self.tcp_identity_outbox_limit:
            return False, None
        guard_ref = self._tcp_identity_admission_guard
        guard = None if guard_ref is None else guard_ref()
        if guard_ref is not None and guard is None:
            self._tcp_identity_admission_guard = None
            self._tcp_identity_admission_guard_token = None
        if guard is not None:
            try:
                if guard(previous, current) is not True:
                    return False, None
            except BaseException:
                return False, None
        sequence = self._tcp_identity_next_sequence
        self._tcp_identity_next_sequence += 1
        transition = TcpIdentityTransition(sequence, previous, current)
        self._tcp_identity_outbox[sequence] = transition
        return True, transition

    def drain_tcp_identity_outbox(self) -> tuple[TcpIdentityTransition, ...]:
        with self._tcp_duplicate_lock:
            return tuple(self._tcp_identity_outbox.values())

    def ack_tcp_identity_transition(self, sequence: int) -> bool:
        with self._tcp_duplicate_lock:
            return self._tcp_identity_outbox.pop(sequence, None) is not None

    def set_tcp_server_identity_observer(self, observer: Any) -> None:
        previous_token = self._tcp_server_identity_observer_token
        if previous_token is not None:
            self.unsubscribe_tcp_identity_observer(previous_token)
            self._tcp_server_identity_observer_token = None
        if observer is not None:
            self._tcp_server_identity_observer_token = (
                self.subscribe_tcp_identity_observer(observer)
            )

    @property
    def _tcp_server_identity_observer(self) -> Any:
        token = self._tcp_server_identity_observer_token
        if token is None:
            return None
        with self._tcp_duplicate_lock:
            callback_ref = self._tcp_identity_observers.get(token)
            return None if callback_ref is None else callback_ref()

    def reset_capture(self, *, clear_dedup: bool = False) -> None:
        self.barcode_first_char_ts = None
        self.barcode_last_char_ts = None
        self.barcode_capture_buffer = ""
        self.barcode_capture_first_ts = None
        self.barcode_capture_last_ts = None
        self.barcode_capture_target_text = None
        self.barcode_capture_target_cursor_position = None
        self.debounce_pending = False
        if clear_dedup:
            self.reset_dedup()

    def reset_dedup(self) -> None:
        self.last_committed_barcode = None
        self.last_committed_barcode_time = 0.0

    @property
    def recent_command_id_count(self) -> int:
        return len(self._recent_command_ids)

    def admit_command_id(self, command_id: str) -> bool:
        if command_id in self._recent_command_ids:
            return False
        self._recent_command_ids.add(command_id)
        self._recent_command_id_order.append(command_id)
        while len(self._recent_command_id_order) > self.recent_command_id_limit:
            retired = self._recent_command_id_order.popleft()
            self._recent_command_ids.discard(retired)
        return True

    def admit_tcp_request_id(
        self,
        request_id: str,
        *,
        lifecycle_generation: int | None = None,
        server_token: str | None = None,
    ) -> bool:
        """Atomically admit one wire request before its Qt command is queued."""
        with self._tcp_duplicate_lock:
            if lifecycle_generation is not None and (
                lifecycle_generation != self.tcp_lifecycle_generation
                or server_token != self.tcp_server_token
                or not self.tcp_enabled
                or not self.tcp_running
                or self.tcp_server is None
            ):
                return False
            if request_id == self.tcp_last_request_id:
                return False
            self.tcp_last_request_id = request_id
            return True

    def activate_tcp_server(
        self,
        server: Any,
        *,
        lifecycle_generation: int,
        server_token: str,
        host: Any = _UNCHANGED,
        port: Any = _UNCHANGED,
    ) -> bool:
        notify = False
        with self._tcp_duplicate_lock:
            previous = self.tcp_server
            admitted, transition = self._reserve_tcp_identity_transition_locked(
                previous, server
            )
            if not admitted:
                return False
            object.__setattr__(self, "tcp_enabled", True)
            object.__setattr__(self, "tcp_running", True)
            object.__setattr__(self, "tcp_connected", False)
            object.__setattr__(self, "tcp_server", server)
            object.__setattr__(self, "tcp_server_token", server_token)
            object.__setattr__(
                self, "tcp_lifecycle_generation", lifecycle_generation
            )
            object.__setattr__(self, "tcp_last_request_id", None)
            if host is not _UNCHANGED:
                object.__setattr__(self, "tcp_host", host)
            if port is not _UNCHANGED:
                object.__setattr__(self, "tcp_port", port)
            notify = transition is not None
        if notify:
            self._notify_tcp_identity_observers()
        return True

    def tcp_server_is_current(
        self, *, lifecycle_generation: int, server_token: str
    ) -> bool:
        with self._tcp_duplicate_lock:
            return bool(
                self.tcp_enabled
                and self.tcp_running
                and self.tcp_server is not None
                and self.tcp_lifecycle_generation == lifecycle_generation
                and self.tcp_server_token == server_token
            )

    def invalidate_tcp_server(self) -> Any:
        notify = False
        with self._tcp_duplicate_lock:
            server = self.tcp_server
            admitted, transition = self._reserve_tcp_identity_transition_locked(
                server, None
            )
            if not admitted:
                return False
            object.__setattr__(self, "tcp_enabled", False)
            object.__setattr__(self, "tcp_running", False)
            object.__setattr__(self, "tcp_connected", False)
            object.__setattr__(self, "tcp_server", None)
            object.__setattr__(self, "tcp_server_token", None)
            object.__setattr__(self, "tcp_lifecycle_generation", None)
            object.__setattr__(self, "tcp_last_request_id", None)
            notify = transition is not None
        if notify:
            self._notify_tcp_identity_observers()
        return server

    def reset_tcp_request_id(self) -> None:
        with self._tcp_duplicate_lock:
            self.tcp_last_request_id = None
