"""Trigger-domain controller for barcode, hardware, shortcut, and TCP inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from threading import RLock
from typing import Any
from uuid import uuid4
from weakref import ref

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, QTimer, Qt, pyqtSignal, pyqtSlot

from base.load_config import LoadUiConfig
from base.tcp_service import TcpServer, check_tcp_msg_format
from consts.action_code import RequestTypeEnum
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import (
    BarcodeCommitted,
    StartTestRequested,
    WorkflowCommandRejected,
    WorkflowStateChanged,
)
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_trigger_view import SequenceTriggerView


_INVALID_FILENAME_CHARACTERS = frozenset('\\/:*?"<>|')
_MANUAL_EDIT_KEYS = frozenset(
    {
        Qt.Key_Backspace,
        Qt.Key_Delete,
        Qt.Key_Left,
        Qt.Key_Right,
        Qt.Key_Up,
        Qt.Key_Down,
        Qt.Key_Home,
        Qt.Key_End,
        Qt.Key_PageUp,
        Qt.Key_PageDown,
    }
)
_EXTERNAL_SOURCES = frozenset(
    {
        "hid",
        "wedge",
        "wedge_enter",
        "wedge_debounce",
        "wedge_global_enter",
        "wedge_global_debounce",
        "optical",
        "shortcut",
        "tcp",
    }
)


def _new_identifier() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class TcpTriggerPackage:
    request_type: int
    timestamp: str
    label: str
    lifecycle_generation: int = -1
    server_token: str = ""


class _QtNativeLifetimeGuard:
    """Thread-safe native QObject lifetime bit without retaining its owner."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._alive = True

    def invalidate(self, *_args: Any) -> None:
        with self._lock:
            self._alive = False

    @property
    def alive(self) -> bool:
        with self._lock:
            return self._alive


@dataclass(slots=True)
class _TrackedQtConnection:
    connection_token: Any
    sender_ref: Any
    sender_guard: _QtNativeLifetimeGuard
    owner_guard: _QtNativeLifetimeGuard
    resource: str
    signal: Any = None
    slot: Any = None


@dataclass(slots=True)
class _TcpMirrorWriteReservation:
    """Capacity already committed for one untrusted mirror writer call."""

    requested: Any
    targets: dict[int, Any]
    unexpected_slots: int = 1
    final_observed: bool = False
    final_actual: Any = None
    final_actual_verified_stopped: bool = False


@dataclass(slots=True)
class _TcpMirrorWriteCorrelation:
    token: int
    previous: Any
    requested: Any
    admission_observed: bool = False


class _TriggerDeliveryGuard(QObject):
    def __init__(self, owner: "SequenceTriggerController", handler: str) -> None:
        super().__init__(owner)
        self._owner_ref = ref(owner)
        self._handler = handler

    @pyqtSlot(object)
    def deliver(self, message: Any) -> None:
        owner = self._owner_ref()
        if owner is not None and owner._accept_queued_delivery:
            getattr(owner, self._handler)(message)


class _TcpInstanceCommandChannel(QObject):
    package_received = pyqtSignal(object)


class SequenceTriggerController(QObject):
    """Normalize trigger inputs and publish immutable start commands only."""

    def __init__(
        self,
        model: SequenceTriggerModel,
        view: SequenceTriggerView,
        *,
        start_publisher: Callable[[StartTestRequested], None],
        configuration_generation_provider: Callable[[], int],
        workflow_active_provider: Callable[[], bool] = lambda: False,
        external_mode_available_provider: Callable[[], bool] = lambda: True,
        acquisition_mode_provider: Callable[[], str | None] = lambda: None,
        regex_rule_loader: Callable[[], Mapping[str, Any]] = (
            lambda: LoadUiConfig.get_selected_sn_regex_rule(
                LoadUiConfig.load_sn_regex_rules_from_json()
            )
        ),
        barcode_publisher: Callable[[BarcodeCommitted], None] | None = None,
        event_bus: SequenceEventBus | None = None,
        monotonic: Callable[[], float] | None = None,
        command_id_factory: Callable[[], str] = _new_identifier,
        debounce_timer: Any = None,
        logger: Any = None,
        hardware_manager: Any = None,
        shortcut_manager: Any = None,
        tcp_server_factory: Callable[..., Any] = TcpServer,
        tcp_message_validator: Callable[[Any], tuple[bool, Any]] = check_tcp_msg_format,
        tcp_config_reader: Callable[[], tuple[Any, Any]] = LoadUiConfig.get_tcp_config,
        tcp_config_writer: Callable[[Any, Any], Any] | None = None,
        tcp_mirror_getter: Callable[[], Any] = lambda: None,
        tcp_mirror_setter: Callable[[Any], None] = lambda _server: None,
        resource_journal_limit: int = 64,
        tcp_journal_limit: int = 128,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(model, SequenceTriggerModel):
            raise TypeError("model must be SequenceTriggerModel")
        self.model = model
        self.view = view
        self.start_publisher = start_publisher
        self.configuration_generation_provider = configuration_generation_provider
        self.workflow_active_provider = workflow_active_provider
        self.external_mode_available_provider = external_mode_available_provider
        self.acquisition_mode_provider = acquisition_mode_provider
        self.regex_rule_loader = regex_rule_loader
        self.barcode_publisher = barcode_publisher
        self.event_bus = event_bus
        if monotonic is None:
            import time

            monotonic = time.monotonic
        self.monotonic = monotonic
        self.command_id_factory = command_id_factory
        self.logger = logger
        self._hardware_manager = hardware_manager
        self._shortcut_manager = shortcut_manager
        self.tcp_server_factory = tcp_server_factory
        self.tcp_message_validator = tcp_message_validator
        self.tcp_config_reader = tcp_config_reader
        self.tcp_config_writer = tcp_config_writer
        self._tcp_mirror_reader = tcp_mirror_getter
        self._tcp_mirror_writer = tcp_mirror_setter
        self._connections: list[Any] = []
        self._hardware_connections: list[Any] = []
        self._shortcut_connections: list[Any] = []
        self._lifecycle_lock = RLock()
        self._resource_journal_limit = max(1, int(resource_journal_limit))
        self._tcp_journal_limit = max(1, int(tcp_journal_limit))
        self._active = True
        self._lifecycle_state = "ACTIVE"
        self._lifecycle_generation = 0
        self._resource_identity_epoch = 0
        self._tcp_mirror_write_inflight = 0
        self._tcp_mirror_write_correlation_sequence = 0
        self._tcp_mirror_write_correlation = None
        self._tcp_mirror_write_reservation_sequence = 0
        self._tcp_mirror_write_reservations: dict[
            int, _TcpMirrorWriteReservation
        ] = {}
        self._accept_queued_delivery = True
        self._tcp_configuration_dialog_pending = False
        self._debounce_timeout_signal = None
        self._disconnect_resource_steps_completed: set[str] = set()
        self._disconnect_resource_journal: dict[str, dict[int, Any]] = {
            "shortcut": {},
            "hardware": {},
        }
        # These dictionaries strongly own only targets whose stop operation has
        # not completed.  Completed targets are released immediately; bounded
        # integer identities are enough to remember a trusted no-state port
        # while its owner still exposes that same instance.
        self._disconnect_trusted_stopped_ids: dict[str, int] = {}
        self._tcp_stop_journal: dict[int, Any] = {}
        self._tcp_stop_completed_handles: dict[int, None] = {}
        self._tcp_mirror_release_journal: dict[int, Any] = {}
        self._tcp_model_release_journal: dict[int, Any] = {}
        self._finalization_steps_completed: set[str] = set()
        self._finalization_in_progress = False
        self._finalization_retry_requested = False
        self._model_identity_observer_unsubscribed_token = None
        self._qt_connection_owner_guard = _QtNativeLifetimeGuard()
        self.destroyed.connect(self._qt_connection_owner_guard.invalidate)
        self._model_identity_guard_token = (
            self.model.subscribe_tcp_identity_admission_guard(
                self._admit_model_tcp_identity_write
            )
        )
        self._model_identity_observer_token = (
            self.model.subscribe_tcp_identity_observer(
                self._drain_model_identity_outbox
            )
        )
        self._drain_model_identity_outbox()

        if debounce_timer is None:
            debounce_timer = QTimer(self)
            debounce_timer.setSingleShot(True)
            debounce_timer.setInterval(self.model.debounce_interval_ms)
            self._debounce_timeout_signal = debounce_timer.timeout
        self.debounce_timer = debounce_timer
        debounce_signal = getattr(debounce_timer, "timeout", None)
        self._debounce_connection = (
            self._connect_tracked_signal(
                debounce_timer,
                debounce_signal,
                self.handle_barcode_debounce_timeout,
                "debounce-signal",
            )
            if debounce_signal is not None
            else None
        )

        self._tcp_channel = _TcpInstanceCommandChannel(self)
        self._tcp_guard = _TriggerDeliveryGuard(self, "handle_tcp_package")
        self._tcp_channel_connection = self._connect_tracked_signal(
            self._tcp_channel,
            self._tcp_channel.package_received,
            self._tcp_guard.deliver,
            "tcp-channel",
            Qt.QueuedConnection,
        )
        self._tcp_channel_connected = True
        if event_bus is not None:
            self._wire_event_bus(event_bus)

    def _wire_event_bus(self, bus: SequenceEventBus) -> None:
        for sender, signal, handler in (
            (
                bus.commands,
                bus.commands.barcode_committed,
                "handle_barcode_committed",
            ),
            (
                bus.events,
                bus.events.workflow_command_rejected,
                "handle_workflow_rejection",
            ),
            (
                bus.events,
                bus.events.workflow_state_changed,
                "handle_workflow_state_changed",
            ),
        ):
            guard = _TriggerDeliveryGuard(self, handler)
            self._connections.append(
                self._connect_tracked_signal(
                    sender,
                    signal,
                    guard.deliver,
                    "event-connection",
                    Qt.QueuedConnection,
                )
            )

    def _connect_tracked_signal(
        self,
        sender: Any,
        signal: Any,
        slot: Any,
        resource: str,
        connection_type: Any = None,
    ) -> _TrackedQtConnection:
        if connection_type is None:
            token = signal.connect(slot)
        else:
            try:
                token = signal.connect(slot, connection_type)
            except TypeError:
                token = signal.connect(slot)
        guard = _QtNativeLifetimeGuard()
        sender_ref = None
        if isinstance(sender, QObject):
            sender_ref = ref(sender)
            try:
                sender.destroyed.connect(guard.invalidate)
            except (RuntimeError, TypeError):
                guard.invalidate()
        else:
            try:
                sender_ref = ref(sender)
            except TypeError:
                sender_ref = lambda: sender
        return _TrackedQtConnection(
            connection_token=token,
            sender_ref=sender_ref,
            sender_guard=guard,
            owner_guard=self._qt_connection_owner_guard,
            resource=resource,
            signal=None if token is not None else signal,
            slot=None if token is not None else slot,
        )

    @property
    def is_active(self) -> bool:
        with self._lifecycle_lock:
            return self._lifecycle_state == "ACTIVE"

    @property
    def lifecycle_state(self) -> str:
        with self._lifecycle_lock:
            return self._lifecycle_state

    @property
    def lifecycle_generation(self) -> int:
        with self._lifecycle_lock:
            return self._lifecycle_generation

    @property
    def shortcut_manager(self) -> Any:
        with self._lifecycle_lock:
            return self._shortcut_manager

    @shortcut_manager.setter
    def shortcut_manager(self, manager: Any) -> None:
        self._set_manager_identity("shortcut", manager)

    @property
    def hardware_manager(self) -> Any:
        with self._lifecycle_lock:
            return self._hardware_manager

    @hardware_manager.setter
    def hardware_manager(self, manager: Any) -> None:
        self._set_manager_identity("hardware", manager)

    def _set_manager_identity(self, resource: str, manager: Any) -> bool:
        attribute = f"_{resource}_manager"
        with self._lifecycle_lock:
            if self._lifecycle_state in {"FINALIZING", "INACTIVE"}:
                return False
            previous = getattr(self, attribute)
            if previous is manager:
                return True
            if (
                self._lifecycle_state == "DISCONNECTING"
                and not self._admit_manager_targets_locked(
                    resource, previous, manager
                )
            ):
                self._report_lifecycle_failure(
                    "admission-capacity", resource, None
                )
                return False
            setattr(self, attribute, manager)
            self._resource_identity_epoch += 1
            self._observe_manager_identity_write_locked(
                resource, previous, manager
            )
            return True

    def _observe_manager_identity_write(
        self, resource: str, previous: Any, current: Any
    ) -> None:
        with self._lifecycle_lock:
            self._observe_manager_identity_write_locked(
                resource, previous, current
            )

    def _observe_manager_identity_write_locked(
        self, resource: str, previous: Any, current: Any
    ) -> bool:
        if self._lifecycle_state != "DISCONNECTING":
            return True
        if not self._admit_manager_targets_locked(
            resource, previous, current
        ):
            return False
        trusted_identity = self._disconnect_trusted_stopped_ids.get(resource)
        if current is None or (
            trusted_identity is not None and trusted_identity != id(current)
        ):
            self._disconnect_trusted_stopped_ids.pop(resource, None)
        return True

    def _admit_manager_targets_locked(
        self, resource: str, *targets: Any
    ) -> bool:
        pending = self._disconnect_resource_journal[resource]
        candidates = []
        for target in targets:
            if target is None:
                continue
            identity = id(target)
            if self._disconnect_trusted_stopped_ids.get(resource) == identity:
                continue
            if identity not in pending and all(
                id(candidate) != identity for candidate in candidates
            ):
                candidates.append(target)
        if len(pending) + len(candidates) > self._resource_journal_limit:
            return False
        for target in candidates:
            pending[id(target)] = target
        return True

    def _admit_tcp_targets_locked(self, *targets: Any) -> bool:
        reserved, unexpected_slots = self._tcp_reserved_capacity_locked()
        candidates = []
        for target in targets:
            if target is None:
                continue
            identity = id(target)
            if identity in self._tcp_stop_completed_handles:
                continue
            if (
                identity not in self._tcp_stop_journal
                and identity not in reserved
                and all(
                    id(candidate) != identity for candidate in candidates
                )
            ):
                candidates.append(target)
        occupied = set(self._tcp_stop_journal).union(reserved)
        if (
            len(occupied) + unexpected_slots + len(candidates)
            > self._tcp_journal_limit
        ):
            return False
        for target in candidates:
            self._tcp_stop_journal[id(target)] = target
        return True

    def _tcp_reserved_capacity_locked(
        self, *, exclude_token: int | None = None
    ) -> tuple[set[int], int]:
        identities: set[int] = set()
        unexpected_slots = 0
        for token, reservation in self._tcp_mirror_write_reservations.items():
            if token == exclude_token:
                continue
            identities.update(reservation.targets)
            unexpected_slots += reservation.unexpected_slots
        return identities, unexpected_slots

    def _reserve_tcp_mirror_targets_locked(
        self, previous: Any, requested: Any
    ) -> int | None:
        """Strongly own exact identities before an injected mirror writer runs."""
        occupied = set(self._tcp_stop_journal)
        reserved, unexpected_slots = self._tcp_reserved_capacity_locked()
        occupied.update(reserved)
        reservation: dict[int, Any] = {}
        for target in (previous, requested):
            if target is None:
                continue
            identity = id(target)
            if identity not in occupied and identity not in reservation:
                reservation[identity] = target
        if (
            len(occupied)
            + unexpected_slots
            + len(reservation)
            + 1
            > self._tcp_journal_limit
        ):
            return None
        if requested is not None:
            self._tcp_stop_completed_handles.pop(id(requested), None)
        self._tcp_mirror_write_reservation_sequence += 1
        token = self._tcp_mirror_write_reservation_sequence
        self._tcp_mirror_write_reservations[token] = (
            _TcpMirrorWriteReservation(
                requested=requested,
                targets=reservation,
            )
        )
        return token

    def _settle_tcp_mirror_reservation_locked(
        self,
        token: int | None,
        *,
        actual: Any,
        final_observed: bool,
        actual_verified_stopped: bool = False,
    ) -> bool:
        reservation = self._tcp_mirror_write_reservations.get(token)
        if reservation is None:
            return token is None
        if final_observed:
            reservation.final_observed = True
            reservation.final_actual = actual
            reservation.final_actual_verified_stopped = bool(
                actual_verified_stopped
            )
            reservation.unexpected_slots = 0
            if actual is not None:
                if actual_verified_stopped:
                    self._tcp_stop_completed_handles[id(actual)] = None
                else:
                    self._tcp_stop_completed_handles.pop(id(actual), None)
                reservation.targets.setdefault(id(actual), actual)
        if not reservation.final_observed:
            return False

        occupied = set(self._tcp_stop_journal)
        other_reserved, other_unexpected_slots = (
            self._tcp_reserved_capacity_locked(exclude_token=token)
        )
        occupied.update(other_reserved)
        candidates = {
            identity: target
            for identity, target in reservation.targets.items()
            if identity not in self._tcp_stop_completed_handles
        }
        if (
            len(occupied.union(candidates)) + other_unexpected_slots
            > self._tcp_journal_limit
        ):
            return False
        if candidates and self._lifecycle_state != "DISCONNECTING":
            return False
        for identity, target in candidates.items():
            self._tcp_stop_journal.setdefault(identity, target)
        self._tcp_mirror_write_reservations.pop(token, None)
        return True

    def _retry_tcp_mirror_reservations_locked(
        self,
        *,
        actual: Any,
        final_observed: bool,
        actual_verified_stopped: bool = False,
    ) -> bool:
        settled_all = True
        for token, reservation in tuple(
            self._tcp_mirror_write_reservations.items()
        ):
            observed = reservation.final_observed or final_observed
            final_actual = (
                reservation.final_actual
                if reservation.final_observed
                else actual
            )
            verified_stopped = (
                reservation.final_actual_verified_stopped
                if reservation.final_observed
                else actual_verified_stopped
            )
            if not self._settle_tcp_mirror_reservation_locked(
                token,
                actual=final_actual,
                final_observed=observed,
                actual_verified_stopped=verified_stopped,
            ):
                settled_all = False
        return settled_all

    def _admit_model_tcp_identity_write(
        self, previous: Any, current: Any
    ) -> bool:
        with self._lifecycle_lock:
            if self._lifecycle_state in {"FINALIZING", "INACTIVE"}:
                return False
            if (
                self._lifecycle_state == "DISCONNECTING"
                and not self._admit_tcp_targets_locked(previous, current)
            ):
                self._report_lifecycle_failure(
                    "admission-capacity", "tcp-model-identity", None
                )
                return False
            self._resource_identity_epoch += 1
            self._observe_tcp_identity_write_locked(previous, current)
            return True

    def _drain_model_identity_outbox(self) -> bool:
        try:
            transitions = self.model.drain_tcp_identity_outbox()
        except BaseException as error:
            self._report_lifecycle_failure(
                "drain", "tcp-model-identity", error
            )
            return False
        completed = True
        for transition in transitions:
            try:
                previous = transition.previous
                current = transition.current
                sequence = transition.sequence
            except BaseException as error:
                self._report_lifecycle_failure(
                    "inspect", "tcp-model-identity", error
                )
                completed = False
                continue
            with self._lifecycle_lock:
                self._observe_tcp_identity_write_locked(previous, current)
            try:
                acknowledged = self.model.ack_tcp_identity_transition(sequence)
            except BaseException as error:
                self._report_lifecycle_failure(
                    "ack", "tcp-model-identity", error
                )
                completed = False
                continue
            if acknowledged is not True:
                completed = False
        return completed

    def _observe_tcp_identity_write(self, *targets: Any) -> None:
        with self._lifecycle_lock:
            self._observe_tcp_identity_write_locked(*targets)

    def _observe_tcp_identity_write_locked(self, *targets: Any) -> None:
        if self._lifecycle_state != "DISCONNECTING":
            return
        if not self._admit_tcp_targets_locked(*targets):
            self._report_lifecycle_failure(
                "observe-capacity", "tcp", None
            )

    def _admit_canonical_tcp_mirror_identity_locked(
        self, previous: Any, current: Any
    ) -> bool:
        if previous is current:
            return True
        state = self._lifecycle_state
        if current is not None and state in {
            "DISCONNECTING",
            "FINALIZING",
            "INACTIVE",
        }:
            self._resource_identity_epoch += 1
            return False
        if state == "DISCONNECTING":
            if not self._admit_tcp_targets_locked(previous, current):
                self._resource_identity_epoch += 1
                return False
            self._observe_tcp_identity_write_locked(previous, current)
        correlation = self._tcp_mirror_write_correlation
        if (
            correlation is not None
            and previous is correlation.previous
            and current is correlation.requested
        ):
            correlation.admission_observed = True
            return True
        self._resource_identity_epoch += 1
        return True

    def tcp_mirror_getter(self) -> Any:
        return self._tcp_mirror_reader()

    def tcp_mirror_setter(self, server: Any) -> bool:
        written, _previous, current = self._write_tcp_mirror_identity(server)
        if written and current is server:
            return True
        self._report_lifecycle_failure("verify-write", "tcp-mirror", None)
        return False

    def _write_tcp_mirror_identity(
        self, server: Any
    ) -> tuple[bool, Any, Any]:
        reservation_token = None
        reservation_epoch = None
        correlation_token = None
        with self._lifecycle_lock:
            if (
                self._lifecycle_state in {"FINALIZING", "INACTIVE"}
                or self._tcp_mirror_write_inflight
            ):
                return False, None, None
            observation_epoch = self._resource_identity_epoch

        before_ok, previous = self._read_tcp_mirror()
        if not before_ok:
            return False, None, None

        with self._lifecycle_lock:
            if (
                self._lifecycle_state in {"FINALIZING", "INACTIVE"}
                or self._tcp_mirror_write_inflight
            ):
                return False, previous, previous
            if self._resource_identity_epoch != observation_epoch:
                return False, previous, previous
            if self._lifecycle_state == "DISCONNECTING":
                reservation_token = self._reserve_tcp_mirror_targets_locked(
                    previous, server
                )
                if reservation_token is None:
                    self._report_lifecycle_failure(
                        "admission-capacity", "tcp-mirror", None
                    )
                    return False, previous, previous
            self._tcp_mirror_write_correlation_sequence += 1
            correlation_token = self._tcp_mirror_write_correlation_sequence
            self._tcp_mirror_write_correlation = _TcpMirrorWriteCorrelation(
                token=correlation_token,
                previous=previous,
                requested=server,
            )
            self._tcp_mirror_write_inflight += 1
            self._resource_identity_epoch += 1
            reservation_epoch = self._resource_identity_epoch
        after_ok = False
        current = None
        actual_verified_stopped = False
        written = False
        epoch_stable = False
        settled = False
        try:
            try:
                written = self._tcp_mirror_writer(server)
            except BaseException as error:
                self._report_lifecycle_failure("write", "tcp-mirror", error)
            after_ok, current = self._read_tcp_mirror()
            actual_verified_stopped = bool(
                after_ok
                and current is not None
                and self._tcp_server_active_state(current) is False
            )
        finally:
            with self._lifecycle_lock:
                epoch_stable = (
                    self._resource_identity_epoch == reservation_epoch
                )
                if reservation_token is not None:
                    settled = self._settle_tcp_mirror_reservation_locked(
                        reservation_token,
                        actual=current if after_ok else None,
                        final_observed=after_ok,
                        actual_verified_stopped=actual_verified_stopped,
                    )
                    if not settled:
                        self._report_lifecycle_failure(
                            "observe-capacity", "tcp-mirror", None
                        )
                else:
                    settled = True
                    if after_ok:
                        self._observe_tcp_identity_write_locked(
                            previous, server, current
                        )
                correlation = self._tcp_mirror_write_correlation
                if (
                    correlation is not None
                    and correlation.token == correlation_token
                ):
                    self._tcp_mirror_write_correlation = None
                self._tcp_mirror_write_inflight -= 1
        return (
            bool(
                settled
                and written is not False
                and after_ok
                and current is server
                and epoch_stable
            ),
            previous,
            current if after_ok else None,
        )

    def _capture_active_generation(self) -> int | None:
        with self._lifecycle_lock:
            if self._lifecycle_state != "ACTIVE":
                return None
            return self._lifecycle_generation

    def _generation_is_active(self, generation: int) -> bool:
        with self._lifecycle_lock:
            return (
                self._lifecycle_state == "ACTIVE"
                and self._lifecycle_generation == generation
            )

    def disconnect(self, _lifecycle_request=None) -> bool:
        with self._lifecycle_lock:
            if self._lifecycle_state == "INACTIVE":
                return True
            if self._lifecycle_state == "FINALIZING":
                finalize_only = True
            else:
                finalize_only = False
            if self._lifecycle_state == "ACTIVE":
                self._lifecycle_state = "DISCONNECTING"
                self._resource_identity_epoch += 1
                self._active = False
                self._accept_queued_delivery = False

        if finalize_only:
            return self._advance_disconnect_finalization()

        all_resources_stopped = self.stop_tcp()
        if all_resources_stopped:
            self._disconnect_resource_steps_completed.add("tcp")
        else:
            self._disconnect_resource_steps_completed.discard("tcp")
        for resource, manager in (
            ("shortcut", self.shortcut_manager),
            ("hardware", self.hardware_manager),
        ):
            stopped = self._stop_disconnect_manager_targets(resource, manager)
            if stopped:
                self._disconnect_resource_steps_completed.add(resource)
            else:
                self._disconnect_resource_steps_completed.discard(resource)
                all_resources_stopped = False
        if not all_resources_stopped:
            return False
        with self._lifecycle_lock:
            review_epoch = self._resource_identity_epoch
        if not self._disconnect_resources_stable():
            return False
        with self._lifecycle_lock:
            journals_empty = not (
                self._disconnect_resource_journal["shortcut"]
                or self._disconnect_resource_journal["hardware"]
                or self._tcp_stop_journal
                or self._tcp_mirror_write_reservations
                or self._tcp_mirror_release_journal
                or self._tcp_model_release_journal
            )
            if (
                self._lifecycle_state != "DISCONNECTING"
                or self._resource_identity_epoch != review_epoch
                or self._tcp_mirror_write_inflight
                or not journals_empty
            ):
                return False
            self._lifecycle_state = "FINALIZING"
            self._resource_identity_epoch += 1
            self._accept_queued_delivery = False
        return self._advance_disconnect_finalization()

    def _advance_disconnect_finalization(self) -> bool:
        with self._lifecycle_lock:
            if self._lifecycle_state == "INACTIVE":
                return True
            if self._lifecycle_state != "FINALIZING":
                return False
            if self._finalization_in_progress:
                self._finalization_retry_requested = True
                return False
            self._finalization_in_progress = True

        result = False
        try:
            # A reentrant disconnect request is coalesced into at most one
            # additional pass owned by this outer runner.  A second request is
            # retained for the next explicit/event-driven call, preventing a
            # persistent callback failure from spinning the UI thread.
            for _pass_index in range(2):
                with self._lifecycle_lock:
                    self._finalization_retry_requested = False
                result = self._run_disconnect_finalization_pass()
                with self._lifecycle_lock:
                    retry_requested = self._finalization_retry_requested
                if result or not retry_requested:
                    break
            return result
        finally:
            with self._lifecycle_lock:
                self._finalization_in_progress = False

    def _run_disconnect_finalization_pass(self) -> bool:
        with self._lifecycle_lock:
            if self._lifecycle_state == "INACTIVE":
                return True
            if self._lifecycle_state != "FINALIZING":
                return False
            finalization_epoch = self._resource_identity_epoch

        steps = (
            ("model-observer", self._finalize_model_identity_observer),
            ("model-state", self._finalize_model_state),
            ("debounce", self._finalize_debounce),
            ("debounce-signal", self._finalize_debounce_signal),
            ("event-connections", self._finalize_event_connections),
            ("tcp-channel", self._finalize_tcp_channel),
            ("hardware-connections", self._finalize_hardware_connections),
            ("shortcut-connections", self._finalize_shortcut_connections),
            ("dialogs", self._finalize_dialogs),
            ("cleanup", self._finalize_cleanup_state),
        )
        completed_all = True
        for name, callback in steps:
            with self._lifecycle_lock:
                already_completed = name in self._finalization_steps_completed
            if already_completed:
                continue
            try:
                completed = callback()
            except BaseException as error:
                self._report_lifecycle_failure(
                    "finalize", name, error
                )
                completed = False
            if completed is not True:
                if completed is False:
                    self._report_lifecycle_failure(
                        "finalize", name, None
                    )
                completed_all = False
                continue
            with self._lifecycle_lock:
                self._finalization_steps_completed.add(name)

        with self._lifecycle_lock:
            if (
                not completed_all
                or len(self._finalization_steps_completed) != len(steps)
                or self._lifecycle_state != "FINALIZING"
                or self._resource_identity_epoch != finalization_epoch
            ):
                return False
            self._lifecycle_state = "INACTIVE"
            self._lifecycle_generation += 1
            self._resource_identity_epoch += 1
            self._accept_queued_delivery = False
            return True

    def _finalize_model_identity_observer(self) -> bool:
        with self._lifecycle_lock:
            observer_token = self._model_identity_observer_token
            unsubscribe_confirmed = (
                self._model_identity_observer_unsubscribed_token
                == observer_token
            )
        if observer_token is None:
            return True
        if not unsubscribe_confirmed:
            try:
                self.model.unsubscribe_tcp_identity_observer(
                    observer_token
                )
            except BaseException as error:
                self._report_lifecycle_failure(
                    "unsubscribe", "tcp-model-identity", error
                )
            with self._lifecycle_lock:
                if self._model_identity_observer_token != observer_token:
                    return False
                self._model_identity_observer_unsubscribed_token = (
                    observer_token
                )
        try:
            still_present = self.model.has_tcp_identity_observer(observer_token)
        except BaseException as error:
            self._report_lifecycle_failure(
                "verify-unsubscribe", "tcp-model-identity", error
            )
            return False
        if still_present is not False:
            with self._lifecycle_lock:
                if self._model_identity_observer_token == observer_token:
                    self._model_identity_observer_unsubscribed_token = None
            return False
        with self._lifecycle_lock:
            if self._model_identity_observer_token != observer_token:
                return False
            self._model_identity_observer_token = None
            self._model_identity_observer_unsubscribed_token = None
        return True

    def _finalize_model_state(self) -> bool:
        self.model.pending_start_command_id = None
        self._tcp_configuration_dialog_pending = False
        return True

    def _finalize_debounce(self) -> bool:
        try:
            stopped = self._stop_debounce()
        except BaseException as error:
            self._report_lifecycle_failure("stop", "debounce", error)
            return False
        return stopped is not False

    def _finalize_debounce_signal(self) -> bool:
        connection = getattr(self, "_debounce_connection", None)
        if connection is None:
            return True
        if not self._disconnect_connection(
            connection, "debounce-signal"
        ):
            return False
        if self._debounce_connection is connection:
            self._debounce_connection = None
            self._debounce_timeout_signal = None
        return self._debounce_connection is None

    def _finalize_event_connections(self) -> bool:
        for connection in reversed(tuple(self._connections)):
            if self._disconnect_connection(
                connection, "event-connection"
            ):
                self._remove_connection(self._connections, connection)
        return not self._connections

    def _finalize_tcp_channel(self) -> bool:
        if not self._tcp_channel_connected:
            return True
        connection = getattr(self, "_tcp_channel_connection", None)
        if connection is not None and not self._disconnect_connection(
            connection, "tcp-channel"
        ):
            return False
        self._tcp_channel_connection = None
        self._tcp_channel_connected = False
        return True

    def _finalize_hardware_connections(self) -> bool:
        return self._finalize_connection_list(
            self._hardware_connections, "hardware-connection"
        )

    def _finalize_shortcut_connections(self) -> bool:
        return self._finalize_connection_list(
            self._shortcut_connections, "shortcut-connection"
        )

    def _finalize_connection_list(
        self, connections: list[Any], resource: str
    ) -> bool:
        for connection in reversed(tuple(connections)):
            if self._disconnect_connection(connection, resource):
                self._remove_connection(connections, connection)
        return not connections

    def _disconnect_connection(self, connection: Any, resource: str) -> bool:
        if not isinstance(connection, _TrackedQtConnection):
            try:
                signal, target = connection
                slot = getattr(target, "deliver", target)
            except BaseException as error:
                self._report_lifecycle_failure(
                    "disconnect-port", resource, error
                )
                return False
            return self._disconnect_exact_signal(signal, slot, resource)

        if self._tracked_connection_is_absent(connection):
            return True
        token = connection.connection_token
        if token is not None:
            try:
                # False means this exact QMetaObject::Connection was already
                # disconnected externally, which is an idempotent success.
                QObject.disconnect(token)
                return True
            except BaseException as error:
                if self._tracked_connection_is_absent(connection):
                    return True
                self._report_lifecycle_failure(
                    "disconnect", resource, error
                )
                return False
        return self._disconnect_exact_signal(
            connection.signal, connection.slot, resource
        )

    @staticmethod
    def _tracked_connection_is_absent(
        connection: _TrackedQtConnection,
    ) -> bool:
        if not connection.owner_guard.alive or not connection.sender_guard.alive:
            return True
        sender_ref = connection.sender_ref
        try:
            sender = None if sender_ref is None else sender_ref()
        except BaseException:
            return False
        if sender is None:
            return True
        if isinstance(sender, QObject):
            try:
                return bool(sip.isdeleted(sender))
            except BaseException:
                return False
        return False

    def _disconnect_exact_signal(
        self, signal: Any, slot: Any, resource: str
    ) -> bool:
        try:
            disconnected = signal.disconnect(slot)
        except BaseException as error:
            self._report_lifecycle_failure("disconnect", resource, error)
            return False
        if disconnected is False:
            return False
        return True

    @staticmethod
    def _remove_connection(connections: list[Any], target: Any) -> None:
        for index, current in enumerate(connections):
            if current is target:
                connections.pop(index)
                return

    def _finalize_dialogs(self) -> bool:
        try:
            close_dialogs = getattr(self.view, "close_dialogs", None)
        except BaseException as error:
            self._report_lifecycle_failure(
                "close-port", "dialogs", error
            )
            return False
        if callable(close_dialogs):
            try:
                closed = close_dialogs()
            except BaseException as error:
                self._report_lifecycle_failure("close", "dialogs", error)
                return False
            if closed is False:
                return False
        return True

    def _finalize_cleanup_state(self) -> bool:
        self._disconnect_trusted_stopped_ids.clear()
        return True

    def _disconnect_resources_stable(self) -> bool:
        stable = True
        if not self._drain_model_identity_outbox():
            stable = False
        try:
            if self.model.drain_tcp_identity_outbox():
                stable = False
        except BaseException as error:
            self._report_lifecycle_failure(
                "inspect", "tcp-model-identity", error
            )
            stable = False
        for resource in ("shortcut", "hardware"):
            current = self._current_manager(resource)
            state = self._manager_active_state(current, resource)
            trusted = bool(
                current is not None
                and state is None
                and self._disconnect_trusted_stopped_ids.get(resource)
                == id(current)
            )
            if current is not None and (
                state is True or (state is None and not trusted)
            ):
                with self._lifecycle_lock:
                    if state is True:
                        self._disconnect_trusted_stopped_ids.pop(
                            resource, None
                        )
                    if not self._admit_manager_targets_locked(
                        resource, current
                    ):
                        self._report_lifecycle_failure(
                            "review-capacity", resource, None
                        )
            if self._disconnect_resource_journal[resource]:
                stable = False

        if self._journal_reentrant_tcp_servers():
            stable = False
        mirror_ok, mirror = self._read_tcp_mirror()
        if not mirror_ok:
            stable = False
        if (
            self._tcp_stop_journal
            or self._tcp_mirror_write_reservations
            or self._tcp_mirror_release_journal
            or self._tcp_model_release_journal
            or self.model.tcp_server is not None
            or (mirror_ok and mirror is not None)
        ):
            stable = False
        return stable

    def _stop_disconnect_manager_targets(
        self, resource: str, manager: Any
    ) -> bool:
        pending = self._disconnect_resource_journal.setdefault(resource, {})
        active_state = self._manager_active_state(manager, resource)
        if manager is not None:
            trusted_stopped = (
                active_state is None
                and self._disconnect_trusted_stopped_ids.get(resource) == id(manager)
            )
            if active_state is True or (active_state is None and not trusted_stopped):
                with self._lifecycle_lock:
                    if active_state is True:
                        self._disconnect_trusted_stopped_ids.pop(
                            resource, None
                        )
                    if not self._admit_manager_targets_locked(
                        resource, manager
                    ):
                        self._report_lifecycle_failure(
                            "admission-capacity", resource, None
                        )
                        return False

        for identity, target in tuple(pending.items()):
            try:
                stop = getattr(target, "stop", None)
            except BaseException as error:
                self._report_lifecycle_failure("stop-port", resource, error)
                self._journal_reentrant_manager(resource, target)
                continue
            if not callable(stop):
                self._report_lifecycle_failure("stop-port", resource, None)
                self._journal_reentrant_manager(resource, target)
                continue
            try:
                stopped = stop()
            except BaseException as error:
                self._report_lifecycle_failure("stop", resource, error)
                self._journal_reentrant_manager(resource, target)
                continue
            if stopped is False:
                self._report_lifecycle_failure("stop", resource, None)
                self._journal_reentrant_manager(resource, target)
                continue
            post_state = self._manager_active_state(target, resource)
            if post_state is True:
                self._report_lifecycle_failure("verify-stop", resource, None)
                self._journal_reentrant_manager(resource, target)
                continue
            pending.pop(identity, None)
            self._disconnect_trusted_stopped_ids[resource] = identity
            self._journal_reentrant_manager(resource, target)

        current_manager = self._current_manager(resource)
        current_state = self._manager_active_state(current_manager, resource)
        current_stopped = bool(
            current_manager is None
            or current_state is False
            or (
                current_state is None
                and self._disconnect_trusted_stopped_ids.get(resource)
                == id(current_manager)
            )
        )
        trusted_identity = self._disconnect_trusted_stopped_ids.get(resource)
        if current_manager is not None and (
            current_state is True
            or (
                trusted_identity is not None
                and trusted_identity != id(current_manager)
            )
        ):
            self._disconnect_trusted_stopped_ids.pop(resource, None)
        return not pending and current_stopped

    def _current_manager(self, resource: str) -> Any:
        return (
            self.shortcut_manager
            if resource == "shortcut"
            else self.hardware_manager
        )

    def _journal_reentrant_manager(self, resource: str, stopped_target: Any) -> None:
        current = self._current_manager(resource)
        if current is None or current is stopped_target:
            return
        active_state = self._manager_active_state(current, resource)
        trusted_stopped = (
            active_state is None
            and self._disconnect_trusted_stopped_ids.get(resource) == id(current)
        )
        if active_state is True or (active_state is None and not trusted_stopped):
            with self._lifecycle_lock:
                if not self._admit_manager_targets_locked(resource, current):
                    self._report_lifecycle_failure(
                        "reentrant-capacity", resource, None
                    )

    def _manager_active_state(self, manager: Any, resource: str) -> bool | None:
        if manager is None:
            return False
        observed = []
        for attribute in (
            "is_active",
            "active",
            "is_enabled",
            "enabled",
            "is_running",
            "running",
            "status",
            "_scanner_enabled",
        ):
            ok, value = self._observe_resource_attribute(
                manager, attribute, resource
            )
            if not ok:
                return True
            if callable(value):
                try:
                    value = value()
                except BaseException as error:
                    self._report_lifecycle_failure(
                        f"inspect-{attribute}-call", resource, error
                    )
                    return True
            if type(value) is bool:
                observed.append(value)
            elif attribute == "status" and type(value) is str:
                observed.append(value.upper() not in {"STOPPED", "INACTIVE", "IDLE"})
        for attribute in ("handle", "_hotkey_handle"):
            ok, value = self._observe_resource_attribute(
                manager, attribute, resource
            )
            if not ok:
                return True
            if value is not None:
                observed.append(True)
        ok, hid_handles = self._observe_resource_attribute(
            manager, "hid_handles", resource
        )
        if not ok:
            return True
        if hid_handles is not None:
            try:
                observed.append(bool(hid_handles))
            except BaseException as error:
                self._report_lifecycle_failure(
                    "inspect-hid_handles-bool", resource, error
                )
                return True
        ok, hotkey_registered = self._observe_resource_attribute(
            manager, "hotkey_registered", resource
        )
        if not ok:
            return True
        if type(hotkey_registered) is bool:
            observed.append(hotkey_registered)
        ok, timer = self._observe_resource_attribute(
            manager, "_hid_poll_timer", resource
        )
        if not ok:
            return True
        ok, timer_active = self._observe_resource_attribute(
            timer, "isActive", resource
        )
        if not ok:
            return True
        if callable(timer_active):
            try:
                observed.append(bool(timer_active()))
            except BaseException as error:
                self._report_lifecycle_failure("inspect", resource, error)
                return True
        if not observed:
            return None
        return any(observed)

    def _observe_resource_attribute(
        self, target: Any, attribute: str, resource: str
    ) -> tuple[bool, Any]:
        if target is None:
            return True, None
        try:
            return True, getattr(target, attribute, None)
        except BaseException as error:
            self._report_lifecycle_failure(
                f"inspect-{attribute}", resource, error
            )
            return False, None

    def _report_lifecycle_failure(
        self, operation: str, resource: str, error: BaseException | None
    ) -> None:
        detail = "returned False" if error is None else type(error).__name__
        try:
            callback = getattr(self.logger, "error", None)
        except BaseException:
            return
        if not callable(callback):
            return
        try:
            callback(f"trigger resource {operation} failed: {resource}/{detail}")
        except BaseException:
            return

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self.logger, level, None)
        if callable(callback):
            callback(message)

    def _allocate_command_id(self) -> str:
        candidate = self.command_id_factory()
        if type(candidate) is not str or not candidate:
            raise TypeError("command id factory must return a non-empty plain string")
        if self.model.admit_command_id(candidate):
            return candidate
        while True:
            replacement = uuid4().hex
            if self.model.admit_command_id(replacement):
                return replacement

    @staticmethod
    def normalize_barcode(text: Any) -> str:
        if text is None:
            return ""
        return str(text).strip()

    @staticmethod
    def barcode_invalid_characters(barcode: str) -> tuple[str, ...]:
        return tuple(
            character
            for character in barcode
            if character in _INVALID_FILENAME_CHARACTERS
        )

    def should_auto_commit_barcode(
        self, text: Any, first_ts: float, last_ts: float
    ) -> bool:
        normalized = self.normalize_barcode(text)
        if len(normalized) < self.model.minimum_auto_commit_length:
            return False
        return max(0.0, last_ts - first_ts) <= self.model.fast_input_max_seconds

    def _start_debounce(self) -> None:
        self.model.debounce_pending = True
        self.debounce_timer.start(self.model.debounce_interval_ms)

    def _stop_debounce(self) -> None:
        self.model.debounce_pending = False
        self.debounce_timer.stop()

    def reset_barcode_state(self, *, clear_dedup: bool = False) -> None:
        self._stop_debounce()
        self.model.reset_capture(clear_dedup=clear_dedup)

    def _trigger_display_name(self, source: str) -> str:
        if source.startswith("wedge") or source == "hid":
            return "扫码枪"
        if source == "optical":
            return "扫码枪/光电"
        if source == "tcp":
            return "TCP"
        if source == "shortcut":
            return "快捷键"
        return source

    def _external_start_allowed(self, source: str) -> bool:
        if not self.is_active:
            return False
        if source not in _EXTERNAL_SOURCES:
            return True
        display_name = self._trigger_display_name(source)
        if self.model.pending_start_command_id is not None:
            self._log("debug", f"触发请求等待工作流确认，忽略 {display_name} 触发")
            self.view.show_busy_rejection(display_name)
            return False
        if not self.model.external_trigger_available:
            self._log("debug", f"外部触发不可用，忽略 {display_name}")
            return False
        if not self.external_mode_available_provider():
            mode = self.acquisition_mode_provider()
            self._log(
                "warning", f"{display_name} 自动启动被阻止：当前工作模式 {mode} 不支持"
            )
            if source.startswith("wedge") or source == "hid":
                self.view.clear_serial_text()
            self.view.show_mode_rejection(display_name, mode)
            return False
        if self.workflow_active_provider():
            self._log("debug", f"工作流忙碌中，忽略 {display_name} 触发")
            self.view.show_busy_rejection(display_name)
            return False
        return True

    def ensure_external_mode_supported(self, trigger_source: str) -> bool:
        if not self.is_active:
            return False
        if self.external_mode_available_provider():
            return True
        mode = self.acquisition_mode_provider()
        self._log(
            "warning", f"{trigger_source} 自动启动被阻止：当前工作模式 {mode} 不支持"
        )
        self.clear_serial_for_external_rejection(trigger_source)
        self.view.show_mode_rejection(trigger_source, mode)
        return False

    def clear_serial_for_external_rejection(self, trigger_source: str) -> None:
        if "扫码枪" in str(trigger_source or ""):
            self.view.clear_serial_text()

    def load_selected_sn_regex_rule(self) -> Mapping[str, Any]:
        return self.regex_rule_loader()

    def validate_sn_regex(
        self,
        sn_text: Any = None,
        *,
        value_label: str = "实际 SN 内容",
        retry_hint: str = "请检查当前 SN 内容或切换正确规则后重试。",
        skip_sn_regex_validation: bool = False,
    ) -> bool:
        if not self.is_active:
            return False
        if skip_sn_regex_validation or not self.view.is_scanner_checked():
            return True
        if sn_text is None:
            sn_text = self.view.serial_text()
        text = "" if sn_text is None else str(sn_text)
        rule = self.regex_rule_loader()
        if not isinstance(rule, Mapping):
            rule = {"name": "未知规则", "pattern": ""}
        try:
            matched = re.fullmatch(rule["pattern"], text) is not None
        except (KeyError, TypeError, re.error) as error:
            self._log("error", f"SN regex rule is invalid: {error}")
            matched = False
        if matched:
            return True
        self._log(
            "warning",
            "SN regex validation failed. "
            f"rule={rule.get('name')}, pattern={rule.get('pattern')}, sn={text}",
        )
        self.view.show_regex_rejection(rule, text, value_label, retry_hint)
        return False

    def _publish_start(
        self,
        *,
        command_id: str,
        source: str,
        label: str,
        skip_sn_regex_validation: bool,
    ) -> StartTestRequested:
        generation = self.configuration_generation_provider()
        message = StartTestRequested(
            command_id=command_id,
            source=source,
            label=label,
            skip_sn_regex_validation=skip_sn_regex_validation,
            configuration_generation=generation,
        )
        previous_pending = self.model.pending_start_command_id
        if self.event_bus is not None:
            self.model.pending_start_command_id = command_id
        try:
            self.start_publisher(message)
        except Exception:
            self.model.pending_start_command_id = previous_pending
            raise
        return message

    def request_start(
        self,
        label: str = "not_labeled",
        *,
        source: str = "manual",
        skip_sn_regex_validation: bool = False,
    ) -> bool:
        if not self.is_active:
            return False
        command_id = self._allocate_command_id()
        if not self._external_start_allowed(source):
            return False
        if not self.validate_sn_regex(
            skip_sn_regex_validation=skip_sn_regex_validation
        ):
            return False
        self._publish_start(
            command_id=command_id,
            source=source,
            label=label,
            skip_sn_regex_validation=skip_sn_regex_validation,
        )
        return True

    def commit_barcode(self, barcode: Any, *, source: str = "wedge") -> bool:
        if not self.is_active:
            return False
        normalized = self.normalize_barcode(barcode)
        if not normalized:
            return False
        command = BarcodeCommitted(
            command_id=self._allocate_command_id(), source=source, barcode=normalized
        )
        if self.barcode_publisher is not None:
            self.barcode_publisher(command)
            return True
        return self.handle_barcode_committed(command)

    @pyqtSlot(object)
    def handle_barcode_committed(self, command: BarcodeCommitted) -> bool:
        if not self.is_active:
            return False
        if type(command) is not BarcodeCommitted:
            return False
        barcode = command.barcode
        if self.model.tcp_enabled or not self.view.is_scanner_checked():
            return False
        if not self._external_start_allowed(command.source):
            self.reset_barcode_state()
            return False

        now = self.monotonic()
        if (
            self.model.last_committed_barcode == barcode
            and now - self.model.last_committed_barcode_time
            < self.model.dedup_window_seconds
        ):
            self._log("debug", f"忽略重复条码提交: {barcode}")
            self.reset_barcode_state()
            return False

        invalid = self.barcode_invalid_characters(barcode)
        if invalid:
            self.view.show_invalid_barcode(barcode, invalid)
            self.view.clear_serial_text()
            self.model.sn_textchange_manual_guard = False
            self.reset_barcode_state(clear_dedup=True)
            return False
        if not self.validate_sn_regex(
            barcode,
            value_label="实际扫码内容",
            retry_hint="请检查扫码内容或切换正确规则后重新扫码。",
        ):
            self.view.clear_serial_text()
            self.model.sn_textchange_manual_guard = False
            self.reset_barcode_state(clear_dedup=True)
            return False

        focus = self.view.focus_widget()
        self._publish_start(
            command_id=command.command_id,
            source=command.source,
            label="not_labeled",
            skip_sn_regex_validation=False,
        )
        self.model.last_committed_barcode = barcode
        self.model.last_committed_barcode_time = now
        self.model.sn_textchange_manual_guard = False
        self.view.set_serial_text(barcode)
        self.reset_barcode_state()
        self.view.prepare_for_continuous_scan()
        if focus is not self.view.product_input and focus is not self.view.count_input:
            try:
                self.view.focus_serial_input()
            except (RuntimeError, TypeError):
                pass
        self._log("info", f"S/N 收到({command.source}): {barcode}，请求开始测试")
        return True

    def handle_barcode_return_pressed(self) -> bool:
        if not self.is_active:
            return False
        if not self.view.is_scanner_checked():
            return False
        self._stop_debounce()
        return self.commit_barcode(
            self.view.serial_text(), source="wedge_enter"
        )

    def handle_barcode_text_changed(self, _text: str) -> None:
        if not self.is_active:
            return
        if not self.view.is_scanner_checked() or not self.view.is_serial_enabled():
            return
        if self.model.sn_textchange_manual_guard:
            self._stop_debounce()
            self.model.barcode_first_char_ts = None
            self.model.barcode_last_char_ts = None
            if not self.normalize_barcode(self.view.serial_text()):
                self.model.sn_textchange_manual_guard = False
            return
        now = self.monotonic()
        if self.model.barcode_first_char_ts is None:
            self.model.barcode_first_char_ts = now
        self.model.barcode_last_char_ts = now
        self._start_debounce()

    def handle_barcode_debounce_timeout(self) -> bool:
        if not self.is_active:
            return False
        self.model.debounce_pending = False
        if not self.view.is_scanner_checked():
            return False
        if (
            self.model.barcode_capture_buffer
            and self.model.barcode_capture_first_ts is not None
            and self.model.barcode_capture_last_ts is not None
        ):
            text = self.normalize_barcode(self.model.barcode_capture_buffer)
            if self.should_auto_commit_barcode(
                text,
                self.model.barcode_capture_first_ts,
                self.model.barcode_capture_last_ts,
            ):
                return self.commit_barcode(text, source="wedge_global_debounce")
            self.model.reset_capture()
            return False

        text = self.normalize_barcode(self.view.serial_text())
        first = self.model.barcode_first_char_ts
        last = self.model.barcode_last_char_ts
        if not text:
            self.model.barcode_first_char_ts = None
            self.model.barcode_last_char_ts = None
            self.model.sn_textchange_manual_guard = False
            return False
        if first is None or last is None:
            return False
        if self.should_auto_commit_barcode(text, first, last):
            return self.commit_barcode(text, source="wedge_debounce")
        self.model.barcode_first_char_ts = None
        self.model.barcode_last_char_ts = None
        return False

    def handle_hid_barcode(self, barcode: Any) -> bool:
        if not self.is_active:
            return False
        self.model.hid_mode_active_until = (
            self.monotonic() + self.model.hid_suppression_seconds
        )
        self.reset_barcode_state()
        return self.commit_barcode(barcode, source="hid")

    def handle_optical_trigger(self) -> bool:
        if not self.is_active:
            return False
        return self.request_start(source="optical")

    def handle_shortcut_trigger(self) -> bool:
        if not self.is_active:
            return False
        if self.model.shortcut_processing:
            return False
        self.model.shortcut_processing = True
        try:
            return self.request_start(source="shortcut")
        finally:
            self.model.shortcut_processing = False

    def handle_keypress(self, _obj: Any, event: Any) -> bool | None:
        if not self.is_active:
            return None
        if event.type() != QEvent.KeyPress or not self.view.is_scanner_checked():
            return None
        now = self.monotonic()
        focus = self.view.focus_widget()
        character = event.text()
        if (
            character
            and character.isprintable()
            and now < self.model.hid_mode_active_until
            and focus is not self.view.product_input
            and focus is not self.view.count_input
        ):
            return True
        if focus is self.view.serial_input:
            if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
                return True
            if event.key() in _MANUAL_EDIT_KEYS:
                self.model.sn_textchange_manual_guard = True
                self.reset_barcode_state()
                return None
            if character and character.isprintable() and not character.isspace():
                current_text = self.view.serial_text()
                selected_text = getattr(self.view.serial_input, "selectedText", lambda: "")()
                if not current_text or (
                    selected_text and len(selected_text) == len(current_text)
                ):
                    self.model.sn_textchange_manual_guard = False
            return None
        if focus is self.view.product_input or focus is self.view.count_input:
            return None
        protected = getattr(self.view, "is_protected_input_widget", lambda _widget: False)
        if protected(focus):
            return None
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._stop_debounce()
            buffer = self.model.barcode_capture_buffer
            first = self.model.barcode_capture_first_ts
            last = self.model.barcode_capture_last_ts
            if buffer and first is not None and last is not None:
                text = self.normalize_barcode(buffer)
                if self.should_auto_commit_barcode(text, first, last):
                    self.commit_barcode(text, source="wedge_global_enter")
                    return True
            self.model.reset_capture()
            return None
        if character and character.isprintable() and not character.isspace():
            last = self.model.barcode_capture_last_ts
            if self.model.barcode_capture_buffer and (
                last is None or now - last >= 0.05
            ):
                self.model.reset_capture()
            if not self.model.barcode_capture_buffer:
                self.model.barcode_capture_first_ts = now
            self.model.barcode_capture_last_ts = now
            self.model.barcode_capture_buffer += character
            self._start_debounce()
            return True
        return None

    def set_scanner_checked(self, checked: bool) -> bool:
        if not self.is_active:
            return False
        checked = bool(checked)
        manager = self.hardware_manager
        if checked:
            if manager is not None:
                try:
                    if not manager.ensure_config_loaded():
                        self._log(
                            "warning",
                            "无法加载扫码枪/光电开关配置，将进入扫码枪自动识别模式。",
                        )
                    started = bool(manager.start())
                except Exception as error:
                    self._log("error", f"硬件监听启动失败: {error}")
                    started = False
            else:
                started = True
            setter = getattr(self.view, "set_serial_enabled", None)
            if callable(setter):
                setter(True)
            try:
                self.view.focus_serial_input(select_all=True)
            except (RuntimeError, TypeError):
                pass
            return started
        self.view.clear_serial_text()
        setter = getattr(self.view, "set_serial_enabled", None)
        if callable(setter):
            setter(False)
        if manager is not None:
            try:
                manager.stop()
            except Exception as error:
                self._log("error", f"硬件监听停止失败: {error}")
        self.reset_barcode_state(clear_dedup=True)
        return True

    def bind_hardware_signals(self) -> None:
        if not self.is_active or self._hardware_connections:
            return
        manager = self.hardware_manager
        if manager is None:
            return
        for signal, slot in (
            (manager.sig_barcode, self.handle_hid_barcode),
            (manager.sig_trigger, self.handle_optical_trigger),
        ):
            self._hardware_connections.append(
                self._connect_tracked_signal(
                    manager,
                    signal,
                    slot,
                    "hardware-connection",
                    Qt.QueuedConnection,
                )
            )

    def bind_shortcut_signal(self) -> None:
        if not self.is_active or self._shortcut_connections:
            return
        manager = self.shortcut_manager
        signal = None if manager is None else getattr(manager, "sig_triggered", None)
        if signal is None:
            return
        slot = self.handle_shortcut_trigger
        self._shortcut_connections.append(
            self._connect_tracked_signal(
                manager,
                signal,
                slot,
                "shortcut-connection",
                Qt.QueuedConnection,
            )
        )

    def shutdown(self) -> None:
        self.disconnect()

    def _tcp_callback(
        self, lifecycle_generation: int, server_token: str
    ) -> Callable[[Any], str]:
        controller_ref = ref(self)

        def callback(info: Any) -> str:
            owner = controller_ref()
            if owner is None or not owner._generation_is_active(
                lifecycle_generation
            ):
                return "error, trigger controller unavailable"
            if not owner._owns_active_tcp_server(
                lifecycle_generation=lifecycle_generation,
                server_token=server_token,
            ):
                return "error, tcp server inactive"
            try:
                ok, data = owner.tcp_message_validator(info)
            except Exception as error:
                owner._log("error", f"TCP message validation failed: {error}")
                return "error, message format error"
            if not ok:
                return str(data)
            try:
                request_type = int(data.get("RequestType"))
                timestamp = data.get("Timestamp")
                if type(timestamp) is not str or not timestamp:
                    raise TypeError("timestamp must be non-empty plain text")
                content = data.get("RequestContent", {})
                if not isinstance(content, Mapping):
                    content = {}
                label = content.get("Label") or content.get("label") or "not_labeled"
                package = TcpTriggerPackage(
                    request_type,
                    timestamp,
                    str(label),
                    lifecycle_generation,
                    server_token,
                )
            except (AttributeError, TypeError, ValueError) as error:
                owner._log("error", f"TCP message normalization failed: {error}")
                return "error, message format error"
            request_id = f"{package.request_type}@{package.timestamp}"
            if not owner.model.admit_tcp_request_id(
                request_id,
                lifecycle_generation=lifecycle_generation,
                server_token=server_token,
            ):
                if owner._owns_active_tcp_server(
                    lifecycle_generation=lifecycle_generation,
                    server_token=server_token,
                ):
                    return "pass"
                return "error, tcp server inactive"
            owner._tcp_channel.package_received.emit(package)
            return "ok"

        return callback

    def _owns_active_tcp_server(
        self,
        *,
        lifecycle_generation: int | None = None,
        server_token: str | None = None,
    ) -> bool:
        if lifecycle_generation is None:
            lifecycle_generation = self.lifecycle_generation
        if server_token is None:
            server_token = self.model.tcp_server_token
        if type(server_token) is not str:
            return False
        server = self.model.tcp_server
        return bool(
            self._generation_is_active(lifecycle_generation)
            and self.model.tcp_server_is_current(
                lifecycle_generation=lifecycle_generation,
                server_token=server_token,
            )
            and server is not None
            and self.tcp_mirror_getter() is server
        )

    @pyqtSlot(object)
    def handle_tcp_package(self, data: TcpTriggerPackage) -> bool:
        if not self.is_active:
            return False
        if type(data) is not TcpTriggerPackage:
            return False
        if not self._owns_active_tcp_server(
            lifecycle_generation=data.lifecycle_generation,
            server_token=data.server_token,
        ):
            self._log("debug", "忽略已停止或已被替换的 TCP 服务消息")
            return False
        server = self.model.tcp_server
        self.model.tcp_connected = bool(
            server is not None and getattr(server, "client_address", None) is not None
        )
        if data.request_type != RequestTypeEnum.RUN_TEST.value:
            return False
        return self.handle_tcp_run_test(
            label=data.label, skip_sn_regex_validation=True
        )

    def handle_tcp_run_test(
        self,
        label: str = "not_labeled",
        *,
        skip_sn_regex_validation: bool = False,
    ) -> bool:
        if not self.is_active:
            return False
        return self.request_start(
            label=label,
            source="tcp",
            skip_sn_regex_validation=skip_sn_regex_validation,
        )

    def set_tcp_enabled(
        self, enabled: bool, *, host: Any = None, port: Any = None
    ) -> bool:
        lifecycle_generation = self._capture_active_generation()
        if lifecycle_generation is None:
            return False
        enabled = bool(enabled)
        if not enabled:
            return self.stop_tcp()
        if host is None or port is None:
            host, port = self.tcp_config_reader()
        host = str(host)
        if (
            self.model.tcp_running
            and self.model.tcp_server is not None
            and self.model.tcp_host == host
            and self.model.tcp_port == port
            and self.tcp_mirror_getter() is self.model.tcp_server
            and not self._tcp_stop_journal
            and self._tcp_server_active_state(self.model.tcp_server) is not False
        ):
            return True
        if not self._stop_tcp_cleanup():
            return False
        existing = self.tcp_mirror_getter()
        if existing is not None:
            try:
                stopped = existing.stop()
            except BaseException as error:
                self._report_lifecycle_failure("stop", "tcp-mirror", error)
                return False
            if stopped is False:
                self._report_lifecycle_failure("stop", "tcp-mirror", None)
                return False
            if not self.tcp_mirror_setter(None):
                return False
        if not self._generation_is_active(lifecycle_generation):
            return False
        server_token = uuid4().hex
        server = self.tcp_server_factory(
            host=host,
            port=port,
            callback=self._tcp_callback(lifecycle_generation, server_token),
        )
        if not self.model.activate_tcp_server(
            server,
            lifecycle_generation=lifecycle_generation,
            server_token=server_token,
            host=host,
            port=port,
        ):
            self._report_lifecycle_failure(
                "admission", "tcp-model-identity", None
            )
            self._stop_exact_unadmitted_tcp_server(server)
            return False
        if not self.tcp_mirror_setter(server):
            self._stop_tcp_cleanup()
            return False
        try:
            started = server.start()
        except BaseException as error:
            self._report_lifecycle_failure("start", "tcp", error)
            self._stop_tcp_cleanup()
            self.view.present_tcp_state(False)
            return False
        if started is False:
            self._report_lifecycle_failure("start", "tcp", None)
            self._stop_tcp_cleanup()
            self.view.present_tcp_state(False)
            return False
        if self._tcp_server_active_state(server) is False:
            self._report_lifecycle_failure("verify-start", "tcp", None)
            self._stop_tcp_cleanup()
            self.view.present_tcp_state(False)
            return False
        if not self._generation_is_active(lifecycle_generation):
            self._stop_tcp_cleanup()
            return False
        self.model.tcp_connected = bool(
            getattr(server, "client_address", None) is not None
        )
        self.view.present_tcp_state(True)
        return True

    def _stop_exact_unadmitted_tcp_server(self, server: Any) -> bool:
        identity = id(server)
        with self._lifecycle_lock:
            if not self._admit_tcp_targets_locked(server):
                self._report_lifecycle_failure(
                    "admission-capacity", "tcp", None
                )
                return False
        try:
            stop = getattr(server, "stop", None)
        except BaseException as error:
            self._report_lifecycle_failure("stop-port", "tcp", error)
            return False
        if not callable(stop):
            self._report_lifecycle_failure("stop-port", "tcp", None)
            return False
        try:
            stopped = stop()
        except BaseException as error:
            self._report_lifecycle_failure("stop", "tcp", error)
            return False
        if stopped is False or self._tcp_server_active_state(server) is True:
            self._report_lifecycle_failure("verify-stop", "tcp", None)
            return False
        self._tcp_stop_journal.pop(identity, None)
        return True

    def stop_tcp(self) -> bool:
        return self._stop_tcp_cleanup()

    def _stop_tcp_cleanup(self) -> bool:
        model_server = self.model.tcp_server
        mirror_ok, mirror_server = self._read_tcp_mirror()
        if not mirror_ok:
            return False
        mirror_verified_stopped = bool(
            mirror_server is not None
            and self._tcp_server_active_state(mirror_server) is False
        )
        with self._lifecycle_lock:
            self._retry_tcp_mirror_reservations_locked(
                actual=mirror_server,
                final_observed=True,
                actual_verified_stopped=mirror_verified_stopped,
            )
        servers = []
        for server in (model_server, mirror_server):
            if server is not None and all(server is not item for item in servers):
                servers.append(server)
        for server in servers:
            identity = id(server)
            if identity in self._tcp_stop_completed_handles:
                continue
            with self._lifecycle_lock:
                if not self._admit_tcp_targets_locked(server):
                    self._report_lifecycle_failure(
                        "admission-capacity", "tcp", None
                    )
                    return False

        stopped_all = True
        for identity, server in tuple(self._tcp_stop_journal.items()):
            try:
                stop = getattr(server, "stop", None)
            except BaseException as error:
                self._report_lifecycle_failure("stop-port", "tcp", error)
                stopped_all = False
                self._journal_reentrant_tcp_servers()
                continue
            if not callable(stop):
                self._report_lifecycle_failure("stop-port", "tcp", None)
                stopped_all = False
                self._journal_reentrant_tcp_servers()
                continue
            try:
                stopped = stop()
            except BaseException as error:
                self._report_lifecycle_failure("stop", "tcp", error)
                stopped_all = False
                self._journal_reentrant_tcp_servers()
                continue
            if stopped is False:
                self._report_lifecycle_failure("stop", "tcp", None)
                stopped_all = False
                self._journal_reentrant_tcp_servers()
                continue
            if self._tcp_server_active_state(server) is True:
                self._report_lifecycle_failure("verify-stop", "tcp", None)
                stopped_all = False
                self._journal_reentrant_tcp_servers()
                continue
            self._tcp_stop_journal.pop(identity, None)
            self._tcp_stop_completed_handles[identity] = None
            if self._journal_reentrant_tcp_servers():
                stopped_all = False
        with self._lifecycle_lock:
            if not self._retry_tcp_mirror_reservations_locked(
                actual=mirror_server,
                final_observed=True,
                actual_verified_stopped=mirror_verified_stopped,
            ):
                stopped_all = False
        current_model = self.model.tcp_server
        current_mirror_ok, current_mirror = self._read_tcp_mirror()
        if not current_mirror_ok:
            return False
        exposed_identities = {
            id(server)
            for server in (current_model, current_mirror)
            if server is not None
        }
        for identity in tuple(self._tcp_stop_completed_handles):
            if identity not in exposed_identities:
                self._tcp_stop_completed_handles.pop(identity, None)
        if not stopped_all or self._tcp_stop_journal:
            return False
        release_model = current_model
        release_mirror = current_mirror
        if release_mirror is not None:
            self._tcp_mirror_release_journal.setdefault(
                id(release_mirror), release_mirror
            )
        if self._tcp_mirror_release_journal:
            released, _previous, actual_mirror = self._write_tcp_mirror_identity(
                None
            )
            if not released:
                return False
            if release_mirror is not None and actual_mirror is not release_mirror:
                self._tcp_mirror_release_journal.pop(id(release_mirror), None)
            if actual_mirror is not None:
                self._journal_reentrant_tcp_servers()
                return False
            self._tcp_mirror_release_journal.clear()
        if self.model.tcp_server is not release_model:
            self._journal_reentrant_tcp_servers()
            return False
        if release_model is not None:
            self._tcp_model_release_journal.setdefault(
                id(release_model), release_model
            )
        if self._tcp_model_release_journal:
            try:
                invalidated = self.model.invalidate_tcp_server()
            except BaseException as error:
                self._report_lifecycle_failure("release", "tcp-model", error)
                self._journal_reentrant_tcp_servers()
                return False
            actual_model = self.model.tcp_server
            if invalidated is False:
                self._journal_reentrant_tcp_servers()
                return False
            if release_model is not None and actual_model is not release_model:
                self._tcp_model_release_journal.pop(id(release_model), None)
            if actual_model is not None:
                self._journal_reentrant_tcp_servers()
                return False
            self._tcp_model_release_journal.clear()
        self._tcp_stop_completed_handles.clear()
        self.view.present_tcp_state(False)
        return True

    def _read_tcp_mirror(self) -> tuple[bool, Any]:
        try:
            return True, self.tcp_mirror_getter()
        except BaseException as error:
            self._report_lifecycle_failure("inspect", "tcp-mirror", error)
            return False, None

    def _journal_reentrant_tcp_servers(self) -> bool:
        discovered = False
        mirror_ok, mirror_server = self._read_tcp_mirror()
        if not mirror_ok:
            return True
        for server in (self.model.tcp_server, mirror_server):
            if server is None:
                continue
            identity = id(server)
            if identity in self._tcp_stop_completed_handles:
                continue
            if identity not in self._tcp_stop_journal:
                with self._lifecycle_lock:
                    if self._admit_tcp_targets_locked(server):
                        discovered = True
                    else:
                        self._report_lifecycle_failure(
                            "reentrant-capacity", "tcp", None
                        )
                        discovered = True
        return discovered

    def _tcp_server_active_state(self, server: Any) -> bool | None:
        if server is None:
            return False
        for attribute in ("is_running", "is_active", "running", "active"):
            ok, value = self._observe_resource_attribute(server, attribute, "tcp")
            if not ok:
                return True
            if callable(value):
                try:
                    value = value()
                except BaseException as error:
                    self._report_lifecycle_failure("inspect", "tcp", error)
                    return True
            if type(value) is bool:
                return value
        ok, stop_flag = self._observe_resource_attribute(
            server, "stop_flag", "tcp"
        )
        if not ok:
            return True
        if type(stop_flag) is bool:
            return stop_flag
        return None

    def open_tcp_configuration(self) -> bool:
        generation = self._capture_active_generation()
        if generation is None:
            return False
        if self._tcp_configuration_dialog_pending:
            return False
        self._tcp_configuration_dialog_pending = True
        lifecycle = {"resolved": False}
        controller_ref = ref(self)

        def accepted(result: Any) -> None:
            owner = controller_ref()
            if lifecycle["resolved"]:
                return
            lifecycle["resolved"] = True
            if owner is None or not owner._generation_is_active(generation):
                return
            owner._tcp_configuration_dialog_pending = False
            if type(result) is not tuple or len(result) != 3:
                owner._log("error", "TCP 配置对话框返回了无效结果")
                return
            enabled, host, port = result
            if owner.tcp_config_writer is not None:
                owner.tcp_config_writer(host, port)
            owner.set_tcp_enabled(bool(enabled), host=host, port=port)

        def rejected() -> None:
            owner = controller_ref()
            if lifecycle["resolved"]:
                return
            lifecycle["resolved"] = True
            if owner is not None and owner._generation_is_active(generation):
                owner._tcp_configuration_dialog_pending = False

        try:
            opened = self.view.open_tcp_dialog(
                self.model.tcp_enabled,
                self.model.tcp_host,
                self.model.tcp_port,
                accepted,
                rejected,
            )
        except Exception:
            self._tcp_configuration_dialog_pending = False
            raise
        if not opened and not lifecycle["resolved"]:
            self._tcp_configuration_dialog_pending = False
        return bool(opened)

    @pyqtSlot(object)
    def handle_workflow_rejection(self, event: WorkflowCommandRejected) -> bool:
        if not self.is_active:
            return False
        if type(event) is not WorkflowCommandRejected:
            return False
        if self.model.pending_start_command_id != event.command_id:
            return False
        self.model.pending_start_command_id = None
        self._log("debug", f"触发命令被工作流拒绝: {event.reason}")
        self.view.show_workflow_rejection(event.reason)
        return True

    @pyqtSlot(object)
    def handle_workflow_state_changed(self, event: WorkflowStateChanged) -> bool:
        if not self.is_active:
            return False
        if type(event) is not WorkflowStateChanged:
            return False
        if event.new_phase != "IDLE":
            self.model.pending_start_command_id = None
        return True
