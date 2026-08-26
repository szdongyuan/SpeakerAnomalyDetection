from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import ref

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_messages import ResourceLifecycleRequested


@dataclass(frozen=True, slots=True)
class SequenceResourceLifecycleRegistration:
    """Exact weak ownership receipt for one permanent lifecycle recipient."""

    operation: str
    name: str
    token: object
    _owner_ref: object
    owner_identity: int
    registration_generation: int

    @classmethod
    def capture(cls, operation, name, token, owner):
        if (
            type(operation) is not str
            or not operation
            or type(name) is not str
            or not name
            or not isinstance(owner, QObject)
            or sip.isdeleted(owner)
            or getattr(token, "operation", None) != operation
            or getattr(token, "name", None) != name
            or getattr(token, "owner_identity", None) != id(owner)
            or type(getattr(token, "version", None)) is not int
        ):
            return None
        return cls(
            operation,
            name,
            token,
            ref(owner),
            id(owner),
            token.version,
        )

    def owner(self):
        return self._owner_ref()

    def owner_is_permanently_retired(self) -> bool:
        owner = self.owner()
        if owner is None:
            return True
        try:
            return isinstance(owner, QObject) and sip.isdeleted(owner)
        except BaseException:
            return False


class SequenceResourceLifecycleModel:
    """Canonical reusable-resource and final shutdown state."""

    def __init__(self) -> None:
        self._reusable_resource_lock = RLock()
        self._reusable_resource_epoch = 0
        self._reusable_cleanup_event_pending = False
        self._reusable_cleanup_generation = 0
        self._reusable_cleanup_attempt = 0
        self._reusable_cleanup_retry_limit = 3
        self._reusable_cleanup_retry_delays_ms = (0, 25, 100)
        self._reusable_cleanup_dispatch_active = False
        self._reusable_pending_identity_limit = 64
        self._reusable_operation_generation = 0
        self._reusable_operation_in_progress = False
        self._reusable_operation_kind = None
        self._reusable_operation_desired = "ACTIVE"
        self._reusable_operation_continuation_pending = False
        self._reusable_detached_pending_stops = {}
        self._reusable_detached_pending_resources = {}
        self._reusable_detached_cleanup_tokens = {}
        self._reusable_child_suspended = False
        self._reusable_resource_snapshot = None
        self._reusable_resource_state = "ACTIVE"
        self._reusable_resource_journal = {}
        self._reusable_trusted_running_ids = {}
        self._reusable_suspend_completed = set()
        self._reusable_resume_completed = set()
        self._reusable_resume_pending = False
        self._shortcut_mgr = None
        self._hw_manager = None
        self._tcp_resource_port = None
        self._tcp_mirror_registration_retired = False
        self._tcp_mirror_owner_token = None
        self._shutdown_prepared_generation = None
        self._shutdown_finalized_generation = None
        self._shutdown_delivery_completed_generation = None
        self._shutdown_dispatchers_closed_generation = None
        self._shutdown_cleanup_trace = []
        self._shutdown_cleanup_steps_completed = set()
        self._lightweight_cleanup_done = False
        self._resource_lifecycle_registrations = []
        self._resource_lifecycle_registration_limit = 32
        self._resource_lifecycle_resolution_failures = []
        self._resource_lifecycle_request_lock = RLock()
        self._resource_lifecycle_requests = {}
        self._resource_lifecycle_request_limit = 32

    @property
    def resource_lifecycle_registrations(self):
        return tuple(self._resource_lifecycle_registrations)

    def retain_resource_lifecycle_registration(
        self, operation, name, token, owner
    ) -> bool:
        registration = SequenceResourceLifecycleRegistration.capture(
            operation, name, token, owner
        )
        if registration is None:
            return False
        if any(
            current.token is registration.token
            for current in self._resource_lifecycle_registrations
        ):
            return True
        if (
            len(self._resource_lifecycle_registrations)
            >= self._resource_lifecycle_registration_limit
        ):
            return False
        self._resource_lifecycle_registrations.append(registration)
        return True

    def resource_lifecycle_request(
        self, shutdown_generation: int, operation: str
    ):
        key = (shutdown_generation, operation)
        with self._resource_lifecycle_request_lock:
            current = self._resource_lifecycle_requests.get(key)
            if current is not None:
                return current
            if (
                len(self._resource_lifecycle_requests)
                >= self._resource_lifecycle_request_limit
            ):
                return None
            try:
                request = ResourceLifecycleRequested(
                    shutdown_generation, operation
                )
            except (TypeError, ValueError):
                return None
            self._resource_lifecycle_requests[key] = request
            return request

    def resolve_retired_resource_lifecycle_registrations(
        self, bus, request
    ) -> int:
        resolve = getattr(
            bus, "resolve_retired_resource_lifecycle_recipient", None
        )
        if not callable(resolve):
            return 0
        resolved = 0
        for registration in tuple(self._resource_lifecycle_registrations):
            if (
                registration.operation != request.operation
                or not registration.owner_is_permanently_retired()
            ):
                continue
            try:
                accepted = resolve(
                    request,
                    registration.token,
                    owner_identity=registration.owner_identity,
                    registration_generation=(
                        registration.registration_generation
                    ),
                )
            except BaseException:
                accepted = False
                self._resource_lifecycle_resolution_failures.append(
                    (
                        request.shutdown_generation,
                        registration.operation,
                        registration.name,
                    )
                )
                del self._resource_lifecycle_resolution_failures[
                    : -self._resource_lifecycle_registration_limit
                ]
            if accepted is True:
                resolved += 1
        return resolved


class SequenceResourceLifecycleView:
    """Weak Qt/native adapter used by the lifecycle owner."""

    def __init__(self, owner: QObject) -> None:
        self._owner_ref = ref(owner)

    def owner(self):
        return self._owner_ref()

    def remove_application_event_filter(self) -> None:
        owner = self.owner()
        app = QApplication.instance()
        if owner is None or app is None:
            return
        try:
            app.removeEventFilter(owner)
        except (RuntimeError, TypeError):
            return

    def disconnect_barcode_inputs(self) -> None:
        owner = self.owner()
        if owner is None:
            return
        router = getattr(owner, "_barcode_router", None)
        serial_input = getattr(owner, "lineedit_s_or_n", None)
        if router is None or serial_input is None:
            return
        for signal, slot in (
            (serial_input.returnPressed, router.on_barcode_return_pressed),
            (serial_input.textChanged, router.on_barcode_text_changed),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                continue

    def close_analysis_windows(self) -> None:
        owner = self.owner()
        close = None if owner is None else getattr(
            owner, "_close_analysis_windows", None
        )
        if callable(close):
            close()

    def close_application_subwindows(self) -> None:
        owner = self.owner()
        if owner is None:
            return
        try:
            top_level = owner.window()
        except (RuntimeError, TypeError):
            return
        close = getattr(top_level, "_close_all_subwindows", None)
        if callable(close):
            close()


@dataclass(frozen=True, slots=True)
class _CanonicalTcpMirrorOwnerToken:
    identity: int
    generation: int


class _CanonicalTcpMirrorState:
    """One process-wide mirror with weak owner admission participants."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._value = None
        self._owners = {}
        self._owner_generation = 0
        self._owner_limit = 128
        self._write_in_progress = False

    def read(self):
        with self._lock:
            return self._value

    def register_owner(self, owner):
        identity = id(owner)
        with self._lock:
            self._live_owners_locked()
            current = self._owners.get(identity)
            if current is not None and current[1]() is owner:
                return current[0]
            if len(self._owners) >= self._owner_limit:
                return None
            self._owner_generation += 1
            token = _CanonicalTcpMirrorOwnerToken(
                identity, self._owner_generation
            )

            def owner_collected(owner_ref) -> None:
                self._retire_owner(token, owner_ref)

            try:
                owner_ref = ref(owner, owner_collected)
            except TypeError:
                return None
            self._owners[identity] = (token, owner_ref)
        return token

    def register(self, owner) -> bool:
        return self.register_owner(owner) is not None

    def _retire_owner(self, token, owner_ref=None) -> bool:
        with self._lock:
            current = self._owners.get(token.identity)
            if current is None or current[0] is not token:
                return False
            if owner_ref is not None and current[1] is not owner_ref:
                return False
            self._owners.pop(token.identity, None)
            return True

    def unregister(self, owner_or_token, token=None) -> bool:
        if type(owner_or_token) is _CanonicalTcpMirrorOwnerToken:
            if token is not None:
                return False
            return self._retire_owner(owner_or_token)
        owner = owner_or_token
        identity = id(owner)
        with self._lock:
            current = self._owners.get(identity)
            if current is None:
                return True
            expected = token if token is not None else current[0]
            if (
                type(expected) is not _CanonicalTcpMirrorOwnerToken
                or current[0] is not expected
                or current[1]() is not owner
            ):
                return False
            self._owners.pop(identity, None)
            return True

    @property
    def owner_count(self) -> int:
        with self._lock:
            self._live_owners_locked()
            return len(self._owners)

    def _live_owners_locked(self):
        owners = []
        for identity, entry in tuple(self._owners.items()):
            _token, owner_ref = entry
            owner = owner_ref()
            if owner is None:
                self._owners.pop(identity, None)
                continue
            if isinstance(owner, QObject):
                try:
                    if sip.isdeleted(owner):
                        self._owners.pop(identity, None)
                        continue
                except BaseException:
                    self._owners.pop(identity, None)
                    continue
            owners.append(owner)
        owners.sort(key=id)
        return owners

    @staticmethod
    def _owner_tcp_port(owner):
        try:
            return getattr(owner, "_tcp_resource_port", None)
        except BaseException:
            return None

    @staticmethod
    def _owner_locks(owners):
        owner_locks = {}
        controller_locks = {}
        for owner in owners:
            try:
                owner_lock = getattr(owner, "_reusable_resource_lock", None)
            except BaseException:
                owner_lock = None
            if owner_lock is not None:
                owner_locks[id(owner_lock)] = owner_lock
            controller = _CanonicalTcpMirrorState._owner_tcp_port(owner)
            try:
                controller_lock = getattr(controller, "_lifecycle_lock", None)
            except BaseException:
                controller_lock = None
            if controller_lock is not None:
                controller_locks[id(controller_lock)] = controller_lock
        locks = [
            owner_locks[identity] for identity in sorted(owner_locks)
        ]
        locks.extend(
            controller_locks[identity]
            for identity in sorted(controller_locks)
            if identity not in owner_locks
        )
        return tuple(locks)

    @staticmethod
    def _admit_controller_locked(controller, previous, current) -> bool:
        if controller is None:
            return True
        try:
            admission = getattr(
                controller,
                "_admit_canonical_tcp_mirror_identity_locked",
                None,
            )
        except BaseException:
            return False
        if callable(admission):
            try:
                return admission(previous, current) is True
            except BaseException:
                return False
        try:
            controller._resource_identity_epoch = (
                getattr(controller, "_resource_identity_epoch", 0) + 1
            )
            state = getattr(controller, "_lifecycle_state", "ACTIVE")
        except BaseException:
            return False
        if current is not None and state in {
            "DISCONNECTING",
            "FINALIZING",
            "INACTIVE",
        }:
            return False
        if state == "DISCONNECTING":
            try:
                completed = getattr(
                    controller, "_tcp_stop_completed_handles", {}
                )
                journal = getattr(controller, "_tcp_stop_journal", None)
                if not isinstance(journal, dict):
                    return False
                for target in (previous, current):
                    if target is None or id(target) in completed:
                        continue
                    journal.setdefault(id(target), target)
            except BaseException:
                return False
        return True

    @staticmethod
    def _admit_owner_locked(owner, previous, current) -> bool:
        try:
            owner._reusable_resource_epoch = (
                getattr(owner, "_reusable_resource_epoch", 0) + 1
            )
            reusable_state = getattr(
                owner, "_reusable_resource_state", "ACTIVE"
            )
        except BaseException:
            return False
        if reusable_state in {
            "SUSPENDING",
            "SUSPENDED",
            "RESUMING",
        }:
            return True
        controller = _CanonicalTcpMirrorState._owner_tcp_port(owner)
        controller_admitted = _CanonicalTcpMirrorState._admit_controller_locked(
            controller, previous, current
        )
        return controller_admitted

    def _registration_is_current_locked(self, owner, token) -> bool:
        if type(token) is not _CanonicalTcpMirrorOwnerToken:
            return False
        current = self._owners.get(id(owner))
        return (
            current is not None
            and current[0] is token
            and current[1]() is owner
        )

    def read_registered(self, owner, token):
        with self._lock:
            self._live_owners_locked()
            if not self._registration_is_current_locked(owner, token):
                return False, None
            return True, self._value

    def write_registered(self, owner, token, value) -> bool:
        with self._lock:
            self._live_owners_locked()
            if not self._registration_is_current_locked(owner, token):
                return False
            return self._write_locked(value)

    def write(self, value) -> bool:
        with self._lock:
            return self._write_locked(value)

    def _write_locked(self, value) -> bool:
        previous = self._value
        if previous is value:
            return True
        if self._write_in_progress:
            return False
        self._write_in_progress = True
        owners = self._live_owners_locked()
        locks = self._owner_locks(owners)
        acquired = []
        try:
            for lock in locks:
                try:
                    lock.acquire()
                except BaseException:
                    return False
                acquired.append(lock)
            admitted = True
            for owner in owners:
                if not self._admit_owner_locked(owner, previous, value):
                    admitted = False
            if not admitted:
                return False
            self._value = value
            return True
        finally:
            for lock in reversed(acquired):
                try:
                    lock.release()
                except BaseException:
                    continue
            self._write_in_progress = False


_CANONICAL_TCP_MIRROR_STATE = _CanonicalTcpMirrorState()


class SequenceResourceLifecycleTcpMirrorPort:
    """Exact-registration access to the process-wide TCP compatibility mirror."""

    def __init__(self, owner, registration_token) -> None:
        self._owner_ref = ref(owner)
        self._registration_token = registration_token

    def read(self):
        owner = self._owner_ref()
        if owner is None:
            return None
        admitted, value = _CANONICAL_TCP_MIRROR_STATE.read_registered(
            owner, self._registration_token
        )
        return value if admitted else None

    def write(self, server) -> bool:
        owner = self._owner_ref()
        if owner is None:
            return False
        return _CANONICAL_TCP_MIRROR_STATE.write_registered(
            owner, self._registration_token, server
        )


class _ReusableCleanupNativeGuard:
    """A queued callback lifetime bit that never retains its QObject owner."""

    def __init__(self):
        self._lock = RLock()
        self._alive = True

    def invalidate(self, *_args):
        with self._lock:
            self._alive = False

    @property
    def alive(self):
        with self._lock:
            return self._alive


class _ReusableCleanupDispatcher(QObject):
    """Move cleanup continuations onto the QObject/application thread."""

    requested = pyqtSignal(int, int)

    def __init__(self, owner, guard):
        super().__init__()
        self._owner_ref = ref(owner)
        self._guard = guard

    def activate(self):
        self.requested.connect(self._schedule, Qt.QueuedConnection)

    def _schedule(self, generation, delay_ms):
        owner_ref = self._owner_ref
        guard = self._guard

        def deliver():
            if not guard.alive:
                return
            owner = owner_ref()
            if owner is None:
                return
            try:
                if sip.isdeleted(owner):
                    guard.invalidate()
                    return
            except BaseException:
                return
            SequenceResourceLifecycleController._run_queued_reusable_cleanup(
                owner, generation=generation
            )

        QTimer.singleShot(max(0, int(delay_ms)), deliver)


class _ReusableDetachedProvisional:
    """Stable broker entry for one exact detached target identity."""

    PROVISIONAL = "PROVISIONAL"
    PROMOTING = "PROMOTING"
    TOKEN = "TOKEN"

    __slots__ = (
        "owner_ref",
        "resource",
        "target",
        "token_factory",
        "retry_limit",
        "delays_ms",
        "state",
        "token",
        "start_inflight",
        "demand_sequence",
        "demand_pending",
        "demand_acceptance",
        "teardown_pending",
        "ensure_queued",
    )

    def __init__(
        self,
        owner,
        resource,
        target,
        token_factory,
        retry_limit,
        delays_ms,
    ):
        try:
            self.owner_ref = ref(owner)
        except TypeError:
            self.owner_ref = lambda: None
        self.resource = resource
        self.target = target
        self.token_factory = token_factory
        self.retry_limit = max(1, int(retry_limit))
        self.delays_ms = tuple(delays_ms) or (0,)
        self.state = self.PROVISIONAL
        self.token = None
        self.start_inflight = True
        self.demand_sequence = 0
        self.demand_pending = False
        self.demand_acceptance = None
        self.teardown_pending = False
        self.ensure_queued = False

    def belongs_to(self, owner, resource, target) -> bool:
        return bool(
            self.owner_ref() is owner
            and self.resource == resource
            and self.target is target
        )

    def request(self, *, teardown=False, acceptance=None) -> None:
        self.demand_sequence += 1
        if not self.demand_pending:
            self.demand_acceptance = acceptance
        elif self.demand_acceptance is not None and acceptance is not None:
            # Every pending request was accepted by an already-active burst;
            # the newest receipt subsumes the older coalesced receipts.
            self.demand_acceptance = acceptance
        else:
            # At least one request arrived while no burst was active and must
            # be delivered normally to allocate a new budget.
            self.demand_acceptance = None
        self.demand_pending = True
        self.teardown_pending = bool(self.teardown_pending or teardown)

    def prepare_cleanup(
        self, token_factory, retry_limit, delays_ms, *, teardown=False
    ) -> None:
        self.token_factory = token_factory
        self.retry_limit = max(1, int(retry_limit))
        self.delays_ms = tuple(delays_ms) or (0,)
        self.start_inflight = False
        self.request(teardown=teardown)


class _ReusableDetachedBroker(QObject):
    """Affinity-gated registry and factory for detached cleanup tokens."""

    ensure_requested = pyqtSignal(object)

    def __init__(self, app, capacity=128):
        super().__init__(app)
        self._lock = RLock()
        self._capacity = max(1, int(capacity))
        self._registry = {}
        self._target_keys = {}
        self._last_diagnostic = None
        self._affinity_thread = self.thread()
        try:
            app.aboutToQuit.connect(self._on_app_teardown)
            self.ensure_requested.connect(
                self._on_ensure_requested, Qt.QueuedConnection
            )
        except (RuntimeError, TypeError):
            self._last_diagnostic = "broker-signal-connect-failed"

    @property
    def capacity(self):
        with self._lock:
            return self._capacity

    @capacity.setter
    def capacity(self, value):
        with self._lock:
            self._capacity = max(1, int(value))

    @property
    def pending_count(self):
        with self._lock:
            return len(self._registry)

    def reserve(
        self,
        key,
        owner,
        resource,
        target,
        *,
        token_factory,
        retry_limit,
        delays_ms,
    ) -> bool:
        with self._lock:
            existing = self._registry.get(key)
            if existing is not None:
                return existing.belongs_to(owner, resource, target)
            identity = id(target)
            if self._target_keys.get(identity) is not None:
                return False
            if len(self._registry) >= self._capacity:
                return False
            self._registry[key] = _ReusableDetachedProvisional(
                owner,
                resource,
                target,
                token_factory,
                retry_limit,
                delays_ms,
            )
            self._target_keys[identity] = key
            return True

    def _release_target_key_locked(self, key, target) -> None:
        identity = id(target)
        if self._target_keys.get(identity) == key:
            self._target_keys.pop(identity, None)

    def release_provisional(self, key, target) -> bool:
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            if entry.target is not target or entry.state != entry.PROVISIONAL:
                return False
            self._registry.pop(key, None)
            self._release_target_key_locked(key, target)
            return True

    def complete(self, key, target, token) -> bool:
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            if entry.state != entry.TOKEN or entry.token is not token:
                return False
            if entry.target is not target:
                return False
            self._registry.pop(key, None)
            self._release_target_key_locked(key, target)
            return True

    def prepare_cleanup(
        self,
        key,
        target,
        token_factory,
        retry_limit,
        delays_ms,
    ) -> bool:
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            if entry.target is not target:
                return False
            entry.prepare_cleanup(
                token_factory, retry_limit, delays_ms
            )
        return self._dispatch_ensure(key)

    def retry_pending(self) -> int:
        dispatch_keys = []
        with self._lock:
            entries = tuple(self._registry.items())
            for key, entry in entries:
                if self._accept_entry_demand_locked(
                    entry, teardown=False
                ):
                    dispatch_keys.append(key)
        for key in dispatch_keys:
            self._dispatch_ensure(key)
        return len(entries)

    def request_key(self, key, reason, *, teardown=False) -> bool:
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            dispatch = self._accept_entry_demand_locked(
                entry, teardown=teardown
            )
        if not dispatch:
            return True
        return self._dispatch_ensure(key)

    def _accept_entry_demand_locked(self, entry, *, teardown) -> bool:
        """Freeze active-burst membership at the broker acceptance point."""
        acceptance = None
        if entry.state == entry.TOKEN and entry.token is not None:
            # A scheduled teardown on the affinity thread must still perform
            # the aboutToQuit synchronous attempt. Other active requests get
            # a receipt for the token's current burst and are consumed after
            # affinity delivery without target.stop() on the caller thread.
            allow_scheduled = not (
                teardown and self._is_affinity_thread()
            )
            try:
                acceptance = entry.token.capture_demand_acceptance(
                    allow_scheduled=allow_scheduled,
                )
            except BaseException as error:
                self._last_diagnostic = (
                    "token-demand-acceptance-" + type(error).__name__
                )
        entry.request(teardown=teardown, acceptance=acceptance)
        return True

    def _is_affinity_thread(self) -> bool:
        try:
            return QThread.currentThread() == self._affinity_thread
        except BaseException:
            return False

    def _dispatch_ensure(self, key) -> bool:
        if self._is_affinity_thread():
            return self._ensure_entry(key)
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            if entry.ensure_queued:
                return True
            entry.ensure_queued = True
        try:
            self.ensure_requested.emit(key)
        except BaseException:
            with self._lock:
                entry = self._registry.get(key)
                if isinstance(entry, _ReusableDetachedProvisional):
                    entry.ensure_queued = False
                self._last_diagnostic = "broker-ensure-emit-failed"
            return False
        return True

    def _on_ensure_requested(self, key) -> None:
        with self._lock:
            entry = self._registry.get(key)
            if isinstance(entry, _ReusableDetachedProvisional):
                entry.ensure_queued = False
        self._ensure_entry(key)

    def _owner_lock(self, entry):
        owner = entry.owner_ref()
        if owner is None:
            return None, None
        try:
            lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        except BaseException:
            self._last_diagnostic = "owner-lock-inspection-failed"
            return owner, None
        return owner, lock

    def _clear_owner_token(self, entry, token) -> None:
        owner, lock = self._owner_lock(entry)
        exact_key = (entry.resource, id(entry.target))
        if lock is not None:
            try:
                with lock:
                    tokens = getattr(
                        owner, "_reusable_detached_cleanup_tokens", {}
                    )
                    if tokens.get(exact_key) is token:
                        tokens.pop(exact_key, None)
            except BaseException:
                self._last_diagnostic = "owner-token-clear-failed"
        try:
            token.release_target()
            token._safe_delete_later()
        except BaseException:
            self._last_diagnostic = "dead-token-dispose-failed"

    def _publish_owner_token(self, entry, token) -> None:
        owner, lock = self._owner_lock(entry)
        if lock is None:
            return
        exact_key = (entry.resource, id(entry.target))
        try:
            with lock:
                pending = getattr(
                    owner, "_reusable_detached_pending_stops", {}
                )
                if pending.get(exact_key) is entry.target:
                    owner._reusable_detached_cleanup_tokens[
                        exact_key
                    ] = token
        except BaseException:
            self._last_diagnostic = "owner-token-publish-failed"

    def _restore_delivery(
        self, key, entry, *, teardown, acceptance=None
    ) -> None:
        with self._lock:
            if self._registry.get(key) is not entry:
                return
            if entry.state != entry.TOKEN:
                return
            if not entry.demand_pending:
                entry.demand_acceptance = acceptance
            elif entry.demand_acceptance is None or acceptance is None:
                entry.demand_acceptance = None
            entry.demand_pending = True
            entry.teardown_pending = bool(
                entry.teardown_pending or teardown
            )

    def _deliver_entry(self, key, entry, token) -> bool:
        with self._lock:
            if self._registry.get(key) is not entry:
                return False
            if entry.state != entry.TOKEN or entry.token is not token:
                return False
            teardown = entry.teardown_pending
            demand = entry.demand_pending
            acceptance = entry.demand_acceptance
            if not teardown and not demand:
                return True
            entry.demand_acceptance = None
            entry.demand_pending = False
            entry.teardown_pending = False
        try:
            if acceptance is not None:
                delivered = bool(
                    token.deliver_accepted_demand(
                        acceptance, teardown=teardown
                    )
                )
            elif teardown:
                delivered = bool(
                    token.request_teardown_now("broker-teardown")
                )
            else:
                delivered = bool(token.request_round("broker-demand"))
        except BaseException:
            delivered = False
        if not delivered:
            self._last_diagnostic = "broker-token-delivery-pending"
            self._restore_delivery(
                key,
                entry,
                teardown=teardown,
                acceptance=acceptance,
            )
        return delivered

    def _promote_entry(self, key, entry) -> bool:
        owner = entry.owner_ref()
        token = None
        try:
            token = entry.token_factory(
                owner,
                self,
                key,
                entry.resource,
                entry.target,
                entry.retry_limit,
                entry.delays_ms,
            )
            token.activate(self._affinity_thread)
            if not token.activated:
                raise RuntimeError("detached token activation failed")
            token.attach_to_target()
        except BaseException as error:
            if token is not None:
                try:
                    token.dispose_unpublished()
                except BaseException:
                    self._last_diagnostic = "token-dispose-failed"
            with self._lock:
                if (
                    self._registry.get(key) is entry
                    and entry.state == entry.PROMOTING
                ):
                    entry.state = entry.PROVISIONAL
                    entry.token = None
                self._last_diagnostic = (
                    "token-promotion-" + type(error).__name__
                )
            return False

        self._publish_owner_token(entry, token)
        with self._lock:
            try:
                token_target = token.target
            except BaseException:
                token_target = None
            if (
                self._registry.get(key) is not entry
                or entry.state != entry.PROMOTING
                or entry.target is not token_target
            ):
                promoted = False
            else:
                entry.token = token
                entry.state = entry.TOKEN
                promoted = True
        if not promoted:
            self._clear_owner_token(entry, token)
            return False
        return self._deliver_entry(key, entry, token)

    def _ensure_entry(self, key) -> bool:
        if not self._is_affinity_thread():
            return self._dispatch_ensure(key)
        dead_token = None
        with self._lock:
            entry = self._registry.get(key)
            if not isinstance(entry, _ReusableDetachedProvisional):
                return False
            if entry.state == entry.PROMOTING:
                return True
            if entry.state == entry.TOKEN:
                token = entry.token
                try:
                    valid = bool(token is not None and token.activated)
                except BaseException:
                    valid = False
                if valid:
                    deliver_token = token
                else:
                    dead_token = token
                    deliver_token = None
                    entry.token = None
                    entry.state = entry.PROVISIONAL
                    entry.start_inflight = False
                    entry.demand_pending = True
                    entry.demand_acceptance = None
            else:
                deliver_token = None
            if deliver_token is None:
                if entry.start_inflight or not (
                    entry.demand_pending or entry.teardown_pending
                ):
                    promote = False
                else:
                    entry.state = entry.PROMOTING
                    promote = True
            else:
                promote = False
        if dead_token is not None:
            self._clear_owner_token(entry, dead_token)
        if deliver_token is not None:
            return self._deliver_entry(key, entry, deliver_token)
        if promote:
            return self._promote_entry(key, entry)
        return True

    def _on_app_teardown(self) -> None:
        dispatch_keys = []
        with self._lock:
            entries = tuple(self._registry.items())
            for key, entry in entries:
                if self._accept_entry_demand_locked(
                    entry, teardown=True
                ):
                    dispatch_keys.append(key)
        for key in dispatch_keys:
            self._dispatch_ensure(key)


class _DetachedReusableCleanupToken(QObject):
    """Bounded exact-stop retry that never strongly owns its dead window."""

    requested = pyqtSignal(int)
    IDLE = "IDLE"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"

    def __init__(
        self,
        owner,
        broker,
        key,
        resource,
        target,
        retry_limit,
        delays_ms,
    ):
        super().__init__()
        self._lock = RLock()
        try:
            self._owner_ref = ref(owner)
        except TypeError:
            self._owner_ref = lambda: None
        self._broker_ref = ref(broker) if broker is not None else lambda: None
        self._key = key
        self._resource = resource
        self._target = target
        self._retry_limit = max(1, int(retry_limit))
        self._delays_ms = tuple(delays_ms) or (0,)
        self._activated = False
        self._affinity_thread = None
        self._state = self.IDLE
        self._request_sequence = 0
        self._coalesced_through_sequence = 0
        self._burst_generation = 0
        self._burst_attempt_count = 0
        self._burst_budget_remaining = 0
        self._teardown_pending = False
        self._teardown_diagnostic = None
        self._last_diagnostic = None

    def activate(self, target_thread):
        if target_thread is not None and self.thread() != target_thread:
            self.moveToThread(target_thread)
        self.requested.connect(self._attempt, Qt.QueuedConnection)
        with self._lock:
            self._affinity_thread = self.thread()
            self._activated = True

    @property
    def activated(self) -> bool:
        with self._lock:
            if not self._activated or self._target is None:
                return False
        try:
            return not sip.isdeleted(self)
        except BaseException:
            return False

    def attach_to_target(self) -> bool:
        """Publish durable ownership without racing a concurrent release."""
        with self._lock:
            target = self._target
            if target is None:
                return False
            setattr(target, "_sequence_detached_cleanup_token", self)
            return True

    @property
    def target(self):
        with self._lock:
            return self._target

    @property
    def key(self):
        return self._key

    def belongs_to(self, owner, resource, target) -> bool:
        with self._lock:
            return bool(
                self._owner_ref() is owner
                and self._resource == resource
                and self._target is target
            )

    def request_round(self, _reason) -> bool:
        with self._lock:
            if self._target is None or not self._activated:
                return False
            self._request_sequence += 1
            self._coalesced_through_sequence = self._request_sequence
            if self._state != self.IDLE:
                return True
            self._burst_generation += 1
            generation = self._burst_generation
            self._burst_attempt_count = 0
            self._burst_budget_remaining = self._retry_limit
            self._state = self.SCHEDULED
        return self._emit_requested(generation)

    def capture_demand_acceptance(self, *, allow_scheduled=True):
        """Return a receipt tying this request to the active burst."""
        with self._lock:
            if self._target is None or not self._activated:
                return None
            try:
                if sip.isdeleted(self):
                    return None
            except BaseException as error:
                self._last_diagnostic = (
                    "demand-native-inspection-" + type(error).__name__
                )
                return None
            active = self._state == self.RUNNING or (
                allow_scheduled and self._state == self.SCHEDULED
            )
            if not active:
                return None
            self._request_sequence += 1
            return (self._burst_generation, self._request_sequence)

    def deliver_accepted_demand(self, acceptance, *, teardown=False) -> bool:
        """Consume an acceptance receipt without allocating another burst."""
        accepted_generation, accepted_sequence = acceptance
        with self._lock:
            if self._target is None or not self._activated:
                return False
            if (
                accepted_generation > self._burst_generation
                or accepted_sequence > self._request_sequence
            ):
                self._last_diagnostic = "demand-acceptance-invalid"
                return False
            self._coalesced_through_sequence = max(
                self._coalesced_through_sequence, accepted_sequence
            )
            if teardown:
                self._teardown_pending = True
                self._teardown_diagnostic = (
                    "inflight"
                    if self._state == self.RUNNING
                    else "accepted-burst"
                )
            return True

    def _emit_requested(self, generation) -> bool:
        try:
            self.requested.emit(generation)
        except BaseException:
            with self._lock:
                if (
                    generation == self._burst_generation
                    and self._state == self.SCHEDULED
                ):
                    self._state = self.IDLE
                    self._burst_budget_remaining = 0
                self._last_diagnostic = "request-emit-failed"
            return False
        return True

    def release_target(self):
        with self._lock:
            target = self._target
            self._target = None
            self._activated = False
            self._burst_generation += 1
            self._state = self.IDLE
            self._burst_budget_remaining = 0
            self._teardown_pending = False
            self._teardown_diagnostic = None
        if target is not None:
            try:
                if getattr(
                    target, "_sequence_detached_cleanup_token", None
                ) is self:
                    delattr(target, "_sequence_detached_cleanup_token")
            except BaseException:
                with self._lock:
                    self._last_diagnostic = "target-token-detach-failed"
                return

    def _safe_delete_later(self) -> None:
        try:
            if not sip.isdeleted(self):
                self.deleteLater()
        except BaseException:
            with self._lock:
                self._last_diagnostic = "delete-later-failed"
            return

    def dispose_unpublished(self) -> None:
        self.release_target()
        self._safe_delete_later()

    def _release_stopped_target(self, target) -> bool:
        broker = self._broker_ref()
        owner = self._owner_ref()
        released = False
        if broker is not None:
            try:
                released = broker.complete(self._key, target, self)
            except BaseException:
                released = False
                with self._lock:
                    self._last_diagnostic = "broker-release-failed"
            if released and owner is not None:
                SequenceResourceLifecycleController._complete_detached_owner_release(
                    owner, self._resource, target, self
                )
        elif owner is not None:
            try:
                released = SequenceResourceLifecycleController._release_detached_reusable_target(
                    owner, self._resource, target
                )
            except BaseException:
                released = False
                with self._lock:
                    self._last_diagnostic = "owner-release-failed"
        if released:
            self.release_target()
            self._safe_delete_later()
        return released

    def _stop_and_release(self, target) -> bool:
        owner = self._owner_ref()
        try:
            stopped = SequenceResourceLifecycleController._stop_detached_reusable_target(
                owner, self._resource, target
            )
        except BaseException:
            stopped = False
            with self._lock:
                self._last_diagnostic = "stop-boundary-failed"
        return bool(stopped and self._release_stopped_target(target))

    def _schedule_generation(self, generation, delay_ms) -> bool:
        try:
            QTimer.singleShot(
                max(0, int(delay_ms)),
                lambda generation=generation: self._emit_requested(
                    generation
                ),
            )
            return True
        except BaseException:
            with self._lock:
                if (
                    generation == self._burst_generation
                    and self._state == self.SCHEDULED
                ):
                    self._state = self.IDLE
                    self._burst_budget_remaining = 0
                self._last_diagnostic = "retry-timer-schedule-failed"
            return False

    def _attempt(self, generation):
        with self._lock:
            target = self._target
            if target is None:
                self._safe_delete_later()
                return
            if (
                not self._activated
                or generation != self._burst_generation
                or self._state != self.SCHEDULED
                or self._burst_budget_remaining <= 0
            ):
                return
            self._state = self.RUNNING
            self._burst_budget_remaining -= 1
            self._burst_attempt_count += 1
            attempt = self._burst_attempt_count
        released = self._stop_and_release(target)

        retry = False
        delay_ms = 0
        with self._lock:
            if released or self._target is None:
                return
            if (
                generation != self._burst_generation
                or self._state != self.RUNNING
            ):
                return
            if self._burst_budget_remaining > 0:
                retry = True
                self._state = self.SCHEDULED
                delay_index = min(attempt, len(self._delays_ms) - 1)
                delay_ms = self._delays_ms[delay_index]
            else:
                # Every request observed before this locked transition belongs
                # to the exhausted burst. Consume it and become stably idle;
                # only a later request may allocate a fresh budget.
                self._state = self.IDLE
                self._coalesced_through_sequence = self._request_sequence
        if retry:
            self._schedule_generation(generation, delay_ms)

    def request_teardown_now(self, _reason) -> bool:
        """Run one stop attempt in the aboutToQuit stack without concurrency."""
        with self._lock:
            affinity_thread = self._affinity_thread
        try:
            on_affinity = bool(
                affinity_thread is not None
                and QThread.currentThread() == affinity_thread
            )
        except BaseException:
            on_affinity = False
        if not on_affinity:
            broker = self._broker_ref()
            if broker is not None:
                try:
                    broker.request_key(
                        self._key,
                        "token-teardown-affinity-forward",
                        teardown=True,
                    )
                except BaseException:
                    with self._lock:
                        self._last_diagnostic = (
                            "teardown-affinity-forward-failed"
                        )
            with self._lock:
                self._teardown_pending = True
                self._teardown_diagnostic = "queued-for-affinity"
            return False
        with self._lock:
            target = self._target
            if target is None or not self._activated:
                return False
            self._request_sequence += 1
            self._coalesced_through_sequence = self._request_sequence
            self._teardown_pending = True
            self._teardown_diagnostic = "pending"
            if self._state == self.RUNNING:
                self._teardown_diagnostic = "inflight"
                return False
            previous_state = self._state
            if self._state == self.IDLE:
                self._burst_generation += 1
                self._burst_attempt_count = 0
                self._burst_budget_remaining = self._retry_limit
            generation = self._burst_generation
            self._state = self.RUNNING
            if self._burst_budget_remaining > 0:
                self._burst_budget_remaining -= 1
            self._burst_attempt_count += 1

        released = self._stop_and_release(target)
        with self._lock:
            if released or self._target is None:
                return True
            if generation == self._burst_generation:
                # aboutToQuit must not depend on one more queued event. Keep
                # ownership and an explicit diagnostic. If teardown claimed an
                # already-scheduled burst, its existing timer still owns the
                # remaining budget and must be allowed to settle that burst.
                if (
                    previous_state == self.SCHEDULED
                    and self._burst_budget_remaining > 0
                ):
                    self._state = self.SCHEDULED
                else:
                    self._burst_generation += 1
                    self._state = self.IDLE
                    self._burst_budget_remaining = 0
                self._teardown_pending = True
                self._teardown_diagnostic = "stop-failed"
        return False




class SequenceResourceLifecycleController(QObject):
    """Sole owner of reusable trigger resources and shutdown cleanup."""

    _reusable_resource_bootstrap_lock = RLock()
    _MODEL_PREFIXES = ("_reusable_", "_shutdown_")
    _MODEL_NAMES = {
        "_shortcut_mgr",
        "_hw_manager",
        "_tcp_resource_port",
        "_tcp_mirror_owner_token",
        "_tcp_mirror_registration_retired",
        "_lightweight_cleanup_done",
    }

    def __init__(
        self,
        view: SequenceResourceLifecycleView,
        model: SequenceResourceLifecycleModel | None = None,
        *,
        lifecycle_bus=None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_lifecycle_bus", lifecycle_bus)
        object.__setattr__(
            self, "_model", model or SequenceResourceLifecycleModel()
        )
        token = _CANONICAL_TCP_MIRROR_STATE.register_owner(self)
        self._tcp_mirror_owner_token = token
        object.__setattr__(
            self,
            "_tcp_mirror_port",
            SequenceResourceLifecycleTcpMirrorPort(self, token),
        )
        if token is None:
            self._tcp_mirror_registration_retired = True

    def __getattribute__(self, name):
        if name not in {"_model", "_MODEL_NAMES", "_MODEL_PREFIXES"}:
            model_names = object.__getattribute__(self, "_MODEL_NAMES")
            model_prefixes = object.__getattribute__(self, "_MODEL_PREFIXES")
            if name in model_names or name.startswith(model_prefixes):
                model = object.__getattribute__(self, "_model")
                return getattr(model, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            model = object.__getattribute__(self, "_model")
        except AttributeError:
            model = None
        if model is not None and (
            name in self._MODEL_NAMES or name.startswith(self._MODEL_PREFIXES)
        ):
            setattr(model, name, value)
            return
        super().__setattr__(name, value)

    @property
    def shortcut_mgr(self):
        return self._shortcut_mgr

    @shortcut_mgr.setter
    def shortcut_mgr(self, manager) -> None:
        self._set_reusable_resource_identity(self, "shortcut", manager)

    @property
    def hw_manager(self):
        return self._hw_manager

    @hw_manager.setter
    def hw_manager(self, manager) -> None:
        self._set_reusable_resource_identity(self, "hardware", manager)

    @property
    def tcp_resource_port(self):
        return self._tcp_resource_port

    @tcp_resource_port.setter
    def tcp_resource_port(self, port) -> None:
        self._set_reusable_resource_identity(self, "tcp", port)

    def bind_lifecycle_bus(self, lifecycle_bus) -> None:
        object.__setattr__(self, "_lifecycle_bus", lifecycle_bus)

    def _publish_lifecycle_request(
        self, shutdown_generation: int, operation: str
    ) -> bool:
        bus = object.__getattribute__(self, "_lifecycle_bus")
        model = object.__getattribute__(self, "_model")
        publish = getattr(bus, "publish_resource_lifecycle", None)
        if not callable(publish):
            return False
        request = model.resource_lifecycle_request(
            shutdown_generation, operation
        )
        if request is None:
            return False
        try:
            if publish(request) is True:
                return True
            model.resolve_retired_resource_lifecycle_registrations(
                bus, request
            )
            completed = getattr(
                bus, "is_resource_lifecycle_request_completed", None
            )
            return callable(completed) and completed(request) is True
        except BaseException:
            return False

    @property
    def tcp_mirror_port(self) -> SequenceResourceLifecycleTcpMirrorPort:
        return object.__getattribute__(self, "_tcp_mirror_port")

    def read_tcp_mirror_identity(self):
        return self.tcp_mirror_port.read()

    def write_tcp_mirror_identity(self, server) -> bool:
        return self.tcp_mirror_port.write(server)

    def suspend_child_resources(self) -> bool:
        return self._suspend_reusable_child_resources()

    def resume_child_resources(self) -> bool:
        resumed = self._resume_reusable_child_resources()
        self._reusable_resume_pending = resumed is not True
        return resumed

    def lightweight_child_cleanup(self) -> bool:
        return self._suspend_reusable_child_resources()

    def prepare_application_shutdown(self, shutdown_generation: int) -> bool:
        return self._prepare_application_shutdown_resources(
            shutdown_generation
        )

    def complete_application_shutdown_delivery(
        self, shutdown_generation: int
    ) -> bool:
        return self._complete_application_shutdown_delivery(
            shutdown_generation
        )

    def complete_application_shutdown_before_ready(
        self, shutdown_generation: int
    ) -> bool:
        """Finish every resource/domain step required before ``ShutdownReady``."""
        if not self.prepare_application_shutdown(shutdown_generation):
            return False
        if not self.finalize_application_shutdown(shutdown_generation):
            return False
        return self.complete_application_shutdown_delivery(shutdown_generation)

    @staticmethod
    def _set_reusable_resource_identity(owner, resource: str, target) -> None:
        backing_attribute = {
            "shortcut": "_shortcut_mgr",
            "hardware": "_hw_manager",
            "tcp": "_tcp_resource_port",
        }[resource]
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        queue_cleanup_generation = None
        with lock:
            previous = getattr(owner, backing_attribute, None)
            if previous is target:
                return
            state = getattr(owner, "_reusable_resource_state", "ACTIVE")
            if state in {"SUSPENDING", "SUSPENDED", "RESUMING"}:
                journal = getattr(owner, "_reusable_resource_journal", None)
                if not isinstance(journal, dict):
                    journal = {}
                    owner._reusable_resource_journal = journal
                entry = journal.setdefault(
                    resource,
                    {
                        "desired": False,
                        "status": "STOPPED",
                        "pending_stops": {},
                    },
                )
                candidates = [target]
                if previous is not None:
                    if resource == "tcp":
                        previous_requires_stop = bool(
                            SequenceResourceLifecycleController._reusable_tcp_target_is_active(
                                owner, previous
                            )
                            or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                                owner, previous
                            )
                        )
                    else:
                        previous_state = (
                            SequenceResourceLifecycleController._reusable_manager_active_state(
                                owner, previous, resource
                            )
                        )
                        previous_requires_stop = bool(
                            previous_state is True
                            or getattr(
                                owner, "_reusable_trusted_running_ids", {}
                            ).get(resource) == id(previous)
                        )
                    if previous_requires_stop:
                        candidates.insert(0, previous)
                if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                    owner, entry, *candidates
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "admission-capacity", resource, None
                    )
                    return
            setattr(owner, backing_attribute, target)
            owner._reusable_resource_epoch = (
                getattr(owner, "_reusable_resource_epoch", 0) + 1
            )
            if state not in {"SUSPENDING", "SUSPENDED", "RESUMING"}:
                return
            if target is not None:
                entry["status"] = "RUNNING"
            if state in {"SUSPENDED", "SUSPENDING"}:
                owner._reusable_resource_state = "SUSPENDING"
                if (
                    not getattr(owner, "_reusable_cleanup_event_pending", False)
                    and not getattr(
                        owner, "_reusable_cleanup_dispatch_active", False
                    )
                ):
                    queue_cleanup_generation = (
                        SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(
                            owner
                        )
                    )
        if queue_cleanup_generation is not None:
            SequenceResourceLifecycleController._queue_reusable_cleanup_event(
                owner, queue_cleanup_generation, delay_ms=0
            )

    @staticmethod
    def _reusable_resource_identity_lock(owner):
        lock = getattr(owner, "_reusable_resource_lock", None)
        if lock is None:
            with SequenceResourceLifecycleController._reusable_resource_bootstrap_lock:
                lock = getattr(owner, "_reusable_resource_lock", None)
                if lock is None:
                    lock = RLock()
                    owner._reusable_resource_lock = lock
        with lock:
            if not hasattr(owner, "_reusable_resource_epoch"):
                owner._reusable_resource_epoch = 0
            if not hasattr(owner, "_reusable_cleanup_event_pending"):
                owner._reusable_cleanup_event_pending = False
            if not hasattr(owner, "_reusable_cleanup_generation"):
                owner._reusable_cleanup_generation = 0
            if not hasattr(owner, "_reusable_cleanup_attempt"):
                owner._reusable_cleanup_attempt = 0
            if not hasattr(owner, "_reusable_cleanup_retry_limit"):
                owner._reusable_cleanup_retry_limit = 3
            if not hasattr(owner, "_reusable_cleanup_retry_delays_ms"):
                owner._reusable_cleanup_retry_delays_ms = (0, 25, 100)
            if not hasattr(owner, "_reusable_cleanup_dispatch_active"):
                owner._reusable_cleanup_dispatch_active = False
            if not hasattr(owner, "_reusable_pending_identity_limit"):
                owner._reusable_pending_identity_limit = 64
            if not hasattr(owner, "_reusable_operation_generation"):
                owner._reusable_operation_generation = 0
            if not hasattr(owner, "_reusable_operation_in_progress"):
                owner._reusable_operation_in_progress = False
            if not hasattr(owner, "_reusable_operation_kind"):
                owner._reusable_operation_kind = None
            if not hasattr(owner, "_reusable_operation_desired"):
                state = getattr(owner, "_reusable_resource_state", "ACTIVE")
                owner._reusable_operation_desired = (
                    "SUSPENDED" if state in {"SUSPENDING", "SUSPENDED"}
                    else "ACTIVE"
                )
            if not hasattr(owner, "_reusable_operation_continuation_pending"):
                owner._reusable_operation_continuation_pending = False
            if not hasattr(owner, "_reusable_detached_pending_stops"):
                owner._reusable_detached_pending_stops = {}
            if not hasattr(owner, "_reusable_detached_pending_resources"):
                owner._reusable_detached_pending_resources = {}
            if not hasattr(owner, "_reusable_detached_cleanup_tokens"):
                owner._reusable_detached_cleanup_tokens = {}
            if not hasattr(owner, "_reusable_cleanup_native_guard"):
                guard = _ReusableCleanupNativeGuard()
                owner._reusable_cleanup_native_guard = guard
                if isinstance(owner, QObject):
                    try:
                        if sip.isdeleted(owner):
                            guard.invalidate()
                        else:
                            owner.destroyed.connect(guard.invalidate)
                    except (RuntimeError, TypeError):
                        guard.invalidate()
            if (
                isinstance(owner, QObject)
                and not hasattr(owner, "_reusable_cleanup_dispatcher")
            ):
                guard = owner._reusable_cleanup_native_guard
                bridge = _ReusableCleanupDispatcher(owner, guard)
                try:
                    owner_thread = owner.thread()
                    if bridge.thread() is not owner_thread:
                        bridge.moveToThread(owner_thread)
                    bridge.activate()
                    owner.destroyed.connect(bridge.deleteLater)
                except (RuntimeError, TypeError):
                    guard.invalidate()
                owner._reusable_cleanup_dispatcher = bridge
        SequenceResourceLifecycleController._reusable_detached_broker(owner)
        return lock

    @staticmethod
    def _reusable_detached_broker(_owner=None):
        app = QCoreApplication.instance()
        if app is None:
            return None
        try:
            broker = getattr(app, "_sequence_reusable_detached_broker", None)
            if broker is not None and sip.isdeleted(broker):
                broker = None
        except BaseException:
            broker = None
        if broker is not None:
            return broker
        with SequenceResourceLifecycleController._reusable_resource_bootstrap_lock:
            try:
                broker = getattr(
                    app, "_sequence_reusable_detached_broker", None
                )
                if broker is not None and not sip.isdeleted(broker):
                    return broker
                if QThread.currentThread() != app.thread():
                    return None
                capacity = getattr(
                    app, "_sequence_reusable_detached_capacity", 128
                )
                broker = _ReusableDetachedBroker(app, capacity)
                setattr(app, "_sequence_reusable_detached_broker", broker)
                return broker
            except BaseException:
                return None

    @staticmethod
    def _reusable_detached_key(owner, resource: str, target):
        return (resource, id(target))

    @staticmethod
    def _admit_reusable_pending_locked(owner, entry, *targets) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            pending = entry.setdefault("pending_stops", {})
            resource = None
            journal = getattr(owner, "_reusable_resource_journal", {})
            if isinstance(journal, dict):
                for candidate_resource, candidate_entry in journal.items():
                    if candidate_entry is entry:
                        resource = candidate_resource
                        break
            detached = {
                exact_key[1]
                for exact_key, candidate_resource in getattr(
                    owner, "_reusable_detached_pending_resources", {}
                ).items()
                if resource is not None and candidate_resource == resource
            }
            candidates = []
            for target in targets:
                if target is None:
                    continue
                identity = id(target)
                if identity not in pending and identity not in detached and all(
                    id(candidate) != identity for candidate in candidates
                ):
                    candidates.append(target)
            try:
                limit = max(1, int(owner._reusable_pending_identity_limit))
            except BaseException:
                limit = 64
            occupied = set(pending).union(detached)
            if len(occupied) + len(candidates) > limit:
                return False
            for target in candidates:
                pending[id(target)] = target
            return True

    @staticmethod
    def _reusable_pending_snapshot(owner, entry):
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            pending = entry.setdefault("pending_stops", {})
            return tuple(pending.items())

    @staticmethod
    def _remove_reusable_pending_exact(owner, entry, identity, target) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            pending = entry.setdefault("pending_stops", {})
            if pending.get(identity) is not target:
                return False
            pending.pop(identity, None)
            return True

    @staticmethod
    def _reusable_pending_is_empty(owner, entry) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            return not bool(entry.setdefault("pending_stops", {}))

    @staticmethod
    def _set_reusable_entry_status(owner, entry, status: str) -> None:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            entry["status"] = status

    @staticmethod
    def _begin_reusable_cleanup_round_locked(
        owner, operation: str = "SUSPENDED"
    ) -> int:
        owner._reusable_cleanup_generation = (
            getattr(owner, "_reusable_cleanup_generation", 0) + 1
        )
        owner._reusable_cleanup_attempt = 0
        owner._reusable_cleanup_event_pending = True
        owner._reusable_cleanup_operation = operation
        return owner._reusable_cleanup_generation

    @staticmethod
    def _queue_reusable_cleanup_event(
        owner, generation: int, *, delay_ms: int
    ) -> None:
        if not isinstance(owner, QObject):
            return
        SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        guard = owner._reusable_cleanup_native_guard
        bridge = getattr(owner, "_reusable_cleanup_dispatcher", None)
        if bridge is None or not guard.alive:
            return
        try:
            bridge.requested.emit(generation, max(0, int(delay_ms)))
        except (RuntimeError, TypeError):
            guard.invalidate()

    @staticmethod
    def _run_queued_reusable_cleanup(owner, *, generation=None) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            current_generation = owner._reusable_cleanup_generation
            if generation is not None and generation != current_generation:
                return False
            if not getattr(owner, "_reusable_cleanup_event_pending", False):
                return getattr(owner, "_reusable_resource_state", None) in {
                    "SUSPENDED",
                    "ACTIVE",
                }
            owner._reusable_cleanup_event_pending = False
            owner._reusable_cleanup_attempt += 1
            attempt = owner._reusable_cleanup_attempt
            owner._reusable_cleanup_dispatch_active = True
            operation = getattr(
                owner, "_reusable_cleanup_operation", "SUSPENDED"
            )
        try:
            detached_completed = (
                SequenceResourceLifecycleController._retry_detached_reusable_stops(owner)
            )
            if operation == "ACTIVE":
                completed = SequenceResourceLifecycleController._resume_reusable_child_resources(owner)
            else:
                completed = SequenceResourceLifecycleController._suspend_reusable_child_resources(owner)
            completed = bool(completed and detached_completed)
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "queued-cleanup", "lifecycle", error
            )
            completed = False
        finally:
            with lock:
                owner._reusable_cleanup_dispatch_active = False
        if completed:
            with lock:
                if owner._reusable_cleanup_generation == current_generation:
                    owner._reusable_cleanup_attempt = 0
            return True

        next_delay = None
        with lock:
            if owner._reusable_cleanup_generation != current_generation:
                return False
            try:
                retry_limit = max(1, int(owner._reusable_cleanup_retry_limit))
            except BaseException:
                retry_limit = 3
            if attempt < retry_limit:
                delays = owner._reusable_cleanup_retry_delays_ms
                try:
                    delay_index = min(attempt, len(delays) - 1)
                    next_delay = max(0, int(delays[delay_index]))
                except BaseException:
                    next_delay = 25
                owner._reusable_cleanup_event_pending = True
        if next_delay is None:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "queued-cleanup-bound", "lifecycle", None
            )
            return False
        SequenceResourceLifecycleController._queue_reusable_cleanup_event(
            owner, current_generation, delay_ms=next_delay
        )
        return False

    @staticmethod
    def _synchronize_reusable_cleanup(owner) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            state = getattr(owner, "_reusable_resource_state", "ACTIVE")
            if state == "ACTIVE" and not getattr(
                owner, "_reusable_resource_snapshot", None
            ):
                return True
            owner._reusable_resource_state = "SUSPENDING"
            generation = SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(
                owner
            )
        SequenceResourceLifecycleController._queue_reusable_cleanup_event(
            owner, generation, delay_ms=0
        )
        return True

    @staticmethod
    def _cancel_reusable_cleanup_events(
        owner, *, release_pending: bool
    ) -> None:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            owner._reusable_cleanup_generation += 1
            owner._reusable_cleanup_event_pending = False
            owner._reusable_cleanup_attempt = 0
            owner._reusable_operation_generation += 1
            owner._reusable_operation_desired = "SUSPENDED"
            owner._reusable_operation_continuation_pending = False
            if release_pending:
                journal = getattr(owner, "_reusable_resource_journal", None)
                if isinstance(journal, dict):
                    for entry in journal.values():
                        if isinstance(entry, dict):
                            pending = entry.get("pending_stops")
                            if isinstance(pending, dict):
                                pending.clear()

    def _disconnect_trigger_inputs(
        self,
        shutdown_generation: int,
        *,
        close_dispatcher: bool = True,
    ) -> bool:
        SequenceResourceLifecycleController._cancel_reusable_cleanup_events(
            self, release_pending=True
        )
        if not self._publish_lifecycle_request(
            shutdown_generation, "disconnect-domains"
        ):
            return False
        self._view.remove_application_event_filter()
        self._view.disconnect_barcode_inputs()
        token = getattr(self, "_tcp_mirror_owner_token", None)
        _CANONICAL_TCP_MIRROR_STATE.unregister(self, token)
        self._tcp_mirror_owner_token = None
        self._tcp_mirror_registration_retired = True
        if close_dispatcher:
            bus = object.__getattribute__(self, "_lifecycle_bus")
            close = getattr(
                bus, "close_workflow_continuation_dispatcher", None
            )
            if callable(close) and close() is False:
                return False
        return True

    @staticmethod
    def _claim_reusable_operation(owner, desired: str):
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            guard = owner._reusable_cleanup_native_guard
            if not guard.alive:
                return None
            try:
                if isinstance(owner, QObject) and sip.isdeleted(owner):
                    guard.invalidate()
                    return None
            except BaseException:
                return None
            previous_desired = owner._reusable_operation_desired
            if desired != previous_desired:
                owner._reusable_operation_desired = desired
                owner._reusable_operation_generation += 1
                owner._reusable_cleanup_generation += 1
                owner._reusable_cleanup_event_pending = False
                owner._reusable_cleanup_attempt = 0
            if owner._reusable_operation_in_progress:
                if owner._reusable_operation_kind != desired:
                    owner._reusable_operation_continuation_pending = True
                    owner._reusable_resume_pending = desired == "ACTIVE"
                return None
            owner._reusable_operation_in_progress = True
            owner._reusable_operation_kind = desired
            owner._reusable_operation_continuation_pending = False
            return owner._reusable_operation_generation

    @staticmethod
    def _reusable_operation_is_current(owner, token: int, desired: str) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            return bool(
                owner._reusable_cleanup_native_guard.alive
                and owner._reusable_operation_in_progress
                and owner._reusable_operation_kind == desired
                and owner._reusable_operation_desired == desired
                and owner._reusable_operation_generation == token
            )

    @staticmethod
    def _reusable_operation_port_allowed(owner, _desired: str) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            if not owner._reusable_cleanup_native_guard.alive:
                return False
            if not owner._reusable_operation_in_progress:
                return True
            return bool(
                owner._reusable_operation_kind
                == owner._reusable_operation_desired
            )

    @staticmethod
    def _finish_reusable_operation(
        owner, token: int, desired: str, completed: bool
    ) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        queue_generation = None
        with lock:
            current = bool(
                owner._reusable_cleanup_native_guard.alive
                and owner._reusable_operation_in_progress
                and owner._reusable_operation_kind == desired
                and owner._reusable_operation_desired == desired
                and owner._reusable_operation_generation == token
            )
            owner._reusable_operation_in_progress = False
            owner._reusable_operation_kind = None
            if (
                not current
                and owner._reusable_cleanup_native_guard.alive
                and owner._reusable_operation_continuation_pending
            ):
                next_desired = owner._reusable_operation_desired
                owner._reusable_operation_continuation_pending = False
                queue_generation = (
                    SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(
                        owner, next_desired
                    )
                )
            elif current:
                owner._reusable_operation_continuation_pending = False
        if queue_generation is not None:
            SequenceResourceLifecycleController._queue_reusable_cleanup_event(
                owner, queue_generation, delay_ms=0
            )
        return bool(current and completed)

    @staticmethod
    def _reserve_detached_reusable_target(
        owner, resource: str, target
    ) -> bool:
        if target is None:
            return True
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
        app = QCoreApplication.instance()
        if app is not None and broker is None:
            return False
        key = SequenceResourceLifecycleController._reusable_detached_key(owner, resource, target)
        with lock:
            pending = owner._reusable_detached_pending_stops
            resources = owner._reusable_detached_pending_resources
            try:
                retry_limit = max(
                    1, int(owner._reusable_cleanup_retry_limit)
                )
            except BaseException:
                retry_limit = 3
            delays = getattr(
                owner, "_reusable_cleanup_retry_delays_ms", (0, 25, 100)
            )
            identity = id(target)
            exact_key = (resource, identity)
            if pending.get(exact_key) is target:
                return bool(
                    broker is None
                    or broker.reserve(
                        key,
                        owner,
                        resource,
                        target,
                        token_factory=_DetachedReusableCleanupToken,
                        retry_limit=retry_limit,
                        delays_ms=delays,
                    )
                )
            try:
                limit = max(1, int(owner._reusable_pending_identity_limit))
            except BaseException:
                limit = 64
            journal = getattr(owner, "_reusable_resource_journal", {})
            entry = journal.get(resource, {}) if isinstance(journal, dict) else {}
            regular = entry.get("pending_stops", {}) if isinstance(entry, dict) else {}
            occupied = set(regular)
            occupied.update(
                pending_key[1]
                for pending_key, pending_resource in resources.items()
                if pending_resource == resource
            )
            if identity not in occupied and len(occupied) >= limit:
                return False
            if broker is not None and not broker.reserve(
                key,
                owner,
                resource,
                target,
                token_factory=_DetachedReusableCleanupToken,
                retry_limit=retry_limit,
                delays_ms=delays,
            ):
                return False
            pending[exact_key] = target
            resources[exact_key] = resource
            return True

    @staticmethod
    def _release_detached_reusable_target(
        owner, resource: str, target
    ) -> bool:
        if target is None:
            return True
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
        exact_key = (resource, id(target))
        token = None
        with lock:
            pending_is_exact = (
                owner._reusable_detached_pending_stops.get(exact_key) is target
            )
            token = owner._reusable_detached_cleanup_tokens.get(
                exact_key
            )
            key = exact_key
            if token is not None:
                key = token.key
            if not pending_is_exact and token is None:
                return True
            if broker is not None:
                if token is None:
                    released = broker.release_provisional(key, target)
                else:
                    released = broker.complete(key, target, token)
                if not released:
                    return False
            if owner._reusable_detached_pending_stops.get(exact_key) is target:
                owner._reusable_detached_pending_stops.pop(exact_key, None)
                owner._reusable_detached_pending_resources.pop(exact_key, None)
            if owner._reusable_detached_cleanup_tokens.get(exact_key) is token:
                owner._reusable_detached_cleanup_tokens.pop(exact_key, None)
        if token is not None:
            token.release_target()
        return True

    @staticmethod
    def _complete_detached_owner_release(
        owner, resource: str, target, token
    ) -> None:
        if owner is None or target is None:
            return
        try:
            lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        except BaseException:
            with token._lock:
                token._last_diagnostic = "owner-release-lock-failed"
            return
        exact_key = (resource, id(target))
        try:
            with lock:
                pending = getattr(
                    owner, "_reusable_detached_pending_stops", {}
                )
                resources = getattr(
                    owner, "_reusable_detached_pending_resources", {}
                )
                tokens = getattr(
                    owner, "_reusable_detached_cleanup_tokens", {}
                )
                if pending.get(exact_key) is target:
                    pending.pop(exact_key, None)
                    resources.pop(exact_key, None)
                if tokens.get(exact_key) is token:
                    tokens.pop(exact_key, None)
        except BaseException:
            with token._lock:
                token._last_diagnostic = "owner-release-map-failed"
            return

    @staticmethod
    def _detached_reusable_target_is_active(resource: str, target):
        if target is None:
            return False
        observed = []
        inspection_target = target
        if resource == "tcp":
            try:
                inspection_target = getattr(target, "model", target)
            except BaseException:
                return True
        for attribute in (
            "tcp_enabled",
            "tcp_running",
            "is_active",
            "active",
            "is_running",
            "running",
        ):
            try:
                value = getattr(inspection_target, attribute, None)
                if callable(value):
                    value = value()
            except BaseException:
                return True
            if type(value) is bool:
                observed.append(value)
        for attribute in ("handle", "_hotkey_handle", "tcp_server"):
            try:
                value = getattr(inspection_target, attribute, None)
            except BaseException:
                return True
            if value is not None:
                observed.append(True)
        return any(observed) if observed else None

    @staticmethod
    def _stop_detached_reusable_target(owner, resource: str, target) -> bool:
        method_name = "stop_tcp" if resource == "tcp" else "stop"
        try:
            stop = getattr(target, method_name, None)
        except BaseException as error:
            if owner is not None:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-stop-port", resource, error
                )
            return False
        if not callable(stop):
            if owner is not None:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-stop-port", resource, None
                )
            return False
        try:
            stopped = stop()
        except BaseException as error:
            if owner is not None:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-stop", resource, error
                )
            return False
        active = SequenceResourceLifecycleController._detached_reusable_target_is_active(
            resource, target
        )
        if stopped is False or active is True:
            if owner is not None:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-verify-stop", resource, None
                )
            return False
        return True

    @staticmethod
    def _schedule_detached_reusable_retry(
        owner, resource: str, target
    ) -> None:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
        key = SequenceResourceLifecycleController._reusable_detached_key(owner, resource, target)
        existing_token = None
        exact_key = (resource, id(target))
        with lock:
            if (
                owner._reusable_detached_pending_stops.get(exact_key)
                is not target
            ):
                return
            existing_token = owner._reusable_detached_cleanup_tokens.get(
                exact_key
            )
            try:
                retry_limit = max(1, int(owner._reusable_cleanup_retry_limit))
            except BaseException:
                retry_limit = 3
            delays = getattr(
                owner, "_reusable_cleanup_retry_delays_ms", (0, 25, 100)
            )
        if broker is not None:
            if not broker.prepare_cleanup(
                key,
                target,
                _DetachedReusableCleanupToken,
                retry_limit,
                delays,
            ):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-retry-prepare", resource, None
                )
            return
        if existing_token is not None:
            try:
                existing_token.request_round("owner-schedule")
            except BaseException as error:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "detached-retry-queue", resource, error
                )
            return
        try:
            token = _DetachedReusableCleanupToken(
                owner,
                broker,
                key,
                resource,
                target,
                retry_limit,
                delays,
            )
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "detached-retry-token", resource, error
            )
            return
        try:
            app = QCoreApplication.instance()
            target_thread = (
                broker.thread()
                if broker is not None
                else app.thread() if app is not None else None
            )
            token.activate(target_thread)
        except BaseException as error:
            token.dispose_unpublished()
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "detached-retry-activate", resource, error
            )
            return
        if not token.activated:
            token.dispose_unpublished()
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "detached-retry-activate", resource, None
            )
            return
        try:
            token.attach_to_target()
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "detached-retry-owner", resource, error
            )
        with lock:
            accepted = not (
                owner._reusable_detached_pending_stops.get(exact_key)
                is not target
                or exact_key in owner._reusable_detached_cleanup_tokens
            )
            if accepted:
                owner._reusable_detached_cleanup_tokens[exact_key] = token
        if not accepted:
            token.dispose_unpublished()
            return
        try:
            token.request_round("owner-schedule")
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "detached-retry-queue", resource, error
            )

    @staticmethod
    def _rollback_detached_reusable_start(
        owner, resource: str, target
    ) -> bool:
        stopped = SequenceResourceLifecycleController._stop_detached_reusable_target(
            owner, resource, target
        )
        if stopped:
            return SequenceResourceLifecycleController._release_detached_reusable_target(
                owner, resource, target
            )
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, resource, target
        )
        return False

    @staticmethod
    def _retry_detached_reusable_stops(owner) -> bool:
        broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            pending = tuple(owner._reusable_detached_pending_stops.items())
        for exact_key, target in pending:
            with lock:
                resource = owner._reusable_detached_pending_resources.get(
                    exact_key, exact_key[0]
                )
                token = owner._reusable_detached_cleanup_tokens.get(
                    exact_key
                )
            key = token.key if token is not None else exact_key
            if broker is not None:
                broker.request_key(key, "owner-retry")
                continue
            if token is not None:
                try:
                    token.request_round("owner-retry")
                except BaseException as error:
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "detached-owner-retry", resource, error
                    )
            else:
                SequenceResourceLifecycleController._schedule_detached_reusable_retry(
                    owner, resource, target
                )
        with lock:
            return not bool(owner._reusable_detached_pending_stops)

    def _suspend_reusable_child_resources(self) -> bool:
        token = SequenceResourceLifecycleController._claim_reusable_operation(self, "SUSPENDED")
        if token is None:
            return False
        completed = False
        try:
            completed = SequenceResourceLifecycleController._perform_suspend_reusable_child_resources(
                self, token
            )
        finally:
            completed = SequenceResourceLifecycleController._finish_reusable_operation(
                self, token, "SUSPENDED", completed
            )
        return completed

    def _perform_suspend_reusable_child_resources(
        self, operation_token: int
    ) -> bool:
        """Pause external trigger resources while retaining all MVC wiring."""
        state = SequenceResourceLifecycleController._ensure_reusable_resource_lifecycle(self)
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(self)
        if not SequenceResourceLifecycleController._reusable_operation_is_current(
            self, operation_token, "SUSPENDED"
        ):
            return False
        with lock:
            snapshot = getattr(self, "_reusable_resource_snapshot", None)
            capture_snapshot = state == "ACTIVE" or snapshot is None
            if capture_snapshot:
                journal = getattr(self, "_reusable_resource_journal", None)
                if not isinstance(journal, dict):
                    journal = {}
                    self._reusable_resource_journal = journal
                for resource in ("tcp", "shortcut", "hardware"):
                    journal.setdefault(
                        resource,
                        {
                            "desired": False,
                            "status": "STOPPED",
                            "pending_stops": {},
                        },
                    )
            self._reusable_resource_state = "SUSPENDING"
            self._reusable_resource_epoch += 1
            self._reusable_cleanup_event_pending = False
        trigger = getattr(self, "tcp_resource_port", None)
        shortcut = getattr(self, "shortcut_mgr", None)
        hardware = getattr(self, "hw_manager", None)
        trigger_model = getattr(trigger, "model", None)
        if capture_snapshot:
            tcp_enabled_ok, tcp_enabled = SequenceResourceLifecycleController._observe_reusable_attribute(
                self, trigger_model, "tcp_enabled", "tcp", "snapshot"
            )
            host_ok, tcp_host = SequenceResourceLifecycleController._observe_reusable_attribute(
                self, trigger_model, "tcp_host", "tcp", "snapshot"
            )
            port_ok, tcp_port = SequenceResourceLifecycleController._observe_reusable_attribute(
                self, trigger_model, "tcp_port", "tcp", "snapshot"
            )
            snapshot = {
                "tcp_enabled": (
                    True if not tcp_enabled_ok or type(tcp_enabled) is not bool
                    else tcp_enabled
                ),
                "tcp_host": tcp_host if host_ok else None,
                "tcp_port": tcp_port if port_ok else None,
                "shortcut": SequenceResourceLifecycleController._reusable_manager_is_active(
                    self, shortcut, "shortcut"
                ),
                "hardware": SequenceResourceLifecycleController._reusable_manager_is_active(
                    self, hardware, "hardware"
                ),
            }
            with lock:
                self._reusable_resource_snapshot = snapshot
                journal = self._reusable_resource_journal
                for resource, desired in (
                    ("tcp", snapshot["tcp_enabled"]),
                    ("shortcut", snapshot["shortcut"]),
                    ("hardware", snapshot["hardware"]),
                ):
                    entry = journal[resource]
                    entry["desired"] = bool(desired)
                    if not entry.setdefault("pending_stops", {}):
                        entry["status"] = (
                            "RUNNING" if desired else "STOPPED"
                        )
        with lock:
            self._reusable_child_suspended = False
            journal = self._reusable_resource_journal

        for resource in ("tcp", "shortcut", "hardware"):
            entry = journal.setdefault(
                resource,
                {"desired": False, "status": "STOPPED", "pending_stops": {}},
            )
            SequenceResourceLifecycleController._stop_reusable_resource(self, resource, entry)
            if not SequenceResourceLifecycleController._reusable_operation_is_current(
                self, operation_token, "SUSPENDED"
            ):
                return False
        if not SequenceResourceLifecycleController._reusable_operation_is_current(
            self, operation_token, "SUSPENDED"
        ):
            return False
        with lock:
            review_epoch = self._reusable_resource_epoch
        suspended = SequenceResourceLifecycleController._review_reusable_resources_after_event(
            self, journal, require_stopped=True
        )
        with lock:
            suspended = bool(
                suspended
                and self._reusable_resource_epoch == review_epoch
                and not any(
                    entry.setdefault("pending_stops", {})
                    for entry in journal.values()
                )
            )
            self._reusable_resource_state = (
                "SUSPENDED" if suspended else "SUSPENDING"
            )
            self._reusable_child_suspended = suspended
            self._reusable_suspend_completed = {
                resource
                for resource, entry in journal.items()
                if entry["status"] == "STOPPED"
            }
            queue_cleanup_generation = None
            if (
                not suspended
                and not getattr(
                    self, "_reusable_cleanup_dispatch_active", False
                )
                and not getattr(
                    self, "_reusable_cleanup_event_pending", False
                )
            ):
                queue_cleanup_generation = (
                    SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(self)
                )
        if queue_cleanup_generation is not None:
            SequenceResourceLifecycleController._queue_reusable_cleanup_event(
                self, queue_cleanup_generation, delay_ms=0
            )
        return suspended

    @staticmethod
    def _review_reusable_resources_after_event(
        owner, journal, *, require_stopped: bool
    ) -> bool:
        stable = True
        for resource in ("shortcut", "hardware", "tcp"):
            lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
            with lock:
                entry = journal.setdefault(
                    resource,
                    {
                        "desired": False,
                        "status": "STOPPED",
                        "pending_stops": {},
                    },
                )
                desired = bool(entry["desired"])
            should_be_stopped = require_stopped or not desired
            if resource == "tcp":
                current = SequenceResourceLifecycleController._reusable_tcp_target(owner)
                active = bool(
                    current is not None
                    and (
                        SequenceResourceLifecycleController._reusable_tcp_target_is_active(
                            owner, current
                        )
                        or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                            owner, current
                        )
                    )
                )
            else:
                current = SequenceResourceLifecycleController._reusable_resource_target(
                    owner, resource
                )
                active = (
                    SequenceResourceLifecycleController._reusable_manager_active_state(
                        owner, current, resource
                    )
                    is True
                )
            if should_be_stopped and active and current is not None:
                if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                    owner, entry, current
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "review-capacity", resource, None
                    )
                SequenceResourceLifecycleController._set_reusable_entry_status(
                    owner, entry, "RUNNING"
                )
            if not SequenceResourceLifecycleController._reusable_pending_is_empty(owner, entry):
                stable = False
            with lock:
                status = entry.get("status")
            if should_be_stopped and status != "STOPPED":
                stable = False
        return stable

    @staticmethod
    def _ensure_reusable_resource_lifecycle(owner) -> str:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            state = getattr(owner, "_reusable_resource_state", None)
            if state not in {"ACTIVE", "SUSPENDING", "SUSPENDED", "RESUMING"}:
                state = (
                    "SUSPENDED"
                    if getattr(owner, "_reusable_child_suspended", False)
                    else "ACTIVE"
                )
                owner._reusable_resource_state = state
            journal = getattr(owner, "_reusable_resource_journal", None)
            if not isinstance(journal, dict):
                owner._reusable_resource_journal = {}
            trusted = getattr(owner, "_reusable_trusted_running_ids", None)
            if not isinstance(trusted, dict):
                owner._reusable_trusted_running_ids = {}
            return state

    @staticmethod
    def _stop_reusable_resource(owner, resource: str, entry) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            entry.setdefault("pending_stops", {})
            entry_status = entry.get("status")
        if resource == "tcp":
            trigger = SequenceResourceLifecycleController._reusable_tcp_target(owner)
            if trigger is not None and (
                SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, trigger)
                or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(owner, trigger)
                or entry_status == "RUNNING"
            ):
                if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                    owner, entry, trigger
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "admission-capacity", resource, None
                    )
                    return False
            for identity, target in SequenceResourceLifecycleController._reusable_pending_snapshot(
                owner, entry
            ):
                if not (
                    SequenceResourceLifecycleController._reusable_tcp_target_is_active(
                        owner, target
                    )
                    or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                        owner, target
                    )
                ):
                    SequenceResourceLifecycleController._remove_reusable_pending_exact(
                        owner, entry, identity, target
                    )
                    continue
                stopped = SequenceResourceLifecycleController._stop_exact_reusable_target(
                    owner, resource, target
                )
                SequenceResourceLifecycleController._journal_reentrant_reusable_target(
                    owner, resource, entry, target
                )
                if not stopped:
                    continue
                SequenceResourceLifecycleController._remove_reusable_pending_exact(
                    owner, entry, identity, target
                )
            current = SequenceResourceLifecycleController._reusable_tcp_target(owner)
            current_running = bool(
                current is not None
                and (
                    SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, current)
                    or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                        owner, current
                    )
                )
            )
            stopped = bool(
                SequenceResourceLifecycleController._reusable_pending_is_empty(owner, entry)
                and not current_running
            )
            SequenceResourceLifecycleController._set_reusable_entry_status(
                owner, entry, "STOPPED" if stopped else "RUNNING"
            )
            return stopped

        manager = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        active_state = SequenceResourceLifecycleController._reusable_manager_active_state(
            owner, manager, resource
        )
        trusted_identity = owner._reusable_trusted_running_ids.get(resource)
        if manager is not None and (
            active_state is True
            or trusted_identity == id(manager)
            or entry_status == "RUNNING"
        ):
            if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                owner, entry, manager
            ):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "admission-capacity", resource, None
                )
                return False

        for identity, target in SequenceResourceLifecycleController._reusable_pending_snapshot(
            owner, entry
        ):
            target_state = SequenceResourceLifecycleController._reusable_manager_active_state(
                owner, target, resource
            )
            trusted_target = (
                owner._reusable_trusted_running_ids.get(resource) == identity
            )
            if target_state is False and not trusted_target:
                SequenceResourceLifecycleController._remove_reusable_pending_exact(
                    owner, entry, identity, target
                )
                continue
            stopped = SequenceResourceLifecycleController._stop_exact_reusable_target(
                owner, resource, target
            )
            SequenceResourceLifecycleController._journal_reentrant_reusable_target(
                owner, resource, entry, target
            )
            if not stopped:
                continue
            SequenceResourceLifecycleController._remove_reusable_pending_exact(
                owner, entry, identity, target
            )
            with lock:
                if owner._reusable_trusted_running_ids.get(resource) == identity:
                    owner._reusable_trusted_running_ids.pop(resource, None)

        manager = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        current_state = SequenceResourceLifecycleController._reusable_manager_active_state(
            owner, manager, resource
        )
        current_running = bool(
            manager is not None
            and (
                current_state is True
                or owner._reusable_trusted_running_ids.get(resource) == id(manager)
            )
        )
        stopped = bool(
            SequenceResourceLifecycleController._reusable_pending_is_empty(owner, entry)
            and not current_running
        )
        SequenceResourceLifecycleController._set_reusable_entry_status(
            owner, entry, "STOPPED" if stopped else "RUNNING"
        )
        return stopped

    @staticmethod
    def _stop_exact_reusable_target(owner, resource: str, target) -> bool:
        if not SequenceResourceLifecycleController._reusable_operation_port_allowed(
            owner, "SUSPENDED"
        ):
            return False
        method_name = "stop_tcp" if resource == "tcp" else "stop"
        ok, stop = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, target, method_name, resource, "suspension-port"
        )
        if not ok or not callable(stop):
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "suspension-port", resource, None
            )
            return False
        try:
            stopped = stop()
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "suspension", resource, error
            )
            return False
        if not SequenceResourceLifecycleController._reusable_operation_port_allowed(
            owner, "SUSPENDED"
        ):
            return False
        if stopped is False:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "suspension", resource, None
            )
            return False
        if resource == "tcp":
            still_running = (
                SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, target)
                or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(owner, target)
            )
        else:
            still_running = (
                SequenceResourceLifecycleController._reusable_manager_active_state(
                    owner, target, resource
                )
                is True
            )
        if still_running:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "verify-suspension", resource, None
            )
            return False
        return True

    @staticmethod
    def _journal_reentrant_reusable_target(
        owner, resource: str, entry, stopped_target
    ) -> None:
        if resource == "tcp":
            current = SequenceResourceLifecycleController._reusable_tcp_target(owner)
            if current is None or current is stopped_target:
                return
            if (
                SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, current)
                or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(owner, current)
            ):
                if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                    owner, entry, current
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "reentrant-capacity", resource, None
                    )
            return
        current = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        if current is None or current is stopped_target:
            return
        active_state = SequenceResourceLifecycleController._reusable_manager_active_state(
            owner, current, resource
        )
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            trusted = (
                owner._reusable_trusted_running_ids.get(resource) == id(current)
            )
        if active_state is True or active_state is None or trusted:
            if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
                owner, entry, current
            ):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "reentrant-capacity", resource, None
                )

    @staticmethod
    def _reusable_resource_method(owner, resource: str, operation: str):
        if resource == "tcp":
            trigger = SequenceResourceLifecycleController._reusable_tcp_target(owner)
            method_name = "stop_tcp" if operation == "stop" else "set_tcp_enabled"
            ok, method = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, trigger, method_name, resource, f"{operation}-port"
            )
            return method if ok else None
        manager = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        ok, method = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, manager, operation, resource, f"{operation}-port"
        )
        return method if ok else None

    @staticmethod
    def _reusable_resource_target(owner, resource: str):
        attribute = "shortcut_mgr" if resource == "shortcut" else "hw_manager"
        ok, target = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, owner, attribute, resource, "target"
        )
        return target if ok else None

    @staticmethod
    def _reusable_tcp_target(owner):
        ok, target = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, owner, "tcp_resource_port", "tcp", "target"
        )
        return target if ok else None

    @staticmethod
    def _reusable_tcp_target_is_active(owner, trigger) -> bool:
        ok, model = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, trigger, "model", "tcp", "state"
        )
        if not ok:
            return True
        observed = []
        for attribute in ("tcp_enabled", "tcp_running"):
            ok, value = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, model, attribute, "tcp", "state"
            )
            if not ok:
                return True
            if type(value) is bool:
                observed.append(value)
        ok, server = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, model, "tcp_server", "tcp", "state"
        )
        if not ok:
            return True
        if server is not None:
            observed.append(True)
        return any(observed)

    @staticmethod
    def _reusable_tcp_target_has_pending(owner, trigger) -> bool:
        ok, pending = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, trigger, "_tcp_stop_journal", "tcp", "state"
        )
        if not ok:
            return True
        if pending is None:
            return False
        try:
            return bool(pending)
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "state-_tcp_stop_journal-bool", "tcp", error
            )
            return True

    @staticmethod
    def _reusable_resource_is_active(owner, resource: str) -> bool:
        if resource == "tcp":
            trigger = SequenceResourceLifecycleController._reusable_tcp_target(owner)
            return SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, trigger)
        manager = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        return SequenceResourceLifecycleController._reusable_manager_is_active(
            owner, manager, resource
        )

    @staticmethod
    def _reusable_manager_is_active(owner, manager, resource: str) -> bool:
        state = SequenceResourceLifecycleController._reusable_manager_active_state(
            owner, manager, resource
        )
        if state is True:
            return True
        if resource != "hardware":
            return bool(state)
        ok, toolbar = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, owner, "toolsbar", resource, "snapshot"
        )
        if not ok:
            return True
        ok, checkbox = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, toolbar, "barcode_scanner_box", resource, "snapshot"
        )
        if not ok:
            return True
        ok, checked = SequenceResourceLifecycleController._observe_reusable_attribute(
            owner, checkbox, "isChecked", resource, "snapshot"
        )
        if not ok:
            return True
        if callable(checked):
            try:
                return bool(checked())
            except BaseException as error:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    owner, "snapshot", resource, error
                )
        return bool(state)

    @staticmethod
    def _reusable_manager_active_state(owner, manager, resource: str) -> bool | None:
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
        ):
            ok, value = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, manager, attribute, resource, "state"
            )
            if not ok:
                return True
            if callable(value):
                try:
                    value = value()
                except BaseException as error:
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "snapshot", resource, error
                    )
                    return True
            if type(value) is bool:
                observed.append(value)
            elif attribute == "status" and type(value) is str:
                observed.append(value.upper() not in {"STOPPED", "INACTIVE", "IDLE"})
        if resource == "shortcut":
            ok, handle = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, manager, "_hotkey_handle", resource, "state"
            )
            if not ok:
                return True
            if handle is not None:
                observed.append(True)
        if resource == "hardware":
            for attribute in ("_scanner_enabled", "hotkey_registered"):
                ok, value = SequenceResourceLifecycleController._observe_reusable_attribute(
                    owner, manager, attribute, resource, "state"
                )
                if not ok:
                    return True
                if type(value) is bool:
                    observed.append(value)
            ok, handles = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, manager, "hid_handles", resource, "state"
            )
            if not ok:
                return True
            if handles is not None:
                try:
                    observed.append(bool(handles))
                except BaseException as error:
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "state-hid_handles-bool", resource, error
                    )
                    return True
            ok, timer = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, manager, "_hid_poll_timer", resource, "state"
            )
            if not ok:
                return True
            ok, is_active = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, timer, "isActive", resource, "state"
            )
            if not ok:
                return True
            if callable(is_active):
                try:
                    observed.append(bool(is_active()))
                except BaseException as error:
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        owner, "snapshot", resource, error
                    )
                    return True
        for attribute in ("handle", "_hotkey_handle"):
            ok, handle = SequenceResourceLifecycleController._observe_reusable_attribute(
                owner, manager, attribute, resource, "state"
            )
            if not ok:
                return True
            if handle is not None:
                observed.append(True)
        if not observed:
            return None
        return any(observed)

    @staticmethod
    def _observe_reusable_attribute(
        owner, target, attribute: str, resource: str, operation: str
    ) -> tuple[bool, object]:
        if target is None:
            return True, None
        try:
            return True, getattr(target, attribute, None)
        except BaseException as error:
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, f"{operation}-{attribute}", resource, error
            )
            return False, None

    @staticmethod
    def _report_reusable_resource_failure(
        owner, operation: str, resource: str, error: BaseException | None
    ) -> None:
        detail = "returned False" if error is None else type(error).__name__
        try:
            logger = getattr(owner, "default_logger", None)
            callback = getattr(logger, "warning", None)
        except BaseException:
            return
        if not callable(callback):
            return
        try:
            callback(f"reusable resource {operation} failed: {resource}/{detail}")
        except BaseException:
            return

    def _resume_reusable_child_resources(self) -> bool:
        token = SequenceResourceLifecycleController._claim_reusable_operation(self, "ACTIVE")
        if token is None:
            return False
        completed = False
        try:
            completed = SequenceResourceLifecycleController._perform_resume_reusable_child_resources(
                self, token
            )
        finally:
            completed = SequenceResourceLifecycleController._finish_reusable_operation(
                self, token, "ACTIVE", completed
            )
        return completed

    def _perform_resume_reusable_child_resources(
        self, operation_token: int
    ) -> bool:
        state = SequenceResourceLifecycleController._ensure_reusable_resource_lifecycle(self)
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(self)
        if not SequenceResourceLifecycleController._reusable_operation_is_current(
            self, operation_token, "ACTIVE"
        ):
            return False
        snapshot = getattr(self, "_reusable_resource_snapshot", None)
        if state == "ACTIVE" and snapshot is None:
            with lock:
                self._reusable_cleanup_generation += 1
                self._reusable_cleanup_event_pending = False
                self._reusable_cleanup_attempt = 0
            return True
        if snapshot is None:
            with lock:
                self._reusable_cleanup_generation += 1
                self._reusable_cleanup_event_pending = False
                self._reusable_cleanup_attempt = 0
                self._reusable_resource_state = "ACTIVE"
                self._reusable_resource_epoch += 1
            return True
        with lock:
            # Reopening owns a new generation.  Any queued hide retry now has
            # a stale generation before a single resource is restarted.
            self._reusable_cleanup_generation += 1
            self._reusable_cleanup_event_pending = False
            self._reusable_cleanup_attempt = 0
            self._reusable_resource_state = "RESUMING"
            self._reusable_resource_epoch += 1
            self._reusable_child_suspended = False
            journal = self._reusable_resource_journal

        for resource in ("shortcut", "hardware", "tcp"):
            with lock:
                entry = journal.setdefault(
                    resource,
                    {
                        "desired": False,
                        "status": "STOPPED",
                        "pending_stops": {},
                    },
                )
                desired = bool(entry["desired"])
            if resource != "tcp":
                manager = SequenceResourceLifecycleController._reusable_resource_target(self, resource)
                active_state = SequenceResourceLifecycleController._reusable_manager_active_state(
                    self, manager, resource
                )
                for identity, target in (
                    SequenceResourceLifecycleController._reusable_pending_snapshot(self, entry)
                ):
                    target_state = SequenceResourceLifecycleController._reusable_manager_active_state(
                        self, target, resource
                    )
                    if target is manager and desired and target_state is True:
                        SequenceResourceLifecycleController._remove_reusable_pending_exact(
                            self, entry, identity, target
                        )
                if (
                    not SequenceResourceLifecycleController._reusable_pending_is_empty(self, entry)
                    and not SequenceResourceLifecycleController._stop_reusable_resource(
                        self, resource, entry
                    )
                ):
                    continue
            if not desired:
                if not SequenceResourceLifecycleController._stop_reusable_resource(
                    self, resource, entry
                ):
                    SequenceResourceLifecycleController._set_reusable_entry_status(
                        self, entry, "RUNNING"
                    )
                continue

            active = SequenceResourceLifecycleController._reusable_resource_is_active(self, resource)
            if resource != "tcp":
                manager = SequenceResourceLifecycleController._reusable_resource_target(self, resource)
                active_state = SequenceResourceLifecycleController._reusable_manager_active_state(
                    self, manager, resource
                )
                active = bool(
                    active_state is True
                    or (
                        manager is not None
                        and self._reusable_trusted_running_ids.get(resource)
                        == id(manager)
                    )
                )
            else:
                trigger = SequenceResourceLifecycleController._reusable_tcp_target(self)
                if SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                    self, trigger
                ):
                    active = False
            if active:
                SequenceResourceLifecycleController._set_reusable_entry_status(
                    self, entry, "RUNNING"
                )
                continue
            start_target = (
                SequenceResourceLifecycleController._reusable_tcp_target(self)
                if resource == "tcp"
                else SequenceResourceLifecycleController._reusable_resource_target(self, resource)
            )
            start = SequenceResourceLifecycleController._reusable_resource_method(self, resource, "start")
            if not callable(start):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "resume", resource, None
                )
                continue
            if not SequenceResourceLifecycleController._reserve_detached_reusable_target(
                self, resource, start_target
            ):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "detached-admission-capacity", resource, None
                )
                continue
            if not SequenceResourceLifecycleController._reusable_operation_is_current(
                self, operation_token, "ACTIVE"
            ):
                if not SequenceResourceLifecycleController._release_detached_reusable_target(
                    self, resource, start_target
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        self, "detached-release", resource, None
                    )
                return False
            try:
                if resource == "tcp":
                    tcp_options = {}
                    if snapshot.get("tcp_host") is not None:
                        tcp_options["host"] = snapshot["tcp_host"]
                    if snapshot.get("tcp_port") is not None:
                        tcp_options["port"] = snapshot["tcp_port"]
                    started = start(True, **tcp_options)
                else:
                    started = start()
            except BaseException as error:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "resume", resource, error
                )
                if not SequenceResourceLifecycleController._reusable_operation_is_current(
                    self, operation_token, "ACTIVE"
                ):
                    SequenceResourceLifecycleController._rollback_detached_reusable_start(
                        self, resource, start_target
                    )
                    return False
                if not SequenceResourceLifecycleController._release_detached_reusable_target(
                    self, resource, start_target
                ):
                    SequenceResourceLifecycleController._report_reusable_resource_failure(
                        self, "detached-release", resource, None
                    )
                    SequenceResourceLifecycleController._rollback_detached_reusable_start(
                        self, resource, start_target
                    )
                    return False
                SequenceResourceLifecycleController._rollback_replaced_reusable_start(
                    self, resource, entry, start_target
                )
                continue
            if not SequenceResourceLifecycleController._reusable_operation_is_current(
                self, operation_token, "ACTIVE"
            ):
                SequenceResourceLifecycleController._rollback_detached_reusable_start(
                    self, resource, start_target
                )
                return False
            if not SequenceResourceLifecycleController._release_detached_reusable_target(
                self, resource, start_target
            ):
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "detached-release", resource, None
                )
                SequenceResourceLifecycleController._rollback_detached_reusable_start(
                    self, resource, start_target
                )
                return False
            if started is False:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "resume", resource, None
                )
                SequenceResourceLifecycleController._rollback_replaced_reusable_start(
                    self, resource, entry, start_target
                )
                continue
            if not SequenceResourceLifecycleController._rollback_replaced_reusable_start(
                self, resource, entry, start_target
            ):
                continue
            if resource == "tcp":
                post_active = SequenceResourceLifecycleController._reusable_resource_is_active(
                    self, resource
                )
            else:
                manager = SequenceResourceLifecycleController._reusable_resource_target(self, resource)
                post_state = SequenceResourceLifecycleController._reusable_manager_active_state(
                    self, manager, resource
                )
                post_active = post_state is True
                if post_state is None and manager is not None:
                    with lock:
                        self._reusable_trusted_running_ids[resource] = id(manager)
                    post_active = True
            if not post_active:
                SequenceResourceLifecycleController._report_reusable_resource_failure(
                    self, "verify-resume", resource, None
                )
                continue
            SequenceResourceLifecycleController._set_reusable_entry_status(
                self, entry, "RUNNING"
            )
        if not SequenceResourceLifecycleController._reusable_operation_is_current(
            self, operation_token, "ACTIVE"
        ):
            return False
        with lock:
            review_epoch = self._reusable_resource_epoch
        stable = SequenceResourceLifecycleController._review_reusable_resources_after_event(
            self, journal, require_stopped=False
        )
        resumed = all(
            SequenceResourceLifecycleController._reusable_entry_is_complete(self, resource, entry)
            for resource, entry in journal.items()
        )
        self._reusable_resume_completed = {
            resource
            for resource, entry in journal.items()
            if SequenceResourceLifecycleController._reusable_entry_is_complete(
                self, resource, entry
            )
        }
        with lock:
            resumed = bool(
                resumed
                and stable
                and self._reusable_resource_epoch == review_epoch
            )
            if resumed:
                self._reusable_resource_state = "ACTIVE"
                self._reusable_resource_epoch += 1
                self._reusable_child_suspended = False
                self._reusable_resource_snapshot = None
                self._reusable_resource_journal = {}
                self._reusable_trusted_running_ids = {}
                self._reusable_suspend_completed = set()
                self._reusable_resume_completed = set()
                self._reusable_resume_pending = False
                return True
            self._reusable_resource_state = "RESUMING"
            queue_cleanup_generation = None
            if (
                not getattr(
                    self, "_reusable_cleanup_dispatch_active", False
                )
                and not getattr(
                    self, "_reusable_cleanup_event_pending", False
                )
            ):
                queue_cleanup_generation = (
                    SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(self)
                )
        if queue_cleanup_generation is not None:
            SequenceResourceLifecycleController._queue_reusable_cleanup_event(
                self, queue_cleanup_generation, delay_ms=0
            )
        return False

    @staticmethod
    def _rollback_replaced_reusable_start(
        owner, resource: str, entry, start_target
    ) -> bool:
        current = (
            SequenceResourceLifecycleController._reusable_tcp_target(owner)
            if resource == "tcp"
            else SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
        )
        if current is start_target:
            return True
        if start_target is None:
            return False
        if resource == "tcp":
            started_running = (
                SequenceResourceLifecycleController._reusable_tcp_target_is_active(owner, start_target)
                or SequenceResourceLifecycleController._reusable_tcp_target_has_pending(
                    owner, start_target
                )
            )
        else:
            state = SequenceResourceLifecycleController._reusable_manager_active_state(
                owner, start_target, resource
            )
            started_running = state is not False
        if not started_running:
            return True
        if not SequenceResourceLifecycleController._admit_reusable_pending_locked(
            owner, entry, start_target
        ):
            SequenceResourceLifecycleController._report_reusable_resource_failure(
                owner, "rollback-capacity", resource, None
            )
            return False
        stopped = SequenceResourceLifecycleController._stop_exact_reusable_target(
            owner, resource, start_target
        )
        SequenceResourceLifecycleController._journal_reentrant_reusable_target(
            owner, resource, entry, start_target
        )
        if not stopped:
            return False
        SequenceResourceLifecycleController._remove_reusable_pending_exact(
            owner, entry, id(start_target), start_target
        )
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        with lock:
            owner._reusable_trusted_running_ids.pop(resource, None)
        return True

    @staticmethod
    def _reusable_entry_is_complete(owner, resource: str, entry) -> bool:
        lock = SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
        if not SequenceResourceLifecycleController._reusable_pending_is_empty(owner, entry):
            return False
        with lock:
            desired = bool(entry["desired"])
            status = entry.get("status")
        if desired:
            active = SequenceResourceLifecycleController._reusable_resource_is_active(owner, resource)
            if resource != "tcp" and not active:
                current = SequenceResourceLifecycleController._reusable_resource_target(owner, resource)
                active = bool(
                    current is not None
                    and owner._reusable_trusted_running_ids.get(resource)
                    == id(current)
                )
            return bool(
                status == "RUNNING"
                and active
            )
        return bool(
            status == "STOPPED"
            and not SequenceResourceLifecycleController._reusable_resource_is_active(owner, resource)
        )

    def _lightweight_child_cleanup(self):
        """Suspend reusable child resources without dismantling MVC ownership."""
        return self._suspend_reusable_child_resources()


    def _prepare_application_shutdown_resources(
        self, shutdown_generation: int
    ) -> bool:
        if self._shutdown_prepared_generation == shutdown_generation:
            return True
        if self._shutdown_prepared_generation is not None:
            return False
        completed = getattr(self, "_shutdown_cleanup_steps_completed", None)
        if completed is None:
            completed = set()
            self._shutdown_cleanup_steps_completed = completed
        if "trigger" not in completed:
            if not self._suspend_reusable_child_resources():
                return False
            completed.add("trigger")
            self._shutdown_cleanup_trace.append("stop-trigger-resources")
        if "analysis" not in completed:
            try:
                self._view.close_analysis_windows()
            except BaseException:
                return False
            completed.add("analysis")
            self._shutdown_cleanup_trace.append("close-analysis-windows")
        if "dialogs" not in completed:
            try:
                self._view.close_application_subwindows()
            except BaseException:
                return False
            completed.add("dialogs")
        self._shutdown_prepared_generation = shutdown_generation
        return True

    def finalize_application_shutdown(self, shutdown_generation: int) -> bool:
        if self._shutdown_finalized_generation == shutdown_generation:
            return True
        if self._shutdown_prepared_generation != shutdown_generation:
            return False
        self._shutdown_finalized_generation = shutdown_generation
        return True

    def _complete_application_shutdown_delivery(
        self, shutdown_generation: int
    ) -> bool:
        if self._shutdown_delivery_completed_generation == shutdown_generation:
            return True
        if self._shutdown_finalized_generation != shutdown_generation:
            return False
        completed = getattr(self, "_shutdown_cleanup_steps_completed", None)
        if completed is None:
            completed = set()
            self._shutdown_cleanup_steps_completed = completed
        if "final-owners" not in completed:
            if not self._disconnect_trigger_inputs(
                shutdown_generation, close_dispatcher=False
            ):
                return False
            completed.add("final-owners")
            self._shutdown_cleanup_trace.append("disconnect-sequence-owners")
        self._shutdown_delivery_completed_generation = shutdown_generation
        return True

    def complete_application_shutdown_after_ready_ack(
        self, shutdown_generation: int
    ) -> bool:
        if (
            getattr(self, "_shutdown_dispatchers_closed_generation", None)
            == shutdown_generation
        ):
            return True
        if self._shutdown_delivery_completed_generation != shutdown_generation:
            return False
        completed = getattr(self, "_shutdown_cleanup_steps_completed", None)
        if completed is None:
            completed = set()
            self._shutdown_cleanup_steps_completed = completed
        bus = object.__getattribute__(self, "_lifecycle_bus")
        close_dispatcher = getattr(
            bus, "close_workflow_continuation_dispatcher", None
        )
        if "final-dispatcher" not in completed and callable(close_dispatcher):
            if close_dispatcher() is False:
                return False
            completed.add("final-dispatcher")
        close_lifecycle_dispatcher = getattr(
            bus, "close_resource_lifecycle_dispatcher", None
        )
        if (
            "final-lifecycle-dispatcher" not in completed
            and callable(close_lifecycle_dispatcher)
        ):
            if close_lifecycle_dispatcher() is False:
                return False
            completed.add("final-lifecycle-dispatcher")
        self._shutdown_dispatchers_closed_generation = shutdown_generation
        return True
