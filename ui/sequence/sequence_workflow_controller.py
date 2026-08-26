"""Admission and terminal-event routing for the canonical sequence workflow."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import logging
from pathlib import PosixPath, PurePosixPath, PureWindowsPath, WindowsPath
from threading import RLock
from typing import Any
from uuid import uuid4
from weakref import WeakMethod, ref

from PyQt5 import sip
from PyQt5.QtCore import QObject, QTimer, Qt, pyqtSignal, pyqtSlot

from ui.sequence.sequence_event_bus import (
    ImportTerminalRecipientResult,
    RetainedCleanupLifecycleRegistrationResult,
    SequenceEventBus,
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
)
from ui.sequence.sequence_messages import (
    AbortShutdownRequested,
    AnalysisCompleted,
    AnalysisExportPrepared,
    AnalysisExportPreparationFailed,
    AnalysisFailed,
    AnalysisRequested,
    AnalysisTransportReady,
    BeginRecordingRequested,
    BeginShutdownFlushRequested,
    CancelAnalysisRequested,
    CancelExportPreparationRequested,
    CancelExportRequested,
    CancelImportedAudioRequested,
    CancelRecordingRequested,
    CancelWorkflowRequested,
    CommitRecordingLabelRequested,
    ConfigurationSnapshot,
    ConfirmShutdownCancellationRequested,
    ExportCompleted,
    ExportFailed,
    ExportPreparationCancelled,
    ExportRetryAccepted,
    ExportRequested,
    IgnoreExportFailureRequested,
    ImportAudioRequested,
    ImportedAudioFailed,
    ImportedAudioReady,
    LoadImportedAudioRequested,
    ManualAnalysisRequested,
    ManualLabelExportPrepared,
    ManualLabelExportPreparationFailed,
    ManualLabelRequested,
    PrepareAnalysisExportRequested,
    PrepareManualLabelExportRequested,
    RecordingCancelled,
    RecordingCompleted,
    RecordingFailed,
    RecordingLabelCommitFailed,
    RecordingLabelCommitted,
    RecordingStarted,
    ReplayRequested,
    RetryExportRequested,
    ShutdownReady,
    ShutdownAborted,
    ShutdownFlushCompleted,
    ShutdownRequested,
    StartTestRequested,
    WorkflowCommandRejected,
    WorkflowStateChanged,
)
from ui.sequence.sequence_workflow_model import (
    ExportContinuation,
    PostAnalysisContinuation,
    SessionOrigin,
    SequenceWorkflowModel,
    WorkflowPhase,
)
from ui.sequence.sequence_workflow_policy import (
    AutomaticAnalysisDecision,
    AutomaticAnalysisPolicyPort,
    AutomaticAnalysisSource,
    SequenceAutomaticAnalysisPolicyService,
)


def _new_identifier() -> str:
    return uuid4().hex


def _default_configuration_snapshot() -> ConfigurationSnapshot:
    return ConfigurationSnapshot(sequence_config=(), analysis_config={})


def _default_session_snapshot(command: Any, snapshot: ConfigurationSnapshot) -> dict:
    return {
        "command_id": command.command_id,
        "source": command.source,
        "record_id": getattr(command, "record_id", command.command_id),
        "configuration": snapshot,
    }


def _default_imported_audio_readiness(event: ImportedAudioReady) -> Any:
    if event.recording_snapshot is None:
        return False, "imported recording snapshot is incomplete"
    return True


def _default_import_readiness(
    command: ImportAudioRequested, _snapshot: ConfigurationSnapshot
) -> Any:
    if command.mode not in {"IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"}:
        return False, "import mode is not supported"
    selected_path = command.selected_path
    if selected_path is None or type(selected_path) is str:
        return True
    if type(selected_path) in {
        PurePosixPath,
        PureWindowsPath,
        PosixPath,
        WindowsPath,
    }:
        return True
    return False, "import path must be plain text or an exact supported PurePath"


def _default_record_id_lookup(snapshot: Any) -> str | None:
    if isinstance(snapshot, Mapping):
        value = snapshot.get("record_id")
        if type(value) is str and value:
            return value
    return None


def _default_diagnostic_callback(context: Mapping[str, Any]) -> None:
    logging.getLogger(__name__).debug(
        "Ignored stale or mismatched sequence workflow input",
        extra={"workflow_diagnostic": dict(context)},
    )


def _qt_object_is_deleted(value: Any) -> bool:
    """Treat an uninspectable Qt wrapper as unavailable during teardown."""
    try:
        return bool(sip.isdeleted(value))
    except BaseException:
        return True


class _QueuedDeliveryGuard(QObject):
    """QObject receiver that can suppress metacalls already posted during teardown."""

    def __init__(self, owner: "SequenceWorkflowController", handler_name: str) -> None:
        super().__init__(owner)
        self._owner_ref = ref(owner)
        self._handler_name = handler_name

    @pyqtSlot(object)
    def deliver(self, message: Any) -> None:
        owner = self._owner_ref()
        if owner is not None and owner._accept_queued_delivery:
            getattr(owner, self._handler_name)(message)


@dataclass(slots=True)
class _PendingContinuationPublication:
    delivery_id: tuple[Any, ...]
    kind: str
    signal: Any
    message: Any
    authorization_established: bool | None = None
    permanent_rejection_callback: (
        Callable[
            ["_PendingContinuationPublication", WorkflowContinuationDeliveryOutcome],
            bool,
        ]
        | None
    ) = None


@dataclass(slots=True)
class _AnalysisExportPreparationCorrelation:
    request: PrepareAnalysisExportRequested
    settlement: str | None = None

    def matches_response(self, response: Any) -> bool:
        request = self.request
        return (
            response.request_id == request.request_id
            and response.analysis_id == request.analysis_id
            and response.source_id == request.source_id
            and response.record_id == request.record_id
            and response.workflow_generation == request.workflow_generation
        )


@dataclass(frozen=True, slots=True)
class _ImportTerminalPublication:
    signal: Any
    message: Any
    critical: bool


@dataclass(slots=True)
class _ImportTerminalWorkflowCommit:
    message: ImportedAudioReady | ImportedAudioFailed
    publications: list[_ImportTerminalPublication]
    next_publication: int = 0
    handler_complete: bool = False
    resolved: bool = False
    in_progress: bool = False


@dataclass(frozen=True, slots=True)
class _StagedExportContinuation:
    kind: str
    outcome: Any
    command: CommitRecordingLabelRequested | None = None
    transport: AnalysisTransportReady | None = None
    labeled_result_snapshot: Any = None


@dataclass(frozen=True, slots=True)
class _PendingRetainedRecordingCleanup:
    terminal: RecordingLabelCommitted
    workflow_generation: int

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (
            self.terminal.command_id,
            self.terminal.record_id,
            self.terminal.label,
            self.workflow_generation,
        )


def _retire_timer_on_owner_thread(
    timer: QTimer,
    timeout_callback: Callable[[], None] | None,
) -> tuple[str, ...]:
    """Run only on ``timer.thread()``; failures remain bounded diagnostics."""
    failures = []
    operations = (
        ("stop", timer.stop),
        (
            "disconnect",
            lambda: timer.timeout.disconnect(timeout_callback),
        ),
        ("unparent", lambda: timer.setParent(None)),
        ("delete", timer.deleteLater),
    )
    for operation, retire in operations:
        try:
            if retire() is False:
                failures.append(operation)
        except BaseException as error:
            failures.append(f"{operation}:{type(error).__name__}")
            logging.getLogger(__name__).debug(
                "Retained cleanup timer %s failed",
                operation,
                exc_info=True,
            )
    return tuple(failures)


class _RetainedCleanupTimerRetirementBridge(QObject):
    """Marshal timer retirement onto the timer's owning Qt thread."""

    retire_requested = pyqtSignal()

    def __init__(
        self,
        timer: QTimer,
        timeout_callback: Callable[[], None],
        failure_callback: Callable[[tuple[str, ...]], None],
    ) -> None:
        super().__init__(timer)
        self._timer_ref = ref(timer)
        self._timeout_callback = timeout_callback
        self._failure_callback_ref = WeakMethod(failure_callback)
        self._request_lock = RLock()
        self._requested = False
        self._completed = False
        timer._sequence_retained_cleanup_retirement_bridge = self
        self.retire_requested.connect(
            self._retire_on_timer_thread,
            Qt.QueuedConnection,
        )

    def request_retirement(self) -> bool:
        with self._request_lock:
            if self._requested or self._completed:
                return True
            self._requested = True
        try:
            self.retire_requested.emit()
        except BaseException:
            logging.getLogger(__name__).debug(
                "Retained cleanup timer retirement dispatch failed",
                exc_info=True,
            )
            return False
        return True

    @pyqtSlot()
    def _retire_on_timer_thread(self) -> None:
        if self._completed:
            return
        timer = self._timer_ref()
        timeout_callback = self._timeout_callback
        self._timeout_callback = None
        failures = ()
        if timer is not None and not _qt_object_is_deleted(timer):
            if (
                getattr(
                    timer,
                    "_sequence_retained_cleanup_retirement_bridge",
                    None,
                )
                is self
            ):
                del timer._sequence_retained_cleanup_retirement_bridge
            failures = _retire_timer_on_owner_thread(
                timer,
                timeout_callback,
            )
        self._completed = True
        failure_callback = self._failure_callback_ref()
        self._failure_callback_ref = lambda: None
        if failures and failure_callback is not None:
            failure_callback(failures)


@dataclass(slots=True, weakref_slot=True)
class _NativeRetainedCleanupLifecycle:
    """Python-only state that remains safe while its QObject owner is dying."""

    model: SequenceWorkflowModel | None
    diagnostic_callback: Callable[[Mapping[str, Any]], None]
    pending: _PendingRetainedRecordingCleanup | None = None
    retry_attempt: int = 0
    retry_delay_ms: int = 0
    last_diagnostic: dict[str, Any] | None = None
    finalized: bool = False
    cleanup_acknowledged: bool = False
    retry_timer_ref: Callable[[], QTimer | None] | None = None
    retry_timer_retirement_bridge: (
        _RetainedCleanupTimerRetirementBridge | None
    ) = None
    registry_owner_ref: Callable[[], SequenceEventBus | None] | None = None
    registry_token: object | None = None
    retiring: bool = False
    retired: bool = False

    def register_native_finalization_root(
        self,
        bus: SequenceEventBus,
        token: object,
    ) -> RetainedCleanupLifecycleRegistrationResult:
        if self.registry_token is token:
            owner = (
                None
                if self.registry_owner_ref is None
                else self.registry_owner_ref()
            )
            if owner is None or _qt_object_is_deleted(owner):
                return (
                    RetainedCleanupLifecycleRegistrationResult
                    .CLOSED_NATIVE_DELETED
                )
            if owner is not bus:
                return (
                    RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
                )
        elif self.registry_token is not None:
            return (
                RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
            )
        if not isinstance(bus, QObject):
            return RetainedCleanupLifecycleRegistrationResult.UNAVAILABLE
        try:
            owner_ref = ref(bus)
        except TypeError:
            # Lightweight non-QObject test adapters may not support weakrefs.
            # They do not own a surviving native timer and use the weak
            # lifecycle fallback in the timeout closure.
            return RetainedCleanupLifecycleRegistrationResult.UNAVAILABLE
        register = getattr(
            bus, "_register_retained_cleanup_lifecycle", None
        )
        if not callable(register):
            return RetainedCleanupLifecycleRegistrationResult.UNAVAILABLE
        result = register(token, self)
        if result not in (
            RetainedCleanupLifecycleRegistrationResult.REGISTERED,
            RetainedCleanupLifecycleRegistrationResult.IDEMPOTENT,
        ):
            return result
        self.registry_owner_ref = owner_ref
        self.registry_token = token
        return result

    def _retire_native_finalization_root(self) -> bool:
        owner_ref = self.registry_owner_ref
        token = self.registry_token
        self.registry_owner_ref = None
        self.registry_token = None
        if owner_ref is None or token is None:
            return False
        bus = owner_ref()
        if bus is None:
            return False
        retire = getattr(bus, "_retire_retained_cleanup_lifecycle", None)
        if not callable(retire):
            return False
        return retire(token, self)

    def _record_timer_retirement_failures(
        self,
        failures: tuple[str, ...],
    ) -> None:
        self.last_diagnostic = {
            "domain": "recording",
            "event_kind": "retained_cleanup_timer_retirement",
            "failure_operations": failures,
        }

    def _retire_retry_timer(self) -> tuple[str, ...]:
        bridge = self.retry_timer_retirement_bridge
        self.retry_timer_retirement_bridge = None
        self.retry_timer_ref = None
        if bridge is None or _qt_object_is_deleted(bridge):
            return ()
        if bridge.request_retirement():
            return ()
        return ("dispatch",)

    def retire(self) -> None:
        """Release bus-owned retry state without performing domain actions."""
        if self.retired or self.retiring:
            return
        self.retiring = True
        self._retire_native_finalization_root()
        self.finalized = True
        self.cleanup_acknowledged = False
        self.pending = None
        self.retry_attempt = 0
        self.retry_delay_ms = 0
        self.model = None
        self.diagnostic_callback = _default_diagnostic_callback
        failures = self._retire_retry_timer()
        if failures:
            self._record_timer_retirement_failures(failures)
        self.retired = True
        self.retiring = False

    def finalize(
        self,
        reason: str,
        *,
        pending: _PendingRetainedRecordingCleanup | None = None,
        cleanup_acknowledged: bool = False,
    ) -> bool:
        terminal = pending if pending is not None else self.pending
        first_resolution = not self.finalized
        self.finalized = True
        self._retire_native_finalization_root()
        self.cleanup_acknowledged = bool(
            self.cleanup_acknowledged or cleanup_acknowledged
        )
        self.pending = None
        self.retry_attempt = 0
        self.retry_delay_ms = 0
        self._retire_retry_timer()
        model = self.model
        if terminal is None or model is None:
            self.diagnostic_callback = _default_diagnostic_callback
            return first_resolution

        if first_resolution:
            diagnostic = {
                "domain": "recording",
                "event_kind": "retained_recording_cleanup_retry",
                "reason": reason,
                "current_phase": model.phase.name,
                "workflow_generation": model.workflow_generation,
                "pending_identity": terminal.identity,
                "retry_attempt": 0,
                "failure_type": "NativeDeletion",
            }
            self.last_diagnostic = diagnostic
            try:
                self.diagnostic_callback(dict(diagnostic))
            except BaseException:
                logging.getLogger(__name__).debug(
                    "Native retained cleanup diagnostic callback failed",
                    exc_info=True,
                )

        if (
            self.cleanup_acknowledged
            and model.retained_record_id == terminal.terminal.record_id
        ):
            model.retained_record_id = None
            model.awaiting_label = False

        model.active_session_id = None
        model.active_session_origin = None
        model.active_import_id = None
        model.active_analysis_id = None
        model.active_job_id = None
        model.active_attempt_id = None
        model.clear_export_attempt_history()
        model.analysis_source_id = None
        model.analysis_record_id = None
        model.export_record_id = None
        model.export_continuation = None
        model.post_analysis_continuation = None
        model.export_failure_pending = False
        model.cancelling_phase = None
        model.cancelling_domain = None
        model.active_label_command_id = None
        model.active_label_record_id = None
        model.active_label = None
        model.configuration_snapshot = None
        model.session_snapshot = None
        model.recording_snapshot = None
        model.import_reference_snapshot = None
        model.labeled_result_snapshot = None
        model.shutdown_generation = None
        model.shutdown_pending = False
        model.shutdown_asserted_active = False
        model.shutdown_cancellation_confirmed = False
        model.phase = WorkflowPhase.IDLE
        try:
            model.assert_invariants()
        except BaseException as error:
            self.last_diagnostic = {
                "domain": "recording",
                "event_kind": "retained_recording_cleanup_abandoned",
                "reason": reason,
                "failure_type": type(error).__name__,
                "pending_identity": terminal.identity,
            }
        self.model = None
        self.diagnostic_callback = _default_diagnostic_callback
        return first_resolution


class SequenceWorkflowController(QObject):
    """Own cross-domain state transitions without invoking another Controller."""

    @property
    def _pending_retained_cleanup(
        self,
    ) -> _PendingRetainedRecordingCleanup | None:
        return self._native_retained_cleanup_lifecycle.pending

    @_pending_retained_cleanup.setter
    def _pending_retained_cleanup(
        self, value: _PendingRetainedRecordingCleanup | None
    ) -> None:
        self._native_retained_cleanup_lifecycle.pending = value

    @property
    def _retained_cleanup_retry_attempt(self) -> int:
        return self._native_retained_cleanup_lifecycle.retry_attempt

    @_retained_cleanup_retry_attempt.setter
    def _retained_cleanup_retry_attempt(self, value: int) -> None:
        self._native_retained_cleanup_lifecycle.retry_attempt = value

    @property
    def _retained_cleanup_retry_delay_ms(self) -> int:
        return self._native_retained_cleanup_lifecycle.retry_delay_ms

    @_retained_cleanup_retry_delay_ms.setter
    def _retained_cleanup_retry_delay_ms(self, value: int) -> None:
        self._native_retained_cleanup_lifecycle.retry_delay_ms = value

    @property
    def _retained_cleanup_last_diagnostic(self) -> dict[str, Any] | None:
        return self._native_retained_cleanup_lifecycle.last_diagnostic

    @_retained_cleanup_last_diagnostic.setter
    def _retained_cleanup_last_diagnostic(
        self, value: dict[str, Any] | None
    ) -> None:
        self._native_retained_cleanup_lifecycle.last_diagnostic = value

    @property
    def diagnostic_callback(self) -> Callable[[Mapping[str, Any]], None]:
        return self._native_retained_cleanup_lifecycle.diagnostic_callback

    @diagnostic_callback.setter
    def diagnostic_callback(
        self, value: Callable[[Mapping[str, Any]], None]
    ) -> None:
        self._native_retained_cleanup_lifecycle.diagnostic_callback = value

    def __init__(
        self,
        model: SequenceWorkflowModel,
        bus: SequenceEventBus,
        *,
        session_id_factory: Callable[[], str] = _new_identifier,
        import_id_factory: Callable[[], str] = _new_identifier,
        analysis_id_factory: Callable[[], str] = _new_identifier,
        job_id_factory: Callable[[], str] = _new_identifier,
        label_id_factory: Callable[[], str] = _new_identifier,
        preparation_id_factory: Callable[[], str] = _new_identifier,
        configuration_snapshot_provider: Callable[[], ConfigurationSnapshot] = (
            _default_configuration_snapshot
        ),
        start_readiness: Callable[[Any, ConfigurationSnapshot], Any] = (
            lambda _command, _snapshot: True
        ),
        replay_readiness: Callable[[Any, ConfigurationSnapshot], Any] = (
            lambda _command, _snapshot: True
        ),
        import_readiness: Callable[[Any, ConfigurationSnapshot], Any] = (
            _default_import_readiness
        ),
        imported_audio_readiness: Callable[[ImportedAudioReady], Any] = (
            _default_imported_audio_readiness
        ),
        session_snapshot_factory: Callable[[Any, ConfigurationSnapshot], Any] = (
            _default_session_snapshot
        ),
        recording_snapshot_lookup: Callable[[str], Any] = lambda _record_id: None,
        retain_recording_snapshot: Callable[..., bool] = (
            lambda _record_id, _recording, _configuration, **_identity: True
        ),
        clear_retained_recording_snapshot: Callable[..., bool | None] = (
            lambda _record_id, **_identity: True
        ),
        record_id_lookup: Callable[[Any], str | None] = _default_record_id_lookup,
        automatic_analysis_policy: AutomaticAnalysisPolicyPort | None = None,
        export_decision_requires_terminal: bool = False,
        diagnostic_callback: Callable[[Mapping[str, Any]], None] = (
            _default_diagnostic_callback
        ),
        parent: QObject | None = None,
        connect_bus: bool = True,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.bus = bus
        bind_recording_admission_owner = getattr(
            self.bus,
            "_bind_canonical_recording_workflow_owner",
            None,
        )
        self._canonical_recording_admission_capability = (
            bind_recording_admission_owner(self)
            if callable(bind_recording_admission_owner)
            else None
        )
        self._recording_admission_closed = False
        self._recording_admission_lock = RLock()
        self._recording_admission_epoch = 0
        self.session_id_factory = session_id_factory
        self.import_id_factory = import_id_factory
        self.analysis_id_factory = analysis_id_factory
        self.job_id_factory = job_id_factory
        self.label_id_factory = label_id_factory
        self.preparation_id_factory = preparation_id_factory
        self.configuration_snapshot_provider = configuration_snapshot_provider
        self.start_readiness = start_readiness
        self.replay_readiness = replay_readiness
        self.import_readiness = import_readiness
        self.imported_audio_readiness = imported_audio_readiness
        # Factories may omit workflow_generation for compatibility. They must return
        # a data Mapping; a supplied generation must match the pending admission.
        self.session_snapshot_factory = session_snapshot_factory
        self.recording_snapshot_lookup = recording_snapshot_lookup
        self.retain_recording_snapshot = retain_recording_snapshot
        self.clear_retained_recording_snapshot = (
            clear_retained_recording_snapshot
        )
        self._native_retained_cleanup_lifecycle = (
            _NativeRetainedCleanupLifecycle(model, diagnostic_callback)
        )
        retained_cleanup_registry_token = object()
        self._retained_cleanup_registry_token = (
            retained_cleanup_registry_token
        )
        self._label_terminal_cleanup_lock = RLock()
        self._label_terminal_cleanup_active = False
        self._label_terminal_cleanup_reentered = False
        self._pending_retained_cleanup: (
            _PendingRetainedRecordingCleanup | None
        ) = None
        self._retained_cleanup_retry_base_delay_ms = 10
        self._retained_cleanup_retry_max_delay_ms = 1_000
        self._retained_cleanup_retry_attempt = 0
        self._retained_cleanup_retry_delay_ms = 0
        self._retained_cleanup_last_diagnostic: dict[str, Any] | None = None
        # The retry timer outlives an independently native-deleted controller.
        # Its callback holds only weak Python ownership and can settle the
        # lifecycle capsule without touching a dead QObject wrapper.
        timer_parent = self.bus if isinstance(self.bus, QObject) else None
        self._retained_cleanup_retry_timer = QTimer(timer_parent)
        self._retained_cleanup_retry_timer.setSingleShot(True)
        self._native_retained_cleanup_lifecycle.retry_timer_ref = ref(
            self._retained_cleanup_retry_timer
        )
        retry_owner_ref = ref(self)
        try:
            retained_cleanup_bus_ref = (
                ref(self.bus) if isinstance(self.bus, QObject) else None
            )
        except TypeError:
            retained_cleanup_bus_ref = None
        fallback_native_lifecycle_ref = ref(
            self._native_retained_cleanup_lifecycle
        )

        def retry_retained_cleanup() -> None:
            native_lifecycle = fallback_native_lifecycle_ref()
            if native_lifecycle is None:
                return
            if retained_cleanup_bus_ref is not None:
                bus = retained_cleanup_bus_ref()
                if bus is None or _qt_object_is_deleted(bus):
                    return
                resolve = getattr(
                    bus, "_resolve_retained_cleanup_lifecycle", None
                )
                if not callable(resolve):
                    return
                resolved_lifecycle = resolve(
                    retained_cleanup_registry_token
                )
                if resolved_lifecycle is not native_lifecycle:
                    # The timeout belongs to the lifecycle captured when this
                    # timer was created. An exact token may later be reused by
                    # a replacement owner; stale callbacks must never dispatch
                    # to or finalize that replacement.
                    return
            owner = retry_owner_ref()
            if owner is None or _qt_object_is_deleted(owner):
                native_lifecycle.finalize(
                    "native workflow owner unavailable at retained cleanup retry",
                )
                return
            owner._retry_pending_retained_cleanup()

        self._retained_cleanup_retry_timer.timeout.connect(
            retry_retained_cleanup
        )
        self._native_retained_cleanup_lifecycle.retry_timer_retirement_bridge = (
            _RetainedCleanupTimerRetirementBridge(
                self._retained_cleanup_retry_timer,
                retry_retained_cleanup,
                self._native_retained_cleanup_lifecycle._record_timer_retirement_failures,
            )
        )
        self.record_id_lookup = record_id_lookup
        if automatic_analysis_policy is None:
            automatic_analysis_policy = SequenceAutomaticAnalysisPolicyService()
        if not isinstance(automatic_analysis_policy, AutomaticAnalysisPolicyPort):
            raise TypeError(
                "automatic_analysis_policy must implement AutomaticAnalysisPolicyPort"
            )
        self.automatic_analysis_policy = automatic_analysis_policy
        self.export_decision_requires_terminal = bool(
            export_decision_requires_terminal
        )
        self.diagnostic_callback = diagnostic_callback
        self._seen_domain_ids: dict[str, set[str]] = {
            "session": set(),
            "import": set(),
            "analysis": set(),
            "job": set(),
            "preparation": set(),
        }
        self._connections: list[tuple[Any, _QueuedDeliveryGuard]] = []
        self._pending_local_import_failure_notifications: dict[
            int, ImportedAudioFailed
        ] = {}
        self._pending_import_terminal_commit: (
            _ImportTerminalWorkflowCommit | None
        ) = None
        self._active_import_terminal_commit: (
            _ImportTerminalWorkflowCommit | None
        ) = None
        self._completed_import_terminal_commits: OrderedDict[
            tuple[str, str], ImportedAudioReady | ImportedAudioFailed
        ] = OrderedDict()
        self._import_terminal_commit_history_limit = 128
        self._import_terminal_publication_limit = 8
        self._pending_continuation_publications: OrderedDict[
            tuple[Any, ...], _PendingContinuationPublication
        ] = OrderedDict()
        self._continuation_outbox_limit = 128
        self._continuation_retry_base_delay_ms = 10
        self._continuation_retry_max_delay_ms = 1_000
        self._continuation_retry_attempt = 0
        self._continuation_retry_delay_ms = 0
        self._continuation_dispatch_active = True
        self._continuation_retry_timer = QTimer(self)
        self._continuation_retry_timer.setSingleShot(True)
        self._continuation_retry_timer.timeout.connect(
            self.retry_pending_continuation_publications
        )
        register_lifecycle_owner = getattr(
            self.bus,
            "register_workflow_continuation_lifecycle_owner",
            None,
        )
        if callable(register_lifecycle_owner):
            register_lifecycle_owner(self)
        self._accept_queued_delivery = False
        self._export_terminal_recipient_name = f"workflow:{id(self)}"
        self._import_terminal_recipient_name = f"workflow-import:{id(self)}"
        self._analysis_prepared_recipient_name = f"workflow-analysis-prepared:{id(self)}"
        self._analysis_preparation_failed_recipient_name = (
            f"workflow-analysis-preparation-failed:{id(self)}"
        )
        self._manual_prepared_recipient_name = f"workflow-manual-prepared:{id(self)}"
        self._manual_preparation_failed_recipient_name = (
            f"workflow-manual-preparation-failed:{id(self)}"
        )
        self._preparation_cancelled_recipient_name = (
            f"workflow-preparation-cancelled:{id(self)}"
        )
        self._pending_export_preparation: (
            PrepareAnalysisExportRequested | PrepareManualLabelExportRequested | None
        ) = None
        self._analysis_export_preparation_correlation: (
            _AnalysisExportPreparationCorrelation | None
        ) = None
        self._register_export_preparation_responses()
        if connect_bus:
            self._wire_bus()

    def _register_export_preparation_responses(self) -> None:
        register = getattr(
            self.bus, "register_workflow_continuation_recipient", None
        )
        if not callable(register):
            return
        for kind, name, handler in (
            (
                "analysis-export-prepared",
                self._analysis_prepared_recipient_name,
                self.handle_analysis_export_prepared,
            ),
            (
                "analysis-export-preparation-failed",
                self._analysis_preparation_failed_recipient_name,
                self.handle_analysis_export_preparation_failed,
            ),
            (
                "manual-label-export-prepared",
                self._manual_prepared_recipient_name,
                self.handle_manual_label_export_prepared,
            ),
            (
                "manual-label-export-preparation-failed",
                self._manual_preparation_failed_recipient_name,
                self.handle_manual_label_export_preparation_failed,
            ),
            (
                "export-preparation-cancelled",
                self._preparation_cancelled_recipient_name,
                self.handle_export_preparation_cancelled,
            ),
        ):
            register(kind, name, handler, owner=self)

    def _wire_bus(self) -> None:
        self._accept_queued_delivery = True
        commands = self.bus.commands
        events = self.bus.events
        register_import_terminal = getattr(
            self.bus, "register_import_terminal_recipient", None
        )
        formal_import_terminal = callable(register_import_terminal)
        connections = [
            (commands.start_test_requested, self.handle_start),
            (commands.replay_requested, self.handle_replay),
            (commands.import_audio_requested, self.handle_import),
            (commands.manual_analysis_requested, self.handle_manual_analysis),
            (commands.manual_label_requested, self.handle_manual_label),
            (commands.cancel_workflow_requested, self.handle_cancel_workflow),
            (commands.retry_export_requested, self.handle_retry_export),
            (
                commands.ignore_export_failure_requested,
                self.handle_ignore_export_failure,
            ),
            (commands.shutdown_requested, self.handle_shutdown),
            (
                commands.confirm_shutdown_cancellation_requested,
                self.handle_confirm_shutdown_cancellation,
            ),
            (commands.abort_shutdown_requested, self.handle_abort_shutdown),
            (events.recording_started, self.handle_recording_started),
            (events.recording_completed, self.handle_recording_completed),
            (events.recording_failed, self.handle_recording_failed),
            (events.recording_cancelled, self.handle_recording_cancelled),
            (events.analysis_completed, self.handle_analysis_completed),
            (events.analysis_failed, self.handle_analysis_failed),
            (events.export_completed, self.handle_export_completed),
            (events.export_failed, self.handle_export_failed),
            (events.recording_label_committed, self.handle_label_committed),
            (events.recording_label_commit_failed, self.handle_label_failed),
            (events.shutdown_ready, self.handle_shutdown_ready),
        ]
        if not formal_import_terminal:
            connections.extend(
                (
                    (events.imported_audio_ready, self.handle_imported_audio_ready),
                    (events.imported_audio_failed, self.handle_imported_audio_failed),
                )
            )
        for signal, slot in connections:
            guard = _QueuedDeliveryGuard(self, slot.__name__)
            signal.connect(guard.deliver, Qt.QueuedConnection)
            self._connections.append((signal, guard))
        register_retry = getattr(
            self.bus, "register_export_retry_recipient", None
        )
        if callable(register_retry):
            register_retry(self.handle_export_retry_accepted)
        register_terminal = getattr(
            self.bus, "register_export_terminal_recipient", None
        )
        if callable(register_terminal):
            register_terminal(
                self._export_terminal_recipient_name,
                self._deliver_export_terminal,
                critical=True,
            )
        if formal_import_terminal:
            register_import_terminal(
                self._import_terminal_recipient_name,
                self._deliver_import_terminal,
                owner=self,
                critical=True,
            )

    def disconnect(self, _lifecycle_request=None) -> None:
        if _qt_object_is_deleted(self):
            self._finalize_native_deletion(
                "native workflow owner disconnected after destruction"
            )
            self._disconnect_native_bus_references()
            return
        with self._recording_admission_lock:
            self._recording_admission_closed = True
            self._recording_admission_epoch += 1
        self._retire_active_recording_admission()
        self._release_recording_admission_capability()
        self._accept_queued_delivery = False
        self._continuation_dispatch_active = False
        self._retire_pending_retained_cleanup_for_disconnect()
        self._pending_local_import_failure_notifications.clear()
        self._pending_import_terminal_commit = None
        self._active_import_terminal_commit = None
        pending_publications = tuple(
            self._pending_continuation_publications.values()
        )
        bus_native_available = not (
            isinstance(self.bus, QObject) and _qt_object_is_deleted(self.bus)
        )
        abandon = (
            getattr(self.bus, "abandon_workflow_continuations", None)
            if bus_native_available
            else None
        )
        if callable(abandon):
            abandon(
                tuple(self._pending_continuation_publications),
                owner=self,
                reason="workflow-disconnect",
            )
        for publication in pending_publications:
            if publication.authorization_established is True:
                self.model.consume_analysis_transport(publication.message)
        self._pending_continuation_publications.clear()
        if not _qt_object_is_deleted(self._continuation_retry_timer):
            try:
                self._continuation_retry_timer.stop()
            except BaseException:
                # A child may be deleted independently while teardown is
                # entering. The controller is already dispatch-disabled.
                logging.getLogger(__name__).debug(
                    "Continuation retry timer unavailable during disconnect",
                    exc_info=True,
                )
        self._continuation_retry_attempt = 0
        self._continuation_retry_delay_ms = 0
        if not bus_native_available:
            self._connections.clear()
            return
        unregister_retry = getattr(
            self.bus, "unregister_export_retry_recipient", None
        )
        if callable(unregister_retry):
            unregister_retry(self.handle_export_retry_accepted)
        unregister_terminal = getattr(
            self.bus, "unregister_export_terminal_recipient", None
        )
        if callable(unregister_terminal):
            unregister_terminal(
                self._export_terminal_recipient_name,
                self._deliver_export_terminal,
            )
        unregister_import_terminal = getattr(
            self.bus, "unregister_import_terminal_recipient", None
        )
        if callable(unregister_import_terminal):
            unregister_import_terminal(
                self._import_terminal_recipient_name,
                self._deliver_import_terminal,
            )
        unregister_continuation = getattr(
            self.bus, "unregister_workflow_continuation_recipient", None
        )
        if callable(unregister_continuation):
            unregister_continuation(
                "analysis-export-prepared",
                self._analysis_prepared_recipient_name,
                self.handle_analysis_export_prepared,
            )
            unregister_continuation(
                "analysis-export-preparation-failed",
                self._analysis_preparation_failed_recipient_name,
                self.handle_analysis_export_preparation_failed,
            )
            unregister_continuation(
                "manual-label-export-prepared",
                self._manual_prepared_recipient_name,
                self.handle_manual_label_export_prepared,
            )
            unregister_continuation(
                "manual-label-export-preparation-failed",
                self._manual_preparation_failed_recipient_name,
                self.handle_manual_label_export_preparation_failed,
            )
            unregister_continuation(
                "export-preparation-cancelled",
                self._preparation_cancelled_recipient_name,
                self.handle_export_preparation_cancelled,
            )
        unregister_lifecycle_owner = getattr(
            self.bus,
            "unregister_workflow_continuation_lifecycle_owner",
            None,
        )
        if callable(unregister_lifecycle_owner):
            unregister_lifecycle_owner(self)
        while self._connections:
            signal, guard = self._connections.pop()
            try:
                signal.disconnect(guard.deliver)
            except TypeError:
                # QObject teardown may already have removed this exact connection.
                continue

    def _disconnect_native_bus_references(self) -> None:
        """Release bus-owned Python callbacks without touching dead Qt children."""
        bus = self.bus
        if isinstance(bus, QObject) and _qt_object_is_deleted(bus):
            return

        def invoke(name: str, *args) -> None:
            operation = getattr(bus, name, None)
            if not callable(operation):
                return
            try:
                operation(*args)
            except BaseException:
                logging.getLogger(__name__).debug(
                    "Native workflow bus retirement failed for %s",
                    name,
                    exc_info=True,
                )

        invoke(
            "unregister_export_retry_recipient",
            self.handle_export_retry_accepted,
        )
        invoke(
            "unregister_export_terminal_recipient",
            self._export_terminal_recipient_name,
            self._deliver_export_terminal,
        )

    def _allocate_domain_id(
        self, domain: str, factory: Callable[[], str]
    ) -> str:
        candidate = factory()
        if type(candidate) is not str or not candidate:
            raise TypeError(f"{domain} identifier factory must return a non-empty plain string")
        seen = self._seen_domain_ids[domain]
        if candidate not in seen:
            unique = candidate
        else:
            longest_seen = max(len(identifier) for identifier in seen)
            unique = f"{domain}:" + ("~" * (longest_seen + 1))
        seen.add(unique)
        return unique

    def _deliver_export_terminal(self, message: Any) -> bool:
        """BaseException-safe canonical dispatcher entry used by Export."""
        try:
            if type(message) is ExportCompleted:
                return self.handle_export_completed(message) is True
            if type(message) is ExportFailed:
                return self.handle_export_failed(message) is True
            return False
        except BaseException:
            return False

    def _deliver_import_terminal(
        self, message: Any
    ) -> ImportTerminalRecipientResult:
        """Stage one exact terminal and acknowledge only after publication."""
        if type(message) not in {ImportedAudioReady, ImportedAudioFailed}:
            return ImportTerminalRecipientResult.PERMANENT_REJECT
        identity = (type(message).__name__, message.import_id)
        completed = self._completed_import_terminal_commits.get(identity)
        if completed is not None:
            return (
                ImportTerminalRecipientResult.ACK
                if completed is message
                else ImportTerminalRecipientResult.PERMANENT_REJECT
            )
        commit = self._pending_import_terminal_commit
        if commit is None:
            commit = _ImportTerminalWorkflowCommit(message, [])
            self._pending_import_terminal_commit = commit
        elif commit.message is not message:
            return ImportTerminalRecipientResult.PERMANENT_REJECT
        if commit.in_progress:
            return ImportTerminalRecipientResult.RETRYABLE_NACK

        commit.in_progress = True
        try:
            if not commit.handler_complete:
                self._active_import_terminal_commit = commit
                try:
                    if type(message) is ImportedAudioReady:
                        handled = self.handle_imported_audio_ready(message)
                    else:
                        handled = self.handle_imported_audio_failed(message)
                except BaseException:
                    return ImportTerminalRecipientResult.RETRYABLE_NACK
                finally:
                    self._active_import_terminal_commit = None
                if handled is not True and not commit.resolved:
                    self._pending_import_terminal_commit = None
                    return ImportTerminalRecipientResult.PERMANENT_REJECT
                commit.handler_complete = True

            for index in range(
                commit.next_publication, len(commit.publications)
            ):
                publication = commit.publications[index]
                try:
                    publication.signal.emit(publication.message)
                except BaseException:
                    if publication.critical:
                        return ImportTerminalRecipientResult.RETRYABLE_NACK
                commit.next_publication = index + 1

            self._pending_import_terminal_commit = None
            self._completed_import_terminal_commits[identity] = message
            self._completed_import_terminal_commits.move_to_end(identity)
            if (
                len(self._completed_import_terminal_commits)
                > self._import_terminal_commit_history_limit
            ):
                self._completed_import_terminal_commits.popitem(last=False)
            return ImportTerminalRecipientResult.ACK
        finally:
            commit.in_progress = False

    def _stage_import_terminal_publication(
        self, signal: Any, message: Any, *, critical: bool
    ) -> bool:
        commit = self._active_import_terminal_commit
        if commit is None:
            signal.emit(message)
            return True
        if len(commit.publications) >= self._import_terminal_publication_limit:
            raise RuntimeError("import terminal publication outbox is full")
        commit.publications.append(
            _ImportTerminalPublication(signal, message, bool(critical))
        )
        return True

    def _resolve_formal_import_terminal(self) -> bool:
        commit = self._active_import_terminal_commit
        if commit is None:
            return False
        commit.resolved = True
        return True

    @property
    def pending_continuation_publication_ids(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(self._pending_continuation_publications)

    @property
    def pending_retained_cleanup_identity(self) -> tuple[str, str, str, int] | None:
        self._settle_native_retained_cleanup_observation()
        pending = self._pending_retained_cleanup
        return None if pending is None else pending.identity

    @property
    def retained_cleanup_retry_attempt(self) -> int:
        self._settle_native_retained_cleanup_observation()
        return self._retained_cleanup_retry_attempt

    @property
    def retained_cleanup_retry_delay_ms(self) -> int:
        self._settle_native_retained_cleanup_observation()
        return self._retained_cleanup_retry_delay_ms

    @property
    def retained_cleanup_last_diagnostic(self) -> Mapping[str, Any] | None:
        self._settle_native_retained_cleanup_observation()
        diagnostic = self._retained_cleanup_last_diagnostic
        return None if diagnostic is None else dict(diagnostic)

    def _settle_native_retained_cleanup_observation(self) -> None:
        if _qt_object_is_deleted(self):
            self._native_retained_cleanup_lifecycle.finalize(
                "native workflow owner observed destroyed"
            )

    @property
    def continuation_retry_delay_ms(self) -> int:
        return self._continuation_retry_delay_ms

    @property
    def continuation_retry_max_delay_ms(self) -> int:
        return self._continuation_retry_max_delay_ms

    def _schedule_continuation_retry(self) -> None:
        if (
            not self._continuation_dispatch_active
            or not self._pending_continuation_publications
            or self._continuation_retry_timer.isActive()
        ):
            return
        exponent = min(self._continuation_retry_attempt, 30)
        delay = min(
            self._continuation_retry_base_delay_ms * (2 ** exponent),
            self._continuation_retry_max_delay_ms,
        )
        self._continuation_retry_attempt += 1
        self._continuation_retry_delay_ms = delay
        self._continuation_retry_timer.start(delay)

    def _deliver_continuation_publication(
        self, publication: _PendingContinuationPublication
    ) -> WorkflowContinuationDeliveryOutcome:
        outcome_type = WorkflowContinuationDeliveryOutcome
        retryable = WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
        if publication.authorization_established is False:
            message = publication.message
            if not self.model.is_analysis_transport_authorized(
                message
            ) and not self.model.authorize_analysis_transport(message):
                return outcome_type(
                    retryable,
                    "analysis transport authorization is pending",
                )
            # Persist this checkpoint before EventBus can re-enter any
            # recipient. Its per-recipient ACK ledger owns all later retries.
            publication.authorization_established = True
        dispatcher = getattr(
            self.bus, "deliver_workflow_continuation_outcome", None
        )
        if callable(dispatcher):
            try:
                outcome = dispatcher(
                    publication.delivery_id,
                    publication.kind,
                    publication.message,
                    owner=self,
                )
            except BaseException:
                return outcome_type(
                    retryable,
                    "continuation dispatcher raised",
                )
            if type(outcome) is WorkflowContinuationDeliveryOutcome:
                return outcome
            return outcome_type(
                retryable,
                "continuation dispatcher returned an invalid outcome",
            )
        legacy_dispatcher = getattr(
            self.bus, "deliver_workflow_continuation", None
        )
        if callable(legacy_dispatcher):
            try:
                acknowledged = legacy_dispatcher(
                    publication.delivery_id,
                    publication.kind,
                    publication.message,
                    owner=self,
                ) is True
            except BaseException:
                acknowledged = False
            return outcome_type(
                (
                    WorkflowContinuationDeliveryStatus.ACK
                    if acknowledged
                    else retryable
                ),
                "" if acknowledged else "continuation acknowledgement is pending",
            )
        try:
            publication.signal.emit(publication.message)
        except BaseException:
            return outcome_type(
                retryable,
                "continuation signal emission raised",
            )
        return outcome_type(WorkflowContinuationDeliveryStatus.ACK)

    def _settle_permanently_rejected_publication(
        self,
        publication: _PendingContinuationPublication,
        outcome: WorkflowContinuationDeliveryOutcome,
    ) -> bool:
        callback = publication.permanent_rejection_callback
        return bool(
            callback is not None
            and callback(publication, outcome) is True
        )

    def _publish_continuation(
        self,
        delivery_id: tuple[Any, ...],
        signal: Any,
        message: Any,
        *,
        requires_analysis_transport_authorization: bool = False,
        permanent_rejection_callback: (
            Callable[
                [_PendingContinuationPublication, WorkflowContinuationDeliveryOutcome],
                bool,
            ]
            | None
        ) = None,
    ) -> bool:
        pending = self._pending_continuation_publications
        if not self._continuation_dispatch_active:
            return False
        if delivery_id in pending:
            return False
        if len(pending) >= self._continuation_outbox_limit:
            raise RuntimeError("workflow continuation outbox is full")
        kind = delivery_id[0] if delivery_id else ""
        if type(kind) is not str or not kind:
            raise ValueError("continuation delivery kind is unavailable")
        publication = _PendingContinuationPublication(
            delivery_id,
            kind,
            signal,
            message,
            (
                not requires_analysis_transport_authorization
                if kind == "analysis-transport"
                else None
            ),
            permanent_rejection_callback,
        )
        if pending:
            pending[delivery_id] = publication
            self._schedule_continuation_retry()
            return False
        outcome = self._deliver_continuation_publication(publication)
        if outcome.status is WorkflowContinuationDeliveryStatus.ACK:
            # A synchronous terminal recipient may re-enter Workflow and enqueue
            # the next formal continuation before this delivery returns. Preserve
            # that new outbox entry and its retry diagnostics.
            if not pending:
                self._continuation_retry_attempt = 0
                self._continuation_retry_delay_ms = 0
            return True
        if (
            outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
            and self._settle_permanently_rejected_publication(
                publication, outcome
            )
        ):
            return False
        pending[delivery_id] = publication
        self._schedule_continuation_retry()
        return False

    def retry_pending_continuation_publications(self) -> bool:
        self._continuation_retry_timer.stop()
        if not self._continuation_dispatch_active:
            return False
        pending = self._pending_continuation_publications
        for delivery_id in tuple(pending):
            publication = pending.get(delivery_id)
            if publication is None:
                continue
            outcome = self._deliver_continuation_publication(publication)
            if outcome.status is WorkflowContinuationDeliveryStatus.ACK:
                pending.pop(delivery_id, None)
                continue
            if (
                outcome.status
                is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
                and self._settle_permanently_rejected_publication(
                    publication, outcome
                )
            ):
                continue
            if outcome.status is not WorkflowContinuationDeliveryStatus.ACK:
                self._schedule_continuation_retry()
                return False
        self._continuation_retry_attempt = 0
        self._continuation_retry_delay_ms = 0
        return True

    @staticmethod
    def _readiness(result: Any, default_reason: str) -> tuple[bool, str]:
        if type(result) is bool:
            return result, "" if result else default_reason
        if type(result) is tuple and len(result) == 2 and type(result[0]) is bool:
            reason = result[1]
            if type(reason) is not str:
                raise TypeError("readiness reason must be a plain string")
            return result[0], reason if not result[0] else ""
        raise TypeError("readiness collaborator must return bool or (bool, reason)")

    def _reject(self, command_id: str, reason: str) -> bool:
        self.bus.events.workflow_command_rejected.emit(
            WorkflowCommandRejected(
                command_id=command_id,
                current_phase=self.model.phase.name,
                reason=reason,
            )
        )
        return False

    def _shutdown_blocks_admission(self, command_id: str) -> bool:
        if self._pending_continuation_publications:
            self._reject(command_id, "workflow continuation publication is pending")
            return True
        if not self.model.shutdown_pending:
            return False
        self._reject(command_id, "shutdown is pending")
        return True

    def _diagnose_stale(
        self,
        domain: str,
        event_kind: str,
        reason: str,
        **context: Any,
    ) -> bool:
        diagnostic = {
            "domain": domain,
            "event_kind": event_kind,
            "reason": reason,
            "current_phase": self.model.phase.name,
            "workflow_generation": self.model.workflow_generation,
            "shutdown_generation": self.model.shutdown_generation,
        }
        diagnostic.update(context)
        self.diagnostic_callback(diagnostic)
        return False

    def _transition(
        self,
        phase: WorkflowPhase,
        *,
        continuation_publication: bool = False,
    ) -> None:
        previous = self.model.phase
        if previous is phase:
            return
        self.model.phase = phase
        self.model.assert_invariants()
        active_session_id, active_import_id, active_analysis_id, active_job_id = (
            self._phase_active_identifiers(phase)
        )
        message = WorkflowStateChanged(
            workflow_generation=self.model.workflow_generation,
            previous_phase=previous.name,
            new_phase=phase.name,
            active_session_id=active_session_id,
            active_import_id=active_import_id,
            active_analysis_id=active_analysis_id,
            active_job_id=active_job_id,
        )
        if continuation_publication:
            self._publish_continuation(
                (
                    "workflow-state",
                    self.model.workflow_generation,
                    previous.name,
                    phase.name,
                ),
                self.bus.events.workflow_state_changed,
                message,
            )
        else:
            self._stage_import_terminal_publication(
                self.bus.events.workflow_state_changed,
                message,
                critical=True,
            )

    def _phase_active_identifiers(
        self, phase: WorkflowPhase
    ) -> tuple[str | None, str | None, str | None, str | None]:
        session_id = None
        import_id = None
        analysis_id = None
        job_id = None
        if phase in {
            WorkflowPhase.PREPARING,
            WorkflowPhase.RECORDING,
            WorkflowPhase.FINALIZING,
        }:
            session_id = self.model.active_session_id
        elif phase is WorkflowPhase.IMPORTING:
            import_id = self.model.active_import_id
        elif phase is WorkflowPhase.ANALYZING:
            analysis_id = self.model.active_analysis_id
        elif phase is WorkflowPhase.RESULT_EXPORTING:
            job_id = self.model.active_job_id
        elif phase is WorkflowPhase.CANCELLING:
            if self.model.cancelling_domain == "recording":
                session_id = self.model.active_session_id
            elif self.model.cancelling_domain == "import":
                import_id = self.model.active_import_id
            elif self.model.cancelling_domain == "analysis":
                analysis_id = self.model.active_analysis_id
            elif self.model.cancelling_domain == "export":
                job_id = self.model.active_job_id
        return session_id, import_id, analysis_id, job_id

    def _clear_active_domain(self) -> None:
        self._retire_active_recording_admission()
        self._pending_export_preparation = None
        self.model.active_session_id = None
        self.model.active_session_origin = None
        self.model.active_import_id = None
        self.model.active_analysis_id = None
        self.model.active_job_id = None
        self.model.active_attempt_id = None
        self.model.clear_export_attempt_history()
        self.model.analysis_source_id = None
        self.model.analysis_record_id = None
        self.model.export_record_id = None
        self.model.export_continuation = None
        self.model.post_analysis_continuation = None
        self.model.export_failure_pending = False
        self.model.cancelling_phase = None
        self.model.cancelling_domain = None
        self.model.active_label_command_id = None
        self.model.active_label_record_id = None
        self.model.active_label = None
        self.model.configuration_snapshot = None
        self.model.session_snapshot = None
        self.model.recording_snapshot = None
        self.model.import_reference_snapshot = None
        self.model.labeled_result_snapshot = None

    def _retire_active_recording_admission(self) -> bool:
        session_id = self.model.active_session_id
        capability = self._canonical_recording_admission_capability
        retire = getattr(
            self.bus,
            "_retire_canonical_recording_identity",
            None,
        )
        if (
            type(session_id) is not str
            or not session_id
            or capability is None
            or not callable(retire)
        ):
            return False
        return retire(
            capability,
            session_id,
            self.model.workflow_generation,
        )

    def _release_recording_admission_capability(self) -> bool:
        capability = self._canonical_recording_admission_capability
        if capability is None:
            return False
        self._canonical_recording_admission_capability = None
        release = getattr(
            self.bus,
            "_release_canonical_recording_workflow_owner",
            None,
        )
        if not callable(release):
            return False
        return bool(release(capability))

    def _register_recording_admission(
        self,
        admitted: BeginRecordingRequested,
        capability: object | None = None,
    ) -> bool:
        register = getattr(
            self.bus,
            "_register_canonical_recording_admission",
            None,
        )
        if not callable(register):
            return True
        if capability is None:
            capability = self._canonical_recording_admission_capability
        if capability is None:
            return False
        return bool(register(capability, admitted))

    def _recording_admission_cas(self) -> tuple[int, object] | None:
        with self._recording_admission_lock:
            capability = self._canonical_recording_admission_capability
            if self._recording_admission_closed or capability is None:
                return None
            return self._recording_admission_epoch, capability

    def _commit_registered_recording_admission(
        self,
        admitted: BeginRecordingRequested,
        configuration: ConfigurationSnapshot,
        origin: SessionOrigin,
        epoch: int,
        capability: object,
    ) -> WorkflowStateChanged | None:
        with self._recording_admission_lock:
            if (
                self._recording_admission_closed
                or self._recording_admission_epoch != epoch
                or self._canonical_recording_admission_capability is not capability
                or self.model.phase is not WorkflowPhase.IDLE
            ):
                return None
            previous = self.model.phase
            self.model.workflow_generation += 1
            self.model.automatic_analysis_decision = None
            self.model.active_session_id = admitted.session_id
            self.model.active_session_origin = origin
            self.model.configuration_snapshot = configuration
            self.model.session_snapshot = admitted.session_snapshot
            self.model.phase = WorkflowPhase.PREPARING
            self.model.assert_invariants()
            return WorkflowStateChanged(
                workflow_generation=self.model.workflow_generation,
                previous_phase=previous.name,
                new_phase=WorkflowPhase.PREPARING.name,
                active_session_id=admitted.session_id,
            )

    def _abandon_provisional_recording_admission(
        self,
        capability: object,
        admitted: BeginRecordingRequested,
    ) -> None:
        abandon = getattr(
            self.bus,
            "_abandon_canonical_recording_admission",
            None,
        )
        if callable(abandon):
            abandon(capability, admitted)

    def _publish_recording_admission_commit(
        self,
        message: WorkflowStateChanged,
        admitted: BeginRecordingRequested,
    ) -> None:
        self._stage_import_terminal_publication(
            self.bus.events.workflow_state_changed,
            message,
            critical=True,
        )
        self.bus.commands.begin_recording_requested.emit(admitted)

    def _begin_workflow(self) -> None:
        self._analysis_export_preparation_correlation = None
        self._clear_active_domain()
        self.model.workflow_generation += 1
        self.model.automatic_analysis_decision = None

    def _finish_idle(self, *, continuation_publication: bool = False) -> None:
        shutdown_was_confirmed = self.model.shutdown_cancellation_confirmed
        self._clear_active_domain()
        self.model.shutdown_asserted_active = False
        if self.model.shutdown_generation is not None and shutdown_was_confirmed:
            self._enter_shutdown_flushing(
                continuation_publication=continuation_publication
            )
        else:
            self.model.shutdown_cancellation_confirmed = False
            self._transition(
                WorkflowPhase.IDLE,
                continuation_publication=continuation_publication,
            )

    def _configuration_snapshot(self) -> ConfigurationSnapshot:
        snapshot = self.configuration_snapshot_provider()
        if type(snapshot) is not ConfigurationSnapshot:
            raise TypeError("configuration provider must return ConfigurationSnapshot")
        return snapshot

    def _retain_recording_analysis_inputs(
        self,
        *,
        record_id: str,
        source_id: str,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> bool:
        try:
            return self.retain_recording_snapshot(
                record_id,
                recording_snapshot,
                configuration_snapshot,
                source_id=source_id,
                workflow_generation=self.model.workflow_generation,
            ) is True
        except BaseException:
            self._finish_idle()
            raise

    def _automatic_analysis_decision(
        self,
        source: AutomaticAnalysisSource,
        *,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> AutomaticAnalysisDecision:
        decide = (
            self.automatic_analysis_policy.decide_recorded
            if source is AutomaticAnalysisSource.RECORDED
            else self.automatic_analysis_policy.decide_imported
        )
        try:
            decision = decide(
                workflow_generation=self.model.workflow_generation,
                recording_snapshot=recording_snapshot,
                configuration_snapshot=configuration_snapshot,
            )
        except BaseException:
            self._finish_idle()
            raise
        if (
            type(decision) is not AutomaticAnalysisDecision
            or decision.workflow_generation != self.model.workflow_generation
        ):
            self._finish_idle()
            raise TypeError(
                "automatic analysis policy must return the current exact decision"
            )
        self.model.automatic_analysis_decision = decision
        return decision

    def _session_snapshot_for_admission(
        self,
        command: StartTestRequested | ReplayRequested,
        configuration: ConfigurationSnapshot,
        workflow_generation: int,
    ) -> tuple[dict[Any, Any] | None, str]:
        raw_snapshot = self.session_snapshot_factory(command, configuration)
        if not isinstance(raw_snapshot, Mapping):
            return None, "session snapshot must be a data mapping"
        missing_generation = object()
        supplied_generation = raw_snapshot.get(
            "workflow_generation", missing_generation
        )
        if supplied_generation is not missing_generation and (
            type(supplied_generation) is not int
            or supplied_generation != workflow_generation
        ):
            return None, "session snapshot workflow generation mismatch"
        normalized_snapshot = dict(raw_snapshot.items())
        normalized_snapshot["workflow_generation"] = workflow_generation
        return normalized_snapshot, ""

    @pyqtSlot(object)
    def handle_start(self, command: StartTestRequested) -> bool:
        if self._recording_admission_closed:
            return False
        if self._shutdown_blocks_admission(command.command_id):
            return False
        if (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        ):
            return self._reject(command.command_id, "workflow is busy")
        if command.configuration_generation != self.model.configuration_generation:
            return self._reject(command.command_id, "stale configuration generation")
        configuration = self._configuration_snapshot()
        ready, reason = self._readiness(
            self.start_readiness(command, configuration), "recording is not ready"
        )
        if not ready:
            return self._reject(command.command_id, reason)
        session_id = self._allocate_domain_id("session", self.session_id_factory)
        next_workflow_generation = self.model.workflow_generation + 1
        session_snapshot, reason = self._session_snapshot_for_admission(
            command, configuration, next_workflow_generation
        )
        if session_snapshot is None:
            return self._reject(command.command_id, reason)
        admitted = BeginRecordingRequested(
            command_id=command.command_id,
            session_id=session_id,
            replay=False,
            session_snapshot=session_snapshot,
        )
        admission_cas = self._recording_admission_cas()
        if admission_cas is None:
            return False
        admission_epoch, admission_capability = admission_cas
        if not self._register_recording_admission(
            admitted,
            admission_capability,
        ):
            return self._reject(
                command.command_id,
                "recording admission handoff is busy",
            )
        transition = self._commit_registered_recording_admission(
            admitted,
            configuration,
            SessionOrigin.CANONICAL,
            admission_epoch,
            admission_capability,
        )
        if transition is None:
            self._abandon_provisional_recording_admission(
                admission_capability,
                admitted,
            )
            return False
        self._publish_recording_admission_commit(transition, admitted)
        return True

    @pyqtSlot(object)
    def handle_replay(self, command: ReplayRequested) -> bool:
        if self._recording_admission_closed:
            return False
        if self._shutdown_blocks_admission(command.command_id):
            return False
        if (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        ):
            return self._reject(command.command_id, "workflow is busy")
        configuration = self._configuration_snapshot()
        ready, reason = self._readiness(
            self.replay_readiness(command, configuration), "replay is not ready"
        )
        if not ready:
            return self._reject(command.command_id, reason)
        session_id = self._allocate_domain_id("session", self.session_id_factory)
        next_workflow_generation = self.model.workflow_generation + 1
        session_snapshot, reason = self._session_snapshot_for_admission(
            command, configuration, next_workflow_generation
        )
        if session_snapshot is None:
            return self._reject(command.command_id, reason)
        admitted = BeginRecordingRequested(
            command_id=command.command_id,
            session_id=session_id,
            replay=True,
            session_snapshot=session_snapshot,
        )
        admission_cas = self._recording_admission_cas()
        if admission_cas is None:
            return False
        admission_epoch, admission_capability = admission_cas
        if not self._register_recording_admission(
            admitted,
            admission_capability,
        ):
            return self._reject(
                command.command_id,
                "recording admission handoff is busy",
            )
        transition = self._commit_registered_recording_admission(
            admitted,
            configuration,
            SessionOrigin.CANONICAL,
            admission_epoch,
            admission_capability,
        )
        if transition is None:
            self._abandon_provisional_recording_admission(
                admission_capability,
                admitted,
            )
            return False
        self._publish_recording_admission_commit(transition, admitted)
        return True

    @pyqtSlot(object)
    def handle_import(self, command: ImportAudioRequested) -> bool:
        if self._shutdown_blocks_admission(command.command_id):
            return False
        if (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        ):
            return self._reject(command.command_id, "workflow is busy")
        configuration = self._configuration_snapshot()
        ready, reason = self._readiness(
            self.import_readiness(command, configuration), "import mode is not supported"
        )
        if not ready:
            return self._reject(command.command_id, reason)
        import_id = self._allocate_domain_id("import", self.import_id_factory)
        admitted = LoadImportedAudioRequested(
            command_id=command.command_id,
            import_id=import_id,
            mode=command.mode,
            selected_path=command.selected_path,
            configuration_snapshot=configuration,
            workflow_generation=self.model.workflow_generation + 1,
        )
        self._begin_workflow()
        self.model.active_import_id = import_id
        self.model.configuration_snapshot = admitted.configuration_snapshot
        self._transition(WorkflowPhase.IMPORTING)
        self.bus.commands.load_imported_audio_requested.emit(admitted)
        return True

    def _analysis_inputs(self, value: Any) -> tuple[Any, ConfigurationSnapshot]:
        if type(value) is tuple and len(value) == 2:
            recording, configuration = value
        else:
            recording = value
            configuration = self.model.configuration_snapshot
        if recording is None:
            raise ValueError("recording snapshot is unavailable")
        if type(configuration) is not ConfigurationSnapshot:
            configuration = self._configuration_snapshot()
        return recording, configuration

    def _admit_analysis(
        self,
        *,
        source_id: str,
        record_id: str,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
        automatic: bool,
        begin_workflow: bool,
    ) -> None:
        analysis_id = self._allocate_domain_id("analysis", self.analysis_id_factory)
        admitted = AnalysisRequested(
            analysis_id=analysis_id,
            source_id=source_id,
            recording_snapshot=recording_snapshot,
            configuration_snapshot=configuration_snapshot,
            automatic=automatic,
            workflow_generation=(
                self.model.workflow_generation + (1 if begin_workflow else 0)
            ),
        )
        if begin_workflow:
            self._begin_workflow()
        self.model.active_session_id = None
        self.model.active_session_origin = None
        self.model.active_import_id = None
        self.model.active_job_id = None
        self.model.active_attempt_id = None
        self.model.clear_export_attempt_history()
        self.model.active_analysis_id = analysis_id
        self.model.analysis_source_id = source_id
        self.model.analysis_record_id = record_id
        self.model.recording_snapshot = admitted.recording_snapshot
        self.model.configuration_snapshot = admitted.configuration_snapshot
        self._transition(WorkflowPhase.ANALYZING)
        self._stage_import_terminal_publication(
            self.bus.commands.analysis_requested,
            admitted,
            critical=True,
        )

    @pyqtSlot(object)
    def handle_manual_analysis(self, command: ManualAnalysisRequested) -> bool:
        if self._shutdown_blocks_admission(command.command_id):
            return False
        if (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        ):
            return self._reject(command.command_id, "workflow is busy")
        if command.record_id != self.model.retained_record_id:
            return self._reject(command.command_id, "record is not retained for analysis")
        lookup = self.recording_snapshot_lookup(command.record_id)
        if lookup is None:
            return self._reject(command.command_id, "record is not analysis-ready")
        recording, configuration = self._analysis_inputs(lookup)
        self._admit_analysis(
            source_id=command.record_id,
            record_id=command.record_id,
            recording_snapshot=recording,
            configuration_snapshot=configuration,
            automatic=False,
            begin_workflow=True,
        )
        return True

    @pyqtSlot(object)
    def handle_recording_started(self, event: RecordingStarted) -> bool:
        if (
            self.model.phase is not WorkflowPhase.PREPARING
            or event.session_id != self.model.active_session_id
        ):
            return self._diagnose_stale(
                "recording",
                "recording_started",
                "recording start does not match the active session",
                expected_session_id=self.model.active_session_id,
                received_session_id=event.session_id,
                expected_phase=WorkflowPhase.PREPARING.name,
            )
        if not isinstance(event.session_snapshot, Mapping):
            return self._diagnose_stale(
                "recording",
                "recording_started",
                "recording start is missing its workflow generation",
                expected_session_id=self.model.active_session_id,
                received_session_id=event.session_id,
                expected_generation=self.model.workflow_generation,
                received_generation=None,
            )
        event_generation = event.session_snapshot.get("workflow_generation")
        if (
            type(event_generation) is not int
            or event_generation != self.model.workflow_generation
        ):
            return self._diagnose_stale(
                "recording",
                "recording_started",
                "recording start has a stale workflow generation",
                expected_session_id=self.model.active_session_id,
                received_session_id=event.session_id,
                expected_generation=self.model.workflow_generation,
                received_generation=event_generation,
            )
        self._retire_active_recording_admission()
        self._transition(WorkflowPhase.RECORDING)
        return True

    def handle_recording_finalizing(self, session_id: str) -> bool:
        if (
            self.model.phase is not WorkflowPhase.RECORDING
            or session_id != self.model.active_session_id
        ):
            return self._diagnose_stale(
                "recording",
                "recording_finalizing",
                "recording finalization does not match the active session",
                expected_session_id=self.model.active_session_id,
                received_session_id=session_id,
                expected_phase=WorkflowPhase.RECORDING.name,
            )
        self._transition(WorkflowPhase.FINALIZING)
        return True

    @pyqtSlot(object)
    def handle_recording_completed(self, event: RecordingCompleted) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "recording"
                or event.session_id != self.model.active_session_id
            ):
                return self._diagnose_stale(
                    "recording",
                    "recording_completed",
                    "recording completion does not match the cancelling session",
                    expected_session_id=self.model.active_session_id,
                    received_session_id=event.session_id,
                    expected_cancelling_domain="recording",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if (
            self.model.phase not in {WorkflowPhase.RECORDING, WorkflowPhase.FINALIZING}
            or event.session_id != self.model.active_session_id
        ):
            return self._diagnose_stale(
                "recording",
                "recording_completed",
                "recording completion does not match the active session",
                expected_session_id=self.model.active_session_id,
                received_session_id=event.session_id,
                expected_phases=(
                    WorkflowPhase.RECORDING.name,
                    WorkflowPhase.FINALIZING.name,
                ),
            )
        if self.model.phase is WorkflowPhase.RECORDING:
            self._transition(WorkflowPhase.FINALIZING)
        record_id = self.record_id_lookup(event.result_snapshot) or event.session_id
        configuration = self.model.configuration_snapshot
        if type(configuration) is not ConfigurationSnapshot:
            configuration = self._configuration_snapshot()
        if not self._retain_recording_analysis_inputs(
            record_id=record_id,
            source_id=event.session_id,
            recording_snapshot=event.result_snapshot,
            configuration_snapshot=configuration,
        ):
            self._finish_idle()
            return False
        self.model.recording_snapshot = event.result_snapshot
        self.model.retained_record_id = record_id
        self.model.awaiting_label = True
        decision = self._automatic_analysis_decision(
            AutomaticAnalysisSource.RECORDED,
            recording_snapshot=event.result_snapshot,
            configuration_snapshot=configuration,
        )
        if decision.enabled:
            self._admit_analysis(
                source_id=event.session_id,
                record_id=record_id,
                recording_snapshot=event.result_snapshot,
                configuration_snapshot=configuration,
                automatic=True,
                begin_workflow=False,
            )
        else:
            self._finish_idle()
        return True

    def _handle_recording_unsuccessful(
        self, event: Any, event_kind: str
    ) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "recording"
                or event.session_id != self.model.active_session_id
            ):
                return self._diagnose_stale(
                    "recording",
                    event_kind,
                    "recording terminal does not match the cancelling session",
                    expected_session_id=self.model.active_session_id,
                    received_session_id=event.session_id,
                    expected_cancelling_domain="recording",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if (
            self.model.phase
            not in {
                WorkflowPhase.PREPARING,
                WorkflowPhase.RECORDING,
                WorkflowPhase.FINALIZING,
            }
            or event.session_id != self.model.active_session_id
        ):
            return self._diagnose_stale(
                "recording",
                event_kind,
                "recording terminal does not match the active session",
                expected_session_id=self.model.active_session_id,
                received_session_id=event.session_id,
                expected_phases=(
                    WorkflowPhase.PREPARING.name,
                    WorkflowPhase.RECORDING.name,
                    WorkflowPhase.FINALIZING.name,
                ),
            )
        self._finish_idle()
        return True

    @pyqtSlot(object)
    def handle_recording_failed(self, event: RecordingFailed) -> bool:
        return self._handle_recording_unsuccessful(event, "recording_failed")

    @pyqtSlot(object)
    def handle_recording_cancelled(self, event: RecordingCancelled) -> bool:
        return self._handle_recording_unsuccessful(event, "recording_cancelled")

    @pyqtSlot(object)
    def handle_imported_audio_ready(self, event: ImportedAudioReady) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "import"
                or event.import_id != self.model.active_import_id
            ):
                return self._diagnose_stale(
                    "import",
                    "imported_audio_ready",
                    "import result does not match the cancelling import",
                    expected_import_id=self.model.active_import_id,
                    received_import_id=event.import_id,
                    expected_cancelling_domain="import",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if (
            self.model.phase is not WorkflowPhase.IMPORTING
            or event.import_id != self.model.active_import_id
        ):
            return self._diagnose_stale(
                "import",
                "imported_audio_ready",
                "import result does not match the active import",
                expected_import_id=self.model.active_import_id,
                received_import_id=event.import_id,
                expected_phase=WorkflowPhase.IMPORTING.name,
            )
        ready, reason = self._readiness(
            self.imported_audio_readiness(event),
            "imported recording snapshot is incomplete",
        )
        if not ready:
            notification = ImportedAudioFailed(event.import_id, reason)
            formal_delivery = self._resolve_formal_import_terminal()
            if self._accept_queued_delivery and not formal_delivery:
                self._pending_local_import_failure_notifications[
                    id(notification)
                ] = notification
            self._stage_import_terminal_publication(
                self.bus.events.imported_audio_failed,
                notification,
                critical=False,
            )
            self._finish_idle()
            return False
        configuration = self.model.configuration_snapshot
        if type(configuration) is not ConfigurationSnapshot:
            configuration = self._configuration_snapshot()
        self.model.import_reference_snapshot = event.reference_snapshot
        record_id = self.record_id_lookup(event.recording_snapshot) or event.import_id
        if not self._retain_recording_analysis_inputs(
            record_id=record_id,
            source_id=event.import_id,
            recording_snapshot=event.recording_snapshot,
            configuration_snapshot=configuration,
        ):
            self._finish_idle()
            return False
        # Imported data remains analysis-ready after the automatic pass, so the
        # Data action can request another Workflow-admitted analysis.
        self.model.retained_record_id = record_id
        self.model.awaiting_label = False
        decision = self._automatic_analysis_decision(
            AutomaticAnalysisSource.IMPORTED,
            recording_snapshot=event.recording_snapshot,
            configuration_snapshot=configuration,
        )
        if not decision.enabled:
            self._finish_idle()
            return False
        self._admit_analysis(
            source_id=event.import_id,
            record_id=record_id,
            recording_snapshot=event.recording_snapshot,
            configuration_snapshot=configuration,
            automatic=True,
            begin_workflow=False,
        )
        return True

    @pyqtSlot(object)
    def handle_imported_audio_failed(self, event: ImportedAudioFailed) -> bool:
        local_notification = self._pending_local_import_failure_notifications.pop(
            id(event),
            None,
        )
        if local_notification is event:
            return True
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "import"
                or event.import_id != self.model.active_import_id
            ):
                return self._diagnose_stale(
                    "import",
                    "imported_audio_failed",
                    "import failure does not match the cancelling import",
                    expected_import_id=self.model.active_import_id,
                    received_import_id=event.import_id,
                    expected_cancelling_domain="import",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if (
            self.model.phase is not WorkflowPhase.IMPORTING
            or event.import_id != self.model.active_import_id
        ):
            return self._diagnose_stale(
                "import",
                "imported_audio_failed",
                "import failure does not match the active import",
                expected_import_id=self.model.active_import_id,
                received_import_id=event.import_id,
                expected_phase=WorkflowPhase.IMPORTING.name,
            )
        self._finish_idle()
        return True

    def _start_export(
        self,
        *,
        record_id: str,
        result_snapshot: Any,
        target_configurations: tuple[Any, ...],
        continuation: ExportContinuation,
    ) -> None:
        job_id = self._allocate_domain_id("job", self.job_id_factory)
        admitted = ExportRequested(
            job_id=job_id,
            record_id=record_id,
            result_snapshot=result_snapshot,
            target_configurations=target_configurations,
        )
        self.model.active_analysis_id = None
        self.model.active_job_id = job_id
        self.model.active_attempt_id = None
        self.model.clear_export_attempt_history()
        self.model.export_record_id = record_id
        self.model.export_continuation = continuation
        self.model.export_failure_pending = False
        self._transition(WorkflowPhase.RESULT_EXPORTING)
        self.bus.commands.export_requested.emit(admitted)

    @staticmethod
    def _automatic_label_for_result(
        result_snapshot: Any,
        *,
        awaiting_label: bool,
        retained_record_id: str | None,
        record_id: str,
    ) -> str | None:
        if (
            not awaiting_label
            or retained_record_id != record_id
            or not isinstance(result_snapshot, Mapping)
            or result_snapshot.get("test_mode") is not True
            or result_snapshot.get("can_output_ok_ng") is not True
        ):
            return None
        summary = result_snapshot.get("ok_ng_summary")
        if not isinstance(summary, (tuple, list)) or len(summary) != 2:
            return None
        label = summary[1]
        return label if label in {"OK", "NG"} else None

    def _continue_post_analysis(self, outcome: Any) -> bool:
        continuation = self.model.post_analysis_continuation
        if continuation is None:
            self._finish_idle()
            return False
        if continuation.automatic_label is None:
            return self._finish_post_analysis_transport()
        command_id = self.label_id_factory()
        if type(command_id) is not str or not command_id:
            raise ValueError("label identifier factory must return a non-empty string")
        self.model.active_analysis_id = None
        self.model.active_label_command_id = command_id
        self.model.active_label_record_id = continuation.record_id
        self.model.active_label = continuation.automatic_label
        self.model.labeled_result_snapshot = continuation.result_snapshot
        commit = CommitRecordingLabelRequested(
            command_id=command_id,
            record_id=continuation.record_id,
            label=continuation.automatic_label,
            export_outcome={
                "continuation": "post-analysis",
                "analysis_id": continuation.analysis_id,
                "source_id": continuation.source_id,
                "workflow_generation": continuation.workflow_generation,
                "export_outcome": outcome,
            },
        )
        self._transition(
            WorkflowPhase.LABEL_COMMITTING,
            continuation_publication=True,
        )
        self._publish_continuation(
            (
                "label-commit",
                commit.command_id,
                self.model.workflow_generation,
            ),
            self.bus.commands.commit_recording_label_requested,
            commit,
        )
        return True

    def _finish_post_analysis_transport(self) -> bool:
        continuation = self.model.post_analysis_continuation
        if continuation is None:
            self._finish_idle()
            return False
        transport = AnalysisTransportReady(
            analysis_id=continuation.analysis_id,
            source_id=continuation.source_id,
            record_id=continuation.record_id,
            workflow_generation=continuation.workflow_generation,
            payload=(
                continuation.result_snapshot.get("tcp_result_payload")
                if isinstance(continuation.result_snapshot, Mapping)
                else None
            ),
        )
        authorized = self.model.authorize_analysis_transport(transport)
        self._finish_idle(continuation_publication=True)
        self._publish_continuation(
            (
                "analysis-transport",
                transport.analysis_id,
                transport.source_id,
                transport.record_id,
                transport.workflow_generation,
            ),
            self.bus.events.analysis_transport_ready,
            transport,
            requires_analysis_transport_authorization=not authorized,
        )
        return True

    @pyqtSlot(object)
    def handle_analysis_completed(self, event: AnalysisCompleted) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "analysis"
                or event.analysis_id != self.model.active_analysis_id
                or event.source_id != self.model.analysis_source_id
            ):
                return self._diagnose_stale(
                    "analysis",
                    "analysis_completed",
                    "analysis completion does not match the cancelling analysis",
                    expected_analysis_id=self.model.active_analysis_id,
                    received_analysis_id=event.analysis_id,
                    expected_source_id=self.model.analysis_source_id,
                    received_source_id=event.source_id,
                    expected_cancelling_domain="analysis",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if (
            self.model.phase is not WorkflowPhase.ANALYZING
            or event.analysis_id != self.model.active_analysis_id
            or event.source_id != self.model.analysis_source_id
        ):
            return self._diagnose_stale(
                "analysis",
                "analysis_completed",
                "analysis completion does not match the active analysis",
                expected_analysis_id=self.model.active_analysis_id,
                received_analysis_id=event.analysis_id,
                expected_source_id=self.model.analysis_source_id,
                received_source_id=event.source_id,
                expected_phase=WorkflowPhase.ANALYZING.name,
            )
        self.model.analysis_result_snapshot = event.result_snapshot
        record_id = (
            self.record_id_lookup(event.result_snapshot)
            or self.model.analysis_record_id
            or event.source_id
        )
        self.model.post_analysis_continuation = PostAnalysisContinuation(
            analysis_id=event.analysis_id,
            source_id=event.source_id,
            record_id=record_id,
            workflow_generation=self.model.workflow_generation,
            result_snapshot=event.result_snapshot,
            automatic_label=self._automatic_label_for_result(
                event.result_snapshot,
                awaiting_label=self.model.awaiting_label,
                retained_record_id=self.model.retained_record_id,
                record_id=record_id,
            ),
        )
        request = PrepareAnalysisExportRequested(
            request_id=self._allocate_domain_id(
                "preparation", self.preparation_id_factory
            ),
            analysis_id=event.analysis_id,
            source_id=event.source_id,
            record_id=record_id,
            workflow_generation=self.model.workflow_generation,
            result_snapshot=event.result_snapshot,
        )
        self._analysis_export_preparation_correlation = (
            _AnalysisExportPreparationCorrelation(request)
        )
        self._pending_export_preparation = request
        self._publish_continuation(
            (
                "analysis-export-prepare",
                request.request_id,
                request.workflow_generation,
            ),
            self.bus.commands.prepare_analysis_export_requested,
            request,
            permanent_rejection_callback=(
                self._settle_rejected_analysis_export_preparation
            ),
        )
        return True

    def _settle_rejected_analysis_export_preparation(
        self,
        publication: _PendingContinuationPublication,
        outcome: WorkflowContinuationDeliveryOutcome,
    ) -> bool:
        request = publication.message
        correlation = self._analysis_export_preparation_correlation
        if (
            publication.kind != "analysis-export-prepare"
            or type(request) is not PrepareAnalysisExportRequested
            or correlation is None
            or correlation.request is not request
            or request.workflow_generation != self.model.workflow_generation
            or publication.delivery_id
            != (
                "analysis-export-prepare",
                request.request_id,
                request.workflow_generation,
            )
        ):
            return False
        if correlation.settlement is None and (
            self._pending_export_preparation is not request
            or self.model.phase is not WorkflowPhase.ANALYZING
        ):
            return False
        pending = self._pending_continuation_publications
        retained = pending.get(publication.delivery_id)
        if retained is not None and retained is not publication:
            return False
        if retained is publication:
            pending.pop(publication.delivery_id, None)
        if not pending:
            self._continuation_retry_timer.stop()
            self._continuation_retry_attempt = 0
            self._continuation_retry_delay_ms = 0
        if correlation.settlement is not None:
            return True
        bounded_delivery_id = tuple(
            value[:256]
            if type(value) is str
            else value
            if type(value) is int and value.bit_length() <= 63
            else "<bounded>"
            for value in publication.delivery_id
        )
        reason = (
            outcome.reason[:256]
            if type(outcome.reason) is str
            else "continuation was permanently rejected"
        )
        correlation.settlement = "outer-permanent-reject"
        self._pending_export_preparation = None
        try:
            self.diagnostic_callback(
                {
                    "kind": publication.kind,
                    "delivery_id": bounded_delivery_id,
                    "reason": reason,
                }
            )
        except BaseException:
            # Diagnostics are an external best-effort boundary. Terminal
            # recovery must continue after the exact publication is settled.
            logging.getLogger(__name__).debug(
                "Analysis export preparation rejection diagnostic callback failed",
                exc_info=True,
            )
        self._finish_idle()
        return True

    def _matches_analysis_export_preparation(self, event: Any) -> bool:
        request = self._pending_export_preparation
        correlation = self._analysis_export_preparation_correlation
        return (
            type(request) is PrepareAnalysisExportRequested
            and correlation is not None
            and correlation.request is request
            and correlation.settlement is None
            and self.model.phase is WorkflowPhase.ANALYZING
            and correlation.matches_response(event)
            and event.workflow_generation == self.model.workflow_generation
        )

    def _matches_settled_analysis_export_preparation(self, event: Any) -> bool:
        correlation = self._analysis_export_preparation_correlation
        return (
            correlation is not None
            and correlation.settlement is not None
            and correlation.matches_response(event)
            and event.workflow_generation == self.model.workflow_generation
        )

    @pyqtSlot(object)
    def handle_analysis_export_prepared(
        self, event: AnalysisExportPrepared
    ) -> bool:
        if (
            type(event) is not AnalysisExportPrepared
            or not self._matches_analysis_export_preparation(event)
        ):
            return self._diagnose_stale(
                "export-preparation",
                "analysis_export_prepared",
                "prepared analysis export does not match the pending request",
                received_request_id=getattr(event, "request_id", None),
            )
        correlation = self._analysis_export_preparation_correlation
        self._pending_export_preparation = None
        continuation = self.model.post_analysis_continuation
        if continuation is None:
            return False
        self.model.analysis_result_snapshot = event.result_snapshot
        self.model.post_analysis_continuation = replace(
            continuation, result_snapshot=event.result_snapshot
        )
        targets = tuple(event.target_configurations)
        if targets:
            self._start_export(
                record_id=event.record_id,
                result_snapshot=event.result_snapshot,
                target_configurations=targets,
                continuation=ExportContinuation.ANALYSIS_DONE,
            )
        else:
            self._continue_post_analysis(())
        if self._analysis_export_preparation_correlation is correlation:
            correlation.settlement = "prepared"
        return True

    @pyqtSlot(object)
    def handle_analysis_export_preparation_failed(
        self, event: AnalysisExportPreparationFailed
    ) -> bool:
        if (
            type(event) is AnalysisExportPreparationFailed
            and self._matches_settled_analysis_export_preparation(event)
        ):
            return True
        if (
            type(event) is not AnalysisExportPreparationFailed
            or not self._matches_analysis_export_preparation(event)
        ):
            return self._diagnose_stale(
                "export-preparation",
                "analysis_export_preparation_failed",
                "failed analysis export preparation does not match the pending request",
                received_request_id=getattr(event, "request_id", None),
            )
        correlation = self._analysis_export_preparation_correlation
        self._pending_export_preparation = None
        self._finish_idle()
        if self._analysis_export_preparation_correlation is correlation:
            correlation.settlement = "failed"
        return True

    @pyqtSlot(object)
    def handle_analysis_failed(self, event: AnalysisFailed) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if (
                self.model.cancelling_domain != "analysis"
                or event.analysis_id != self.model.active_analysis_id
                or event.source_id != self.model.analysis_source_id
            ):
                return self._diagnose_stale(
                    "analysis",
                    "analysis_failed",
                    "analysis failure does not match the cancelling analysis",
                    expected_analysis_id=self.model.active_analysis_id,
                    received_analysis_id=event.analysis_id,
                    expected_source_id=self.model.analysis_source_id,
                    received_source_id=event.source_id,
                    expected_cancelling_domain="analysis",
                    received_cancelling_domain=self.model.cancelling_domain,
                )
            return self._resolve_cancelled_domain()
        if self._pending_export_preparation is not None:
            return self._diagnose_stale(
                "analysis",
                "analysis_failed",
                "analysis already entered export preparation",
                received_analysis_id=event.analysis_id,
            )
        if (
            self.model.phase is not WorkflowPhase.ANALYZING
            or event.analysis_id != self.model.active_analysis_id
            or event.source_id != self.model.analysis_source_id
        ):
            return self._diagnose_stale(
                "analysis",
                "analysis_failed",
                "analysis failure does not match the active analysis",
                expected_analysis_id=self.model.active_analysis_id,
                received_analysis_id=event.analysis_id,
                expected_source_id=self.model.analysis_source_id,
                received_source_id=event.source_id,
                expected_phase=WorkflowPhase.ANALYZING.name,
            )
        self._finish_idle()
        return True

    @pyqtSlot(object)
    def handle_manual_label(self, command: ManualLabelRequested) -> bool:
        if self._shutdown_blocks_admission(command.command_id):
            return False
        if (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        ):
            return self._reject(command.command_id, "workflow is busy")
        if command.label not in {"OK", "NG"}:
            return self._reject(command.command_id, "label must be OK or NG")
        if (
            not self.model.awaiting_label
            or command.record_id != self.model.retained_record_id
        ):
            return self._reject(command.command_id, "record is not awaiting a label")
        self._begin_workflow()
        self.model.active_label_command_id = command.command_id
        self.model.active_label_record_id = command.record_id
        self.model.active_label = command.label
        request = PrepareManualLabelExportRequested(
            request_id=self._allocate_domain_id(
                "preparation", self.preparation_id_factory
            ),
            command_id=command.command_id,
            record_id=command.record_id,
            label=command.label,
            workflow_generation=self.model.workflow_generation,
        )
        self._pending_export_preparation = request
        self._publish_continuation(
            (
                "manual-label-export-prepare",
                request.request_id,
                request.workflow_generation,
            ),
            self.bus.commands.prepare_manual_label_export_requested,
            request,
        )
        return True

    def _matches_manual_export_preparation(self, event: Any) -> bool:
        request = self._pending_export_preparation
        return (
            type(request) is PrepareManualLabelExportRequested
            and self.model.phase is WorkflowPhase.IDLE
            and event.request_id == request.request_id
            and event.command_id == request.command_id
            and event.record_id == request.record_id
            and event.label == request.label
            and event.workflow_generation == request.workflow_generation
            and event.workflow_generation == self.model.workflow_generation
        )

    @pyqtSlot(object)
    def handle_manual_label_export_prepared(
        self, event: ManualLabelExportPrepared
    ) -> bool:
        if (
            type(event) is not ManualLabelExportPrepared
            or not self._matches_manual_export_preparation(event)
        ):
            return self._diagnose_stale(
                "export-preparation",
                "manual_label_export_prepared",
                "prepared manual export does not match the pending request",
                received_request_id=getattr(event, "request_id", None),
            )
        self._pending_export_preparation = None
        self.model.labeled_result_snapshot = event.result_snapshot
        targets = tuple(event.target_configurations)
        if targets:
            self._start_export(
                record_id=event.record_id,
                result_snapshot=event.result_snapshot,
                target_configurations=targets,
                continuation=ExportContinuation.LABEL_COMMIT,
            )
            return True
        commit = CommitRecordingLabelRequested(
            command_id=event.command_id,
            record_id=event.record_id,
            label=event.label,
            export_outcome=(),
        )
        self._transition(
            WorkflowPhase.LABEL_COMMITTING,
            continuation_publication=True,
        )
        self._publish_continuation(
            (
                "label-commit",
                commit.command_id,
                self.model.workflow_generation,
            ),
            self.bus.commands.commit_recording_label_requested,
            commit,
        )
        return True

    @pyqtSlot(object)
    def handle_manual_label_export_preparation_failed(
        self, event: ManualLabelExportPreparationFailed
    ) -> bool:
        if (
            type(event) is not ManualLabelExportPreparationFailed
            or not self._matches_manual_export_preparation(event)
        ):
            return self._diagnose_stale(
                "export-preparation",
                "manual_label_export_preparation_failed",
                "failed manual export preparation does not match the pending request",
                received_request_id=getattr(event, "request_id", None),
            )
        self._pending_export_preparation = None
        self._finish_idle()
        return True

    @pyqtSlot(object)
    def handle_export_preparation_cancelled(
        self, event: ExportPreparationCancelled
    ) -> bool:
        request = self._pending_export_preparation
        if (
            type(event) is not ExportPreparationCancelled
            or request is None
            or self.model.phase is not WorkflowPhase.CANCELLING
            or self.model.cancelling_domain != "preparation"
            or event.request_id != request.request_id
            or event.workflow_generation != request.workflow_generation
            or event.workflow_generation != self.model.workflow_generation
        ):
            return self._diagnose_stale(
                "export-preparation",
                "export_preparation_cancelled",
                "preparation cancellation does not match the pending request",
                received_request_id=getattr(event, "request_id", None),
            )
        self._finish_idle()
        return True

    def _valid_export_event(
        self, event: Any, event_kind: str, *, cancelling: bool = False
    ) -> bool:
        if cancelling:
            valid = (
                self.model.cancelling_domain == "export"
                and event.job_id == self.model.active_job_id
                and event.record_id == self.model.export_record_id
            )
            reason = "export terminal does not match the cancelling export"
        else:
            valid = (
                self.model.phase is WorkflowPhase.RESULT_EXPORTING
                and event.job_id == self.model.active_job_id
                and event.record_id == self.model.export_record_id
            )
            reason = "export terminal does not match the active export"
        if valid:
            return True
        return self._diagnose_stale(
            "export",
            event_kind,
            reason,
            expected_job_id=self.model.active_job_id,
            received_job_id=event.job_id,
            expected_record_id=self.model.export_record_id,
            received_record_id=event.record_id,
            expected_cancelling_domain="export" if cancelling else None,
            received_cancelling_domain=(
                self.model.cancelling_domain if cancelling else None
            ),
            expected_phase=(
                None if cancelling else WorkflowPhase.RESULT_EXPORTING.name
            ),
        )

    def _accept_export_attempt(
        self,
        attempt_id: str,
        event_kind: str,
        *,
        install: bool = True,
    ) -> bool:
        if attempt_id in self.model.retired_attempt_ids:
            return self._diagnose_stale(
                "export",
                event_kind,
                "retired export attempt",
                expected_job_id=self.model.active_job_id,
                expected_attempt_id=self.model.active_attempt_id,
                received_attempt_id=attempt_id,
            )
        if self.model.active_attempt_id is None:
            if install:
                self.model.active_attempt_id = attempt_id
            return True
        if attempt_id == self.model.active_attempt_id:
            return True
        return self._diagnose_stale(
            "export",
            event_kind,
            "export terminal has a mismatched attempt",
            expected_job_id=self.model.active_job_id,
            expected_attempt_id=self.model.active_attempt_id,
            received_attempt_id=attempt_id,
        )

    def _valid_export_decision(self, command: Any, event_kind: str) -> bool:
        if self.model.phase is not WorkflowPhase.RESULT_EXPORTING:
            reason = "export decision arrived outside result export"
        elif not self.model.export_failure_pending:
            reason = "export decision arrived without a pending failure"
        elif command.job_id != self.model.active_job_id:
            reason = "export decision has a mismatched job"
        elif command.attempt_id != self.model.active_attempt_id:
            reason = "export decision has a mismatched attempt"
        else:
            return True
        return self._diagnose_stale(
            "export",
            event_kind,
            reason,
            expected_job_id=self.model.active_job_id,
            received_job_id=command.job_id,
            expected_attempt_id=self.model.active_attempt_id,
            received_attempt_id=command.attempt_id,
            expected_phase=WorkflowPhase.RESULT_EXPORTING.name,
        )

    def _stage_export_continuation(
        self, outcome: Any
    ) -> _StagedExportContinuation:
        continuation_kind = self.model.export_continuation
        if continuation_kind is ExportContinuation.LABEL_COMMIT:
            command = CommitRecordingLabelRequested(
                command_id=self.model.active_label_command_id,
                record_id=self.model.active_label_record_id,
                label=self.model.active_label,
                export_outcome=outcome,
            )
            return _StagedExportContinuation(
                "label", outcome, command=command
            )
        if continuation_kind is not ExportContinuation.ANALYSIS_DONE:
            raise ValueError("export continuation is unavailable")
        continuation = self.model.post_analysis_continuation
        if continuation is None:
            raise ValueError("post-analysis continuation is unavailable")
        if continuation.automatic_label is not None:
            command_id = self.label_id_factory()
            if type(command_id) is not str or not command_id:
                raise ValueError(
                    "label identifier factory must return a non-empty string"
                )
            command = CommitRecordingLabelRequested(
                command_id=command_id,
                record_id=continuation.record_id,
                label=continuation.automatic_label,
                export_outcome={
                    "continuation": "post-analysis",
                    "analysis_id": continuation.analysis_id,
                    "source_id": continuation.source_id,
                    "workflow_generation": continuation.workflow_generation,
                    "export_outcome": outcome,
                },
            )
            return _StagedExportContinuation(
                "label",
                outcome,
                command=command,
                labeled_result_snapshot=continuation.result_snapshot,
            )
        payload = (
            continuation.result_snapshot.get("tcp_result_payload")
            if isinstance(continuation.result_snapshot, Mapping)
            else None
        )
        transport = AnalysisTransportReady(
            analysis_id=continuation.analysis_id,
            source_id=continuation.source_id,
            record_id=continuation.record_id,
            workflow_generation=continuation.workflow_generation,
            payload=payload,
        )
        return _StagedExportContinuation(
            "transport", outcome, transport=transport
        )

    def _clear_completed_export(self, outcome: Any) -> None:
        self.model.export_outcome = outcome
        self.model.active_job_id = None
        self.model.active_attempt_id = None
        self.model.clear_export_attempt_history()
        self.model.export_record_id = None
        self.model.export_failure_pending = False
        self.model.export_continuation = None

    def _commit_export_continuation(
        self, staged: _StagedExportContinuation
    ) -> bool:
        if staged.kind == "transport":
            transport = staged.transport
            if transport is None:
                return False
            authorized = self.model.authorize_analysis_transport(transport)
            self._clear_completed_export(staged.outcome)
            self._finish_idle(continuation_publication=True)
            self._publish_continuation(
                (
                    "analysis-transport",
                    transport.analysis_id,
                    transport.source_id,
                    transport.record_id,
                    transport.workflow_generation,
                ),
                self.bus.events.analysis_transport_ready,
                transport,
                requires_analysis_transport_authorization=not authorized,
            )
            return True
        command = staged.command
        if staged.kind != "label" or command is None:
            return False
        self._clear_completed_export(staged.outcome)
        self.model.cancelling_phase = None
        self.model.cancelling_domain = None
        if staged.labeled_result_snapshot is not None:
            self.model.active_analysis_id = None
            self.model.active_label_command_id = command.command_id
            self.model.active_label_record_id = command.record_id
            self.model.active_label = command.label
            self.model.labeled_result_snapshot = staged.labeled_result_snapshot
        self._transition(
            WorkflowPhase.LABEL_COMMITTING,
            continuation_publication=True,
        )
        self._publish_continuation(
            (
                "label-commit",
                command.command_id,
                self.model.workflow_generation,
            ),
            self.bus.commands.commit_recording_label_requested,
            command,
        )
        return True

    def _continue_after_export(self, outcome: Any) -> bool:
        staged = self._stage_export_continuation(outcome)
        return self._commit_export_continuation(staged)

    @pyqtSlot(object)
    def handle_export_completed(self, event: ExportCompleted) -> bool:
        try:
            if self.model.phase is WorkflowPhase.CANCELLING:
                if not self._valid_export_event(
                    event, "export_completed", cancelling=True
                ):
                    return False
                if self.model.shutdown_cancellation_confirmed:
                    if not self._accept_export_attempt(
                        event.attempt_id,
                        "export_completed",
                        install=False,
                    ):
                        return False
                    staged = self._stage_export_continuation(
                        event.target_results
                    )
                    return self._commit_export_continuation(staged)
                if not self._accept_export_attempt(
                    event.attempt_id, "export_completed"
                ):
                    return False
                return self._resolve_cancelled_domain()
            if not self._valid_export_event(event, "export_completed"):
                return False
            if self.model.export_failure_pending:
                return self._diagnose_stale(
                    "export",
                    "export_completed",
                    "export completion arrived while a failure decision is pending",
                    expected_job_id=self.model.active_job_id,
                    received_job_id=event.job_id,
                    expected_attempt_id=self.model.active_attempt_id,
                    received_attempt_id=event.attempt_id,
                )
            if not self._accept_export_attempt(
                event.attempt_id, "export_completed", install=False
            ):
                return False
            staged = self._stage_export_continuation(event.target_results)
            return self._commit_export_continuation(staged)
        except BaseException as error:
            try:
                logging.getLogger(__name__).error(
                    "Failed to stage or commit export continuation: %s",
                    type(error).__name__,
                )
            except BaseException:
                pass
            return False

    @pyqtSlot(object)
    def handle_export_failed(self, event: ExportFailed) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if not self._valid_export_event(
                event, "export_failed", cancelling=True
            ):
                return False
            if not self._accept_export_attempt(event.attempt_id, "export_failed"):
                return False
            if self.model.shutdown_cancellation_confirmed:
                self.model.export_outcome = event.failures
                self.model.export_failure_pending = True
                self.model.cancelling_phase = None
                self.model.cancelling_domain = None
                self._transition(WorkflowPhase.RESULT_EXPORTING)
                return True
            return self._resolve_cancelled_domain()
        if not self._valid_export_event(event, "export_failed"):
            return False
        if self.model.export_failure_pending:
            return self._diagnose_stale(
                "export",
                "export_failed",
                "export failure arrived while a failure decision is pending",
                expected_job_id=self.model.active_job_id,
                received_job_id=event.job_id,
                expected_attempt_id=self.model.active_attempt_id,
                received_attempt_id=event.attempt_id,
            )
        if not self._accept_export_attempt(event.attempt_id, "export_failed"):
            return False
        self.model.active_attempt_id = event.attempt_id
        self.model.export_outcome = event.failures
        self.model.export_failure_pending = True
        return True

    @pyqtSlot(object)
    def handle_retry_export(self, command: RetryExportRequested) -> bool:
        if not self._valid_export_decision(command, "retry_export_requested"):
            return False
        # This command is only a request. Export owns attempt allocation and
        # Workflow retains the actionable failed identity until the exact
        # installation acknowledgement arrives.
        return True

    @pyqtSlot(object)
    def handle_export_retry_accepted(
        self, event: ExportRetryAccepted
    ) -> bool:
        if type(event) is not ExportRetryAccepted:
            return False
        if self.model.phase is not WorkflowPhase.RESULT_EXPORTING:
            reason = "export retry acknowledgement arrived outside result export"
        elif not self.model.export_failure_pending:
            reason = "export retry acknowledgement has no pending failure"
        elif event.job_id != self.model.active_job_id:
            reason = "export retry acknowledgement has a mismatched job"
        elif event.previous_attempt_id != self.model.active_attempt_id:
            reason = "export retry acknowledgement has a mismatched attempt"
        else:
            self.model.retire_export_attempt(event.previous_attempt_id)
            self.model.active_attempt_id = event.new_attempt_id
            self.model.export_failure_pending = False
            return True
        return self._diagnose_stale(
            "export",
            "export_retry_accepted",
            reason,
            expected_job_id=self.model.active_job_id,
            received_job_id=event.job_id,
            expected_attempt_id=self.model.active_attempt_id,
            received_attempt_id=event.previous_attempt_id,
        )

    @pyqtSlot(object)
    def handle_ignore_export_failure(
        self, command: IgnoreExportFailureRequested
    ) -> bool:
        if not self._valid_export_decision(
            command, "ignore_export_failure_requested"
        ):
            return False
        if self.export_decision_requires_terminal:
            # Export Controller owns the ignored terminal and publishes the
            # matching ExportCompleted event. Workflow keeps its continuation
            # frozen until that event arrives.
            self.model.export_failure_pending = False
            return True
        try:
            return self._continue_after_export(self.model.export_outcome)
        except BaseException:
            return False

    def _valid_label_event(
        self, event: Any, event_kind: str, *, cancelling: bool = False
    ) -> bool:
        valid_identity = (
            event.command_id == self.model.active_label_command_id
            and event.record_id == self.model.active_label_record_id
            and event.label == self.model.active_label
        )
        if cancelling:
            valid = self.model.cancelling_domain == "label" and valid_identity
            reason = "label terminal does not match the cancelling label commit"
        else:
            valid = (
                self.model.phase is WorkflowPhase.LABEL_COMMITTING
                and valid_identity
            )
            reason = "label terminal does not match the active label commit"
        if valid:
            return True
        return self._diagnose_stale(
            "label",
            event_kind,
            reason,
            expected_command_id=self.model.active_label_command_id,
            received_command_id=event.command_id,
            expected_record_id=self.model.active_label_record_id,
            received_record_id=event.record_id,
            expected_label=self.model.active_label,
            received_label=event.label,
            expected_cancelling_domain="label" if cancelling else None,
            received_cancelling_domain=(
                self.model.cancelling_domain if cancelling else None
            ),
            expected_phase=(
                None if cancelling else WorkflowPhase.LABEL_COMMITTING.name
            ),
        )

    def _diagnose_retained_cleanup_failure(
        self,
        pending: _PendingRetainedRecordingCleanup,
        reason: str,
        *,
        failure_type: str | None = None,
    ) -> None:
        diagnostic = {
            "domain": "recording",
            "event_kind": "retained_recording_cleanup_retry",
            "reason": reason,
            "current_phase": self.model.phase.name,
            "workflow_generation": self.model.workflow_generation,
            "pending_identity": pending.identity,
            "retry_attempt": self._retained_cleanup_retry_attempt,
        }
        if failure_type is not None:
            diagnostic["failure_type"] = failure_type
        self._retained_cleanup_last_diagnostic = diagnostic
        try:
            self.diagnostic_callback(dict(diagnostic))
        except BaseException:
            # Diagnostics are an external best-effort boundary. The immutable
            # local diagnostic remains available and cleanup retry must survive.
            logging.getLogger(__name__).debug(
                "Retained cleanup diagnostic callback failed",
                exc_info=True,
            )

    def _finalize_native_deletion(
        self,
        reason: str,
        *,
        pending: _PendingRetainedRecordingCleanup | None = None,
        cleanup_acknowledged: bool = False,
    ) -> bool:
        """Resolve Python workflow state without invoking a dead Qt wrapper."""
        return self._native_retained_cleanup_lifecycle.finalize(
            reason,
            pending=pending,
            cleanup_acknowledged=cleanup_acknowledged,
        )

    def _retained_cleanup_timer_is_available(self) -> bool:
        if _qt_object_is_deleted(self):
            return False
        return not _qt_object_is_deleted(self._retained_cleanup_retry_timer)

    def _stop_retained_cleanup_retry_timer(self) -> bool:
        if not self._retained_cleanup_timer_is_available():
            return False
        try:
            self._retained_cleanup_retry_timer.stop()
        except BaseException:
            return False
        return True

    def _schedule_retained_cleanup_retry(
        self, pending: _PendingRetainedRecordingCleanup
    ) -> bool:
        if (
            self._pending_retained_cleanup is not pending
            or not self._continuation_dispatch_active
        ):
            return False
        registration = (
            self._native_retained_cleanup_lifecycle.register_native_finalization_root(
                self.bus,
                self._retained_cleanup_registry_token,
            )
        )
        if registration not in (
            RetainedCleanupLifecycleRegistrationResult.REGISTERED,
            RetainedCleanupLifecycleRegistrationResult.IDEMPOTENT,
        ):
            if (
                registration
                is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
            ):
                reason = "retained cleanup lifecycle token collision"
            else:
                reason = "event bus unavailable for retained cleanup retry"
            self._finalize_native_deletion(reason, pending=pending)
            return False
        if not self._retained_cleanup_timer_is_available():
            self._finalize_native_deletion(
                "native retained cleanup timer unavailable",
                pending=pending,
            )
            return False
        try:
            if self._retained_cleanup_retry_timer.isActive():
                return True
        except BaseException:
            self._finalize_native_deletion(
                "native retained cleanup timer inspection failed",
                pending=pending,
            )
            return False
        exponent = min(self._retained_cleanup_retry_attempt, 30)
        delay = min(
            self._retained_cleanup_retry_base_delay_ms * (2 ** exponent),
            self._retained_cleanup_retry_max_delay_ms,
        )
        delay = max(1, int(delay))
        self._retained_cleanup_retry_attempt += 1
        self._retained_cleanup_retry_delay_ms = delay
        try:
            self._retained_cleanup_retry_timer.start(delay)
        except BaseException:
            self._finalize_native_deletion(
                "native retained cleanup timer scheduling failed",
                pending=pending,
            )
            return False
        return True

    def _abandon_pending_retained_cleanup(self, reason: str) -> bool:
        pending = self._pending_retained_cleanup
        if pending is not None and not self._stop_retained_cleanup_retry_timer():
            return self._finalize_native_deletion(
                "native retained cleanup timer unavailable during abandon",
                pending=pending,
            )
        self._retained_cleanup_retry_attempt = 0
        self._retained_cleanup_retry_delay_ms = 0
        self._pending_retained_cleanup = None
        self._native_retained_cleanup_lifecycle._retire_native_finalization_root()
        with self._label_terminal_cleanup_lock:
            self._label_terminal_cleanup_reentered = False
        if pending is None:
            return False
        self._diagnose_retained_cleanup_failure(pending, reason)
        return True

    def _retire_pending_retained_cleanup_for_disconnect(self) -> bool:
        """Abandon live cleanup without invoking native domain finalization."""
        pending = self._pending_retained_cleanup
        self._retained_cleanup_retry_attempt = 0
        self._retained_cleanup_retry_delay_ms = 0
        self._pending_retained_cleanup = None
        with self._label_terminal_cleanup_lock:
            self._label_terminal_cleanup_reentered = False
        if pending is not None:
            self._diagnose_retained_cleanup_failure(
                pending,
                "workflow-disconnect",
            )
        self._native_retained_cleanup_lifecycle.retire()
        return pending is not None

    def _complete_pending_retained_cleanup(
        self, pending: _PendingRetainedRecordingCleanup
    ) -> bool:
        event = pending.terminal
        if (
            self._pending_retained_cleanup is not pending
            or self.model.phase is not WorkflowPhase.LABEL_COMMITTING
            or self.model.workflow_generation != pending.workflow_generation
            or self.model.retained_record_id != event.record_id
            or self.model.active_label_command_id != event.command_id
            or self.model.active_label_record_id != event.record_id
            or self.model.active_label != event.label
        ):
            self._abandon_pending_retained_cleanup(
                "workflow state changed before retained cleanup acknowledgement"
            )
            return False
        if not self._stop_retained_cleanup_retry_timer():
            self._finalize_native_deletion(
                "native retained cleanup timer unavailable after acknowledgement",
                pending=pending,
                cleanup_acknowledged=True,
            )
            return False
        self._retained_cleanup_retry_attempt = 0
        self._retained_cleanup_retry_delay_ms = 0
        self._pending_retained_cleanup = None
        self._native_retained_cleanup_lifecycle._retire_native_finalization_root()
        self.model.retained_record_id = None
        self.model.awaiting_label = False
        if self.model.post_analysis_continuation is not None:
            return self._finish_post_analysis_transport()
        self._finish_idle()
        return True

    def _attempt_pending_retained_cleanup(self) -> bool:
        pending = self._pending_retained_cleanup
        if pending is None:
            return False
        with self._label_terminal_cleanup_lock:
            if self._label_terminal_cleanup_active:
                self._label_terminal_cleanup_reentered = True
                return False
            self._label_terminal_cleanup_active = True
            self._label_terminal_cleanup_reentered = False
        failure_reason: str | None = None
        failure_type: str | None = None
        try:
            try:
                cleared = self.clear_retained_recording_snapshot(
                    pending.terminal.record_id,
                    workflow_generation=pending.workflow_generation,
                )
            except BaseException as error:
                cleared = False
                failure_reason = "retained cleanup collaborator raised"
                failure_type = type(error).__name__
            with self._label_terminal_cleanup_lock:
                reentered = self._label_terminal_cleanup_reentered
            if reentered:
                failure_reason = "retained cleanup reentered its terminal"
            elif cleared is False and failure_reason is None:
                failure_reason = "retained cleanup collaborator returned False"
            elif cleared is not None and type(cleared) is not bool:
                failure_reason = "retained cleanup collaborator violated bool-or-None"
            if _qt_object_is_deleted(self):
                self._finalize_native_deletion(
                    "native workflow owner destroyed during retained cleanup",
                    pending=pending,
                    cleanup_acknowledged=failure_reason is None,
                )
                return False
            if failure_reason is None:
                return self._complete_pending_retained_cleanup(pending)
            self._diagnose_retained_cleanup_failure(
                pending,
                failure_reason,
                failure_type=failure_type,
            )
            return False
        finally:
            with self._label_terminal_cleanup_lock:
                self._label_terminal_cleanup_active = False
                self._label_terminal_cleanup_reentered = False
            if (
                self._pending_retained_cleanup is pending
                and not _qt_object_is_deleted(self)
            ):
                self._schedule_retained_cleanup_retry(pending)

    @pyqtSlot()
    def _retry_pending_retained_cleanup(self) -> None:
        self._attempt_pending_retained_cleanup()

    @pyqtSlot(object)
    def handle_label_committed(self, event: RecordingLabelCommitted) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if not self._valid_label_event(
                event, "recording_label_committed", cancelling=True
            ):
                return False
            return self._resolve_cancelled_domain()
        if not self._valid_label_event(event, "recording_label_committed"):
            return False
        if self.model.retained_record_id != event.record_id:
            return False
        pending = self._pending_retained_cleanup
        if pending is not None:
            with self._label_terminal_cleanup_lock:
                if self._label_terminal_cleanup_active:
                    self._label_terminal_cleanup_reentered = True
            return self._diagnose_stale(
                "recording",
                "recording_label_committed",
                "retained cleanup is already pending for this terminal",
                pending_identity=pending.identity,
                received_command_id=event.command_id,
                received_record_id=event.record_id,
                received_label=event.label,
            )
        pending = _PendingRetainedRecordingCleanup(
            event,
            self.model.workflow_generation,
        )
        self._pending_retained_cleanup = pending
        return self._attempt_pending_retained_cleanup()

    @pyqtSlot(object)
    def handle_label_failed(self, event: RecordingLabelCommitFailed) -> bool:
        if self.model.phase is WorkflowPhase.CANCELLING:
            if not self._valid_label_event(
                event, "recording_label_commit_failed", cancelling=True
            ):
                return False
            return self._resolve_cancelled_domain()
        if not self._valid_label_event(
            event, "recording_label_commit_failed"
        ):
            return False
        if self._pending_retained_cleanup is not None:
            return self._diagnose_stale(
                "recording",
                "recording_label_commit_failed",
                "label failure arrived after committed cleanup became pending",
                pending_identity=self.pending_retained_cleanup_identity,
            )
        if self.model.post_analysis_continuation is not None:
            return self._finish_post_analysis_transport()
        self._finish_idle()
        return True

    def handle_legacy_recording_flags(
        self,
        player_active: bool,
        workflow_busy: bool,
        *,
        activation_edge: bool = False,
    ) -> bool:
        """Reconcile transitional writable flags with the canonical recording phase."""
        if (
            type(player_active) is not bool
            or type(workflow_busy) is not bool
            or type(activation_edge) is not bool
        ):
            return False
        phase = self.model.phase
        if phase is WorkflowPhase.IDLE:
            if self._pending_export_preparation is not None:
                return False
            if (
                not activation_edge
                or self.model.shutdown_pending
                or not (player_active or workflow_busy)
            ):
                return False
            session_id = self._allocate_domain_id("session", self.session_id_factory)
            self._begin_workflow()
            self.model.active_session_id = session_id
            self.model.active_session_origin = SessionOrigin.LEGACY_BRIDGE
            self.model.session_snapshot = {
                "legacy": True,
                "workflow_generation": self.model.workflow_generation,
            }
            self._transition(WorkflowPhase.PREPARING)
            if player_active:
                self._transition(WorkflowPhase.RECORDING)
            return True
        if phase not in {
            WorkflowPhase.PREPARING,
            WorkflowPhase.RECORDING,
            WorkflowPhase.FINALIZING,
            WorkflowPhase.CANCELLING,
        }:
            return False
        if self.model.active_session_origin is not SessionOrigin.LEGACY_BRIDGE:
            return False
        if phase is WorkflowPhase.CANCELLING:
            if self.model.cancelling_domain != "recording":
                return False
            if not player_active and not workflow_busy:
                self.model.shutdown_asserted_active = False
                return self._resolve_cancelled_domain()
            return True
        if not player_active and not workflow_busy:
            self.model.shutdown_asserted_active = False
            self._finish_idle()
            return True
        if phase is WorkflowPhase.PREPARING and player_active:
            self._transition(WorkflowPhase.RECORDING)
        elif phase is WorkflowPhase.RECORDING and not player_active and workflow_busy:
            self._transition(WorkflowPhase.FINALIZING)
        return True

    @pyqtSlot(object)
    def handle_cancel_workflow(self, command: CancelWorkflowRequested) -> bool:
        if command.workflow_generation != self.model.workflow_generation:
            self._diagnose_stale(
                "workflow",
                "cancel_workflow_requested",
                "cancellation has a stale workflow generation",
                expected_generation=self.model.workflow_generation,
                received_generation=command.workflow_generation,
                command_id=command.command_id,
            )
            return self._reject(command.command_id, "stale workflow generation")
        phase = self.model.phase
        preparation = self._pending_export_preparation
        if preparation is not None and (
            (
                type(preparation) is PrepareAnalysisExportRequested
                and phase is WorkflowPhase.ANALYZING
            )
            or (
                type(preparation) is PrepareManualLabelExportRequested
                and phase is WorkflowPhase.IDLE
            )
        ):
            request_kind = (
                "analysis-export-prepare"
                if type(preparation) is PrepareAnalysisExportRequested
                else "manual-label-export-prepare"
            )
            request_delivery_id = (
                request_kind,
                preparation.request_id,
                preparation.workflow_generation,
            )
            abandon = getattr(self.bus, "abandon_workflow_continuations", None)
            if callable(abandon):
                abandon(
                    (request_delivery_id,),
                    owner=self,
                    reason="export preparation cancelled",
                )
            self._pending_continuation_publications.pop(
                request_delivery_id, None
            )
            if not self._pending_continuation_publications:
                self._continuation_retry_timer.stop()
                self._continuation_retry_attempt = 0
                self._continuation_retry_delay_ms = 0
            domain = "preparation"
            cancellation = CancelExportPreparationRequested(
                preparation.request_id,
                preparation.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_export_preparation_requested
            self.model.active_analysis_id = None
        elif phase is WorkflowPhase.IMPORTING and self.model.active_import_id is not None:
            domain = "import"
            cancellation = CancelImportedAudioRequested(
                self.model.active_import_id,
                self.model.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_imported_audio_requested
        elif phase in {
            WorkflowPhase.PREPARING,
            WorkflowPhase.RECORDING,
            WorkflowPhase.FINALIZING,
        } and self.model.active_session_id is not None:
            domain = "recording"
            cancellation = CancelRecordingRequested(
                self.model.active_session_id,
                self.model.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_recording_requested
        elif phase is WorkflowPhase.ANALYZING and self.model.active_analysis_id is not None:
            domain = "analysis"
            cancellation = CancelAnalysisRequested(
                self.model.active_analysis_id,
                self.model.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_analysis_requested
        elif phase is WorkflowPhase.RESULT_EXPORTING and self.model.active_job_id is not None:
            domain = "export"
            cancellation = CancelExportRequested(
                self.model.active_job_id,
                self.model.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_export_requested
        elif (
            phase is WorkflowPhase.LABEL_COMMITTING
            and self.model.active_label_command_id is not None
        ):
            if self._pending_retained_cleanup is not None:
                return self._reject(
                    command.command_id,
                    "retained recording cleanup is pending",
                )
            domain = "label"
            cancellation = CancelRecordingRequested(
                self.model.active_label_command_id,
                self.model.workflow_generation,
                command.reason,
            )
            signal = self.bus.commands.cancel_recording_requested
        else:
            return self._reject(command.command_id, "workflow phase is not cancellable")
        if domain == "recording":
            record_cancellation = getattr(
                self.bus,
                "_record_canonical_recording_cancellation",
                None,
            )
            capability = self._canonical_recording_admission_capability
            if (
                callable(record_cancellation)
                and (
                    capability is None
                    or not record_cancellation(capability, cancellation)
                )
            ):
                return self._reject(
                    command.command_id,
                    "recording cancellation lifecycle is stale",
                )
        self.model.cancelling_phase = phase
        self.model.cancelling_domain = domain
        self._transition(WorkflowPhase.CANCELLING)
        if domain == "preparation":
            self._publish_continuation(
                (
                    "export-preparation-cancel",
                    cancellation.request_id,
                    cancellation.workflow_generation,
                ),
                signal,
                cancellation,
            )
            return True
        signal.emit(cancellation)
        return True

    def _resolve_cancelled_domain(self) -> bool:
        self._finish_idle()
        return True

    def _enter_shutdown_flushing(
        self, *, continuation_publication: bool = False
    ) -> None:
        self.model.shutdown_pending = False
        self.model.shutdown_asserted_active = False
        self.model.shutdown_cancellation_confirmed = False
        self._transition(
            WorkflowPhase.CLOSING,
            continuation_publication=continuation_publication,
        )
        self._transition(
            WorkflowPhase.SHUTDOWN_FLUSHING,
            continuation_publication=continuation_publication,
        )

    @pyqtSlot(object)
    def handle_shutdown(self, command: ShutdownRequested) -> bool:
        generation = command.shutdown_generation
        if self.model.shutdown_generation is not None:
            return self._diagnose_stale(
                "shutdown",
                "shutdown_requested",
                "a shutdown generation is already active",
                expected_generation=self.model.shutdown_generation,
                received_generation=generation,
            )
        if generation <= self.model.last_shutdown_generation:
            return self._diagnose_stale(
                "shutdown",
                "shutdown_requested",
                "shutdown request has a stale generation",
                expected_generation=self.model.last_shutdown_generation + 1,
                received_generation=generation,
                last_shutdown_generation=self.model.last_shutdown_generation,
            )
        actual_activity = (
            self.model.phase is not WorkflowPhase.IDLE
            or self._pending_export_preparation is not None
        )
        self.model.shutdown_generation = generation
        self.model.last_shutdown_generation = generation
        self.model.shutdown_pending = True
        self.model.shutdown_asserted_active = actual_activity
        self.model.shutdown_cancellation_confirmed = False
        if not actual_activity:
            self._enter_shutdown_flushing()
        return True

    @pyqtSlot(object)
    def handle_confirm_shutdown_cancellation(
        self, command: ConfirmShutdownCancellationRequested
    ) -> bool:
        if command.shutdown_generation != self.model.shutdown_generation:
            return self._diagnose_stale(
                "shutdown",
                "confirm_shutdown_cancellation_requested",
                "shutdown confirmation has a mismatched generation",
                expected_generation=self.model.shutdown_generation,
                received_generation=command.shutdown_generation,
            )
        if not self.model.shutdown_pending:
            return self._diagnose_stale(
                "shutdown",
                "confirm_shutdown_cancellation_requested",
                "shutdown confirmation arrived without a pending shutdown",
                expected_generation=self.model.shutdown_generation,
                received_generation=command.shutdown_generation,
            )
        if (
            self.model.phase is WorkflowPhase.IDLE
            and self._pending_export_preparation is None
        ):
            self._enter_shutdown_flushing()
            return True
        if self.model.phase is WorkflowPhase.CANCELLING:
            self.model.shutdown_cancellation_confirmed = True
            return True
        if (
            self.model.phase is WorkflowPhase.RESULT_EXPORTING
            and self.model.export_failure_pending
        ):
            self.model.shutdown_cancellation_confirmed = True
            self.model.assert_invariants()
            return True
        if (
            self.model.phase is WorkflowPhase.LABEL_COMMITTING
            and self._pending_retained_cleanup is not None
        ):
            self.model.shutdown_cancellation_confirmed = True
            self.model.assert_invariants()
            return True
        self.model.shutdown_cancellation_confirmed = True
        accepted = self.handle_cancel_workflow(
            CancelWorkflowRequested(
                command_id=f"shutdown-{command.shutdown_generation}",
                workflow_generation=self.model.workflow_generation,
                reason="shutdown",
            )
        )
        if not accepted:
            self.model.shutdown_cancellation_confirmed = False
        return accepted

    @pyqtSlot(object)
    def handle_abort_shutdown(self, command: AbortShutdownRequested) -> bool:
        if command.shutdown_generation != self.model.shutdown_generation:
            return self._diagnose_stale(
                "shutdown",
                "abort_shutdown_requested",
                "shutdown abort has a mismatched generation",
                expected_generation=self.model.shutdown_generation,
                received_generation=command.shutdown_generation,
            )
        if not self.model.shutdown_pending:
            return self._diagnose_stale(
                "shutdown",
                "abort_shutdown_requested",
                "shutdown abort arrived without a pending shutdown",
                expected_generation=self.model.shutdown_generation,
                received_generation=command.shutdown_generation,
            )
        if self.model.shutdown_cancellation_confirmed or self.model.phase in {
            WorkflowPhase.CLOSING,
            WorkflowPhase.SHUTDOWN_FLUSHING,
            WorkflowPhase.SHUTDOWN_READY,
        }:
            return self._diagnose_stale(
                "shutdown",
                "abort_shutdown_requested",
                "shutdown abort arrived after shutdown cancellation or flushing began",
                expected_generation=self.model.shutdown_generation,
                received_generation=command.shutdown_generation,
            )
        generation = self.model.shutdown_generation
        self.model.shutdown_generation = None
        self.model.shutdown_pending = False
        self.model.shutdown_asserted_active = False
        self.model.shutdown_cancellation_confirmed = False
        self.bus.events.shutdown_aborted.emit(ShutdownAborted(generation))
        return True

    @pyqtSlot(object)
    def handle_shutdown_ready(self, event: ShutdownReady) -> bool:
        if type(event) is not ShutdownReady:
            return False
        if event.shutdown_generation != self.model.shutdown_generation:
            return self._diagnose_stale(
                "shutdown",
                "shutdown_ready",
                "shutdown ready has a mismatched generation",
                expected_generation=self.model.shutdown_generation,
                received_generation=event.shutdown_generation,
                expected_phase=WorkflowPhase.SHUTDOWN_FLUSHING.name,
            )
        if self.model.phase is WorkflowPhase.SHUTDOWN_READY:
            return True
        if self.model.phase is not WorkflowPhase.SHUTDOWN_FLUSHING:
            return self._diagnose_stale(
                "shutdown",
                "shutdown_ready",
                "shutdown ready arrived outside shutdown flushing",
                expected_generation=self.model.shutdown_generation,
                received_generation=event.shutdown_generation,
                expected_phase=WorkflowPhase.SHUTDOWN_FLUSHING.name,
            )
        self._transition(WorkflowPhase.SHUTDOWN_READY)
        return True


class SequenceShutdownCoordinator(QObject):
    """Bridge top-level close decisions to Workflow and Export messages."""

    def __init__(
        self,
        model: SequenceWorkflowModel,
        bus: SequenceEventBus,
        *,
        view: Any,
        cleanup_resources: Callable[[int], bool] | None = None,
        shutdown_ready: Callable[[ShutdownReady], Any] | None = None,
        finalize_after_ready_ack: Callable[[int], bool] | None = None,
        release_shutdown_close: Callable[[int], bool] | None = None,
        shutdown_aborted: Callable[[ShutdownAborted], Any] | None = None,
        logger: Any = None,
        parent: QObject | None = None,
        connect_bus: bool = True,
    ) -> None:
        super().__init__(parent)
        if type(model) is not SequenceWorkflowModel:
            raise TypeError("model must be SequenceWorkflowModel")
        self.model = model
        self.bus = bus
        self.view = view
        self.cleanup_resources = cleanup_resources or (lambda _generation: True)
        self.shutdown_ready = shutdown_ready or (
            lambda event: self._project_ready_without_workflow_owner(event)
        )
        self.finalize_after_ready_ack = finalize_after_ready_ack or (
            lambda _generation: True
        )
        self.release_shutdown_close = release_shutdown_close or (
            lambda _generation: True
        )
        self.shutdown_aborted = shutdown_aborted or (lambda _event: None)
        self.logger = logger
        self._confirmation_generation: int | None = None
        self._waiting_generation: int | None = None
        self._flush_requested_generation: int | None = None
        self._cleanup_pending_generation: int | None = None
        self._ready_pending_generation: int | None = None
        self._workflow_ready_generation: int | None = None
        self._ready_ack_generation: int | None = None
        self._finalization_attempt_generation: int | None = None
        self._finalization_attempt_token: object | None = None
        self._retry_queued_generation: int | None = None
        self._completed_generations: set[int] = set()
        self._automatic_retry_attempt = 0
        self._automatic_retry_limit = 5
        self._automatic_retry_base_delay_ms = 25
        self._automatic_retry_max_delay_ms = 400
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._handle_retry_timeout)
        self._connections: list[tuple[Any, Any]] = []
        self._disconnect_steps_completed: set[str] = set()
        self._active = True
        register_owner = getattr(
            bus, "register_workflow_continuation_lifecycle_owner", None
        )
        if callable(register_owner):
            register_owner(self)
        if connect_bus:
            self._connect(
                bus.events.workflow_state_changed,
                self.handle_workflow_state_changed,
            )
            self._connect(
                bus.commands.confirm_shutdown_cancellation_requested,
                self._observe_shutdown_confirmation,
            )
            register_continuation = getattr(
                bus, "register_workflow_continuation_recipient", None
            )
            if callable(register_continuation):
                self._shutdown_flush_recipient_name = (
                    f"shutdown-coordinator:{id(self)}"
                )
                register_continuation(
                    "shutdown-flush-completed",
                    self._shutdown_flush_recipient_name,
                    self._receive_shutdown_flush_completed,
                    owner=self,
                )
            else:
                self._shutdown_flush_recipient_name = None
                self._connect(
                    bus.events.shutdown_flush_completed,
                    self.handle_shutdown_flush_completed,
                )
            self._connect(bus.events.shutdown_aborted, self.handle_shutdown_aborted)
        else:
            self._shutdown_flush_recipient_name = None
        confirm = getattr(view, "confirm_shutdown_requested", None)
        if confirm is not None and hasattr(confirm, "connect"):
            self._connect(
                confirm,
                self.bus.commands.confirm_shutdown_cancellation_requested.emit,
            )
        abort = getattr(view, "abort_shutdown_requested", None)
        if abort is not None and hasattr(abort, "connect"):
            self._connect(abort, self.bus.commands.abort_shutdown_requested.emit)

    def _project_ready_without_workflow_owner(self, event: ShutdownReady) -> bool:
        if (
            type(event) is not ShutdownReady
            or event.shutdown_generation != self.model.shutdown_generation
        ):
            return False
        if self.model.phase is WorkflowPhase.SHUTDOWN_READY:
            self.model.assert_invariants()
            return True
        if self.model.phase is not WorkflowPhase.SHUTDOWN_FLUSHING:
            return False
        self.model.phase = WorkflowPhase.SHUTDOWN_READY
        self.model.assert_invariants()
        return True

    def _connect(self, signal: Any, slot: Any) -> None:
        try:
            signal.connect(slot, Qt.QueuedConnection)
        except TypeError:
            signal.connect(slot)
        self._connections.append((signal, slot))

    @pyqtSlot(object)
    def _observe_shutdown_confirmation(self, _command: Any) -> None:
        if self._active:
            # Workflow receives the same queued command first. One subsequent
            # event-loop turn projects its accepted confirmation without polling.
            QTimer.singleShot(0, self.synchronize)

    def _log(self, message: str) -> None:
        try:
            callback = getattr(self.logger, "warning", None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def request_shutdown(self, generation: int, has_active_workflow: bool) -> bool:
        if not self._active or type(generation) is not int or generation < 0:
            return False
        current = self.model.shutdown_generation
        if current is not None:
            if current == generation:
                return self.raise_progress(generation)
            return False
        try:
            self.bus.commands.shutdown_requested.emit(
                ShutdownRequested(generation, bool(has_active_workflow))
            )
            # The Workflow command is queued. Schedule one event-loop turn to
            # project its accepted decision; this is a one-shot continuation,
            # not a state poll or retry loop.
            QTimer.singleShot(0, self.synchronize)
        except BaseException as error:
            self._log(f"shutdown request publication failed: {type(error).__name__}")
            return False
        return True

    @pyqtSlot(object)
    def handle_workflow_state_changed(self, _event: Any) -> bool:
        return self.synchronize()

    def synchronize(self) -> bool:
        if not self._active:
            return False
        if (
            self._cleanup_pending_generation is not None
            or self._ready_pending_generation is not None
        ):
            return self._restart_retry_round()
        generation = self.model.shutdown_generation
        if generation is None:
            return False
        if self.model.shutdown_cancellation_confirmed:
            if self._confirmation_generation == generation:
                getattr(
                    self.view,
                    "finish_shutdown_confirmation",
                    lambda _g: False,
                )(generation)
                self._confirmation_generation = None
            if self._waiting_generation != generation:
                shown = getattr(
                    self.view, "show_shutdown_waiting", lambda _g: False
                )(generation)
                if shown is False:
                    return False
                self._waiting_generation = generation
            return True
        if self.model.shutdown_pending and self.model.shutdown_asserted_active:
            if self._confirmation_generation == generation:
                return True
            shown = getattr(self.view, "show_shutdown_confirmation", lambda _g: False)(
                generation
            )
            if shown is False:
                return False
            self._confirmation_generation = generation
            return True
        if self.model.phase is WorkflowPhase.SHUTDOWN_FLUSHING:
            if self._confirmation_generation == generation:
                getattr(
                    self.view,
                    "finish_shutdown_confirmation",
                    lambda _g: False,
                )(generation)
                self._confirmation_generation = None
            if self._waiting_generation == generation:
                getattr(
                    self.view,
                    "finish_shutdown_waiting",
                    lambda _g: False,
                )(generation)
                self._waiting_generation = None
            if self._flush_requested_generation == generation:
                return True
            self._flush_requested_generation = generation
            self.bus.commands.begin_shutdown_flush_requested.emit(
                BeginShutdownFlushRequested(generation)
            )
            return True
        return False

    def raise_progress(self, generation: int) -> bool:
        if not self._active or generation != self.model.shutdown_generation:
            return False
        if (
            self._cleanup_pending_generation == generation
            or self._ready_pending_generation == generation
        ):
            return self._restart_retry_round()
        try:
            raised = (
                getattr(self.view, "raise_shutdown", lambda _g: False)(generation)
                is not False
            )
        except BaseException as error:
            self._log(f"shutdown progress raise failed: {type(error).__name__}")
            raised = False
        if raised:
            return True
        if self.model.phase is WorkflowPhase.SHUTDOWN_FLUSHING:
            try:
                self.bus.commands.begin_shutdown_flush_requested.emit(
                    BeginShutdownFlushRequested(generation)
                )
                return True
            except BaseException as error:
                self._log(
                    f"shutdown flush presentation retry failed: {type(error).__name__}"
                )
                return False
        return self.synchronize()

    def _schedule_retry(self) -> None:
        if _qt_object_is_deleted(self):
            return
        timer = self._retry_timer
        if _qt_object_is_deleted(timer):
            return
        if (
            not self._active
            or timer.isActive()
            or self._automatic_retry_attempt >= self._automatic_retry_limit
        ):
            return
        delay = min(
            self._automatic_retry_base_delay_ms
            * (2**self._automatic_retry_attempt),
            self._automatic_retry_max_delay_ms,
        )
        self._automatic_retry_attempt += 1
        if self._active and not _qt_object_is_deleted(timer):
            timer.start(delay)

    @pyqtSlot()
    def _handle_retry_timeout(self) -> None:
        self.retry_pending_shutdown()

    def _restart_retry_round(self) -> bool:
        if _qt_object_is_deleted(self):
            return False
        timer = self._retry_timer
        if _qt_object_is_deleted(timer):
            return False
        timer.stop()
        self._automatic_retry_attempt = 0
        # A repeated close may arrive from inside MainWindow.closeEvent().  Queue
        # the retry so a successful Ready callback cannot nest a second close
        # event inside the first one.
        if not self._active or _qt_object_is_deleted(timer):
            return False
        timer.start(0)
        return True

    def _attempt_is_current(self, generation: int, token: object) -> bool:
        return (
            not _qt_object_is_deleted(self)
            and self._active
            and self._finalization_attempt_generation == generation
            and self._finalization_attempt_token is token
            and generation == self.model.shutdown_generation
        )

    def _retire_after_ready_ack(self) -> bool:
        """Retire local Qt state after the formal Ready delivery is ACKed."""
        if _qt_object_is_deleted(self):
            return False
        timer = self._retry_timer
        if _qt_object_is_deleted(timer):
            return False
        timer.stop()
        while self._connections:
            signal, slot = self._connections[-1]
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            except BaseException as error:
                self._log(
                    "shutdown coordinator signal disconnect failed after Ready ACK: "
                    f"{type(error).__name__}"
                )
                return False
            if _qt_object_is_deleted(self):
                return False
            self._connections.pop()
        self._disconnect_steps_completed.update({"timer", "connections"})
        self._active = False
        return True

    def _deliver_shutdown_ready(self, generation: int, token: object) -> bool:
        ready = ShutdownReady(generation)
        if self._workflow_ready_generation != generation:
            try:
                projected = self.shutdown_ready(ready) is True
            except BaseException as error:
                self._log(
                    f"shutdown ready domain projection failed: {type(error).__name__}"
                )
                projected = False
            if _qt_object_is_deleted(self):
                return False
            if not self._attempt_is_current(generation, token):
                return False
            if not projected:
                self._schedule_retry()
                return False
            self._workflow_ready_generation = generation
        if self._ready_ack_generation != generation:
            delivery_outcome = getattr(
                self.bus, "deliver_workflow_continuation_outcome", None
            )
            if not callable(delivery_outcome):
                return False
            try:
                outcome = delivery_outcome(
                    ("shutdown-ready", generation),
                    "shutdown-ready",
                    ready,
                    owner=self,
                )
                delivered = (
                    type(outcome) is WorkflowContinuationDeliveryOutcome
                    and outcome.status is WorkflowContinuationDeliveryStatus.ACK
                )
            except BaseException as error:
                self._log(f"shutdown ready delivery failed: {type(error).__name__}")
                delivered = False
            if _qt_object_is_deleted(self):
                return False
            if not self._attempt_is_current(generation, token):
                return False
            if not delivered:
                self._schedule_retry()
                return False
            self._ready_ack_generation = generation
        try:
            finalized = self.finalize_after_ready_ack(generation) is True
        except BaseException as error:
            self._log(
                f"shutdown post-ack finalization failed: {type(error).__name__}"
            )
            finalized = False
        if _qt_object_is_deleted(self):
            return False
        if not self._attempt_is_current(generation, token):
            return False
        if not finalized:
            self._schedule_retry()
            return False
        self._completed_generations.add(generation)
        self._ready_pending_generation = None
        self._retry_queued_generation = None
        self._automatic_retry_attempt = 0
        release_shutdown_close = self.release_shutdown_close
        if not self._retire_after_ready_ack():
            return False
        self.cleanup_resources = lambda _generation: False
        self.shutdown_ready = lambda _event: False
        self.finalize_after_ready_ack = lambda _generation: False
        self.release_shutdown_close = lambda _generation: False
        self.shutdown_aborted = lambda _event: False
        try:
            return release_shutdown_close(generation) is True
        except BaseException as error:
            if not _qt_object_is_deleted(self):
                self._log(
                    f"shutdown close release failed: {type(error).__name__}"
                )
            return False

    @pyqtSlot()
    def retry_pending_shutdown(self) -> bool:
        if _qt_object_is_deleted(self) or not self._active:
            return False
        pending_generation = self._cleanup_pending_generation
        if pending_generation is None:
            pending_generation = self._ready_pending_generation
        if self._finalization_attempt_token is not None:
            if pending_generation == self._finalization_attempt_generation:
                self._retry_queued_generation = pending_generation
                timer = self._retry_timer
                if (
                    not _qt_object_is_deleted(timer)
                    and not timer.isActive()
                ):
                    timer.start(0)
                return True
            return False
        token = object()
        self._finalization_attempt_generation = pending_generation
        self._finalization_attempt_token = token
        result = self._run_finalization_attempt(pending_generation, token)
        if _qt_object_is_deleted(self):
            return False
        if self._finalization_attempt_token is token:
            self._finalization_attempt_generation = None
            self._finalization_attempt_token = None
        if result:
            self._retry_queued_generation = None
        return result

    def _run_finalization_attempt(
        self, pending_generation: int | None, token: object
    ) -> bool:
        generation = self._cleanup_pending_generation
        if generation is not None:
            if (
                generation != pending_generation
                or generation != self.model.shutdown_generation
                or self.model.phase is not WorkflowPhase.SHUTDOWN_FLUSHING
            ):
                return False
            try:
                cleaned = self.cleanup_resources(generation)
            except BaseException as error:
                self._log(
                    f"shutdown resource cleanup failed: {type(error).__name__}"
                )
                if _qt_object_is_deleted(self):
                    return False
                self._finalization_attempt_generation = None
                self._finalization_attempt_token = None
                self._schedule_retry()
                return False
            if _qt_object_is_deleted(self):
                return False
            if not self._attempt_is_current(generation, token):
                return False
            if cleaned is False:
                self._finalization_attempt_generation = None
                self._finalization_attempt_token = None
                self._schedule_retry()
                return False
            self._cleanup_pending_generation = None
            self._ready_pending_generation = generation
        generation = self._ready_pending_generation
        if generation is None:
            self._finalization_attempt_generation = None
            self._finalization_attempt_token = None
            return False
        return self._deliver_shutdown_ready(generation, token)

    @pyqtSlot(object)
    def _receive_shutdown_flush_completed(
        self, event: ShutdownFlushCompleted
    ) -> bool:
        if (
            not self._active
            or type(event) is not ShutdownFlushCompleted
            or event.shutdown_generation != self.model.shutdown_generation
            or self.model.phase is not WorkflowPhase.SHUTDOWN_FLUSHING
            or event.shutdown_generation in self._completed_generations
        ):
            return False
        generation = event.shutdown_generation
        if self._cleanup_pending_generation not in {None, generation}:
            return False
        if self._ready_pending_generation not in {None, generation}:
            return False
        if self._ready_pending_generation is None:
            if _qt_object_is_deleted(self._retry_timer):
                return False
            self._retry_timer.stop()
            self._automatic_retry_attempt = 0
            self._cleanup_pending_generation = generation
        QTimer.singleShot(0, self.retry_pending_shutdown)
        return True

    @pyqtSlot(object)
    def handle_shutdown_flush_completed(self, event: ShutdownFlushCompleted) -> bool:
        if (
            not self._active
            or type(event) is not ShutdownFlushCompleted
            or event.shutdown_generation != self.model.shutdown_generation
            or self.model.phase is not WorkflowPhase.SHUTDOWN_FLUSHING
            or event.shutdown_generation in self._completed_generations
        ):
            return False
        generation = event.shutdown_generation
        if self._cleanup_pending_generation not in {None, generation}:
            return False
        if self._ready_pending_generation not in {None, generation}:
            return False
        if self._ready_pending_generation is None:
            if _qt_object_is_deleted(self._retry_timer):
                return False
            self._retry_timer.stop()
            self._automatic_retry_attempt = 0
            self._cleanup_pending_generation = generation
        return self.retry_pending_shutdown()

    @pyqtSlot(object)
    def handle_shutdown_aborted(self, event: ShutdownAborted) -> bool:
        if not self._active or type(event) is not ShutdownAborted:
            return False
        if self._confirmation_generation not in {None, event.shutdown_generation}:
            return False
        getattr(
            self.view,
            "finish_shutdown_confirmation",
            lambda _generation: False,
        )(event.shutdown_generation)
        self._confirmation_generation = None
        if self._waiting_generation == event.shutdown_generation:
            getattr(
                self.view,
                "finish_shutdown_waiting",
                lambda _generation: False,
            )(event.shutdown_generation)
            self._waiting_generation = None
        if self._flush_requested_generation == event.shutdown_generation:
            self._flush_requested_generation = None
        if self._workflow_ready_generation == event.shutdown_generation:
            self._workflow_ready_generation = None
        self.shutdown_aborted(event)
        return True

    def disconnect(self, _lifecycle_request=None) -> bool:
        if not self._active:
            return True
        completed = self._disconnect_steps_completed
        pending_ids = []
        if self._ready_pending_generation is not None:
            pending_ids.append(("shutdown-ready", self._ready_pending_generation))
        abandon = getattr(self.bus, "abandon_workflow_continuations", None)
        if "abandon" not in completed and callable(abandon) and pending_ids:
            try:
                abandon(
                    tuple(pending_ids),
                    owner=self,
                    reason="shutdown coordinator disconnected",
                )
            except BaseException as error:
                self._log(
                    f"shutdown ready abandonment failed: {type(error).__name__}"
                )
                return False
        completed.add("abandon")
        unregister_recipient = getattr(
            self.bus, "unregister_workflow_continuation_recipient", None
        )
        if (
            "recipient" not in completed
            and callable(unregister_recipient)
            and self._shutdown_flush_recipient_name is not None
        ):
            try:
                unregister_recipient(
                    "shutdown-flush-completed",
                    self._shutdown_flush_recipient_name,
                )
            except BaseException as error:
                self._log(
                    f"shutdown flush recipient unregister failed: {type(error).__name__}"
                )
                return False
        completed.add("recipient")
        unregister_owner = getattr(
            self.bus, "unregister_workflow_continuation_lifecycle_owner", None
        )
        if "owner" not in completed and callable(unregister_owner):
            try:
                unregister_owner(self)
            except BaseException as error:
                self._log(
                    f"shutdown lifecycle owner unregister failed: {type(error).__name__}"
                )
                return False
        completed.add("owner")
        if "timer" not in completed:
            try:
                self._retry_timer.stop()
            except BaseException as error:
                self._log(f"shutdown retry timer stop failed: {type(error).__name__}")
                return False
            completed.add("timer")
        while self._connections:
            signal, slot = self._connections[-1]
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            except BaseException as error:
                self._log(
                    f"shutdown coordinator signal disconnect failed: {type(error).__name__}"
                )
                return False
            self._connections.pop()
        completed.add("connections")
        self._active = False
        return True
