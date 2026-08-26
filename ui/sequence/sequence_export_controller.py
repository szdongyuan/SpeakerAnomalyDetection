"""Event-driven controller for per-record and spool-rebuild exports."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable
from uuid import uuid4

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSlot

from ui.sequence.sequence_event_bus import (
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
    WorkflowContinuationRecipientResult,
)
from ui.sequence.sequence_export_model import (
    ExportAttempt,
    ExportJob,
    RecordExportWork,
    SequenceExportModel,
    SpoolRebuildJob,
    SpoolTarget,
)
from ui.sequence.sequence_export_service import (
    ExportTargetFailure,
    SequenceExportService,
)
from ui.sequence.sequence_export_worker import SequenceExportWorker
from ui.sequence.sequence_messages import (
    AnalysisExportPrepared,
    AnalysisExportPreparationFailed,
    CancelExportPreparationRequested,
    BeginShutdownFlushRequested,
    CancelExportRequested,
    ExportCompleted,
    ExportFailed,
    ExportPreparationCancelled,
    ExportRequested,
    ExportRetryAccepted,
    IgnoreExportFailureRequested,
    IgnoreShutdownFlushFailureRequested,
    ManualLabelExportPrepared,
    ManualLabelExportPreparationFailed,
    PrepareAnalysisExportRequested,
    PrepareManualLabelExportRequested,
    RetryExportRequested,
    RetryShutdownFlushRequested,
    ShutdownFlushCompleted,
    ShutdownFlushFailed,
)


def _new_attempt_identifier(job_id: str, number: int) -> str:
    return f"{job_id}-attempt-{number}-{uuid4().hex}"


def _plain_target_result(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _plain_target_result(item) for key, item in value.items()}
    if hasattr(value, "target_type"):
        return {
            "target": str(getattr(value, "target_type", "unknown")),
            "config_name": str(getattr(value, "config_name", "unknown")),
            "message": str(getattr(value, "message", "")),
        }
    if isinstance(value, dict):
        return {key: _plain_target_result(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_target_result(item) for item in value)
    return value


def _bounded_controller_error(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        return "export controller failed"
    return message[:1024] if message else "export controller failed"


@dataclass(slots=True)
class _PendingTerminalPublication:
    identity: tuple[str, str]
    signal: Any
    message: Any
    finalize: Callable[[], None]


class SequenceExportController(QObject):
    """Coordinate immutable jobs and publish attempt-tagged Workflow terminals."""

    def __init__(
        self,
        model: SequenceExportModel,
        view: Any,
        *,
        bus: Any,
        service: SequenceExportService | None = None,
        submit_attempt: Callable[[Any, ExportAttempt], None] | None = None,
        thread_factory: Callable[[], Any] = QThread,
        worker_factory: Callable[..., Any] = SequenceExportWorker,
        attempt_id_factory: Callable[[str, int], str] = _new_attempt_identifier,
        debounce_ms: int = 30_000,
        logger: Any = None,
        parent: QObject | None = None,
        connect_bus: bool = True,
    ) -> None:
        super().__init__(parent)
        if type(model) is not SequenceExportModel:
            raise TypeError("model must be SequenceExportModel")
        if type(debounce_ms) is not int or debounce_ms < 0:
            raise ValueError("debounce_ms must be a non-negative integer")
        self.model = model
        self.view = view
        self.bus = bus
        self.service = service or SequenceExportService(logger=logger)
        self.attempt_id_factory = attempt_id_factory
        self.debounce_ms = debounce_ms
        self.logger = logger
        self._submit_attempt_port = submit_attempt
        self._thread_factory = thread_factory
        self._worker_factory = worker_factory
        self._accept_worker_results = True
        self._connections: list[tuple[Any, Any]] = []
        self._worker_thread: QThread | None = None
        self._worker: SequenceExportWorker | None = None
        self._active_worker_identity: tuple[str, str] | None = None
        self._pending_worker_jobs: deque[tuple[Any, str]] = deque()
        self._owned_thread_handles: dict[Any, Any] = {}
        self._rebuild_jobs: dict[tuple[str, str], SpoolRebuildJob] = {}
        self._failed_rebuild_jobs: dict[
            tuple[str, str], SpoolRebuildJob
        ] = {}
        self._rebuild_failures: dict[tuple[str, str], tuple[Any, ...]] = {}
        self._pending_terminal_publication: (
            _PendingTerminalPublication | None
        ) = None
        self._pending_presentation: tuple[
            str, str, str, Any
        ] | None = None
        self._shutdown_failure_presented_identity: (
            tuple[str, str] | None
        ) = None
        self._shutdown_last_rebuild_identity: tuple[str, str] | None = None
        self._shutdown_completion_retry_attempt = 0
        self._shutdown_completion_retry_limit = 5
        self._shutdown_completion_retry_base_delay_ms = 25
        self._shutdown_completion_retry_max_delay_ms = 400
        self._shutdown_completion_retry_timer = QTimer(self)
        self._shutdown_completion_retry_timer.setSingleShot(True)
        self._shutdown_completion_retry_timer.timeout.connect(
            self._retry_shutdown_completion_publication
        )
        self._formal_shutdown_completion_delivery = False
        self._analysis_preparation_recipient_name = (
            f"export-analysis-preparation:{id(self)}"
        )
        self._manual_preparation_recipient_name = (
            f"export-manual-preparation:{id(self)}"
        )
        self._preparation_cancel_recipient_name = (
            f"export-preparation-cancel:{id(self)}"
        )
        if connect_bus:
            register_owner = getattr(
                self.bus,
                "register_workflow_continuation_lifecycle_owner",
                None,
            )
            deliver = getattr(self.bus, "deliver_workflow_continuation", None)
            deliver_outcome = getattr(
                self.bus, "deliver_workflow_continuation_outcome", None
            )
            if callable(register_owner) and (
                callable(deliver_outcome) or callable(deliver)
            ):
                register_owner(self)
                self._formal_shutdown_completion_delivery = True
                register_recipient = getattr(
                    self.bus,
                    "register_workflow_continuation_recipient",
                    None,
                )
                if callable(register_recipient):
                    register_recipient(
                        "analysis-export-prepare",
                        self._analysis_preparation_recipient_name,
                        self.handle_prepare_analysis_export,
                        owner=self,
                    )
                    register_recipient(
                        "manual-label-export-prepare",
                        self._manual_preparation_recipient_name,
                        self.handle_prepare_manual_label_export,
                        owner=self,
                    )
                    register_recipient(
                        "export-preparation-cancel",
                        self._preparation_cancel_recipient_name,
                        self.handle_cancel_export_preparation,
                        owner=self,
                    )
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.handle_rebuild_debounce)
        if connect_bus:
            self._wire_bus()
        self._wire_view()

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def retain_result_snapshot(self, record_id: str, snapshot: Any) -> bool:
        return self.model.retain_result_snapshot(record_id, snapshot)

    def build_labeled_result(self, record_id: str, label: str) -> dict[str, Any]:
        source = self.model.retained_result_snapshot(record_id)
        return self.service.build_labeled_result(record_id, label, source)

    def _deliver_preparation_response(
        self, kind: str, message: Any
    ) -> WorkflowContinuationDeliveryOutcome:
        outcome_type = WorkflowContinuationDeliveryOutcome
        ack = WorkflowContinuationDeliveryStatus.ACK
        retryable = WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
        detailed_delivery = getattr(
            self.bus, "deliver_workflow_continuation_outcome", None
        )
        if callable(detailed_delivery) and self._formal_shutdown_completion_delivery:
            outcome = detailed_delivery(
                (kind, message.request_id, message.workflow_generation),
                kind,
                message,
                owner=self,
            )
            if type(outcome) is WorkflowContinuationDeliveryOutcome:
                return outcome
            return outcome_type(
                retryable,
                "preparation response dispatcher returned an invalid outcome",
            )
        delivery = getattr(self.bus, "deliver_workflow_continuation", None)
        if callable(delivery) and self._formal_shutdown_completion_delivery:
            acknowledged = delivery(
                (kind, message.request_id, message.workflow_generation),
                kind,
                message,
                owner=self,
            ) is True
            return outcome_type(
                ack if acknowledged else retryable,
                "" if acknowledged else "preparation response acknowledgement pending",
            )
        signal = getattr(self.bus.events, kind.replace("-", "_"), None)
        if signal is None:
            return outcome_type(
                retryable, "preparation response signal is unavailable"
            )
        acknowledged = self._safe_emit(signal, message)
        return outcome_type(
            ack if acknowledged else retryable,
            "" if acknowledged else "preparation response acknowledgement pending",
        )

    @staticmethod
    def _preparation_recipient_result(
        outcome: WorkflowContinuationDeliveryOutcome,
    ) -> WorkflowContinuationRecipientResult:
        return {
            WorkflowContinuationDeliveryStatus.ACK: (
                WorkflowContinuationRecipientResult.ACK
            ),
            WorkflowContinuationDeliveryStatus.RETRYABLE_NACK: (
                WorkflowContinuationRecipientResult.RETRYABLE_NACK
            ),
            WorkflowContinuationDeliveryStatus.PERMANENT_REJECT: (
                WorkflowContinuationRecipientResult.PERMANENT_REJECT
            ),
        }[outcome.status]

    @staticmethod
    def _preparation_response_ack(
        outcome: WorkflowContinuationDeliveryOutcome,
    ) -> bool:
        return outcome.status is WorkflowContinuationDeliveryStatus.ACK

    @pyqtSlot(object)
    def handle_prepare_analysis_export(
        self, request: PrepareAnalysisExportRequested
    ) -> bool | WorkflowContinuationRecipientResult:
        if type(request) is not PrepareAnalysisExportRequested:
            return False
        if self.model.is_export_preparation_cancelled(request):
            return False
        cached = self.model.prepared_export_response(request)
        if cached is False:
            return False
        response = cached
        if response is None:
            try:
                prepared = self.service.prepare_analysis_export(
                    request.record_id,
                    request.result_snapshot,
                    request.analysis_configuration,
                )
                if type(prepared) is not tuple or len(prepared) != 2:
                    raise ValueError(
                        "analysis export preparation returned an invalid result"
                    )
                snapshot, targets = prepared
                if not self.model.retain_result_snapshot(
                    request.record_id, snapshot
                ):
                    raise ValueError(
                        "analysis export preparation could not retain its result"
                    )
                response = AnalysisExportPrepared(
                    request.request_id,
                    request.analysis_id,
                    request.source_id,
                    request.record_id,
                    request.workflow_generation,
                    snapshot,
                    targets,
                )
            except BaseException as error:
                reason = _bounded_controller_error(error)
                self._log(
                    "warning",
                    f"analysis export preparation failed: {reason}",
                )
                response = AnalysisExportPreparationFailed(
                    request.request_id,
                    request.analysis_id,
                    request.source_id,
                    request.record_id,
                    request.workflow_generation,
                    reason,
                )
            if not self.model.remember_export_preparation(request, response):
                return False
        if type(response) is AnalysisExportPrepared:
            outcome = self._deliver_preparation_response(
                "analysis-export-prepared", response
            )
            if (
                outcome.status
                is not WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
            ):
                return self._preparation_recipient_result(outcome)
            failure = AnalysisExportPreparationFailed(
                request.request_id,
                request.analysis_id,
                request.source_id,
                request.record_id,
                request.workflow_generation,
                "analysis export prepared response was permanently rejected",
            )
            if not self.model.replace_export_preparation_response(
                request, response, failure
            ):
                return WorkflowContinuationRecipientResult.PERMANENT_REJECT
            response = failure
            self._log("warning", failure.reason)
        outcome = self._deliver_preparation_response(
            "analysis-export-preparation-failed", response
        )
        return self._preparation_recipient_result(outcome)

    @pyqtSlot(object)
    def handle_prepare_manual_label_export(
        self, request: PrepareManualLabelExportRequested
    ) -> bool:
        if type(request) is not PrepareManualLabelExportRequested:
            return False
        if self.model.is_export_preparation_cancelled(request):
            return False
        cached = self.model.prepared_export_response(request)
        if cached is False:
            return False
        response = cached
        if response is None:
            try:
                source = self.model.retained_result_snapshot(request.record_id)
                prepared = self.service.prepare_manual_label_export(
                    request.record_id, request.label, source
                )
                if type(prepared) is not tuple or len(prepared) != 2:
                    raise ValueError(
                        "manual label export preparation returned an invalid result"
                    )
                snapshot, targets = prepared
                response = ManualLabelExportPrepared(
                    request.request_id,
                    request.command_id,
                    request.record_id,
                    request.label,
                    request.workflow_generation,
                    snapshot,
                    targets,
                )
            except BaseException as error:
                reason = _bounded_controller_error(error)
                self._log(
                    "warning",
                    f"manual label export preparation failed: {reason}",
                )
                response = ManualLabelExportPreparationFailed(
                    request.request_id,
                    request.command_id,
                    request.record_id,
                    request.label,
                    request.workflow_generation,
                    reason,
                )
            if not self.model.remember_export_preparation(request, response):
                return False
        kind = (
            "manual-label-export-prepared"
            if type(response) is ManualLabelExportPrepared
            else "manual-label-export-preparation-failed"
        )
        return self._preparation_response_ack(
            self._deliver_preparation_response(kind, response)
        )

    @pyqtSlot(object)
    def handle_cancel_export_preparation(
        self, request: CancelExportPreparationRequested
    ) -> bool:
        if type(request) is not CancelExportPreparationRequested:
            return False
        if not self.model.cancel_export_preparation(
            request.request_id, request.workflow_generation
        ):
            return False
        terminal = ExportPreparationCancelled(
            request.request_id, request.workflow_generation
        )
        return self._preparation_response_ack(
            self._deliver_preparation_response(
                "export-preparation-cancelled", terminal
            )
        )

    @staticmethod
    def export_targets(result_snapshot: Any) -> tuple[Any, ...]:
        if not isinstance(result_snapshot, Mapping):
            return ()
        targets = result_snapshot.get("export_targets", ())
        return tuple(targets) if isinstance(targets, (tuple, list)) else ()

    @classmethod
    def has_export_targets(cls, result_snapshot: Any) -> bool:
        return bool(cls.export_targets(result_snapshot))

    def schedule_spool_configurations(self, configurations: Any) -> int:
        """Normalize legacy spool tuples at the Export ownership boundary."""
        targets = []
        for config_name, configuration, file_path, spool_dir in tuple(
            configurations or ()
        ):
            try:
                targets.append(
                    SpoolTarget.create(
                        str(config_name),
                        configuration,
                        str(file_path),
                        str(spool_dir),
                    )
                )
            except (TypeError, ValueError) as error:
                self._log(
                    "error",
                    f"excel_spool_schedule_path_error[{config_name}]: {error}",
                )
        return self.schedule_spool_targets(targets)

    def _wire_bus(self) -> None:
        commands = self.bus.commands
        for signal, slot in (
            (commands.export_requested, self.handle_export_requested),
            (commands.retry_export_requested, self.handle_retry_requested),
            (
                commands.ignore_export_failure_requested,
                self.handle_ignore_requested,
            ),
            (commands.cancel_export_requested, self.handle_cancel_requested),
            (
                commands.begin_shutdown_flush_requested,
                self.handle_begin_shutdown_flush,
            ),
            (
                commands.retry_shutdown_flush_requested,
                self.handle_retry_shutdown_flush,
            ),
            (
                commands.ignore_shutdown_flush_failure_requested,
                self.handle_ignore_shutdown_flush_failure,
            ),
        ):
            signal.connect(slot, Qt.QueuedConnection)
            self._connections.append((signal, slot))

    def _wire_view(self) -> None:
        retry_signal = getattr(self.view, "retry_requested", None)
        ignore_signal = getattr(self.view, "ignore_requested", None)
        if retry_signal is not None and hasattr(retry_signal, "connect"):
            retry_signal.connect(self.bus.commands.retry_export_requested.emit)
        if ignore_signal is not None and hasattr(ignore_signal, "connect"):
            ignore_signal.connect(
                self.bus.commands.ignore_export_failure_requested.emit
            )
        shutdown_retry = getattr(self.view, "shutdown_retry_requested", None)
        shutdown_ignore = getattr(self.view, "shutdown_ignore_requested", None)
        retry_command = getattr(
            self.bus.commands, "retry_shutdown_flush_requested", None
        )
        ignore_command = getattr(
            self.bus.commands, "ignore_shutdown_flush_failure_requested", None
        )
        if (
            shutdown_retry is not None
            and hasattr(shutdown_retry, "connect")
            and retry_command is not None
            and hasattr(retry_command, "emit")
        ):
            shutdown_retry.connect(
                retry_command.emit
            )
        if (
            shutdown_ignore is not None
            and hasattr(shutdown_ignore, "connect")
            and ignore_command is not None
            and hasattr(ignore_command, "emit")
        ):
            shutdown_ignore.connect(
                ignore_command.emit
            )

    def _safe_view_call(self, method: str, *args: Any) -> bool:
        try:
            callback = getattr(self.view, method)
            result = callback(*args)
            return result is not False
        except BaseException as error:
            self._log(
                "warning",
                f"export view {method} failed: {_bounded_controller_error(error)}",
            )
            return False

    def _safe_emit(self, signal: Any, message: Any) -> bool:
        try:
            signal.emit(message)
            return True
        except BaseException as error:
            self._log(
                "error",
                f"export terminal publication failed: {_bounded_controller_error(error)}",
            )
            return False

    def _deliver_terminal(
        self, signal: Any, message: Any, identity: tuple[str, str]
    ) -> bool:
        has_recipients = getattr(
            self.bus, "has_export_terminal_recipients", None
        )
        delivery = getattr(self.bus, "deliver_export_terminal", None)
        if (
            callable(has_recipients)
            and has_recipients()
            and callable(delivery)
        ):
            delivery_id = (
                type(message).__name__,
                identity[0],
                identity[1],
            )
            try:
                return delivery(delivery_id, message) is True
            except BaseException as error:
                self._log(
                    "error",
                    "export terminal delivery failed: "
                    f"{_bounded_controller_error(error)}",
                )
                return False
        return self._safe_emit(signal, message)

    def _present_failure(
        self, job_id: str, attempt_id: str, failures: Any
    ) -> bool:
        if self._safe_view_call(
            "show_failure", job_id, attempt_id, failures
        ):
            if self.presentation_pending_identity == (job_id, attempt_id):
                self._pending_presentation = None
            return True
        self._pending_presentation = (
            job_id, attempt_id, "failure", failures
        )
        return False

    def _present_publication_failure(
        self, job_id: str, attempt_id: str, failures: Any
    ) -> bool:
        method = (
            "show_publication_failure"
            if hasattr(self.view, "show_publication_failure")
            else "show_failure"
        )
        if self._safe_view_call(method, job_id, attempt_id, failures):
            if self.presentation_pending_identity == (job_id, attempt_id):
                self._pending_presentation = None
            return True
        self._pending_presentation = (
            job_id, attempt_id, "publication", failures
        )
        return False

    @property
    def presentation_pending_identity(self) -> tuple[str, str] | None:
        pending = self._pending_presentation
        return None if pending is None else pending[:2]

    def retry_pending_presentation(
        self, job_id: str, attempt_id: str
    ) -> bool:
        pending = self._pending_presentation
        if pending is None or pending[:2] != (job_id, attempt_id):
            return False
        _job_id, _attempt_id, kind, payload = pending
        self._pending_presentation = None
        self._safe_view_call(
            "prepare_failure_identity", job_id, attempt_id
        )
        if kind == "publication":
            return self._present_publication_failure(
                job_id, attempt_id, payload
            )
        return self._present_failure(job_id, attempt_id, payload)

    def _deliver_retry_ack(self, message: ExportRetryAccepted) -> bool:
        delivery = getattr(
            self.bus, "deliver_export_retry_accepted", None
        )
        if callable(delivery):
            try:
                accepted = delivery(message) is True
            except BaseException as error:
                self._log(
                    "error",
                    "export retry acknowledgement failed: "
                    f"{_bounded_controller_error(error)}",
                )
                return False
            if not accepted:
                return False
            return True
        signal = getattr(
            getattr(self.bus, "events", None),
            "export_retry_accepted",
            None,
        )
        if signal is None:
            # Compatibility fakes without a Workflow collaborator have no ack
            # recipient; installing the local attempt is sufficient.
            return True
        try:
            signal.emit(message)
            return True
        except BaseException:
            return False

    @property
    def pending_terminal_publication_identity(
        self,
    ) -> tuple[str, str] | None:
        pending = self._pending_terminal_publication
        return None if pending is None else pending.identity

    def _publish_terminal(
        self,
        signal: Any,
        message: Any,
        identity: tuple[str, str],
        finalize: Callable[[], None],
    ) -> bool:
        if self._pending_terminal_publication is not None:
            return False
        pending = _PendingTerminalPublication(
            identity, signal, message, finalize
        )
        if not self._deliver_terminal(signal, message, identity):
            self._pending_terminal_publication = pending
            self._present_publication_failure(
                *identity,
                (
                    ExportTargetFailure(
                        "publication",
                        "event-bus",
                        "export terminal publication failed",
                    ),
                ),
            )
            return False
        finalize()
        return True

    def retry_pending_terminal_publication(
        self, job_id: str, attempt_id: str
    ) -> bool:
        pending = self._pending_terminal_publication
        if pending is None or pending.identity != (job_id, attempt_id):
            return False
        if not self._deliver_terminal(
            pending.signal, pending.message, pending.identity
        ):
            self._present_publication_failure(
                job_id,
                attempt_id,
                (
                    ExportTargetFailure(
                        "publication",
                        "event-bus",
                        "export terminal publication failed",
                    ),
                ),
            )
            return False
        self._pending_terminal_publication = None
        pending.finalize()
        return True

    def _fallback_active_identity(self) -> tuple[str, str] | None:
        if self._active_worker_identity is not None:
            return self._active_worker_identity
        attempt = self.model.active_record_attempt
        if attempt is not None:
            return attempt.job_id, attempt.attempt_id
        if len(self._rebuild_jobs) == 1:
            return next(iter(self._rebuild_jobs))
        return None

    def _fail_current_attempt(
        self,
        error: BaseException,
        identity: tuple[str, str] | None = None,
    ) -> bool:
        identity = identity or self._fallback_active_identity()
        if identity is None:
            self._log(
                "error",
                f"unowned export controller failure: {_bounded_controller_error(error)}",
            )
            return False
        job_id, attempt_id = identity
        failure = ExportTargetFailure(
            "controller", "controller", _bounded_controller_error(error)
        )
        failures = (failure,)
        rebuild = self._rebuild_jobs.get(identity)
        if rebuild is None:
            rebuild = self._failed_rebuild_jobs.get(identity)
        if rebuild is not None:
            sealed = self.model.fail_rebuild_boundary(
                job_id, attempt_id, failures
            )
            if sealed is None:
                return False
            self._rebuild_jobs.pop(identity, None)
            self._rebuild_failures[rebuild.target.key] = failures
            self._failed_rebuild_jobs[identity] = rebuild
            if self.model.shutdown_flush_pending:
                self._advance_shutdown_flush()
            else:
                self._present_failure(job_id, attempt_id, failures)
            return True
        attempt = self.model.active_record_attempt
        job = self.model.active_record_job
        if (
            attempt is None
            or job is None
            or not self.model.fail_record_attempt(
                job_id, attempt_id, failures
            )
        ):
            return False
        message = ExportFailed(
                job_id,
                attempt_id,
                job.record_id,
                tuple(_plain_target_result(item) for item in failures),
        )
        return self._publish_terminal(
            self.bus.events.export_failed,
            message,
            (job_id, attempt_id),
            lambda: self._present_failure(job_id, attempt_id, failures),
        )

    @pyqtSlot(object)
    def handle_export_requested(self, command: ExportRequested) -> bool:
        if type(command) is not ExportRequested:
            return False
        try:
            self.model.enqueue_record(command)
            self._start_next_record_job()
        except (TypeError, ValueError):
            return False
        except BaseException as error:
            self._fail_current_attempt(error)
            return True
        return True

    def _start_next_record_job(self) -> bool:
        if self._failed_rebuild_jobs or self._pending_terminal_publication:
            return False
        job = self.model.begin_next_record_job()
        if job is None:
            return False
        try:
            attempt = self.model.begin_record_attempt(self.attempt_id_factory)
        except BaseException as error:
            attempt = self.model.begin_record_recovery_attempt()
            self._safe_view_call(
                "prepare_failure_identity", attempt.job_id, attempt.attempt_id
            )
            self._fail_current_attempt(
                error, (attempt.job_id, attempt.attempt_id)
            )
            return False
        self._submit(self.model.active_record_work(), attempt.attempt_id)
        return True

    def _submit(
        self,
        job: ExportJob | RecordExportWork | SpoolRebuildJob,
        attempt_id: str,
    ) -> None:
        if (
            self._failed_rebuild_jobs
            or self._pending_terminal_publication is not None
            or (
                type(job) is SpoolRebuildJob
                and self.model.record_failure is not None
            )
        ):
            self._pending_worker_jobs.append((job, attempt_id))
            return
        identity = (job.job_id, attempt_id)
        if not self._safe_view_call("show_progress", *identity):
            self._fail_current_attempt(
                RuntimeError("export progress presentation failed"),
                identity,
            )
            return
        if self.presentation_pending_identity == identity:
            self._pending_presentation = None
        if self._submit_attempt_port is not None:
            attempt_number = getattr(job, "attempt_number", None)
            if attempt_number is None:
                active_attempt = self.model.active_record_attempt
                attempt_number = (
                    active_attempt.attempt_number
                    if active_attempt is not None
                    and active_attempt.job_id == job.job_id
                    and active_attempt.attempt_id == attempt_id
                    else 1
                )
            try:
                self._submit_attempt_port(
                    job,
                    ExportAttempt(job.job_id, attempt_id, int(attempt_number)),
                )
            except BaseException as error:
                self._fail_current_attempt(error, (job.job_id, attempt_id))
            return
        if self._active_worker_identity is not None:
            self._pending_worker_jobs.append((job, attempt_id))
            return
        self._start_worker(job, attempt_id, progress_presented=True)

    def _execute_job(self, job: Any, attempt_id: str) -> Any:
        if isinstance(job, (ExportJob, RecordExportWork)):
            return self.service.execute_record_job(job, attempt_id)
        if type(job) is SpoolRebuildJob:
            return self.service.execute_rebuild_job(job)
        raise TypeError("unsupported export worker job")

    def _start_worker(
        self,
        job: Any,
        attempt_id: str,
        *,
        progress_presented: bool = False,
    ) -> None:
        if not self._accept_worker_results:
            return
        identity = (job.job_id, attempt_id)
        if not progress_presented:
            if not self._safe_view_call("show_progress", *identity):
                self._fail_current_attempt(
                    RuntimeError("export progress presentation failed"),
                    identity,
                )
                return
        # Threads intentionally have no QObject parent. This controller owns
        # their Python handles until each real ``finished`` signal arrives, so
        # teardown never converts a still-running external call into success.
        thread = None
        worker = None
        self._active_worker_identity = identity
        try:
            thread = self._thread_factory()
            worker = self._worker_factory(
                job,
                attempt_id,
                execute=self._execute_job,
                validate_dirty_checkpoint=getattr(
                    self.service, "validate_dirty_checkpoint", None
                ),
            )
            self._owned_thread_handles[thread] = worker
            self._worker_thread = thread
            self._worker = worker
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.completed.connect(self.handle_worker_completed)
            worker.failed.connect(self.handle_worker_failed)
            worker.finished.connect(thread.quit, Qt.DirectConnection)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(
                lambda thread=thread, worker=worker: self._worker_finished(
                    thread, worker
                )
            )
            thread.finished.connect(thread.deleteLater)
            thread.start()
        except BaseException as error:
            running = False
            if thread is not None:
                try:
                    running = bool(thread.isRunning())
                except BaseException:
                    running = False
            if not running:
                if thread is not None:
                    self._owned_thread_handles.pop(thread, None)
                    try:
                        thread.deleteLater()
                    except BaseException:
                        pass
                if self._worker_thread is thread:
                    self._worker_thread = None
                    self._worker = None
                    self._active_worker_identity = None
            self._fail_current_attempt(error, identity)

    def _worker_finished(self, thread: Any, worker: Any) -> None:
        self._owned_thread_handles.pop(thread, None)
        terminal_pending = False
        if self._worker_thread is thread:
            identity = self._active_worker_identity
            terminal_pending = bool(
                identity is not None
                and (
                    identity in self._rebuild_jobs
                    or self.model.accept_worker_terminal(*identity)
                )
            )
            self._worker_thread = None
            self._worker = None
            self._active_worker_identity = None
        if terminal_pending:
            # A worker's direct lifecycle signal may stop its QThread before
            # the queued business terminal reaches this controller. The
            # terminal handler will resume the queue after sealing the model.
            return
        if (
            self._accept_worker_results
            and not self._failed_rebuild_jobs
            and self._pending_terminal_publication is None
            and self.model.record_failure is None
            and self._pending_worker_jobs
        ):
            job, attempt_id = self._pending_worker_jobs.popleft()
            self._start_worker(job, attempt_id)
        self._advance_shutdown_flush()

    @staticmethod
    def _dirty_targets(outcome: Any) -> tuple[SpoolTarget, ...]:
        return tuple(
            target
            for target in tuple(getattr(outcome, "dirty_targets", ()) or ())
            if type(target) is SpoolTarget
        )

    def _record_dirty_targets(self, outcome: Any) -> None:
        dirty = self._dirty_targets(outcome)
        for target in dirty:
            self.model.mark_target_dirty(target)
        if dirty:
            self._debounce_timer.start(self.debounce_ms)

    def _record_terminal_matches(self, outcome: Any) -> bool:
        return self.model.accept_worker_terminal(
            getattr(outcome, "job_id", None),
            getattr(outcome, "attempt_id", None),
        )

    @pyqtSlot(object)
    def handle_worker_completed(self, outcome: Any) -> bool:
        try:
            return self._handle_worker_completed(outcome)
        except BaseException as error:
            self._fail_current_attempt(
                error,
                (
                    getattr(outcome, "job_id", None),
                    getattr(outcome, "attempt_id", None),
                ),
            )
            return False

    def _handle_worker_completed(self, outcome: Any) -> bool:
        if not self._accept_worker_results:
            return False
        identity = (
            getattr(outcome, "job_id", None),
            getattr(outcome, "attempt_id", None),
        )
        rebuild = self._rebuild_jobs.get(identity)
        if rebuild is not None:
            follow_up = self.model.complete_rebuild(
                rebuild.job_id,
                rebuild.attempt_id,
                succeeded=True,
            )
            self._rebuild_jobs.pop(identity, None)
            self._rebuild_failures.pop(rebuild.target.key, None)
            self._failed_rebuild_jobs.pop(identity, None)
            self._safe_view_call("finish", *identity)
            if follow_up is not None:
                self._queue_rebuild(follow_up)
            self._resume_pending_worker()
            self._start_next_record_job()
            self._advance_shutdown_flush()
            return True
        if not self._record_terminal_matches(outcome):
            self._log("warning", f"ignored stale export completion {identity!r}")
            return False
        job_id, attempt_id = identity
        record_id = self.model.active_record_job.record_id
        target_results = tuple(getattr(outcome, "target_results", ()) or ())
        dirty_targets = self._dirty_targets(outcome)
        completed_indices = tuple(
            getattr(outcome, "completed_target_indices", ()) or ()
        )
        if not self.model.complete_record_attempt(
            job_id,
            attempt_id,
            completed_indices,
            target_results,
            dirty_targets,
        ):
            return False
        message = ExportCompleted(
            job_id,
            attempt_id,
            record_id,
            tuple(
                _plain_target_result(item)
                for item in self.model.record_terminal_results()
            ),
        )
        for target in dirty_targets:
            self.model.mark_target_dirty(target)
        if dirty_targets:
            self._debounce_timer.start(self.debounce_ms)
        def finalize() -> None:
            self._safe_view_call("finish", job_id, attempt_id)
            self.model.complete_record_job(job_id, attempt_id)
            self._resume_pending_worker()
            self.handle_rebuild_debounce()
            self._start_next_record_job()
            self._advance_shutdown_flush()

        return self._publish_terminal(
            self.bus.events.export_completed,
            message,
            (job_id, attempt_id),
            finalize,
        )

    @pyqtSlot(object)
    def handle_worker_failed(self, outcome: Any) -> bool:
        try:
            return self._handle_worker_failed(outcome)
        except BaseException as error:
            self._fail_current_attempt(
                error,
                (
                    getattr(outcome, "job_id", None),
                    getattr(outcome, "attempt_id", None),
                ),
            )
            return False

    def _handle_worker_failed(self, outcome: Any) -> bool:
        if not self._accept_worker_results:
            return False
        identity = (
            getattr(outcome, "job_id", None),
            getattr(outcome, "attempt_id", None),
        )
        rebuild = self._rebuild_jobs.get(identity)
        if rebuild is not None:
            failures = tuple(getattr(outcome, "failures", ()) or ())
            self.model.complete_rebuild(
                rebuild.job_id,
                rebuild.attempt_id,
                succeeded=False,
                failure=failures,
            )
            self._rebuild_jobs.pop(identity, None)
            self._rebuild_failures[rebuild.target.key] = failures
            self._failed_rebuild_jobs[identity] = rebuild
            if not self.model.shutdown_flush_pending:
                self._present_failure(
                    rebuild.job_id, rebuild.attempt_id, failures
                )
            else:
                self._advance_shutdown_flush()
            self._log(
                "warning",
                f"spool rebuild failed[{rebuild.target.config_name}]: {failures!r}",
            )
            return True
        if not self._record_terminal_matches(outcome):
            self._log("warning", f"ignored stale export failure {identity!r}")
            return False
        failures = tuple(getattr(outcome, "failures", ()) or ())
        plain_failures = tuple(
            _plain_target_result(item) for item in failures
        )
        dirty_targets = self._dirty_targets(outcome)
        completed_indices = tuple(
            getattr(outcome, "completed_target_indices", ()) or ()
        )
        failed_indices = tuple(
            getattr(outcome, "failed_target_indices", ()) or ()
        )
        target_results = tuple(getattr(outcome, "target_results", ()) or ())
        job_id, attempt_id = identity
        record_id = self.model.active_record_job.record_id
        message = ExportFailed(
            job_id,
            attempt_id,
            record_id,
            plain_failures,
        )
        if not self.model.fail_record_attempt(
            job_id,
            attempt_id,
            failures,
            completed_indices,
            target_results,
            failed_indices,
            dirty_targets,
        ):
            return False
        for target in dirty_targets:
            self.model.mark_target_dirty(target)
        if dirty_targets:
            self._debounce_timer.start(self.debounce_ms)
        return self._publish_terminal(
            self.bus.events.export_failed,
            message,
            (job_id, attempt_id),
            lambda: self._present_failure(job_id, attempt_id, failures),
        )

    @pyqtSlot(object)
    def handle_retry_requested(self, command: RetryExportRequested) -> bool:
        try:
            return self._handle_retry_requested(command)
        except BaseException as error:
            self._log(
                "error",
                f"export retry request failed: {_bounded_controller_error(error)}",
            )
            return False

    def _handle_retry_requested(self, command: RetryExportRequested) -> bool:
        if type(command) is not RetryExportRequested:
            return False
        if self.presentation_pending_identity == (
            command.job_id,
            command.attempt_id,
        ):
            return self.retry_pending_presentation(
                command.job_id, command.attempt_id
            )
        if self.pending_terminal_publication_identity == (
            command.job_id,
            command.attempt_id,
        ):
            return self.retry_pending_terminal_publication(
                command.job_id, command.attempt_id
            )
        try:
            previous_attempt = self.model.active_record_attempt
            previous_failure = self.model.record_failure
            attempt = self.model.retry_record_attempt(
                command.job_id,
                command.attempt_id,
                self.attempt_id_factory,
            )
        except BaseException as error:
            active = self.model.active_record_attempt
            if (
                active is not None
                and active.job_id == command.job_id
                and active.attempt_id == command.attempt_id
                and self.model.record_failure is not None
            ):
                self._log(
                    "error",
                    "export retry identifier allocation failed: "
                    f"{_bounded_controller_error(error)}",
                )
                self._present_failure(
                    command.job_id,
                    command.attempt_id,
                    self.model.record_failure,
                )
            return False
        if attempt is not None:
            acknowledgement = ExportRetryAccepted(
                attempt.job_id,
                command.attempt_id,
                attempt.attempt_id,
                attempt.attempt_number,
            )
            if not self._deliver_retry_ack(acknowledgement):
                if previous_attempt is not None:
                    self.model.rollback_record_retry(
                        previous_attempt,
                        attempt,
                        previous_failure,
                    )
                    self._present_failure(
                        previous_attempt.job_id,
                        previous_attempt.attempt_id,
                        previous_failure,
                    )
                return False
            self._submit(
                self.model.active_record_work(), attempt.attempt_id
            )
            return True
        rebuild = self.model.retry_rebuild(
            command.job_id, command.attempt_id
        )
        if rebuild is None:
            return False
        failed = self._failed_rebuild_jobs.pop(
            (command.job_id, command.attempt_id), None
        )
        if failed is not None:
            self._rebuild_failures.pop(failed.target.key, None)
        self._rebuild_jobs[(rebuild.job_id, rebuild.attempt_id)] = rebuild
        self._submit(rebuild, rebuild.attempt_id)
        return True

    @pyqtSlot(object)
    def handle_ignore_requested(
        self, command: IgnoreExportFailureRequested
    ) -> bool:
        try:
            return self._handle_ignore_requested(command)
        except BaseException as error:
            self._log(
                "error",
                f"export ignore request failed: {_bounded_controller_error(error)}",
            )
            return False

    def _handle_ignore_requested(
        self, command: IgnoreExportFailureRequested
    ) -> bool:
        if type(command) is not IgnoreExportFailureRequested:
            return False
        if self._pending_terminal_publication is not None:
            return False
        job = self.model.ignore_record_failure(
            command.job_id, command.attempt_id
        )
        if job is None:
            rebuild = self._failed_rebuild_jobs.get(
                (command.job_id, command.attempt_id)
            )
            if rebuild is None or not self.model.ignore_rebuild_failure(
                command.job_id, command.attempt_id
            ):
                return False
            self._failed_rebuild_jobs.pop(
                (command.job_id, command.attempt_id), None
            )
            self._rebuild_failures.pop(rebuild.target.key, None)
            self._safe_view_call(
                "finish", command.job_id, command.attempt_id
            )
            follow_up = self.model.begin_rebuild(rebuild.target.key)
            if follow_up is not None:
                self._queue_rebuild(follow_up)
            self._resume_pending_worker()
            self._start_next_record_job()
            return True
        failures = self.model.record_failure
        completed = ExportCompleted(
            job.job_id,
            command.attempt_id,
            job.record_id,
            tuple(
                _plain_target_result(item)
                for item in self.model.record_terminal_results(ignored=True)
            ),
        )
        def finalize() -> None:
            self._safe_view_call(
                "finish", command.job_id, command.attempt_id
            )
            self.model.complete_record_job(job.job_id, command.attempt_id)
            self._resume_pending_worker()
            self.handle_rebuild_debounce()
            self._start_next_record_job()
            self._advance_shutdown_flush()

        return self._publish_terminal(
            self.bus.events.export_completed,
            completed,
            (command.job_id, command.attempt_id),
            finalize,
        )

    def _resume_pending_worker(self) -> None:
        if (
            self._accept_worker_results
            and self._active_worker_identity is None
            and not self._failed_rebuild_jobs
            and self._pending_terminal_publication is None
            and self.model.record_failure is None
            and self._pending_worker_jobs
        ):
            job, attempt_id = self._pending_worker_jobs.popleft()
            self._submit(job, attempt_id)

    @pyqtSlot(object)
    def handle_cancel_requested(self, command: CancelExportRequested) -> bool:
        if type(command) is not CancelExportRequested:
            return False
        if not self.model.request_record_cancel(command.job_id, command.reason):
            return False
        attempt = self.model.active_record_attempt
        if attempt is not None and self.model.record_failure is not None:
            job = self.model.active_record_job
            self._safe_emit(
                self.bus.events.export_failed,
                ExportFailed(
                    job.job_id,
                    attempt.attempt_id,
                    job.record_id,
                    tuple(
                        _plain_target_result(item)
                        for item in tuple(self.model.record_failure or ())
                    ),
                ),
            )
        return True

    def schedule_spool_targets(self, targets: Any) -> bool:
        accepted = False
        for target in tuple(targets or ()):
            if type(target) is not SpoolTarget:
                continue
            self.model.mark_target_dirty(target)
            accepted = True
        if accepted:
            self._debounce_timer.start(self.debounce_ms)
        return accepted

    @pyqtSlot()
    def handle_rebuild_debounce(self) -> None:
        if (
            self.model.record_failure is not None
            or self._pending_terminal_publication is not None
            or self._pending_presentation is not None
        ):
            return
        for key in self.model.dirty_target_keys():
            job = self.model.begin_rebuild(key)
            if job is not None:
                self._queue_rebuild(job)

    def _queue_rebuild(self, job: SpoolRebuildJob) -> None:
        identity = (job.job_id, job.attempt_id)
        self._rebuild_jobs[identity] = job
        if (
            self.model.shutdown_flush_pending
            and self.model.shutdown_flush_final_started
        ):
            self._shutdown_last_rebuild_identity = identity
        self._submit(job, job.attempt_id)

    def has_active_jobs(self) -> bool:
        return bool(
            self.model.has_active_work()
            or self._active_worker_identity is not None
            or self._pending_worker_jobs
            or self._pending_terminal_publication is not None
            or self._pending_presentation is not None
        )

    def pending_rebuild_failures(self) -> tuple[tuple[SpoolTarget, tuple[Any, ...]], ...]:
        return tuple(
            (job.target, self._rebuild_failures.get(job.target.key, ()))
            for job in self._failed_rebuild_jobs.values()
            if job.target.key in self._rebuild_failures
        )

    def begin_shutdown_flush(self, shutdown_generation: int) -> bool:
        if type(shutdown_generation) is not int or shutdown_generation < 0:
            return False
        if self.model.shutdown_flush_pending:
            if self.model.shutdown_flush_generation != shutdown_generation:
                return False
            self._shutdown_completion_retry_timer.stop()
            self._shutdown_completion_retry_attempt = 0
            self._advance_shutdown_flush()
            return True
        if self.model.shutdown_flush_terminal:
            return self.model.shutdown_flush_generation == shutdown_generation
        self.model.shutdown_flush_pending = True
        self.model.shutdown_flush_generation = shutdown_generation
        self.model.shutdown_flush_failures = ()
        self.model.shutdown_flush_final_started = False
        self.model.shutdown_flush_failure_identity = None
        self.model.shutdown_flush_completion_identity = None
        self._shutdown_failure_presented_identity = None
        self._shutdown_last_rebuild_identity = None
        self._shutdown_completion_retry_attempt = 0
        self._shutdown_completion_retry_timer.stop()
        self._debounce_timer.stop()
        self._advance_shutdown_flush()
        return True

    @pyqtSlot(object)
    def handle_begin_shutdown_flush(
        self, command: BeginShutdownFlushRequested
    ) -> bool:
        if type(command) is not BeginShutdownFlushRequested:
            return False
        return self.begin_shutdown_flush(command.shutdown_generation)

    def _advance_shutdown_flush(self) -> bool:
        model = self.model
        generation = model.shutdown_flush_generation
        if (
            not self._accept_worker_results
            or not model.shutdown_flush_pending
            or generation is None
            or model.shutdown_flush_terminal
        ):
            return False
        if self._failed_rebuild_jobs:
            identity, rebuild = next(iter(self._failed_rebuild_jobs.items()))
            failures = tuple(self._rebuild_failures.get(rebuild.target.key, ()))
            plain_failures = tuple(_plain_target_result(item) for item in failures)
            if model.shutdown_flush_failure_identity != identity:
                model.shutdown_flush_failure_identity = identity
                model.shutdown_flush_failures = plain_failures
                failed = ShutdownFlushFailed(
                    generation, identity[0], identity[1], plain_failures
                )
                self._safe_emit(self.bus.events.shutdown_flush_failed, failed)
            if self._shutdown_failure_presented_identity != identity:
                presented = self._safe_view_call(
                    "show_shutdown_failure",
                    generation,
                    identity[0],
                    identity[1],
                    plain_failures,
                )
                if presented:
                    self._shutdown_failure_presented_identity = identity
            return True
        if (
            self._active_worker_identity is not None
            or self._pending_worker_jobs
            or self._pending_terminal_publication is not None
            or self._pending_presentation is not None
            or model.active_record_job is not None
            or model.record_failure is not None
        ):
            return True
        if not model.shutdown_flush_final_started:
            model.shutdown_flush_final_started = True
            for target in model.tracked_spool_targets():
                model.mark_target_dirty(target)
            self.handle_rebuild_debounce()
            if self._failed_rebuild_jobs:
                return self._advance_shutdown_flush()
            if (
                self._active_worker_identity is not None
                or self._pending_worker_jobs
                or self._rebuild_jobs
                or model.has_active_work()
            ):
                return True
        if (
            self._rebuild_jobs
            or self._failed_rebuild_jobs
            or model.has_active_work()
        ):
            return True
        return self._publish_shutdown_flush_completed(generation)

    def _schedule_shutdown_completion_retry(self) -> None:
        if (
            not self._accept_worker_results
            or self._shutdown_completion_retry_timer.isActive()
            or self._shutdown_completion_retry_attempt
            >= self._shutdown_completion_retry_limit
        ):
            return
        delay = min(
            self._shutdown_completion_retry_base_delay_ms
            * (2**self._shutdown_completion_retry_attempt),
            self._shutdown_completion_retry_max_delay_ms,
        )
        self._shutdown_completion_retry_attempt += 1
        self._shutdown_completion_retry_timer.start(delay)

    @pyqtSlot()
    def _retry_shutdown_completion_publication(self) -> None:
        self._advance_shutdown_flush()

    def _publish_shutdown_flush_completed(self, generation: int) -> bool:
        model = self.model
        identity = model.shutdown_flush_completion_identity
        if identity is None:
            job_id, attempt_id = self._shutdown_last_rebuild_identity or (
                f"shutdown:{generation}",
                "attempt:0",
            )
            identity = (generation, job_id, attempt_id)
            model.shutdown_flush_completion_identity = identity
        if identity[0] != generation:
            return False
        message = ShutdownFlushCompleted(generation)
        if self._formal_shutdown_completion_delivery:
            deliver_outcome = getattr(
                self.bus, "deliver_workflow_continuation_outcome", None
            )
            try:
                outcome = (
                    deliver_outcome(
                        (
                            "shutdown-flush-completed",
                            identity[0],
                            identity[1],
                            identity[2],
                        ),
                        "shutdown-flush-completed",
                        message,
                        owner=self,
                    )
                    if callable(deliver_outcome)
                    else None
                )
                delivered = (
                    type(outcome) is WorkflowContinuationDeliveryOutcome
                    and outcome.status is WorkflowContinuationDeliveryStatus.ACK
                )
            except BaseException as error:
                self._log(
                    "error",
                    "shutdown completion delivery failed: "
                    f"{_bounded_controller_error(error)}",
                )
                delivered = False
        else:
            delivered = self._safe_emit(
                self.bus.events.shutdown_flush_completed,
                message,
            )
        if not delivered:
            self._schedule_shutdown_completion_retry()
            return False
        model.shutdown_flush_pending = False
        model.shutdown_flush_terminal = True
        model.shutdown_flush_failure_identity = None
        self._shutdown_failure_presented_identity = None
        model.shutdown_flush_failures = ()
        self._shutdown_completion_retry_timer.stop()
        self._shutdown_completion_retry_attempt = 0
        return True

    @pyqtSlot(object)
    def handle_retry_shutdown_flush(
        self, command: RetryShutdownFlushRequested
    ) -> bool:
        if type(command) is not RetryShutdownFlushRequested:
            return False
        identity = (command.job_id, command.attempt_id)
        if (
            not self.model.shutdown_flush_pending
            or command.shutdown_generation
            != self.model.shutdown_flush_generation
            or identity != self.model.shutdown_flush_failure_identity
        ):
            return False
        rebuild = self.model.retry_rebuild(*identity)
        if rebuild is None:
            return False
        failed = self._failed_rebuild_jobs.pop(identity, None)
        if failed is not None:
            self._rebuild_failures.pop(failed.target.key, None)
        self._safe_view_call(
            "finish_shutdown_failure",
            command.shutdown_generation,
            command.job_id,
            command.attempt_id,
        )
        self.model.shutdown_flush_failure_identity = None
        self.model.shutdown_flush_failures = ()
        self._shutdown_failure_presented_identity = None
        self._rebuild_jobs[(rebuild.job_id, rebuild.attempt_id)] = rebuild
        self._submit(rebuild, rebuild.attempt_id)
        self._advance_shutdown_flush()
        return True

    @pyqtSlot(object)
    def handle_ignore_shutdown_flush_failure(
        self, command: IgnoreShutdownFlushFailureRequested
    ) -> bool:
        if type(command) is not IgnoreShutdownFlushFailureRequested:
            return False
        identity = (command.job_id, command.attempt_id)
        if (
            not self.model.shutdown_flush_pending
            or command.shutdown_generation
            != self.model.shutdown_flush_generation
            or identity != self.model.shutdown_flush_failure_identity
        ):
            return False
        rebuild = self._failed_rebuild_jobs.get(identity)
        if rebuild is None or not self.model.ignore_rebuild_failure(*identity):
            return False
        self._failed_rebuild_jobs.pop(identity, None)
        self._rebuild_failures.pop(rebuild.target.key, None)
        self._safe_view_call(
            "finish_shutdown_failure",
            command.shutdown_generation,
            command.job_id,
            command.attempt_id,
        )
        self.model.shutdown_flush_failure_identity = None
        self.model.shutdown_flush_failures = ()
        self._shutdown_failure_presented_identity = None
        self._resume_pending_worker()
        self._start_next_record_job()
        self._advance_shutdown_flush()
        return True

    def disconnect(self, _lifecycle_request=None) -> None:
        self._accept_worker_results = False
        self._debounce_timer.stop()
        self._shutdown_completion_retry_timer.stop()
        identity = self.model.shutdown_flush_completion_identity
        abandon = getattr(self.bus, "abandon_workflow_continuations", None)
        if (
            callable(abandon)
            and self._formal_shutdown_completion_delivery
            and self.model.shutdown_flush_pending
            and identity is not None
        ):
            try:
                abandon(
                    ((
                        "shutdown-flush-completed",
                        identity[0],
                        identity[1],
                        identity[2],
                    ),),
                    owner=self,
                    reason="export controller disconnected",
                )
            except BaseException as error:
                self._log(
                    "error",
                    "shutdown completion abandonment failed: "
                    f"{_bounded_controller_error(error)}",
                )
        unregister_owner = getattr(
            self.bus, "unregister_workflow_continuation_lifecycle_owner", None
        )
        unregister_recipient = getattr(
            self.bus, "unregister_workflow_continuation_recipient", None
        )
        if callable(unregister_recipient):
            unregister_recipient(
                "analysis-export-prepare",
                self._analysis_preparation_recipient_name,
                self.handle_prepare_analysis_export,
            )
            unregister_recipient(
                "manual-label-export-prepare",
                self._manual_preparation_recipient_name,
                self.handle_prepare_manual_label_export,
            )
            unregister_recipient(
                "export-preparation-cancel",
                self._preparation_cancel_recipient_name,
                self.handle_cancel_export_preparation,
            )
        if callable(unregister_owner) and self._formal_shutdown_completion_delivery:
            try:
                unregister_owner(self)
            except BaseException as error:
                self._log(
                    "error",
                    "shutdown completion owner unregister failed: "
                    f"{_bounded_controller_error(error)}",
                )
        self._pending_worker_jobs.clear()
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                continue
        self._connections.clear()
        try:
            self.view.disconnect()
        except (AttributeError, RuntimeError, TypeError):
            pass
        # Do not clear ``_owned_thread_handles`` here. Its instance-scoped
        # references keep detached work alive until the real thread terminal;
        # ``_worker_finished`` releases each handle without a timeout.


__all__ = ["SequenceExportController"]
