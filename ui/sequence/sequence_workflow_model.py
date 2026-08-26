"""Canonical cross-domain workflow state for one sequence window."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock
from typing import Any

from ui.sequence.sequence_messages import AnalysisTransportReady
from ui.sequence.sequence_workflow_policy import AutomaticAnalysisDecision


class WorkflowPhase(Enum):
    IDLE = auto()
    IMPORTING = auto()
    PREPARING = auto()
    RECORDING = auto()
    FINALIZING = auto()
    ANALYZING = auto()
    RESULT_EXPORTING = auto()
    LABEL_COMMITTING = auto()
    CANCELLING = auto()
    CLOSING = auto()
    SHUTDOWN_FLUSHING = auto()
    SHUTDOWN_READY = auto()


class ExportContinuation(Enum):
    ANALYSIS_DONE = auto()
    LABEL_COMMIT = auto()


class SessionOrigin(Enum):
    CANONICAL = auto()
    LEGACY_BRIDGE = auto()


@dataclass(frozen=True, slots=True)
class PostAnalysisContinuation:
    analysis_id: str
    source_id: str
    record_id: str
    workflow_generation: int
    result_snapshot: Any
    automatic_label: str | None


@dataclass(frozen=True, slots=True)
class WorkflowModelSnapshot:
    phase: WorkflowPhase
    workflow_generation: int
    configuration_generation: int
    shutdown_generation: int | None
    shutdown_pending: bool
    shutdown_asserted_active: bool
    shutdown_cancellation_confirmed: bool
    active_session_id: str | None
    active_session_origin: SessionOrigin | None
    active_import_id: str | None
    active_analysis_id: str | None
    active_job_id: str | None
    active_attempt_id: str | None
    retired_attempt_ids: frozenset[str]
    retained_record_id: str | None
    awaiting_label: bool
    automatic_analysis_decision: AutomaticAnalysisDecision | None
    export_continuation: ExportContinuation | None
    post_analysis_continuation: PostAnalysisContinuation | None
    export_failure_pending: bool
    active_label_command_id: str | None


class _AnalysisTransportClaim:
    """Opaque exact-identity token; Workflow retains the only reverse mapping."""

    __slots__ = ()


@dataclass(slots=True)
class _AnalysisTransportAuthorization:
    event: AnalysisTransportReady | None
    claim: _AnalysisTransportClaim | None = None


class SequenceWorkflowModel:
    """Main-thread-owned state shared only through the workflow controller."""

    ANALYSIS_TRANSPORT_HISTORY_LIMIT = 128

    def __init__(
        self,
        *,
        workflow_generation: int = 0,
        configuration_generation: int = 0,
        export_attempt_history_limit: int = 128,
    ) -> None:
        if (
            type(export_attempt_history_limit) is not int
            or export_attempt_history_limit < 1
        ):
            raise ValueError(
                "export_attempt_history_limit must be a positive integer"
            )
        self.export_attempt_history_limit = export_attempt_history_limit
        self.workflow_generation = self._generation(
            "workflow_generation", workflow_generation
        )
        self.configuration_generation = self._generation(
            "configuration_generation", configuration_generation
        )
        self.phase = WorkflowPhase.IDLE

        self.shutdown_generation: int | None = None
        self.last_shutdown_generation = -1
        self.shutdown_pending = False
        self.shutdown_asserted_active = False
        self.shutdown_cancellation_confirmed = False

        self.active_session_id: str | None = None
        self.active_session_origin: SessionOrigin | None = None
        self.active_import_id: str | None = None
        self.active_analysis_id: str | None = None
        self.active_job_id: str | None = None
        self.active_attempt_id: str | None = None
        self.retired_attempt_ids: set[str] = set()
        self._retired_attempt_order: OrderedDict[str, None] = OrderedDict()
        self.analysis_source_id: str | None = None
        self.analysis_record_id: str | None = None
        self.export_record_id: str | None = None

        self.export_continuation: ExportContinuation | None = None
        self.post_analysis_continuation: PostAnalysisContinuation | None = None
        self.export_failure_pending = False
        self.cancelling_phase: WorkflowPhase | None = None
        self.cancelling_domain: str | None = None

        self.retained_record_id: str | None = None
        self.awaiting_label = False
        self.automatic_analysis_decision: AutomaticAnalysisDecision | None = None
        self.active_label_command_id: str | None = None
        self.active_label_record_id: str | None = None
        self.active_label: str | None = None

        self.configuration_snapshot: Any = None
        self.session_snapshot: Any = None
        self.recording_snapshot: Any = None
        self.import_reference_snapshot: Any = None
        self.analysis_result_snapshot: Any = None
        self.labeled_result_snapshot: Any = None
        self.export_outcome: Any = None
        self._analysis_transport_authorization_lock = Lock()
        self._analysis_transport_authorizations: OrderedDict[
            tuple[str, str, str, int], _AnalysisTransportAuthorization
        ] = OrderedDict()

    def retire_export_attempt(self, attempt_id: str) -> None:
        self.retired_attempt_ids.add(attempt_id)
        self._retired_attempt_order[attempt_id] = None
        self._retired_attempt_order.move_to_end(attempt_id)
        while (
            len(self._retired_attempt_order)
            > self.export_attempt_history_limit
        ):
            retired, _value = self._retired_attempt_order.popitem(last=False)
            self.retired_attempt_ids.discard(retired)

    def clear_export_attempt_history(self) -> None:
        self.retired_attempt_ids.clear()
        self._retired_attempt_order.clear()

    @staticmethod
    def _analysis_transport_identity(
        event: AnalysisTransportReady,
    ) -> tuple[str, str, str, int] | None:
        try:
            analysis_id = object.__getattribute__(event, "analysis_id")
            source_id = object.__getattribute__(event, "source_id")
            record_id = object.__getattribute__(event, "record_id")
            workflow_generation = object.__getattribute__(
                event, "workflow_generation"
            )
            if any(
                type(value) is not str or not str.strip(value)
                for value in (analysis_id, source_id, record_id)
            ):
                return None
            if type(workflow_generation) is not int or workflow_generation < 0:
                return None
            return (
                analysis_id,
                source_id,
                record_id,
                workflow_generation,
            )
        except BaseException:
            return None

    def _make_analysis_transport_authorization_room(self) -> bool:
        history = self._analysis_transport_authorizations
        if len(history) < self.ANALYSIS_TRANSPORT_HISTORY_LIMIT:
            return True
        for identity, authorization in tuple(history.items()):
            # An unclaimed event is still a live retry authorization. Only a
            # consumed/retired tombstone is safe to replace.
            if authorization.event is None and authorization.claim is None:
                history.pop(identity, None)
                return True
        return False

    def authorize_analysis_transport(self, event: AnalysisTransportReady) -> bool:
        """Retain one exact Workflow-created transport until its queued consumer runs."""
        if type(event) is not AnalysisTransportReady:
            return False
        identity = self._analysis_transport_identity(event)
        if identity is None:
            return False
        with self._analysis_transport_authorization_lock:
            history = self._analysis_transport_authorizations
            if identity in history:
                return False
            if not self._make_analysis_transport_authorization_room():
                return False
            history[identity] = _AnalysisTransportAuthorization(event)
            return True

    def consume_analysis_transport(self, event: AnalysisTransportReady) -> bool:
        """Consume only the exact immutable object previously authorized by Workflow."""
        if type(event) is not AnalysisTransportReady:
            return False
        identity = self._analysis_transport_identity(event)
        if identity is None:
            return False
        with self._analysis_transport_authorization_lock:
            history = self._analysis_transport_authorizations
            authorization = history.get(identity)
            if (
                authorization is None
                or authorization.event is not event
                or authorization.claim is not None
            ):
                return False
            authorization.event = None
            history.move_to_end(identity)
            return True

    def is_analysis_transport_authorized(
        self, event: AnalysisTransportReady
    ) -> bool:
        """Check an exact pending authorization without consuming its retry token."""
        if type(event) is not AnalysisTransportReady:
            return False
        identity = self._analysis_transport_identity(event)
        if identity is None:
            return False
        with self._analysis_transport_authorization_lock:
            authorization = self._analysis_transport_authorizations.get(identity)
            return authorization is not None and authorization.event is event

    def claim_analysis_transport(self, event: AnalysisTransportReady) -> Any:
        """Atomically reserve one exact pending authorization for its sender."""
        if type(event) is not AnalysisTransportReady:
            return None
        identity = self._analysis_transport_identity(event)
        if identity is None:
            return None
        with self._analysis_transport_authorization_lock:
            authorization = self._analysis_transport_authorizations.get(identity)
            if (
                authorization is None
                or authorization.event is not event
                or authorization.claim is not None
            ):
                return None
            claim = _AnalysisTransportClaim()
            authorization.claim = claim
            return claim

    def _find_analysis_transport_claim(self, claim: Any):
        if type(claim) is not _AnalysisTransportClaim:
            return None
        for identity, authorization in self._analysis_transport_authorizations.items():
            if authorization.claim is claim:
                return identity, authorization
        return None

    def release_analysis_transport_claim(self, claim: Any) -> bool:
        """Restore a claimed authorization after no transport side effect occurred."""
        if type(claim) is not _AnalysisTransportClaim:
            return False
        with self._analysis_transport_authorization_lock:
            claimed = self._find_analysis_transport_claim(claim)
            if claimed is None:
                return False
            identity, authorization = claimed
            authorization.claim = None
            self._analysis_transport_authorizations.move_to_end(identity)
            return True

    def commit_analysis_transport_claim(self, claim: Any) -> bool:
        """Atomically consume the authorization after its send succeeded."""
        return self._finish_analysis_transport_claim(claim)

    def abandon_analysis_transport_claim(self, claim: Any) -> bool:
        """Consume a sent claim when its controller is being retired."""
        return self._finish_analysis_transport_claim(claim)

    def _finish_analysis_transport_claim(self, claim: Any) -> bool:
        if type(claim) is not _AnalysisTransportClaim:
            return False
        with self._analysis_transport_authorization_lock:
            claimed = self._find_analysis_transport_claim(claim)
            if claimed is None:
                return False
            identity, authorization = claimed
            if authorization.event is None:
                return False
            authorization.event = None
            authorization.claim = None
            self._analysis_transport_authorizations.move_to_end(identity)
            return True

    @staticmethod
    def _generation(name: str, value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @property
    def player_status_flag(self) -> bool:
        recording_phase = self.phase in {
            WorkflowPhase.PREPARING,
            WorkflowPhase.RECORDING,
            WorkflowPhase.FINALIZING,
        }
        cancelling_recording = (
            self.phase is WorkflowPhase.CANCELLING
            and self.cancelling_domain == "recording"
            and self.cancelling_phase
            in {
                WorkflowPhase.PREPARING,
                WorkflowPhase.RECORDING,
                WorkflowPhase.FINALIZING,
            }
        )
        return recording_phase or cancelling_recording

    @property
    def _record_workflow_busy(self) -> bool:
        return self.player_status_flag

    @property
    def record_workflow_busy(self) -> bool:
        return self._record_workflow_busy

    def is_workflow_active(self) -> bool:
        return self.phase is not WorkflowPhase.IDLE

    def apply_configuration(self, snapshot: Any, *, generation: int) -> bool:
        generation = self._generation("configuration_generation", generation)
        if generation < self.configuration_generation:
            return False
        self.configuration_snapshot = snapshot
        self.configuration_generation = generation
        return True

    def snapshot(self) -> WorkflowModelSnapshot:
        return WorkflowModelSnapshot(
            phase=self.phase,
            workflow_generation=self.workflow_generation,
            configuration_generation=self.configuration_generation,
            shutdown_generation=self.shutdown_generation,
            shutdown_pending=self.shutdown_pending,
            shutdown_asserted_active=self.shutdown_asserted_active,
            shutdown_cancellation_confirmed=self.shutdown_cancellation_confirmed,
            active_session_id=self.active_session_id,
            active_session_origin=self.active_session_origin,
            active_import_id=self.active_import_id,
            active_analysis_id=self.active_analysis_id,
            active_job_id=self.active_job_id,
            active_attempt_id=self.active_attempt_id,
            retired_attempt_ids=frozenset(self.retired_attempt_ids),
            retained_record_id=self.retained_record_id,
            awaiting_label=self.awaiting_label,
            automatic_analysis_decision=self.automatic_analysis_decision,
            export_continuation=self.export_continuation,
            post_analysis_continuation=self.post_analysis_continuation,
            export_failure_pending=self.export_failure_pending,
            active_label_command_id=self.active_label_command_id,
        )

    def assert_invariants(self) -> None:
        decision = self.automatic_analysis_decision
        if decision is not None and (
            type(decision) is not AutomaticAnalysisDecision
            or decision.workflow_generation != self.workflow_generation
        ):
            raise AssertionError(
                "automatic analysis decision must match the workflow generation"
            )
        if (
            self.active_session_id is None
            and self.active_session_origin is not None
        ):
            raise AssertionError("session origin requires an active recording session")

        active_identifiers = {
            "session": self.active_session_id,
            "import": self.active_import_id,
            "analysis": self.active_analysis_id,
            "job": self.active_job_id,
        }
        expected_identifier: str | None
        if self.phase is WorkflowPhase.IMPORTING:
            expected_identifier = "import"
        elif self.phase in {
            WorkflowPhase.PREPARING,
            WorkflowPhase.RECORDING,
            WorkflowPhase.FINALIZING,
        }:
            expected_identifier = "session"
        elif self.phase is WorkflowPhase.ANALYZING:
            expected_identifier = "analysis"
        elif self.phase is WorkflowPhase.RESULT_EXPORTING:
            expected_identifier = "job"
        elif self.phase is WorkflowPhase.CANCELLING:
            expected_identifier = {
                "recording": "session",
                "import": "import",
                "analysis": "analysis",
                "export": "job",
                "label": None,
                "preparation": None,
            }.get(self.cancelling_domain)
            if self.cancelling_domain not in {
                "recording",
                "import",
                "analysis",
                "export",
                "label",
                "preparation",
            }:
                raise AssertionError("cancelling workflow requires a domain identifier")
        else:
            expected_identifier = None

        present_identifiers = {
            name for name, identifier in active_identifiers.items() if identifier is not None
        }
        expected_identifiers = (
            set() if expected_identifier is None else {expected_identifier}
        )
        if present_identifiers != expected_identifiers:
            raise AssertionError(
                f"{self.phase.name} has illegal active domain identifier combination"
            )
        if self.active_session_id is not None and not isinstance(
            self.active_session_origin, SessionOrigin
        ):
            raise AssertionError(
                "active recording session requires a valid session origin"
            )

        if self.active_job_id is None and (
            self.active_attempt_id is not None or self.retired_attempt_ids
        ):
            raise AssertionError("export attempt identifiers require an active job identifier")
        if self.active_attempt_id in self.retired_attempt_ids:
            raise AssertionError("current export attempt cannot also be retired")

        continuation = self.post_analysis_continuation
        if continuation is not None:
            if self.phase not in {
                WorkflowPhase.ANALYZING,
                WorkflowPhase.RESULT_EXPORTING,
                WorkflowPhase.LABEL_COMMITTING,
                WorkflowPhase.CANCELLING,
            }:
                raise AssertionError(
                    "post-analysis continuation requires an active continuation phase"
                )
            if continuation.workflow_generation != self.workflow_generation:
                raise AssertionError(
                    "post-analysis continuation generation must match workflow"
                )

        if self.shutdown_cancellation_confirmed and (
            self.shutdown_generation is None
            or not self.shutdown_pending
            or self.phase
            not in {
                WorkflowPhase.RESULT_EXPORTING,
                WorkflowPhase.LABEL_COMMITTING,
                WorkflowPhase.CANCELLING,
            }
        ):
            raise AssertionError(
                "confirmed shutdown cancellation requires a pending decision phase"
            )

        if (
            self.phase is WorkflowPhase.LABEL_COMMITTING
            and self.active_label_command_id is None
        ):
            raise AssertionError("label commit requires a command identifier")
        if self.phase in {
            WorkflowPhase.CLOSING,
            WorkflowPhase.SHUTDOWN_FLUSHING,
            WorkflowPhase.SHUTDOWN_READY,
        } and self.shutdown_generation is None:
            raise AssertionError("shutdown phases require a shutdown generation")
