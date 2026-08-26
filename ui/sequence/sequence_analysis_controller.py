"""Controller for Workflow-admitted sequence analysis."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from threading import Lock
from typing import Any
from uuid import uuid4

from PyQt5.QtCore import QObject, pyqtSlot

from base.analysis_warning_preferences import (
    is_uncalibrated_microphone_warning_suppressed,
)
from base.soundcard_calibration_manager import (
    AnalysisV2paBatch,
    format_input_channel_label,
    resolve_analysis_v2pa_factor_for_channel,
)
from consts import ui_style_const
from consts.acoustic_analysis.common_consts import GOLDEN_SAMPLE_RESULT_PATH_KEY
from ui.signal_analysis_window import get_class_mapping

from ui.sequence.sequence_analysis_model import (
    DEFAULT_ANALYSIS_CALIBRATION_TYPES,
    AnalysisState,
    SequenceAnalysisCalibrationPolicyService,
    SequenceAnalysisModel,
    mutable_analysis_value,
)
from ui.sequence.sequence_analysis_view import SequenceAnalysisView
from ui.sequence.sequence_analysis_transport_service import (
    SequenceAnalysisTransportService,
)
from ui.sequence.sequence_messages import (
    AnalysisCompleted,
    AnalysisFailed,
    AnalysisRequested,
    AnalysisTransportReady,
    CancelAnalysisRequested,
    ManualAnalysisRequested,
    ManualLabelRequested,
    ResourceLifecycleRequested,
)


@dataclass(frozen=True, slots=True)
class AnalysisExecutionContext:
    analysis_config: Any
    sequence_config: Any
    mode: str
    active_channels: tuple[int, ...]
    recording_snapshot: Any
    test_mode: bool
    calibration_types: frozenset[str] = DEFAULT_ANALYSIS_CALIBRATION_TYPES


class _WorkflowIdentityProviderFailure(RuntimeError):
    """Contain a failed coordinator identity read at the Qt slot boundary."""


class _AnalysisTransportReservationPhase(Enum):
    AUTHORIZING = auto()
    SENDING = auto()
    COMMITTING = auto()
    RELEASING = auto()
    RELEASE_PENDING = auto()
    SEND_SUCCEEDED_PENDING_COMMIT = auto()
    ABANDONING = auto()
    ABANDON_PENDING = auto()


@dataclass(frozen=True, slots=True)
class _LegacyAnalysisTransportClaim:
    event: AnalysisTransportReady


@dataclass(slots=True)
class _AnalysisTransportReservation:
    event: AnalysisTransportReady
    phase: _AnalysisTransportReservationPhase
    claim: Any = None
    active_handler: Any = None


class SequenceAnalysisTransportController(QObject):
    """Own Workflow-authorized TCP delivery and exact retry bookkeeping."""

    def __init__(
        self,
        *,
        bus: Any,
        authorization_claimer: Callable[
            [AnalysisTransportReady], Any
        ] | None = None,
        claim_releaser: Callable[[Any], bool] | None = None,
        claim_committer: Callable[[Any], bool] | None = None,
        claim_abandoner: Callable[[Any], bool] | None = None,
        authorization_provider: Callable[
            [AnalysisTransportReady], bool
        ] | None = None,
        authorization_consumer: Callable[
            [AnalysisTransportReady], bool
        ] | None = None,
        tcp_enabled_provider: Callable[[], bool] | None = None,
        tcp_server_provider: Callable[[], Any] | None = None,
        service: SequenceAnalysisTransportService | Any | None = None,
        logger: Any = None,
        history_limit: int = 128,
        parent: QObject | None = None,
        connect_bus: bool = True,
    ) -> None:
        super().__init__(parent)
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        self.bus = bus
        atomic_ports = (
            authorization_claimer,
            claim_releaser,
            claim_committer,
            claim_abandoner,
        )
        if any(port is not None for port in atomic_ports):
            if not all(callable(port) for port in atomic_ports):
                raise ValueError("all analysis transport claim ports are required")
            self.authorization_claimer = authorization_claimer
            self.claim_releaser = claim_releaser
            self.claim_committer = claim_committer
            self.claim_abandoner = claim_abandoner
        else:
            if not callable(authorization_provider) or not callable(
                authorization_consumer
            ):
                raise ValueError(
                    "analysis transport requires atomic claim ports or "
                    "legacy authorization callbacks"
                )
            self.authorization_claimer = self._legacy_claim_authorization
            self.claim_releaser = self._legacy_release_claim
            self.claim_committer = self._legacy_commit_claim
            self.claim_abandoner = self._legacy_commit_claim
        self.authorization_provider = authorization_provider
        self.authorization_consumer = authorization_consumer
        self.logger = logger
        if service is None:
            if not callable(tcp_enabled_provider) or not callable(
                tcp_server_provider
            ):
                raise ValueError(
                    "transport service or TCP providers are required"
                )
            service = SequenceAnalysisTransportService(
                tcp_enabled_provider=tcp_enabled_provider,
                tcp_server_provider=tcp_server_provider,
                logger=logger,
            )
        self.service = service
        self.history_limit = history_limit
        self._state_lock = Lock()
        self._reservations: dict[
            tuple[str, str, str, int], _AnalysisTransportReservation
        ] = {}
        self._applied: OrderedDict[
            tuple[str, str, str, int], AnalysisTransportReady
        ] = OrderedDict()
        self._recipient_name = f"analysis-transport:{id(self)}"
        self._recipient_token = None
        self._connected = False
        self._closed = False
        if connect_bus:
            self._recipient_token = (
                self.bus.register_workflow_continuation_recipient(
                    "analysis-transport",
                    self._recipient_name,
                    self.handle_analysis_transport_ready,
                    owner=self,
                )
            )
            self._connected = True
        self.destroyed.connect(self._handle_destroyed)

    @staticmethod
    def _identity(
        event: AnalysisTransportReady,
    ) -> tuple[str, str, str, int] | None:
        try:
            analysis_id = object.__getattribute__(event, "analysis_id")
            source_id = object.__getattribute__(event, "source_id")
            record_id = object.__getattribute__(event, "record_id")
            workflow_generation = object.__getattribute__(
                event, "workflow_generation"
            )
            identifiers = (analysis_id, source_id, record_id)
            if any(
                type(value) is not str
                or not str.strip(value)
                or str.__len__(value) > 8_192
                for value in identifiers
            ):
                return None
            if (
                type(workflow_generation) is not int
                or workflow_generation < 0
                or int.bit_length(workflow_generation) > 4_096
            ):
                return None
            return (
                analysis_id,
                source_id,
                record_id,
                workflow_generation,
            )
        except BaseException:
            # Exact message validation is an ingress boundary: corrupted
            # frozen objects must not escape into lock-owned dictionaries.
            return None

    def send_payload(self, payload: Any) -> bool:
        """Compatibility delegate for callers that previously used the Controller."""
        return self.service.send_payload(payload)

    def _diagnose_transport_boundary(self, operation: str) -> None:
        try:
            warning = getattr(self.logger, "warning", None)
            if callable(warning):
                warning(f"analysis_transport_{operation}_failed")
        except BaseException:
            # Diagnostics are best-effort and cannot change claim ownership.
            return

    def _legacy_claim_authorization(self, event: AnalysisTransportReady) -> Any:
        if self.authorization_provider(event) is not True:
            return None
        return _LegacyAnalysisTransportClaim(event)

    @staticmethod
    def _legacy_release_claim(claim: Any) -> bool:
        return type(claim) is _LegacyAnalysisTransportClaim

    def _legacy_commit_claim(self, claim: Any) -> bool:
        if type(claim) is not _LegacyAnalysisTransportClaim:
            return False
        return self.authorization_consumer(claim.event)

    def _release_reserved_claim(
        self,
        identity: tuple[str, str, str, int],
        reservation: _AnalysisTransportReservation,
    ) -> bool:
        try:
            released = self.claim_releaser(reservation.claim)
        except BaseException:
            released = False
            self._diagnose_transport_boundary("claim_release")
        with self._state_lock:
            if self._reservations.get(identity) is not reservation:
                return False
            if released is True:
                self._reservations.pop(identity, None)
                return True
            reservation.phase = _AnalysisTransportReservationPhase.RELEASE_PENDING
            return False

    def _record_applied_reservation(
        self,
        identity: tuple[str, str, str, int],
        reservation: _AnalysisTransportReservation,
    ) -> None:
        self._reservations.pop(identity, None)
        self._applied[identity] = reservation.event
        self._applied.move_to_end(identity)
        while len(self._applied) > self.history_limit:
            self._applied.popitem(last=False)

    def _abandon_reserved_claim(
        self,
        identity: tuple[str, str, str, int],
        reservation: _AnalysisTransportReservation,
    ) -> bool:
        try:
            abandoned = self.claim_abandoner(reservation.claim)
        except BaseException:
            abandoned = False
            self._diagnose_transport_boundary("claim_abandon")
        settled_event = None
        with self._state_lock:
            if self._reservations.get(identity) is not reservation:
                return False
            if abandoned is True:
                self._reservations.pop(identity, None)
                settled_event = reservation.event
            else:
                reservation.phase = (
                    _AnalysisTransportReservationPhase.ABANDON_PENDING
                )
        if settled_event is not None:
            self._settle_abandoned_delivery(identity, settled_event)
            return True
        self._diagnose_transport_boundary("claim_abandon_rejected")
        return False

    def _settle_abandoned_delivery(
        self,
        identity: tuple[str, str, str, int],
        event: AnalysisTransportReady,
    ) -> None:
        settle = getattr(
            self.bus, "acknowledge_workflow_continuation_recipient", None
        )
        if not callable(settle) or self._recipient_token is None:
            return
        try:
            settled = settle(
                ("analysis-transport", *identity),
                "analysis-transport",
                event,
                self._recipient_token,
            )
        except BaseException:
            settled = False
            self._diagnose_transport_boundary("delivery_settle")
        if settled is not True:
            self._diagnose_transport_boundary("delivery_settle_rejected")

    def _finish_active_handler(
        self,
        identity: tuple[str, str, str, int],
        reservation: _AnalysisTransportReservation,
        handler_token: Any,
        result: bool,
    ) -> bool:
        abandon = False
        with self._state_lock:
            if self._reservations.get(identity) is not reservation:
                return result
            if reservation.active_handler is not handler_token:
                return False
            if (
                result is False
                and self._closed
                and reservation.phase
                is _AnalysisTransportReservationPhase.SEND_SUCCEEDED_PENDING_COMMIT
            ):
                reservation.phase = _AnalysisTransportReservationPhase.ABANDONING
                abandon = True
            else:
                reservation.active_handler = None
                return result
        if abandon:
            abandoned = self._abandon_reserved_claim(identity, reservation)
            if not abandoned:
                with self._state_lock:
                    if self._reservations.get(identity) is reservation:
                        reservation.active_handler = None
            return abandoned
        return result

    def _commit_reserved_claim(
        self,
        identity: tuple[str, str, str, int],
        reservation: _AnalysisTransportReservation,
    ) -> bool:
        try:
            committed = self.claim_committer(reservation.claim)
        except BaseException:
            committed = False
            self._diagnose_transport_boundary("claim_commit")
        with self._state_lock:
            if self._reservations.get(identity) is not reservation:
                return False
            if committed is True:
                self._record_applied_reservation(identity, reservation)
                return True
            if self._closed:
                reservation.phase = _AnalysisTransportReservationPhase.ABANDONING
            else:
                reservation.phase = (
                    _AnalysisTransportReservationPhase.SEND_SUCCEEDED_PENDING_COMMIT
                )
                return False
        return self._abandon_reserved_claim(identity, reservation)

    def handle_analysis_transport_ready(self, event: Any) -> bool:
        if type(event) is not AnalysisTransportReady:
            return False
        identity = self._identity(event)
        if identity is None:
            return False
        with self._state_lock:
            if self._closed:
                return False
            applied_event = self._applied.get(identity)
            if applied_event is not None:
                return False
            reservation = self._reservations.get(identity)
            if reservation is not None:
                if reservation.event is not event:
                    return False
                if reservation.active_handler is not None:
                    return False
                if (
                    reservation.phase
                    is _AnalysisTransportReservationPhase.SEND_SUCCEEDED_PENDING_COMMIT
                ):
                    reservation.phase = _AnalysisTransportReservationPhase.COMMITTING
                    retry_action = "commit"
                elif (
                    reservation.phase
                    is _AnalysisTransportReservationPhase.RELEASE_PENDING
                ):
                    reservation.phase = _AnalysisTransportReservationPhase.RELEASING
                    retry_action = "release"
                else:
                    return False
            else:
                if len(self._reservations) >= self.history_limit:
                    return False
                reservation = _AnalysisTransportReservation(
                    event,
                    _AnalysisTransportReservationPhase.AUTHORIZING,
                )
                self._reservations[identity] = reservation
                retry_action = "claim"
            handler_token = object()
            reservation.active_handler = handler_token

        if retry_action == "commit":
            result = self._commit_reserved_claim(identity, reservation)
            return self._finish_active_handler(
                identity, reservation, handler_token, result
            )
        if retry_action == "release":
            self._release_reserved_claim(identity, reservation)
            return self._finish_active_handler(
                identity, reservation, handler_token, False
            )

        try:
            claim = self.authorization_claimer(event)
        except BaseException:
            claim = None
            self._diagnose_transport_boundary("claim")
        if claim is None:
            with self._state_lock:
                if self._reservations.get(identity) is reservation:
                    self._reservations.pop(identity, None)
            return self._finish_active_handler(
                identity, reservation, handler_token, False
            )

        release_claim = False
        with self._state_lock:
            if (
                self._reservations.get(identity) is not reservation
                or self._closed
            ):
                if self._reservations.get(identity) is reservation:
                    self._reservations.pop(identity, None)
                release_claim = True
            else:
                reservation.claim = claim
                reservation.phase = _AnalysisTransportReservationPhase.SENDING
        if release_claim:
            try:
                released = self.claim_releaser(claim)
            except BaseException:
                released = False
                self._diagnose_transport_boundary("claim_release")
            if released is not True:
                self._diagnose_transport_boundary("claim_release_rejected")
            return self._finish_active_handler(
                identity, reservation, handler_token, False
            )

        try:
            sent = event.payload is None or self.send_payload(event.payload) is True
        except BaseException:
            sent = False
            self._diagnose_transport_boundary("send")
        if not sent:
            with self._state_lock:
                reservation_current = (
                    self._reservations.get(identity) is reservation
                )
                if reservation_current:
                    reservation.phase = (
                        _AnalysisTransportReservationPhase.RELEASING
                    )
            if not reservation_current:
                return self._finish_active_handler(
                    identity, reservation, handler_token, False
                )
            self._release_reserved_claim(identity, reservation)
            return self._finish_active_handler(
                identity, reservation, handler_token, False
            )

        with self._state_lock:
            reservation_current = (
                self._reservations.get(identity) is reservation
            )
            if reservation_current:
                reservation.phase = (
                    _AnalysisTransportReservationPhase.COMMITTING
                )
        if not reservation_current:
            return self._finish_active_handler(
                identity, reservation, handler_token, False
            )
        result = self._commit_reserved_claim(identity, reservation)
        return self._finish_active_handler(
            identity, reservation, handler_token, result
        )

    def _retire_transport_claims(self) -> None:
        releases = []
        abandonments = []
        with self._state_lock:
            self._closed = True
            for identity, reservation in tuple(self._reservations.items()):
                if reservation.active_handler is not None:
                    continue
                if reservation.phase is _AnalysisTransportReservationPhase.AUTHORIZING:
                    self._reservations.pop(identity, None)
                elif reservation.phase is _AnalysisTransportReservationPhase.RELEASE_PENDING:
                    self._reservations.pop(identity, None)
                    releases.append(reservation.claim)
                elif reservation.phase in {
                    _AnalysisTransportReservationPhase.SEND_SUCCEEDED_PENDING_COMMIT,
                    _AnalysisTransportReservationPhase.ABANDON_PENDING,
                }:
                    reservation.phase = _AnalysisTransportReservationPhase.ABANDONING
                    abandonments.append((identity, reservation))
        for claim in releases:
            try:
                released = self.claim_releaser(claim)
            except BaseException:
                released = False
                self._diagnose_transport_boundary("claim_release")
            if released is not True:
                self._diagnose_transport_boundary("claim_release_rejected")
        for identity, reservation in abandonments:
            self._abandon_reserved_claim(identity, reservation)

    def _handle_destroyed(self, *_args: Any) -> None:
        self._retire_transport_claims()

    def disconnect(self, _lifecycle_request=None) -> bool:
        self._retire_transport_claims()
        with self._state_lock:
            connected = self._connected
            self._connected = False
        if not connected:
            return type(_lifecycle_request) is ResourceLifecycleRequested
        self.bus.unregister_workflow_continuation_recipient(
            "analysis-transport",
            self._recipient_name,
            self.handle_analysis_transport_ready,
        )
        return True


class SequenceAnalysisController(QObject):
    """Validate and execute one analysis task, then publish one terminal event."""

    MISSING_CALIBRATION_MESSAGE = "麦克风未进行校准，结果仅供参考。"

    def __init__(
        self,
        model: SequenceAnalysisModel,
        view: SequenceAnalysisView | Any,
        *,
        bus: Any,
        readiness: Callable[[AnalysisRequested], Any] | None = None,
        execute: Callable[[AnalysisRequested], Mapping[str, Any]] | None = None,
        runtime: Any = None,
        class_mapping_provider: Callable[[], Mapping[str, Any]] = get_class_mapping,
        calibration_batch_factory: Callable[..., Any] = AnalysisV2paBatch,
        calibration_resolver: Callable[..., Any] = resolve_analysis_v2pa_factor_for_channel,
        calibration_policy_service: SequenceAnalysisCalibrationPolicyService | None = None,
        warning_suppressed: Callable[..., bool] = is_uncalibrated_microphone_warning_suppressed,
        instance_factory: Callable[..., Any] | None = None,
        workflow_identity_provider: Callable[[], Mapping[str, Any]] | None = None,
        timestamp_provider: Callable[[], datetime] = datetime.now,
        transport_service: SequenceAnalysisTransportService | Any | None = None,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.view = view
        self.bus = bus
        self.runtime = runtime
        self.readiness = readiness or self._runtime_readiness
        self.execute = execute or self._execute_admitted
        self.class_mapping_provider = class_mapping_provider
        self.calibration_batch_factory = calibration_batch_factory
        self.calibration_resolver = calibration_resolver
        self.calibration_policy_service = (
            calibration_policy_service or SequenceAnalysisCalibrationPolicyService()
        )
        self.warning_suppressed = warning_suppressed
        self.instance_factory = instance_factory
        self.workflow_identity_provider = workflow_identity_provider
        self.timestamp_provider = timestamp_provider
        self.transport_service = transport_service
        self.logger = logger
        self._connected = True
        self._active_context: AnalysisExecutionContext | None = None

    def disconnect(self, _lifecycle_request=None) -> bool:
        if not self._connected:
            return type(_lifecycle_request) is ResourceLifecycleRequested
        self._connected = False
        return True

    @property
    def active_context(self) -> AnalysisExecutionContext | None:
        return self._active_context

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    @staticmethod
    def _normalize_readiness(value: Any) -> tuple[bool, str]:
        if type(value) is tuple and len(value) == 2:
            ready, reason = value
            return bool(ready), str(reason or "")
        return bool(value), "" if value else "analysis is not ready"

    def _runtime_readiness(self, command: AnalysisRequested) -> tuple[bool, str]:
        runtime = self.runtime
        if runtime is None:
            return False, "analysis runtime is unavailable"
        context = self._active_context or self._context_from_command(command)
        if context.mode == "IMPORT_STIMULUS_AUDIO":
            data_struct = getattr(runtime, "data_struct", None)
            if not self._import_stimulus_runtime_ready(data_struct):
                return False, "import stimulus reference is unavailable"
        return True, ""

    @pyqtSlot(object)
    def handle_analysis_requested(self, command: AnalysisRequested) -> bool:
        if type(command) is not AnalysisRequested:
            return False
        if self.model.is_retired(
            command.analysis_id, command.source_id, command.workflow_generation
        ):
            self._log("warning", f"ignored retired analysis command {command.analysis_id}")
            return False
        if self.model.state is AnalysisState.RUNNING:
            self._log(
                "warning", f"ignored reentrant analysis command {command.analysis_id}"
            )
            return False
        try:
            admitted = self._accept_admitted_identity(command)
        except _WorkflowIdentityProviderFailure as error:
            return self._fail_identity_provider_admission(command, error)
        if not admitted:
            return False
        self.model.begin(command)
        event: AnalysisCompleted | AnalysisFailed
        succeeded = False
        try:
            self._active_context = self._context_from_command(command)
            identity = self._workflow_identity()
            if (
                identity is not None
                and identity.get("phase") == "CANCELLING"
                and identity.get("cancelling_domain") == "analysis"
            ):
                self.model.request_cancel("analysis cancelled before execution")
            if self.model.cancel_pending:
                ready, reason = True, ""
            else:
                ready, reason = self._normalize_readiness(self.readiness(command))
            if self.model.cancel_pending:
                failure_reason = self._cancel_failure_reason()
                self.model.fail()
                event = AnalysisFailed(
                    command.analysis_id, command.source_id, failure_reason
                )
            elif not ready:
                failure_reason = reason or "analysis is not ready"
                self._present_readiness_failure(failure_reason)
                self.model.fail()
                event = AnalysisFailed(
                    command.analysis_id, command.source_id, failure_reason
                )
            else:
                result = self.execute(command)
                self._synchronize_cancellation_identity(command)
                if self.model.cancel_pending:
                    failure_reason = self._cancel_failure_reason()
                    self.model.fail()
                    event = AnalysisFailed(
                        command.analysis_id, command.source_id, failure_reason
                    )
                else:
                    if not isinstance(result, Mapping):
                        raise TypeError("analysis result snapshot must be a mapping")
                    normalized_result = dict(result)
                    normalized_result["analysis_id"] = command.analysis_id
                    normalized_result["source_id"] = command.source_id
                    normalized_result["workflow_generation"] = (
                        command.workflow_generation
                    )
                    event = AnalysisCompleted(
                        command.analysis_id,
                        command.source_id,
                        normalized_result,
                    )
                    # Constructing the immutable public message validates and
                    # recursively freezes the complete payload before Model
                    # state can become COMPLETED.
                    self.model.complete(event.result_snapshot)
                    succeeded = True
        except BaseException as error:
            # Qt slot and third-party analysis code are a BaseException boundary:
            # leave the model terminal and clear all per-run context before Qt
            # regains control, including interruption-style failures.
            reason = self._safe_error_text(error)
            self._log("error", f"analysis[{command.analysis_id}] failed: {reason}")
            if self.model.state in {AnalysisState.RUNNING, AnalysisState.COMPLETED}:
                self.model.fail()
            event = AnalysisFailed(command.analysis_id, command.source_id, reason)
        finally:
            self._active_context = None
        self.model.retire(
            command.analysis_id, command.source_id, command.workflow_generation
        )
        if succeeded:
            self.bus.events.analysis_completed.emit(event)
            return True
        self.bus.events.analysis_failed.emit(event)
        return False

    def _fail_identity_provider_admission(
        self,
        command: AnalysisRequested,
        error: _WorkflowIdentityProviderFailure,
    ) -> bool:
        self.model.begin(command)
        reason = self._safe_error_text(error)
        self.model.fail()
        self.model.retire(
            command.analysis_id, command.source_id, command.workflow_generation
        )
        self._active_context = None
        self.bus.events.analysis_failed.emit(
            AnalysisFailed(command.analysis_id, command.source_id, reason)
        )
        return False

    def _synchronize_cancellation_identity(self, command: AnalysisRequested) -> None:
        if self.model.cancel_pending:
            return
        identity = self._workflow_identity()
        if identity is None:
            return
        exact = (
            identity.get("analysis_id") == command.analysis_id
            and identity.get("source_id") == command.source_id
            and identity.get("workflow_generation") == command.workflow_generation
        )
        if (
            exact
            and identity.get("phase") == "CANCELLING"
            and identity.get("cancelling_domain") == "analysis"
        ):
            if not self.model.cancel_pending:
                self.model.request_cancel("analysis cancelled before completion")
            return
        if not (exact and identity.get("phase") == "ANALYZING"):
            raise RuntimeError("workflow identity changed during analysis")

    @pyqtSlot(object)
    def handle_cancel_analysis_requested(
        self, command: CancelAnalysisRequested
    ) -> bool:
        if type(command) is not CancelAnalysisRequested:
            return False
        if self.model.is_retired_analysis_id(
            command.analysis_id, command.workflow_generation
        ):
            self._log("warning", f"ignored retired analysis cancellation {command.analysis_id}")
            return False
        if (
            self.model.state is not AnalysisState.RUNNING
            or command.analysis_id != self.model.active_analysis_id
            or command.workflow_generation != self.model.active_workflow_generation
        ):
            self._log("warning", f"ignored stale analysis cancellation {command.analysis_id}")
            return False
        try:
            identity = self._workflow_identity()
        except _WorkflowIdentityProviderFailure as error:
            # The locally active Model already owns the exact admitted identity.
            # A matching Workflow-issued cancel remains safe to accept when the
            # coordinator identity read itself fails at this DirectConnection
            # boundary.
            self._log(
                "error",
                "accepted matching analysis cancellation after identity "
                f"provider failure: {self._bounded_error_text(error)}",
            )
            self.model.request_cancel(command.reason)
            return True
        if identity is not None and not (
            identity.get("analysis_id") == command.analysis_id
            and identity.get("workflow_generation") == command.workflow_generation
            and identity.get("phase") == "CANCELLING"
            and identity.get("cancelling_domain") == "analysis"
        ):
            self._log("warning", f"ignored unmatched analysis cancellation {command.analysis_id}")
            return False
        self.model.request_cancel(command.reason)
        return True

    def _accept_admitted_identity(self, command: AnalysisRequested) -> bool:
        identity = self._workflow_identity()
        if identity is None:
            return True
        accepted = (
            identity.get("analysis_id") == command.analysis_id
            and identity.get("source_id") == command.source_id
            and identity.get("workflow_generation") == command.workflow_generation
            and identity.get("phase") in {"ANALYZING", "CANCELLING"}
        )
        if not accepted:
            self._log("warning", f"ignored stale analysis command {command.analysis_id}")
        return accepted

    def _workflow_identity(self) -> Mapping[str, Any] | None:
        if self.workflow_identity_provider is None:
            return None
        try:
            value = self.workflow_identity_provider()
        except BaseException as error:
            detail = self._bounded_error_text(error)
            self._log("error", f"analysis identity provider failed: {detail}")
            raise _WorkflowIdentityProviderFailure(
                f"analysis identity provider failed: {detail}"
            ) from None
        if not isinstance(value, Mapping):
            raise _WorkflowIdentityProviderFailure(
                "analysis identity provider returned a non-mapping value"
            )
        return value

    @staticmethod
    def _safe_error_text(error: BaseException) -> str:
        try:
            detail = str(error)
        except BaseException:
            detail = "<unprintable error>"
        return detail or type(error).__name__

    @classmethod
    def _bounded_error_text(cls, error: BaseException) -> str:
        detail = cls._safe_error_text(error)
        if len(detail) > 512:
            return f"{detail[:509]}..."
        return detail

    def _cancel_failure_reason(self) -> str:
        reason = self.model.cancellation_reason or "analysis cancelled"
        return reason if "cancel" in reason.lower() else f"analysis cancelled: {reason}"

    def _present_readiness_failure(self, reason: str) -> None:
        presenter = getattr(self.view, "warning_presenter", None)
        if callable(presenter):
            context = self._active_context
            if context is not None and context.mode == "IMPORT_STIMULUS_AUDIO":
                _ready, title, text = self._import_stimulus_readiness_detail(
                    getattr(self.runtime, "data_struct", None)
                )
                presenter(title, text)
            else:
                presenter("提示", reason)

    @staticmethod
    def _has_samples(value: Any) -> bool:
        try:
            return len(value) > 0
        except (TypeError, ValueError):
            return False

    @classmethod
    def _import_stimulus_runtime_ready(cls, data_struct: Any) -> bool:
        return cls._import_stimulus_readiness_detail(data_struct)[0]

    @classmethod
    def _import_stimulus_readiness_detail(
        cls, data_struct: Any
    ) -> tuple[bool, str, str]:
        generic = "分析参考激励尚未就绪或采样率与导入音频不一致，请检查激励配置后重试。"
        if data_struct is None:
            return False, "提示", generic
        stimulus_info = getattr(data_struct, "stimulus_info", None)
        if not isinstance(stimulus_info, Mapping):
            return False, "提示", generic
        try:
            recording_rate = float(getattr(data_struct, "sample_rate"))
            reference_rate = float(stimulus_info.get("sample_rate"))
            audio_length = int(getattr(data_struct, "audio_lenth"))
            total_time = float(stimulus_info.get("total_time"))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False, "提示", generic
        base_ready = bool(
            math.isfinite(recording_rate)
            and recording_rate > 0
            and math.isfinite(reference_rate)
            and reference_rate > 0
            and recording_rate == reference_rate
            and cls._has_samples(getattr(data_struct, "store_wave_data", None))
            and cls._has_samples(getattr(data_struct, "store_wave_data_multi", None))
            and cls._has_samples(getattr(data_struct, "stimulus_data", None))
            and math.isfinite(total_time)
            and total_time > 0
        )
        if not base_ready:
            return False, "提示", generic
        stimulus_length = round(total_time * reference_rate)
        if audio_length != stimulus_length:
            return (
                False,
                "音频长度校验失败",
                f"导入音频长度({audio_length})\n"
                f"与激励信号长度({stimulus_length})不一致！无法分析！",
            )
        return True, "", ""

    def request_manual_analysis(self) -> bool:
        """Translate the Data action into a Workflow command."""
        runtime = self.runtime
        if runtime is None:
            return False
        workflow_model = getattr(runtime, "workflow_model", None)
        record_id = getattr(workflow_model, "retained_record_id", None)
        if not record_id:
            record_id = getattr(runtime, "recorded_path", None)
        if type(record_id) is not str or not record_id:
            return False
        command = ManualAnalysisRequested(f"manual-analysis-{uuid4().hex}", record_id)
        runtime.sequence_event_bus.commands.manual_analysis_requested.emit(command)
        return True

    def update_v2pa_factor(self) -> None:
        if self.runtime is not None:
            self.runtime.v2pa_factor = None

    def configure_calibration_types(
        self, analysis_types: Any, *, generation: int
    ) -> bool:
        """Apply a validated policy update; stale generations are ignored."""
        if (
            type(generation) is int
            and generation >= 0
            and generation
            <= self.model.calibration_policy_snapshot.generation
        ):
            return False
        snapshot = self.calibration_policy_service.snapshot(
            analysis_types,
            generation=generation,
        )
        return self.model.apply_calibration_policy(snapshot)

    def _context_from_command(
        self, command: AnalysisRequested
    ) -> AnalysisExecutionContext:
        sequence_config = mutable_analysis_value(
            self._snapshot_field(
                command.configuration_snapshot, "sequence_config"
            )
        )
        analysis_config = mutable_analysis_value(
            self._snapshot_field(
                command.configuration_snapshot, "analysis_config"
            )
        )
        mode = self._mode_from_sequence_config(sequence_config)
        active_channels = self._snapshot_active_channels(
            command.recording_snapshot, command.configuration_snapshot
        )
        count_board = getattr(self.runtime, "count_board", None)
        return AnalysisExecutionContext(
            analysis_config=analysis_config,
            sequence_config=sequence_config,
            mode=mode,
            active_channels=active_channels,
            recording_snapshot=command.recording_snapshot,
            test_mode=getattr(count_board, "mode", None) == "test",
            calibration_types=(
                self.model.calibration_policy_snapshot.analysis_types
            ),
        )

    def _legacy_context(
        self,
        *,
        analysis_config: Any = None,
        sequence_config: Any = None,
    ) -> AnalysisExecutionContext:
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("analysis runtime is unavailable")
        resolved_analysis = (
            getattr(runtime, "analysis_config", None)
            if analysis_config is None
            else analysis_config
        )
        resolved_sequence = (
            getattr(runtime, "sequence_config", None)
            if sequence_config is None
            else sequence_config
        )
        mode = getattr(runtime, "mode", None)
        if type(mode) is not str or not mode:
            mode = self._mode_from_sequence_config(resolved_sequence)
        return AnalysisExecutionContext(
            analysis_config=resolved_analysis,
            sequence_config=resolved_sequence,
            mode=mode,
            active_channels=tuple(self._active_channels(runtime, allow_live=True)),
            recording_snapshot={
                "record_id": getattr(runtime, "recorded_path", None),
                "recorded_path": getattr(runtime, "recorded_path", None),
                "recorded_signal_info": getattr(
                    runtime, "recorded_signal_info", None
                ),
            },
            test_mode=getattr(getattr(runtime, "count_board", None), "mode", None)
            == "test",
            calibration_types=(
                self.model.calibration_policy_snapshot.analysis_types
            ),
        )

    @staticmethod
    def _mode_from_sequence_config(sequence_config: Any) -> str:
        try:
            mode = sequence_config[0]["seq1"]["acq"]["mode"]
        except (AttributeError, IndexError, KeyError, TypeError):
            raise ValueError("analysis configuration is unavailable")
        if type(mode) is not str or not mode:
            raise ValueError("analysis configuration mode is unavailable")
        return mode

    @classmethod
    def _snapshot_active_channels(
        cls, recording_snapshot: Any, configuration_snapshot: Any
    ) -> tuple[int, ...]:
        candidates: Any = None
        if isinstance(recording_snapshot, Mapping):
            session = recording_snapshot.get("session")
            if isinstance(session, Mapping):
                candidates = session.get("input_channels")
            if candidates is None:
                candidates = recording_snapshot.get("input_channels")
            if candidates is None:
                channel_count = recording_snapshot.get("channel_count")
                if type(channel_count) is int and channel_count > 0:
                    candidates = range(channel_count)
        if candidates is None:
            candidates = cls._snapshot_field(configuration_snapshot, "mic_channels")
        normalized = cls._normalize_channels(candidates)
        return tuple(normalized or [0])

    @staticmethod
    def _normalize_channels(channels: Any) -> list[int]:
        if channels is None:
            return []
        result: list[int] = []
        try:
            iterator = iter(channels)
        except TypeError:
            return []
        for channel in iterator:
            try:
                value = int(channel)
            except (TypeError, ValueError, OverflowError):
                continue
            if value >= 0:
                result.append(value)
        return result

    def _execute_admitted(self, command: AnalysisRequested) -> Mapping[str, Any]:
        context = self._active_context
        if context is None:
            raise RuntimeError("analysis execution context is unavailable")
        self.run(prepare_downstream=False, readiness_checked=True, context=context)
        runtime = self.runtime
        result_dict = dict(
            getattr(getattr(runtime, "data_struct", None), "analysis_result_dict", {})
            or {}
        )
        record_id = None
        if isinstance(command.recording_snapshot, Mapping):
            record_id = command.recording_snapshot.get("record_id")
        record_id = record_id or command.source_id
        snapshot: dict[str, Any] = {
            "record_id": record_id,
            "analysis_id": command.analysis_id,
            "source_id": command.source_id,
            "workflow_generation": command.workflow_generation,
            "automatic": command.automatic,
            "analysis_result_dict": result_dict,
        }
        now = self.timestamp_provider()
        summary = self.summarize_ok_ng(result_dict)
        snapshot["ok_ng_summary"] = summary
        snapshot["can_output_ok_ng"] = self.can_output_ok_ng(
            context.analysis_config
        )[0]
        snapshot["test_mode"] = context.test_mode
        snapshot["tcp_result_payload"] = self.build_tcp_result_payload(
            context.recording_snapshot, result_dict, now
        )
        snapshot["export_handoff"] = self.build_export_handoff(
            context, result_dict, now
        )
        return snapshot

    @staticmethod
    def summarize_ok_ng(result_dict: Any) -> tuple[bool, str]:
        if not isinstance(result_dict, Mapping) or not result_dict:
            return False, "NG"
        for value in result_dict.values():
            try:
                if not bool(value[0]):
                    return False, "NG"
            except (IndexError, KeyError, TypeError):
                return False, "NG"
        return True, "OK"

    @staticmethod
    def can_output_ok_ng(analysis_config: Any) -> tuple[bool, str]:
        if not isinstance(analysis_config, Mapping):
            return False, "当前配置未选择任何分析项"
        sequence = analysis_config.get("display_sequence")
        if not isinstance(sequence, (list, tuple)) or not sequence:
            return False, "当前配置未选择任何分析项"
        for key in sequence:
            item = analysis_config.get(key)
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "AI" or (
                item_type in ("SPL", "SPLF", "FFT", "FR", "HD", "RB", "PRB")
                and item.get("limit_checked")
            ):
                return True, ""
        return False, "当前配置未启用阈值对比，无法产出OK/NG"

    @classmethod
    def build_tcp_result_payload(
        cls, recording_snapshot: Any, result_dict: Any, now: datetime
    ) -> dict[str, str]:
        return {
            "TimeStamp": (
                f"{now.strftime('%Y-%m-%d %H:%M:%S')},"
                f"{now.microsecond // 1000:03d}"
            ),
            "Label": cls.tcp_result_label(result_dict),
            "FileName": cls.recorded_file_name(recording_snapshot),
        }

    @classmethod
    def tcp_result_label(cls, result_dict: Any) -> str:
        if not isinstance(result_dict, Mapping) or not result_dict:
            return "not_labeled"
        _passed, label = cls.summarize_ok_ng(result_dict)
        return label if label in {"OK", "NG"} else "not_labeled"

    @classmethod
    def recorded_file_name(cls, recording_snapshot: Any) -> str:
        file_path = cls._recorded_file_path(recording_snapshot)
        normalized = str(file_path or "").replace("\\", "/")
        return os.path.basename(normalized) if normalized else ""

    @staticmethod
    def _recorded_file_path(recording_snapshot: Any) -> Any:
        if not isinstance(recording_snapshot, Mapping):
            return None
        direct = recording_snapshot.get("recorded_path") or recording_snapshot.get(
            "file_path"
        )
        if direct:
            return direct
        info = recording_snapshot.get("recorded_signal_info")
        return info.get("file_path") if isinstance(info, Mapping) else None

    def build_export_handoff(
        self,
        context: AnalysisExecutionContext,
        result_dict: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        recording = context.recording_snapshot
        info = recording.get("recorded_signal_info") if isinstance(recording, Mapping) else None
        session = recording.get("session") if isinstance(recording, Mapping) else None
        analysis_items_data: dict[str, Any] = {}
        config = context.analysis_config
        if isinstance(config, Mapping):
            for instance in self.model.analysis_instances:
                key = getattr(instance, "_sequence_analysis_key", None)
                item_config = config.get(key) if key else None
                if not isinstance(item_config, Mapping):
                    continue
                item_type = item_config.get("type")
                if not item_type or item_type == "Excel":
                    continue
                item = {"type": item_type, "result": getattr(instance, "result", None)}
                detail = getattr(instance, "export_detail", None)
                if isinstance(detail, Mapping):
                    item.update(dict(detail))
                analysis_items_data[key] = item
        return {
            "record_id": self._recorded_file_path(recording),
            "sn": info.get("barcode", "") if isinstance(info, Mapping) else "",
            "product_model": (
                session.get("product_model", "") if isinstance(session, Mapping) else ""
            ),
            "date_text": f"{now.year}/{now.month}/{now.day} {now.strftime('%H:%M:%S')}",
            "analysis_items_data": analysis_items_data,
            "analysis_result_dict": dict(result_dict),
            "analysis_config": context.analysis_config,
            "ok_ng_summary": self.summarize_ok_ng(result_dict),
            "can_output_ok_ng": self.can_output_ok_ng(context.analysis_config)[0],
        }

    def run(
        self,
        *,
        prepare_downstream: bool = True,
        readiness_checked: bool = False,
        analysis_config_override: Any = None,
        sequence_config_override: Any = None,
        context: AnalysisExecutionContext | None = None,
    ) -> bool:
        """Execute analysis synchronously on the existing UI-thread boundary."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("analysis runtime is unavailable")
        owns_context = self._active_context is None
        if context is None:
            context = self._legacy_context(
                analysis_config=analysis_config_override,
                sequence_config=sequence_config_override,
            )
        if owns_context:
            self._active_context = context
        if not readiness_checked:
            probe = AnalysisRequested(
                "compatibility-analysis",
                str(getattr(runtime, "recorded_path", None) or "compatibility-source"),
                {},
                {
                    "sequence_config": context.sequence_config,
                    "analysis_config": context.analysis_config,
                },
                False,
            )
            ready, _reason = self._normalize_readiness(self._runtime_readiness(probe))
            if not ready:
                self._present_readiness_failure(
                    _reason or "import stimulus reference is unavailable"
                )
                if owns_context:
                    self._active_context = None
                return False
        try:
            runtime.data_struct.analysis_result_dict.clear()
            self.view.reset_output()
            screen_size = runtime.screen().size()
            width = int((screen_size.width() - 400) / 3)
            height = int((screen_size.height() - 400) / 3)
            window_width = ui_style_const.scale_size_px(600)
            window_height = ui_style_const.scale_size_px(500)

            analysis_config = context.analysis_config
            if analysis_config:
                item_sort_list = analysis_config.get("display_sequence", [])
                messages, uncalibrated_channels = self._instantiate_configured_items(
                    item_sort_list
                )
                suppress_missing = (
                    self.MISSING_CALIBRATION_MESSAGE in messages
                    and self.warning_suppressed(logger=self.logger)
                )
                self.view.present_calibration_warnings(
                    messages,
                    missing_message=self.MISSING_CALIBRATION_MESSAGE,
                    suppress_missing=suppress_missing,
                    record_only_channels=(
                        uncalibrated_channels
                        if context.mode == "RECORD_ONLY"
                        else []
                    ),
                    channel_formatter=format_input_channel_label,
                )

                for instance in list(self.model.analysis_instances):
                    instance_key = getattr(instance, "_sequence_analysis_key", None)
                    mismatch_info = getattr(instance, "_channel_mismatch_info", None)
                    if getattr(instance, "_channel_mismatch", False):
                        self.view.show_channel_mismatch(
                            instance_key or "分析项", mismatch_info=mismatch_info
                        )
                        continue
                    if not self._calculate(instance):
                        continue
                    self.view.show_instance(
                        instance,
                        key=instance_key,
                        default_geometry={
                            "x": width,
                            "y": height,
                            "w": window_width,
                            "h": window_height,
                        },
                    )
                    width += 20
                    height += 20

                if prepare_downstream:
                    runtime._handle_post_analysis_exports()
                    self._complete_legacy_test_mode()

            self.view.show_summary(
                getattr(runtime.data_struct, "analysis_result_dict", {}), width, height
            )
            if prepare_downstream and self.transport_service is not None:
                payload = self.build_tcp_result_payload(
                    {
                        "recorded_path": getattr(runtime, "recorded_path", None),
                        "recorded_signal_info": getattr(
                            runtime, "recorded_signal_info", None
                        ),
                    },
                    getattr(runtime.data_struct, "analysis_result_dict", None),
                    self.timestamp_provider(),
                )
                self.transport_service.send_payload(payload)
            return True
        finally:
            if owns_context:
                self._active_context = None

    @staticmethod
    def _snapshot_field(snapshot: Any, field: str) -> Any:
        if isinstance(snapshot, Mapping):
            return snapshot.get(field)
        return getattr(snapshot, field, None)

    def _instantiate_configured_items(
        self, item_sort_list: Any
    ) -> tuple[list[str], list[int]]:
        runtime = self.runtime
        messages: list[str] = []
        seen_messages: set[str] = set()
        uncalibrated_channels: list[int] = []
        seen_uncalibrated: set[int] = set()

        def collect_once(message: str) -> None:
            if message not in seen_messages:
                seen_messages.add(message)
                messages.append(message)

        context = self._active_context or self._legacy_context()
        if context.mode == "RECORD_ONLY":
            def resolver(raw_channel: int, warn_callback: Callable[[str], None] | None = None):
                def capture(message: str) -> None:
                    if (
                        message == self.MISSING_CALIBRATION_MESSAGE
                        and raw_channel not in seen_uncalibrated
                    ):
                        seen_uncalibrated.add(raw_channel)
                        uncalibrated_channels.append(raw_channel)
                    if warn_callback is not None:
                        warn_callback(message)

                return self.calibration_resolver(
                    raw_channel, warn_callback=capture
                )

            batch = self.calibration_batch_factory(resolver=resolver)
        else:
            batch = self.calibration_batch_factory()

        missing = object()
        previous_batch = getattr(runtime, "_analysis_v2pa_batch", missing)
        previous_callback = getattr(
            runtime, "_analysis_v2pa_warning_callback", missing
        )
        runtime._analysis_v2pa_batch = batch
        runtime._analysis_v2pa_warning_callback = collect_once
        try:
            active_config = context.analysis_config
            if not isinstance(active_config, Mapping):
                active_config = {}
            for key in item_sort_list:
                config = active_config.get(key)
                if not isinstance(config, Mapping):
                    continue
                if self.instance_factory is None:
                    self.instance_analysis_class(
                        key,
                        config.get("type"),
                        config,
                        calibration_batch=batch,
                        warning_callback=collect_once,
                    )
                else:
                    self.instance_factory(
                        key, config.get("type"), config, batch, collect_once
                    )
                for message in batch.messages:
                    collect_once(message)
        finally:
            if previous_batch is missing:
                del runtime._analysis_v2pa_batch
            else:
                runtime._analysis_v2pa_batch = previous_batch
            if previous_callback is missing:
                del runtime._analysis_v2pa_warning_callback
            else:
                runtime._analysis_v2pa_warning_callback = previous_callback
        self.model.set_calibration_results(
            getattr(batch, "preparations", ()) or ()
        )
        return messages, uncalibrated_channels

    def instance_analysis_class(
        self,
        key: str,
        analysis_type: str,
        params: Mapping[str, Any],
        *,
        calibration_batch: Any = None,
        warning_callback: Callable[[str], None] | None = None,
    ) -> Any:
        """Create one configured analysis instance using active-channel mapping."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("analysis runtime is unavailable")
        class_mapping = self.class_mapping_provider()
        factory = class_mapping.get(analysis_type)
        if factory is None:
            return None
        if calibration_batch is None:
            calibration_batch = getattr(runtime, "_analysis_v2pa_batch", None)
        if warning_callback is None:
            warning_callback = getattr(
                runtime, "_analysis_v2pa_warning_callback", None
            )
        context = self._active_context or self._legacy_context()
        requires_v2pa = (
            analysis_type in context.calibration_types
            and analysis_type not in {"HD", "RB"}
        )
        use_batch = calibration_batch is not None and analysis_type not in {
            "PD",
            "PM",
            "ED",
        }
        warning_callback = warning_callback or (
            lambda message: self.view.warning_presenter("提示", message)
        )
        channel_config: Mapping[str, Any] = params if isinstance(params, Mapping) else {}
        if analysis_type == "ED":
            head = channel_config.get("head")
            if isinstance(head, Mapping) and isinstance(head.get("config"), Mapping):
                channel_config = head["config"]
        raw_channel = self._raw_channel(channel_config)

        if context.mode == "RECORD_ONLY":
            active_channels = list(context.active_channels)
            mismatch = raw_channel not in active_channels
            mapped_channel = (
                int(active_channels.index(raw_channel)) if not mismatch else 0
            )
            instance = factory(f"{key}--通道{raw_channel + 1}")
            instance.data_struct = runtime.data_struct
            setattr(instance, "_channel_mismatch", mismatch)
            setattr(
                instance,
                "_channel_mismatch_info",
                {
                    "raw_channel": raw_channel,
                    "active_input_channels": list(active_channels),
                },
            )
            if requires_v2pa and not mismatch:
                if not self._prepare_v2pa(
                    instance,
                    raw_channel,
                    calibration_batch,
                    use_batch,
                    warning_callback,
                ):
                    return None
            runtime_params = dict(params) if isinstance(params, Mapping) else {}
            runtime_params["analysis_channel"] = mapped_channel
        else:
            instance = factory(key)
            if requires_v2pa and context.mode not in {
                "IMPORT_AUDIO",
                "IMPORT_STIMULUS_AUDIO",
            }:
                if not self._prepare_v2pa(
                    instance,
                    raw_channel,
                    calibration_batch,
                    use_batch,
                    warning_callback,
                ):
                    return None
            runtime_params = dict(params) if isinstance(params, Mapping) else {}

        setattr(instance, "_sequence_analysis_key", key)
        analysis_config = context.analysis_config
        if isinstance(analysis_config, Mapping):
            golden_path = analysis_config.get(GOLDEN_SAMPLE_RESULT_PATH_KEY)
            if golden_path:
                runtime_params[GOLDEN_SAMPLE_RESULT_PATH_KEY] = golden_path
        instance.analysis_config = runtime_params
        self.model.register_instance(key, instance)
        return instance

    @staticmethod
    def _raw_channel(config: Mapping[str, Any]) -> int:
        try:
            channel = int(config.get("analysis_channel", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, channel)

    @staticmethod
    def _active_channels(runtime: Any, *, allow_live: bool = False) -> list[int]:
        if not allow_live:
            return [0]
        channels = getattr(runtime, "_active_input_channels", None) or [0]
        result: list[int] = []
        for channel in channels:
            try:
                result.append(int(channel))
            except (TypeError, ValueError, OverflowError):
                continue
        return result or [0]

    def _prepare_v2pa(
        self,
        instance: Any,
        raw_channel: int,
        batch: Any,
        use_batch: bool,
        warning_callback: Callable[[str], None],
    ) -> bool:
        if use_batch:
            preparation = batch.resolve(raw_channel)
            if preparation.factor is None:
                return False
            instance.v2pa_factor = preparation.factor
        else:
            try:
                instance.v2pa_factor = self.calibration_resolver(
                    raw_channel, warn_callback=warning_callback
                )
            except ValueError as error:
                warning_callback(str(error))
                return False
        if hasattr(instance, "_resolve_v2pa_factor_for_analysis") or getattr(
            instance, "_supports_pre_resolved_v2pa_factor", False
        ):
            instance._v2pa_raw_analysis_channel = raw_channel
            instance._use_pre_resolved_v2pa_factor = True
        return True

    def _calculate(self, instance: Any) -> bool:
        runtime = self.runtime
        if hasattr(instance, "calculate_spl"):
            return bool(instance.calculate_spl())
        if hasattr(instance, "calculate_fr"):
            return bool(instance.calculate_fr())
        if hasattr(instance, "calculate_fft"):
            return bool(instance.calculate_fft())
        if hasattr(instance, "calculate_thd"):
            instance.calculate_thd()
        elif hasattr(instance, "calculate_ai_scores"):
            context = self._active_context or self._legacy_context()
            sequence_config = context.sequence_config
            instance.calculate_ai_scores(
                runtime.count_board.mode,
                context.analysis_config,
                context.mode,
            )
        elif hasattr(instance, "calculate_spec"):
            return bool(instance.calculate_spec())
        elif hasattr(instance, "calculate_reference_spectrum"):
            return bool(instance.calculate_reference_spectrum())
        elif hasattr(instance, "calculate_peak_detection"):
            instance.calculate_peak_detection()
        elif hasattr(instance, "calculate_loose_particle"):
            instance.calculate_loose_particle()
        elif hasattr(instance, "calculate_pattern_match"):
            instance.calculate_pattern_match()
        elif hasattr(instance, "calculate_pipeline_pd_pm"):
            instance.calculate_pipeline_pd_pm()
        elif hasattr(instance, "calculate_fba"):
            return bool(instance.calculate_fba())
        elif hasattr(instance, "calculate_loudness"):
            return bool(instance.calculate_loudness())
        return True

    def _complete_legacy_test_mode(self) -> None:
        runtime = self.runtime
        if runtime.count_board.mode != "test":
            return
        context = self._active_context or self._legacy_context()
        can_output, _reason = self.can_output_ok_ng(context.analysis_config)
        if not can_output:
            self.view.warning_presenter(
                "提示", "当前配置无法产出 OK/NG 汇总结果，无法执行测试模式自动判定。"
            )
            return
        _passed, label = self.summarize_ok_ng(
            getattr(runtime.data_struct, "analysis_result_dict", None)
        )
        workflow_model = getattr(runtime, "workflow_model", None)
        record_id = getattr(workflow_model, "retained_record_id", None)
        if not record_id:
            info = getattr(runtime, "recorded_signal_info", None)
            if isinstance(info, Mapping):
                record_id = info.get("file_path")
        record_id = str(record_id or getattr(runtime, "recorded_path", "") or "")
        if not record_id:
            self._log("warning", "test-mode label has no retained recording")
            return
        self.bus.commands.manual_label_requested.emit(
            ManualLabelRequested(
                f"test-mode-label-{uuid4().hex}",
                record_id,
                label,
            )
        )
