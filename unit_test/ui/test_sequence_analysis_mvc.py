from __future__ import annotations

import ast
import os
import sys
import threading
import tracemalloc
import types
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

try:
    import base.analysis_warning_preferences  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "base.analysis_warning_preferences":
        raise
    warning_preferences = types.ModuleType("base.analysis_warning_preferences")
    warning_preferences.is_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: False
    )
    warning_preferences.save_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: None
    )
    sys.modules[warning_preferences.__name__] = warning_preferences

from base.excel_result_exporter import ExportResult
from base.core_algorithm.harmonic_distortion.harmonic_index_builder import (
    HarmonicIndexBuilder,
)
from base.data_struct.data_deal_struct import DataDealStruct
from base.pre_processing.audio_thd_frequency_response_analysis import (
    AudioThdFrequencyResponseAnalysis,
)
from ui.signal_analysis_window import Distortion, RubAndBuzz
from ui.sequence.sequence_analysis_controller import (
    AnalysisExecutionContext,
    SequenceAnalysisController,
    SequenceAnalysisTransportController,
)
from ui.sequence.sequence_analysis_model import (
    AnalysisCalibrationPolicySnapshot,
    AnalysisState,
    MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH,
    MAX_ANALYSIS_CALIBRATION_TYPES,
    SequenceAnalysisCalibrationPolicyService,
    SequenceAnalysisModel,
    mutable_analysis_value,
)
from ui.sequence.sequence_analysis_view import SequenceAnalysisView
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import SequenceExportModel
from ui.sequence.sequence_export_service import SequenceExportService
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_import_owner import (
    SequenceRecordingImportController,
)
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_view import (
    SequenceRecordingImportView,
    SequenceRecordingView,
)
from ui.sequence.sequence_resource_lifecycle_controller import (
    SequenceResourceLifecycleController,
    SequenceResourceLifecycleModel,
    SequenceResourceLifecycleView,
)
from ui.sequence import sequence_messages as sequence_message_types
from ui.sequence.sequence_messages import (
    AnalysisCompleted,
    AnalysisFailed,
    AnalysisRequested,
    CancelAnalysisRequested,
    CancelWorkflowRequested,
    ConfigurationSnapshot,
    CommitRecordingLabelRequested,
    ExportCompleted,
    ExportFailed,
    ExportRequested,
    ExportRetryAccepted,
    IgnoreExportFailureRequested,
    ImportedAudioReady,
    ManualAnalysisRequested,
    ManualLabelRequested,
    RecordingCompleted,
    RecordingLabelCommitted,
    RetryExportRequested,
    StartTestRequested,
)
from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import (
    SequenceWorkflowModel,
    SessionOrigin,
    WorkflowPhase,
)
from ui.sequence.sequence_workflow_policy import (
    AutomaticAnalysisDecision,
    AutomaticAnalysisSource,
)


class _StaticAutomaticAnalysisPolicy:
    def __init__(self, recorded: bool) -> None:
        self.recorded = recorded

    def decide_recorded(self, *, workflow_generation, **_kwargs):
        return AutomaticAnalysisDecision(
            workflow_generation,
            AutomaticAnalysisSource.RECORDED,
            "RECORD_ONLY",
            self.recorded,
            "test policy",
        )

    def decide_imported(self, *, workflow_generation, **_kwargs):
        return AutomaticAnalysisDecision(
            workflow_generation,
            AutomaticAnalysisSource.IMPORTED,
            None,
            True,
            "test policy",
        )


_QAPP = QApplication.instance() or QApplication([])


def _analysis_transport_delivery_id(event):
    return (
        "analysis-transport",
        event.analysis_id,
        event.source_id,
        event.record_id,
        event.workflow_generation,
    )


def test_analysis_transport_reserves_before_reentrant_event_bus_delivery():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-reentrant", "source", "record", 4, {"Label": "OK"}
    )
    delivery_id = _analysis_transport_delivery_id(event)
    nested_results = []
    sends = []
    consumes = []

    class ReentrantService:
        def send_payload(self, payload):
            sends.append(payload)
            if len(sends) == 1:
                nested_results.append(
                    bus.deliver_workflow_continuation(
                        delivery_id,
                        "analysis-transport",
                        event,
                        owner=bus,
                    )
                )
            return True

    def consume(candidate):
        consumes.append(candidate)
        return model.consume_analysis_transport(candidate)

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=consume,
        service=ReentrantService(),
    )
    assert model.authorize_analysis_transport(event)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is True
    assert nested_results == [False]
    assert sends == [event.payload]
    assert consumes == [event]
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert owner.handle_analysis_transport_ready(event) is False


def test_analysis_transport_concurrent_delivery_has_one_external_side_effect():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-concurrent", "source", "record", 5, {"Label": "NG"}
    )
    send_entered = threading.Event()
    release_send = threading.Event()
    sends = []
    consumes = []
    outcomes = []

    class BarrierService:
        def send_payload(self, payload):
            sends.append(payload)
            send_entered.set()
            assert release_send.wait(5)
            return True

    def consume(candidate):
        consumes.append(candidate)
        return model.consume_analysis_transport(candidate)

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=consume,
        service=BarrierService(),
    )
    assert model.authorize_analysis_transport(event)
    first = threading.Thread(
        target=lambda: outcomes.append(
            owner.handle_analysis_transport_ready(event)
        )
    )
    first.start()
    assert send_entered.wait(5)

    assert owner.handle_analysis_transport_ready(event) is False
    release_send.set()
    first.join(5)

    assert not first.is_alive()
    assert outcomes == [True]
    assert sends == [event.payload]
    assert consumes == [event]


def test_analysis_transport_send_failure_releases_reservation_for_retry():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-send-retry", "source", "record", 6, {"Label": "OK"}
    )
    outcomes = iter((False, True))
    sends = []

    class RetryService:
        def send_payload(self, payload):
            sends.append(payload)
            return next(outcomes)

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        service=RetryService(),
    )
    assert model.authorize_analysis_transport(event)

    assert owner.handle_analysis_transport_ready(event) is False
    assert owner.handle_analysis_transport_ready(event) is True
    assert sends == [event.payload, event.payload]


@pytest.mark.parametrize(
    "first_consume_outcome",
    (
        False,
        RuntimeError("ordinary consume failure"),
        KeyboardInterrupt("consume interrupted"),
        SystemExit("consume exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_analysis_transport_consume_retry_does_not_resend_successful_payload(
    first_consume_outcome,
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-consume-retry", "source", "record", 7, {"Label": "OK"}
    )
    sends = []
    consume_calls = []

    class Service:
        def send_payload(self, payload):
            sends.append(payload)
            return True

    def consume(candidate):
        consume_calls.append(candidate)
        if len(consume_calls) == 1:
            if isinstance(first_consume_outcome, BaseException):
                raise first_consume_outcome
            return first_consume_outcome
        return model.consume_analysis_transport(candidate)

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=consume,
        service=Service(),
    )
    assert model.authorize_analysis_transport(event)
    delivery_id = _analysis_transport_delivery_id(event)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is False
    assert model.is_analysis_transport_authorized(event)
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert sends == [event.payload]
    assert consume_calls == [event, event]


def test_analysis_transport_disconnect_rejects_new_direct_delivery():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    sends = []
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-disconnect", "source", "record", 8, {"Label": "OK"}
    )
    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        service=SimpleNamespace(
            send_payload=lambda payload: sends.append(payload) or True
        ),
    )
    assert model.authorize_analysis_transport(event)

    assert owner.disconnect() is True
    assert owner.handle_analysis_transport_ready(event) is False
    assert sends == []
    assert model.is_analysis_transport_authorized(event)


def test_analysis_transport_conflicting_object_cannot_reuse_pending_identity():
    bus = SequenceEventBus()
    original = sequence_message_types.AnalysisTransportReady(
        "analysis-conflict", "source", "record", 9, {"Label": "OK"}
    )
    conflict = sequence_message_types.AnalysisTransportReady(
        "analysis-conflict", "source", "record", 9, {"Label": "NG"}
    )
    sends = []
    consume_calls = []

    def consume(candidate):
        consume_calls.append(candidate)
        return len(consume_calls) > 1

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=lambda _event: True,
        authorization_consumer=consume,
        service=SimpleNamespace(
            send_payload=lambda payload: sends.append(payload) or True
        ),
    )

    assert owner.handle_analysis_transport_ready(original) is False
    assert owner.handle_analysis_transport_ready(conflict) is False
    assert owner.handle_analysis_transport_ready(original) is True
    assert sends == [original.payload]
    assert consume_calls == [original, original]


def test_analysis_transport_disconnect_during_send_finishes_started_delivery_once():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-disconnect-in-flight", "source", "record", 10, {"Label": "OK"}
    )
    sends = []
    consumes = []
    owner = None

    def send(payload):
        sends.append(payload)
        assert owner.disconnect() is True
        return True

    def consume(candidate):
        consumes.append(candidate)
        return model.consume_analysis_transport(candidate)

    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=consume,
        service=SimpleNamespace(send_payload=send),
    )
    assert model.authorize_analysis_transport(event)

    assert owner.handle_analysis_transport_ready(event) is True
    assert owner.handle_analysis_transport_ready(event) is False
    assert sends == [event.payload]
    assert consumes == [event]


def test_analysis_transport_rejects_corrupted_identity_without_external_calls():
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-hostile-identity", "source", "record", 11, {"Label": "OK"}
    )
    object.__setattr__(event, "analysis_id", [])
    calls = []
    owner = SequenceAnalysisTransportController(
        bus=SequenceEventBus(),
        authorization_provider=lambda _event: calls.append("authorize") or True,
        authorization_consumer=lambda _event: calls.append("consume") or True,
        service=SimpleNamespace(
            send_payload=lambda _payload: calls.append("send") or True
        ),
    )

    assert owner.handle_analysis_transport_ready(event) is False
    assert calls == []


def _atomic_transport_owner(bus, model, service, **overrides):
    ports = {
        "authorization_claimer": model.claim_analysis_transport,
        "claim_releaser": model.release_analysis_transport_claim,
        "claim_committer": model.commit_analysis_transport_claim,
        "claim_abandoner": model.abandon_analysis_transport_claim,
    }
    ports.update(overrides)
    return SequenceAnalysisTransportController(
        bus=bus,
        service=service,
        **ports,
    )


def test_analysis_transport_disconnect_while_claim_blocks_releases_without_send():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-claim-disconnect", "source", "record", 12, {"Label": "OK"}
    )
    claim_acquired = threading.Event()
    return_claim = threading.Event()
    sends = []
    results = []

    def blocked_claim(candidate):
        claim = model.claim_analysis_transport(candidate)
        claim_acquired.set()
        assert return_claim.wait(5)
        return claim

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        authorization_claimer=blocked_claim,
    )
    assert model.authorize_analysis_transport(event)
    delivery_id = _analysis_transport_delivery_id(event)
    delivery = threading.Thread(
        target=lambda: results.append(
            bus.deliver_workflow_continuation(
                delivery_id,
                "analysis-transport",
                event,
                owner=bus,
            )
        )
    )
    delivery.start()
    assert claim_acquired.wait(5)

    assert owner.disconnect() is True
    return_claim.set()
    delivery.join(5)

    assert not delivery.is_alive()
    assert results == [False]
    assert sends == []
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert model.is_analysis_transport_authorized(event)
    retry_claim = model.claim_analysis_transport(event)
    assert retry_claim is not None
    assert model.release_analysis_transport_claim(retry_claim) is True


def test_workflow_analysis_transport_claim_excludes_legacy_consume_across_generation():
    model = SequenceWorkflowModel(workflow_generation=13)
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-atomic", "source", "record", 13, {"Label": "OK"}
    )
    assert model.authorize_analysis_transport(event)

    claim = model.claim_analysis_transport(event)
    assert claim is not None
    model.workflow_generation = 14
    assert model.consume_analysis_transport(event) is False
    assert model.commit_analysis_transport_claim(object()) is False
    assert model.commit_analysis_transport_claim(claim) is True
    assert model.release_analysis_transport_claim(claim) is False
    assert model.claim_analysis_transport(event) is None


def test_workflow_analysis_transport_revocation_before_claim_prevents_send():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-revoked", "source", "record", 14, {"Label": "NG"}
    )
    sends = []
    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
    )
    assert model.authorize_analysis_transport(event)
    assert model.consume_analysis_transport(event) is True

    assert owner.handle_analysis_transport_ready(event) is False
    assert sends == []


def test_workflow_analysis_transport_claim_and_revocation_are_one_atomic_winner():
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-claim-revoke-race", "source", "record", 141, None
    )
    barrier = threading.Barrier(2)
    results = {}

    def claim():
        barrier.wait(5)
        results["claim"] = model.claim_analysis_transport(event)

    def revoke():
        barrier.wait(5)
        results["revoke"] = model.consume_analysis_transport(event)

    assert model.authorize_analysis_transport(event)
    contenders = [threading.Thread(target=claim), threading.Thread(target=revoke)]
    for contender in contenders:
        contender.start()
    for contender in contenders:
        contender.join(5)

    assert all(not contender.is_alive() for contender in contenders)
    if results["claim"] is None:
        assert results["revoke"] is True
    else:
        assert results["revoke"] is False
        assert model.release_analysis_transport_claim(results["claim"]) is True


def test_atomic_analysis_transport_send_failure_releases_claim_for_retry():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-atomic-send-retry", "source", "record", 15, {"Label": "OK"}
    )
    outcomes = iter((False, True))
    sends = []

    def send(payload):
        sends.append(payload)
        return next(outcomes)

    owner = _atomic_transport_owner(
        bus, model, SimpleNamespace(send_payload=send)
    )
    assert model.authorize_analysis_transport(event)

    assert owner.handle_analysis_transport_ready(event) is False
    assert model.is_analysis_transport_authorized(event)
    assert owner.handle_analysis_transport_ready(event) is True
    assert sends == [event.payload, event.payload]
    assert not model.is_analysis_transport_authorized(event)


@pytest.mark.parametrize(
    "first_commit_outcome",
    (
        False,
        RuntimeError("ordinary commit failure"),
        KeyboardInterrupt("commit interrupted"),
        SystemExit("commit exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_atomic_analysis_transport_commit_retry_never_resends(
    first_commit_outcome,
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-atomic-commit-retry", "source", "record", 16, {"Label": "OK"}
    )
    sends = []
    commit_calls = []

    def commit(claim):
        commit_calls.append(claim)
        if len(commit_calls) == 1:
            if isinstance(first_commit_outcome, BaseException):
                raise first_commit_outcome
            return first_commit_outcome
        return model.commit_analysis_transport_claim(claim)

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=commit,
    )
    assert model.authorize_analysis_transport(event)
    delivery_id = _analysis_transport_delivery_id(event)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is True
    assert sends == [event.payload]
    assert len(commit_calls) == 2
    assert commit_calls[0] is commit_calls[1]


@pytest.mark.parametrize(
    "commit_outcome",
    (
        False,
        RuntimeError("ordinary commit failure"),
        KeyboardInterrupt("commit interrupted"),
        SystemExit("commit exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_disconnect_during_failed_commit_returns_terminal_ack(commit_outcome):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-commit-disconnect", "source", "record", 160, {"Label": "OK"}
    )
    commit_entered = threading.Event()
    finish_commit = threading.Event()
    sends = []
    results = []

    def commit(_claim):
        commit_entered.set()
        assert finish_commit.wait(5)
        if isinstance(commit_outcome, BaseException):
            raise commit_outcome
        return commit_outcome

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=commit,
    )
    assert model.authorize_analysis_transport(event)
    delivery_id = _analysis_transport_delivery_id(event)
    delivery = threading.Thread(
        target=lambda: results.append(
            bus.deliver_workflow_continuation(
                delivery_id,
                "analysis-transport",
                event,
                owner=bus,
            )
        )
    )
    delivery.start()
    assert commit_entered.wait(5)

    assert owner.disconnect() is True
    finish_commit.set()
    delivery.join(5)

    assert not delivery.is_alive()
    assert results == [True]
    assert sends == [event.payload]
    assert model.claim_analysis_transport(event) is None
    assert bus.pending_workflow_continuation_delivery_count == 0


@pytest.mark.parametrize(
    "commit_outcome",
    (
        False,
        RuntimeError("ordinary commit failure"),
        KeyboardInterrupt("commit interrupted"),
        SystemExit("commit exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_disconnect_after_pending_commit_write_is_owned_by_active_handler(
    commit_outcome,
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-pending-write-disconnect",
        "source",
        "record",
        1_602,
        {"Label": "OK"},
    )
    pending_written = threading.Event()
    finish_handler = threading.Event()
    sends = []
    abandon_calls = []
    results = []

    def commit(_claim):
        if isinstance(commit_outcome, BaseException):
            raise commit_outcome
        return commit_outcome

    def abandon(claim):
        abandon_calls.append(claim)
        return model.abandon_analysis_transport_claim(claim)

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=commit,
        claim_abandoner=abandon,
    )
    original_commit = owner._commit_reserved_claim

    def commit_then_block(identity, reservation):
        result = original_commit(identity, reservation)
        assert result is False
        pending_written.set()
        assert finish_handler.wait(5)
        return result

    owner._commit_reserved_claim = commit_then_block
    assert model.authorize_analysis_transport(event)
    delivery = threading.Thread(
        target=lambda: results.append(
            bus.deliver_workflow_continuation(
                _analysis_transport_delivery_id(event),
                "analysis-transport",
                event,
                owner=bus,
            )
        )
    )
    delivery.start()
    assert pending_written.wait(5)

    assert owner.disconnect() is True
    finish_handler.set()
    delivery.join(5)

    assert not delivery.is_alive()
    assert results == [True]
    assert sends == [event.payload]
    assert len(abandon_calls) == 1
    assert model.claim_analysis_transport(event) is None
    assert bus.pending_workflow_continuation_delivery_count == 0


def test_disconnect_settles_recipient_after_handler_already_returned_nack():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-old-pending-disconnect",
        "source",
        "record",
        1_603,
        {"Label": "OK"},
    )
    sends = []
    abandon_calls = []

    def abandon(claim):
        abandon_calls.append(claim)
        return model.abandon_analysis_transport_claim(claim)

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=lambda _claim: False,
        claim_abandoner=abandon,
    )
    assert model.authorize_analysis_transport(event)

    assert bus.deliver_workflow_continuation(
        _analysis_transport_delivery_id(event),
        "analysis-transport",
        event,
        owner=bus,
    ) is False
    assert bus.pending_workflow_continuation_delivery_count == 1

    assert owner.disconnect() is True
    assert owner.disconnect() is False

    assert sends == [event.payload]
    assert len(abandon_calls) == 1
    assert model.claim_analysis_transport(event) is None
    assert bus.pending_workflow_continuation_delivery_count == 0


def test_failed_disconnect_abandon_retains_explicit_pending_claim_ownership():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-abandon-rejected", "source", "record", 1601, {"Label": "OK"}
    )
    commit_entered = threading.Event()
    finish_commit = threading.Event()
    results = []

    def commit(_claim):
        commit_entered.set()
        assert finish_commit.wait(5)
        return False

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda _payload: True),
        claim_committer=commit,
        claim_abandoner=lambda _claim: False,
    )
    assert model.authorize_analysis_transport(event)
    delivery = threading.Thread(
        target=lambda: results.append(owner.handle_analysis_transport_ready(event))
    )
    delivery.start()
    assert commit_entered.wait(5)

    assert owner.disconnect() is True
    finish_commit.set()
    delivery.join(5)

    assert not delivery.is_alive()
    assert results == [False]
    reservation = owner._reservations[_analysis_transport_delivery_id(event)[1:]]
    assert reservation.event is event
    assert reservation.claim is not None
    assert reservation.phase.name == "ABANDON_PENDING"


def test_atomic_commit_retry_is_not_truncated_by_applied_history_limit():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-small-history", "source", "record", 161, {"Label": "OK"}
    )
    sends = []
    commits = []

    def commit(claim):
        commits.append(claim)
        if len(commits) == 1:
            return False
        return model.commit_analysis_transport_claim(claim)

    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=commit,
        history_limit=1,
    )
    assert model.authorize_analysis_transport(event)

    assert owner.handle_analysis_transport_ready(event) is False
    assert owner.handle_analysis_transport_ready(event) is True
    assert sends == [event.payload]
    assert commits[0] is commits[1]


def test_concurrent_atomic_analysis_transport_claims_send_exactly_once():
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-atomic-concurrent", "source", "record", 17, {"Label": "OK"}
    )
    claim_barrier = threading.Barrier(2)
    sends = []
    results = []

    def claim(candidate):
        claim_barrier.wait(5)
        return model.claim_analysis_transport(candidate)

    owners = [
        _atomic_transport_owner(
            SequenceEventBus(),
            model,
            SimpleNamespace(
                send_payload=lambda payload: sends.append(payload) or True
            ),
            authorization_claimer=claim,
        )
        for _index in range(2)
    ]
    assert model.authorize_analysis_transport(event)
    deliveries = [
        threading.Thread(
            target=lambda owner=owner: results.append(
                owner.handle_analysis_transport_ready(event)
            )
        )
        for owner in owners
    ]
    for delivery in deliveries:
        delivery.start()
    for delivery in deliveries:
        delivery.join(5)

    assert all(not delivery.is_alive() for delivery in deliveries)
    assert sorted(results) == [False, True]
    assert sends == [event.payload]


@pytest.mark.parametrize("retirement", ("disconnect", "destroyed"))
def test_transport_retirement_abandons_sent_pending_claim_without_replay(
    retirement,
):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-pending-disconnect", "source", "record", 18, {"Label": "OK"}
    )
    sends = []
    owner = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(send_payload=lambda payload: sends.append(payload) or True),
        claim_committer=lambda _claim: False,
    )
    assert model.authorize_analysis_transport(event)
    assert owner.handle_analysis_transport_ready(event) is False

    if retirement == "disconnect":
        assert owner.disconnect() is True
    else:
        owner._handle_destroyed()
    assert not model.is_analysis_transport_authorized(event)
    assert model.claim_analysis_transport(event) is None
    assert sends == [event.payload]


def test_workflow_analysis_transport_capacity_never_evicts_active_authorizations():
    model = SequenceWorkflowModel()
    events = []
    claims = []
    for generation in range(model.ANALYSIS_TRANSPORT_HISTORY_LIMIT):
        event = sequence_message_types.AnalysisTransportReady(
            f"analysis-capacity-{generation}",
            "source",
            "record",
            generation,
            None,
        )
        assert model.authorize_analysis_transport(event)
        claim = model.claim_analysis_transport(event)
        assert claim is not None
        events.append(event)
        claims.append(claim)
    extra = sequence_message_types.AnalysisTransportReady(
        "analysis-capacity-extra", "source", "record", 999, None
    )

    assert model.authorize_analysis_transport(extra) is False
    assert model.release_analysis_transport_claim(claims[0]) is True
    assert model.authorize_analysis_transport(extra) is False
    restored = model.claim_analysis_transport(events[0])
    assert restored is not None
    assert model.commit_analysis_transport_claim(restored) is True
    assert model.authorize_analysis_transport(extra) is True
    extra_claim = model.claim_analysis_transport(extra)
    assert extra_claim is not None
    assert model.abandon_analysis_transport_claim(extra_claim) is True
    replacement = sequence_message_types.AnalysisTransportReady(
        "analysis-capacity-replacement", "source", "record", 1_000, None
    )
    assert model.authorize_analysis_transport(replacement) is True


def test_workflow_analysis_transport_capacity_reclaims_only_retired_tombstone():
    model = SequenceWorkflowModel()
    events = []
    for generation in range(model.ANALYSIS_TRANSPORT_HISTORY_LIMIT):
        event = sequence_message_types.AnalysisTransportReady(
            f"analysis-pending-{generation}",
            "source",
            "record",
            generation,
            None,
        )
        assert model.authorize_analysis_transport(event)
        events.append(event)
    retained_order = tuple(model._analysis_transport_authorizations)
    extra = sequence_message_types.AnalysisTransportReady(
        "analysis-pending-extra", "source", "record", 999, None
    )

    model.workflow_generation += 1
    assert model.authorize_analysis_transport(extra) is False
    assert tuple(model._analysis_transport_authorizations) == retained_order
    claims = [model.claim_analysis_transport(event) for event in events]
    assert all(claim is not None for claim in claims)
    for claim in claims:
        assert model.release_analysis_transport_claim(claim) is True

    assert model.consume_analysis_transport(events[17]) is True
    assert model.authorize_analysis_transport(extra) is True
    remaining = tuple(model._analysis_transport_authorizations)
    assert retained_order[17] not in remaining
    expected_active_order = tuple(
        identity for identity in retained_order if identity != retained_order[17]
    )
    assert expected_active_order == remaining[:-1]


def test_workflow_analysis_transport_concurrent_authorize_uses_one_retired_slot():
    model = SequenceWorkflowModel()
    active = []
    for generation in range(model.ANALYSIS_TRANSPORT_HISTORY_LIMIT):
        event = sequence_message_types.AnalysisTransportReady(
            f"analysis-concurrent-active-{generation}",
            "source",
            "record",
            generation,
            None,
        )
        assert model.authorize_analysis_transport(event)
        active.append(event)
    assert model.consume_analysis_transport(active[31]) is True
    candidates = [
        sequence_message_types.AnalysisTransportReady(
            f"analysis-concurrent-new-{index}",
            "source",
            "record",
            1_000 + index,
            None,
        )
        for index in range(16)
    ]
    barrier = threading.Barrier(len(candidates))
    results = []

    def authorize(candidate):
        barrier.wait()
        results.append((candidate, model.authorize_analysis_transport(candidate)))

    workers = [
        threading.Thread(target=authorize, args=(candidate,))
        for candidate in candidates
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)

    assert all(not worker.is_alive() for worker in workers)
    winners = [candidate for candidate, accepted in results if accepted]
    assert len(winners) == 1
    for index, event in enumerate(active):
        if index == 31:
            continue
        claim = model.claim_analysis_transport(event)
        assert claim is not None
        assert model.release_analysis_transport_claim(claim) is True
    assert model.claim_analysis_transport(winners[0]) is not None


@pytest.mark.parametrize(
    "transient_outcome",
    (
        False,
        RuntimeError("ordinary recipient failure"),
        KeyboardInterrupt("recipient interrupted"),
        SystemExit("recipient exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_capacity_retry_does_not_reauthorize_after_event_bus_delivery_started(
    transient_outcome,
):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    retained = []
    for generation in range(model.ANALYSIS_TRANSPORT_HISTORY_LIMIT):
        event = sequence_message_types.AnalysisTransportReady(
            f"analysis-retained-{generation}",
            "source",
            "record",
            generation,
            None,
        )
        assert model.authorize_analysis_transport(event)
        retained.append(event)
    target = sequence_message_types.AnalysisTransportReady(
        "analysis-target", "source", "record", 999, {"Label": "OK"}
    )
    sends = []
    transport = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(
            send_payload=lambda payload: sends.append(payload) or True
        ),
    )
    second_calls = []

    def second_recipient(message):
        second_calls.append(message)
        if len(second_calls) <= 3:
            if isinstance(transient_outcome, BaseException):
                raise transient_outcome
            return transient_outcome
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "second", second_recipient
    )
    delivery_id = _analysis_transport_delivery_id(target)

    assert controller._publish_continuation(
        delivery_id,
        bus.events.analysis_transport_ready,
        target,
        requires_analysis_transport_authorization=True,
    ) is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert model.consume_analysis_transport(retained[0]) is True

    for attempt in range(1, 4):
        assert controller.retry_pending_continuation_publications() is False
        assert sends == [target.payload]
        assert second_calls == [target] * attempt
        assert bus.pending_workflow_continuation_delivery_count == 1
    assert controller.retry_pending_continuation_publications() is True
    assert sends == [target.payload]
    assert second_calls == [target] * 4
    assert controller.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    transport.disconnect()
    controller.disconnect()


def test_disconnect_abandons_partial_transport_event_bus_delivery():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-partial-disconnect",
        "source",
        "record",
        1_002,
        {"Label": "OK"},
    )
    sends = []
    transport = _atomic_transport_owner(
        bus,
        model,
        SimpleNamespace(
            send_payload=lambda payload: sends.append(payload) or True
        ),
    )
    second_calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "blocked-second",
        lambda message: second_calls.append(message) or False,
    )
    assert model.authorize_analysis_transport(event)

    assert controller._publish_continuation(
        _analysis_transport_delivery_id(event),
        bus.events.analysis_transport_ready,
        event,
    ) is False
    assert sends == [event.payload]
    assert second_calls == [event]
    assert bus.pending_workflow_continuation_delivery_count == 1

    controller.disconnect()

    assert controller.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1
    assert not model.is_analysis_transport_authorized(event)
    transport.disconnect()


def test_analysis_transport_owner_retries_exact_authorized_event_across_generation():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    workflow_model = SequenceWorkflowModel()
    sends = []

    class _Server:
        def send_to_current_client(self, payload):
            sends.append(payload)
            if len(sends) == 1:
                raise SystemExit("transient")
            return True

    transport = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=workflow_model.is_analysis_transport_authorized,
        authorization_consumer=workflow_model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: _Server(),
    )
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-1", "source-1", "record-1", 7, {"Label": "OK"}
    )
    assert workflow_model.authorize_analysis_transport(event)
    workflow_model.workflow_generation = 8
    delivery_id = (
        "analysis-transport",
        event.analysis_id,
        event.source_id,
        event.record_id,
        event.workflow_generation,
    )

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is False
    assert workflow_model.is_analysis_transport_authorized(event)
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is True
    assert not workflow_model.is_analysis_transport_authorized(event)
    assert len(sends) == 2
    assert transport.handle_analysis_transport_ready(event) is False
    assert len(sends) == 2


def test_analysis_transport_owner_disabled_acks_but_missing_server_retries():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    enabled = {"value": False}
    server = {"value": None}
    transport = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: enabled["value"],
        tcp_server_provider=lambda: server["value"],
    )
    disabled = sequence_message_types.AnalysisTransportReady(
        "analysis-disabled", "source", "record", 0, {"Label": "OK"}
    )
    assert model.authorize_analysis_transport(disabled)
    assert transport.handle_analysis_transport_ready(disabled) is True
    assert not model.is_analysis_transport_authorized(disabled)

    enabled["value"] = True
    missing = sequence_message_types.AnalysisTransportReady(
        "analysis-missing", "source", "record", 1, {"Label": "NG"}
    )
    assert model.authorize_analysis_transport(missing)
    assert transport.handle_analysis_transport_ready(missing) is False
    assert model.is_analysis_transport_authorized(missing)


def test_analysis_transport_owner_keeps_hostile_payload_retryable_and_disconnects():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_lifecycle_owner(bus)
    model = SequenceWorkflowModel()
    sends = []
    owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: SimpleNamespace(
            send_to_current_client=lambda payload: sends.append(payload) or True
        ),
    )
    event = sequence_message_types.AnalysisTransportReady(
        "analysis-hostile", "source", "record", 3, {"bad": complex(1, 2)}
    )
    assert model.authorize_analysis_transport(event)
    delivery_id = (
        "analysis-transport", "analysis-hostile", "source", "record", 3
    )

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is False
    assert model.is_analysis_transport_authorized(event)
    assert sends == []
    assert owner.disconnect() is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", event, owner=bus
    ) is False
    assert model.is_analysis_transport_authorized(event)


def test_sequence_window_source_has_no_analysis_transport_algorithm():
    source = Path(SequenceWindow.__module__.replace(".", "/") + ".py")
    text = source.read_text(encoding="utf-8")
    analysis_source = Path(
        SequenceAnalysisController.__module__.replace(".", "/") + ".py"
    ).read_text(encoding="utf-8")

    assert "_handle_analysis_transport_ready" not in text
    assert "_send_tcp_analysis_result_payload" not in text
    assert "_send_tcp_analysis_result_callback" not in text
    assert "analysis_transport_controller.send_payload" not in text
    assert "legacy_transport_sender" not in analysis_source
    assert "authorization_claimer=(" in text
    assert "claim_releaser=self.workflow_model.release_analysis_transport_claim" in text
    assert "claim_committer=self.workflow_model.commit_analysis_transport_claim" in text
    assert "claim_abandoner=self.workflow_model.abandon_analysis_transport_claim" in text
    assert "authorization_provider=(" not in text


@pytest.mark.parametrize(
    ("enabled", "server_outcome", "expected_sends"),
    (
        (False, True, 0),
        (True, None, 0),
        (True, True, 1),
        (True, False, 1),
        (True, RuntimeError("ordinary"), 1),
        (True, KeyboardInterrupt("interrupt"), 1),
        (True, SystemExit("exit"), 1),
    ),
    ids=("disabled", "missing", "normal", "false", "ordinary", "ki", "system-exit"),
)
def test_legacy_analysis_uses_transport_service_without_consuming_formal_authorization(
    enabled, server_outcome, expected_sends
):
    from ui.sequence.sequence_analysis_transport_service import (
        SequenceAnalysisTransportService,
    )

    sends = []
    logs = []

    def send(payload):
        sends.append(payload)
        if isinstance(server_outcome, BaseException):
            raise server_outcome
        return server_outcome

    server = None if server_outcome is None else SimpleNamespace(
        send_to_current_client=send
    )
    service = SequenceAnalysisTransportService(
        tcp_enabled_provider=lambda: enabled,
        tcp_server_provider=lambda: server,
        logger=SimpleNamespace(
            info=lambda message: logs.append(message),
            warning=lambda message: logs.append(message),
            error=lambda message: logs.append(message),
        ),
    )
    workflow_model = SequenceWorkflowModel()
    formal = sequence_message_types.AnalysisTransportReady(
        "formal-analysis", "source", "record", 41, {"Label": "OK"}
    )
    assert workflow_model.authorize_analysis_transport(formal)
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(analysis_result_dict={"SPL": (True, "ok")}),
        recorded_path="record.wav",
        recorded_signal_info={"file_path": "record.wav"},
        screen=lambda: SimpleNamespace(
            size=lambda: SimpleNamespace(width=lambda: 1200, height=lambda: 800)
        ),
    )
    view = SimpleNamespace(
        reset_output=lambda: None,
        show_summary=lambda *_args: None,
    )
    controller = SequenceAnalysisController(
        SequenceAnalysisModel(),
        view,
        bus=SequenceEventBus(),
        runtime=runtime,
        transport_service=service,
    )
    context = AnalysisExecutionContext(
        analysis_config={},
        sequence_config={},
        mode="RECORD_ONLY",
        active_channels=(),
        recording_snapshot={},
        test_mode=False,
    )

    assert controller.run(readiness_checked=True, context=context) is True
    assert len(sends) == expected_sends
    assert workflow_model.is_analysis_transport_authorized(formal)
    assert max(map(len, logs), default=0) <= 600


def test_transport_service_contains_hostile_payload_and_logger_failures():
    from ui.sequence.sequence_analysis_transport_service import (
        SequenceAnalysisTransportService,
    )

    class HostileLogger:
        def __getattribute__(self, _name):
            raise SystemExit("hostile logger")

    service = SequenceAnalysisTransportService(
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: SimpleNamespace(
            send_to_current_client=lambda _payload: pytest.fail(
                "non-serializable payload must not reach TCP"
            )
        ),
        logger=HostileLogger(),
    )

    assert service.send_payload({"bad": complex(1, 2)}) is False


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class _RaisingSignal(_Signal):
    def emit(self, value):
        super().emit(value)
        raise RuntimeError("downstream slot failed")


class _Bus:
    def __init__(self):
        self.events = SimpleNamespace(
            analysis_completed=_Signal(),
            analysis_failed=_Signal(),
        )


class _View:
    def close_windows(self):
        return None


def test_sequence_window_registers_formal_continuation_recipients_without_raw_qt():
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    bus = SequenceEventBus(window)
    bus.register_workflow_continuation_lifecycle_owner(window)
    window.sequence_event_bus = bus
    window.workflow_model = SequenceWorkflowModel()
    window.analysis_controller = SimpleNamespace(
        handle_analysis_requested=lambda _message: True,
        handle_cancel_analysis_requested=lambda _message: True,
    )
    window.export_controller = object()
    delivered = []
    window._project_workflow_state = lambda message: delivered.append(
        ("state", message)
    )
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=window.workflow_model.is_analysis_transport_authorized,
        authorization_consumer=window.workflow_model.consume_analysis_transport,
        tcp_enabled_provider=lambda: False,
        tcp_server_provider=lambda: None,
    )
    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        label_service=SimpleNamespace(
            commit=lambda message, _project: (
                delivered.append(("label", message)) or {}
            )
        ),
        connect_queued=True,
    )
    raw = []
    bus.events.analysis_transport_ready.connect(
        lambda message: raw.append(("transport", message))
    )
    bus.commands.commit_recording_label_requested.connect(
        lambda message: raw.append(("label", message))
    )

    SequenceWindow._wire_workflow_continuation_ports(window)
    SequenceWindow._wire_analysis_workflow_channels(window)
    transport = sequence_message_types.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    assert window.workflow_model.authorize_analysis_transport(transport)
    label = CommitRecordingLabelRequested(
        "command", "record", "OK", ()
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 0),
        "analysis-transport",
        transport,
        owner=window,
    )
    assert bus.deliver_workflow_continuation(
        ("label-commit", "command", 0),
        "label-commit",
        label,
        owner=window,
    )

    assert delivered == [("label", label)]
    assert raw == []
    assert recording is not None


def test_analysis_transport_authorization_is_acked_only_after_side_effect_succeeds():
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    bus = SequenceEventBus(window)
    bus.register_workflow_continuation_lifecycle_owner(window)
    model = SequenceWorkflowModel()
    window.sequence_event_bus = bus
    window.workflow_model = model
    calls = []

    def send(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise SystemExit("transient transport failure")
        return True

    SequenceWindow._wire_workflow_continuation_ports(window)
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: None,
    )
    transport_owner.send_payload = send
    transport = sequence_message_types.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    assert model.authorize_analysis_transport(transport)
    delivery_id = ("analysis-transport", "analysis", "source", "record", 0)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", transport, owner=window
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", transport, owner=window
    ) is True
    assert calls == [transport.payload, transport.payload]


def test_label_business_failure_terminal_acks_continuation_and_allows_next_start():
    from PyQt5.QtTest import QTest

    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    _admit_workflow_analysis(model)
    configuration = ConfigurationSnapshot([], {})
    workflow = SequenceWorkflowController(
        model,
        bus,
        label_id_factory=lambda: "label-failure",
        configuration_snapshot_provider=lambda: configuration,
        connect_bus=True,
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = model
    bus.register_workflow_continuation_lifecycle_owner(window)
    window._project_workflow_state = lambda _message: True
    side_effects = []
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: False,
        tcp_server_provider=lambda: None,
    )
    SequenceWindow._wire_workflow_continuation_ports(window)
    def fail_label(command, _project):
        side_effects.extend(
            [
                ("result", command.label),
                ("text", None),
                ("persist", command.label),
            ]
        )
        raise RuntimeError("label persistence failed")

    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        label_service=SimpleNamespace(commit=fail_label),
        connect_queued=True,
    )

    assert workflow.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "record-1",
            _analysis_continuation_snapshot(targets=()),
        )
    )
    for _ in range(8):
        _QAPP.processEvents()
    QTest.qWait(30)
    _QAPP.processEvents()

    assert model.phase is WorkflowPhase.IDLE
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert side_effects == [
        ("result", "OK"),
        ("text", None),
        ("persist", "OK"),
    ]
    export_owner.disconnect()
    assert recording is not None
    assert workflow.handle_start(
        StartTestRequested("next-start", "manual", "SN", False, 0)
    )
    assert side_effects == [
        ("result", "OK"),
        ("text", None),
        ("persist", "OK"),
    ]


@pytest.mark.parametrize(
    "first_outcome",
    (
        False,
        RuntimeError("tcp ordinary " + "x" * 2000),
        KeyboardInterrupt("tcp interrupted"),
        SystemExit("tcp exited"),
    ),
    ids=("false", "ordinary", "keyboard-interrupt", "system-exit"),
)
def test_tcp_transport_failure_keeps_authorization_until_exact_retry_succeeds(
    first_outcome, monkeypatch
):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    logs = []
    logger = SimpleNamespace(
        debug=lambda message: logs.append(message),
        info=lambda message: logs.append(message),
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )
    outcomes = iter((first_outcome, True))
    sends = []

    def send(message):
        sends.append(message)
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    server = SimpleNamespace(send_to_current_client=send)
    bus.register_workflow_continuation_lifecycle_owner(bus)
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: server,
        logger=logger,
    )
    transport = sequence_message_types.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    assert model.authorize_analysis_transport(transport)
    delivery_id = ("analysis-transport", "analysis", "source", "record", 0)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", transport, owner=bus
    ) is False
    assert model.is_analysis_transport_authorized(transport)
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", transport, owner=bus
    ) is True

    assert not model.is_analysis_transport_authorized(transport)
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert len(sends) == 2
    assert logs
    assert max(map(len, logs)) <= 600


def test_tcp_sender_defines_disabled_and_missing_server_outcomes(monkeypatch):
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    logs = []
    window.default_logger = SimpleNamespace(
        info=lambda message: logs.append(message),
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )
    payload = {"Label": "OK"}

    enabled = {"value": False}
    transport_owner = SequenceAnalysisTransportController(
        bus=SequenceEventBus(),
        authorization_provider=lambda _event: False,
        authorization_consumer=lambda _event: False,
        tcp_enabled_provider=lambda: enabled["value"],
        tcp_server_provider=lambda: SequenceWindow.tcp_server,
        logger=window.default_logger,
    )
    assert transport_owner.send_payload(payload) is True

    enabled["value"] = True
    monkeypatch.setattr(SequenceWindow, "tcp_server", None)
    assert transport_owner.send_payload(payload) is False
    assert any("no tcp server" in message for message in logs)


def test_analysis_transport_missing_payload_is_an_intentional_no_op():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = model
    bus.register_workflow_continuation_lifecycle_owner(window)
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: pytest.fail(
            "missing payload must not invoke the TCP sender"
        ),
    )
    transport = sequence_message_types.AnalysisTransportReady(
        "analysis", "source", "record", 0, None
    )
    assert model.authorize_analysis_transport(transport)

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", "source", "record", 0),
        "analysis-transport",
        transport,
        owner=window,
    ) is True
    assert not model.is_analysis_transport_authorized(transport)


def test_formal_lifecycle_disconnect_abandons_outbox_before_next_recipient():
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        SequenceWorkflowModel(), bus, connect_bus=False
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "blocked", lambda _message: False
    )
    message = CommitRecordingLabelRequested(
        "command", "record", "OK", ()
    )
    assert workflow._publish_continuation(
        ("label-commit", "command", 0),
        bus.commands.commit_recording_label_requested,
        message,
    ) is False
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    lifecycle = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(window),
        SequenceResourceLifecycleModel(),
        lifecycle_bus=bus,
        parent=window,
    )
    observations = []
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        workflow.disconnect,
        owner=workflow,
    )
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "observer",
        lambda _request: observations.append(
            (
                workflow.pending_continuation_publication_ids,
                bus.pending_workflow_continuation_delivery_count,
            )
        )
        or True,
    )
    lifecycle._shutdown_prepared_generation = 1
    assert lifecycle.finalize_application_shutdown(1)

    assert lifecycle.complete_application_shutdown_delivery(1)

    assert observations == [((), 0)]
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def _command(**overrides):
    values = {
        "analysis_id": "analysis-1",
        "source_id": "record-1",
        "recording_snapshot": {"record_id": "record-1", "samples": [1, 2]},
        "configuration_snapshot": ConfigurationSnapshot(
            [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
        ),
        "automatic": False,
        "workflow_generation": 7,
    }
    values.update(overrides)
    return AnalysisRequested(**values)


def test_mutable_analysis_value_detaches_frozen_runtime_containers():
    source_array = np.array([1.0, 2.0])
    command = AnalysisRequested(
        "analysis-adapter",
        "record-adapter",
        {},
        {
            "sequence_config": [{"seq1": {"acq": {"mode": "PLAY_AND_RECORD"}}}],
            "analysis_config": {
                "display_sequence": ["HD 1"],
                "HD 1": {
                    "type": "HD",
                    "selected_labels": [2, 3],
                    "nested": [{"enabled": True}],
                },
                "hashable_members": {("left", "right")},
                "array": source_array,
            },
        },
        False,
    )
    frozen = command.configuration_snapshot["analysis_config"]

    mutable = mutable_analysis_value(frozen)

    assert isinstance(mutable, dict)
    assert mutable["display_sequence"] == ["HD 1"]
    assert mutable["HD 1"]["selected_labels"] == [2, 3]
    assert mutable["HD 1"]["nested"] == [{"enabled": True}]
    assert mutable["hashable_members"] == {("left", "right")}
    assert type(mutable["hashable_members"]) is set
    assert mutable["array"].flags.writeable
    assert not np.shares_memory(mutable["array"], frozen["array"])

    mutable["HD 1"]["selected_labels"].append(4)
    mutable["HD 1"]["nested"][0]["enabled"] = False
    mutable["hashable_members"].add(("new", "member"))
    mutable["array"][0] = 99.0
    assert frozen["HD 1"]["selected_labels"] == (2, 3)
    assert frozen["HD 1"]["nested"][0]["enabled"] is True
    assert frozen["hashable_members"] == frozenset({("left", "right")})
    assert float(frozen["array"][0]) == 1.0


def test_mutable_analysis_value_preserves_hashable_shapes_and_rejects_mappings():
    command = AnalysisRequested(
        "analysis-hashable-adapter",
        "record-hashable-adapter",
        {},
        {
            "analysis_config": {
                ("left", "right"): "value",
                "mapping_member": {"nested": 1},
            }
        },
        False,
    )
    frozen = command.configuration_snapshot["analysis_config"]

    mutable = mutable_analysis_value(frozen)

    assert mutable[("left", "right")] == "value"
    assert type(next(key for key in mutable if type(key) is tuple)) is tuple
    with pytest.raises(
        TypeError, match="analysis mappings cannot be thawed as hashable values"
    ):
        mutable_analysis_value(frozen["mapping_member"], _hashable=True)


def test_admitted_analysis_thaws_hd_rb_config_before_legacy_execution(
    monkeypatch,
):
    executed = []
    shown = []
    received_types = {}
    model = SequenceAnalysisModel()

    class LaterAnalysis:
        result = None

        def __init__(self, title_name):
            self.title_name = title_name

        def calculate_thd(self):
            executed.append(self.title_name)

    def probe_three_phase(_self, _signal, _sample_rate, thd_kwargs):
        orders = thd_kwargs["harmonic_orders"]
        received_types[tuple(orders)] = type(orders)
        HarmonicIndexBuilder().create_mask_from_indices(
            np.zeros((1, 36), dtype=np.int32), orders, 2
        )
        executed.append(tuple(orders))
        return np.array([100.0]), np.zeros((6, 1)), np.array([1.0])

    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "calculate_thd_three_phase",
        probe_three_phase,
    )
    monkeypatch.setattr(Distortion, "plot_graph", lambda *_args, **_kwargs: {})

    data_struct = SimpleNamespace(
        analysis_result_dict={},
        store_wave_data=np.zeros(128, dtype=np.float32),
        store_wave_data_multi=None,
        sample_rate=48000,
        stimulus_info={
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 100.0,
            "stop_freq": 200.0,
            "num_steps": 1,
            "total_time": 0.1,
            "repeat_times": 1,
        },
    )
    monkeypatch.setattr(DataDealStruct, "_instance", data_struct)
    runtime = SimpleNamespace(
        data_struct=data_struct,
        screen=lambda: SimpleNamespace(
            size=lambda: SimpleNamespace(width=lambda: 1200, height=lambda: 800)
        ),
    )
    view = SimpleNamespace(
        reset_output=lambda: None,
        present_calibration_warnings=lambda *_args, **_kwargs: None,
        show_channel_mismatch=lambda *_args, **_kwargs: None,
        show_instance=lambda _instance, *, key, **_kwargs: shown.append(key),
        show_summary=lambda *_args: None,
        warning_presenter=lambda *_args: None,
    )
    controller = SequenceAnalysisController(
        model,
        view,
        bus=SimpleNamespace(),
        runtime=runtime,
        class_mapping_provider=lambda: {
            "HD": Distortion,
            "RB": RubAndBuzz,
            "PRB": LaterAnalysis,
        },
    )
    assert controller.configure_calibration_types(
        ["HD", "RB"], generation=1
    ) is True
    command = AnalysisRequested(
        "analysis-hd-rb",
        "record-hd-rb",
        {},
        ConfigurationSnapshot(
            [{"seq1": {"acq": {"mode": "PLAY_AND_RECORD"}}}],
            {
                "display_sequence": ["HD 1", "RB 1", "PRB 1"],
                "HD 1": {
                    "type": "HD",
                    "selected_labels": [2, 3],
                    "display": {"traces": ["raw", "limit"]},
                    "manual_limit": {"frequencies": [100.0, 200.0]},
                },
                "RB 1": {
                    "type": "RB",
                    "selected_labels": [10, 11],
                },
                "PRB 1": {"type": "PRB"},
            },
        ),
        False,
    )

    assert (
        command.configuration_snapshot.analysis_config["HD 1"][
            "selected_labels"
        ]
        == (2, 3)
    )
    context = controller._context_from_command(command)
    assert controller.run(
        prepare_downstream=False,
        readiness_checked=True,
        context=context,
    ) is True
    assert received_types == {(2, 3): list, (10, 11): list}
    assert executed == [(2, 3), (10, 11), "PRB 1"]
    assert shown == ["HD 1", "RB 1", "PRB 1"]

    context.analysis_config["HD 1"]["selected_labels"].append(4)
    context.analysis_config["HD 1"]["display"]["traces"].append("extra")
    context.analysis_config["HD 1"]["manual_limit"]["frequencies"][0] = 50.0
    frozen_hd = command.configuration_snapshot.analysis_config["HD 1"]
    assert frozen_hd["selected_labels"] == (2, 3)
    assert frozen_hd["display"]["traces"] == ("raw", "limit")
    assert frozen_hd["manual_limit"]["frequencies"] == (100.0, 200.0)


def test_analysis_model_detaches_terminal_result_snapshot():
    model = SequenceAnalysisModel()
    model.begin(_command())
    mutable = {"record_id": "record-1", "analysis_result_dict": {"SPL": [True, "OK"]}}

    frozen = model.complete(mutable)
    mutable["analysis_result_dict"]["SPL"][1] = "changed"

    assert model.state is AnalysisState.COMPLETED
    assert frozen["analysis_result_dict"]["SPL"] == (True, "OK")
    with pytest.raises(TypeError):
        frozen["record_id"] = "other"


def test_admitted_analysis_result_has_single_configuration_authority(monkeypatch):
    admitted_config = {
        "display_sequence": ["Result"],
        "Result": {"type": "Excel", "enabled": True},
    }
    command = _command(
        configuration_snapshot=ConfigurationSnapshot(
            [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], admitted_config
        )
    )
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            analysis_result_dict={"FFT": (True, "within limits")}
        )
    )
    controller = SequenceAnalysisController(
        SequenceAnalysisModel(),
        _View(),
        bus=SimpleNamespace(),
        runtime=runtime,
        timestamp_provider=lambda: datetime(2026, 8, 24, 12, 0, 0),
    )
    monkeypatch.setattr(controller, "run", lambda **_kwargs: True)
    controller._active_context = controller._context_from_command(command)

    result_snapshot = controller._execute_admitted(command)

    assert "analysis_configuration" not in result_snapshot
    assert result_snapshot["export_handoff"]["analysis_config"] == admitted_config


def test_controller_emits_matching_completed_event_only_after_readiness():
    model = SequenceAnalysisModel()
    bus = _Bus()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: (True, ""),
        execute=lambda _command: {
            "record_id": "record-1",
            "analysis_result_dict": {"SPL": (True, "OK")},
        },
    )

    assert controller.handle_analysis_requested(_command()) is True

    assert model.state is AnalysisState.COMPLETED
    assert bus.events.analysis_failed.values == []
    assert len(bus.events.analysis_completed.values) == 1
    event = bus.events.analysis_completed.values[0]
    assert isinstance(event, AnalysisCompleted)
    assert (event.analysis_id, event.source_id) == ("analysis-1", "record-1")


def test_controller_normalizes_readiness_failure_to_matching_failed_event():
    model = SequenceAnalysisModel()
    bus = _Bus()
    executed = []
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: (False, "import reference is unavailable"),
        execute=lambda command: executed.append(command),
    )

    assert controller.handle_analysis_requested(_command()) is False

    assert executed == []
    assert bus.events.analysis_completed.values == []
    failure = bus.events.analysis_failed.values[0]
    assert isinstance(failure, AnalysisFailed)
    assert (failure.analysis_id, failure.source_id) == ("analysis-1", "record-1")
    assert failure.reason == "import reference is unavailable"


def test_controller_normalizes_analysis_exception_to_one_failed_terminal():
    model = SequenceAnalysisModel()
    bus = _Bus()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: (_ for _ in ()).throw(ValueError("bad input")),
    )

    assert controller.handle_analysis_requested(_command()) is False

    assert model.state is AnalysisState.FAILED
    assert bus.events.analysis_completed.values == []
    assert [event.reason for event in bus.events.analysis_failed.values] == [
        "bad input"
    ]


def test_unpublishable_result_fails_without_leaving_completed_model_state():
    model = SequenceAnalysisModel()
    bus = _Bus()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: {"mutable_object": SimpleNamespace(value=1)},
    )

    assert controller.handle_analysis_requested(_command()) is False

    assert model.state is AnalysisState.FAILED
    assert bus.events.analysis_completed.values == []
    assert len(bus.events.analysis_failed.values) == 1


def test_completed_event_delivery_error_does_not_publish_second_terminal():
    model = SequenceAnalysisModel()
    bus = _Bus()
    bus.events.analysis_completed = _RaisingSignal()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: {"record_id": "record-1"},
    )

    with pytest.raises(RuntimeError, match="downstream slot failed"):
        controller.handle_analysis_requested(_command())

    assert model.state is AnalysisState.COMPLETED
    assert len(bus.events.analysis_completed.values) == 1
    assert bus.events.analysis_failed.values == []


def test_instance_creation_maps_active_channel_and_injects_runtime_parameters():
    created = []
    calibration_calls = []

    class Analysis:
        _supports_pre_resolved_v2pa_factor = True

        def __init__(self, name):
            self.name = name
            created.append(self)

    model = SequenceAnalysisModel()
    runtime = SimpleNamespace(
        mode="RECORD_ONLY",
        data_struct=object(),
        _active_input_channels=[2, 5],
        analysis_config={"golden_sample_result_path": "golden.json"},
        analysis_types_requiring_v2pa={"FFT"},
    )
    view = SimpleNamespace(warning_presenter=lambda *_args: None)
    controller = SequenceAnalysisController(
        model,
        view,
        bus=SimpleNamespace(),
        runtime=runtime,
        class_mapping_provider=lambda: {"FFT": Analysis},
        calibration_resolver=lambda channel, warn_callback=None: (
            calibration_calls.append(channel) or 2.5
        ),
    )

    instance = controller.instance_analysis_class(
        "fft-item", "FFT", {"analysis_channel": 5, "average": 3}
    )

    assert instance is created[0]
    assert instance.name == "fft-item--通道6"
    assert instance.analysis_config == {
        "analysis_channel": 1,
        "average": 3,
        "golden_sample_result_path": "golden.json",
    }
    assert instance.v2pa_factor == 2.5
    assert instance._v2pa_raw_analysis_channel == 5
    assert instance._use_pre_resolved_v2pa_factor is True
    assert calibration_calls == [5]


def test_analysis_model_owns_immutable_bounded_calibration_policy_snapshot():
    service = SequenceAnalysisCalibrationPolicyService(max_types=3, max_type_length=8)
    source = ["FFT", "PD"]
    snapshot = service.snapshot(source, generation=4)
    source.append("ED")

    assert snapshot == AnalysisCalibrationPolicySnapshot(
        generation=4,
        analysis_types=frozenset({"FFT", "PD"}),
    )
    assert isinstance(snapshot.analysis_types, frozenset)

    model = SequenceAnalysisModel()
    assert model.apply_calibration_policy(snapshot) is True
    assert model.calibration_policy_snapshot is snapshot
    assert model.apply_calibration_policy(
        service.snapshot(["ED"], generation=4)
    ) is False
    assert model.calibration_policy_snapshot is snapshot


@pytest.mark.parametrize(
    "analysis_types",
    (
        "FFT",
        {"FFT": True},
        [""],
        ["TOO-LONG-TYPE"],
        ["FFT", object()],
        ["FFT", "PD", "ED", "SPL"],
    ),
)
def test_calibration_policy_rejects_malformed_or_oversized_types(analysis_types):
    service = SequenceAnalysisCalibrationPolicyService(max_types=3, max_type_length=8)

    with pytest.raises((TypeError, ValueError)):
        service.snapshot(analysis_types, generation=1)


def test_calibration_policy_rejects_hostile_container_without_executing_iteration():
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("hostile iterator executed")

    service = SequenceAnalysisCalibrationPolicyService()

    with pytest.raises(TypeError, match="plain sequence or set"):
        service.snapshot(HostileList(["FFT"]), generation=1)


@pytest.mark.parametrize(
    "analysis_types",
    (
        [object(), *(f"T{index}" for index in range(128))],
        tuple([object(), *(f"T{index}" for index in range(128))]),
        {object(), *(f"T{index}" for index in range(128))},
        frozenset({object(), *(f"T{index}" for index in range(128))}),
    ),
    ids=("list", "tuple", "set", "frozenset"),
)
def test_calibration_policy_rejects_exact_oversized_container_before_items(
    analysis_types,
):
    service = SequenceAnalysisCalibrationPolicyService()

    with pytest.raises(ValueError, match="too many"):
        service.snapshot(analysis_types, generation=1)


def test_calibration_policy_rejects_huge_list_without_copying_payload():
    service = SequenceAnalysisCalibrationPolicyService()
    analysis_types = ["FFT"] * 500_000

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="too many"):
            service.snapshot(analysis_types, generation=1)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 256_000


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_types": 0},
        {"max_types": -1},
        {"max_types": 129},
        {"max_types": True},
        {"max_type_length": 0},
        {"max_type_length": -1},
        {"max_type_length": 65},
        {"max_type_length": True},
    ),
)
def test_calibration_policy_service_rejects_noncanonical_bounds(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SequenceAnalysisCalibrationPolicyService(**kwargs)


def test_calibration_policy_service_accepts_canonical_maxima_end_to_end():
    service = SequenceAnalysisCalibrationPolicyService(
        max_types=MAX_ANALYSIS_CALIBRATION_TYPES,
        max_type_length=MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH,
    )
    analysis_types = [
        f"T{index}".ljust(MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH, "X")
        for index in range(MAX_ANALYSIS_CALIBRATION_TYPES)
    ]

    snapshot = service.snapshot(analysis_types, generation=1)

    assert len(snapshot.analysis_types) == MAX_ANALYSIS_CALIBRATION_TYPES
    assert max(map(len, snapshot.analysis_types)) == MAX_ANALYSIS_CALIBRATION_TYPE_LENGTH


def test_calibration_policy_service_does_not_coerce_hostile_numeric_bounds():
    class HostileNumber:
        def __int__(self):
            raise AssertionError("numeric bound was coerced")

        def __index__(self):
            raise AssertionError("numeric bound index was read")

    with pytest.raises((TypeError, ValueError)):
        SequenceAnalysisCalibrationPolicyService(max_types=HostileNumber())


@pytest.mark.parametrize(
    "generation, analysis_types",
    (
        (True, frozenset({"FFT"})),
        (-1, frozenset({"FFT"})),
        (1, {"FFT"}),
        (1, frozenset({object()})),
        (1, frozenset({""})),
        (1, frozenset({"X" * 65})),
        (1, frozenset(f"T{index}" for index in range(129))),
    ),
)
def test_calibration_policy_snapshot_rejects_direct_malformed_construction(
    generation, analysis_types
):
    with pytest.raises((TypeError, ValueError)):
        AnalysisCalibrationPolicySnapshot(generation, analysis_types)


def test_controller_calibration_policy_update_is_generation_guarded_and_context_frozen():
    created = []
    calibration_calls = []

    class Analysis:
        _supports_pre_resolved_v2pa_factor = True

        def __init__(self, name):
            self.name = name
            created.append(self)

    class Runtime:
        mode = "PLAY_AND_RECORD"
        data_struct = object()
        analysis_config = {}
        sequence_config = []

        @property
        def analysis_types_requiring_v2pa(self):
            raise AssertionError("facade calibration policy was read")

    model = SequenceAnalysisModel()
    controller = SequenceAnalysisController(
        model,
        SimpleNamespace(warning_presenter=lambda *_args: None),
        bus=SimpleNamespace(),
        runtime=Runtime(),
        class_mapping_provider=lambda: {"CUSTOM": Analysis, "NEXT": Analysis},
        calibration_resolver=lambda channel, warn_callback=None: (
            calibration_calls.append(channel) or 2.0
        ),
    )
    assert controller.configure_calibration_types(
        ["CUSTOM"], generation=1
    ) is True
    frozen_context = controller._legacy_context()
    assert controller.configure_calibration_types(["NEXT"], generation=2) is True
    assert controller.configure_calibration_types(
        ["CUSTOM"], generation=1
    ) is False

    controller._active_context = frozen_context
    try:
        first = controller.instance_analysis_class(
            "custom", "CUSTOM", {"analysis_channel": 0}
        )
        second = controller.instance_analysis_class(
            "next", "NEXT", {"analysis_channel": 1}
        )
    finally:
        controller._active_context = None

    assert first.v2pa_factor == 2.0
    assert not hasattr(second, "v2pa_factor")
    assert calibration_calls == [0]
    assert model.calibration_policy_snapshot.analysis_types == frozenset({"NEXT"})


def test_stale_calibration_policy_update_does_not_inspect_hostile_payload():
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("stale payload was inspected")

    model = SequenceAnalysisModel()
    controller = SequenceAnalysisController(
        model,
        SimpleNamespace(),
        bus=SimpleNamespace(),
    )
    assert controller.configure_calibration_types(["FFT"], generation=1) is True

    assert controller.configure_calibration_types(
        HostileList(["PD"]), generation=1
    ) is False
    assert model.calibration_policy_snapshot.analysis_types == frozenset({"FFT"})


def test_readiness_uses_workflow_frozen_configuration_snapshot():
    samples = [0.0, 0.0]
    runtime = SimpleNamespace(
        sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
        data_struct=SimpleNamespace(
            sample_rate=48000,
            audio_lenth=2,
            store_wave_data=samples,
            store_wave_data_multi=[[0.0], [0.0]],
            stimulus_info=None,
            stimulus_data=None,
        ),
    )
    controller = SequenceAnalysisController(
        SequenceAnalysisModel(),
        _View(),
        bus=SimpleNamespace(),
        runtime=runtime,
    )
    command = _command(
        configuration_snapshot=ConfigurationSnapshot(
            [{"seq1": {"acq": {"mode": "IMPORT_STIMULUS_AUDIO"}}}],
            {},
        )
    )

    assert controller._runtime_readiness(command) == (
        False,
        "import stimulus reference is unavailable",
    )


def test_view_normalizes_calibration_warning_and_channel_context():
    ordinary = []
    dedicated = []
    view = object.__new__(SequenceAnalysisView)
    view.warning_presenter = lambda title, text: ordinary.append((title, text))
    view.uncalibrated_warning_presenter = dedicated.append

    view.present_calibration_warnings(
        ["missing", "extra"],
        missing_message="missing",
        suppress_missing=False,
        record_only_channels=[2, 0],
        channel_formatter=lambda channel: f"In{channel + 1}",
    )

    assert dedicated == ["missing\n未校准通道：\n• In3\n• In1\nextra"]
    assert ordinary == []


def test_view_persists_and_restores_normalized_geometry(tmp_path):
    path = tmp_path / "analysis-geometry.json"
    first_model = SequenceAnalysisModel()
    first = SequenceAnalysisView(first_model, geometry_path=path)
    first.set_geometry("FFT1", {"x": 10, "y": 20, "w": 600, "h": 500})
    first.flush_geometry()

    second_model = SequenceAnalysisModel()
    SequenceAnalysisView(second_model, geometry_path=path)

    assert second_model.geometry == {
        "FFT1": {"x": 10, "y": 20, "w": 600, "h": 500}
    }


def _workflow(*, automatic=False, recording_lookup=lambda _record_id: None):
    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "analysis-helper-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "analysis-helper-transport", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "analysis-helper-label", lambda _message: True
    )
    captured = []
    bus.commands.analysis_requested.connect(captured.append)
    configuration = ConfigurationSnapshot([], {})
    controller = SequenceWorkflowController(
        model,
        bus,
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=recording_lookup,
        automatic_analysis_policy=_StaticAutomaticAnalysisPolicy(automatic),
        connect_bus=False,
    )
    controller._test_export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )
    return model, controller, captured, configuration


def test_manual_automatic_and_import_analysis_are_workflow_admitted():
    manual_model, manual, manual_events, _configuration = _workflow(
        recording_lookup=lambda record_id: {"record_id": record_id}
    )
    manual_model.retained_record_id = "record-1"
    assert manual.handle_manual_analysis(
        ManualAnalysisRequested("manual-command", "record-1")
    )

    automatic_model, automatic, automatic_events, configuration = _workflow(
        automatic=True
    )
    automatic_model.phase = WorkflowPhase.RECORDING
    automatic_model.active_session_id = "session-1"
    automatic_model.active_session_origin = SessionOrigin.CANONICAL
    automatic_model.configuration_snapshot = configuration
    assert automatic.handle_recording_completed(
        RecordingCompleted("session-1", 3, {"record_id": "record-2"})
    )

    import_model, imported, import_events, configuration = _workflow(
        recording_lookup=lambda record_id: {"record_id": record_id}
    )
    import_model.phase = WorkflowPhase.IMPORTING
    import_model.active_import_id = "import-1"
    import_model.configuration_snapshot = configuration
    assert imported.handle_imported_audio_ready(
        ImportedAudioReady("import-1", {"record_id": "record-3"})
    )
    imported_analysis = import_events[0]
    assert import_model.retained_record_id == "record-3"
    assert imported.handle_analysis_completed(
        AnalysisCompleted(
            imported_analysis.analysis_id,
            imported_analysis.source_id,
            {"record_id": "record-3"},
        )
    )
    assert imported.handle_manual_analysis(
        ManualAnalysisRequested("manual-import", "record-3")
    )

    assert [event.source_id for event in manual_events] == ["record-1"]
    assert manual_events[0].automatic is False
    assert [event.source_id for event in automatic_events] == ["session-1"]
    assert automatic_events[0].automatic is True
    assert [event.source_id for event in import_events] == [
        "import-1",
        "record-3",
    ]
    assert import_events[0].automatic is True
    assert manual_events[0].workflow_generation == manual_model.workflow_generation
    assert automatic_events[0].workflow_generation == automatic_model.workflow_generation
    assert import_events[0].workflow_generation == 0
    assert import_events[1].workflow_generation == import_model.workflow_generation


def test_controller_source_does_not_call_recording_or_export_controller():
    source = (
        Path(__file__).parents[2]
        / "ui"
        / "sequence"
        / "sequence_analysis_controller.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }

    assert "recording_controller" not in attributes
    assert "export_controller" not in attributes


def test_facade_analysis_methods_are_explicit_controller_delegates():
    source = (
        Path(__file__).parents[2] / "ui" / "sequence" / "sequence_widget.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in facade.body
        if isinstance(node, ast.FunctionDef)
    }

    for name in (
        "run",
        "instance_analysis_class",
        "update_v2pa_factor",
        "_summarize_ok_ng",
    ):
        calls = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "analysis_controller"
            for call in calls
        ), name


def test_facade_routes_import_and_data_actions_through_workflow_commands():
    source = (
        Path(__file__).parents[2] / "ui" / "sequence" / "sequence_widget.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in facade.body
        if isinstance(node, ast.FunctionDef)
    }
    player_calls = {
        node.func.attr
        for node in ast.walk(methods["on_clicked_player_btn"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    member_calls = {
        node.func.attr
        for node in ast.walk(methods["set_member_connect"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "emit" in player_calls
    assert "import_audio_and_analyze" not in player_calls
    assert "connect" in member_calls
    assert "run" not in {
        node.attr
        for node in ast.walk(methods["set_member_connect"])
        if isinstance(node, ast.Attribute)
    }


def test_recording_owner_receives_import_command_and_facade_is_only_a_delegate():
    source = (
        Path(__file__).parents[2] / "ui" / "sequence" / "sequence_widget.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in facade.body
        if isinstance(node, ast.FunctionDef)
    }
    import_delegate = methods["import_audio_and_analyze"]
    calls = [node for node in ast.walk(import_delegate) if isinstance(node, ast.Call)]

    assert len(import_delegate.body) == 1
    assert not any(isinstance(node, (ast.Try, ast.If, ast.For, ast.While)) for node in ast.walk(import_delegate))
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "recording_controller"
        and call.func.attr == "handle_load_imported_audio_requested"
        for call in calls
    )
    assert not {
        "QFileDialog",
        "load_audio_preserve_rate",
        "read_wav_calibration_metadata",
        "set_data_struct_analysis_reference_signal",
    } & {node.id for node in ast.walk(import_delegate) if isinstance(node, ast.Name)}

    constructor = methods["__init__"]
    connections = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Attribute)
        and isinstance(call.args[0].value, ast.Attribute)
        and call.args[0].value.attr == "recording_controller"
        and call.args[0].attr == "handle_load_imported_audio_requested"
        for call in connections
    )


def test_recording_import_handler_uses_admitted_snapshot_and_commits_one_ready_event():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    configuration = ConfigurationSnapshot(
        [
            {
                "seq1": {
                    "acq": {
                        "mode": "IMPORT_AUDIO",
                        "detail": {"sample_rate": 32_000},
                    }
                }
            }
        ],
        {"auto_analysis": False},
        using_config_path="frozen.json",
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-1", "import-1", "IMPORT_AUDIO", "selected.wav", configuration
    )
    multi = __import__("numpy").array(
        [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]], dtype="float32"
    )
    mono = multi.mean(axis=1).astype("float32")
    stage = ImportedAudioStage(
        file_path="selected.wav",
        mode="IMPORT_AUDIO",
        sample_rate=32_000,
        audio_multi=multi,
        audio_mono=mono,
        sample_count=3,
        calibration_metadata={"recorded_channels": []},
        reference=None,
    )
    service_calls = []
    service = SimpleNamespace(
        load=lambda admitted, path: service_calls.append((admitted, path)) or stage
    )
    projections = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *args: projections.append(("warning", args)),
        capture_import_projection=lambda: "old-projection",
        restore_import_projection=lambda checkpoint: projections.append(
            ("restore", checkpoint)
        ),
        clear_import_projection=lambda: projections.append(("clear",)),
        show_imported_audio=lambda audio, rate: projections.append(
            ("plot", audio, rate)
        ),
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(),
        recorded_path="live-old.wav",
        recorded_signal_info={"file_path": "live-old.wav"},
        configuration_model=SimpleNamespace(
            current_snapshot=lambda: (_ for _ in ()).throw(
                AssertionError("mutable live configuration must not be read")
            )
        ),
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=service,
        workflow_identity_provider=lambda: {
            "import_id": "import-1",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is True
    assert service_calls == [(command, "selected.wav")]
    assert runtime.recorded_path == "selected.wav"
    assert runtime.recorded_signal_info == {
        "file_path": "selected.wav",
        "barcode": None,
        "labels": "not_labeled",
    }
    assert runtime.data_struct.sample_rate == 32_000
    assert runtime.data_struct.audio_lenth == 3
    assert runtime.data_struct.store_wave_data_multi is multi
    assert runtime.data_struct.store_wave_data is mono
    assert runtime.data_struct.wav_calibration_metadata_authoritative is True
    assert [item[0] for item in projections] == ["plot", "enabled"]
    assert len(bus.events.imported_audio_ready.values) == 1
    assert bus.events.imported_audio_ready.values[0].import_id == "import-1"
    assert bus.events.imported_audio_failed.values == []


@pytest.mark.parametrize(
    "failure", [RuntimeError("decode failed"), KeyboardInterrupt(), SystemExit()]
)
def test_recording_import_boundary_contains_failures_and_retires_duplicate(
    failure,
):
    from ui.sequence.sequence_recording_import_service import AudioImportFailure

    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-fail", "import-fail", "IMPORT_AUDIO", "broken.wav", configuration
    )
    calls = []

    def fail_load(_command, _path):
        calls.append("load")
        if isinstance(failure, RuntimeError):
            raise AudioImportFailure("audio import failed", "提示", "导入失败")
        raise failure

    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *args: calls.append(("warning", args)),
        clear_import_projection=lambda: calls.append("clear"),
        set_import_data_enabled=lambda enabled: calls.append(("enabled", enabled)),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(sample_rate=48_000),
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=fail_load),
        workflow_identity_provider=lambda: {
            "import_id": "import-fail",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert controller.handle_load_imported_audio_requested(command) is False
    assert calls.count("load") == 1
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert controller.model.active_import_id is None
    assert bus.events.imported_audio_failed.values[0].import_id == "import-fail"


def test_recording_import_cancel_preserves_existing_runtime_and_selected_path_semantics():
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-cancel", "import-cancel", "IMPORT_AUDIO", None, configuration
    )
    old_multi = __import__("numpy").array([[1.0]], dtype="float32")
    data_struct = SimpleNamespace(
        store_wave_data_multi=old_multi,
        store_wave_data=old_multi[:, 0],
        sample_rate=48_000,
        audio_lenth=1,
    )
    runtime = SimpleNamespace(
        data_struct=data_struct,
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    selected_values = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected_values.append(selected),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda *_args: (_ for _ in ()).throw(
                AssertionError("cancelled selection must not load")
            )
        ),
        workflow_identity_provider=lambda: {
            "import_id": "import-cancel",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert selected_values == [None]
    assert runtime.recorded_path == "old.wav"
    assert runtime.recorded_signal_info == {"file_path": "old.wav"}
    assert data_struct.store_wave_data_multi is old_multi
    assert data_struct.sample_rate == 48_000
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert bus.events.imported_audio_failed.values[0].reason == "audio import was cancelled"


def test_recording_import_stale_after_staging_cannot_mutate_newer_runtime():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-stale", "import-old", "IMPORT_AUDIO", "old.wav", configuration
    )
    old_multi = np.array([[9.0]], dtype=np.float32)
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data_multi=old_multi,
            store_wave_data=old_multi[:, 0],
            sample_rate=48_000,
            audio_lenth=1,
        ),
        recorded_path="newer.wav",
        recorded_signal_info={"file_path": "newer.wav"},
    )
    identity = {"import_id": "import-old", "phase": "IMPORTING"}

    def stage_then_advance(_command, _path):
        identity["import_id"] = "import-new"
        return ImportedAudioStage(
            "old.wav",
            "IMPORT_AUDIO",
            32_000,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            3,
            {"recorded_channels": []},
        )

    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(choose_import_audio_path=lambda selected: selected),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=stage_then_advance),
        workflow_identity_provider=lambda: identity,
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert runtime.recorded_path == "newer.wav"
    assert runtime.recorded_signal_info == {"file_path": "newer.wav"}
    assert runtime.data_struct.store_wave_data_multi is old_multi
    assert runtime.data_struct.sample_rate == 48_000
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert bus.events.imported_audio_failed.values[0].import_id == "import-old"


def test_recording_import_cancel_during_view_projection_rolls_back_before_terminal():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-race", "import-race", "IMPORT_AUDIO", "new.wav", configuration
    )
    old_multi = np.array([[7.0]], dtype=np.float32)
    old_mono = np.array([7.0], dtype=np.float32)
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data_multi=old_multi,
            store_wave_data=old_mono,
            sample_rate=48_000,
            audio_lenth=1,
            stimulus_data=None,
            stimulus_info=None,
        ),
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    identity = {"import_id": "import-race", "phase": "IMPORTING"}
    projections = []

    def project(_audio, _rate):
        projections.append("new-plot")
        identity["phase"] = "CANCELLING"

    stage = ImportedAudioStage(
        "new.wav",
        "IMPORT_AUDIO",
        32_000,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        3,
        {"recorded_channels": []},
    )
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=lambda: "old-plot",
        restore_import_projection=lambda checkpoint: projections.append(
            ("restored", checkpoint)
        ),
        show_imported_audio=project,
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
        workflow_identity_provider=lambda: identity,
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert runtime.recorded_path == "old.wav"
    assert runtime.recorded_signal_info == {"file_path": "old.wav"}
    assert runtime.data_struct.store_wave_data_multi is old_multi
    assert runtime.data_struct.store_wave_data is old_mono
    assert runtime.data_struct.sample_rate == 48_000
    assert projections[-1] == ("restored", "old-plot")
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1


def test_recording_import_commit_failure_restores_checkpoint_and_next_import_runs():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    class FailCommitOnce(SimpleNamespace):
        armed = False
        failed = False

        def __setattr__(self, name, value):
            if name == "audio_lenth" and self.armed and not self.failed:
                self.failed = True
                raise RuntimeError("commit failed")
            super().__setattr__(name, value)

    data_struct = FailCommitOnce(
        store_wave_data_multi=np.array([[7.0]], dtype=np.float32),
        store_wave_data=np.array([7.0], dtype=np.float32),
        sample_rate=48_000,
        audio_lenth=1,
        stimulus_data=np.array([1.0], dtype=np.float32),
        stimulus_info={"sample_rate": 48_000},
        wav_calibration_metadata={"old": True},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=False,
    )
    data_struct.armed = True
    old_multi = data_struct.store_wave_data_multi
    old_mono = data_struct.store_wave_data
    old_stimulus = data_struct.stimulus_data
    runtime = SimpleNamespace(
        data_struct=data_struct,
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    identity = {"import_id": "import-commit", "phase": "IMPORTING"}
    first = sequence_message_types.LoadImportedAudioRequested(
        "command-commit", "import-commit", "IMPORT_AUDIO", "new.wav", configuration
    )
    second = sequence_message_types.LoadImportedAudioRequested(
        "command-next", "import-next", "IMPORT_AUDIO", "next.wav", configuration
    )
    stages = {
        "import-commit": ImportedAudioStage(
            "new.wav",
            "IMPORT_AUDIO",
            32_000,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            3,
            {"recorded_channels": []},
        ),
        "import-next": ImportedAudioStage(
            "next.wav",
            "IMPORT_AUDIO",
            44_100,
            np.ones((2, 1), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            2,
            {"recorded_channels": []},
        ),
    }
    projections = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *_args: None,
        capture_import_projection=lambda: ("old-plot", True),
        restore_import_projection=lambda checkpoint: projections.append(
            ("restored", checkpoint)
        ),
        clear_import_projection=lambda: projections.append(("cleared",)),
        show_imported_audio=lambda *_args: projections.append(("plot",)),
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda command, _path: stages[command.import_id]
        ),
        workflow_identity_provider=lambda: identity,
    )

    assert controller.handle_load_imported_audio_requested(first) is False
    assert runtime.recorded_path == "old.wav"
    assert runtime.recorded_signal_info == {"file_path": "old.wav"}
    assert data_struct.store_wave_data_multi is old_multi
    assert data_struct.store_wave_data is old_mono
    assert data_struct.stimulus_data is old_stimulus
    assert data_struct.sample_rate == 48_000
    assert data_struct.audio_lenth == 1
    assert projections == [("restored", ("old-plot", True))]
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1

    identity["import_id"] = "import-next"
    assert controller.handle_load_imported_audio_requested(second) is True
    assert runtime.recorded_path == "next.wav"
    assert len(bus.events.imported_audio_ready.values) == 1
    assert len(bus.events.imported_audio_failed.values) == 1


@pytest.mark.parametrize(
    "projection_failure",
    [RuntimeError("plot failed"), KeyboardInterrupt("plot interrupted"), SystemExit("plot exited")],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_recording_import_projection_failure_restores_checkpoint_without_clear(
    projection_failure,
):
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    old_multi = np.array([[5.0]], dtype=np.float32)
    old_mono = np.array([5.0], dtype=np.float32)
    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data_multi=old_multi,
            store_wave_data=old_mono,
            sample_rate=48_000,
            audio_lenth=1,
            stimulus_data=None,
            stimulus_info=None,
        ),
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-projection",
        "import-projection",
        "IMPORT_AUDIO",
        "new.wav",
        configuration,
    )
    stage = ImportedAudioStage(
        "new.wav",
        "IMPORT_AUDIO",
        32_000,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        3,
        {"recorded_channels": []},
    )
    projections = []

    def fail_projection(*_args):
        projections.append("new-plot")
        raise projection_failure

    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *_args: None,
        capture_import_projection=lambda: ("old-plot", False),
        restore_import_projection=lambda checkpoint: projections.append(
            ("restored", checkpoint)
        ),
        clear_import_projection=lambda: projections.append(("cleared",)),
        show_imported_audio=fail_projection,
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
        workflow_identity_provider=lambda: {
            "import_id": "import-projection",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert runtime.recorded_path == "old.wav"
    assert runtime.recorded_signal_info == {"file_path": "old.wav"}
    assert runtime.data_struct.store_wave_data_multi is old_multi
    assert runtime.data_struct.store_wave_data is old_mono
    assert runtime.data_struct.sample_rate == 48_000
    assert projections == ["new-plot", ("restored", ("old-plot", False))]
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1


def test_recording_import_attempts_projection_restore_when_runtime_restore_fails():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    class RuntimeWithRestoreFailure:
        def __init__(self):
            self.data_struct = SimpleNamespace(
                store_wave_data_multi=np.array([[4.0]], dtype=np.float32),
                store_wave_data=np.array([4.0], dtype=np.float32),
                sample_rate=48_000,
                audio_lenth=1,
                stimulus_data=None,
                stimulus_info=None,
            )
            self._recorded_path = "old.wav"
            self.recorded_signal_info = {"file_path": "old.wav"}
            self.fail_old_restore = False

        @property
        def recorded_path(self):
            return self._recorded_path

        @recorded_path.setter
        def recorded_path(self, value):
            if value == "old.wav" and self.fail_old_restore:
                raise RuntimeError("runtime restore failed")
            self._recorded_path = value

    runtime = RuntimeWithRestoreFailure()
    old_multi = runtime.data_struct.store_wave_data_multi
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-restore", "import-restore", "IMPORT_AUDIO", "new.wav", configuration
    )
    stage = ImportedAudioStage(
        "new.wav",
        "IMPORT_AUDIO",
        32_000,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        3,
        {"recorded_channels": []},
    )
    projections = []

    def fail_projection(*_args):
        runtime.fail_old_restore = True
        raise RuntimeError("projection failed")

    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *_args: None,
        capture_import_projection=lambda: "old-projection",
        restore_import_projection=lambda checkpoint: projections.append(
            ("restore-attempted", checkpoint)
        ),
        clear_import_projection=lambda: projections.append(("cleared",)),
        show_imported_audio=fail_projection,
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
        workflow_identity_provider=lambda: {
            "import_id": "import-restore",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert ("restore-attempted", "old-projection") in projections
    assert ("cleared",) in projections
    assert runtime.recorded_path is None
    assert runtime.data_struct.store_wave_data_multi is None
    assert runtime.data_struct.store_wave_data_multi is not old_multi
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert controller.model.active_import_id is None


@pytest.mark.parametrize(
    "warning_failure",
    [RuntimeError("warning failed"), KeyboardInterrupt("warning interrupted"), SystemExit("warning exited")],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_recording_import_warning_failure_is_contained_without_changing_terminal(
    warning_failure,
):
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-warning", "import-warning", "IMPORT_AUDIO", "new.wav", configuration
    )
    runtime = SimpleNamespace(data_struct=SimpleNamespace())
    stage = ImportedAudioStage(
        "new.wav",
        "IMPORT_AUDIO",
        32_000,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        3,
        None,
    )

    def fail_warning(*_args):
        raise warning_failure

    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            present_import_warning=fail_warning,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _enabled: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
        workflow_identity_provider=lambda: {
            "import_id": "import-warning",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is True
    assert runtime.recorded_path == "new.wav"
    assert runtime.data_struct.wav_calibration_warning_shown is True
    assert len(bus.events.imported_audio_ready.values) == 1
    assert bus.events.imported_audio_failed.values == []


def test_recording_import_projection_restore_failure_uses_cleared_safe_fallback():
    from ui.sequence.sequence_recording_import_service import ImportedAudioStage

    import numpy as np

    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data_multi=np.array([[8.0]], dtype=np.float32),
            store_wave_data=np.array([8.0], dtype=np.float32),
            sample_rate=48_000,
            audio_lenth=1,
        ),
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-restore-view",
        "import-restore-view",
        "IMPORT_AUDIO",
        "new.wav",
        configuration,
    )
    stage = ImportedAudioStage(
        "new.wav",
        "IMPORT_AUDIO",
        32_000,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        3,
        {"recorded_channels": []},
    )
    projections = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda *_args: None,
        capture_import_projection=lambda: "old-projection",
        restore_import_projection=lambda _checkpoint: (_ for _ in ()).throw(
            KeyboardInterrupt("projection restore interrupted")
        ),
        clear_import_projection=lambda: projections.append("cleared"),
        show_imported_audio=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("projection failed")
        ),
        set_import_data_enabled=lambda enabled: projections.append(
            ("enabled", enabled)
        ),
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
        workflow_identity_provider=lambda: {
            "import_id": "import-restore-view",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert runtime.recorded_path is None
    assert runtime.recorded_signal_info is None
    assert runtime.data_struct.store_wave_data_multi is None
    assert projections == ["cleared", ("enabled", False)]
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert controller.model.active_import_id is None


@pytest.mark.parametrize(
    "warning_failure",
    [RuntimeError("warning failed"), KeyboardInterrupt("warning interrupted"), SystemExit("warning exited")],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_recording_import_staging_failure_warning_is_contained_and_clears_runtime(
    warning_failure,
):
    from ui.sequence.sequence_recording_import_service import AudioImportFailure

    import numpy as np

    runtime = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data_multi=np.array([[8.0]], dtype=np.float32),
            store_wave_data=np.array([8.0], dtype=np.float32),
            sample_rate=48_000,
            audio_lenth=1,
        ),
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
    )
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "IMPORT_AUDIO", "detail": {}}}}], {}
    )
    command = sequence_message_types.LoadImportedAudioRequested(
        "command-stage-warning",
        "import-stage-warning",
        "IMPORT_AUDIO",
        "broken.wav",
        configuration,
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            present_import_warning=lambda *_args: (_ for _ in ()).throw(
                warning_failure
            ),
            clear_import_projection=lambda: None,
            set_import_data_enabled=lambda _enabled: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda *_args: (_ for _ in ()).throw(
                AudioImportFailure("audio import failed", "提示", "导入失败")
            )
        ),
        workflow_identity_provider=lambda: {
            "import_id": "import-stage-warning",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(command) is False
    assert runtime.recorded_path is None
    assert runtime.recorded_signal_info is None
    assert runtime.data_struct.store_wave_data_multi is None
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert controller.model.active_import_id is None


def _active_identity(command, *, phase="ANALYZING"):
    return {
        "analysis_id": command.analysis_id,
        "source_id": command.source_id,
        "workflow_generation": command.workflow_generation,
        "phase": phase,
        "cancelling_domain": "analysis" if phase == "CANCELLING" else None,
    }


def test_controller_ignores_late_duplicate_and_reentrant_analysis_commands():
    command = _command()
    model = SequenceAnalysisModel()
    bus = _Bus()
    calls = []
    controller = None

    def execute(admitted):
        calls.append(admitted.analysis_id)
        assert controller.handle_analysis_requested(admitted) is False
        return {"record_id": admitted.source_id}

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=execute,
        workflow_identity_provider=lambda: _active_identity(command),
    )

    assert controller.handle_analysis_requested(command) is True
    assert controller.handle_analysis_requested(command) is False
    assert controller.handle_analysis_requested(
        _command(analysis_id="late-analysis")
    ) is False

    assert calls == ["analysis-1"]
    assert len(bus.events.analysis_completed.values) == 1
    assert bus.events.analysis_failed.values == []


def test_controller_cancellation_is_identity_bound_and_first_terminal_wins():
    command = _command()
    model = SequenceAnalysisModel()
    bus = _Bus()
    identity = _active_identity(command)
    controller = None

    def execute(_admitted):
        identity["phase"] = "CANCELLING"
        identity["cancelling_domain"] = "analysis"
        assert controller.handle_cancel_analysis_requested(
            CancelAnalysisRequested("analysis-1", 7, "user stop")
        ) is True
        assert controller.handle_cancel_analysis_requested(
            CancelAnalysisRequested("analysis-1", 6, "stale")
        ) is False
        return {"record_id": "record-1"}

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=execute,
        workflow_identity_provider=lambda: dict(identity),
    )

    assert controller.handle_analysis_requested(command) is False
    assert model.state is AnalysisState.FAILED
    assert bus.events.analysis_completed.values == []
    assert len(bus.events.analysis_failed.values) == 1
    assert "cancel" in bus.events.analysis_failed.values[0].reason

    assert controller.handle_cancel_analysis_requested(
        CancelAnalysisRequested("analysis-1", 7, "duplicate")
    ) is False
    assert len(bus.events.analysis_failed.values) == 1


def test_cancellation_pending_before_analysis_delivery_skips_readiness_and_execute():
    command = _command()
    calls = []
    bus = _Bus()
    controller = SequenceAnalysisController(
        SequenceAnalysisModel(),
        _View(),
        bus=bus,
        readiness=lambda _command: calls.append("readiness"),
        execute=lambda _command: calls.append("execute"),
        workflow_identity_provider=lambda: _active_identity(
            command, phase="CANCELLING"
        ),
    )

    assert controller.handle_analysis_requested(command) is False
    assert calls == []
    assert len(bus.events.analysis_failed.values) == 1
    assert "cancel" in bus.events.analysis_failed.values[0].reason


def test_controller_begins_before_readiness_and_recovers_from_base_exception():
    first = _command()
    second = _command(analysis_id="analysis-2", source_id="record-2")
    identity = _active_identity(first)
    model = SequenceAnalysisModel()
    bus = _Bus()
    observed_states = []

    def readiness(_command):
        observed_states.append(model.state)
        raise KeyboardInterrupt("readiness interrupted")

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=readiness,
        execute=lambda command: {"record_id": command.source_id},
        workflow_identity_provider=lambda: dict(identity),
    )

    assert controller.handle_analysis_requested(first) is False
    assert observed_states == [AnalysisState.RUNNING]
    assert model.state is AnalysisState.FAILED
    assert controller.active_context is None
    assert len(bus.events.analysis_failed.values) == 1

    identity.update(_active_identity(second))
    controller.readiness = lambda _command: True
    assert controller.handle_analysis_requested(second) is True
    assert model.state is AnalysisState.COMPLETED
    assert controller.active_context is None


def test_calculation_base_exception_cleans_context_and_next_task_succeeds():
    first = _command()
    second = _command(analysis_id="analysis-2", source_id="record-2")
    identity = _active_identity(first)
    model = SequenceAnalysisModel()
    bus = _Bus()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: (_ for _ in ()).throw(SystemExit("calculation stop")),
        workflow_identity_provider=lambda: dict(identity),
    )

    assert controller.handle_analysis_requested(first) is False
    assert model.state is AnalysisState.FAILED
    assert controller.active_context is None
    assert [event.reason for event in bus.events.analysis_failed.values] == [
        "calculation stop"
    ]

    identity.update(_active_identity(second))
    controller.execute = lambda command: {"record_id": command.source_id}
    assert controller.handle_analysis_requested(second) is True
    assert controller.active_context is None
    assert len(bus.events.analysis_failed.values) == 1


def test_frozen_mode_and_empty_analysis_config_never_fall_back_to_live_runtime():
    created = []

    class Analysis:
        def __init__(self, name):
            self.name = name
            created.append(self)

    command = _command(
        recording_snapshot={
            "record_id": "record-1",
            "session": {"input_channels": [2, 5]},
        },
        configuration_snapshot=ConfigurationSnapshot(
            [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
            {},
            mic_channels=(9,),
        ),
    )
    runtime = SimpleNamespace(
        mode="IMPORT_AUDIO",
        data_struct=object(),
        _active_input_channels=[9],
        analysis_config={"golden_sample_result_path": "live.json"},
        analysis_types_requiring_v2pa=set(),
    )
    model = SequenceAnalysisModel()
    bus = _Bus()
    controller = None

    def execute(_admitted):
        runtime.mode = "PLAY_AND_RECORD"
        instance = controller.instance_analysis_class(
            "item", "X", {"analysis_channel": 5}
        )
        assert instance.name == "item--通道6"
        assert instance.analysis_config == {"analysis_channel": 1}
        return {"record_id": "record-1"}

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        runtime=runtime,
        readiness=lambda _command: True,
        execute=execute,
        class_mapping_provider=lambda: {"X": Analysis},
        workflow_identity_provider=lambda: _active_identity(command),
    )

    assert controller.handle_analysis_requested(command) is True
    assert len(created) == 1


def test_controller_owns_summary_and_tcp_payload_from_frozen_inputs():
    now = datetime(2026, 8, 20, 1, 2, 3, 456000)
    results = {"FFT": (True, "ok"), "SPL": (False, "ng")}

    assert SequenceAnalysisController.summarize_ok_ng(results) == (False, "NG")
    assert SequenceAnalysisController.build_tcp_result_payload(
        {"recorded_path": r"C:\records\sample.wav"}, results, now
    ) == {
        "TimeStamp": "2026-08-20 01:02:03,456",
        "Label": "NG",
        "FileName": "sample.wav",
    }


def test_geometry_loader_keeps_valid_subset_and_ignores_legacy_values(tmp_path):
    path = tmp_path / "analysis-geometry.json"
    path.write_text(
        '{"FFT": 1, "SPL": {"x": 1, "y": 2, "w": 600, "h": 500}, '
        '"BAD": {"x": 1, "y": 2, "w": 3, "h": 4}}',
        encoding="utf-8",
    )
    model = SequenceAnalysisModel()

    SequenceAnalysisView(model, geometry_path=path)

    assert model.geometry == {"SPL": {"x": 1, "y": 2, "w": 600, "h": 500}}
    model.replace_geometry({"FFT": 1})
    assert model.geometry == {}

    class HostileGeometry(dict):
        def items(self):
            raise RuntimeError("hostile mapping")

    model.replace_geometry(HostileGeometry())
    assert model.geometry == {}


def test_view_close_windows_releases_registry_and_all_window_references():
    closed = []

    class Window:
        def close(self):
            closed.append(self)

    model = SequenceAnalysisModel()
    first = Window()
    second = Window()
    model.register_instance("first", first)
    model.register_instance("second", second)
    view = object.__new__(SequenceAnalysisView)
    view.model = model
    view.summary_window = None
    view.feedback_dialogs = []
    view.window_keys = {}
    view.logger = None

    view.close_windows()

    assert closed == [first, second]
    assert model.analysis_instances == []
    assert model.analysis_registry == {}
    assert view.window_keys == {}




def test_real_bus_workflow_cancellation_consumes_one_analysis_failed_terminal():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-cancel",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda _record_id: (
            {"record_id": "record-1"},
            configuration,
        ),
        connect_bus=True,
    )
    model = SequenceAnalysisModel()
    failures = []
    bus.events.analysis_failed.connect(failures.append)
    controller = None

    def execute(command):
        assert workflow.handle_cancel_workflow(
            CancelWorkflowRequested(
                "cancel-1", command.workflow_generation, "stop"
            )
        )
        return {"record_id": command.source_id}

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=execute,
        workflow_identity_provider=lambda: {
            "analysis_id": workflow_model.active_analysis_id,
            "source_id": workflow_model.analysis_source_id,
            "workflow_generation": workflow_model.workflow_generation,
            "phase": workflow_model.phase.name,
            "cancelling_domain": workflow_model.cancelling_domain,
        },
    )
    bus.commands.analysis_requested.connect(controller.handle_analysis_requested)
    bus.commands.cancel_analysis_requested.connect(
        controller.handle_cancel_analysis_requested
    )

    bus.commands.manual_analysis_requested.emit(
        ManualAnalysisRequested("manual-cancel", "record-1")
    )
    _QAPP.processEvents()
    _QAPP.processEvents()

    assert workflow_model.phase is WorkflowPhase.IDLE
    assert len(failures) == 1
    assert failures[0].analysis_id == "analysis-cancel"
    assert "cancel" in failures[0].reason


def _admit_workflow_analysis(model, *, generation=4):
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = generation
    model.active_analysis_id = "analysis-1"
    model.analysis_source_id = "record-1"
    model.analysis_record_id = "record-1"
    model.retained_record_id = "record-1"
    model.awaiting_label = True


def _analysis_continuation_snapshot(*, targets=(), test_mode=True):
    snapshot = {
        "record_id": "record-1",
        "analysis_id": "analysis-1",
        "source_id": "record-1",
        "workflow_generation": 4,
        "analysis_result_dict": {"FFT": (True, "ok")},
        "ok_ng_summary": (True, "OK"),
        "can_output_ok_ng": True,
        "test_mode": test_mode,
        "tcp_result_payload": {"Label": "OK", "FileName": "sample.wav"},
        "export_handoff": {"record_id": "record-1"},
    }
    snapshot["analysis_configuration"] = (
        {
            "display_sequence": ("Output",),
            "Output": {
                "type": "Excel",
                "enabled": True,
                "save_mes_enabled": True,
            },
        }
        if targets
        else {}
    )
    return snapshot


def test_workflow_orders_export_auto_label_and_authorized_transport():
    assert hasattr(sequence_message_types, "AnalysisTransportReady")
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    _admit_workflow_analysis(model)
    order = []
    exports = []
    labels = []
    transports = []
    bus.commands.export_requested.connect(
        lambda command: (exports.append(command), order.append("export"))
    )
    bus.register_workflow_continuation_recipient(
        "workflow-state", "analysis-order-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "analysis-order-label",
        lambda command: (labels.append(command), order.append("label")) or True,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "analysis-order-transport",
        lambda event: (transports.append(event), order.append("tcp")) or True,
    )
    controller = SequenceWorkflowController(
        model,
        bus,
        job_id_factory=lambda: "job-1",
        label_id_factory=lambda: "auto-label-1",
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )
    event = AnalysisCompleted(
        "analysis-1",
        "record-1",
        _analysis_continuation_snapshot(
            targets=(
                {"type": "mes", "config_name": "Output"},
                {"type": "excel", "config_name": "Output"},
            )
        ),
    )

    assert controller.handle_analysis_completed(event) is True
    assert order == ["export"]
    assert model.phase is WorkflowPhase.RESULT_EXPORTING
    assert model.awaiting_label is True

    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "busy"},))
    )
    assert controller.handle_retry_export(RetryExportRequested("job-1", "attempt-1"))
    assert controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-2", "record-1", ({"reason": "busy"},))
    )
    assert order == ["export"]
    assert controller.handle_ignore_export_failure(
        IgnoreExportFailureRequested("job-1", "attempt-2")
    )

    assert order == ["export", "label"]
    assert len(labels) == 1
    assert isinstance(labels[0], CommitRecordingLabelRequested)
    assert labels[0].record_id == "record-1"
    assert labels[0].label == "OK"
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert model.retained_record_id == "record-1"
    assert model.awaiting_label is True
    assert transports == []

    assert controller.handle_label_committed(
        RecordingLabelCommitted(
            labels[0].command_id,
            labels[0].record_id,
            labels[0].label,
            {"saved": True},
        )
    )
    assert order == ["export", "label", "tcp"]
    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id is None
    assert model.awaiting_label is False
    assert model.post_analysis_continuation is None
    assert transports[0].analysis_id == "analysis-1"
    assert transports[0].workflow_generation == 4
    assert controller.handle_manual_label(
        ManualLabelRequested("duplicate-label", "record-1", "OK")
    ) is False
    export_owner.disconnect()


def test_workflow_with_no_real_export_target_skips_result_exporting():
    assert hasattr(sequence_message_types, "AnalysisTransportReady")
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    _admit_workflow_analysis(model)
    phases = []
    exports = []
    transports = []
    bus.commands.export_requested.connect(exports.append)
    bus.register_workflow_continuation_recipient(
        "workflow-state",
        "no-target-state",
        lambda event: phases.append(event.new_phase) or True,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "no-target-transport",
        lambda event: transports.append(event) or True,
    )
    controller = SequenceWorkflowController(
        model,
        bus,
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )
    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "record-1",
            _analysis_continuation_snapshot(targets=(), test_mode=False),
        )
    )

    assert exports == []
    assert WorkflowPhase.RESULT_EXPORTING not in phases
    assert model.phase is WorkflowPhase.IDLE
    assert model.post_analysis_continuation is None
    assert len(transports) == 1
    export_owner.disconnect()


def test_failed_auto_label_terminal_retains_canonical_waiting_then_authorizes_transport():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    _admit_workflow_analysis(model)
    labels = []
    transports = []
    bus.register_workflow_continuation_recipient(
        "workflow-state", "failed-label-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "failed-label-command",
        lambda command: labels.append(command) or True,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "failed-label-transport",
        lambda event: transports.append(event) or True,
    )
    controller = SequenceWorkflowController(
        model,
        bus,
        label_id_factory=lambda: "auto-label-failed",
        connect_bus=False,
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )

    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "record-1",
            _analysis_continuation_snapshot(targets=()),
        )
    )
    assert len(labels) == 1
    command = labels[0]
    assert transports == []

    assert controller.handle_label_failed(
        sequence_message_types.RecordingLabelCommitFailed(
            command.command_id,
            command.record_id,
            command.label,
            "database unavailable",
        )
    )
    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id == "record-1"
    assert model.awaiting_label is True
    assert model.post_analysis_continuation is None
    assert len(transports) == 1
    export_owner.disconnect()


def test_export_service_builds_only_real_frozen_export_targets():
    assert not hasattr(SequenceAnalysisController, "build_export_targets")
    assert SequenceExportService.resolve_target_configurations({}) == ()
    assert SequenceExportService.resolve_target_configurations(
        {
            "display_sequence": ["Output"],
            "Output": {
                "type": "Excel",
                "enabled": True,
                "save_mes_enabled": True,
                "file_path": "result.xlsx",
            },
            "FFT": {"type": "FFT"},
        }
    ) == (
        {
            "type": "mes",
            "config_name": "Output",
            "configuration": {
                "type": "Excel",
                "enabled": True,
                "save_mes_enabled": True,
                "file_path": "result.xlsx",
            },
        },
        {
            "type": "excel",
            "config_name": "Output",
            "configuration": {
                "type": "Excel",
                "enabled": True,
                "save_mes_enabled": True,
                "file_path": "result.xlsx",
            },
        },
    )


def test_facade_has_no_analysis_completed_business_adapter():
    assert not hasattr(SequenceWindow, "_handle_analysis_completed_compatibility")
    assert not hasattr(SequenceWindow, "_apply_test_mode_analysis_handoff")


@pytest.mark.parametrize("failure", [KeyboardInterrupt("identity stop"), SystemExit("identity exit"), RuntimeError("identity bad")])
def test_identity_provider_failure_publishes_one_matching_failed_terminal(failure):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-identity",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda _record_id: (
            {"record_id": "record-1"},
            configuration,
        ),
        connect_bus=True,
    )
    model = SequenceAnalysisModel()
    failures = []
    bus.events.analysis_failed.connect(failures.append)

    def broken_identity():
        raise failure

    analysis = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda command: {"record_id": command.source_id},
        workflow_identity_provider=broken_identity,
    )
    bus.commands.analysis_requested.connect(analysis.handle_analysis_requested)

    bus.commands.manual_analysis_requested.emit(
        ManualAnalysisRequested("manual-identity", "record-1")
    )
    _QAPP.processEvents()
    _QAPP.processEvents()

    assert len(failures) == 1
    assert failures[0].analysis_id == "analysis-identity"
    assert model.state is AnalysisState.FAILED
    assert model.result_snapshot == {}
    assert analysis.active_context is None
    assert workflow.model.phase is WorkflowPhase.IDLE


def test_completed_payload_canonicalizes_identity_before_atomic_model_commit():
    command = _command()
    result = {
        "analysis_id": "forged-analysis",
        "source_id": "forged-source",
        "workflow_generation": 999,
        "values": [1, 2],
    }
    bus = _Bus()
    model = SequenceAnalysisModel()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: result,
        workflow_identity_provider=lambda: _active_identity(command),
    )

    assert controller.handle_analysis_requested(command) is True
    result["values"].append(3)

    event = bus.events.analysis_completed.values[0]
    assert event.result_snapshot["analysis_id"] == command.analysis_id
    assert event.result_snapshot["source_id"] == command.source_id
    assert event.result_snapshot["workflow_generation"] == command.workflow_generation
    assert event.result_snapshot["values"] == (1, 2)
    assert model.result_snapshot["values"] == (1, 2)


def test_unfreezable_or_hostile_result_never_leaves_partial_snapshot_and_next_runs():
    command = _command()
    second = _command(analysis_id="analysis-2", source_id="record-2")
    identity = _active_identity(command)
    bus = _Bus()
    model = SequenceAnalysisModel()
    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda _command: {"bad": SimpleNamespace(value=1)},
        workflow_identity_provider=lambda: dict(identity),
    )

    assert controller.handle_analysis_requested(command) is False
    assert model.state is AnalysisState.FAILED
    assert model.result_snapshot == {}
    assert bus.events.analysis_completed.values == []

    class HostileMapping(Mapping):
        def __iter__(self):
            raise RuntimeError("hostile keys")

        def __len__(self):
            return 1

        def __getitem__(self, key):
            return 1

    identity.update(_active_identity(second))
    controller.execute = lambda _command: HostileMapping()
    assert controller.handle_analysis_requested(second) is False
    assert model.result_snapshot == {}

    third = _command(analysis_id="analysis-3", source_id="record-3")
    identity.update(_active_identity(third))
    controller.execute = lambda command: {"record_id": command.source_id}
    assert controller.handle_analysis_requested(third) is True
    assert model.state is AnalysisState.COMPLETED


def test_view_close_windows_isolates_destroyed_feedback_dialogs():
    closed = []

    class DestroyedDialog:
        def close(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class Dialog:
        def close(self):
            closed.append("closed")

    view = object.__new__(SequenceAnalysisView)
    view.model = SequenceAnalysisModel()
    view.summary_window = None
    view.feedback_dialogs = [DestroyedDialog(), Dialog()]
    view.window_keys = {}
    view.logger = None

    view.close_windows()

    assert closed == ["closed"]
    assert view.feedback_dialogs == []


def test_production_analysis_wiring_makes_admitted_cancel_win_completion_race():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    workflow_model.awaiting_label = True
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-cancel-production",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda _record_id: (
            {"record_id": "record-1"},
            configuration,
        ),
        connect_bus=True,
    )
    side_effects = []
    model = SequenceAnalysisModel()
    controller = None

    def execute(command):
        assert workflow.handle_cancel_workflow(
            CancelWorkflowRequested(
                "cancel-production",
                command.workflow_generation,
                "shutdown",
            )
        )
        return _analysis_continuation_snapshot(
            targets=({"type": "excel", "config_name": "Output"},)
        )

    controller = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=execute,
        workflow_identity_provider=lambda: {
            "analysis_id": workflow_model.active_analysis_id,
            "source_id": workflow_model.analysis_source_id,
            "workflow_generation": workflow_model.workflow_generation,
            "phase": workflow_model.phase.name,
            "cancelling_domain": workflow_model.cancelling_domain,
        },
    )
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = workflow_model
    window.analysis_controller = controller
    window._project_workflow_state = lambda _message: True
    window._handle_legacy_analysis_export_requested = lambda _command: side_effects.append("export")
    window._handle_legacy_analysis_label_commit_requested = lambda _command: side_effects.append("label")
    SequenceWindow._wire_analysis_workflow_channels(window)

    bus.commands.manual_analysis_requested.emit(
        ManualAnalysisRequested("manual-cancel-production", "record-1")
    )
    for _ in range(5):
        _QAPP.processEvents()

    assert workflow_model.phase is WorkflowPhase.IDLE
    assert model.state is AnalysisState.FAILED
    assert workflow_model.post_analysis_continuation is None
    assert side_effects == []


def _formal_mes_export_command(tmp_path, *, summary=(True, "OK")):
    return ExportRequested(
        "job-mes",
        "record-1",
        {
            "export_handoff": {
                "record_id": "record-1",
                "sn": "SN-1",
                "analysis_result_dict": {"FFT": (True, "ok")},
                "ok_ng_summary": summary,
                "can_output_ok_ng": True,
                "analysis_config": {
                    "display_sequence": ["Output"],
                    "Output": {
                        "type": "Excel",
                        "enabled": True,
                        "save_mes_enabled": True,
                        "mes_file_base": str(tmp_path),
                        "mes_file_name": "MES_Result",
                    },
                },
            }
        },
        (
            {"type": "mes", "config_name": "Output", "configuration": {}},
            {
                "type": "excel",
                "config_name": "Output",
                "configuration": {"fast_mode": False},
            },
        ),
    )


def _legacy_export_window(bus):
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.default_logger = SimpleNamespace(
        debug=lambda _message: None,
        info=lambda _message: None,
        warning=lambda _message: None,
        error=lambda _message: None,
    )
    return window





def test_workflow_authorized_queued_transport_survives_next_generation_once():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    _admit_workflow_analysis(model)
    model.awaiting_label = False
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = model
    sent = []
    window._project_workflow_state = lambda _message: True
    authorized = []
    SequenceWindow._wire_workflow_continuation_ports(window)
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "authorized-capture",
        lambda event: authorized.append(event) or True,
    )
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=model.is_analysis_transport_authorized,
        authorization_consumer=model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: SimpleNamespace(
            send_to_current_client=lambda payload: sent.append(payload) or True
        ),
    )
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    controller = SequenceWorkflowController(
        model,
        bus,
        analysis_id_factory=lambda: "analysis-next",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda record_id: (
            {"record_id": record_id},
            configuration,
        ),
        connect_bus=False,
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _FormalExportView(), bus=bus
    )

    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "record-1",
            _analysis_continuation_snapshot(targets=(), test_mode=False),
        )
    )
    assert len(authorized) == 1
    transport = authorized[0]
    assert len(sent) == 1
    assert '"Label": "OK"' in sent[0]
    forged = sequence_message_types.AnalysisTransportReady(
        transport.analysis_id,
        transport.source_id,
        transport.record_id,
        transport.workflow_generation,
        transport.payload,
    )
    assert transport_owner.handle_analysis_transport_ready(forged) is False
    assert len(sent) == 1

    model.retained_record_id = "record-2"
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("manual-next", "record-2")
    )
    assert model.workflow_generation == 5
    assert len(sent) == 1
    # The retained raw signal remains a non-canonical compatibility surface.
    bus.events.analysis_transport_ready.emit(transport)
    _QAPP.processEvents()
    assert len(sent) == 1
    export_owner.disconnect()


@pytest.mark.parametrize(
    "provider_failure",
    [
        RuntimeError("identity unavailable " + "x" * 2000),
        KeyboardInterrupt("identity interrupted"),
        SystemExit("identity exited"),
    ],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_production_direct_cancel_contains_identity_provider_failure(
    provider_failure, monkeypatch
):
    qt_failures = []
    monkeypatch.setattr(
        sys,
        "excepthook",
        lambda error_type, error, traceback: qt_failures.append(
            (error_type, error, traceback)
        ),
    )
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    workflow_model.awaiting_label = True
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-cancel-provider",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda _record_id: (
            {"record_id": "record-1"},
            configuration,
        ),
        connect_bus=True,
    )
    model = SequenceAnalysisModel()
    observed_pending = []
    terminal_failures = []
    terminal_completions = []
    side_effects = []
    logs = []
    bus.events.analysis_failed.connect(terminal_failures.append)
    bus.events.analysis_completed.connect(terminal_completions.append)

    def workflow_identity():
        if workflow_model.phase is WorkflowPhase.CANCELLING:
            raise provider_failure
        return {
            "analysis_id": workflow_model.active_analysis_id,
            "source_id": workflow_model.analysis_source_id,
            "workflow_generation": workflow_model.workflow_generation,
            "phase": workflow_model.phase.name,
            "cancelling_domain": workflow_model.cancelling_domain,
        }

    def execute(command):
        assert workflow.handle_cancel_workflow(
            CancelWorkflowRequested(
                "cancel-provider",
                command.workflow_generation,
                "shutdown",
            )
        )
        observed_pending.append(model.cancel_pending)
        return _analysis_continuation_snapshot(
            targets=({"type": "excel", "config_name": "Output"},)
        )

    analysis = SequenceAnalysisController(
        model,
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=execute,
        workflow_identity_provider=workflow_identity,
        logger=SimpleNamespace(
            debug=logs.append,
            info=logs.append,
            warning=logs.append,
            error=logs.append,
        ),
    )
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = workflow_model
    window.analysis_controller = analysis
    window._project_workflow_state = lambda _message: True
    window._handle_legacy_analysis_export_requested = (
        lambda _command: side_effects.append("export")
    )
    window._handle_legacy_analysis_label_commit_requested = (
        lambda _command: side_effects.append("label")
    )
    SequenceWindow._wire_analysis_workflow_channels(window)

    bus.commands.manual_analysis_requested.emit(
        ManualAnalysisRequested("manual-cancel-provider", "record-1")
    )
    for _ in range(5):
        _QAPP.processEvents()

    assert qt_failures == []
    assert observed_pending == [True]
    assert model.state is AnalysisState.FAILED
    assert workflow_model.phase is WorkflowPhase.IDLE
    assert len(terminal_failures) == 1
    assert terminal_completions == []
    assert side_effects == []
    assert any("identity provider failed" in message for message in logs)
    assert max(map(len, logs), default=0) <= 700


class _FormalExportView:
    def show_progress(self, *_args):
        return None

    def show_failure(self, *_args):
        return None

    def finish(self, *_args):
        return None


def _run_formal_export(bus, command, service):
    submissions = []
    completed = []
    failed = []
    bus.events.export_completed.connect(completed.append)
    bus.events.export_failed.connect(failed.append)
    controller = SequenceExportController(
        SequenceExportModel(),
        _FormalExportView(),
        bus=bus,
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=True,
    )
    bus.commands.export_requested.emit(command)
    for _ in range(8):
        _QAPP.processEvents()
        if submissions:
            break
    assert len(submissions) == 1
    work, attempt = submissions[0]
    outcome = service.execute_record_job(work, attempt.attempt_id)
    if outcome.ok:
        assert controller.handle_worker_completed(outcome) is True
    else:
        assert controller.handle_worker_failed(outcome) is True
    return outcome, completed, failed, controller


def _formal_export_service(order, *, mes_result=None):
    def write_mes(_configuration, *, sn, label, logger):
        order.append(("mes", sn, label))
        return mes_result or SimpleNamespace(ok=True, message="written")

    return SequenceExportService(
        mes_validator=lambda _configuration: (True, ""),
        mes_writer=write_mes,
        output_path_resolver=lambda _configuration, *, product_model: "result.xlsx",
        excel_exporter=lambda _configuration, **kwargs: (
            order.append(("excel", kwargs)) or ExportResult(True, "saved")
        ),
    )


def test_real_facade_bus_workflow_analysis_chain_uses_formal_export_owner():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    configuration = ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}], {}
    )
    order = []
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.sequence_event_bus = bus
    window.workflow_model = workflow_model
    workflow_model.awaiting_label = True
    submissions = []
    export_service = _formal_export_service(order)
    export_controller = SequenceExportController(
        SequenceExportModel(),
        _FormalExportView(),
        bus=bus,
        service=export_service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=True,
    )
    transport_owner = SequenceAnalysisTransportController(
        bus=bus,
        authorization_provider=workflow_model.is_analysis_transport_authorized,
        authorization_consumer=workflow_model.consume_analysis_transport,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: SimpleNamespace(
            send_to_current_client=lambda _payload: order.append(("tcp",)) or True
        ),
    )
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-1",
        job_id_factory=lambda: "job-1",
        label_id_factory=lambda: "auto-label-1",
        configuration_snapshot_provider=lambda: configuration,
        recording_snapshot_lookup=lambda _record_id: (
            {"record_id": "record-1", "recorded_path": "sample.wav"},
            configuration,
        ),
        connect_bus=True,
    )
    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        label_service=SimpleNamespace(
            commit=lambda command, _project: (
                order.append(("label", command.label)) or {}
            )
        ),
        connect_queued=True,
    )
    controller = SequenceAnalysisController(
        SequenceAnalysisModel(),
        _View(),
        bus=bus,
        readiness=lambda _command: True,
        execute=lambda command: {
            "record_id": command.source_id,
            "analysis_result_dict": {"FFT": (True, "ok")},
            "ok_ng_summary": (True, "OK"),
            "can_output_ok_ng": True,
            "test_mode": True,
            "tcp_result_payload": {"Label": "OK", "FileName": "sample.wav"},
            "export_handoff": {
                "record_id": "record-1",
                "sn": "SN-1",
                "analysis_result_dict": {"FFT": (True, "ok")},
                "ok_ng_summary": (True, "OK"),
                "can_output_ok_ng": True,
                "analysis_config": {
                    "display_sequence": ("Output",),
                    "Output": {
                        "type": "Excel",
                        "enabled": True,
                        "save_mes_enabled": True,
                        "fast_mode": False,
                    },
                },
            },
            "analysis_configuration": {
                "display_sequence": ("Output",),
                "Output": {
                    "type": "Excel",
                    "enabled": True,
                    "save_mes_enabled": True,
                    "fast_mode": False,
                },
            },
        },
        workflow_identity_provider=lambda: SequenceWindow._analysis_workflow_identity(
            window
        ),
    )
    window.analysis_controller = controller
    window.export_controller = export_controller
    window._project_workflow_state = lambda _message: True
    SequenceWindow._wire_workflow_continuation_ports(window)
    SequenceWindow._wire_analysis_workflow_channels(window)

    bus.commands.manual_analysis_requested.emit(
        ManualAnalysisRequested("manual-1", "record-1")
    )
    for _ in range(12):
        _QAPP.processEvents()
        if submissions:
            break
    assert len(submissions) == 1
    work, attempt = submissions[0]
    outcome = export_service.execute_record_job(work, attempt.attempt_id)
    assert export_controller.handle_worker_completed(outcome) is True
    for _ in range(12):
        _QAPP.processEvents()

    assert workflow.model.phase is WorkflowPhase.IDLE
    assert [item[0] for item in order] == ["mes", "excel", "label", "tcp"]
    assert workflow.model.retained_record_id is None
    assert workflow.model.awaiting_label is False
    assert recording is not None


def test_formal_export_owner_thaws_only_the_completed_frozen_handoff():
    bus = SequenceEventBus()
    received = []
    command = ExportRequested(
        "job-1",
        "record-1",
        {
            "export_handoff": {
                "record_id": "record-1",
                "sn": "SN-1",
                "analysis_config": {"display_sequence": ("Excel",)},
                "analysis_result_dict": {"FFT": (True, "ok")},
                "ok_ng_summary": (True, "OK"),
                "can_output_ok_ng": True,
            }
        },
        (
            {"type": "mes", "config_name": "MES", "configuration": {}},
            {
                "type": "excel",
                "config_name": "Excel",
                "configuration": {"fast_mode": False},
            },
        ),
    )
    service = SequenceExportService(
        mes_validator=lambda _configuration: (True, ""),
        mes_writer=lambda _configuration, *, sn, label, logger: (
            received.append(("mes", sn, label))
            or SimpleNamespace(ok=True, message="written")
        ),
        output_path_resolver=lambda _configuration, *, product_model: "result.xlsx",
        excel_exporter=lambda _configuration, **kwargs: (
            received.append(("excel", kwargs)) or ExportResult(True, "saved")
        ),
    )

    outcome, completed, failed, _controller = _run_formal_export(
        bus, command, service
    )

    assert outcome.ok is True
    assert [item[0] for item in received] == ["mes", "excel"]
    assert received[1][1]["analysis_config"]["display_sequence"] == ["Excel"]
    assert received[1][1]["analysis_result_dict"]["FFT"] == [True, "ok"]
    assert len(completed) == 1
    assert failed == []


def test_formal_export_service_normalizes_frozen_summary_before_excel(tmp_path):
    bus = SequenceEventBus()
    order = []
    service = _formal_export_service(order)

    outcome, completed, failed, _controller = _run_formal_export(
        bus, _formal_mes_export_command(tmp_path), service
    )

    assert outcome.ok is True
    assert [item[0] for item in order] == ["mes", "excel"]
    assert order[0][1:3] == ("SN-1", "OK")
    assert len(completed) == 1
    assert failed == []


def test_formal_export_service_reports_mes_failure_without_running_excel(tmp_path):
    bus = SequenceEventBus()
    order = []
    service = _formal_export_service(
        order, mes_result=SimpleNamespace(ok=False, message="MES disk unavailable")
    )

    outcome, completed, failed, _controller = _run_formal_export(
        bus, _formal_mes_export_command(tmp_path), service
    )

    assert outcome.ok is False
    assert [item[0] for item in order] == ["mes"]
    assert completed == []
    assert len(failed) == 1
    assert "MES disk unavailable" in failed[0].failures[0]["message"]


@pytest.mark.parametrize("summary", [(True,), {"passed": True}])
def test_formal_export_service_rejects_malformed_summary_without_side_effects(
    tmp_path, summary
):
    bus = SequenceEventBus()
    writes = []
    service = _formal_export_service(writes)

    outcome, completed, failed, _controller = _run_formal_export(
        bus, _formal_mes_export_command(tmp_path, summary=summary), service
    )

    assert outcome.ok is False
    assert writes == []
    assert completed == []
    assert len(failed) == 1
    assert "mes_write_skip_bad_summary" in failed[0].failures[0]["message"]
