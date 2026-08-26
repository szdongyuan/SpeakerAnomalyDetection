from __future__ import annotations

import ast
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread, get_ident
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from PyQt5 import sip
from PyQt5.QtCore import (
    QCoreApplication,
    QEvent,
    QObject,
    QEventLoop,
    QTimer,
    pyqtSignal,
    qInstallMessageHandler,
)
from PyQt5.QtWidgets import QApplication

import ui.sequence.sequence_workflow_controller as workflow_controller_module
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_event_bus import (
    RetainedCleanupLifecycleRegistrationResult,
    SequenceEventBus,
)
from ui.sequence.sequence_messages import (
    AnalysisCompleted,
    AnalysisExportPrepared,
    AbortShutdownRequested,
    ConfirmShutdownCancellationRequested,
    ConfigurationSnapshot,
    ImportedAudioReady,
    ManualAnalysisRequested,
    ManualLabelRequested,
    RecordingCompleted,
    RecordingLabelCommitted,
    ShutdownFlushCompleted,
    ShutdownRequested,
)
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingStreamingStimulusPort,
)
from ui.sequence.sequence_recording_service import RecordingManualLabelRequestService
from ui.sequence.sequence_workflow_controller import (
    SequenceShutdownCoordinator,
    SequenceWorkflowController,
)
from ui.sequence.sequence_workflow_model import (
    PostAnalysisContinuation,
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


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET = ROOT / "ui" / "sequence" / "sequence_widget.py"
WORKFLOW_CONTROLLER = ROOT / "ui" / "sequence" / "sequence_workflow_controller.py"
_QT_APPLICATIONS = []


def _configuration(mode: str = "RECORD_ONLY") -> ConfigurationSnapshot:
    return ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": mode, "detail": {}}, "analysis_list": {}}}],
        {},
    )


def test_streaming_stimulus_is_recording_owned_and_facade_is_a_thin_projection():
    recording = RecordingModel()
    configuration = SequenceConfigurationModel(
        streaming_stimulus_port=RecordingStreamingStimulusPort(recording)
    )
    payload = [0.25, -0.25]

    configuration.streaming_stimulus_data = payload

    assert recording.streaming_stimulus_data is payload
    assert configuration.streaming_stimulus_data is payload
    assert "_streaming_stimulus_data" not in vars(configuration)

    source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    accessors = [
        node
        for node in window.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "streaming_stimulus_data"
    ]
    assert len(accessors) == 2
    assert all("recording_model" in ast.unparse(node) for node in accessors)
    assert all("configuration_model" not in ast.unparse(node) for node in accessors)


def test_retained_recording_snapshot_is_immutable_exact_and_generation_guarded():
    model = RecordingModel()
    recording = {"record_id": "record-1", "samples": [1.0, 2.0]}
    configuration = _configuration()

    assert model.retain_recording_snapshot(
        "record-1",
        recording,
        configuration,
        source_id="session-1",
        workflow_generation=4,
    )
    retained = model.retained_recording_snapshot("record-1")
    assert retained.record_id == "record-1"
    assert retained.source_id == "session-1"
    assert retained.workflow_generation == 4
    assert retained.recording_snapshot["samples"] == (1.0, 2.0)
    recording["samples"].append(3.0)
    with pytest.raises(TypeError):
        retained.recording_snapshot["record_id"] = "changed"

    first = retained
    assert model.retain_recording_snapshot(
        "record-1",
        {"record_id": "record-1", "samples": [99.0]},
        configuration,
        source_id="session-1",
        workflow_generation=4,
    )
    assert model.retained_recording_snapshot("record-1") is first
    assert not model.retain_recording_snapshot(
        "stale",
        {"record_id": "stale"},
        configuration,
        source_id="old-session",
        workflow_generation=3,
    )
    assert model.retained_recording_snapshot("stale") is None
    assert model.retained_analysis_inputs("wrong") is None
    assert not model.clear_retained_recording_snapshot(
        "record-1", workflow_generation=3
    )
    assert model.clear_retained_recording_snapshot(
        "record-1", workflow_generation=4
    )
    assert model.retained_recording_snapshot("record-1") is None


def test_retained_recording_can_be_cleared_by_the_same_record_in_a_later_generation():
    model = RecordingModel()
    assert model.retain_recording_snapshot(
        "record-1",
        {"record_id": "record-1"},
        _configuration(),
        source_id="session-1",
        workflow_generation=4,
    )
    assert model.clear_retained_recording_snapshot(
        "record-1", workflow_generation=5
    )
    assert model.retained_recording_snapshot("record-1") is None


@pytest.mark.parametrize(
    "source",
    [
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        np.arange(12, dtype=np.int16).reshape(3, 4)[:, ::2],
        np.empty((0, 3), dtype=np.float64),
    ],
    ids=("owning-float32", "noncontiguous-int16", "empty-float64"),
)
def test_retained_numpy_payload_is_irreversible_and_detached(source):
    model = RecordingModel()
    expected = np.array(source, copy=True)
    source_alias = source
    recording = {"record_id": "record-1", "samples": source}

    assert model.retain_recording_snapshot(
        "record-1",
        recording,
        _configuration(),
        source_id="session-1",
        workflow_generation=1,
    )
    frozen = model.retained_recording_snapshot("record-1").recording_snapshot[
        "samples"
    ]
    assert frozen.dtype == expected.dtype
    assert frozen.shape == expected.shape
    np.testing.assert_array_equal(frozen, expected)
    assert frozen.flags.writeable is False
    assert type(frozen.base) is bytes
    with pytest.raises(ValueError):
        frozen.setflags(write=True)

    if source_alias.size:
        source_alias[...] = 99
    np.testing.assert_array_equal(frozen, expected)

    returned_recording, _returned_configuration = model.retained_analysis_inputs(
        "record-1"
    )
    returned_samples = returned_recording["samples"]
    with pytest.raises(ValueError):
        returned_samples.setflags(write=True)
    with pytest.raises(ValueError):
        returned_samples[...] = 123
    returned_recording["samples"] = np.zeros(expected.shape, dtype=expected.dtype)
    np.testing.assert_array_equal(
        model.retained_recording_snapshot("record-1").recording_snapshot["samples"],
        expected,
    )


def test_retained_recording_reentry_is_rejected_and_baseexception_releases_gate():
    model = RecordingModel()
    configuration = _configuration()
    nested_results = []

    class ReenteringMapping(dict):
        def items(self):
            nested_results.append(
                model.retain_recording_snapshot(
                    "nested",
                    {"record_id": "nested"},
                    configuration,
                    source_id="nested-source",
                    workflow_generation=1,
                )
            )
            return super().items()

    assert model.retain_recording_snapshot(
        "record-1",
        ReenteringMapping(record_id="record-1"),
        configuration,
        source_id="session-1",
        workflow_generation=1,
    )
    assert nested_results == [False]

    class StopNow(BaseException):
        pass

    class ExplodingMapping(dict):
        def items(self):
            raise StopNow()

    with pytest.raises(StopNow):
        model.retain_recording_snapshot(
            "record-2",
            ExplodingMapping(record_id="record-2"),
            configuration,
            source_id="session-2",
            workflow_generation=2,
        )
    assert model.retained_recording_snapshot("record-1") is not None
    assert model.retain_recording_snapshot(
        "record-2",
        {"record_id": "record-2"},
        configuration,
        source_id="session-2",
        workflow_generation=2,
    )


def _workflow_with_recording_owner(*, automatic: bool):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    recording = RecordingModel()
    analysis = []
    preparations = []
    bus.commands.analysis_requested.connect(analysis.append)
    controller = SequenceWorkflowController(
        model,
        bus,
        configuration_snapshot_provider=_configuration,
        recording_snapshot_lookup=recording.retained_analysis_inputs,
        retain_recording_snapshot=recording.retain_recording_snapshot,
        clear_retained_recording_snapshot=recording.clear_retained_recording_snapshot,
        automatic_analysis_policy=_StaticAutomaticAnalysisPolicy(automatic),
        connect_bus=False,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "recording-facade-port-test",
        lambda message: preparations.append(message) or True,
        owner=controller,
    )
    bus.register_workflow_continuation_recipient(
        "workflow-state",
        "recording-facade-port-state",
        lambda _message: True,
        owner=controller,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "recording-facade-port-transport",
        lambda _message: True,
        owner=controller,
    )
    return bus, model, recording, controller, analysis, preparations


def test_recording_completion_idle_then_manual_analysis_uses_retained_owner():
    _bus, workflow, recording, controller, analysis, _preparations = (
        _workflow_with_recording_owner(automatic=False)
    )
    workflow.phase = WorkflowPhase.RECORDING
    workflow.active_session_id = "session-1"
    workflow.active_session_origin = SessionOrigin.CANONICAL
    workflow.configuration_snapshot = _configuration()
    completed = RecordingCompleted(
        "session-1", 2, {"record_id": "record-1", "samples": [1.0, 2.0]}
    )

    assert controller.handle_recording_completed(completed)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.recording_snapshot is None
    retained = recording.retained_recording_snapshot("record-1")
    assert retained is not None
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("manual-1", "record-1")
    )
    assert analysis[-1].recording_snapshot["samples"] == (1.0, 2.0)


def test_import_analysis_completion_idle_then_repeat_manual_uses_retained_owner():
    _bus, workflow, recording, controller, analysis, preparations = (
        _workflow_with_recording_owner(automatic=True)
    )
    workflow.phase = WorkflowPhase.IMPORTING
    workflow.active_import_id = "import-1"
    workflow.configuration_snapshot = _configuration("IMPORT_AUDIO")

    assert controller.handle_imported_audio_ready(
        ImportedAudioReady(
            "import-1", {"record_id": "import-record", "samples": [3.0]}
        )
    )
    admitted = analysis[-1]
    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            admitted.analysis_id,
            admitted.source_id,
            {"record_id": "import-record"},
        )
    )
    preparation = preparations[-1]
    assert controller.handle_analysis_export_prepared(
        AnalysisExportPrepared(
            preparation.request_id,
            preparation.analysis_id,
            preparation.source_id,
            preparation.record_id,
            preparation.workflow_generation,
            preparation.result_snapshot,
            (),
        )
    )
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.recording_snapshot is None
    assert recording.retained_recording_snapshot("import-record") is not None
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("manual-repeat", "import-record")
    )
    assert analysis[-1].automatic is False
    assert analysis[-1].recording_snapshot["samples"] == (3.0,)


def _label_committing_workflow(*, clear, connect_bus=False):
    _wait_for_qt(0)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.phase = WorkflowPhase.LABEL_COMMITTING
    workflow.workflow_generation = 4
    workflow.retained_record_id = "record-1"
    workflow.awaiting_label = True
    workflow.active_label_command_id = "label-1"
    workflow.active_label_record_id = "record-1"
    workflow.active_label = "OK"
    states = []
    bus.events.workflow_state_changed.connect(states.append)
    controller = SequenceWorkflowController(
        workflow,
        bus,
        clear_retained_recording_snapshot=clear,
        connect_bus=connect_bus,
    )
    event = RecordingLabelCommitted("label-1", "record-1", "OK", {})
    return workflow, controller, event, states


def _wait_for_qt(milliseconds: int) -> None:
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
        _QT_APPLICATIONS.append(application)
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _drain_deferred_timer_deletion(timer) -> None:
    _wait_for_qt(0)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(timer)


def _workflow_logical_snapshot(workflow):
    return {
        name: deepcopy(value)
        for name, value in vars(workflow).items()
        if name != "_analysis_transport_authorization_lock"
    }


class _InjectedRetirementTimeout:
    def __init__(self, invoke):
        self._invoke = invoke

    def disconnect(self, *_callbacks):
        return self._invoke("disconnect")


class _InjectedRetirementTimer:
    def __init__(self, invoke):
        self._invoke = invoke
        self.timeout = _InjectedRetirementTimeout(invoke)

    def stop(self):
        return self._invoke("stop")

    def setParent(self, parent):
        assert parent is None
        return self._invoke("unparent")

    def deleteLater(self):
        return self._invoke("delete")


class _InjectedRetirementBridge:
    def __init__(self, timer, lifecycle):
        self._timer = timer
        self._lifecycle = lifecycle

    def request_retirement(self):
        failures = workflow_controller_module._retire_timer_on_owner_thread(
            self._timer,
            lambda: None,
        )
        if failures:
            self._lifecycle._record_timer_retirement_failures(failures)
        return True


def test_queued_label_terminal_automatically_retries_cleanup_only_until_success():
    _wait_for_qt(0)
    outcomes = iter(
        (
            False,
            RuntimeError("ordinary"),
            KeyboardInterrupt("interrupt"),
            SystemExit("exit"),
            "reenter",
            True,
        )
    )
    calls = []
    nested = []
    diagnostics = []
    holder = {}

    def clear(record_id, *, workflow_generation):
        calls.append((record_id, workflow_generation))
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "reenter":
            nested.append(
                holder["controller"].handle_label_committed(holder["event"])
            )
            return True
        return outcome

    workflow, controller, event, states = _label_committing_workflow(
        clear=clear,
        connect_bus=True,
    )
    holder.update(controller=controller, event=event)
    controller.diagnostic_callback = diagnostics.append
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 8
    retry_timer = controller._retained_cleanup_retry_timer
    repeated_label_commits = []
    repeated_exports = []
    controller.bus.commands.commit_recording_label_requested.connect(
        repeated_label_commits.append
    )
    controller.bus.commands.export_requested.connect(repeated_exports.append)

    controller.bus.events.recording_label_committed.emit(event)
    _wait_for_qt(100)

    assert calls == [("record-1", 4)] * 6
    assert nested == [False]
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert [state.new_phase for state in states] == ["IDLE"]
    assert controller.pending_retained_cleanup_identity is None
    assert retry_timer.isSingleShot()
    assert retry_timer.isActive() is False
    assert repeated_label_commits == []
    assert repeated_exports == []
    assert sum(
        diagnostic["event_kind"] == "retained_recording_cleanup_retry"
        for diagnostic in diagnostics
    ) == 5


def test_pending_cleanup_ignores_duplicate_and_stale_terminals_and_is_diagnostic():
    _wait_for_qt(0)
    calls = []
    diagnostics = []

    def clear(record_id, *, workflow_generation):
        calls.append((record_id, workflow_generation))
        return False

    workflow, controller, event, _states = _label_committing_workflow(
        clear=clear,
        connect_bus=True,
    )
    controller.diagnostic_callback = diagnostics.append
    controller._retained_cleanup_retry_base_delay_ms = 50
    controller._retained_cleanup_retry_max_delay_ms = 50
    controller.bus.events.recording_label_committed.emit(event)
    _wait_for_qt(5)
    assert calls == [("record-1", 4)]
    assert controller.pending_retained_cleanup_identity == (
        "label-1",
        "record-1",
        "OK",
        4,
    )
    assert controller.retained_cleanup_retry_attempt == 1
    assert controller.retained_cleanup_retry_delay_ms == 50
    assert controller._retained_cleanup_retry_timer.isActive()

    controller.bus.events.recording_label_committed.emit(event)
    controller.bus.events.recording_label_committed.emit(
        RecordingLabelCommitted("stale-label", "record-1", "OK", {})
    )
    _wait_for_qt(5)
    assert calls == [("record-1", 4)]
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert controller.pending_retained_cleanup_identity is not None
    assert diagnostics

    controller.disconnect()
    assert controller.pending_retained_cleanup_identity is None
    assert not sip.isdeleted(controller._retained_cleanup_retry_timer)
    _drain_deferred_timer_deletion(
        controller._retained_cleanup_retry_timer
    )
    _wait_for_qt(60)
    assert calls == [("record-1", 4)]


def test_live_disconnect_retires_bus_owned_cleanup_capsule_while_bus_survives():
    class Cleanup:
        def __init__(self):
            self.calls = 0

        def __call__(self, _record_id, **_identity):
            self.calls += 1
            return False

    class Diagnostic:
        def __call__(self, _context):
            return None

    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.phase = WorkflowPhase.LABEL_COMMITTING
    workflow.workflow_generation = 4
    workflow.retained_record_id = "record-1"
    workflow.awaiting_label = True
    workflow.active_label_command_id = "label-1"
    workflow.active_label_record_id = "record-1"
    workflow.active_label = "OK"
    cleanup = Cleanup()
    diagnostic = Diagnostic()
    controller = SequenceWorkflowController(
        workflow,
        bus,
        clear_retained_recording_snapshot=cleanup,
        diagnostic_callback=diagnostic,
        connect_bus=True,
    )
    timer = controller._retained_cleanup_retry_timer
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    event = RecordingLabelCommitted("label-1", "record-1", "OK", {})

    assert controller.handle_label_committed(event) is False
    assert timer.isActive()
    assert cleanup.calls == 1

    workflow_reference = weakref.ref(workflow)
    controller_reference = weakref.ref(controller)
    cleanup_reference = weakref.ref(cleanup)
    diagnostic_reference = weakref.ref(diagnostic)
    timer_reference = weakref.ref(timer)
    registry = bus._retained_cleanup_lifecycle_registry
    assert registry.active_count == 1
    controller.disconnect()
    controller.disconnect()
    assert registry.active_count == 0
    assert not sip.isdeleted(timer)
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert workflow.active_label_command_id == "label-1"
    assert controller.pending_retained_cleanup_identity is None
    assert controller._native_retained_cleanup_lifecycle.model is None
    diagnostic_record = (
        controller._native_retained_cleanup_lifecycle.last_diagnostic
    )
    assert diagnostic_record["event_kind"] == "retained_recording_cleanup_retry"
    assert diagnostic_record["reason"] == "workflow-disconnect"

    del workflow
    del controller
    del cleanup
    del diagnostic
    gc.collect()

    assert not sip.isdeleted(bus)
    assert controller_reference() is None
    assert workflow_reference() is None
    assert cleanup_reference() is None
    assert diagnostic_reference() is None
    assert not sip.isdeleted(timer)
    _wait_for_qt(0)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(timer)
    del timer
    gc.collect()
    assert timer_reference() is None
    _wait_for_qt(120)


def test_retained_cleanup_registry_rejects_collision_and_double_root():
    bus = SequenceEventBus()
    registry = bus._retained_cleanup_lifecycle_registry
    token = object()
    first = object()
    replacement = object()

    assert registry.register(
        token, first
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert registry.register(
        token, first
    ) is RetainedCleanupLifecycleRegistrationResult.IDEMPOTENT
    assert registry.register(
        token, replacement
    ) is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
    assert registry.resolve(token) is first
    assert registry.retire(token, replacement) is False
    assert registry.active_count == 1
    assert registry.retire(token, first) is True
    assert registry.active_count == 0

    lifecycle = workflow_controller_module._NativeRetainedCleanupLifecycle(
        SequenceWorkflowModel(),
        lambda _context: None,
    )
    assert lifecycle.register_native_finalization_root(
        bus, object()
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert lifecycle.register_native_finalization_root(
        bus, object()
    ) is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
    assert registry.active_count == 1
    assert lifecycle._retire_native_finalization_root() is True
    assert registry.active_count == 0


def test_retained_cleanup_idempotence_revalidates_exact_registry_owner():
    bus = SequenceEventBus()
    registry = bus._retained_cleanup_lifecycle_registry
    token = object()
    lifecycle = workflow_controller_module._NativeRetainedCleanupLifecycle(
        SequenceWorkflowModel(),
        lambda _context: None,
    )
    replacement = object()

    assert lifecycle.register_native_finalization_root(
        bus, token
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert registry.retire(token, lifecycle) is True
    assert registry.register(
        token, replacement
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED

    assert lifecycle.register_native_finalization_root(
        bus, token
    ) is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
    assert registry.resolve(token) is replacement
    assert lifecycle._retire_native_finalization_root() is False
    assert registry.resolve(token) is replacement
    assert registry.retire(token, replacement) is True


def test_live_cleanup_token_collision_fails_closed_without_retiring_other_root():
    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda *_args, **_identity: False,
        connect_bus=True,
    )
    bus = controller.bus
    token = controller._retained_cleanup_registry_token
    other_lifecycle = object()
    assert bus._register_retained_cleanup_lifecycle(
        token, other_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED

    assert controller.handle_label_committed(event) is False

    assert controller.pending_retained_cleanup_identity is None
    assert controller.retained_cleanup_retry_attempt == 0
    assert controller.retained_cleanup_retry_delay_ms == 0
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert workflow.active_label_command_id is None
    assert states == []
    assert "token collision" in (
        controller.retained_cleanup_last_diagnostic["reason"]
    )
    assert bus._resolve_retained_cleanup_lifecycle(token) is other_lifecycle
    assert bus._retained_cleanup_lifecycle_count() == 1

    controller.disconnect()
    controller.disconnect()
    assert bus._resolve_retained_cleanup_lifecycle(token) is other_lifecycle
    assert bus._retire_retained_cleanup_lifecycle(token, other_lifecycle)


def test_stale_retired_timer_callback_cannot_resolve_replacement_capsule():
    _wait_for_qt(0)
    bus = SequenceEventBus()
    old_model = SequenceWorkflowModel()
    old_model.phase = WorkflowPhase.LABEL_COMMITTING
    old_model.workflow_generation = 4
    old_model.retained_record_id = "old-record"
    old_model.awaiting_label = True
    old_model.active_label_command_id = "old-label"
    old_model.active_label_record_id = "old-record"
    old_model.active_label = "OK"
    old_controller = SequenceWorkflowController(
        old_model,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_identity: False,
        connect_bus=False,
    )
    old_controller._retained_cleanup_retry_base_delay_ms = 1_000
    old_controller._retained_cleanup_retry_max_delay_ms = 1_000
    assert old_controller.handle_label_committed(
        RecordingLabelCommitted("old-label", "old-record", "OK", {})
    ) is False
    old_timer = old_controller._retained_cleanup_retry_timer
    old_callback = (
        old_controller._native_retained_cleanup_lifecycle
        .retry_timer_retirement_bridge._timeout_callback
    )
    registry = bus._retained_cleanup_lifecycle_registry
    assert registry.active_count == 1
    old_controller.disconnect()
    assert registry.active_count == 0

    replacement_model = SequenceWorkflowModel()
    replacement_model.phase = WorkflowPhase.LABEL_COMMITTING
    replacement_model.workflow_generation = 9
    replacement_model.retained_record_id = "replacement-record"
    replacement_model.awaiting_label = True
    replacement_model.active_label_command_id = "replacement-label"
    replacement_model.active_label_record_id = "replacement-record"
    replacement_model.active_label = "NG"
    replacement = SequenceWorkflowController(
        replacement_model,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_identity: False,
        connect_bus=False,
    )
    replacement._retained_cleanup_retry_base_delay_ms = 1_000
    replacement._retained_cleanup_retry_max_delay_ms = 1_000
    assert replacement.handle_label_committed(
        RecordingLabelCommitted(
            "replacement-label",
            "replacement-record",
            "NG",
            {},
        )
    ) is False
    replacement_timer = replacement._retained_cleanup_retry_timer
    before = _workflow_logical_snapshot(replacement_model)
    assert registry.active_count == 1

    old_callback()

    assert _workflow_logical_snapshot(replacement_model) == before
    assert registry.active_count == 1
    replacement.disconnect()
    assert registry.active_count == 0
    _drain_deferred_timer_deletion(old_timer)
    _drain_deferred_timer_deletion(replacement_timer)


@pytest.mark.parametrize("owner_state", ("live", "native-deleted", "gc"))
def test_stale_retry_callback_never_dispatches_to_same_token_replacement(
    owner_state,
):
    _wait_for_qt(0)
    bus = SequenceEventBus()
    old_model = SequenceWorkflowModel()
    old_model.phase = WorkflowPhase.LABEL_COMMITTING
    old_model.workflow_generation = 4
    old_model.retained_record_id = "old-record"
    old_model.awaiting_label = True
    old_model.active_label_command_id = "old-label"
    old_model.active_label_record_id = "old-record"
    old_model.active_label = "OK"
    old_calls = []
    old_controller = SequenceWorkflowController(
        old_model,
        bus,
        clear_retained_recording_snapshot=(
            lambda *_args, **_identity: old_calls.append("old") or False
        ),
        connect_bus=False,
    )
    old_controller._retained_cleanup_retry_base_delay_ms = 10_000
    old_controller._retained_cleanup_retry_max_delay_ms = 10_000
    old_token = old_controller._retained_cleanup_registry_token
    old_lifecycle = old_controller._native_retained_cleanup_lifecycle
    old_lifecycle_reference = weakref.ref(old_lifecycle)
    old_timer = old_controller._retained_cleanup_retry_timer
    old_callback = old_lifecycle.retry_timer_retirement_bridge._timeout_callback
    assert old_controller.handle_label_committed(
        RecordingLabelCommitted("old-label", "old-record", "OK", {})
    ) is False
    old_before = _workflow_logical_snapshot(old_model)
    assert old_calls == ["old"]
    assert old_lifecycle._retire_native_finalization_root() is True

    replacement_model = SequenceWorkflowModel()
    replacement_model.phase = WorkflowPhase.LABEL_COMMITTING
    replacement_model.workflow_generation = 9
    replacement_model.retained_record_id = "replacement-record"
    replacement_model.awaiting_label = True
    replacement_model.active_label_command_id = "replacement-label"
    replacement_model.active_label_record_id = "replacement-record"
    replacement_model.active_label = "NG"
    replacement_calls = []
    replacement = SequenceWorkflowController(
        replacement_model,
        bus,
        clear_retained_recording_snapshot=(
            lambda *_args, **_identity: replacement_calls.append("replacement")
            or False
        ),
        connect_bus=False,
    )
    replacement._retained_cleanup_registry_token = old_token
    replacement._retained_cleanup_retry_base_delay_ms = 10_000
    replacement._retained_cleanup_retry_max_delay_ms = 10_000
    replacement_lifecycle = replacement._native_retained_cleanup_lifecycle
    replacement_timer = replacement._retained_cleanup_retry_timer
    assert replacement.handle_label_committed(
        RecordingLabelCommitted(
            "replacement-label", "replacement-record", "NG", {}
        )
    ) is False
    replacement_before = _workflow_logical_snapshot(replacement_model)
    replacement_pending = replacement.pending_retained_cleanup_identity
    assert bus._resolve_retained_cleanup_lifecycle(
        old_token
    ) is replacement_lifecycle

    if owner_state != "live":
        sip.delete(old_controller)
        assert sip.isdeleted(old_controller)
    if owner_state == "gc":
        del old_lifecycle
        del old_controller
        gc.collect()
        assert old_lifecycle_reference() is None
    else:
        assert old_lifecycle_reference() is old_lifecycle

    old_callback()
    old_callback()

    assert _workflow_logical_snapshot(replacement_model) == replacement_before
    assert replacement.pending_retained_cleanup_identity == replacement_pending
    assert replacement_calls == ["replacement"]
    assert bus._resolve_retained_cleanup_lifecycle(
        old_token
    ) is replacement_lifecycle
    assert bus._retained_cleanup_lifecycle_count() == 1
    if owner_state == "live":
        assert _workflow_logical_snapshot(old_model) == old_before
        assert old_calls == ["old"]

    if owner_state != "gc":
        old_controller.disconnect()
        if not sip.isdeleted(old_timer):
            _drain_deferred_timer_deletion(old_timer)
    elif not sip.isdeleted(old_timer):
        sip.delete(old_timer)
    replacement.disconnect()
    assert bus._retained_cleanup_lifecycle_count() == 0
    _drain_deferred_timer_deletion(replacement_timer)


def test_live_disconnect_stop_failure_never_runs_native_domain_finalization(
    monkeypatch,
):
    workflow, controller, event, _states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    assert controller.handle_label_committed(event) is False
    before = _workflow_logical_snapshot(workflow)
    bridge = controller._native_retained_cleanup_lifecycle.retry_timer_retirement_bridge
    monkeypatch.setattr(bridge, "request_retirement", lambda: False)

    controller.disconnect()

    assert _workflow_logical_snapshot(workflow) == before
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True


def test_live_disconnect_inside_timeout_defers_native_timer_deletion():
    workflow, controller, _event, _states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    timer = controller._retained_cleanup_retry_timer
    observed_native_state = []

    def disconnect_inside_timeout():
        controller.disconnect()
        observed_native_state.append(sip.isdeleted(timer))

    timer.timeout.connect(disconnect_inside_timeout)
    timer.start(0)
    _wait_for_qt(10)

    assert observed_native_state == [False]
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(timer)
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING


def test_worker_disconnect_marshals_real_timer_retirement_to_owner_thread(
    monkeypatch,
):
    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    assert controller.handle_label_committed(event) is False
    before = _workflow_logical_snapshot(workflow)
    timer = controller._retained_cleanup_retry_timer
    late_timeouts = []
    timer.timeout.connect(lambda: late_timeouts.append("late-timeout"))
    main_thread_identity = get_ident()
    retirement_threads = []
    original_retirement = (
        workflow_controller_module._retire_timer_on_owner_thread
    )

    def observe_retirement(*args):
        retirement_threads.append(get_ident())
        return original_retirement(*args)

    monkeypatch.setattr(
        workflow_controller_module,
        "_retire_timer_on_owner_thread",
        observe_retirement,
    )
    worker_errors = []
    worker_identities = []
    qt_messages = []
    previous_handler = qInstallMessageHandler(
        lambda _kind, _context, message: qt_messages.append(message)
    )
    try:
        def disconnect_from_worker():
            worker_identities.append(get_ident())
            try:
                controller.disconnect()
            except BaseException as error:
                worker_errors.append(error)

        worker = Thread(target=disconnect_from_worker)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert worker_errors == []
        assert retirement_threads == []
        assert timer.isActive()

        _wait_for_qt(0)
        assert retirement_threads == [main_thread_identity]
        assert retirement_threads[0] != worker_identities[0]
        assert timer.isActive() is False
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(timer)
        _wait_for_qt(120)
        assert late_timeouts == []
    finally:
        qInstallMessageHandler(previous_handler)

    assert _workflow_logical_snapshot(workflow) == before
    assert states == []
    assert not any(
        "timer" in message.lower() or "setparent" in message.lower()
        for message in qt_messages
    )


def test_event_bus_native_delete_race_keeps_live_disconnect_state_neutral():
    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    assert controller.handle_label_committed(event) is False
    before = _workflow_logical_snapshot(workflow)
    bus = controller.bus
    timer = controller._retained_cleanup_retry_timer

    sip.delete(bus)
    assert sip.isdeleted(bus)
    assert sip.isdeleted(timer)
    controller.disconnect()
    controller.disconnect()

    assert _workflow_logical_snapshot(workflow) == before
    assert states == []
    assert controller.pending_retained_cleanup_identity is None


@pytest.mark.parametrize(
    "outcome",
    (
        False,
        RuntimeError("ordinary"),
        KeyboardInterrupt("interrupt"),
        SystemExit("exit"),
    ),
    ids=("false", "runtime-error", "keyboard-interrupt", "system-exit"),
)
def test_event_bus_native_delete_inside_cleanup_is_bounded_and_collectable(
    outcome,
):
    holder = {}
    calls = []

    class Diagnostic:
        def __call__(self, _context):
            return None

    def clear(record_id, **identity):
        calls.append((record_id, identity))
        sip.delete(holder["bus"])
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    workflow, controller, event, states = _label_committing_workflow(
        clear=clear,
        connect_bus=True,
    )
    bus = controller.bus
    timer = controller._retained_cleanup_retry_timer
    diagnostic = Diagnostic()
    controller.diagnostic_callback = diagnostic
    holder["bus"] = bus
    controller_reference = weakref.ref(controller)
    bus_reference = weakref.ref(bus)
    workflow_reference = weakref.ref(workflow)
    diagnostic_reference = weakref.ref(diagnostic)
    qt_messages = []
    previous_handler = qInstallMessageHandler(
        lambda _kind, _context, message: qt_messages.append(message)
    )
    try:
        assert controller.handle_label_committed(event) is False
        controller.disconnect()
        controller.disconnect()
    finally:
        qInstallMessageHandler(previous_handler)

    assert calls == [("record-1", {"workflow_generation": 4})]
    assert sip.isdeleted(bus)
    assert sip.isdeleted(timer)
    assert controller.pending_retained_cleanup_identity is None
    assert controller.retained_cleanup_retry_attempt == 0
    assert controller.retained_cleanup_retry_delay_ms == 0
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert workflow.active_label_command_id is None
    assert states == []
    assert "event bus" in controller.retained_cleanup_last_diagnostic["reason"]
    assert not any(
        "timer" in message.lower() or "deleted" in message.lower()
        for message in qt_messages
    )

    if isinstance(outcome, BaseException):
        outcome.__traceback__ = None
    holder.clear()
    del diagnostic
    del controller
    del bus
    del workflow
    gc.collect()
    assert controller_reference() is None
    assert bus_reference() is None
    assert workflow_reference() is None
    assert diagnostic_reference() is None


def test_event_bus_native_delete_inside_cleanup_subprocess_has_no_native_failure():
    script = r'''
import gc
import os
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import sip
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import RecordingLabelCommitted
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase

app = QApplication.instance() or QApplication([])
outcomes = (False, RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit"))
for index, outcome in enumerate(outcomes):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.LABEL_COMMITTING
    model.workflow_generation = index + 1
    model.retained_record_id = "record"
    model.awaiting_label = True
    model.active_label_command_id = "label"
    model.active_label_record_id = "record"
    model.active_label = "OK"
    calls = []

    def clear(*_args, _bus=bus, _outcome=outcome, **_kwargs):
        calls.append("clear")
        sip.delete(_bus)
        if isinstance(_outcome, BaseException):
            raise _outcome
        return _outcome

    controller = SequenceWorkflowController(
        model,
        bus,
        clear_retained_recording_snapshot=clear,
        connect_bus=True,
    )
    timer = controller._retained_cleanup_retry_timer
    controller_ref = weakref.ref(controller)
    bus_ref = weakref.ref(bus)
    model_ref = weakref.ref(model)
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label", "record", "OK", {})
    ) is False
    controller.disconnect()
    controller.disconnect()
    assert calls == ["clear"]
    assert sip.isdeleted(bus)
    assert sip.isdeleted(timer)
    assert controller.pending_retained_cleanup_identity is None
    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id == "record"
    if isinstance(outcome, BaseException):
        outcome.__traceback__ = None
    del clear
    del timer
    del controller
    del bus
    del model
    gc.collect()
    assert controller_ref() is None
    assert bus_ref() is None
    assert model_ref() is None
print("bus-delete-cleanup-race-ok")
'''
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bus-delete-cleanup-race-ok"
    stderr = result.stderr.lower()
    assert "access violation" not in stderr
    assert "wrapped c/c++ object" not in stderr
    assert "qobject::" not in stderr


def test_event_bus_gc_releases_registry_capsule_without_python_cycle():
    _wait_for_qt(0)

    class Diagnostic:
        def __call__(self, _context):
            return None

    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.phase = WorkflowPhase.LABEL_COMMITTING
    workflow.workflow_generation = 4
    workflow.retained_record_id = "record-1"
    workflow.awaiting_label = True
    workflow.active_label_command_id = "label-1"
    workflow.active_label_record_id = "record-1"
    workflow.active_label = "OK"
    diagnostic = Diagnostic()
    controller = SequenceWorkflowController(
        workflow,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_identity: False,
        diagnostic_callback=diagnostic,
        connect_bus=False,
    )
    controller._retained_cleanup_retry_base_delay_ms = 1_000
    controller._retained_cleanup_retry_max_delay_ms = 1_000
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label-1", "record-1", "OK", {})
    ) is False
    timer = controller._retained_cleanup_retry_timer
    registry = bus._retained_cleanup_lifecycle_registry
    bus_reference = weakref.ref(bus)
    controller_reference = weakref.ref(controller)
    capsule_reference = weakref.ref(
        controller._native_retained_cleanup_lifecycle
    )
    diagnostic_reference = weakref.ref(diagnostic)
    timer_reference = weakref.ref(timer)
    assert registry.active_count == 1

    del timer
    del diagnostic
    del controller
    del bus
    del registry
    gc.collect()

    assert controller_reference() is None
    assert bus_reference() is None
    assert capsule_reference() is None
    assert diagnostic_reference() is None
    assert timer_reference() is None


def test_registry_roots_only_an_active_scheduled_native_cleanup():
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        SequenceWorkflowModel(),
        bus,
        connect_bus=False,
    )
    registry = bus._retained_cleanup_lifecycle_registry

    assert registry.active_count == 0
    controller.disconnect()


def test_thread_affine_retirement_stress_has_no_qt_warning_or_native_crash():
    script = r'''
import gc
import os
from threading import Thread
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import RecordingLabelCommitted
import ui.sequence.sequence_workflow_controller as workflow_controller_module
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase

app = QApplication.instance() or QApplication([])
for iteration in range(100):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.LABEL_COMMITTING
    model.workflow_generation = iteration + 1
    model.retained_record_id = "record"
    model.awaiting_label = True
    model.active_label_command_id = "label"
    model.active_label_record_id = "record"
    model.active_label = "OK"
    controller = SequenceWorkflowController(
        model,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_kwargs: False,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = 1_000
    controller._retained_cleanup_retry_max_delay_ms = 1_000
    timer = controller._retained_cleanup_retry_timer
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label", "record", "OK", {})
    ) is False
    registry = bus._retained_cleanup_lifecycle_registry
    assert registry.active_count == 1
    worker = Thread(target=controller.disconnect)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert registry.active_count == 0
    QCoreApplication.sendPostedEvents(None, 0)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(timer)

native_case_count = 500
native_cases = []
timer_iterations = {}
retirement_iterations = set()
destroyed_iterations = set()
loop = QEventLoop()
original_retirement = workflow_controller_module._retire_timer_on_owner_thread

def observe_retirement(timer, timeout_callback):
    retirement_iterations.add(timer_iterations[id(timer)])
    return original_retirement(timer, timeout_callback)

def observe_destroyed(iteration):
    destroyed_iterations.add(iteration)
    if len(destroyed_iterations) == native_case_count:
        loop.quit()

workflow_controller_module._retire_timer_on_owner_thread = observe_retirement
for iteration in range(native_case_count):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.LABEL_COMMITTING
    model.workflow_generation = iteration + 201
    model.retained_record_id = "native-record"
    model.awaiting_label = True
    model.active_label_command_id = "native-label"
    model.active_label_record_id = "native-record"
    model.active_label = "OK"
    controller = SequenceWorkflowController(
        model,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_kwargs: False,
        connect_bus=False,
    )
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1
    timer = controller._retained_cleanup_retry_timer
    timer_iterations[id(timer)] = iteration
    timer.destroyed.connect(
        lambda _object=None, iteration=iteration: observe_destroyed(iteration)
    )
    capsule_ref = weakref.ref(controller._native_retained_cleanup_lifecycle)
    controller_ref = weakref.ref(controller)
    assert controller.handle_label_committed(
        RecordingLabelCommitted("native-label", "native-record", "OK", {})
    ) is False
    registry = bus._retained_cleanup_lifecycle_registry
    sip.delete(controller)
    del controller
    gc.collect()
    assert controller_ref() is None
    assert capsule_ref() is not None
    assert registry.active_count == 1
    native_cases.append((bus, model, timer, capsule_ref, registry))

deadline = QTimer()
deadline.setSingleShot(True)
deadline.timeout.connect(loop.quit)
deadline.start(10_000)
loop.exec()
completed_before_deadline = len(destroyed_iterations) == native_case_count
deadline.stop()
QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
gc.collect()
assert completed_before_deadline
assert retirement_iterations == set(range(native_case_count))
assert destroyed_iterations == set(range(native_case_count))
for bus, model, timer, capsule_ref, registry in native_cases:
    assert model.phase is WorkflowPhase.IDLE
    assert capsule_ref() is None
    assert registry.active_count == 0
    assert sip.isdeleted(timer)
    assert not sip.isdeleted(bus)
print("retirement-stress-ok")
'''
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "retirement-stress-ok"
    stderr = result.stderr.lower()
    assert "access violation" not in stderr
    assert "wrapped c/c++ object" not in stderr
    assert "timers cannot be stopped" not in stderr
    assert "qobject::killtimer" not in stderr
    assert "qobject::setparent" not in stderr


@pytest.mark.parametrize(
    "operation",
    ("stop", "disconnect", "unparent", "delete"),
)
@pytest.mark.parametrize(
    "outcome",
    (
        False,
        RuntimeError("ordinary"),
        KeyboardInterrupt("interrupt"),
        SystemExit("exit"),
    ),
    ids=("false", "runtime-error", "keyboard-interrupt", "system-exit"),
)
def test_live_disconnect_timer_retirement_failures_are_bounded_and_state_neutral(
    monkeypatch,
    operation,
    outcome,
):
    calls = []

    def invoke(name):
        calls.append(name)
        if name != operation:
            return None
        if outcome is False:
            return False
        raise outcome

    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    assert controller.handle_label_committed(event) is False
    before = _workflow_logical_snapshot(workflow)
    original_timer = controller._retained_cleanup_retry_timer
    controller._native_retained_cleanup_lifecycle._retire_native_finalization_root()
    sip.delete(original_timer)
    timer = _InjectedRetirementTimer(invoke)
    timer_reference = weakref.ref(timer)
    controller._retained_cleanup_retry_timer = timer
    controller._native_retained_cleanup_lifecycle.retry_timer_ref = lambda: timer
    bridge = _InjectedRetirementBridge(
        timer,
        controller._native_retained_cleanup_lifecycle,
    )
    controller._native_retained_cleanup_lifecycle.retry_timer_retirement_bridge = bridge
    original_deleted_check = workflow_controller_module._qt_object_is_deleted
    monkeypatch.setattr(
        workflow_controller_module,
        "_qt_object_is_deleted",
        lambda value: False if value is bridge else original_deleted_check(value),
    )

    controller.disconnect()
    controller.disconnect()

    assert _workflow_logical_snapshot(workflow) == before
    assert states == []
    assert calls == ["stop", "disconnect", "unparent", "delete"]
    diagnostic = controller.retained_cleanup_last_diagnostic
    assert diagnostic["event_kind"] == "retained_cleanup_timer_retirement"
    assert len(diagnostic["failure_operations"]) == 1
    assert diagnostic["failure_operations"][0].startswith(operation)
    if isinstance(outcome, BaseException):
        outcome.__traceback__ = None
    del bridge
    del timer
    del controller
    gc.collect()
    assert timer_reference() is None


@pytest.mark.parametrize(
    "operation",
    ("stop", "disconnect", "unparent", "delete"),
)
def test_live_disconnect_timer_retirement_is_reentrant_and_idempotent(
    monkeypatch,
    operation,
):
    calls = []
    holder = {}
    reentered = False

    def invoke(name):
        nonlocal reentered
        calls.append(name)
        if name == operation and not reentered:
            reentered = True
            holder["controller"].disconnect()
        return None

    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: False,
        connect_bus=True,
    )
    holder["controller"] = controller
    controller._retained_cleanup_retry_base_delay_ms = 100
    controller._retained_cleanup_retry_max_delay_ms = 100
    assert controller.handle_label_committed(event) is False
    before = _workflow_logical_snapshot(workflow)
    original_timer = controller._retained_cleanup_retry_timer
    controller._native_retained_cleanup_lifecycle._retire_native_finalization_root()
    sip.delete(original_timer)
    timer = _InjectedRetirementTimer(invoke)
    controller._retained_cleanup_retry_timer = timer
    controller._native_retained_cleanup_lifecycle.retry_timer_ref = lambda: timer
    bridge = _InjectedRetirementBridge(
        timer,
        controller._native_retained_cleanup_lifecycle,
    )
    controller._native_retained_cleanup_lifecycle.retry_timer_retirement_bridge = bridge
    original_deleted_check = workflow_controller_module._qt_object_is_deleted
    monkeypatch.setattr(
        workflow_controller_module,
        "_qt_object_is_deleted",
        lambda value: False if value is bridge else original_deleted_check(value),
    )

    controller.disconnect()
    controller.disconnect()

    assert reentered is True
    assert calls == ["stop", "disconnect", "unparent", "delete"]
    assert _workflow_logical_snapshot(workflow) == before
    assert states == []


class _PendingCleanupShutdownView(QObject):
    confirm_shutdown_requested = pyqtSignal(object)
    abort_shutdown_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.confirmations = []
        self.finished_confirmations = []
        self.waiting = []

    def show_shutdown_confirmation(self, generation):
        self.confirmations.append(generation)
        return True

    def finish_shutdown_confirmation(self, generation):
        self.finished_confirmations.append(generation)
        return True

    def show_shutdown_waiting(self, generation):
        self.waiting.append(generation)
        return True

    def finish_shutdown_waiting(self, _generation):
        return True


def _shutdown_with_pending_cleanup(*, clear, retry_delay_ms):
    workflow, controller, event, states = _label_committing_workflow(
        clear=clear,
        connect_bus=True,
    )
    controller._retained_cleanup_retry_base_delay_ms = retry_delay_ms
    controller._retained_cleanup_retry_max_delay_ms = retry_delay_ms
    view = _PendingCleanupShutdownView()
    coordinator = SequenceShutdownCoordinator(
        workflow,
        controller.bus,
        view=view,
        shutdown_ready=controller.handle_shutdown_ready,
    )
    controller.bus.register_workflow_continuation_recipient(
        "workflow-state",
        "pending-cleanup-shutdown-state",
        coordinator.handle_workflow_state_changed,
        owner=coordinator,
    )
    ready = []

    def main_ready(message):
        ready.append(message)
        return True

    controller.bus.register_workflow_continuation_recipient(
        "shutdown-ready",
        "pending-cleanup-main",
        main_ready,
    )
    flush_owner = QObject()
    controller.bus.register_workflow_continuation_lifecycle_owner(flush_owner)

    def complete_flush(command):
        return controller.bus.deliver_workflow_continuation(
            ("shutdown-flush-completed", command.shutdown_generation),
            "shutdown-flush-completed",
            ShutdownFlushCompleted(command.shutdown_generation),
            owner=flush_owner,
        )

    controller.bus.commands.begin_shutdown_flush_requested.connect(complete_flush)
    cancellations = []
    controller.bus.commands.cancel_recording_requested.connect(cancellations.append)
    controller.bus.events.recording_label_committed.emit(event)
    _wait_for_qt(5)
    return (
        workflow,
        controller,
        coordinator,
        view,
        event,
        states,
        ready,
        cancellations,
        flush_owner,
    )


def test_abortable_shutdown_preserves_pending_cleanup_and_abort_reaches_idle():
    calls = []
    outcomes = iter((False, True))

    def clear(record_id, **identity):
        calls.append((record_id, identity))
        return next(outcomes)

    (
        workflow,
        controller,
        coordinator,
        view,
        _event,
        _states,
        ready,
        cancellations,
        _flush_owner,
    ) = _shutdown_with_pending_cleanup(clear=clear, retry_delay_ms=50)
    timer = controller._retained_cleanup_retry_timer
    pending_identity = controller.pending_retained_cleanup_identity

    assert coordinator.request_shutdown(30, True)
    _wait_for_qt(5)
    assert view.confirmations == [30]
    assert controller.pending_retained_cleanup_identity == pending_identity
    assert timer.isActive()

    view.abort_shutdown_requested.emit(AbortShutdownRequested(30))
    _wait_for_qt(70)

    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.shutdown_generation is None
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert controller.pending_retained_cleanup_identity is None
    assert timer.isActive() is False
    assert len(calls) == 2
    assert ready == []
    assert cancellations == []


def test_confirmed_shutdown_waits_for_cleanup_ack_then_reaches_ready():
    calls = []
    outcomes = iter(
        (
            False,
            RuntimeError("ordinary"),
            KeyboardInterrupt("interrupt"),
            SystemExit("exit"),
            True,
        )
    )

    def clear(record_id, **identity):
        calls.append((record_id, identity))
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    (
        workflow,
        controller,
        coordinator,
        view,
        _event,
        _states,
        ready,
        cancellations,
        _flush_owner,
    ) = _shutdown_with_pending_cleanup(clear=clear, retry_delay_ms=10)

    assert coordinator.request_shutdown(31, True)
    _wait_for_qt(5)
    view.confirm_shutdown_requested.emit(
        ConfirmShutdownCancellationRequested(31)
    )
    _wait_for_qt(80)

    assert len(calls) == 5
    assert workflow.phase is WorkflowPhase.SHUTDOWN_READY
    assert workflow.shutdown_generation == 31
    assert controller.pending_retained_cleanup_identity is None
    assert controller._retained_cleanup_retry_timer.isActive() is False
    assert view.finished_confirmations == [31]
    assert view.waiting == [31]
    assert len(ready) == 1
    assert ready[0].shutdown_generation == 31
    assert cancellations == []


def test_confirmed_shutdown_permanent_cleanup_failure_never_reports_ready():
    calls = []
    diagnostics = []

    (
        workflow,
        controller,
        coordinator,
        view,
        _event,
        _states,
        ready,
        cancellations,
        _flush_owner,
    ) = _shutdown_with_pending_cleanup(
        clear=lambda record_id, **identity: calls.append((record_id, identity))
        or False,
        retry_delay_ms=10,
    )
    controller.diagnostic_callback = diagnostics.append
    timer = controller._retained_cleanup_retry_timer
    timer_identity = id(timer)

    assert coordinator.request_shutdown(32, True)
    _wait_for_qt(5)
    pending_before_stale = controller.pending_retained_cleanup_identity
    view.confirm_shutdown_requested.emit(
        ConfirmShutdownCancellationRequested(31)
    )
    view.abort_shutdown_requested.emit(AbortShutdownRequested(31))
    _wait_for_qt(5)
    assert workflow.shutdown_generation == 32
    assert workflow.shutdown_cancellation_confirmed is False
    assert controller.pending_retained_cleanup_identity == pending_before_stale
    assert timer.isActive()

    view.confirm_shutdown_requested.emit(
        ConfirmShutdownCancellationRequested(32)
    )
    _wait_for_qt(40)

    assert len(calls) >= 3
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.shutdown_generation == 32
    assert workflow.shutdown_cancellation_confirmed is True
    assert controller.pending_retained_cleanup_identity is not None
    assert id(controller._retained_cleanup_retry_timer) == timer_identity
    assert timer.isActive()
    assert ready == []
    assert cancellations == []
    assert diagnostics

    before = controller.pending_retained_cleanup_identity
    assert controller.handle_shutdown(ShutdownRequested(33, True)) is False
    assert controller.pending_retained_cleanup_identity == before
    assert timer.isActive()
    controller.disconnect()


def test_native_delete_during_cleanup_abandons_without_touching_dead_timer():
    class Diagnostic:
        def __call__(self, _context):
            return None

    holder = {}
    calls = []

    def clear(record_id, **identity):
        calls.append((record_id, identity))
        sip.delete(holder["controller"])
        return False

    workflow, controller, event, _states = _label_committing_workflow(
        clear=clear,
        connect_bus=True,
    )
    holder["controller"] = controller
    bus = controller.bus
    timer = controller._retained_cleanup_retry_timer
    reference = weakref.ref(controller)
    diagnostic = Diagnostic()
    diagnostic_reference = weakref.ref(diagnostic)
    controller.diagnostic_callback = diagnostic

    assert controller.handle_label_committed(event) is False
    assert sip.isdeleted(controller)
    assert controller.pending_retained_cleanup_identity is None
    assert not sip.isdeleted(timer)
    _drain_deferred_timer_deletion(timer)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.active_label_command_id is None
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert controller.retained_cleanup_retry_attempt == 0
    assert controller.retained_cleanup_retry_delay_ms == 0
    assert "native" in controller.retained_cleanup_last_diagnostic["reason"]

    controller.disconnect()
    controller.disconnect()
    holder.clear()
    del diagnostic
    del controller
    del timer
    gc.collect()
    assert reference() is None
    assert diagnostic_reference() is None
    assert not sip.isdeleted(bus)


def test_native_delete_with_scheduled_cleanup_then_disconnect_is_idempotent():
    calls = []
    workflow, controller, event, _states = _label_committing_workflow(
        clear=lambda record_id, **identity: calls.append((record_id, identity))
        or False
    )
    controller._retained_cleanup_retry_base_delay_ms = 50
    controller._retained_cleanup_retry_max_delay_ms = 50
    timer = controller._retained_cleanup_retry_timer

    assert controller.handle_label_committed(event) is False
    assert timer.isActive()
    sip.delete(controller)

    assert sip.isdeleted(controller)
    assert controller.pending_retained_cleanup_identity is None
    assert not sip.isdeleted(timer)
    _drain_deferred_timer_deletion(timer)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    controller.disconnect()
    controller.disconnect()
    _wait_for_qt(70)
    assert len(calls) == 1


def test_native_deleted_wrapper_gc_keeps_capsule_until_timer_finalization():
    _wait_for_qt(0)

    class Diagnostic:
        def __call__(self, _context):
            return None

    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.phase = WorkflowPhase.LABEL_COMMITTING
    workflow.workflow_generation = 4
    workflow.retained_record_id = "record-1"
    workflow.awaiting_label = True
    workflow.active_label_command_id = "label-1"
    workflow.active_label_record_id = "record-1"
    workflow.active_label = "OK"
    diagnostic = Diagnostic()
    controller = SequenceWorkflowController(
        workflow,
        bus,
        clear_retained_recording_snapshot=lambda *_args, **_identity: False,
        diagnostic_callback=diagnostic,
        connect_bus=False,
    )
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1
    timer = controller._retained_cleanup_retry_timer
    capsule_reference = weakref.ref(
        controller._native_retained_cleanup_lifecycle
    )
    controller_reference = weakref.ref(controller)
    diagnostic_reference = weakref.ref(diagnostic)
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label-1", "record-1", "OK", {})
    ) is False

    sip.delete(controller)
    del controller
    del diagnostic
    gc.collect()

    assert controller_reference() is None
    assert capsule_reference() is not None
    assert diagnostic_reference() is not None
    registry = bus._retained_cleanup_lifecycle_registry
    assert registry.active_count == 1
    assert timer.isActive()
    _wait_for_qt(20)
    assert sip.isdeleted(timer)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()

    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.active_label_command_id is None
    assert capsule_reference() is None
    assert diagnostic_reference() is None
    assert sip.isdeleted(timer)
    assert not sip.isdeleted(bus)


@pytest.mark.parametrize("decision", ["abort", "confirm"])
def test_native_delete_resolves_pending_cleanup_shutdown_without_false_ready(decision):
    calls = []
    (
        workflow,
        controller,
        coordinator,
        view,
        _event,
        _states,
        ready,
        cancellations,
        _flush_owner,
    ) = _shutdown_with_pending_cleanup(
        clear=lambda record_id, **identity: calls.append((record_id, identity))
        or False,
        retry_delay_ms=100,
    )
    timer = controller._retained_cleanup_retry_timer
    generation = 40 if decision == "abort" else 41

    assert coordinator.request_shutdown(generation, True)
    _wait_for_qt(5)
    if decision == "abort":
        view.abort_shutdown_requested.emit(
            AbortShutdownRequested(generation)
        )
    else:
        view.confirm_shutdown_requested.emit(
            ConfirmShutdownCancellationRequested(generation)
        )
    _wait_for_qt(5)
    sip.delete(controller)

    assert sip.isdeleted(controller)
    assert controller.pending_retained_cleanup_identity is None
    assert not sip.isdeleted(timer)
    _drain_deferred_timer_deletion(timer)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.shutdown_generation is None
    assert workflow.shutdown_pending is False
    assert workflow.shutdown_cancellation_confirmed is False
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert ready == []
    assert cancellations == []
    controller.disconnect()
    controller.disconnect()
    _wait_for_qt(120)
    assert len(calls) == 1
    coordinator.disconnect()


def test_permanent_cleanup_failure_remains_pending_with_bounded_backoff():
    _wait_for_qt(0)
    calls = []
    diagnostics = []
    workflow, controller, event, _states = _label_committing_workflow(
        clear=lambda record_id, **identity: calls.append((record_id, identity))
        or False,
        connect_bus=True,
    )
    controller.diagnostic_callback = diagnostics.append
    controller._retained_cleanup_retry_base_delay_ms = 10
    controller._retained_cleanup_retry_max_delay_ms = 40
    timer = controller._retained_cleanup_retry_timer
    observed_intervals = []
    retry_ready = QEventLoop()
    retry_deadline = QTimer()
    retry_deadline.setSingleShot(True)
    retry_deadline.timeout.connect(retry_ready.quit)

    def retry_condition_met():
        return (
            len(calls) >= 3
            and controller.retained_cleanup_retry_attempt >= 3
            and len(observed_intervals) >= 2
        )

    def observe_retry_interval():
        observed_intervals.append(timer.interval())
        if retry_condition_met():
            retry_ready.quit()

    timer.timeout.connect(observe_retry_interval)
    controller.bus.events.recording_label_committed.emit(event)
    retry_deadline.start(2_000)
    if not retry_condition_met():
        retry_ready.exec()
    retry_deadline.stop()

    assert len(calls) >= 3
    assert controller.pending_retained_cleanup_identity is not None
    assert controller.retained_cleanup_retry_attempt >= 3
    assert observed_intervals[:2] == [20, 40]
    assert 0 < controller.retained_cleanup_retry_delay_ms <= 40
    assert timer.isActive()
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert diagnostics
    controller.disconnect()


def test_retained_cleanup_retry_uses_one_single_shot_without_polling_loops():
    tree = ast.parse(WORKFLOW_CONTROLLER.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceWorkflowController"
    )
    retry_methods = [
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and "retained_cleanup" in node.name
    ]
    rendered = "\n".join(ast.unparse(node) for node in retry_methods)
    assert "processEvents" not in rendered
    assert not any(
        isinstance(node, ast.While)
        for method in retry_methods
        for node in ast.walk(method)
    )
    timer_assignments = [
        node
        for node in ast.walk(controller)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_retained_cleanup_retry_timer"
            for target in node.targets
        )
    ]
    assert len(timer_assignments) == 1
    timer_source = ast.unparse(timer_assignments[0])
    assert "QTimer(timer_parent)" in timer_source
    controller_source = ast.unparse(controller)
    assert "timer_parent = self.bus if isinstance(self.bus, QObject) else None" in (
        controller_source
    )
    assert "retry_owner_ref = ref(self)" in controller_source
    assert "_qt_object_is_deleted(owner)" in controller_source
    assert "self.destroyed.connect" not in controller_source


def test_label_clear_false_preserves_exact_workflow_identity_for_retry():
    outcomes = iter((False, True))
    calls = []

    def clear(record_id, *, workflow_generation):
        calls.append((record_id, workflow_generation))
        return next(outcomes)

    workflow, controller, event, states = _label_committing_workflow(clear=clear)
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1
    registry = controller.bus._retained_cleanup_lifecycle_registry

    assert controller.handle_label_committed(event) is False
    assert registry.active_count == 1
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert workflow.active_label_command_id == "label-1"
    assert states == []

    _wait_for_qt(20)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert calls == [("record-1", 4), ("record-1", 4)]
    assert [state.new_phase for state in states] == ["IDLE"]
    assert registry.active_count == 0


def test_label_clear_none_is_a_successful_no_return_contract():
    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: None
    )

    assert controller.handle_label_committed(event) is True
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert [state.new_phase for state in states] == ["IDLE"]


def test_label_clear_rejects_values_outside_the_bool_or_none_contract():
    workflow, controller, event, states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: 1
    )

    assert controller.handle_label_committed(event) is False
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert states == []
    controller.disconnect()


def test_label_clear_retry_publishes_post_analysis_continuation_exactly_once():
    outcomes = iter((False, True))
    workflow, controller, event, _states = _label_committing_workflow(
        clear=lambda _record_id, **_identity: next(outcomes)
    )
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1
    workflow.post_analysis_continuation = PostAnalysisContinuation(
        "analysis-1",
        "source-1",
        "record-1",
        4,
        {"tcp_result_payload": {"result": "OK"}},
        "OK",
    )
    transports = []
    controller.bus.register_workflow_continuation_recipient(
        "workflow-state",
        "recording-facade-retry-state",
        lambda _message: True,
        owner=controller,
    )
    controller.bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "recording-facade-retry-transport",
        lambda message: transports.append(message) or True,
        owner=controller,
    )

    assert controller.handle_label_committed(event) is False
    assert transports == []
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING

    _wait_for_qt(20)
    assert workflow.phase is WorkflowPhase.IDLE
    assert len(transports) == 1
    assert transports[0].record_id == "record-1"
    assert controller.handle_label_committed(event) is False
    assert len(transports) == 1


def test_label_clear_baseexception_is_automatically_retried_without_escape():
    class StopNow(BaseException):
        pass

    outcomes = iter((StopNow(), True))

    def clear(_record_id, **_identity):
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    workflow, controller, event, states = _label_committing_workflow(clear=clear)
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1

    assert controller.handle_label_committed(event) is False
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert workflow.active_label_command_id == "label-1"
    assert states == []
    _wait_for_qt(20)
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert [state.new_phase for state in states] == ["IDLE"]


def test_label_clear_reentry_is_rejected_and_terminal_emits_once():
    nested = []
    calls = []
    holder = {}
    owner = RecordingModel()
    assert owner.retain_recording_snapshot(
        "record-1",
        {"record_id": "record-1", "samples": [1.0]},
        _configuration(),
        source_id="session-1",
        workflow_generation=4,
    )

    def clear(record_id, *, workflow_generation):
        calls.append((record_id, workflow_generation))
        cleared = owner.clear_retained_recording_snapshot(
            record_id, workflow_generation=workflow_generation
        )
        if len(calls) == 1:
            nested.append(
                holder["controller"].handle_label_committed(holder["event"])
            )
        return cleared

    workflow, controller, event, states = _label_committing_workflow(clear=clear)
    holder.update(controller=controller, event=event)
    controller._retained_cleanup_retry_base_delay_ms = 1
    controller._retained_cleanup_retry_max_delay_ms = 1

    assert controller.handle_label_committed(event) is False
    assert nested == [False]
    assert calls == [("record-1", 4)]
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert owner.retained_recording_snapshot("record-1") is None
    assert states == []

    _wait_for_qt(20)
    assert calls == [("record-1", 4), ("record-1", 4)]
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.retained_record_id is None
    assert workflow.awaiting_label is False
    assert [state.new_phase for state in states] == ["IDLE"]


def test_stale_cleanup_and_retry_cannot_clear_concurrent_replacement():
    owner = RecordingModel()
    assert owner.retain_recording_snapshot(
        "record-1",
        {"record_id": "record-1", "samples": [1.0]},
        _configuration(),
        source_id="session-1",
        workflow_generation=4,
    )
    replaced = False

    def replace_then_clear(record_id, *, workflow_generation):
        nonlocal replaced
        if not replaced:
            replaced = True
            assert owner.retain_recording_snapshot(
                "record-1",
                {"record_id": "record-1", "samples": [2.0]},
                _configuration(),
                source_id="session-2",
                workflow_generation=5,
            )
        return owner.clear_retained_recording_snapshot(
            record_id, workflow_generation=workflow_generation
        )

    workflow, controller, event, states = _label_committing_workflow(
        clear=replace_then_clear
    )

    assert controller.handle_label_committed(event) is False
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    retained = owner.retained_recording_snapshot("record-1")
    assert retained.workflow_generation == 5
    assert retained.recording_snapshot["samples"] == (2.0,)
    assert states == []

    assert controller.handle_label_committed(event) is False
    retained = owner.retained_recording_snapshot("record-1")
    assert retained.workflow_generation == 5
    assert retained.recording_snapshot["samples"] == (2.0,)
    assert workflow.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.retained_record_id == "record-1"
    assert workflow.awaiting_label is True
    assert states == []
    controller.disconnect()


def _label_service(**overrides):
    ok_button = object()
    ng_button = object()
    published = []
    warnings = []
    values = {
        "data_provider": lambda: [1.0],
        "sequence_config_provider": lambda: _configuration().sequence_config,
        "retained_record_id_provider": lambda: "retained-record",
        "recorded_signal_info_provider": lambda: {"file_path": "info-record"},
        "recorded_path_provider": lambda: "path-record",
        "ok_button": ok_button,
        "ng_button": ng_button,
        "publish": published.append,
        "present_warning": lambda title, text: warnings.append((title, text)),
        "command_id_factory": lambda: "manual-label-fixed",
    }
    values.update(overrides)
    return RecordingManualLabelRequestService(**values), ok_button, ng_button, published, warnings


def test_manual_label_owner_selects_label_and_exact_fallback_before_publication():
    service, ok_button, ng_button, published, _warnings = _label_service()

    assert service.request(ok_button)
    assert service.request(ng_button)
    assert published == [
        ManualLabelRequested("manual-label-fixed", "retained-record", "OK"),
        ManualLabelRequested("manual-label-fixed", "retained-record", "NG"),
    ]

    info_published = []
    info_service, ok_button, _ng, _unused, _warnings = _label_service(
        retained_record_id_provider=lambda: None,
        publish=info_published.append,
    )
    assert info_service.request(ok_button)
    assert info_published[0].record_id == "info-record"

    path_published = []
    path_service, ok_button, _ng, _unused, _warnings = _label_service(
        retained_record_id_provider=lambda: None,
        recorded_signal_info_provider=lambda: object(),
        publish=path_published.append,
    )
    assert path_service.request(ok_button)
    assert path_published[0].record_id == "path-record"


@pytest.mark.parametrize("mode", ["IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"])
def test_manual_label_owner_rejects_import_modes_without_publication(mode):
    service, ok_button, _ng, published, warnings = _label_service(
        sequence_config_provider=lambda: _configuration(mode).sequence_config
    )

    assert not service.request(ok_button)
    assert published == []
    assert warnings and "导入" in warnings[0][1]


def test_manual_label_owner_handles_missing_or_hostile_inputs_and_reentry():
    class HostileData:
        def __len__(self):
            raise ValueError("bad data")

    service, ok_button, _ng, published, warnings = _label_service(
        data_provider=lambda: HostileData()
    )
    assert not service.request(ok_button)
    assert published == []
    assert warnings == [("警告", "请先录制声音！")]

    nested = []
    holder = {}

    def publish(command):
        nested.append(holder["service"].request(ok_button))
        nested.append(command)

    service, ok_button, _ng, _published, _warnings = _label_service(publish=publish)
    holder["service"] = service
    assert service.request(ok_button)
    assert nested[0] is False
    assert isinstance(nested[1], ManualLabelRequested)


def test_manual_label_owner_releases_gate_after_baseexception():
    class StopNow(BaseException):
        pass

    calls = []

    def publish(command):
        calls.append(command)
        if len(calls) == 1:
            raise StopNow()

    service, ok_button, _ng, _published, _warnings = _label_service(publish=publish)
    with pytest.raises(StopNow):
        service.request(ok_button)
    assert service.request(ok_button)
    assert len(calls) == 2


def test_facade_manual_label_handler_is_one_step_recording_delegate():
    source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    method = next(
        node
        for node in window.body
        if isinstance(node, ast.FunctionDef) and node.name == "clicked_ok_or_ng"
    )
    assert len(method.body) == 1
    rendered = ast.unparse(method)
    assert "recording_controller.request_manual_label" in rendered
    assert "ManualLabelRequested" not in rendered
    assert not any(
        isinstance(node, (ast.If, ast.Try, ast.For, ast.While))
        for node in ast.walk(method)
    )
