import ast
import logging
import os
from pathlib import Path
from pathlib import PurePosixPath
import textwrap
from threading import Event, Thread
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_event_bus import (
    ImportTerminalRecipientResult,
    SequenceEventBus,
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
)
from ui.sequence.sequence_messages import (
    AbortShutdownRequested,
    AnalysisCompleted,
    AnalysisExportPrepared,
    AnalysisFailed,
    AnalysisTransportReady,
    CancelWorkflowRequested,
    CommitRecordingLabelRequested,
    ConfirmShutdownCancellationRequested,
    ConfigurationSnapshot,
    ExportCompleted,
    ExportFailed,
    ExportRetryAccepted,
    IgnoreExportFailureRequested,
    ImportAudioRequested,
    ImportedAudioFailed,
    ImportedAudioReady,
    ManualAnalysisRequested,
    ManualLabelExportPrepared,
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
    ShutdownRequested,
    StartTestRequested,
)
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import (
    ExportContinuation,
    PostAnalysisContinuation,
    SessionOrigin,
    SequenceWorkflowModel,
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


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def configuration_snapshot():
    return ConfigurationSnapshot(
        sequence_config={"mode": "RECORD_ONLY"},
        analysis_config={"auto_analysis": False},
        mic={"name": "input"},
        speaker=None,
        mic_channels=(0,),
    )


def start_command(command_id="start-1"):
    return StartTestRequested(
        command_id=command_id,
        source="manual",
        label="SN-001",
        skip_sn_regex_validation=False,
        configuration_generation=3,
    )


def test_importing_cancel_uses_canonical_import_command(qapp):
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        model,
        bus,
        import_id_factory=lambda: "import-1",
        configuration_snapshot_provider=configuration_snapshot,
    )
    loads = []
    cancellations = []
    wrong_domain = []
    bus.commands.load_imported_audio_requested.connect(loads.append)
    bus.commands.cancel_imported_audio_requested.connect(cancellations.append)
    bus.commands.cancel_recording_requested.connect(wrong_domain.append)
    assert controller.handle_import(
        ImportAudioRequested("command-import", "IMPORT_AUDIO", "audio.wav")
    )
    generation = model.workflow_generation

    assert controller.handle_cancel_workflow(
        CancelWorkflowRequested("cancel-1", generation, "operator cancelled")
    )

    assert model.phase is WorkflowPhase.CANCELLING
    assert len(cancellations) == 1
    assert cancellations[0].import_id == "import-1"
    assert cancellations[0].workflow_generation == generation
    assert cancellations[0].reason == "operator cancelled"
    assert wrong_domain == []
    assert loads[0].workflow_generation == generation


def test_import_terminal_does_not_infer_ack_from_active_identity_change():
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)
    model.phase = WorkflowPhase.IMPORTING
    model.active_import_id = "import-1"

    calls = []

    def consume_then_interrupt(event):
        calls.append(event)
        if len(calls) == 1:
            controller._finish_idle()
            raise SystemExit("observer failed after transition")
        return True

    controller.handle_imported_audio_failed = consume_then_interrupt

    event = ImportedAudioFailed("import-1", "cancelled")
    assert controller._deliver_import_terminal(event) is (
        ImportTerminalRecipientResult.RETRYABLE_NACK
    )
    assert model.phase is WorkflowPhase.IDLE
    assert controller._deliver_import_terminal(event) is (
        ImportTerminalRecipientResult.ACK
    )
    assert calls == [event, event]


def test_workflow_import_terminal_classifies_malformed_and_stale_permanently():
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)

    assert controller._deliver_import_terminal(object()) is (
        ImportTerminalRecipientResult.PERMANENT_REJECT
    )
    assert controller._deliver_import_terminal(
        ImportedAudioFailed("stale-import", "late")
    ) is ImportTerminalRecipientResult.PERMANENT_REJECT
    assert controller._pending_import_terminal_commit is None


@pytest.mark.parametrize(
    "failures",
    [
        [RuntimeError("dispatch failed")],
        [KeyboardInterrupt("dispatch interrupted")],
        [SystemExit("dispatch exited")],
        [
            RuntimeError("dispatch failed"),
            KeyboardInterrupt("dispatch interrupted"),
            SystemExit("dispatch exited"),
        ],
    ],
    ids=["ordinary", "keyboard-interrupt", "system-exit", "repeated"],
)
def test_import_ready_retries_exact_staged_analysis_command_without_reapplying_state(
    failures,
):
    class Signal:
        def __init__(self, outcomes=()):
            self.outcomes = list(outcomes)
            self.messages = []

        def emit(self, message):
            self.messages.append(message)
            if self.outcomes:
                raise self.outcomes.pop(0)

    analysis_signal = Signal(failures)
    workflow_state_signal = Signal()
    bus = SimpleNamespace(
        commands=SimpleNamespace(analysis_requested=analysis_signal),
        events=SimpleNamespace(
            workflow_state_changed=workflow_state_signal,
            imported_audio_failed=Signal(),
        ),
    )
    model = SequenceWorkflowModel(configuration_generation=3)
    model.phase = WorkflowPhase.IMPORTING
    model.active_import_id = "import-1"
    model.configuration_snapshot = configuration_snapshot()
    allocated = []
    controller = SequenceWorkflowController(
        model,
        bus,
        analysis_id_factory=lambda: allocated.append("analysis-1") or "analysis-1",
        connect_bus=False,
    )
    event = ImportedAudioReady(
        "import-1", {"record_id": "record-1", "sample_rate": 48_000}
    )

    for _failure in failures:
        assert controller._deliver_import_terminal(event) is (
            ImportTerminalRecipientResult.RETRYABLE_NACK
        )
        assert model.phase is WorkflowPhase.ANALYZING
        assert model.active_analysis_id == "analysis-1"
    assert controller._deliver_import_terminal(event) is (
        ImportTerminalRecipientResult.ACK
    )

    assert allocated == ["analysis-1"]
    assert len({id(message) for message in analysis_signal.messages}) == 1
    assert len(workflow_state_signal.messages) == 1


def test_import_ready_analysis_observer_reentry_commits_and_publishes_once():
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    allocations = []
    controller = SequenceWorkflowController(
        model,
        bus,
        analysis_id_factory=lambda: allocations.append("analysis-1")
        or "analysis-1",
    )
    model.phase = WorkflowPhase.IMPORTING
    model.active_import_id = "import-reentrant"
    model.configuration_snapshot = configuration_snapshot()
    event = ImportedAudioReady(
        "import-reentrant",
        {"record_id": "record-reentrant", "sample_rate": 48_000},
    )
    delivery_id = ("ImportedAudioReady", "import-reentrant")
    publications = []
    nested_results = []

    def reenter(message):
        publications.append(message)
        if len(publications) == 1:
            nested_results.append(
                bus.deliver_import_terminal(delivery_id, event)
            )

    bus.commands.analysis_requested.connect(reenter)

    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert nested_results == [False]
    assert allocations == ["analysis-1"]
    assert len(publications) == 1
    assert controller._completed_import_terminal_commits[delivery_id] is event
    assert model.phase is WorkflowPhase.ANALYZING


def test_formal_malformed_import_failures_do_not_accumulate_local_tombstones():
    model, bus, controller = build_workflow()

    for index in range(1_000):
        import_id = f"import-{index}"
        model.phase = WorkflowPhase.IMPORTING
        model.active_import_id = import_id
        model.configuration_snapshot = configuration_snapshot()
        event = ImportedAudioReady(import_id, None, None)
        assert bus.deliver_import_terminal(
            ("ImportedAudioReady", import_id), event
        ) is True

    assert controller._pending_local_import_failure_notifications == {}
    assert (
        len(controller._completed_import_terminal_commits)
        == controller._import_terminal_commit_history_limit
    )
    assert (
        bus.completed_import_terminal_delivery_count
        == bus.import_terminal_history_limit
    )


def test_formal_import_terminal_is_consumed_once_without_raw_stale_diagnostic(qapp):
    diagnostics = []
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        model,
        bus,
        diagnostic_callback=diagnostics.append,
        analysis_id_factory=lambda: "analysis-1",
    )
    model.phase = WorkflowPhase.IMPORTING
    model.active_import_id = "import-1"
    model.configuration_snapshot = configuration_snapshot()
    analyses = []
    raw = []
    bus.commands.analysis_requested.connect(analyses.append)
    bus.events.imported_audio_ready.connect(raw.append)
    event = ImportedAudioReady(
        "import-1", {"record_id": "record-1", "sample_rate": 48_000}
    )

    assert bus.deliver_import_terminal(
        ("ImportedAudioReady", "import-1"), event
    ) is True
    bus.events.imported_audio_ready.emit(event)
    qapp.processEvents()

    assert model.phase is WorkflowPhase.ANALYZING
    assert len(analyses) == 1
    assert raw == [event]
    assert diagnostics == []


def build_workflow(
    *,
    auto_analysis=False,
    analysis_export=False,
    label_export=False,
    start_ready=True,
    diagnostic_callback=None,
):
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    # The production SequenceWindow registers direct domain recipients. This
    # isolated Workflow harness projects them onto its capture-friendly Qt
    # channels without making those raw signals canonical in application code.
    bus.register_workflow_continuation_recipient(
        "workflow-state",
        "test-workflow-state",
        lambda message: bus.events.workflow_state_changed.emit(message) or True,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "test-analysis-transport",
        lambda message: bus.events.analysis_transport_ready.emit(message) or True,
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "test-label-commit",
        lambda message: (
            bus.commands.commit_recording_label_requested.emit(message) or True
        ),
    )
    identifiers = {
        "session": iter(("session-1", "session-2", "session-3")),
        "import": iter(("import-1", "import-2")),
        "analysis": iter(("analysis-1", "analysis-2", "analysis-3")),
        "job": iter(("job-1", "job-2", "job-3")),
    }
    controller_arguments = dict(
        model=model,
        bus=bus,
        session_id_factory=lambda: next(identifiers["session"]),
        import_id_factory=lambda: next(identifiers["import"]),
        analysis_id_factory=lambda: next(identifiers["analysis"]),
        job_id_factory=lambda: next(identifiers["job"]),
        configuration_snapshot_provider=configuration_snapshot,
        start_readiness=lambda _command, _snapshot: start_ready,
        replay_readiness=lambda _command, _snapshot: True,
        import_readiness=lambda _command, _snapshot: True,
        session_snapshot_factory=lambda command, snapshot: {
            "record_id": getattr(command, "record_id", "record-1"),
            "label": getattr(command, "label", ""),
            "configuration": snapshot,
        },
        recording_snapshot_lookup=lambda record_id: (
            {"record_id": record_id, "samples": (1.0, 2.0)},
            configuration_snapshot(),
        ),
        record_id_lookup=lambda snapshot: snapshot["record_id"],
        automatic_analysis_policy=_StaticAutomaticAnalysisPolicy(auto_analysis),
    )
    if diagnostic_callback is not None:
        controller_arguments["diagnostic_callback"] = diagnostic_callback
    controller = SequenceWorkflowController(**controller_arguments)

    def prepare_analysis(request):
        assert type(request) is PrepareAnalysisExportRequested
        response = AnalysisExportPrepared(
            request.request_id,
            request.analysis_id,
            request.source_id,
            request.record_id,
            request.workflow_generation,
            request.result_snapshot,
            ({"target": "excel"},) if analysis_export else (),
        )
        return bus.deliver_workflow_continuation(
            (
                "analysis-export-prepared",
                response.request_id,
                response.workflow_generation,
            ),
            "analysis-export-prepared",
            response,
            owner=controller,
        )

    def prepare_manual(request):
        assert type(request) is PrepareManualLabelExportRequested
        response = ManualLabelExportPrepared(
            request.request_id,
            request.command_id,
            request.record_id,
            request.label,
            request.workflow_generation,
            {"record_id": request.record_id, "label": request.label},
            ({"target": "excel"},) if label_export else (),
        )
        return bus.deliver_workflow_continuation(
            (
                "manual-label-export-prepared",
                response.request_id,
                response.workflow_generation,
            ),
            "manual-label-export-prepared",
            response,
            owner=controller,
        )

    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "test-analysis-export-preparer",
        prepare_analysis,
        owner=controller,
    )
    bus.register_workflow_continuation_recipient(
        "manual-label-export-prepare",
        "test-manual-export-preparer",
        prepare_manual,
        owner=controller,
    )
    return model, bus, controller


def capture(signal):
    values = []
    signal.connect(values.append)
    return values


def admit_recording(model, controller):
    assert controller.handle_start(start_command()) is True
    assert controller.handle_recording_started(
        RecordingStarted("session-1", model.session_snapshot)
    ) is True
    assert model.phase is WorkflowPhase.RECORDING


def complete_recording(model, controller):
    admit_recording(model, controller)
    assert controller.handle_recording_completed(
        RecordingCompleted("session-1", 2, {"record_id": "record-1"})
    ) is True


def test_model_compatibility_truth_tables_are_exact():
    model = SequenceWorkflowModel()
    recording_phases = {
        WorkflowPhase.PREPARING,
        WorkflowPhase.RECORDING,
        WorkflowPhase.FINALIZING,
    }

    for phase in WorkflowPhase:
        model.phase = phase
        assert model.player_status_flag is (phase in recording_phases)
        assert model.record_workflow_busy is (phase in recording_phases)
        assert model.is_workflow_active() is (phase is not WorkflowPhase.IDLE)


def test_start_and_replay_admission_emit_domain_commands_not_controller_calls():
    model, bus, controller = build_workflow()
    recordings = capture(bus.commands.begin_recording_requested)
    states = capture(bus.events.workflow_state_changed)

    assert controller.handle_start(start_command()) is True

    assert model.phase is WorkflowPhase.PREPARING
    assert model.workflow_generation == 1
    assert recordings[0].session_id == "session-1"
    assert recordings[0].replay is False
    assert recordings[0].session_snapshot["workflow_generation"] == 1
    assert type(recordings[0].session_snapshot["workflow_generation"]) is int
    assert recordings[0].session_snapshot["configuration"] == configuration_snapshot()
    assert states[-1].previous_phase == "IDLE"
    assert states[-1].new_phase == "PREPARING"
    assert type(states[-1].new_phase) is str

    controller.handle_recording_failed(RecordingFailed("session-1", "device"))
    assert controller.handle_replay(ReplayRequested("replay-1", "button", "record-9")) is True
    assert recordings[-1].session_id == "session-2"
    assert recordings[-1].replay is True
    assert recordings[-1].session_snapshot["record_id"] == "record-9"


def test_start_rejects_readiness_and_busy_with_plain_phase_token():
    model, bus, controller = build_workflow(start_ready=(False, "device unavailable"))
    rejected = capture(bus.events.workflow_command_rejected)
    recordings = capture(bus.commands.begin_recording_requested)

    assert controller.handle_start(start_command()) is False
    assert model.phase is WorkflowPhase.IDLE
    assert rejected[-1].reason == "device unavailable"
    assert rejected[-1].current_phase == "IDLE"

    model2, bus2, controller2 = build_workflow()
    rejected2 = capture(bus2.events.workflow_command_rejected)
    assert controller2.handle_start(start_command()) is True
    before = model2.snapshot()
    assert controller2.handle_start(start_command("start-2")) is False
    assert model2.snapshot() == before
    assert rejected2[-1].current_phase == "PREPARING"
    assert recordings == []


def test_stale_configuration_generation_is_rejected_before_snapshot_lookup():
    model, bus, controller = build_workflow()
    rejected = capture(bus.events.workflow_command_rejected)
    calls = []
    controller.configuration_snapshot_provider = lambda: calls.append(True)
    stale = StartTestRequested("start-stale", "manual", "SN", False, 2)

    assert controller.handle_start(stale) is False
    assert calls == []
    assert model.phase is WorkflowPhase.IDLE
    assert rejected[-1].reason == "stale configuration generation"


def test_import_success_routes_to_analysis_and_failure_returns_idle():
    model, bus, controller = build_workflow()
    loads = capture(bus.commands.load_imported_audio_requested)
    analyses = capture(bus.commands.analysis_requested)

    assert controller.handle_import(ImportAudioRequested("load-1", "IMPORT_AUDIO", "a.wav"))
    assert model.phase is WorkflowPhase.IMPORTING
    assert loads[-1].import_id == "import-1"
    assert controller.handle_imported_audio_ready(
        ImportedAudioReady("import-1", {"record_id": "import-record"}, {"rate": 48000})
    )
    assert model.phase is WorkflowPhase.ANALYZING
    assert model.active_import_id is None
    assert model.active_session_id is None
    assert analyses[-1].analysis_id == "analysis-1"
    assert analyses[-1].source_id == "import-1"
    assert analyses[-1].automatic is True

    assert controller.handle_analysis_failed(AnalysisFailed("analysis-1", "import-1", "bad"))
    assert controller.handle_import(ImportAudioRequested("load-2", "IMPORT_AUDIO", None))
    before = model.snapshot()
    assert controller.handle_imported_audio_failed(ImportedAudioFailed("wrong-import", "bad")) is False
    assert model.snapshot() == before
    assert controller.handle_imported_audio_failed(ImportedAudioFailed("import-2", "bad"))
    assert model.phase is WorkflowPhase.IDLE


def test_manual_analysis_requires_matching_retained_ready_record():
    model, bus, controller = build_workflow()
    analyses = capture(bus.commands.analysis_requested)
    rejected = capture(bus.events.workflow_command_rejected)
    model.retained_record_id = "record-1"
    model.awaiting_label = True

    assert controller.handle_manual_analysis(ManualAnalysisRequested("analysis-cmd-1", "other")) is False
    assert model.phase is WorkflowPhase.IDLE
    assert rejected[-1].reason == "record is not retained for analysis"

    assert controller.handle_manual_analysis(ManualAnalysisRequested("analysis-cmd-2", "record-1"))
    assert model.phase is WorkflowPhase.ANALYZING
    assert analyses[-1].source_id == "record-1"
    assert analyses[-1].automatic is False


@pytest.mark.parametrize("auto_analysis", [False, True])
def test_recording_completion_retains_label_and_routes_optional_auto_analysis(auto_analysis):
    model, bus, controller = build_workflow(auto_analysis=auto_analysis)
    analyses = capture(bus.commands.analysis_requested)
    states = capture(bus.events.workflow_state_changed)

    complete_recording(model, controller)

    assert model.retained_record_id == "record-1"
    assert model.awaiting_label is True
    assert model.phase is (WorkflowPhase.ANALYZING if auto_analysis else WorkflowPhase.IDLE)
    assert [state.new_phase for state in states][-2:] == (
        ["FINALIZING", "ANALYZING"] if auto_analysis else ["FINALIZING", "IDLE"]
    )
    assert bool(analyses) is auto_analysis
    if auto_analysis:
        assert model.active_session_id is None
        assert model.active_import_id is None
        assert model.active_analysis_id == "analysis-1"
        analyzing_state = next(
            state for state in reversed(states) if state.new_phase == "ANALYZING"
        )
        assert analyzing_state.active_session_id is None
        assert analyzing_state.active_import_id is None
        assert analyzing_state.active_analysis_id == "analysis-1"


def test_recording_database_warning_still_routes_automatic_analysis():
    model, bus, controller = build_workflow(auto_analysis=True)
    analyses = capture(bus.commands.analysis_requested)
    admit_recording(model, controller)
    result_snapshot = {
        "record_id": "record-1",
        "warnings": ({"stage": "database", "message": "db unavailable"},),
    }

    assert controller.handle_recording_completed(
        RecordingCompleted("session-1", 2, result_snapshot)
    )

    assert model.phase is WorkflowPhase.ANALYZING
    assert model.retained_record_id == "record-1"
    assert len(analyses) == 1
    assert analyses[0].source_id == "session-1"
    assert analyses[0].automatic is True


@pytest.mark.parametrize(
    "terminal",
    [
        RecordingFailed("session-1", "device"),
        RecordingCancelled("session-1", "user"),
    ],
)
def test_recording_failure_or_cancel_resolves_active_session_and_rejects_stale(terminal):
    model, _bus, controller = build_workflow()
    admit_recording(model, controller)
    stale = RecordingFailed("wrong-session", "late")
    before = model.snapshot()

    assert controller.handle_recording_failed(stale) is False
    assert model.snapshot() == before
    handler = (
        controller.handle_recording_failed
        if isinstance(terminal, RecordingFailed)
        else controller.handle_recording_cancelled
    )
    assert handler(terminal) is True
    assert model.phase is WorkflowPhase.IDLE
    assert model.active_session_id is None


def test_analysis_completion_routes_optional_export_and_rejects_stale_source():
    model, bus, controller = build_workflow(auto_analysis=True, analysis_export=True)
    exports = capture(bus.commands.export_requested)
    complete_recording(model, controller)
    before = model.snapshot()

    assert controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "wrong-source", {"record_id": "record-1"})
    ) is False
    assert model.snapshot() == before
    assert controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    ) is True
    assert model.phase is WorkflowPhase.RESULT_EXPORTING
    assert model.export_continuation is ExportContinuation.ANALYSIS_DONE
    assert exports[-1].job_id == "job-1"
    assert exports[-1].record_id == "record-1"


@pytest.mark.parametrize("with_export", [False, True])
def test_manual_label_with_or_without_export_reaches_label_commit(with_export):
    model, bus, controller = build_workflow(label_export=with_export)
    commits = capture(bus.commands.commit_recording_label_requested)
    exports = capture(bus.commands.export_requested)
    model.retained_record_id = "record-1"
    model.awaiting_label = True

    assert controller.handle_manual_label(ManualLabelRequested("label-1", "record-1", "OK"))
    if with_export:
        assert model.phase is WorkflowPhase.RESULT_EXPORTING
        assert model.export_continuation is ExportContinuation.LABEL_COMMIT
        assert exports[-1].record_id == "record-1"
        assert controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-1", "record-1", ({"ok": True},))
        )
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert commits[-1].command_id == "label-1"
    assert commits[-1].record_id == "record-1"
    assert commits[-1].label == "OK"


def test_label_terminal_events_clear_only_successful_matching_retained_state():
    model, bus, controller = build_workflow()
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    controller.handle_manual_label(ManualLabelRequested("label-1", "record-1", "NG"))
    before = model.snapshot()

    assert controller.handle_label_committed(
        RecordingLabelCommitted("wrong", "record-1", "NG", {"saved": True})
    ) is False
    assert model.snapshot() == before
    assert controller.handle_label_failed(
        RecordingLabelCommitFailed("label-1", "record-1", "NG", "db")
    )
    assert model.phase is WorkflowPhase.IDLE
    assert model.awaiting_label is True

    controller.handle_manual_label(ManualLabelRequested("label-2", "record-1", "OK"))
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label-2", "record-1", "OK", {"saved": True})
    )
    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id is None
    assert model.awaiting_label is False


def test_export_failure_retry_and_ignore_validate_job_and_attempt_before_continuation():
    model, _bus, controller = build_workflow(auto_analysis=True, analysis_export=True)
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )
    before = model.snapshot()

    assert controller.handle_ignore_export_failure(
        IgnoreExportFailureRequested("job-1", "wrong-attempt")
    ) is False
    assert model.snapshot() == before
    assert controller.handle_retry_export(RetryExportRequested("job-1", "attempt-1"))
    assert model.phase is WorkflowPhase.RESULT_EXPORTING
    assert model.active_attempt_id == "attempt-1"
    assert model.export_failure_pending is True
    assert controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    assert model.active_attempt_id == "attempt-2"
    assert model.export_failure_pending is False
    assert controller.handle_export_completed(
        ExportCompleted("job-1", "attempt-2", "record-1", ())
    )
    assert model.phase is WorkflowPhase.IDLE

    model2, _bus2, controller2 = build_workflow(auto_analysis=True, analysis_export=True)
    complete_recording(model2, controller2)
    controller2.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    controller2.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )
    assert controller2.handle_ignore_export_failure(
        IgnoreExportFailureRequested("job-1", "attempt-1")
    )
    assert model2.phase is WorkflowPhase.IDLE


def test_retry_command_keeps_failed_identity_until_exact_retry_ack_or_ignore():
    model, _bus, controller = build_workflow(
        auto_analysis=True, analysis_export=True
    )
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )

    assert controller.handle_retry_export(
        RetryExportRequested("job-1", "attempt-1")
    )
    assert model.active_attempt_id == "attempt-1"
    assert model.export_failure_pending is True
    assert model.retired_attempt_ids == set()
    assert controller.handle_ignore_export_failure(
        IgnoreExportFailureRequested("job-1", "attempt-1")
    )
    assert model.phase is WorkflowPhase.IDLE


def test_retry_ack_switches_attempt_before_fast_terminal_is_accepted():
    model, _bus, controller = build_workflow(
        auto_analysis=True, analysis_export=True
    )
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )

    assert controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    assert model.active_attempt_id == "attempt-2"
    assert model.retired_attempt_ids == {"attempt-1"}
    assert controller.handle_export_completed(
        ExportCompleted("job-1", "attempt-2", "record-1", ())
    )
    assert model.phase is WorkflowPhase.IDLE


def test_workflow_retry_tombstones_are_bounded_for_long_active_job():
    model = SequenceWorkflowModel(export_attempt_history_limit=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    model.phase = WorkflowPhase.RESULT_EXPORTING
    model.active_job_id = "job-1"
    model.export_record_id = "record-1"
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )

    for number in range(2, 102):
        previous = model.active_attempt_id
        current = f"attempt-{number}"
        assert controller.handle_export_retry_accepted(
            ExportRetryAccepted("job-1", previous, current, number)
        )
        assert controller.handle_export_failed(
            ExportFailed("job-1", current, "record-1", ({"reason": "io"},))
        )

    assert len(model.retired_attempt_ids) <= 3


class _ContinuationSignal:
    def __init__(self, *errors):
        self.errors = list(errors)
        self.values = []
        self.calls = 0

    def emit(self, value):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        self.values.append(value)


def _export_continuation_workflow(
    *,
    automatic_label="OK",
    result_snapshot=None,
    label_id_factory=lambda: "label-1",
    commit_signal=None,
    transport_signal=None,
    state_signal=None,
):
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.RESULT_EXPORTING
    model.active_job_id = "job-1"
    model.export_record_id = "record-1"
    model.export_continuation = ExportContinuation.ANALYSIS_DONE
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    model.post_analysis_continuation = PostAnalysisContinuation(
        "analysis-1",
        "source-1",
        "record-1",
        model.workflow_generation,
        {"tcp_result_payload": "payload"}
        if result_snapshot is None
        else result_snapshot,
        automatic_label,
    )
    bus = SimpleNamespace(
        commands=SimpleNamespace(
            commit_recording_label_requested=(
                commit_signal or _ContinuationSignal()
            )
        ),
        events=SimpleNamespace(
            workflow_state_changed=state_signal or _ContinuationSignal(),
            analysis_transport_ready=transport_signal
            or _ContinuationSignal(),
        ),
    )
    controller = SequenceWorkflowController(
        model,
        bus,
        label_id_factory=label_id_factory,
        connect_bus=False,
    )
    return model, bus, controller


def test_export_completed_stages_label_factory_before_any_model_mutation():
    values = iter((RuntimeError("identifier"), "label-1"))

    def label_id():
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    model, bus, controller = _export_continuation_workflow(
        label_id_factory=label_id
    )
    event = ExportCompleted("job-1", "attempt-1", "record-1", ({"ok": True},))
    before = model.snapshot()
    before_outcome = model.export_outcome

    assert controller.handle_export_completed(event) is False
    assert model.snapshot() == before
    assert model.export_outcome is before_outcome
    assert bus.commands.commit_recording_label_requested.values == []

    assert controller.handle_export_completed(event) is True
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert len(bus.commands.commit_recording_label_requested.values) == 1
    assert bus.commands.commit_recording_label_requested.values[0].command_id == "label-1"


@pytest.mark.parametrize(
    "error", (RuntimeError("identifier"), KeyboardInterrupt(), SystemExit())
)
def test_export_completed_permanent_label_factory_failure_is_transactional(error):
    model, bus, controller = _export_continuation_workflow(
        label_id_factory=lambda: (_ for _ in ()).throw(error)
    )
    event = ExportCompleted("job-1", "attempt-1", "record-1", ())
    before = model.snapshot()

    assert controller.handle_export_completed(event) is False
    assert model.snapshot() == before
    assert model.export_outcome is None
    assert bus.commands.commit_recording_label_requested.values == []


@pytest.mark.parametrize(
    "error", (RuntimeError("emit"), KeyboardInterrupt(), SystemExit())
)
def test_export_completed_postcommit_emit_failure_uses_exact_retryable_outbox(error):
    commits = _ContinuationSignal(error)
    model, _bus, controller = _export_continuation_workflow(
        commit_signal=commits
    )
    event = ExportCompleted("job-1", "attempt-1", "record-1", ())

    assert controller.handle_export_completed(event) is True
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert controller.handle_export_completed(event) is False
    pending = controller.pending_continuation_publication_ids
    assert len(pending) == 1
    assert controller.retry_pending_continuation_publications() is True
    assert controller.pending_continuation_publication_ids == ()
    assert commits.calls == 2
    assert len(commits.values) == 1
    assert commits.values[0].command_id == "label-1"


def test_export_completed_transport_emit_failure_is_acked_then_retried_once():
    transports = _ContinuationSignal(RuntimeError("transport observer"))
    model, _bus, controller = _export_continuation_workflow(
        automatic_label=None,
        transport_signal=transports,
    )
    event = ExportCompleted("job-1", "attempt-1", "record-1", ())

    assert controller.handle_export_completed(event) is True
    assert model.phase is WorkflowPhase.IDLE
    assert len(controller.pending_continuation_publication_ids) == 1
    assert controller.retry_pending_continuation_publications() is True
    assert transports.calls == 2
    assert len(transports.values) == 1


def test_export_completed_transport_capacity_pressure_uses_retryable_outbox(
    qapp,
):
    model, _bus, controller = _export_continuation_workflow(
        automatic_label=None,
    )
    retained = []
    for generation in range(model.ANALYSIS_TRANSPORT_HISTORY_LIMIT):
        event = AnalysisTransportReady(
            f"analysis-capacity-{generation}",
            "source",
            "record",
            100 + generation,
            None,
        )
        assert model.authorize_analysis_transport(event)
        retained.append(event)
    event = ExportCompleted("job-1", "attempt-1", "record-1", ())

    assert controller.handle_export_completed(event) is True
    assert model.phase is WorkflowPhase.IDLE
    assert controller.pending_continuation_publication_ids == (
        ("analysis-transport", "analysis-1", "source-1", "record-1", 0),
    )
    assert controller._continuation_retry_timer.isSingleShot()
    assert controller._continuation_retry_timer.isActive()
    assert controller.continuation_retry_delay_ms > 0
    assert controller.retry_pending_continuation_publications() is False

    assert model.consume_analysis_transport(retained[0]) is True
    assert controller.retry_pending_continuation_publications() is True
    assert controller.pending_continuation_publication_ids == ()
    published = _bus.events.analysis_transport_ready.values
    assert len(published) == 1
    assert model.is_analysis_transport_authorized(published[0])


def test_disconnect_retires_authorized_transport_staged_behind_outbox():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    bus.register_workflow_continuation_recipient(
        "label-commit", "blocked", lambda _message: False
    )
    blocked = CommitRecordingLabelRequested(
        "blocked-command", "record", "OK", ()
    )
    assert controller._publish_continuation(
        ("label-commit", "blocked-command", 0),
        bus.commands.commit_recording_label_requested,
        blocked,
    ) is False
    transport = AnalysisTransportReady(
        "analysis-staged", "source", "record", 7, None
    )
    assert model.authorize_analysis_transport(transport)
    assert controller._publish_continuation(
        ("analysis-transport", "analysis-staged", "source", "record", 7),
        bus.events.analysis_transport_ready,
        transport,
    ) is False
    assert model.is_analysis_transport_authorized(transport)

    controller.disconnect()

    assert controller.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert not model.is_analysis_transport_authorized(transport)


def test_export_completed_outbox_preserves_state_before_command_publication_order():
    states = _ContinuationSignal(RuntimeError("state observer"))
    commits = _ContinuationSignal()
    model, _bus, controller = _export_continuation_workflow(
        state_signal=states,
        commit_signal=commits,
    )

    assert controller.handle_export_completed(
        ExportCompleted("job-1", "attempt-1", "record-1", ())
    )
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert len(controller.pending_continuation_publication_ids) == 2
    assert commits.calls == 0

    assert controller.retry_pending_continuation_publications()
    assert states.calls == 2
    assert commits.calls == 1
    assert controller.pending_continuation_publication_ids == ()


def test_export_completed_outbox_has_one_next_turn_retry_consumer(qapp):
    commits = _ContinuationSignal(RuntimeError("first publication"))
    _model, _bus, controller = _export_continuation_workflow(
        commit_signal=commits
    )

    assert controller.handle_export_completed(
        ExportCompleted("job-1", "attempt-1", "record-1", ())
    )
    assert len(controller.pending_continuation_publication_ids) == 1

    assert controller.retry_pending_continuation_publications()

    assert controller.pending_continuation_publication_ids == ()
    assert commits.calls == 2
    assert len(commits.values) == 1


def test_continuation_outbox_is_bounded_without_evicting_pending_delivery():
    _model, _bus, controller = _export_continuation_workflow()
    signal = _ContinuationSignal(RuntimeError("blocked"))

    for index in range(controller._continuation_outbox_limit):
        controller._publish_continuation(
            ("test", index), signal, {"index": index}
        )
    retained = controller.pending_continuation_publication_ids

    with pytest.raises(RuntimeError, match="outbox is full"):
        controller._publish_continuation(
            ("test", "overflow"), signal, {"overflow": True}
        )
    assert controller.pending_continuation_publication_ids == retained
    assert len(retained) == controller._continuation_outbox_limit
    controller.disconnect()


@pytest.mark.parametrize("failures_before_success", (2, 10, 100))
def test_continuation_outbox_retries_only_unacked_recipient_until_success(
    qapp, failures_before_success
):
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        SequenceWorkflowModel(), bus, connect_bus=False
    )
    calls = {"first": 0, "second": 0}

    def first(_message):
        calls["first"] += 1
        return True

    def second(_message):
        calls["second"] += 1
        if calls["second"] <= failures_before_success:
            raise SystemExit("transient continuation failure")
        return True

    bus.register_workflow_continuation_recipient(
        "label-commit", "first", first
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "second", second
    )
    command = CommitRecordingLabelRequested(
        "command-1", "record-1", "OK", ()
    )

    assert controller._publish_continuation(
        ("label-commit", "command-1", 0),
        bus.commands.commit_recording_label_requested,
        command,
    ) is False
    while controller.pending_continuation_publication_ids:
        controller.retry_pending_continuation_publications()

    assert calls == {"first": 1, "second": failures_before_success + 1}
    assert controller.pending_continuation_publication_ids == ()
    assert controller._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    controller.disconnect()


@pytest.mark.parametrize(
    ("initial_status", "expected_pending"),
    (
        (WorkflowContinuationDeliveryStatus.ACK, False),
        (WorkflowContinuationDeliveryStatus.RETRYABLE_NACK, True),
        (WorkflowContinuationDeliveryStatus.PERMANENT_REJECT, True),
    ),
    ids=("ack", "retryable", "generic-permanent"),
)
def test_detailed_continuation_outbox_preserves_generic_behavior(
    qapp, monkeypatch, initial_status, expected_pending
):
    model, bus, controller = build_workflow()
    outcomes = [
        WorkflowContinuationDeliveryOutcome(initial_status, "test outcome"),
        WorkflowContinuationDeliveryOutcome(
            WorkflowContinuationDeliveryStatus.ACK
        ),
    ]
    calls = []

    def deliver(delivery_id, kind, message, *, owner):
        calls.append((delivery_id, kind, message, owner))
        return outcomes.pop(0)

    monkeypatch.setattr(bus, "deliver_workflow_continuation_outcome", deliver)
    monkeypatch.setattr(
        bus,
        "deliver_workflow_continuation",
        lambda *_args, **_kwargs: pytest.fail("legacy boolean dispatcher used"),
    )
    command = CommitRecordingLabelRequested(
        "command-detailed", "record-1", "OK", ()
    )
    delivery_id = ("label-commit", command.command_id, 0)

    assert controller._publish_continuation(
        delivery_id,
        bus.commands.commit_recording_label_requested,
        command,
    ) is (not expected_pending)
    assert (delivery_id in controller.pending_continuation_publication_ids) is (
        expected_pending
    )
    assert controller._continuation_retry_timer.isActive() is expected_pending

    if expected_pending:
        assert controller.retry_pending_continuation_publications() is True
        assert controller.pending_continuation_publication_ids == ()
        assert controller._continuation_retry_timer.isActive() is False
        assert len(calls) == 2
    else:
        assert len(calls) == 1

    controller.disconnect()


def test_analysis_prepare_direct_permanent_rejection_finishes_idle(qapp):
    diagnostics = []
    model, bus, controller = build_workflow(
        auto_analysis=True,
        diagnostic_callback=diagnostics.append,
    )
    received = []
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "must-not-receive-invalid-delivery",
        lambda message: received.append(message) or True,
        owner=controller,
    )
    controller.preparation_id_factory = lambda: "p" * 513
    complete_recording(model, controller)

    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "session-1",
            {"record_id": "record-1"},
        )
    ) is True

    assert received == []
    assert model.phase is WorkflowPhase.IDLE
    assert controller._pending_export_preparation is None
    assert controller.pending_continuation_publication_ids == ()
    assert controller._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert set(diagnostic) == {"kind", "delivery_id", "reason"}
    assert diagnostic["kind"] == "analysis-export-prepare"
    assert diagnostic["delivery_id"][0] == "analysis-export-prepare"
    assert all(
        type(value) is not str or len(value) <= 256
        for value in diagnostic["delivery_id"]
    )
    assert diagnostic["reason"] == "continuation delivery identifier is invalid"
    assert len(diagnostic["reason"]) <= 256

    controller.disconnect()


def test_analysis_prepare_direct_permanent_survives_diagnostic_failure(qapp):
    diagnostic_calls = []

    def raise_diagnostic(diagnostic):
        diagnostic_calls.append(diagnostic)
        raise RuntimeError("diagnostic sink unavailable")

    model, bus, controller = build_workflow(
        auto_analysis=True,
        diagnostic_callback=raise_diagnostic,
    )
    received = []
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "must-not-receive-invalid-delivery-with-broken-diagnostics",
        lambda message: received.append(message) or True,
        owner=controller,
    )
    controller.preparation_id_factory = lambda: "p" * 513
    complete_recording(model, controller)

    assert controller.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-1",
            "session-1",
            {"record_id": "record-1"},
        )
    ) is True

    assert len(diagnostic_calls) == 1
    assert received == []
    assert model.phase is WorkflowPhase.IDLE
    assert controller._pending_export_preparation is None
    assert controller.pending_continuation_publication_ids == ()
    assert controller._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0

    controller.disconnect()


def test_permanent_continuation_failure_rearms_one_bounded_backoff_timer(qapp):
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        SequenceWorkflowModel(), bus, connect_bus=False
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "blocked", lambda _message: False
    )
    command = CommitRecordingLabelRequested(
        "command-1", "record-1", "OK", ()
    )

    assert controller._publish_continuation(
        ("label-commit", "command-1", 0),
        bus.commands.commit_recording_label_requested,
        command,
    ) is False
    delays = [controller.continuation_retry_delay_ms]
    for _ in range(20):
        assert controller.retry_pending_continuation_publications() is False
        assert controller._continuation_retry_timer.isSingleShot()
        assert controller._continuation_retry_timer.isActive()
        delays.append(controller.continuation_retry_delay_ms)

    assert delays == sorted(delays)
    assert min(delays) > 0
    assert max(delays) == controller.continuation_retry_max_delay_ms
    assert controller.pending_continuation_publication_ids == (
        ("label-commit", "command-1", 0),
    )
    assert bus.pending_workflow_continuation_delivery_count == 1

    controller.disconnect()
    assert controller._continuation_retry_timer.isActive() is False
    assert controller.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_retired_recipient_keeps_workflow_outbox_on_one_bounded_retry_timer(qapp):
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        SequenceWorkflowModel(), bus, connect_bus=False
    )
    calls = {"old": 0, "new": 0}

    def old(_message):
        calls["old"] += 1
        return False

    def new(_message):
        calls["new"] += 1
        return True

    bus.register_workflow_continuation_recipient(
        "label-commit", "replaceable", old
    )
    command = CommitRecordingLabelRequested(
        "command-1", "record-1", "OK", ()
    )
    assert controller._publish_continuation(
        ("label-commit", "command-1", 0),
        bus.commands.commit_recording_label_requested,
        command,
    ) is False
    bus.register_workflow_continuation_recipient(
        "label-commit", "replaceable", new
    )

    delays = []
    for _ in range(20):
        assert controller.retry_pending_continuation_publications() is False
        assert controller._continuation_retry_timer.isActive()
        delays.append(controller.continuation_retry_delay_ms)

    assert calls == {"old": 1, "new": 0}
    assert len(controller.pending_continuation_publication_ids) == 1
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert max(delays) == controller.continuation_retry_max_delay_ms
    assert controller.handle_start(start_command("blocked-by-outbox")) is False

    controller.disconnect()
    assert controller.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_export_completed_hostile_transport_snapshot_does_not_mutate_workflow():
    class HostileSnapshot(dict):
        def get(self, *_args, **_kwargs):
            raise SystemExit("hostile snapshot")

    model, bus, controller = _export_continuation_workflow(
        automatic_label=None,
        result_snapshot=HostileSnapshot(),
    )
    event = ExportCompleted("job-1", "attempt-1", "record-1", ())
    before = model.snapshot()

    assert controller.handle_export_completed(event) is False
    assert model.snapshot() == before
    assert model.export_outcome is None
    assert bus.events.analysis_transport_ready.values == []


def build_export_workflow_for_continuation(continuation):
    if continuation is ExportContinuation.ANALYSIS_DONE:
        model, bus, controller = build_workflow(
            auto_analysis=True,
            analysis_export=True,
        )
        complete_recording(model, controller)
        assert controller.handle_analysis_completed(
            AnalysisCompleted(
                "analysis-1",
                "session-1",
                {"record_id": "record-1"},
            )
        )
    else:
        model, bus, controller = build_workflow(label_export=True)
        model.retained_record_id = "record-1"
        model.awaiting_label = True
        assert controller.handle_manual_label(
            ManualLabelRequested("label-1", "record-1", "OK")
        )
    assert model.phase is WorkflowPhase.RESULT_EXPORTING
    assert model.export_continuation is continuation
    return model, bus, controller


def finish_confirmed_export_continuation(
    model,
    controller,
    continuation,
    commits,
    expected_outcome,
):
    if continuation is ExportContinuation.ANALYSIS_DONE:
        assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
        assert model.export_outcome == expected_outcome
        assert commits == []
        return
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert model.shutdown_cancellation_confirmed is True
    assert model.shutdown_pending is True
    assert len(commits) == 1
    assert commits[0].command_id == "label-1"
    assert commits[0].record_id == "record-1"
    assert commits[0].export_outcome == expected_outcome
    assert controller.handle_label_committed(
        RecordingLabelCommitted("label-1", "record-1", "OK", ())
    )
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING


@pytest.mark.parametrize(
    "continuation",
    [ExportContinuation.ANALYSIS_DONE, ExportContinuation.LABEL_COMMIT],
)
@pytest.mark.parametrize("decision", ["retry_success", "ignore"])
def test_shutdown_confirmed_pending_export_failure_waits_for_decision_and_continuation(
    continuation,
    decision,
):
    model, bus, controller = build_export_workflow_for_continuation(continuation)
    cancellations = capture(bus.commands.cancel_export_requested)
    commits = capture(bus.commands.commit_recording_label_requested)
    first_failure = ({"reason": "first"},)
    assert controller.handle_shutdown(ShutdownRequested(30, True))
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", first_failure)
    )
    assert model.export_failure_pending is True

    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(30)
    )

    assert model.phase is WorkflowPhase.RESULT_EXPORTING
    assert model.shutdown_cancellation_confirmed is True
    assert model.export_failure_pending is True
    assert cancellations == []

    if decision == "retry_success":
        assert controller.handle_retry_export(
            RetryExportRequested("job-1", "attempt-1")
        )
        assert controller.handle_export_retry_accepted(
            ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
        )
        assert model.phase is WorkflowPhase.RESULT_EXPORTING
        assert model.active_attempt_id == "attempt-2"
        assert model.export_failure_pending is False
        assert controller.handle_export_failed(
            ExportFailed(
                "job-1",
                "attempt-2",
                "record-1",
                ({"reason": "second"},),
            )
        )
        assert model.phase is WorkflowPhase.RESULT_EXPORTING
        assert model.export_failure_pending is True
        assert controller.handle_retry_export(
            RetryExportRequested("job-1", "attempt-2")
        )
        assert controller.handle_export_retry_accepted(
            ExportRetryAccepted("job-1", "attempt-2", "attempt-3", 3)
        )
        expected_outcome = ({"ok": True},)
        assert controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-3", "record-1", expected_outcome)
        )
    else:
        expected_outcome = first_failure
        assert controller.handle_ignore_export_failure(
            IgnoreExportFailureRequested("job-1", "attempt-1")
        )

    finish_confirmed_export_continuation(
        model,
        controller,
        continuation,
        commits,
        expected_outcome,
    )


@pytest.mark.parametrize(
    "continuation",
    [ExportContinuation.ANALYSIS_DONE, ExportContinuation.LABEL_COMMIT],
)
@pytest.mark.parametrize("terminal_kind", ["failed", "completed"])
def test_export_terminal_racing_with_confirmed_shutdown_preserves_continuation(
    continuation,
    terminal_kind,
):
    model, bus, controller = build_export_workflow_for_continuation(continuation)
    cancellations = capture(bus.commands.cancel_export_requested)
    commits = capture(bus.commands.commit_recording_label_requested)
    assert controller.handle_shutdown(ShutdownRequested(31, True))
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(31)
    )
    assert model.phase is WorkflowPhase.CANCELLING
    assert model.cancelling_domain == "export"
    assert len(cancellations) == 1

    if terminal_kind == "failed":
        failure = ({"reason": "raced"},)
        assert controller.handle_export_failed(
            ExportFailed("job-1", "attempt-1", "record-1", failure)
        )
        assert model.phase is WorkflowPhase.RESULT_EXPORTING
        assert model.shutdown_cancellation_confirmed is True
        assert model.cancelling_phase is None
        assert model.cancelling_domain is None
        assert model.export_failure_pending is True
        assert model.export_outcome == failure
        assert controller.handle_ignore_export_failure(
            IgnoreExportFailureRequested("job-1", "attempt-1")
        )
    else:
        success = ({"ok": True},)
        assert controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-1", "record-1", success)
        )
        failure = success

    finish_confirmed_export_continuation(
        model,
        controller,
        continuation,
        commits,
        failure,
    )


@pytest.mark.parametrize("late_terminal_kind", ["failed", "completed"])
def test_retry_rejects_late_terminal_from_retired_export_attempt(late_terminal_kind):
    model, _bus, controller = build_workflow(auto_analysis=True, analysis_export=True)
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )
    assert controller.handle_retry_export(RetryExportRequested("job-1", "attempt-1"))
    assert controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    assert model.active_attempt_id == "attempt-2"
    assert model.retired_attempt_ids == {"attempt-1"}
    before = model.snapshot()

    if late_terminal_kind == "failed":
        accepted = controller.handle_export_failed(
            ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "late"},))
        )
    else:
        accepted = controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-1", "record-1", ())
        )

    assert accepted is False
    assert model.snapshot() == before
    assert controller.handle_export_completed(
        ExportCompleted("job-1", "attempt-2", "record-1", ())
    )
    assert model.phase is WorkflowPhase.IDLE


@pytest.mark.parametrize("late_terminal_kind", ["failed", "completed"])
def test_cancelling_export_rejects_late_retired_attempt_terminal(
    late_terminal_kind,
):
    model, _bus, controller = build_workflow(auto_analysis=True, analysis_export=True)
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "io"},))
    )
    controller.handle_retry_export(RetryExportRequested("job-1", "attempt-1"))
    controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    controller.handle_cancel_workflow(
        CancelWorkflowRequested(
            "cancel-export", model.workflow_generation, "shutdown"
        )
    )
    before = model.snapshot()

    if late_terminal_kind == "failed":
        accepted = controller.handle_export_failed(
            ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "late"},))
        )
    else:
        accepted = controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-1", "record-1", ())
        )

    assert accepted is False
    assert model.snapshot() == before
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-2", "record-1", ({"reason": "cancelled"},))
    ) is True
    assert model.phase is WorkflowPhase.IDLE


@pytest.mark.parametrize(
    "phase, expected_signal, identifier_field, expected_id",
    [
        (WorkflowPhase.IMPORTING, "cancel_imported_audio_requested", "import_id", "import-1"),
        (WorkflowPhase.PREPARING, "cancel_recording_requested", "session_id", "session-1"),
        (WorkflowPhase.RECORDING, "cancel_recording_requested", "session_id", "session-1"),
        (WorkflowPhase.FINALIZING, "cancel_recording_requested", "session_id", "session-1"),
        (WorkflowPhase.ANALYZING, "cancel_analysis_requested", "analysis_id", "analysis-1"),
        (WorkflowPhase.RESULT_EXPORTING, "cancel_export_requested", "job_id", "job-1"),
        (WorkflowPhase.LABEL_COMMITTING, "cancel_recording_requested", "session_id", "label-1"),
    ],
)
def test_cancel_routes_to_exactly_one_active_domain(
    phase, expected_signal, identifier_field, expected_id
):
    model, bus, controller = build_workflow()
    model.phase = phase
    model.workflow_generation = 7
    if phase is WorkflowPhase.IMPORTING:
        model.active_import_id = "import-1"
    elif phase in {
        WorkflowPhase.PREPARING,
        WorkflowPhase.RECORDING,
        WorkflowPhase.FINALIZING,
    }:
        model.active_session_id = "session-1"
        model.active_session_origin = SessionOrigin.CANONICAL
    elif phase is WorkflowPhase.ANALYZING:
        model.active_analysis_id = "analysis-1"
    elif phase is WorkflowPhase.RESULT_EXPORTING:
        model.active_job_id = "job-1"
    elif phase is WorkflowPhase.LABEL_COMMITTING:
        model.active_label_command_id = "label-1"
    emissions = {
        name: capture(getattr(bus.commands, name))
        for name in (
            "cancel_recording_requested",
            "cancel_imported_audio_requested",
            "cancel_analysis_requested",
            "cancel_export_requested",
        )
    }

    command = CancelWorkflowRequested("cancel-1", 7, "close")
    assert controller.handle_cancel_workflow(command) is True
    assert model.phase is WorkflowPhase.CANCELLING
    assert sum(len(values) for values in emissions.values()) == 1
    emitted = emissions[expected_signal][0]
    assert getattr(emitted, identifier_field) == expected_id
    assert controller.handle_cancel_workflow(command) is False
    assert sum(len(values) for values in emissions.values()) == 1


def test_cancel_terminal_waits_for_matching_domain_and_continues_pending_shutdown():
    model, _bus, controller = build_workflow()
    admit_recording(model, controller)
    assert controller.handle_shutdown(ShutdownRequested(11, True)) is True
    assert model.phase is WorkflowPhase.RECORDING
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(11)
    )
    before = model.snapshot()
    assert controller.handle_recording_cancelled(RecordingCancelled("wrong", "late")) is False
    assert model.snapshot() == before
    assert controller.handle_recording_cancelled(RecordingCancelled("session-1", "close"))
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.shutdown_generation == 11


def test_recording_cancellation_keeps_legacy_guards_until_queued_terminal(qapp):
    model, bus, _controller = build_workflow()
    cancellations = capture(bus.commands.cancel_recording_requested)
    bus.commands.start_test_requested.emit(start_command())
    qapp.processEvents()
    assert model.phase is WorkflowPhase.PREPARING

    bus.commands.cancel_workflow_requested.emit(
        CancelWorkflowRequested("cancel-preparing", model.workflow_generation, "close")
    )
    qapp.processEvents()

    assert model.phase is WorkflowPhase.CANCELLING
    assert len(cancellations) == 1
    assert model.player_status_flag is True
    assert model.record_workflow_busy is True

    bus.events.recording_cancelled.emit(
        RecordingCancelled(model.active_session_id, "cancelled")
    )
    assert model.player_status_flag is True
    assert model.record_workflow_busy is True
    qapp.processEvents()
    assert model.phase is WorkflowPhase.IDLE
    assert model.player_status_flag is False
    assert model.record_workflow_busy is False


def test_shutdown_generation_is_idempotent_and_stale_ready_is_ignored():
    model, _bus, controller = build_workflow()

    assert controller.handle_shutdown(ShutdownRequested(4, False)) is True
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    before = model.snapshot()
    assert controller.handle_shutdown(ShutdownRequested(4, False)) is False
    assert controller.handle_shutdown_ready(ShutdownReady(3)) is False
    assert model.snapshot() == before
    assert controller.handle_shutdown_ready(ShutdownReady(4)) is True
    assert model.phase is WorkflowPhase.SHUTDOWN_READY


def test_shutdown_stale_active_hint_accepts_generation_and_flushes_when_idle():
    model, _bus, controller = build_workflow()

    assert controller.handle_shutdown(ShutdownRequested(12, True)) is True
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.shutdown_generation == 12
    assert model.last_shutdown_generation == 12


@pytest.mark.parametrize("reported_active", [False, True])
def test_shutdown_uses_canonical_idle_when_command_activity_hint_is_stale(
    reported_active,
):
    model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)
    assert controller.handle_shutdown(ShutdownRequested(13, reported_active)) is True
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.active_session_id is None
    assert model.shutdown_pending is False
    assert model.shutdown_asserted_active is False


@pytest.mark.parametrize(
    "admission_kind, command",
    [
        ("start", start_command("pending-start")),
        ("replay", ReplayRequested("pending-replay", "button", "record-1")),
        (
            "import",
            ImportAudioRequested("pending-import", "IMPORT_AUDIO", None),
        ),
        (
            "analysis",
            ManualAnalysisRequested("pending-analysis", "record-1"),
        ),
        ("label", ManualLabelRequested("pending-label", "record-1", "OK")),
    ],
)
def test_pending_shutdown_rejects_every_new_workflow_admission(
    admission_kind, command
):
    model, bus, controller = build_workflow()
    assert controller.handle_start(start_command())
    assert controller.handle_shutdown(ShutdownRequested(14, True))
    assert controller.handle_recording_failed(RecordingFailed("session-1", "done"))
    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_pending is True
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    rejected = capture(bus.events.workflow_command_rejected)
    handler = {
        "start": controller.handle_start,
        "replay": controller.handle_replay,
        "import": controller.handle_import,
        "analysis": controller.handle_manual_analysis,
        "label": controller.handle_manual_label,
    }[admission_kind]
    before = model.snapshot()

    assert handler(command) is False
    assert model.snapshot() == before
    assert rejected[-1].command_id == command.command_id
    assert rejected[-1].reason == "shutdown is pending"


@pytest.mark.parametrize(
    "domain",
    [
        "recording",
        "import",
        "analysis",
        "export",
        "label",
        "recording_analysis_continuation",
        "import_analysis_continuation",
        "analysis_export_continuation",
        "label_export_continuation",
    ],
)
@pytest.mark.parametrize("resolution", ["confirm", "abort"])
def test_natural_terminal_during_pending_shutdown_preserves_close_decision(
    domain, resolution
):
    model, bus, controller = build_workflow(
        auto_analysis=domain in {"export", "recording_analysis_continuation"},
        analysis_export=domain in {"export", "analysis_export_continuation"},
        label_export=domain == "label_export_continuation",
    )
    states = capture(bus.events.workflow_state_changed)
    if domain in {"recording", "recording_analysis_continuation"}:
        admit_recording(model, controller)
        if domain == "recording":
            terminal = lambda: controller.handle_recording_failed(
                RecordingFailed("session-1", "natural completion")
            )
        else:
            def terminal():
                assert controller.handle_recording_completed(
                    RecordingCompleted(
                        "session-1",
                        1,
                        {"record_id": "record-1"},
                    )
                )
                assert model.phase is WorkflowPhase.ANALYZING
                assert model.shutdown_asserted_active is True
                return controller.handle_analysis_failed(
                    AnalysisFailed(
                        "analysis-1",
                        "session-1",
                        "natural completion",
                    )
                )
    elif domain in {"import", "import_analysis_continuation"}:
        assert controller.handle_import(
            ImportAudioRequested("race-import", "IMPORT_AUDIO", None)
        )
        if domain == "import":
            terminal = lambda: controller.handle_imported_audio_failed(
                ImportedAudioFailed("import-1", "natural completion")
            )
        else:
            def terminal():
                assert controller.handle_imported_audio_ready(
                    ImportedAudioReady(
                        "import-1",
                        {"record_id": "record-1"},
                        None,
                    )
                )
                assert model.phase is WorkflowPhase.ANALYZING
                assert model.shutdown_asserted_active is True
                return controller.handle_analysis_failed(
                    AnalysisFailed(
                        "analysis-1",
                        "import-1",
                        "natural completion",
                    )
                )
    elif domain in {"analysis", "analysis_export_continuation"}:
        model.retained_record_id = "record-1"
        model.awaiting_label = True
        assert controller.handle_manual_analysis(
            ManualAnalysisRequested("race-analysis", "record-1")
        )
        if domain == "analysis":
            terminal = lambda: controller.handle_analysis_failed(
                AnalysisFailed("analysis-1", "record-1", "natural completion")
            )
        else:
            def terminal():
                assert controller.handle_analysis_completed(
                    AnalysisCompleted(
                        "analysis-1",
                        "record-1",
                        {"record_id": "record-1"},
                    )
                )
                assert model.phase is WorkflowPhase.RESULT_EXPORTING
                assert model.shutdown_asserted_active is True
                return controller.handle_export_completed(
                    ExportCompleted("job-1", "attempt-1", "record-1", ())
                )
    elif domain == "export":
        complete_recording(model, controller)
        assert controller.handle_analysis_completed(
            AnalysisCompleted(
                "analysis-1",
                "session-1",
                {"record_id": "record-1"},
            )
        )
        terminal = lambda: controller.handle_export_completed(
            ExportCompleted("job-1", "attempt-1", "record-1", ())
        )
    else:
        model.retained_record_id = "record-1"
        model.awaiting_label = True
        assert controller.handle_manual_label(
            ManualLabelRequested("race-label", "record-1", "OK")
        )
        if domain == "label":
            terminal = lambda: controller.handle_label_committed(
                RecordingLabelCommitted(
                    "race-label",
                    "record-1",
                    "OK",
                    (),
                )
            )
        else:
            def terminal():
                assert controller.handle_export_completed(
                    ExportCompleted("job-1", "attempt-1", "record-1", ())
                )
                assert model.phase is WorkflowPhase.LABEL_COMMITTING
                assert model.shutdown_pending is True
                return controller.handle_label_committed(
                    RecordingLabelCommitted(
                        "race-label",
                        "record-1",
                        "OK",
                        (),
                    )
                )

    assert model.phase is not WorkflowPhase.IDLE
    assert controller.handle_shutdown(ShutdownRequested(15, True))
    assert model.shutdown_pending is True
    assert terminal()
    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_generation == 15
    assert model.shutdown_pending is True
    assert model.shutdown_asserted_active is False

    before = model.snapshot()
    assert controller.handle_start(start_command("blocked-after-terminal")) is False
    assert model.snapshot() == before
    if resolution == "confirm":
        assert controller.handle_confirm_shutdown_cancellation(
            ConfirmShutdownCancellationRequested(15)
        )
        assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
        assert [state.new_phase for state in states[-2:]] == [
            "CLOSING",
            "SHUTDOWN_FLUSHING",
        ]
    else:
        assert controller.handle_abort_shutdown(AbortShutdownRequested(15))
        assert model.phase is WorkflowPhase.IDLE
        assert model.shutdown_generation is None
        assert model.shutdown_pending is False
        assert controller.handle_start(start_command("admitted-after-abort"))
        assert model.phase is WorkflowPhase.PREPARING


def test_queued_shutdown_uses_activity_at_delivery_time(qapp):
    model, bus, controller = build_workflow()
    bus.commands.shutdown_requested.emit(ShutdownRequested(16, False))
    assert controller.handle_start(start_command("wins-before-shutdown-delivery"))
    assert model.phase is WorkflowPhase.PREPARING

    qapp.processEvents()

    assert model.phase is WorkflowPhase.PREPARING
    assert model.shutdown_generation == 16
    assert model.shutdown_pending is True
    assert model.shutdown_asserted_active is True


def test_queued_shutdown_uses_idle_state_at_delivery_when_active_hint_is_stale(qapp):
    model, bus, controller = build_workflow()
    states = capture(bus.events.workflow_state_changed)
    admit_recording(model, controller)
    bus.commands.shutdown_requested.emit(ShutdownRequested(17, True))

    assert controller.handle_recording_failed(
        RecordingFailed("session-1", "completed before close delivery")
    )
    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_generation is None

    qapp.processEvents()

    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.shutdown_generation == 17
    assert model.shutdown_pending is False
    assert model.shutdown_asserted_active is False
    assert [state.new_phase for state in states[-2:]] == [
        "CLOSING",
        "SHUTDOWN_FLUSHING",
    ]


def test_unconfirmed_cancellation_terminal_returns_to_pending_idle_without_flush():
    model, _bus, controller = build_workflow()
    admit_recording(model, controller)
    assert controller.handle_shutdown(ShutdownRequested(17, True))
    assert controller.handle_cancel_workflow(
        CancelWorkflowRequested(
            "independent-cancel",
            model.workflow_generation,
            "operator",
        )
    )
    assert controller.handle_recording_cancelled(
        RecordingCancelled("session-1", "operator")
    )

    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_generation == 17
    assert model.shutdown_pending is True
    assert model.shutdown_cancellation_confirmed is False
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(17)
    )
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING


def test_abort_shutdown_during_unrelated_cancellation_preserves_domain_cancellation():
    model, bus, controller = build_workflow()
    cancellations = capture(bus.commands.cancel_recording_requested)
    admit_recording(model, controller)
    assert controller.handle_cancel_workflow(
        CancelWorkflowRequested(
            "operator-cancel",
            model.workflow_generation,
            "operator",
        )
    )
    assert model.phase is WorkflowPhase.CANCELLING
    assert len(cancellations) == 1

    assert controller.handle_shutdown(ShutdownRequested(18, True))
    assert model.shutdown_cancellation_confirmed is False
    assert controller.handle_abort_shutdown(AbortShutdownRequested(18))

    assert model.phase is WorkflowPhase.CANCELLING
    assert model.cancelling_phase is WorkflowPhase.RECORDING
    assert model.cancelling_domain == "recording"
    assert model.active_session_id == "session-1"
    assert model.shutdown_generation is None
    assert model.shutdown_pending is False
    assert model.shutdown_asserted_active is False
    assert model.shutdown_cancellation_confirmed is False
    assert len(cancellations) == 1

    assert controller.handle_recording_cancelled(
        RecordingCancelled("session-1", "operator")
    )
    assert model.phase is WorkflowPhase.IDLE
    assert controller.handle_start(start_command("admitted-after-cancel"))
    assert model.phase is WorkflowPhase.PREPARING


def test_abort_shutdown_rejects_after_shutdown_confirms_unrelated_cancellation():
    model, _bus, controller = build_workflow()
    admit_recording(model, controller)
    assert controller.handle_cancel_workflow(
        CancelWorkflowRequested(
            "operator-cancel",
            model.workflow_generation,
            "operator",
        )
    )
    assert controller.handle_shutdown(ShutdownRequested(19, True))
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(19)
    )
    assert model.shutdown_cancellation_confirmed is True
    before = model.snapshot()

    assert controller.handle_abort_shutdown(AbortShutdownRequested(19)) is False
    assert model.snapshot() == before
    assert controller.handle_recording_cancelled(
        RecordingCancelled("session-1", "operator")
    )
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING


def test_recording_started_requires_echoed_workflow_generation_before_mutation():
    model, _bus, controller = build_workflow()
    assert controller.handle_start(start_command())
    before = model.snapshot()

    assert controller.handle_recording_started(
        RecordingStarted("session-1", {"record_id": "record-1"})
    ) is False
    assert model.snapshot() == before
    assert controller.handle_recording_started(
        RecordingStarted(
            "session-1", {"record_id": "record-1", "workflow_generation": 0}
        )
    ) is False
    assert model.snapshot() == before
    assert controller.handle_recording_started(
        RecordingStarted("session-1", model.session_snapshot)
    ) is True
    assert model.phase is WorkflowPhase.RECORDING


def test_repeated_session_factory_still_rejects_old_recording_generation():
    model, _bus, controller = build_workflow()
    controller.session_id_factory = lambda: "reused-session"
    assert controller.handle_start(start_command("start-generation-1"))
    old_snapshot = model.session_snapshot
    assert old_snapshot["workflow_generation"] == 1
    assert controller.handle_recording_failed(
        RecordingFailed("reused-session", "first start failed")
    )

    assert controller.handle_start(start_command("start-generation-2"))
    new_session_id = model.active_session_id
    assert new_session_id != "reused-session"
    new_snapshot = model.session_snapshot
    assert new_snapshot["workflow_generation"] == 2
    before = model.snapshot()
    assert controller.handle_recording_started(
        RecordingStarted(new_session_id, old_snapshot)
    ) is False
    assert model.snapshot() == before
    assert controller.handle_recording_started(
        RecordingStarted(new_session_id, new_snapshot)
    ) is True


@pytest.mark.parametrize(
    "factory_result",
    [
        ("not", "a", "mapping"),
        {"record_id": "record-1", "workflow_generation": 999},
    ],
)
def test_session_snapshot_factory_contract_rejects_invalid_output_before_admission(
    factory_result,
):
    model, bus, controller = build_workflow()
    rejected = capture(bus.events.workflow_command_rejected)
    recordings = capture(bus.commands.begin_recording_requested)
    controller.session_snapshot_factory = lambda _command, _configuration: factory_result
    before = model.snapshot()

    assert controller.handle_start(start_command()) is False
    assert model.snapshot() == before
    assert recordings == []
    assert rejected[-1].current_phase == "IDLE"


def test_incomplete_import_ready_follows_failure_semantics_without_analysis():
    model, bus, controller = build_workflow()
    analyses = capture(bus.commands.analysis_requested)
    failures = capture(bus.events.imported_audio_failed)
    assert controller.handle_import(
        ImportAudioRequested("load-incomplete", "IMPORT_AUDIO", "bad.wav")
    )

    assert controller.handle_imported_audio_ready(
        ImportedAudioReady("import-1", None, None)
    ) is False

    assert model.phase is WorkflowPhase.IDLE
    assert analyses == []
    assert failures[-1].import_id == "import-1"
    assert failures[-1].reason == "imported recording snapshot is incomplete"


def test_formal_incomplete_import_notification_is_not_reprocessed_via_raw_signal(qapp):
    diagnostics = []
    model, bus, controller = build_workflow(diagnostic_callback=diagnostics.append)
    analyses = capture(bus.commands.analysis_requested)
    failures = capture(bus.events.imported_audio_failed)
    assert controller.handle_import(
        ImportAudioRequested("load-incomplete", "IMPORT_AUDIO", "bad.wav")
    )

    event = ImportedAudioReady("import-1", None, None)
    assert bus.deliver_import_terminal(("ImportedAudioReady", "import-1"), event)
    qapp.processEvents()
    qapp.processEvents()

    assert model.phase is WorkflowPhase.IDLE
    assert analyses == []
    assert failures == [
        ImportedAudioFailed(
            "import-1",
            "imported recording snapshot is incomplete",
        )
    ]
    assert diagnostics == []

    bus.events.imported_audio_failed.emit(
        ImportedAudioFailed("import-1", "external late failure")
    )
    qapp.processEvents()

    assert diagnostics == []


def test_import_and_recording_analysis_state_events_expose_only_analysis_id():
    model, bus, controller = build_workflow(auto_analysis=True)
    states = capture(bus.events.workflow_state_changed)
    complete_recording(model, controller)
    recording_analysis_state = states[-1]
    assert recording_analysis_state.new_phase == "ANALYZING"
    assert recording_analysis_state.active_session_id is None
    assert recording_analysis_state.active_import_id is None
    assert recording_analysis_state.active_analysis_id == "analysis-1"
    assert recording_analysis_state.active_job_id is None

    model2, bus2, controller2 = build_workflow()
    states2 = capture(bus2.events.workflow_state_changed)
    controller2.handle_import(ImportAudioRequested("load", "IMPORT_AUDIO", "a.wav"))
    controller2.handle_imported_audio_ready(
        ImportedAudioReady("import-1", {"record_id": "import-record"}, None)
    )
    import_analysis_state = states2[-1]
    assert import_analysis_state.new_phase == "ANALYZING"
    assert import_analysis_state.active_session_id is None
    assert import_analysis_state.active_import_id is None
    assert import_analysis_state.active_analysis_id == "analysis-1"
    assert import_analysis_state.active_job_id is None


@pytest.mark.parametrize(
    "phase, identifiers",
    [
        (WorkflowPhase.IDLE, {"active_session_id": "session"}),
        (
            WorkflowPhase.IMPORTING,
            {"active_import_id": "import", "active_session_id": "session"},
        ),
        (
            WorkflowPhase.ANALYZING,
            {"active_analysis_id": "analysis", "active_import_id": "import"},
        ),
        (
            WorkflowPhase.RESULT_EXPORTING,
            {"active_job_id": "job", "active_analysis_id": "analysis"},
        ),
    ],
)
def test_model_invariants_reject_illegal_active_identifier_combinations(
    phase, identifiers
):
    model = SequenceWorkflowModel()
    model.phase = phase
    for name, value in identifiers.items():
        setattr(model, name, value)

    with pytest.raises(AssertionError, match="identifier"):
        model.assert_invariants()


def test_model_invariants_keep_current_and_retired_export_attempts_exclusive():
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.RESULT_EXPORTING
    model.active_job_id = "job"
    model.active_attempt_id = "attempt"
    model.retired_attempt_ids.add("attempt")

    with pytest.raises(AssertionError, match="attempt"):
        model.assert_invariants()


def test_model_invariants_reject_confirmed_shutdown_outside_pending_cancellation():
    model = SequenceWorkflowModel()
    model.shutdown_cancellation_confirmed = True

    with pytest.raises(AssertionError, match="confirmed shutdown cancellation"):
        model.assert_invariants()


@pytest.mark.parametrize(
    "late_event_factory",
    [
        lambda identifier: RecordingCompleted(identifier, 1, {"record_id": "old"}),
        lambda identifier: RecordingFailed(identifier, "late failure"),
        lambda identifier: RecordingCancelled(identifier, "late cancellation"),
    ],
)
def test_repeated_session_factory_never_reuses_lifetime_identifier(
    late_event_factory,
):
    model, bus, controller = build_workflow()
    controller.session_id_factory = lambda: "factory-session"
    recordings = capture(bus.commands.begin_recording_requested)
    assert controller.handle_start(start_command("first-session"))
    first_id = recordings[-1].session_id
    assert first_id == "factory-session"
    assert controller.handle_recording_failed(RecordingFailed(first_id, "reset"))

    assert controller.handle_start(start_command("second-session"))
    second_id = recordings[-1].session_id
    assert second_id != first_id
    assert controller.handle_recording_started(
        RecordingStarted(second_id, model.session_snapshot)
    )
    before = model.snapshot()
    event = late_event_factory(first_id)
    handler = {
        RecordingCompleted: controller.handle_recording_completed,
        RecordingFailed: controller.handle_recording_failed,
        RecordingCancelled: controller.handle_recording_cancelled,
    }[type(event)]
    assert handler(event) is False
    assert model.snapshot() == before


@pytest.mark.parametrize("late_event_kind", ["ready", "failed"])
def test_repeated_import_factory_rejects_prior_identifier_events(late_event_kind):
    model, bus, controller = build_workflow()
    controller.import_id_factory = lambda: "factory-import"
    loads = capture(bus.commands.load_imported_audio_requested)
    assert controller.handle_import(ImportAudioRequested("first-import", "IMPORT_AUDIO", None))
    first_id = loads[-1].import_id
    assert controller.handle_imported_audio_failed(ImportedAudioFailed(first_id, "reset"))

    assert controller.handle_import(ImportAudioRequested("second-import", "IMPORT_AUDIO", None))
    second_id = loads[-1].import_id
    assert second_id != first_id
    before = model.snapshot()
    if late_event_kind == "ready":
        accepted = controller.handle_imported_audio_ready(
            ImportedAudioReady(first_id, {"record_id": "old"}, None)
        )
    else:
        accepted = controller.handle_imported_audio_failed(
            ImportedAudioFailed(first_id, "late")
        )
    assert accepted is False
    assert model.snapshot() == before


@pytest.mark.parametrize("late_event_kind", ["completed", "failed"])
def test_repeated_analysis_factory_rejects_prior_identifier_events(late_event_kind):
    model, _bus, controller = build_workflow()
    controller.analysis_id_factory = lambda: "factory-analysis"
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("first-analysis", "record-1")
    )
    first_id = model.active_analysis_id
    assert controller.handle_analysis_failed(
        AnalysisFailed(first_id, "record-1", "reset")
    )
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("second-analysis", "record-1")
    )
    second_id = model.active_analysis_id
    assert second_id != first_id
    before = model.snapshot()
    if late_event_kind == "completed":
        accepted = controller.handle_analysis_completed(
            AnalysisCompleted(first_id, "record-1", {"record_id": "record-1"})
        )
    else:
        accepted = controller.handle_analysis_failed(
            AnalysisFailed(first_id, "record-1", "late")
        )
    assert accepted is False
    assert model.snapshot() == before


@pytest.mark.parametrize("late_event_kind", ["completed", "failed"])
def test_repeated_job_factory_rejects_prior_identifier_events(late_event_kind):
    model, _bus, controller = build_workflow(
        auto_analysis=True, analysis_export=True
    )
    controller.job_id_factory = lambda: "factory-job"
    complete_recording(model, controller)
    controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    first_id = model.active_job_id
    assert controller.handle_export_completed(
        ExportCompleted(first_id, "attempt-1", "record-1", ())
    )

    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("second-analysis", "record-1")
    )
    second_analysis_id = model.active_analysis_id
    controller.handle_analysis_completed(
        AnalysisCompleted(
            second_analysis_id, "record-1", {"record_id": "record-1"}
        )
    )
    second_id = model.active_job_id
    assert second_id != first_id
    before = model.snapshot()
    if late_event_kind == "completed":
        accepted = controller.handle_export_completed(
            ExportCompleted(first_id, "attempt-late", "record-1", ())
        )
    else:
        accepted = controller.handle_export_failed(
            ExportFailed(first_id, "attempt-late", "record-1", ({"late": True},))
        )
    assert accepted is False
    assert model.snapshot() == before


@pytest.mark.parametrize(
    "mode, selected_path",
    [
        ("UNKNOWN", None),
        ("IMPORT_AUDIO", 1),
        ("IMPORT_AUDIO", {"path": "audio.wav"}),
    ],
)
def test_default_import_admission_rejects_unknown_mode_and_unsafe_path(
    mode, selected_path
):
    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)
    rejected = capture(bus.events.workflow_command_rejected)
    loads = capture(bus.commands.load_imported_audio_requested)
    command = ImportAudioRequested("unsafe-import", mode, selected_path)
    before = model.snapshot()

    assert controller.handle_import(command) is False
    assert model.snapshot() == before
    assert loads == []
    assert rejected[-1].command_id == "unsafe-import"


def test_default_import_admission_rejects_behavioral_path_without_hooks():
    class BehavioralPath(PurePosixPath):
        calls = 0

        def __str__(self):
            type(self).calls += 1
            return super().__str__()

    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)
    rejected = capture(bus.events.workflow_command_rejected)
    unsafe_path = BehavioralPath("audio.wav")
    command = SimpleNamespace(
        command_id="behavioral-path",
        mode="IMPORT_AUDIO",
        selected_path=unsafe_path,
    )
    before = model.snapshot()

    assert controller.handle_import(command) is False
    assert model.snapshot() == before
    assert rejected[-1].command_id == "behavioral-path"
    assert BehavioralPath.calls == 0


def test_injected_import_readiness_can_override_default_mode_policy():
    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(
        model,
        bus,
        import_readiness=lambda _command, _configuration: True,
    )

    assert controller.handle_import(
        ImportAudioRequested("custom-mode", "CUSTOM_IMPORT", None)
    ) is True
    assert model.phase is WorkflowPhase.IMPORTING


@pytest.mark.parametrize(
    "mode, selected_path",
    [
        ("IMPORT_AUDIO", None),
        ("IMPORT_AUDIO", "audio.wav"),
        ("IMPORT_STIMULUS_AUDIO", PurePosixPath("audio.wav")),
    ],
)
def test_default_import_admission_allows_existing_modes_and_safe_paths(
    mode, selected_path
):
    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus)

    assert controller.handle_import(
        ImportAudioRequested("safe-import", mode, selected_path)
    ) is True
    assert model.phase is WorkflowPhase.IMPORTING


def test_legacy_flag_reconciliation_requires_edge_and_tracks_recording_phases():
    model, _bus, controller = build_workflow()

    assert controller.handle_legacy_recording_flags(False, True) is False
    assert model.phase is WorkflowPhase.IDLE
    assert model.workflow_generation == 0
    assert controller.handle_legacy_recording_flags(
        False,
        True,
        activation_edge=True,
    ) is True
    assert model.phase is WorkflowPhase.PREPARING
    assert model.is_workflow_active() is True
    assert controller.handle_legacy_recording_flags(
        True,
        True,
        activation_edge=True,
    ) is True
    assert model.phase is WorkflowPhase.RECORDING
    assert controller.handle_legacy_recording_flags(False, True) is True
    assert model.phase is WorkflowPhase.FINALIZING
    assert controller.handle_legacy_recording_flags(False, False) is True
    assert model.phase is WorkflowPhase.IDLE
    assert model.is_workflow_active() is False


@pytest.mark.parametrize("admission", ["start", "replay"])
def test_canonical_recording_admission_tracks_origin_in_model_snapshot(admission):
    model, _bus, controller = build_workflow()

    if admission == "start":
        assert controller.handle_start(start_command())
    else:
        assert controller.handle_replay(
            ReplayRequested("replay-origin", "button", "record-9")
        )

    assert model.active_session_origin is SessionOrigin.CANONICAL
    assert model.snapshot().active_session_origin is model.active_session_origin
    model.assert_invariants()


@pytest.mark.parametrize("admission", ["start", "replay"])
@pytest.mark.parametrize(
    "target_phase",
    [
        WorkflowPhase.PREPARING,
        WorkflowPhase.RECORDING,
        WorkflowPhase.FINALIZING,
    ],
)
def test_legacy_false_flags_cannot_finish_canonical_recording_phase(
    admission,
    target_phase,
):
    model, _bus, controller = build_workflow()
    if admission == "start":
        assert controller.handle_start(start_command())
    else:
        assert controller.handle_replay(
            ReplayRequested("replay-flags", "button", "record-9")
        )
    session_id = model.active_session_id
    if target_phase in {WorkflowPhase.RECORDING, WorkflowPhase.FINALIZING}:
        assert controller.handle_recording_started(
            RecordingStarted(session_id, model.session_snapshot)
        )
    if target_phase is WorkflowPhase.FINALIZING:
        assert controller.handle_recording_finalizing(session_id)
    before = model.snapshot()

    assert controller.handle_legacy_recording_flags(False, False) is False

    assert model.snapshot() == before
    assert model.phase is target_phase
    assert model.active_session_id == session_id
    assert model.active_session_origin is SessionOrigin.CANONICAL


@pytest.mark.parametrize("admission", ["start", "replay"])
def test_legacy_false_flags_cannot_resolve_canonical_shutdown_cancellation(admission):
    model, _bus, controller = build_workflow()
    if admission == "start":
        assert controller.handle_start(start_command())
    else:
        assert controller.handle_replay(
            ReplayRequested("replay-shutdown", "button", "record-9")
        )
    session_id = model.active_session_id
    session_origin = model.active_session_origin
    assert controller.handle_shutdown(ShutdownRequested(31, True))
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(31)
    )
    assert model.phase is WorkflowPhase.CANCELLING

    assert controller.handle_legacy_recording_flags(False, False) is False

    assert model.phase is WorkflowPhase.CANCELLING
    assert model.active_session_id == session_id
    assert model.active_session_origin is session_origin
    assert model.shutdown_cancellation_confirmed is True
    assert controller.handle_recording_cancelled(
        RecordingCancelled(session_id, "shutdown")
    )
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.active_session_id is None
    assert model.active_session_origin is None


def test_legacy_recording_origin_can_resolve_confirmed_shutdown_cancellation():
    model, _bus, controller = build_workflow()
    assert controller.handle_legacy_recording_flags(
        True,
        True,
        activation_edge=True,
    )
    assert model.active_session_origin is SessionOrigin.LEGACY_BRIDGE
    assert controller.handle_shutdown(ShutdownRequested(32, True))
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(32)
    )
    assert model.phase is WorkflowPhase.CANCELLING

    assert controller.handle_legacy_recording_flags(False, False) is True

    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert model.active_session_id is None
    assert model.active_session_origin is None


def test_model_invariants_require_valid_origin_for_active_recording_session():
    model, _bus, controller = build_workflow()
    assert controller.handle_start(start_command())

    model.active_session_origin = None
    with pytest.raises(AssertionError, match="session origin"):
        model.assert_invariants()

    model.active_session_origin = object()
    with pytest.raises(AssertionError, match="session origin"):
        model.assert_invariants()


def test_model_invariants_reject_session_origin_without_active_session():
    model, _bus, controller = build_workflow()
    assert controller.handle_start(start_command())
    session_origin = model.active_session_origin
    model.active_session_id = None
    model.active_session_origin = session_origin

    with pytest.raises(AssertionError, match="session origin"):
        model.assert_invariants()


@pytest.mark.parametrize(
    "player_active, workflow_busy",
    [(True, False), (False, True), (True, True)],
)
def test_pending_shutdown_rejects_new_legacy_recording_activation(
    player_active,
    workflow_busy,
):
    model, _bus, controller = build_workflow()
    admit_recording(model, controller)
    assert controller.handle_shutdown(ShutdownRequested(20, True))
    assert controller.handle_recording_failed(
        RecordingFailed("session-1", "completed before legacy activation")
    )
    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_pending is True
    before = model.snapshot()

    assert controller.handle_legacy_recording_flags(
        player_active,
        workflow_busy,
        activation_edge=True,
    ) is False
    assert model.snapshot() == before
    assert model.active_session_id is None


def test_legacy_workflow_active_before_shutdown_can_finish_without_losing_close_decision():
    model, _bus, controller = build_workflow()
    assert controller.handle_legacy_recording_flags(
        False,
        True,
        activation_edge=True,
    )
    assert controller.handle_legacy_recording_flags(
        True,
        True,
        activation_edge=True,
    )
    assert model.phase is WorkflowPhase.RECORDING
    assert controller.handle_shutdown(ShutdownRequested(21, True))

    assert controller.handle_legacy_recording_flags(False, True)
    assert model.phase is WorkflowPhase.FINALIZING
    assert controller.handle_legacy_recording_flags(False, False)

    assert model.phase is WorkflowPhase.IDLE
    assert model.shutdown_generation == 21
    assert model.shutdown_pending is True
    assert model.shutdown_asserted_active is False
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(21)
    )
    assert model.phase is WorkflowPhase.SHUTDOWN_FLUSHING


def assert_stale_diagnostic(
    diagnostics,
    model,
    before,
    *,
    domain,
    event_kind,
):
    assert model.snapshot() == before
    diagnostic = diagnostics[-1]
    assert diagnostic["domain"] == domain
    assert diagnostic["event_kind"] == event_kind
    assert type(diagnostic["reason"]) is str and diagnostic["reason"]
    assert diagnostic["current_phase"] == before.phase.name
    assert diagnostic["workflow_generation"] == before.workflow_generation
    assert diagnostic["shutdown_generation"] == before.shutdown_generation
    assert all("snapshot" not in field for field in diagnostic)
    return diagnostic


def test_stale_recording_events_report_diagnostics_before_mutation():
    diagnostics = []
    model, _bus, controller = build_workflow(diagnostic_callback=diagnostics.append)
    assert controller.handle_start(start_command())

    before = model.snapshot()
    assert controller.handle_recording_started(
        RecordingStarted(
            "session-1",
            {"workflow_generation": model.workflow_generation - 1},
        )
    ) is False
    diagnostic = assert_stale_diagnostic(
        diagnostics,
        model,
        before,
        domain="recording",
        event_kind="recording_started",
    )
    assert diagnostic["expected_session_id"] == "session-1"
    assert diagnostic["received_session_id"] == "session-1"
    assert diagnostic["expected_generation"] == model.workflow_generation
    assert diagnostic["received_generation"] == model.workflow_generation - 1

    assert controller.handle_recording_started(
        RecordingStarted("session-1", model.session_snapshot)
    )
    stale_events = (
        (
            "recording_completed",
            controller.handle_recording_completed,
            RecordingCompleted("old-session", 1, {"record_id": "old"}),
        ),
        (
            "recording_failed",
            controller.handle_recording_failed,
            RecordingFailed("old-session", "late"),
        ),
        (
            "recording_cancelled",
            controller.handle_recording_cancelled,
            RecordingCancelled("old-session", "late"),
        ),
    )
    for event_kind, handler, event in stale_events:
        before = model.snapshot()
        assert handler(event) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="recording",
            event_kind=event_kind,
        )
        assert diagnostic["expected_session_id"] == "session-1"
        assert diagnostic["received_session_id"] == "old-session"


def test_stale_import_and_analysis_events_report_diagnostics_before_mutation():
    diagnostics = []
    model, _bus, controller = build_workflow(diagnostic_callback=diagnostics.append)
    assert controller.handle_import(
        ImportAudioRequested("import-diagnostics", "IMPORT_AUDIO", None)
    )
    for event_kind, handler, event in (
        (
            "imported_audio_ready",
            controller.handle_imported_audio_ready,
            ImportedAudioReady("old-import", {"record_id": "old"}, None),
        ),
        (
            "imported_audio_failed",
            controller.handle_imported_audio_failed,
            ImportedAudioFailed("old-import", "late"),
        ),
    ):
        before = model.snapshot()
        assert handler(event) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="import",
            event_kind=event_kind,
        )
        assert diagnostic["expected_import_id"] == "import-1"
        assert diagnostic["received_import_id"] == "old-import"

    assert controller.handle_imported_audio_failed(
        ImportedAudioFailed("import-1", "reset")
    )
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    assert controller.handle_manual_analysis(
        ManualAnalysisRequested("analysis-diagnostics", "record-1")
    )
    for event_kind, handler, event in (
        (
            "analysis_completed",
            controller.handle_analysis_completed,
            AnalysisCompleted("old-analysis", "record-1", {"record_id": "record-1"}),
        ),
        (
            "analysis_failed",
            controller.handle_analysis_failed,
            AnalysisFailed("old-analysis", "record-1", "late"),
        ),
    ):
        before = model.snapshot()
        assert handler(event) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="analysis",
            event_kind=event_kind,
        )
        assert diagnostic["expected_analysis_id"] == "analysis-1"
        assert diagnostic["received_analysis_id"] == "old-analysis"
        assert diagnostic["expected_source_id"] == "record-1"


def test_stale_export_attempts_and_decisions_report_before_mutation():
    diagnostics = []
    model, _bus, controller = build_workflow(
        auto_analysis=True,
        analysis_export=True,
        diagnostic_callback=diagnostics.append,
    )
    complete_recording(model, controller)
    assert controller.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "session-1", {"record_id": "record-1"})
    )
    assert controller.handle_export_failed(
        ExportFailed("job-1", "attempt-1", "record-1", ({"reason": "first"},))
    )

    for event_kind, handler, command in (
        (
            "retry_export_requested",
            controller.handle_retry_export,
            RetryExportRequested("wrong-job", "attempt-1"),
        ),
        (
            "ignore_export_failure_requested",
            controller.handle_ignore_export_failure,
            IgnoreExportFailureRequested("job-1", "wrong-attempt"),
        ),
    ):
        before = model.snapshot()
        assert handler(command) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="export",
            event_kind=event_kind,
        )
        assert diagnostic["expected_job_id"] == "job-1"
        assert diagnostic["expected_attempt_id"] == "attempt-1"

    assert controller.handle_retry_export(RetryExportRequested("job-1", "attempt-1"))
    assert controller.handle_export_retry_accepted(
        ExportRetryAccepted("job-1", "attempt-1", "attempt-2", 2)
    )
    for event_kind, handler, event in (
        (
            "export_completed",
            controller.handle_export_completed,
            ExportCompleted("job-1", "attempt-1", "record-1", ()),
        ),
        (
            "export_failed",
            controller.handle_export_failed,
            ExportFailed("job-1", "attempt-1", "record-1", ({"late": True},)),
        ),
    ):
        before = model.snapshot()
        assert handler(event) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="export",
            event_kind=event_kind,
        )
        assert diagnostic["reason"] == "retired export attempt"
        assert diagnostic["received_attempt_id"] == "attempt-1"


def test_stale_label_events_and_cancel_generation_report_before_mutation():
    diagnostics = []
    model, bus, controller = build_workflow(diagnostic_callback=diagnostics.append)
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    assert controller.handle_manual_label(
        ManualLabelRequested("label-diagnostics", "record-1", "OK")
    )
    for event_kind, handler, event in (
        (
            "recording_label_committed",
            controller.handle_label_committed,
            RecordingLabelCommitted("old-label", "record-1", "OK", ()),
        ),
        (
            "recording_label_commit_failed",
            controller.handle_label_failed,
            RecordingLabelCommitFailed("old-label", "record-1", "OK", "late"),
        ),
    ):
        before = model.snapshot()
        assert handler(event) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="label",
            event_kind=event_kind,
        )
        assert diagnostic["expected_command_id"] == "label-diagnostics"
        assert diagnostic["received_command_id"] == "old-label"

    rejected = capture(bus.events.workflow_command_rejected)
    before = model.snapshot()
    assert controller.handle_cancel_workflow(
        CancelWorkflowRequested(
            "stale-cancel",
            model.workflow_generation - 1,
            "close",
        )
    ) is False
    diagnostic = assert_stale_diagnostic(
        diagnostics,
        model,
        before,
        domain="workflow",
        event_kind="cancel_workflow_requested",
    )
    assert diagnostic["expected_generation"] == model.workflow_generation
    assert diagnostic["received_generation"] == model.workflow_generation - 1
    assert rejected[-1].command_id == "stale-cancel"


def test_stale_shutdown_generations_report_for_every_lifecycle_path():
    diagnostics = []
    model, _bus, controller = build_workflow(diagnostic_callback=diagnostics.append)
    assert controller.handle_start(start_command())
    assert controller.handle_shutdown(ShutdownRequested(30, True))

    for event_kind, handler, command in (
        (
            "confirm_shutdown_cancellation_requested",
            controller.handle_confirm_shutdown_cancellation,
            ConfirmShutdownCancellationRequested(29),
        ),
        (
            "abort_shutdown_requested",
            controller.handle_abort_shutdown,
            AbortShutdownRequested(29),
        ),
    ):
        before = model.snapshot()
        assert handler(command) is False
        diagnostic = assert_stale_diagnostic(
            diagnostics,
            model,
            before,
            domain="shutdown",
            event_kind=event_kind,
        )
        assert diagnostic["expected_generation"] == 30
        assert diagnostic["received_generation"] == 29

    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(30)
    )
    assert controller.handle_recording_cancelled(
        RecordingCancelled(model.active_session_id, "closed")
    )
    before = model.snapshot()
    assert controller.handle_shutdown_ready(ShutdownReady(29)) is False
    diagnostic = assert_stale_diagnostic(
        diagnostics,
        model,
        before,
        domain="shutdown",
        event_kind="shutdown_ready",
    )
    assert diagnostic["expected_generation"] == 30
    assert diagnostic["received_generation"] == 29

    before = model.snapshot()
    assert controller.handle_shutdown(ShutdownRequested(30, False)) is False
    assert_stale_diagnostic(
        diagnostics,
        model,
        before,
        domain="shutdown",
        event_kind="shutdown_requested",
    )


def test_default_stale_diagnostic_logs_structured_context(caplog):
    model = SequenceWorkflowModel()
    bus = SequenceEventBus()
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    before = model.snapshot()

    with caplog.at_level(
        logging.DEBUG,
        logger="ui.sequence.sequence_workflow_controller",
    ):
        assert controller.handle_recording_failed(
            RecordingFailed("closed-session", "late")
        ) is False

    assert model.snapshot() == before
    record = caplog.records[-1]
    assert record.levelno == logging.DEBUG
    assert record.workflow_diagnostic["domain"] == "recording"
    assert record.workflow_diagnostic["event_kind"] == "recording_failed"
    assert record.workflow_diagnostic["received_session_id"] == "closed-session"


def test_queued_composition_wiring_and_disconnect_are_explicit(qapp):
    model, bus, controller = build_workflow()
    emitted = capture(bus.commands.begin_recording_requested)

    bus.commands.start_test_requested.emit(start_command())
    assert model.phase is WorkflowPhase.IDLE
    qapp.processEvents()
    assert model.phase is WorkflowPhase.PREPARING
    assert len(emitted) == 1

    controller.disconnect()
    controller.handle_recording_failed(RecordingFailed("session-1", "reset"))
    bus.commands.start_test_requested.emit(start_command("start-after-disconnect"))
    qapp.processEvents()
    assert model.phase is WorkflowPhase.IDLE


def test_disconnect_blocks_already_queued_delivery_and_is_idempotent(qapp):
    model, bus, controller = build_workflow()
    recordings = capture(bus.commands.begin_recording_requested)
    rejected = capture(bus.events.workflow_command_rejected)
    bus.commands.start_test_requested.emit(start_command())

    controller.disconnect()
    controller.disconnect()
    qapp.processEvents()

    assert model.phase is WorkflowPhase.IDLE
    assert recordings == []
    assert rejected == []
    assert controller.handle_start(start_command("direct-after-disconnect")) is False
    assert controller.handle_replay(
        ReplayRequested("replay-after-disconnect", "button", "record-1")
    ) is False
    assert model.phase is WorkflowPhase.IDLE

    replacement_model = SequenceWorkflowModel(configuration_generation=3)
    replacement = SequenceWorkflowController(
        replacement_model,
        bus,
        session_id_factory=lambda: "replacement-session",
        configuration_snapshot_provider=configuration_snapshot,
    )
    assert replacement.handle_start(start_command("replacement-owner")) is True
    assert replacement_model.phase is WorkflowPhase.PREPARING
    replacement.disconnect()


def test_reentrant_disconnect_after_registration_retires_without_model_commit():
    model, bus, controller = build_workflow()
    original_register = bus._register_canonical_recording_admission
    before = model.snapshot()

    def register_then_disconnect(capability, admitted):
        assert original_register(capability, admitted) is True
        controller.disconnect()
        return True

    bus._register_canonical_recording_admission = register_then_disconnect

    assert controller.handle_start(start_command("reentrant-disconnect")) is False
    assert model.snapshot() == before

    replacement = SequenceWorkflowController(
        SequenceWorkflowModel(configuration_generation=3),
        bus,
        session_id_factory=lambda: "replacement-after-reentry",
        configuration_snapshot_provider=configuration_snapshot,
    )
    assert replacement.handle_start(start_command("replacement-after-reentry"))
    replacement.disconnect()


def test_concurrent_start_and_disconnect_linearize_before_model_commit():
    model, bus, controller = build_workflow()
    original_register = bus._register_canonical_recording_admission
    registered = Event()
    release = Event()
    before = model.snapshot()
    results = []

    def register_then_wait(capability, admitted):
        assert original_register(capability, admitted) is True
        registered.set()
        assert release.wait(2)
        return True

    bus._register_canonical_recording_admission = register_then_wait
    worker = Thread(
        target=lambda: results.append(
            controller.handle_start(start_command("concurrent-disconnect"))
        )
    )
    worker.start()
    assert registered.wait(2)
    controller.disconnect()
    release.set()
    worker.join(2)

    assert results == [False]
    assert model.snapshot() == before


def build_read_only_sequence_window_facade():
    source_path = Path(__file__).resolve().parents[2] / "ui" / "sequence" / "sequence_widget.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sequence_window = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    method_names = {
        "player_status_flag",
        "_record_workflow_busy",
        "is_workflow_active",
    }
    source_lines = source.splitlines()
    method_sources = [
        textwrap.dedent(
            "\n".join(
                source_lines[
                    min([node.lineno] + [item.lineno for item in node.decorator_list])
                    - 1 : node.end_lineno
                ]
            )
        )
        for node in sequence_window.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {node.name for node in sequence_window.body if isinstance(node, ast.FunctionDef) and node.name in method_names} == method_names
    namespace = {}
    facade_source = "class SequenceWindowFacade:\n" + textwrap.indent(
        "\n\n".join(method_sources), "    "
    )
    exec(facade_source, namespace)
    return namespace["SequenceWindowFacade"], sequence_window


def test_sequence_window_busy_surfaces_are_read_only_canonical_projections():
    facade_type, sequence_window = build_read_only_sequence_window_facade()
    model = SequenceWorkflowModel()
    window = facade_type()
    window.workflow_model = model

    assert window.player_status_flag is False
    assert window._record_workflow_busy is False
    assert window.is_workflow_active() is False

    model.phase = WorkflowPhase.PREPARING
    assert window.player_status_flag is True
    assert window._record_workflow_busy is True
    assert window.is_workflow_active() is True

    model.phase = WorkflowPhase.ANALYZING
    assert window.player_status_flag is False
    assert window._record_workflow_busy is False
    assert window.is_workflow_active() is True

    properties = {
        node.name: node
        for node in sequence_window.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"player_status_flag", "_record_workflow_busy"}
    }
    assert all(
        not any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
            for decorator in node.decorator_list
        )
        for node in properties.values()
    )


def test_sequence_window_facade_has_no_legacy_busy_state_or_arbitration():
    source_path = Path(__file__).resolve().parents[2] / "ui" / "sequence" / "sequence_widget.py"
    source = source_path.read_text(encoding="utf-8")
    assert "_sync_legacy_workflow_flags" not in source
    assert "_legacy_player_status_flag" not in source
    assert "_legacy_record_workflow_busy" not in source
