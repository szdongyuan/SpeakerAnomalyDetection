from __future__ import annotations

import ast
import os
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget

from base.excel_result_exporter import ExportResult
from base.load_config import LoadUiConfig

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

from ui.sequence.sequence_analysis_controller import (
    SequenceAnalysisTransportController,
)
from ui.sequence.sequence_event_bus import (
    SequenceEventBus,
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
    WorkflowContinuationRecipientResult,
)
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import (
    ExportJob,
    ExportJobKind,
    SequenceExportModel,
    SpoolTarget,
    mutable_export_value,
)
from ui.sequence.sequence_export_service import (
    ExportExecutionOutcome,
    ExportTargetFailure,
    ExportTargetResult,
    SequenceExportService,
)
from ui.sequence.sequence_export_view import SequenceExportView
from ui.sequence.sequence_export_worker import SequenceExportWorker
from ui.sequence.sequence_messages import (
    AnalysisCompleted,
    AnalysisTransportReady,
    AnalysisExportPrepared,
    AnalysisExportPreparationFailed,
    CancelExportPreparationRequested,
    CancelWorkflowRequested,
    ConfigurationSnapshot,
    ExportCompleted,
    ExportFailed,
    ExportRetryAccepted,
    ExportRequested,
    IgnoreExportFailureRequested,
    ManualLabelExportPreparationFailed,
    ManualLabelRequested,
    PrepareAnalysisExportRequested,
    PrepareManualLabelExportRequested,
    RecordingCompleted,
    RecordingStarted,
    RetryExportRequested,
    StartTestRequested,
)
from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_service import RecordingManualLabelRequestService
from ui.sequence.sequence_workflow_controller import (
    SequenceShutdownCoordinator,
    SequenceWorkflowController,
)
from ui.sequence.sequence_workflow_model import (
    ExportContinuation,
    SequenceWorkflowModel,
    WorkflowPhase,
)


_QAPP = QApplication.instance() or QApplication([])


def _complete_window_resource_shutdown(window, generation=0):
    lifecycle = window.resource_lifecycle_controller
    assert lifecycle.prepare_application_shutdown(generation)
    assert lifecycle.finalize_application_shutdown(generation)
    assert lifecycle.complete_application_shutdown_delivery(generation)
    assert lifecycle.complete_application_shutdown_after_ready_ack(generation)


def test_export_retry_ack_message_freezes_exact_attempt_transition():
    message = ExportRetryAccepted(
        "job-1", "attempt-1", "attempt-2", 2
    )

    assert message.job_id == "job-1"
    assert message.previous_attempt_id == "attempt-1"
    assert message.new_attempt_id == "attempt-2"
    assert message.attempt_number == 2


def test_export_owner_retains_exact_record_and_builds_frozen_manual_label_result():
    model = SequenceExportModel()
    service = SequenceExportService()
    controller = SequenceExportController(
        model, _View(), bus=SequenceEventBus(), service=service
    )
    current = mutable_export_value(
        _request("job-current", "record-current").result_snapshot
    )
    current["export_handoff"]["analysis_config"] = {
        "display_sequence": ("Result",),
        "Result": {
            "type": "Excel",
            "save_mes_enabled": True,
            "fast_mode": True,
        },
    }

    assert controller.retain_result_snapshot("wrong-record", current) is False
    assert controller.retain_result_snapshot("record-current", current)
    current["export_handoff"]["analysis_config"].clear()
    snapshot = controller.build_labeled_result("record-current", "NG")

    assert snapshot["record_id"] == "record-current"
    assert snapshot["manual_label"] == "NG"
    assert snapshot["export_handoff"]["ok_ng_summary"] == (False, "NG")
    assert [target["type"] for target in snapshot["export_targets"]] == ["excel"]
    assert controller.build_labeled_result("other-record", "OK")[
        "export_targets"
    ] == ()
    controller.disconnect()


def test_workflow_analysis_completion_retains_result_in_export_owner():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "retain-test-transport", lambda _message: True
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus
    )
    workflow_model = SequenceWorkflowModel()
    workflow_model.phase = WorkflowPhase.ANALYZING
    workflow_model.active_analysis_id = "analysis-1"
    workflow_model.analysis_source_id = "record-1"
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
    )
    source = {
        "record_id": "record-1",
        "export_handoff": {"record_id": "record-1", "analysis_config": {}},
    }

    assert workflow.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "record-1", source)
    )
    source["export_handoff"]["analysis_config"]["mutated"] = True
    retained = export_owner.model.retained_result_snapshot("record-1")
    assert "mutated" not in retained["export_handoff"]["analysis_config"]
    export_owner.disconnect()
    workflow.disconnect()


def test_sequence_window_source_has_no_export_result_cache_or_factory():
    source = Path(SequenceWindow.__module__.replace(".", "/") + ".py")
    text = source.read_text(encoding="utf-8")

    assert "_excel_export_cache" not in text
    assert "_excel_exported_record_id" not in text
    assert "_workflow_labeled_result_factory" not in text
    assert "SpoolTarget" not in text


def test_export_preparation_has_formal_owner_boundary_and_no_analysis_target_resolver():
    workflow_source = Path(
        SequenceWorkflowController.__module__.replace(".", "/") + ".py"
    ).read_text(encoding="utf-8")
    facade_source = Path(
        SequenceWindow.__module__.replace(".", "/") + ".py"
    ).read_text(encoding="utf-8")
    analysis_source = Path("ui/sequence/sequence_analysis_controller.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "analysis_export_policy",
        "label_export_policy",
        "export_targets_provider",
        "labeled_result_factory",
        "analysis_result_observer",
    ):
        assert forbidden not in workflow_source
        assert forbidden not in facade_source
    assert "build_export_targets" not in analysis_source
    assert 'snapshot["export_targets"]' not in analysis_source


def test_real_event_bus_export_owner_uses_handoff_configuration_authority():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "export-preparation-test-state", lambda _message: True
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=SequenceExportService()
    )
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = 7
    model.active_analysis_id = "analysis-7"
    model.analysis_source_id = "source-7"
    model.analysis_record_id = "record-7"
    exports = QSignalSpy(bus.commands.export_requested)
    requests = []
    prepared_events = []
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "export-preparation-test-observer",
        lambda message: requests.append(message) or True,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepared",
        "export-preparation-terminal-observer",
        lambda message: prepared_events.append(message) or True,
    )
    workflow = SequenceWorkflowController(
        model, bus, job_id_factory=lambda: "job-7"
    )
    snapshot = {
        "record_id": "record-7",
        "export_handoff": {
            "record_id": "record-7",
            "analysis_config": {
                "display_sequence": ("Result",),
                "Result": {
                    "type": "Excel",
                    "enabled": True,
                    "save_mes_enabled": True,
                },
            },
        },
    }

    assert workflow.handle_analysis_completed(
        AnalysisCompleted("analysis-7", "source-7", snapshot)
    )

    assert len(requests) == 1
    assert isinstance(requests[0], PrepareAnalysisExportRequested)
    assert requests[0].workflow_generation == 7
    assert requests[0].analysis_configuration is None
    assert len(exports) == 1
    export = exports[0][0]
    assert [target["type"] for target in export.target_configurations] == [
        "mes",
        "excel",
    ]
    retained = export_owner.model.retained_result_snapshot("record-7")
    assert retained == export.result_snapshot
    assert len(prepared_events) == 1
    assert isinstance(prepared_events[0], AnalysisExportPrepared)
    assert prepared_events[0].request_id == requests[0].request_id
    export_owner.disconnect()
    workflow.disconnect()


def test_two_production_cycles_complete_without_stuck_continuations_or_active_shutdown():
    def wait_until(predicate, timeout_seconds=2.0):
        deadline = time.monotonic() + timeout_seconds
        while True:
            _QAPP.processEvents()
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "two-cycle-workflow-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "two-cycle-analysis-transport", lambda _message: True
    )
    preparation_requests = []
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "two-cycle-preparation-observer",
        lambda message: preparation_requests.append(message) or True,
    )
    submissions = []
    export = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=SequenceExportService(),
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-attempt-{number}",
    )
    configuration = ConfigurationSnapshot(
        sequence_config={"mode": "RECORD_ONLY"},
        analysis_config={"auto_analysis": True},
        mic={"name": "two-cycle-input"},
        speaker=None,
        mic_channels=(0,),
    )
    session_ids = iter(("cycle-1-session", "cycle-2-session"))
    analysis_ids = iter(("cycle-1-analysis", "cycle-2-analysis"))
    preparation_ids = iter(("cycle-1-preparation", "cycle-2-preparation"))
    job_ids = iter(("cycle-1-job", "cycle-2-job"))
    observed_attempt_ids = set()
    model = SequenceWorkflowModel(configuration_generation=8)
    workflow = SequenceWorkflowController(
        model,
        bus,
        session_id_factory=lambda: next(session_ids),
        analysis_id_factory=lambda: next(analysis_ids),
        preparation_id_factory=lambda: next(preparation_ids),
        job_id_factory=lambda: next(job_ids),
        configuration_snapshot_provider=lambda: configuration,
        session_snapshot_factory=lambda command, snapshot: {
            "label": command.label,
            "configuration": snapshot,
        },
    )
    shutdown_ready = []
    shutdown_confirmations = []
    shutdown_cleanup = []
    shutdown_finalized = []
    shutdown_released = []
    bus.register_workflow_continuation_recipient(
        "shutdown-ready",
        "two-cycle-shutdown-ready",
        lambda event: shutdown_ready.append(event) or True,
    )
    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=SimpleNamespace(
            show_shutdown_confirmation=lambda generation: (
                shutdown_confirmations.append(generation) or True
            )
        ),
        cleanup_resources=lambda generation: (
            shutdown_cleanup.append(generation) or True
        ),
        shutdown_ready=workflow.handle_shutdown_ready,
        finalize_after_ready_ack=lambda generation: (
            shutdown_finalized.append(generation) or True
        ),
        release_shutdown_close=lambda generation: (
            shutdown_released.append(generation) or True
        ),
    )
    expected_ids = (
        (
            "cycle-1-session",
            "cycle-1-analysis",
            "cycle-1-preparation",
            "cycle-1-job",
            "cycle-1-record",
        ),
        (
            "cycle-2-session",
            "cycle-2-analysis",
            "cycle-2-preparation",
            "cycle-2-job",
            "cycle-2-record",
        ),
    )

    try:
        for cycle, identifiers in enumerate(expected_ids, start=1):
            session_id, analysis_id, preparation_id, job_id, record_id = identifiers
            assert workflow.handle_start(
                StartTestRequested(
                    f"cycle-{cycle}-start",
                    "manual",
                    f"SN-{cycle}",
                    False,
                    8,
                )
            )
            assert model.active_session_id == session_id
            assert workflow.handle_recording_started(
                RecordingStarted(session_id, model.session_snapshot)
            )
            assert workflow.handle_recording_completed(
                RecordingCompleted(
                    session_id,
                    cycle,
                    {
                        "record_id": record_id,
                        "session": {
                            "workflow_generation": cycle,
                            "mode": "RECORD_ONLY",
                            "analysis_config": {"auto_analysis": True},
                        },
                    },
                )
            )
            assert model.active_analysis_id == analysis_id
            assert model.analysis_source_id == session_id

            analysis_config = {
                "display_sequence": ("HD 1", "RB 1", "PRB 1", "Workbook"),
                "HD 1": {
                    "type": "HD",
                    "selected_labels": (2, 3),
                    "limit_checked": True,
                },
                "RB 1": {"type": "RB", "bands": (100.0, 1_000.0)},
                "PRB 1": {"type": "PRB", "enabled": True},
                "Workbook": {
                    "type": "Excel",
                    "enabled": True,
                    "save_mes_enabled": False,
                    "fast_mode": True,
                },
            }
            hd_frequency = np.array([100.0, 1_000.0], dtype=np.float64)
            hd_values = np.array([0.01, 0.02], dtype=np.float64)
            rb_frequency = np.array([200.0, 2_000.0], dtype=np.float32)
            rb_values = np.array([0.03, 0.04], dtype=np.float32)
            prb_frequency = np.array([250.0, 2_500.0], dtype=np.float64)
            prb_values = np.array([0.05, 0.06], dtype=np.float64)
            item_results = {
                "HD 1": {
                    "freq_value": hd_frequency,
                    "harmonic": (),
                    "thd": hd_values,
                    "thd_raw": hd_values,
                },
                "RB 1": {
                    "freq_value": rb_frequency,
                    "harmonic": (),
                    "thd": rb_values,
                    "thd_raw": rb_values,
                },
                "PRB 1": {
                    "freq_value": prb_frequency,
                    "harmonic": (),
                    "thd": prb_values,
                    "thd_raw": prb_values,
                },
            }
            analysis_result_dict = {
                "HD 1": (True, 0.02),
                "RB 1": (True, 0.04),
                "PRB 1": (True, 0.06),
            }
            analysis_items_data = {
                name: {
                    "type": analysis_config[name]["type"],
                    "result": result,
                    "frequency": result["freq_value"],
                    "values": result["thd"],
                }
                for name, result in item_results.items()
            }
            result_snapshot = {
                "record_id": record_id,
                "analysis_id": analysis_id,
                "source_id": session_id,
                "workflow_generation": cycle,
                "automatic": True,
                "analysis_result_dict": analysis_result_dict,
                "ok_ng_summary": (True, "OK"),
                "can_output_ok_ng": True,
                "test_mode": False,
                "tcp_result_payload": {
                    "TimeStamp": f"2026-08-24 12:00:0{cycle},000",
                    "Label": "OK",
                    "FileName": f"cycle-{cycle}.wav",
                },
                "export_handoff": {
                    "record_id": record_id,
                    "sn": f"SN-{cycle}",
                    "product_model": "two-cycle-product",
                    "date_text": f"2026/8/24 12:00:0{cycle}",
                    "analysis_items_data": analysis_items_data,
                    "analysis_result_dict": analysis_result_dict,
                    "analysis_config": analysis_config,
                    "ok_ng_summary": (True, "OK"),
                    "can_output_ok_ng": True,
                },
            }
            submissions_before = len(submissions)
            assert workflow.handle_analysis_completed(
                AnalysisCompleted(analysis_id, session_id, result_snapshot)
            )
            assert wait_until(lambda: len(submissions) == submissions_before + 1)
            assert preparation_requests[-1].request_id == preparation_id
            work, attempt = submissions[-1]
            assert work.job_id == job_id
            assert work.record_id == record_id
            assert attempt.attempt_id.startswith(f"{job_id}-attempt-1")
            assert attempt.attempt_id not in observed_attempt_ids
            observed_attempt_ids.add(attempt.attempt_id)
            assert [target["type"] for target in work.target_configurations] == [
                "excel"
            ]
            handoff = work.result_snapshot["export_handoff"]
            assert handoff["analysis_config"] == analysis_config
            assert handoff["analysis_result_dict"] == analysis_result_dict
            assert work.result_snapshot["analysis_result_dict"] == analysis_result_dict
            for name, expected_frequency, expected_values in (
                ("HD 1", hd_frequency, hd_values),
                ("RB 1", rb_frequency, rb_values),
                ("PRB 1", prb_frequency, prb_values),
            ):
                item = handoff["analysis_items_data"][name]
                assert item["type"] == analysis_config[name]["type"]
                assert np.array_equal(item["frequency"], expected_frequency)
                assert np.array_equal(item["values"], expected_values)
                assert np.array_equal(
                    item["result"]["freq_value"], expected_frequency
                )
                assert np.array_equal(item["result"]["thd"], expected_values)
            assert export.handle_worker_completed(
                ExportExecutionOutcome(
                    True,
                    work.job_id,
                    attempt.attempt_id,
                    work.record_id,
                    (ExportTargetResult("excel", "Workbook", "saved"),),
                    (),
                    (),
                    (0,),
                )
            )
            assert wait_until(lambda: model.phase is WorkflowPhase.IDLE)
            assert len(submissions) == cycle
            assert workflow._pending_export_preparation is None
            assert workflow.pending_continuation_publication_ids == ()
            assert bus.pending_workflow_continuation_delivery_count == 0

        assert model.is_workflow_active() is False
        assert coordinator.request_shutdown(91, model.is_workflow_active()) is True
        assert wait_until(
            lambda: model.phase is WorkflowPhase.SHUTDOWN_READY
            and shutdown_released == [91]
        )
        assert len(shutdown_ready) == 1
        assert shutdown_ready[0].shutdown_generation == 91
        assert shutdown_cleanup == [91]
        assert shutdown_finalized == [91]
        assert shutdown_confirmations == []
        assert model.shutdown_pending is False
        assert model.shutdown_asserted_active is False
        assert workflow.pending_continuation_publication_ids == ()
        assert bus.pending_workflow_continuation_delivery_count == 0
    finally:
        coordinator.disconnect()
        export.disconnect()
        workflow.disconnect()


def _configuration_target(name):
    return {
        "display_sequence": (name,),
        name: {
            "type": "Excel",
            "enabled": True,
            "save_mes_enabled": False,
        },
    }


def test_analysis_configuration_authority_handoff_wins_without_merging():
    service = SequenceExportService()
    handoff = _configuration_target("Handoff")
    legacy_request = _configuration_target("Request")
    legacy_top_level = _configuration_target("TopLevel")
    snapshot = {
        "record_id": "record-authority",
        "analysis_configuration": legacy_top_level,
        "export_handoff": {
            "record_id": "record-authority",
            "analysis_config": handoff,
        },
    }

    resolved = service.resolve_analysis_configuration(snapshot, legacy_request)
    prepared, targets = service.prepare_analysis_export(
        "record-authority", snapshot, legacy_request
    )

    assert resolved == handoff
    assert [target["config_name"] for target in targets] == ["Handoff"]
    assert prepared["export_targets"] == targets


def test_analysis_configuration_authority_empty_handoff_mapping_wins():
    service = SequenceExportService()
    snapshot = {
        "record_id": "record-empty-authority",
        "analysis_configuration": _configuration_target("TopLevel"),
        "export_handoff": {
            "record_id": "record-empty-authority",
            "analysis_config": {},
        },
    }

    resolved = service.resolve_analysis_configuration(
        snapshot, _configuration_target("Request")
    )
    _prepared, targets = service.prepare_analysis_export(
        "record-empty-authority", snapshot, _configuration_target("Request")
    )

    assert resolved == {}
    assert targets == ()


@pytest.mark.parametrize(
    "snapshot",
    [
        {"record_id": "record-request-fallback"},
        {"record_id": "record-request-fallback", "export_handoff": None},
        {"record_id": "record-request-fallback", "export_handoff": {}},
        {
            "record_id": "record-request-fallback",
            "export_handoff": {"analysis_config": None},
        },
    ],
    ids=[
        "handoff-missing",
        "handoff-none",
        "handoff-configuration-missing",
        "handoff-configuration-none",
    ],
)
def test_analysis_configuration_fallback_uses_legacy_request(snapshot):
    service = SequenceExportService()
    legacy_request = _configuration_target("Request")

    assert service.resolve_analysis_configuration(snapshot, legacy_request) == (
        legacy_request
    )


@pytest.mark.parametrize("request_present", [False, True], ids=["missing", "none"])
def test_analysis_configuration_fallback_uses_legacy_top_level(request_present):
    service = SequenceExportService()
    top_level = _configuration_target("TopLevel")
    snapshot = {
        "record_id": "record-top-level-fallback",
        "analysis_configuration": top_level,
    }

    if request_present:
        resolved = service.resolve_analysis_configuration(snapshot, None)
    else:
        resolved = service.resolve_analysis_configuration(snapshot)

    assert resolved == top_level


@pytest.mark.parametrize(
    ("snapshot", "legacy_request"),
    [
        (
            {
                "export_handoff": "malformed",
                "analysis_configuration": _configuration_target("TopLevel"),
            },
            _configuration_target("Request"),
        ),
        (
            {
                "export_handoff": {"analysis_config": ["malformed"]},
                "analysis_configuration": _configuration_target("TopLevel"),
            },
            _configuration_target("Request"),
        ),
        (
            {
                "export_handoff": {},
                "analysis_configuration": _configuration_target("TopLevel"),
            },
            ["malformed"],
        ),
        (
            {"analysis_configuration": ["malformed"]},
            None,
        ),
    ],
    ids=[
        "handoff",
        "handoff-configuration",
        "request-configuration",
        "top-level-configuration",
    ],
)
def test_analysis_configuration_fallback_rejects_present_malformed_values(
    snapshot, legacy_request
):
    service = SequenceExportService()

    with pytest.raises(ValueError):
        service.resolve_analysis_configuration(snapshot, legacy_request)


@pytest.mark.parametrize(
    "failure",
    [
        False,
        RuntimeError("prepare failed"),
        KeyboardInterrupt("prepare interrupted"),
        SystemExit("prepare exited"),
    ],
    ids=["invalid-result", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("recipient_failure", [False, RuntimeError("recipient")])
def test_analysis_export_preparation_failure_is_cached_and_terminal_retry_is_exact(
    monkeypatch, failure, recipient_failure
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "preparation-retry-state", lambda _message: True
    )
    service = SequenceExportService()
    calls = []

    def failed_prepare(*args):
        calls.append(args)
        if failure is False:
            return False
        raise failure

    monkeypatch.setattr(service, "prepare_analysis_export", failed_prepare)
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=service
    )
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = 9
    model.active_analysis_id = "analysis-9"
    model.analysis_source_id = "source-9"
    model.analysis_record_id = "record-9"
    workflow = SequenceWorkflowController(
        model,
        bus,
        job_id_factory=lambda: "job-9",
        preparation_id_factory=lambda: "prepare-9",
    )
    exports = QSignalSpy(bus.commands.export_requested)
    failures = []

    def flaky_terminal_recipient(message):
        failures.append(message)
        if len(failures) == 1:
            if recipient_failure is False:
                return False
            raise recipient_failure
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-export-preparation-failed",
        "preparation-failure-flaky-recipient",
        flaky_terminal_recipient,
    )
    event = AnalysisCompleted(
        "analysis-9",
        "source-9",
        {
            "record_id": "record-9",
            "analysis_configuration": {
                "Result": {"type": "Excel", "fast_mode": True}
            },
            "export_handoff": {"record_id": "record-9"},
        },
    )

    assert workflow.handle_analysis_completed(event)
    pending = workflow.pending_continuation_publication_ids
    assert pending == (("analysis-export-prepare", "prepare-9", 9),)
    assert model.phase is WorkflowPhase.IDLE
    assert len(exports) == 0
    request = next(iter(export_owner.model._export_preparations.values()))[0]
    cached = export_owner.model.prepared_export_response(request)
    assert isinstance(cached, AnalysisExportPreparationFailed)
    assert len(cached.reason) <= 1024
    assert failures == [cached]

    assert workflow.retry_pending_continuation_publications()

    assert len(calls) == 1
    assert workflow._pending_export_preparation is None
    assert workflow.pending_continuation_publication_ids == ()
    assert model.phase is WorkflowPhase.IDLE
    assert failures == [cached, cached]
    assert len(exports) == 0
    export_owner.disconnect()
    workflow.disconnect()


def test_manual_label_preparation_exception_publishes_bounded_failure_and_returns_idle(
    monkeypatch,
):
    class HostileFailure(SystemExit):
        def __str__(self):
            raise RuntimeError("hostile reason")

    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "manual-preparation-failure-state", lambda _message: True
    )
    service = SequenceExportService()
    calls = []

    def failed_prepare(*args):
        calls.append(args)
        raise HostileFailure()

    monkeypatch.setattr(service, "prepare_manual_label_export", failed_prepare)
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=service
    )
    export_owner.retain_result_snapshot("record-1", {"record_id": "record-1"})
    model = SequenceWorkflowModel()
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    workflow = SequenceWorkflowController(
        model, bus, preparation_id_factory=lambda: "manual-prepare-1"
    )
    states = []
    failures = []
    bus.events.workflow_state_changed.connect(states.append)
    bus.register_workflow_continuation_recipient(
        "manual-label-export-preparation-failed",
        "manual-preparation-failure-observer",
        lambda message: failures.append(message) or True,
    )

    assert workflow.handle_manual_label(
        ManualLabelRequested("label-1", "record-1", "NG")
    )

    assert len(calls) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ManualLabelExportPreparationFailed)
    assert failures[0].reason == "export controller failed"
    assert model.phase is WorkflowPhase.IDLE
    assert workflow._pending_export_preparation is None
    assert states == []
    export_owner.disconnect()
    workflow.disconnect()


def test_pending_export_preparation_cancellation_routes_to_export_owner():
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "preparation-cancel-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepare",
        "analysis-preparation-holder",
        lambda _message: False,
    )
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = 11
    model.active_analysis_id = "analysis-11"
    model.analysis_source_id = "source-11"
    workflow = SequenceWorkflowController(
        model, bus, preparation_id_factory=lambda: "prepare-11"
    )
    analysis_cancellations = QSignalSpy(bus.commands.cancel_analysis_requested)
    preparation_cancellations = []
    bus.register_workflow_continuation_recipient(
        "export-preparation-cancel",
        "preparation-cancel-observer",
        lambda message: preparation_cancellations.append(message) or True,
    )
    assert workflow.handle_analysis_completed(
        AnalysisCompleted(
            "analysis-11",
            "source-11",
            {"record_id": "record-11", "analysis_configuration": {}},
        )
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus
    )

    assert workflow.handle_cancel_workflow(
        CancelWorkflowRequested("cancel-11", 11, "shutdown")
    )

    assert len(analysis_cancellations) == 0
    assert len(preparation_cancellations) == 1
    assert isinstance(
        preparation_cancellations[0], CancelExportPreparationRequested
    )
    assert preparation_cancellations[0].request_id == "prepare-11"
    assert model.phase is WorkflowPhase.IDLE
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    export_owner.disconnect()
    workflow.disconnect()


def test_export_preparation_cache_and_retained_snapshots_are_bounded_and_exact():
    model = SequenceExportModel(history_limit=2)
    requests = [
        PrepareAnalysisExportRequested(
            f"request-{index}",
            f"analysis-{index}",
            f"source-{index}",
            f"record-{index}",
            index,
            {"record_id": f"record-{index}"},
            {},
        )
        for index in range(3)
    ]
    responses = [
        AnalysisExportPrepared(
            request.request_id,
            request.analysis_id,
            request.source_id,
            request.record_id,
            request.workflow_generation,
            request.result_snapshot,
            (),
        )
        for request in requests
    ]
    for request, response in zip(requests, responses):
        assert model.remember_export_preparation(request, response)
        assert model.retain_result_snapshot(
            request.record_id, request.result_snapshot
        )

    assert model.prepared_export_response(requests[0]) is None
    assert model.retained_result_snapshot("record-0") is None
    assert model.prepared_export_response(requests[1]) is responses[1]
    conflicting = PrepareAnalysisExportRequested(
        requests[1].request_id,
        "analysis-conflict",
        "source-conflict",
        "record-conflict",
        99,
        {"record_id": "record-conflict"},
        {},
    )
    assert model.prepared_export_response(conflicting) is False


def test_replace_export_preparation_response_uses_exact_compare_and_swap_without_reordering():
    model = SequenceExportModel(history_limit=2)
    requests = [
        PrepareAnalysisExportRequested(
            f"replace-request-{index}",
            f"analysis-{index}",
            f"source-{index}",
            f"record-{index}",
            index,
            {"record_id": f"record-{index}"},
        )
        for index in range(3)
    ]
    prepared = [
        AnalysisExportPrepared(
            request.request_id,
            request.analysis_id,
            request.source_id,
            request.record_id,
            request.workflow_generation,
            request.result_snapshot,
            (),
        )
        for request in requests
    ]
    failures = [
        AnalysisExportPreparationFailed(
            request.request_id,
            request.analysis_id,
            request.source_id,
            request.record_id,
            request.workflow_generation,
            "prepared response rejected",
        )
        for request in requests
    ]
    assert model.remember_export_preparation(requests[0], prepared[0])
    assert model.remember_export_preparation(requests[1], prepared[1])
    original_order = tuple(model._export_preparations)
    equal_request = PrepareAnalysisExportRequested(
        requests[0].request_id,
        requests[0].analysis_id,
        requests[0].source_id,
        requests[0].record_id,
        requests[0].workflow_generation,
        requests[0].result_snapshot,
    )
    equal_prepared = AnalysisExportPrepared(
        prepared[0].request_id,
        prepared[0].analysis_id,
        prepared[0].source_id,
        prepared[0].record_id,
        prepared[0].workflow_generation,
        prepared[0].result_snapshot,
        prepared[0].target_configurations,
    )

    assert model.replace_export_preparation_response(
        equal_request, prepared[0], failures[0]
    ) is False
    assert model.replace_export_preparation_response(
        requests[2], prepared[0], failures[2]
    ) is False
    assert model.replace_export_preparation_response(
        requests[0], equal_prepared, failures[0]
    ) is False
    assert tuple(model._export_preparations) == original_order

    assert model.replace_export_preparation_response(
        requests[0], prepared[0], failures[0]
    ) is True
    assert tuple(model._export_preparations) == original_order
    assert model.prepared_export_response(requests[0]) is failures[0]
    assert model.replace_export_preparation_response(
        requests[0], prepared[0], failures[1]
    ) is False
    assert tuple(model._export_preparations) == (
        requests[1].request_id,
        requests[0].request_id,
    )

    assert model.remember_export_preparation(requests[2], prepared[2])
    assert tuple(model._export_preparations) == (
        requests[0].request_id,
        requests[2].request_id,
    )


def _analysis_prepared_response_permanent_boundary(
    monkeypatch,
    failure_response_recipient,
    *,
    failure_recipient_before_workflow=False,
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "permanent-prepared-state", lambda _message: True
    )
    prepared_deliveries = []
    bus.register_workflow_continuation_recipient(
        "analysis-export-prepared",
        "permanent-prepared-rejector",
        lambda message: prepared_deliveries.append(message)
        or WorkflowContinuationRecipientResult.PERMANENT_REJECT,
    )
    service = SequenceExportService()
    preparation_calls = []
    original_prepare = service.prepare_analysis_export

    def counted_prepare(*args):
        preparation_calls.append(args)
        return original_prepare(*args)

    monkeypatch.setattr(service, "prepare_analysis_export", counted_prepare)
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=service
    )
    failure_deliveries = []

    def observe_failure(message):
        failure_deliveries.append(message)
        return failure_response_recipient(len(failure_deliveries))

    if failure_recipient_before_workflow:
        bus.register_workflow_continuation_recipient(
            "analysis-export-preparation-failed",
            "permanent-prepared-failure-observer",
            observe_failure,
        )
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = 21
    model.active_analysis_id = "analysis-permanent"
    model.analysis_source_id = "source-permanent"
    model.analysis_record_id = "record-permanent"
    workflow = SequenceWorkflowController(
        model,
        bus,
        preparation_id_factory=lambda: "prepare-permanent",
    )
    if not failure_recipient_before_workflow:
        bus.register_workflow_continuation_recipient(
            "analysis-export-preparation-failed",
            "permanent-prepared-failure-observer",
            observe_failure,
        )
    event = AnalysisCompleted(
        "analysis-permanent",
        "source-permanent",
        {
            "record_id": "record-permanent",
            "secret-result-marker": "must-not-enter-failure",
            "export_handoff": {
                "record_id": "record-permanent",
                "analysis_config": {},
            },
        },
    )
    return (
        bus,
        export_owner,
        workflow,
        model,
        event,
        preparation_calls,
        prepared_deliveries,
        failure_deliveries,
    )


def test_analysis_prepared_response_permanent_downgrades_once_and_failure_ack_settles(
    monkeypatch,
):
    (
        bus,
        export_owner,
        workflow,
        model,
        event,
        preparation_calls,
        prepared_deliveries,
        failure_deliveries,
    ) = _analysis_prepared_response_permanent_boundary(
        monkeypatch, lambda _attempt: True
    )

    assert workflow.handle_analysis_completed(event) is True

    request, cached = next(iter(export_owner.model._export_preparations.values()))
    assert type(cached) is AnalysisExportPreparationFailed
    assert cached.request_id == request.request_id == "prepare-permanent"
    assert cached.analysis_id == request.analysis_id
    assert cached.source_id == request.source_id
    assert cached.record_id == request.record_id
    assert cached.workflow_generation == request.workflow_generation
    assert len(cached.reason) <= 256
    assert "secret-result-marker" not in cached.reason
    assert prepared_deliveries != []
    assert failure_deliveries == [cached]
    assert len(preparation_calls) == 1
    assert workflow._pending_export_preparation is None
    assert model.phase is WorkflowPhase.IDLE
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    export_owner.disconnect()
    workflow.disconnect()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (WorkflowContinuationDeliveryStatus.ACK, True),
        (WorkflowContinuationDeliveryStatus.RETRYABLE_NACK, False),
        (WorkflowContinuationDeliveryStatus.PERMANENT_REJECT, False),
    ],
)
@pytest.mark.parametrize("publisher", ["manual", "cancel"])
def test_preparation_publishers_project_detailed_outcome_to_exact_bool(
    monkeypatch, status, expected, publisher
):
    bus = SequenceEventBus()
    controller = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, connect_bus=False
    )
    controller._formal_shutdown_completion_delivery = True
    deliveries = []

    def deliver_outcome(delivery_id, kind, message, *, owner):
        deliveries.append((delivery_id, kind, message, owner))
        return WorkflowContinuationDeliveryOutcome(status, "test outcome")

    monkeypatch.setattr(bus, "deliver_workflow_continuation_outcome", deliver_outcome)
    monkeypatch.setattr(
        bus,
        "deliver_workflow_continuation",
        lambda *_args, **_kwargs: pytest.fail("legacy boolean dispatcher used"),
    )
    if publisher == "manual":
        message = PrepareManualLabelExportRequested(
            "manual-detailed", "command-detailed", "record-detailed", "OK", 3
        )
        result = controller.handle_prepare_manual_label_export(message)
        expected_kind = "manual-label-export-prepared"
    else:
        message = CancelExportPreparationRequested(
            "cancel-detailed", 3, "cancel"
        )
        result = controller.handle_cancel_export_preparation(message)
        expected_kind = "export-preparation-cancelled"

    assert type(result) is bool
    assert result is expected
    assert len(deliveries) == 1
    assert deliveries[0][0][0] == expected_kind
    assert deliveries[0][1] == expected_kind
    assert deliveries[0][3] is controller
    controller.disconnect()


def test_preparation_publisher_keeps_legacy_signal_fallback():
    bus = SequenceEventBus()
    controller = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, connect_bus=False
    )
    delivered = QSignalSpy(bus.events.export_preparation_cancelled)
    request = CancelExportPreparationRequested("cancel-legacy", 4, "cancel")

    result = controller.handle_cancel_export_preparation(request)

    assert type(result) is bool
    assert result is True
    assert len(delivered) == 1
    assert delivered[0][0].request_id == request.request_id
    controller.disconnect()


def test_analysis_prepared_response_permanent_retryable_failure_reuses_cached_failure(
    monkeypatch,
):
    (
        bus,
        export_owner,
        workflow,
        model,
        event,
        preparation_calls,
        prepared_deliveries,
        failure_deliveries,
    ) = _analysis_prepared_response_permanent_boundary(
        monkeypatch, lambda attempt: attempt > 1
    )

    assert workflow.handle_analysis_completed(event) is True

    request, cached = next(iter(export_owner.model._export_preparations.values()))
    assert type(cached) is AnalysisExportPreparationFailed
    assert failure_deliveries == [cached]
    assert prepared_deliveries != []
    assert workflow.pending_continuation_publication_ids == (
        ("analysis-export-prepare", request.request_id, request.workflow_generation),
    )

    assert workflow.retry_pending_continuation_publications() is True

    assert failure_deliveries == [cached, cached]
    assert len(preparation_calls) == 1
    assert len(prepared_deliveries) == 1
    assert workflow._pending_export_preparation is None
    assert model.phase is WorkflowPhase.IDLE
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    export_owner.disconnect()
    workflow.disconnect()


def test_analysis_prepared_response_permanent_failure_permanent_settles_outer_request(
    monkeypatch,
):
    (
        bus,
        export_owner,
        workflow,
        model,
        event,
        preparation_calls,
        prepared_deliveries,
        failure_deliveries,
    ) = _analysis_prepared_response_permanent_boundary(
        monkeypatch,
        lambda _attempt: WorkflowContinuationRecipientResult.PERMANENT_REJECT,
        failure_recipient_before_workflow=True,
    )

    assert workflow.handle_analysis_completed(event) is True

    request, cached = next(iter(export_owner.model._export_preparations.values()))
    assert type(cached) is AnalysisExportPreparationFailed
    assert failure_deliveries == [cached]
    assert len(preparation_calls) == 1
    assert len(prepared_deliveries) == 1
    assert workflow._pending_export_preparation is None
    assert model.phase is WorkflowPhase.IDLE
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_diagnostics != ()
    assert request.request_id == "prepare-permanent"
    export_owner.disconnect()
    workflow.disconnect()


@pytest.mark.parametrize(
    "response_path", ["prepared", "failed"], ids=["prepared", "failed"]
)
@pytest.mark.parametrize(
    "workflow_first",
    [True, False],
    ids=["workflow-before-terminal", "terminal-before-workflow"],
)
def test_analysis_preparation_terminal_rejection_order_is_correlated_and_final(
    monkeypatch, response_path, workflow_first
):
    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "ordered-rejection-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "ordered-rejection-transport", lambda _message: True
    )
    rejected_responses = []
    rejected_kind = (
        "analysis-export-prepared"
        if response_path == "prepared"
        else "analysis-export-preparation-failed"
    )

    def reject_response(message):
        rejected_responses.append(message)
        return WorkflowContinuationRecipientResult.PERMANENT_REJECT

    if not workflow_first:
        bus.register_workflow_continuation_recipient(
            rejected_kind,
            "ordered-rejection-terminal",
            reject_response,
        )
    service = SequenceExportService()
    if response_path == "failed":
        def fail_preparation(*_args):
            raise RuntimeError("bounded preparation failure")

        monkeypatch.setattr(
            service,
            "prepare_analysis_export",
            fail_preparation,
        )
    submissions = []
    export_owner = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=service,
        submit_attempt=lambda work, attempt: submissions.append((work, attempt)),
    )
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.workflow_generation = 31
    model.active_analysis_id = "analysis-ordered"
    model.analysis_source_id = "source-ordered"
    model.analysis_record_id = "record-ordered"
    workflow = SequenceWorkflowController(
        model,
        bus,
        job_id_factory=lambda: "job-ordered",
        preparation_id_factory=lambda: "prepare-ordered",
    )
    if workflow_first:
        bus.register_workflow_continuation_recipient(
            rejected_kind,
            "ordered-rejection-terminal",
            reject_response,
        )
    analysis_configuration = (
        _configuration_target("Ordered") if response_path == "prepared" else {}
    )
    event = AnalysisCompleted(
        "analysis-ordered",
        "source-ordered",
        {
            "record_id": "record-ordered",
            "export_handoff": {
                "record_id": "record-ordered",
                "analysis_config": analysis_configuration,
            },
        },
    )

    assert workflow.handle_analysis_completed(event) is True
    _QAPP.processEvents()

    request = next(iter(export_owner.model._export_preparations.values()))[0]
    assert len(rejected_responses) == 1
    assert workflow._pending_export_preparation is None
    assert workflow.pending_continuation_publication_ids == ()
    assert workflow._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    if response_path == "prepared" and workflow_first:
        assert model.phase is WorkflowPhase.RESULT_EXPORTING
        assert len(submissions) == 1
        work, attempt = submissions[0]
        outcome = ExportExecutionOutcome(
            True,
            work.job_id,
            attempt.attempt_id,
            work.record_id,
            (ExportTargetResult("excel", "Ordered", "written"),),
            (),
            (),
            completed_target_indices=(0,),
        )
        assert export_owner.handle_worker_completed(outcome) is True
    else:
        assert submissions == []

    assert model.phase is WorkflowPhase.IDLE
    assert workflow._pending_export_preparation is None
    assert workflow.pending_continuation_publication_ids == ()
    assert workflow._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    stale_response = AnalysisExportPreparationFailed(
        "stale-prepare",
        request.analysis_id,
        request.source_id,
        request.record_id,
        request.workflow_generation,
        "stale",
    )
    assert workflow.handle_analysis_export_preparation_failed(stale_response) is False
    stale_request = replace(request, request_id="stale-prepare")
    stale_publication = SimpleNamespace(
        kind="analysis-export-prepare",
        delivery_id=(
            "analysis-export-prepare",
            stale_request.request_id,
            stale_request.workflow_generation,
        ),
        message=stale_request,
    )
    assert workflow._settle_rejected_analysis_export_preparation(
        stale_publication,
        WorkflowContinuationDeliveryOutcome(
            WorkflowContinuationDeliveryStatus.PERMANENT_REJECT,
            "stale request",
        ),
    ) is False
    if response_path == "prepared":
        assert type(rejected_responses[0]) is AnalysisExportPrepared
        assert workflow.handle_analysis_export_prepared(rejected_responses[0]) is False
    exact_old_failure = AnalysisExportPreparationFailed(
        request.request_id,
        request.analysis_id,
        request.source_id,
        request.record_id,
        request.workflow_generation,
        "old workflow",
    )
    workflow._begin_workflow()
    assert workflow.handle_analysis_export_preparation_failed(exact_old_failure) is False
    export_owner.disconnect()
    workflow.disconnect()


def _failed_workflow_for_retry(bus, attempt_id="attempt-1"):
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.RESULT_EXPORTING
    model.active_job_id = "job-1"
    model.export_record_id = "record-1"
    model.export_continuation = ExportContinuation.ANALYSIS_DONE
    controller = SequenceWorkflowController(
        model,
        bus,
        export_decision_requires_terminal=True,
        connect_bus=False,
    )
    bus.register_export_retry_recipient(
        controller.handle_export_retry_accepted
    )
    assert controller.handle_export_failed(
        ExportFailed(
            "job-1", attempt_id, "record-1", ({"reason": "locked"},)
        )
    )
    return model, controller


def test_controller_retry_allocation_failure_leaves_workflow_ignore_actionable():
    bus = SequenceEventBus()
    values = iter(("attempt-1", RuntimeError("id allocation")))

    def allocate(*_args):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    submissions = []
    export = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=allocate,
        connect_bus=False,
    )
    export.handle_export_requested(_request())
    work, attempt = submissions.pop()
    assert attempt.attempt_id.startswith("attempt-1::")
    workflow_model, workflow = _failed_workflow_for_retry(
        bus, attempt.attempt_id
    )
    export.model.fail_record_attempt(
        work.job_id,
        attempt.attempt_id,
        (ExportTargetFailure("excel", "Result", "locked"),),
    )

    command = RetryExportRequested("job-1", attempt.attempt_id)
    assert workflow.handle_retry_export(command)
    assert export.handle_retry_requested(command) is False
    assert workflow_model.active_attempt_id == attempt.attempt_id
    assert workflow_model.export_failure_pending is True
    assert workflow.handle_ignore_export_failure(
        IgnoreExportFailureRequested("job-1", attempt.attempt_id)
    )


def test_controller_delivers_retry_ack_before_fast_retry_terminal():
    bus = SequenceEventBus()
    submissions = []

    def submit(job, attempt):
        submissions.append((job, attempt))
        if attempt.attempt_number == 2:
            assert workflow_model.active_attempt_id == attempt.attempt_id
            assert workflow_model.export_failure_pending is False

    export = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=submit,
        attempt_id_factory=lambda _job_id, number: f"attempt-{number}",
        connect_bus=False,
    )
    export.handle_export_requested(_request())
    work, attempt = submissions.pop()
    workflow_model, workflow = _failed_workflow_for_retry(
        bus, attempt.attempt_id
    )
    export.model.fail_record_attempt(
        work.job_id,
        attempt.attempt_id,
        (ExportTargetFailure("excel", "Result", "locked"),),
    )
    command = RetryExportRequested("job-1", attempt.attempt_id)

    assert workflow.handle_retry_export(command)
    assert export.handle_retry_requested(command)
    assert submissions[-1][1].attempt_number == 2
    assert workflow_model.active_attempt_id == submissions[-1][1].attempt_id
    assert workflow_model.export_failure_pending is False


def test_record_attempt_identity_history_is_bounded_under_constant_factory():
    model = SequenceExportModel(history_limit=3)
    model.enqueue_record(_request())
    model.begin_next_record_job()
    attempt = model.begin_record_attempt(lambda *_args: "same-hint")
    identities = {attempt.attempt_id}

    for _index in range(100):
        assert model.fail_record_attempt(
            attempt.job_id,
            attempt.attempt_id,
            (ExportTargetFailure("excel", "Result", "locked"),),
        )
        attempt = model.retry_record_attempt(
            attempt.job_id, attempt.attempt_id, lambda *_args: "same-hint"
        )
        assert attempt.attempt_id not in identities
        identities.add(attempt.attempt_id)

    assert not hasattr(model, "_record_used_attempt_ids")
    assert len(model._retired_attempts) <= 3


class _Loop3Dialog:
    Warning = 1
    AcceptRole = 2
    RejectRole = 3

    def __init__(self, *_args):
        self.buttonClicked = _ConnectableSignal()
        self.buttons = []
        self.opened = False
        self.closed = False
        self.deleted = False

    def setWindowModality(self, value):
        assert value == Qt.WindowModal

    def setIcon(self, _value):
        pass

    def setWindowTitle(self, _value):
        pass

    def setLabelText(self, _value):
        pass

    def setCancelButton(self, _value):
        pass

    def setRange(self, _minimum, _maximum):
        pass

    def setText(self, _value):
        pass

    def setInformativeText(self, _value):
        pass

    def addButton(self, text, _role):
        button = object()
        self.buttons.append((text, button))
        return button

    def setDefaultButton(self, _button):
        pass

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def deleteLater(self):
        self.deleted = True


def test_initial_identifier_recovery_prepares_real_view_identity_before_dialog():
    decision = _Loop3Dialog()
    view = SequenceExportView(
        progress_dialog_factory=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("progress must not open")
        ),
        failure_dialog_factory=lambda *_args: decision,
        fallback_failure_dialog_factory=lambda *_args: _Loop3Dialog(),
    )
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=_Bus(),
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("identifier")
        ),
        connect_bus=False,
    )

    controller.handle_export_requested(_request())
    recovery = controller.model.active_record_attempt

    assert recovery is not None
    assert decision.opened is True
    assert view.recovery_pending_identity is None
    assert submissions == []


@pytest.mark.parametrize(
    ("stage", "error"),
    (
        ("factory", RuntimeError("factory")),
        ("setup", KeyboardInterrupt("setup")),
        ("open", SystemExit("open")),
    ),
)
def test_progress_presentation_failure_becomes_retryable_terminal(stage, error):
    candidates = []

    class FailingProgress(_Loop3Dialog):
        def setWindowTitle(self, value):
            if stage == "setup":
                raise error
            super().setWindowTitle(value)

        def open(self):
            if stage == "open":
                raise error
            super().open()

    def progress_factory(*_args):
        if stage == "factory":
            raise error
        candidate = FailingProgress()
        candidates.append(candidate)
        return candidate

    decision = _Loop3Dialog()
    view = SequenceExportView(
        progress_dialog_factory=progress_factory,
        failure_dialog_factory=lambda *_args: decision,
        fallback_failure_dialog_factory=_Loop3Dialog,
    )
    submissions = []
    bus = _Bus()
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )

    controller.handle_export_requested(_request())
    attempt = controller.model.active_record_attempt

    assert submissions == []
    assert controller.presentation_pending_identity is None
    assert controller.model.record_failure is not None
    assert len(bus.events.export_failed.values) == 1
    assert bus.events.export_failed.values[0].attempt_id == attempt.attempt_id
    assert decision.opened is True
    if candidates:
        assert candidates[0].closed is True
        assert candidates[0].deleted is True
    view._progress_dialog_factory = _Loop3Dialog
    assert controller.handle_retry_requested(
        RetryExportRequested("job-1", attempt.attempt_id)
    )
    assert len(submissions) == 1


def test_progress_presentation_failure_can_be_ignored_and_fifo_continues():
    decision = _Loop3Dialog()
    view = SequenceExportView(
        progress_dialog_factory=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("progress")
        ),
        failure_dialog_factory=lambda *_args: decision,
        fallback_failure_dialog_factory=_Loop3Dialog,
    )
    submissions = []
    bus = _Bus()
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )

    assert controller.handle_export_requested(_request("job-1", "record-1"))
    attempt = controller.model.active_record_attempt
    assert controller.handle_export_requested(_request("job-2", "record-2"))
    view._progress_dialog_factory = _Loop3Dialog

    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested("job-1", attempt.attempt_id)
    )
    assert len(bus.events.export_failed.values) == 1
    assert len(bus.events.export_completed.values) == 1
    assert submissions[-1][0].job_id == "job-2"


def test_failure_dialog_setup_failure_disposes_candidate_close_and_delete_independently():
    primary = _Loop3Dialog()

    def fail_text(_value):
        raise KeyboardInterrupt("setup")

    def fail_close():
        primary.closed = True
        raise SystemExit("close")

    primary.setText = fail_text
    primary.close = fail_close
    fallback = _Loop3Dialog()
    view = SequenceExportView(
        progress_dialog_factory=_Loop3Dialog,
        failure_dialog_factory=lambda *_args: primary,
        fallback_failure_dialog_factory=lambda *_args: fallback,
    )
    assert view.show_progress("job-1", "attempt-1")

    assert view.show_failure("job-1", "attempt-1", ())
    assert primary.closed is True
    assert primary.deleted is True
    assert fallback.opened is True


def test_terminal_publication_failure_uses_retry_only_decision_dialog():
    decisions = []
    view = SequenceExportView(
        progress_dialog_factory=_Loop3Dialog,
        failure_dialog_factory=lambda *_args: decisions.append(_Loop3Dialog())
        or decisions[-1],
        fallback_failure_dialog_factory=_Loop3Dialog,
    )
    bus = _Bus()
    bus.events.export_completed = _FailOnceSignal(RuntimeError("observer"))
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    outcome = ExportExecutionOutcome(
        True,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (
            ExportTargetResult("mes", "Result", "ok"),
            ExportTargetResult("excel", "Result", "ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )

    assert controller.handle_worker_completed(outcome) is False
    assert [text for text, _button in decisions[-1].buttons] == ["重试"]


def test_failed_terminal_publication_restores_retry_and_ignore_after_delivery():
    decisions = []
    view = SequenceExportView(
        progress_dialog_factory=_Loop3Dialog,
        failure_dialog_factory=lambda *_args: decisions.append(_Loop3Dialog())
        or decisions[-1],
        fallback_failure_dialog_factory=_Loop3Dialog,
    )
    bus = _Bus()
    bus.events.export_failed = _FailOnceSignal(RuntimeError("observer"))
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    outcome = ExportExecutionOutcome(
        False,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (ExportTargetResult("mes", "Result", "ok"),),
        (ExportTargetFailure("excel", "Result", "locked"),),
        (),
        (0,),
        (1,),
    )

    assert controller.handle_worker_failed(outcome) is False
    assert [text for text, _button in decisions[-1].buttons] == ["重试"]
    assert controller.retry_pending_terminal_publication(
        work.job_id, attempt.attempt_id
    )
    assert [text for text, _button in decisions[-1].buttons] == [
        "重试",
        "忽略",
    ]


def test_worker_accepts_only_service_attested_dirty_target_provenance():
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, message="mes"
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: ExportResult(True, "csv"),
    )
    job = ExportJob.from_request(_request())
    trusted = service.execute_record_job(job, "attempt-1")
    forged = (
        replace(
            trusted,
            dirty_targets=(
                SpoolTarget.create(
                    "Result",
                    {"fast_mode": True},
                    "C:/attacker/result.xlsx",
                    "C:/out/.spool",
                ),
            ),
        ),
        replace(trusted, dirty_target_indices=(0,)),
        replace(
            trusted,
            dirty_targets=trusted.dirty_targets + trusted.dirty_targets,
        ),
        replace(trusted, dirty_provenance=object()),
    )

    for outcome in (trusted,) + forged:
        worker = SequenceExportWorker(
            job,
            "attempt-1",
            execute=lambda *_args, outcome=outcome: outcome,
            validate_dirty_checkpoint=service.validate_dirty_checkpoint,
        )
        completed = []
        failed = []
        worker.completed.connect(completed.append)
        worker.failed.connect(failed.append)
        worker.run()
        if outcome is trusted:
            assert len(completed) == 1
            assert failed == []
        else:
            assert completed == []
            assert len(failed) == 1

    direct_request = ExportRequested(
        job.job_id,
        job.record_id,
        mutable_export_value(job.result_snapshot),
        (
            mutable_export_value(job.target_configurations[0]),
            {
                "type": "excel",
                "config_name": "Result",
                "configuration": {"fast_mode": False},
            },
        ),
    )
    direct_worker = SequenceExportWorker(
        ExportJob.from_request(direct_request),
        "attempt-1",
        execute=lambda *_args: trusted,
        validate_dirty_checkpoint=service.validate_dirty_checkpoint,
    )
    direct_completed = []
    direct_failed = []
    direct_worker.completed.connect(direct_completed.append)
    direct_worker.failed.connect(direct_failed.append)
    direct_worker.run()
    assert direct_completed == []
    assert len(direct_failed) == 1


def test_dirty_checkpoint_freezes_dynamic_paths_and_retry_skips_completed_side_effects():
    request = _request()
    request = ExportRequested(
        request.job_id,
        request.record_id,
        request.result_snapshot,
        (
            request.target_configurations[0],
            {
                "type": "excel",
                "config_name": "Fast-A",
                "configuration": {"fast_mode": True, "marker": "A"},
            },
            {
                "type": "excel",
                "config_name": "Direct-B",
                "configuration": {"fast_mode": False, "marker": "B"},
            },
        ),
    )
    calls = {"mes": 0, "csv": 0, "direct": 0, "paths": []}

    def resolve_path(configuration, **_kwargs):
        marker = configuration["marker"]
        calls["paths"].append(marker)
        return f"C:/out/{marker}-{len(calls['paths'])}.xlsx"

    def write_mes(*_args, **_kwargs):
        calls["mes"] += 1
        return SimpleNamespace(ok=True, message="mes")

    def write_csv(*_args, **_kwargs):
        calls["csv"] += 1
        return ExportResult(True, "csv")

    def write_direct(*_args, **_kwargs):
        calls["direct"] += 1
        if calls["direct"] == 1:
            return ExportResult(False, "locked")
        return ExportResult(True, "excel")

    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=write_mes,
        output_path_resolver=resolve_path,
        spool_dir_resolver=lambda _cfg, *, file_path: f"{file_path}.spool",
        csv_exporter=write_csv,
        excel_exporter=write_direct,
    )
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=_Bus(),
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    controller.handle_export_requested(request)
    first_work, first_attempt = submissions.pop()
    first_worker = SequenceExportWorker(
        first_work,
        first_attempt.attempt_id,
        execute=service.execute_record_job,
        validate_dirty_checkpoint=service.validate_dirty_checkpoint,
    )
    first_worker.completed.connect(controller.handle_worker_completed)
    first_worker.failed.connect(controller.handle_worker_failed)
    first_worker.run()

    assert controller.model.record_failure is not None
    assert controller.handle_retry_requested(
        RetryExportRequested(request.job_id, first_attempt.attempt_id)
    )
    retry_work, retry_attempt = submissions.pop()
    assert retry_work.target_indices == (2,)
    retry_worker = SequenceExportWorker(
        retry_work,
        retry_attempt.attempt_id,
        execute=service.execute_record_job,
        validate_dirty_checkpoint=service.validate_dirty_checkpoint,
    )
    retry_worker.completed.connect(controller.handle_worker_completed)
    retry_worker.failed.connect(controller.handle_worker_failed)
    retry_worker.run()

    assert calls["mes"] == 1
    assert calls["csv"] == 1
    assert calls["direct"] == 2
    assert calls["paths"].count("A") == 1
    assert controller.model.active_record_job is None


def test_dirty_checkpoint_rejects_fake_validator_and_reused_attempt_or_work_provenance():
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, message="mes"
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: ExportResult(True, "csv"),
    )
    job = ExportJob.from_request(_request())
    trusted = service.execute_record_job(job, "attempt-1")
    reused_attempt = replace(trusted, attempt_id="attempt-2")
    other_job = ExportJob.from_request(_request("job-2", "record-2"))
    reused_work = replace(
        trusted,
        job_id=other_job.job_id,
        record_id=other_job.record_id,
    )

    cases = (
        (
            job,
            "attempt-1",
            trusted,
            lambda *_args: True,
        ),
        (
            job,
            "attempt-2",
            reused_attempt,
            service.validate_dirty_checkpoint,
        ),
        (
            other_job,
            "attempt-1",
            reused_work,
            service.validate_dirty_checkpoint,
        ),
    )
    for work, attempt_id, outcome, validator in cases:
        worker = SequenceExportWorker(
            work,
            attempt_id,
            execute=lambda *_args, outcome=outcome: outcome,
            validate_dirty_checkpoint=validator,
        )
        completed = []
        failed = []
        worker.completed.connect(completed.append)
        worker.failed.connect(failed.append)
        worker.run()
        assert completed == []
        assert len(failed) == 1


def test_worker_fails_closed_when_dirty_checkpoint_has_no_trusted_validator():
    job = ExportJob.from_request(_request())
    target = SpoolTarget.create(
        "Result",
        {"fast_mode": True},
        "C:/attacker/result.xlsx",
        "C:/attacker/.spool",
    )
    outcome = ExportExecutionOutcome(
        True,
        job.job_id,
        "attempt-1",
        job.record_id,
        (
            ExportTargetResult("mes", "Result", "ok"),
            ExportTargetResult("excel", "Result", "ok"),
        ),
        (),
        (target,),
        (0, 1),
        (),
    )
    worker = SequenceExportWorker(
        job, "attempt-1", execute=lambda *_args: outcome
    )
    completed = []
    failed = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert completed == []
    assert len(failed) == 1


@pytest.mark.parametrize(
    "error", (RuntimeError("observer"), KeyboardInterrupt(), SystemExit())
)
def test_terminal_dispatch_retries_only_undelivered_canonical_recipient(error):
    bus = SequenceEventBus()
    calls = []
    fail_once = {"value": True}

    def first(message):
        calls.append(("first", message.attempt_id))
        return True

    def second(message):
        calls.append(("second", message.attempt_id))
        if fail_once["value"]:
            fail_once["value"] = False
            raise error
        return True

    bus.register_export_terminal_recipient("first", first, critical=True)
    bus.register_export_terminal_recipient("second", second, critical=True)
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    outcome = ExportExecutionOutcome(
        True,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (
            ExportTargetResult("mes", "Result", "ok"),
            ExportTargetResult("excel", "Result", "ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )

    assert controller.handle_worker_completed(outcome) is False
    assert [name for name, _attempt in calls] == ["first", "second"]
    assert controller.retry_pending_terminal_publication(
        work.job_id, attempt.attempt_id
    )
    assert [name for name, _attempt in calls] == [
        "first",
        "second",
        "second",
    ]
    assert controller.model.active_record_job is None


def test_noncritical_terminal_observer_failure_never_blocks_fifo():
    bus = SequenceEventBus()
    delivered = []
    bus.register_export_terminal_recipient(
        "workflow",
        lambda message: delivered.append(("workflow", message)) or True,
        critical=True,
    )
    bus.register_export_terminal_recipient(
        "public-observer",
        lambda _message: (_ for _ in ()).throw(SystemExit("observer")),
    )
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    outcome = ExportExecutionOutcome(
        True,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (
            ExportTargetResult("mes", "Result", "ok"),
            ExportTargetResult("excel", "Result", "ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )

    assert controller.handle_worker_completed(outcome)
    assert len(delivered) == 1
    assert controller.model.active_record_job is None


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class _Bus:
    def __init__(self):
        self.events = SimpleNamespace(
            export_completed=_Signal(),
            export_failed=_Signal(),
        )
        self.commands = SimpleNamespace(
            export_requested=_Signal(),
            retry_export_requested=_Signal(),
            ignore_export_failure_requested=_Signal(),
            cancel_export_requested=_Signal(),
        )


class _View:
    def __init__(self):
        self.started = []
        self.failures = []
        self.finished = []

    def show_progress(self, job_id, attempt_id):
        self.started.append((job_id, attempt_id))

    def show_failure(self, job_id, attempt_id, failures):
        self.failures.append((job_id, attempt_id, failures))

    def finish(self, job_id, attempt_id):
        self.finished.append((job_id, attempt_id))


class _ConnectableSignal:
    def __init__(self, *, fail_connect=False):
        self._slots = []
        self._fail_connect = fail_connect

    def connect(self, slot, *_args):
        if self._fail_connect:
            raise SystemExit("connect exploded")
        self._slots.append(slot)

    def emit(self, *args):
        for slot in tuple(self._slots):
            slot(*args)


class _SetupThread:
    def __init__(self, stage):
        self.stage = stage
        self.started = _ConnectableSignal(fail_connect=stage == "connect")
        self.finished = _ConnectableSignal()
        self._running = False

    def start(self):
        if self.stage == "start":
            raise KeyboardInterrupt("start exploded")
        self._running = True

    def quit(self):
        self._running = False
        self.finished.emit()

    def isRunning(self):
        return self._running

    def isFinished(self):
        return not self._running

    def deleteLater(self):
        pass


class _SetupWorker:
    def __init__(self, stage):
        self.stage = stage
        self.completed = _ConnectableSignal()
        self.failed = _ConnectableSignal()
        self.finished = _ConnectableSignal()

    def moveToThread(self, _thread):
        if self.stage == "move":
            raise SystemExit("move exploded")

    def run(self):
        pass

    def deleteLater(self):
        pass


def _request(job_id="job-1", record_id="record-1", *, value=1):
    return ExportRequested(
        job_id,
        record_id,
        {
            "record_id": record_id,
            "export_handoff": {
                "record_id": record_id,
                "sn": "SN-1",
                "product_model": "MODEL",
                "date_text": "2026/8/20 10:11:12",
                "analysis_items_data": {"SPL": {"type": "SPL", "result": value}},
                "analysis_result_dict": {"SPL": (True, value)},
                "analysis_config": {"display_sequence": ("SPL",)},
                "ok_ng_summary": (True, "OK"),
                "can_output_ok_ng": True,
            },
        },
        (
            {
                "type": "mes",
                "config_name": "Result",
                "configuration": {"save_mes_enabled": True},
            },
            {
                "type": "excel",
                "config_name": "Result",
                "configuration": {"fast_mode": True},
            },
        ),
    )


def _multi_target_request(job_id="job-multi", record_id="record-multi"):
    request = _request(job_id, record_id)
    return ExportRequested(
        request.job_id,
        request.record_id,
        request.result_snapshot,
        (
            request.target_configurations[0],
            {
                "type": "excel",
                "config_name": "Excel-A",
                "configuration": {"fast_mode": False, "marker": "A"},
            },
            {
                "type": "excel",
                "config_name": "Excel-B",
                "configuration": {"fast_mode": False, "marker": "B"},
            },
        ),
    )


def test_model_keeps_fifo_record_jobs_without_record_or_target_deduplication():
    model = SequenceExportModel(history_limit=3)
    first = model.enqueue_record(_request("job-1", "same", value=1))
    second = model.enqueue_record(_request("job-2", "same", value=2))

    assert first is not second
    assert model.begin_next_record_job().job_id == "job-1"
    model.complete_record_job("job-1", "attempt-1")
    assert model.begin_next_record_job().job_id == "job-2"
    assert model.active_record_job.result_snapshot["export_handoff"]["analysis_items_data"]["SPL"]["result"] == 2


def test_model_retry_retires_old_attempt_and_ignore_only_matches_current_identity():
    model = SequenceExportModel()
    model.enqueue_record(_request())
    model.begin_next_record_job()
    first = model.begin_record_attempt(lambda job_id, number: f"{job_id}-a{number}")
    model.fail_record_attempt(first.job_id, first.attempt_id, (("excel", "locked"),))

    assert model.retry_record_attempt("other", first.attempt_id, lambda *_: "bad") is None
    second = model.retry_record_attempt(
        first.job_id, first.attempt_id, lambda job_id, number: f"{job_id}-a{number}"
    )
    assert second.attempt_id.startswith("job-1-a2::")
    assert model.accept_worker_terminal(first.job_id, first.attempt_id) is False
    model.fail_record_attempt(second.job_id, second.attempt_id, (("excel", "locked"),))
    assert model.ignore_record_failure(second.job_id, "wrong") is None
    ignored = model.ignore_record_failure(second.job_id, second.attempt_id)
    assert ignored.job_id == "job-1"


@pytest.mark.parametrize(
    "factory",
    (
        lambda *_args: "",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("factory")),
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt("factory")),
        lambda *_args: (_ for _ in ()).throw(SystemExit("factory")),
    ),
)
def test_initial_attempt_id_failure_creates_recoverable_decision_and_fifo(factory):
    bus = _Bus()
    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=factory,
        connect_bus=False,
    )

    assert controller.handle_export_requested(_request("job-1", "record-1"))
    recovery = controller.model.active_record_attempt
    assert recovery is not None
    assert controller.model.record_failure is not None
    assert view.failures[-1][:2] == ("job-1", recovery.attempt_id)
    assert controller.handle_export_requested(_request("job-2", "record-2"))
    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested("job-1", recovery.attempt_id)
    )
    assert controller.model.active_record_job.job_id == "job-2"


@pytest.mark.parametrize(
    "bad_id",
    (
        "",
        RuntimeError("factory"),
        KeyboardInterrupt("factory"),
        SystemExit("factory"),
    ),
)
def test_retry_attempt_id_failure_keeps_old_decision_then_can_retry(bad_id):
    values = iter(("attempt-1", bad_id, "attempt-2"))

    def attempt_factory(*_args):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=_Bus(),
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=attempt_factory,
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    failure = ExportExecutionOutcome(
        False,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (ExportTargetResult("mes", "Result", "mes-ok"),),
        (ExportTargetFailure("excel", "Result", "locked"),),
        (),
        (0,),
        (1,),
    )
    assert controller.handle_worker_failed(failure)
    failure_count = len(view.failures)

    assert controller.handle_retry_requested(
        RetryExportRequested(work.job_id, attempt.attempt_id)
    ) is False
    assert controller.model.active_record_attempt == attempt
    assert controller.model.record_failure is not None
    assert len(view.failures) == failure_count + 1

    assert controller.handle_retry_requested(
        RetryExportRequested(work.job_id, attempt.attempt_id)
    )
    assert submissions[-1][1].attempt_id.startswith("attempt-2::")


def test_spool_dirty_generations_coalesce_only_the_latest_follow_up():
    model = SequenceExportModel()
    target = SpoolTarget.create("Result", {"fast_mode": True}, "C:/out/a.xlsx", "C:/out/.spool")

    assert model.mark_target_dirty(target) == 1
    active = model.begin_rebuild(target.key)
    assert active.generation == 1
    assert model.mark_target_dirty(target) == 2
    assert model.mark_target_dirty(target) == 3

    follow_up = model.complete_rebuild(active.job_id, active.attempt_id, succeeded=True)
    assert follow_up.generation == 3
    assert model.active_rebuild(target.key) is follow_up


def test_failed_rebuild_retries_same_job_with_new_attempt_then_ignore_is_terminal():
    model = SequenceExportModel()
    target = SpoolTarget.create("Result", {}, "C:/out/a.xlsx", "C:/out/.spool")
    model.mark_target_dirty(target)
    first = model.begin_rebuild(target.key)
    model.complete_rebuild(
        first.job_id,
        first.attempt_id,
        succeeded=False,
        failure=(("Result", "locked"),),
    )

    second = model.retry_rebuild(first.job_id, first.attempt_id)
    assert second.job_id == first.job_id
    assert second.attempt_id != first.attempt_id
    model.complete_rebuild(
        second.job_id,
        second.attempt_id,
        succeeded=False,
        failure=(("Result", "locked"),),
    )
    assert model.ignore_rebuild_failure(second.job_id, "wrong") is False
    assert model.ignore_rebuild_failure(second.job_id, second.attempt_id) is True
    assert model.dirty_target_keys() == ()


def test_rebuild_retry_keeps_failed_generation_target_then_uses_latest_follow_up():
    model = SequenceExportModel()
    v1 = SpoolTarget.create(
        "Result", {"version": 1}, "C:/out/a.xlsx", "C:/out/spool-v1"
    )
    v2 = SpoolTarget.create(
        "Result", {"version": 2}, "C:/out/a.xlsx", "C:/out/spool-v2"
    )
    v3 = SpoolTarget.create(
        "Result", {"version": 3}, "C:/out/a.xlsx", "C:/out/spool-v3"
    )
    model.mark_target_dirty(v1)
    first = model.begin_rebuild(v1.key)
    model.mark_target_dirty(v2)
    model.complete_rebuild(
        first.job_id,
        first.attempt_id,
        succeeded=False,
        failure=(("Result", "locked"),),
    )
    model.mark_target_dirty(v3)

    retried = model.retry_rebuild(first.job_id, first.attempt_id)

    assert retried.job_id == first.job_id
    assert retried.generation == first.generation == 1
    assert retried.target.configuration["version"] == 1
    assert retried.target.spool_dir.endswith("spool-v1")
    rebuilt_versions = []
    service = SequenceExportService(
        spool_builder=lambda configuration, **_kwargs: (
            rebuilt_versions.append(configuration["version"])
            or ExportResult(True, "saved")
        )
    )
    assert service.execute_rebuild_job(retried).ok is True
    follow_up = model.complete_rebuild(
        retried.job_id, retried.attempt_id, succeeded=True
    )
    assert follow_up.generation == 3
    assert follow_up.target.configuration["version"] == 3
    assert follow_up.target.spool_dir.endswith("spool-v3")
    assert service.execute_rebuild_job(follow_up).ok is True
    assert rebuilt_versions == [1, 3]


def test_service_preserves_mes_then_excel_target_order_and_marks_spool_dirty():
    calls = []
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda cfg, *, sn, label, logger: (
            calls.append(("mes", sn, label)) or SimpleNamespace(ok=True, message="mes-ok")
        ),
        output_path_resolver=lambda cfg, *, product_model: "C:/out/result.xlsx",
        spool_dir_resolver=lambda cfg, *, file_path: "C:/out/.spool",
        csv_exporter=lambda cfg, **kwargs: (
            calls.append(("excel-spool", kwargs["sn"])) or ExportResult(True, "csv-ok")
        ),
    )
    job = ExportJob.from_request(_request())

    outcome = service.execute_record_job(job, "attempt-1")

    assert outcome.ok is True
    assert [call[0] for call in calls] == ["mes", "excel-spool"]
    assert [result.target_type for result in outcome.target_results] == ["mes", "excel"]
    assert len(outcome.dirty_targets) == 1


def test_service_attempts_every_excel_after_mes_and_checkpoints_each_target():
    calls = []

    def excel_exporter(configuration, **_kwargs):
        marker = configuration["marker"]
        calls.append(marker)
        return ExportResult(marker != "A", f"{marker}-result")

    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: (
            calls.append("MES") or SimpleNamespace(ok=True, message="mes-ok")
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        excel_exporter=excel_exporter,
    )

    outcome = service.execute_record_job(
        ExportJob.from_request(_multi_target_request()), "attempt-1"
    )

    assert calls == ["MES", "A", "B"]
    assert outcome.completed_target_indices == (0, 2)
    assert outcome.failed_target_indices == (1,)
    assert [result.config_name for result in outcome.target_results] == [
        "Result",
        "Excel-B",
    ]
    assert [failure.config_name for failure in outcome.failures] == ["Excel-A"]


def test_multi_target_retry_runs_only_failures_and_publishes_full_ordered_results():
    calls = []
    attempts = {"A": 0}

    def excel_exporter(configuration, **_kwargs):
        marker = configuration["marker"]
        calls.append(marker)
        attempts[marker] = attempts.get(marker, 0) + 1
        return ExportResult(
            marker != "A" or attempts[marker] > 2,
            f"{marker}-{attempts[marker]}",
        )

    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: (
            calls.append("MES") or SimpleNamespace(ok=True, message="mes-ok")
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        excel_exporter=excel_exporter,
    )
    submissions = []
    bus = _Bus()
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_multi_target_request())
    first_work, first_attempt = submissions.pop()
    first_outcome = service.execute_record_job(
        first_work, first_attempt.attempt_id
    )
    assert controller.handle_worker_failed(first_outcome)

    assert controller.handle_retry_requested(
        RetryExportRequested("job-multi", first_attempt.attempt_id)
    )
    second_work, second_attempt = submissions.pop()
    assert second_work.target_indices == (1,)
    assert controller.handle_worker_failed(
        service.execute_record_job(second_work, second_attempt.attempt_id)
    )
    assert controller.handle_retry_requested(
        RetryExportRequested("job-multi", second_attempt.attempt_id)
    )
    third_work, third_attempt = submissions.pop()
    assert third_work.target_indices == (1,)
    assert controller.handle_worker_completed(
        service.execute_record_job(third_work, third_attempt.attempt_id)
    )

    assert calls == ["MES", "A", "B", "A", "A"]
    terminal = bus.events.export_completed.values[-1]
    assert [item["config_name"] for item in terminal.target_results] == [
        "Result",
        "Excel-A",
        "Excel-B",
    ]


def test_multi_target_ignore_publishes_successes_and_explicit_ignored_failure():
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, message="mes-ok"
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        excel_exporter=lambda configuration, **_kwargs: ExportResult(
            configuration["marker"] != "A", configuration["marker"]
        ),
    )
    submissions = []
    bus = _Bus()
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_multi_target_request())
    work, attempt = submissions.pop()
    assert controller.handle_worker_failed(
        service.execute_record_job(work, attempt.attempt_id)
    )

    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested("job-multi", attempt.attempt_id)
    )
    terminal = bus.events.export_completed.values[-1]
    assert [item["config_name"] for item in terminal.target_results] == [
        "Result",
        "Excel-A",
        "Excel-B",
    ]
    assert terminal.target_results[1]["ignored"] is True


@pytest.mark.parametrize("indices", ((2,), (1,), (0, 0)))
def test_worker_rejects_hostile_record_checkpoint_indices(indices):
    outcome = ExportExecutionOutcome(
        True,
        "job-1",
        "attempt-1",
        "record-1",
        (ExportTargetResult("mes", "Result", "ok"),),
        (),
        (),
        indices,
    )
    worker = SequenceExportWorker(
        ExportJob.from_request(_multi_target_request("job-1", "record-1")),
        "attempt-1",
        execute=lambda *_args: outcome,
    )
    completed = []
    failed = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert completed == []
    assert len(failed) == 1
    assert failed[0].failures[0].target_type == "worker"


@pytest.mark.parametrize(
    "outcome",
    (
        ExportExecutionOutcome(
            True,
            "job-1",
            "attempt-1",
            "record-1",
            (
                ExportTargetResult("excel", "Excel-A", "excel-a-ok"),
                ExportTargetResult("mes", "Result", "mes-ok"),
                ExportTargetResult("excel", "Excel-B", "excel-b-ok"),
            ),
            (),
            (),
            (1, 0, 2),
            (),
        ),
        ExportExecutionOutcome(
            False,
            "job-1",
            "attempt-1",
            "record-1",
            (ExportTargetResult("mes", "Result", "mes-ok"),),
            (ExportTargetFailure("excel", "Excel-A", "locked"),),
            (),
            (0,),
            (1,),
        ),
    ),
)
def test_worker_rejects_out_of_order_or_partial_excel_checkpoint(outcome):
    worker = SequenceExportWorker(
        ExportJob.from_request(_multi_target_request("job-1", "record-1")),
        "attempt-1",
        execute=lambda *_args: outcome,
    )
    completed = []
    failed = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert completed == []
    assert len(failed) == 1
    assert failed[0].failures[0].target_type == "worker"


def test_service_attempt_is_failed_with_frozen_semantic_error_text():
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("MES unavailable")),
    )

    outcome = service.execute_record_job(_request_job := ExportJob.from_request(_request()), "attempt-1")

    assert outcome.ok is False
    assert outcome.job_id == _request_job.job_id
    assert outcome.failures[0].target_type == "mes"
    assert "MES unavailable" in outcome.failures[0].message


def test_worker_emits_exactly_one_attempt_tagged_terminal():
    outcome = ExportExecutionOutcome(
        True,
        "job-1",
        "attempt-1",
        "record-1",
        (
            ExportTargetResult("mes", "Result", "mes-ok"),
            ExportTargetResult("excel", "Result", "excel-ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )
    worker = SequenceExportWorker(
        ExportJob.from_request(_request()),
        "attempt-1",
        execute=lambda _job, _attempt: outcome,
    )
    completed = []
    failed = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)

    worker.run()
    worker.run()

    assert len(completed) == 1
    assert (completed[0].job_id, completed[0].attempt_id) == (
        "job-1",
        "attempt-1",
    )
    assert failed == []


@pytest.mark.parametrize(
    "execute",
    (
        lambda *_: (_ for _ in ()).throw(RuntimeError("ordinary")),
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("interrupt")),
        lambda *_: (_ for _ in ()).throw(SystemExit("exit")),
        lambda *_: (_ for _ in ()).throw(
            type(
                "HostileTextError",
                (BaseException,),
                {"__str__": lambda self: (_ for _ in ()).throw(SystemExit())},
            )()
        ),
        lambda *_: type(
            "HostileOutcome",
            (),
            {
                "job_id": "job-1",
                "attempt_id": "attempt-1",
                "ok": property(
                    lambda self: (_ for _ in ()).throw(SystemExit("hostile ok"))
                ),
            },
        )(),
        lambda *_: SimpleNamespace(
            ok=True,
            job_id="job-1",
            attempt_id="attempt-1",
            record_id="record-1",
            target_results=(
                type(
                    "HostileResult",
                    (),
                    {
                        "target_type": "excel",
                        "config_name": "Result",
                        "message": property(
                            lambda self: (_ for _ in ()).throw(
                                KeyboardInterrupt("hostile message")
                            )
                        ),
                    },
                )(),
            ),
            failures=(),
            dirty_targets=(),
        ),
    ),
)
def test_worker_freezes_every_base_exception_as_one_matching_failure(execute):
    worker = SequenceExportWorker(
        ExportJob.from_request(_request()),
        "attempt-1",
        execute=execute,
    )
    completed = []
    failed = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)

    worker.run()
    worker.run()

    assert completed == []
    assert len(failed) == 1
    outcome = failed[0]
    assert (outcome.job_id, outcome.attempt_id) == ("job-1", "attempt-1")
    assert len(outcome.failures[0].message) <= 1024


@pytest.mark.parametrize("stage", ("thread", "worker", "move", "connect", "start"))
def test_controller_setup_base_exception_becomes_retryable_decision_and_fifo_continues(stage):
    def thread_factory():
        if stage == "thread":
            raise KeyboardInterrupt("thread exploded")
        return _SetupThread(stage)

    def worker_factory(*_args, **_kwargs):
        if stage == "worker":
            raise SystemExit("worker exploded")
        return _SetupWorker(stage)

    bus = _Bus()
    view = _View()
    continued = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        thread_factory=thread_factory,
        worker_factory=worker_factory,
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )

    assert controller.handle_export_requested(_request("job-1", "record-1"))
    failed_attempt = controller.model.active_record_attempt
    assert controller.model.record_failure is not None
    assert view.failures[-1][:2] == ("job-1", failed_attempt.attempt_id)
    assert controller.handle_export_requested(_request("job-2", "record-2"))
    controller._submit_attempt_port = (
        lambda job, attempt: continued.append((job, attempt))
    )

    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested("job-1", failed_attempt.attempt_id)
    )
    assert continued[-1][0].job_id == "job-2"


def test_controller_terminal_base_exception_becomes_current_failed_decision():
    bus = _Bus()
    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    job, attempt = submissions.pop()

    class _HostileCompleted:
        job_id = job.job_id
        attempt_id = attempt.attempt_id
        dirty_targets = ()

        @property
        def target_results(self):
            raise KeyboardInterrupt("terminal exploded")

    assert controller.handle_worker_completed(_HostileCompleted()) is False
    assert controller.model.record_failure is not None
    assert view.failures[-1][:2] == (job.job_id, attempt.attempt_id)
    assert bus.events.export_completed.values == []


def test_rebuild_terminal_base_exception_keeps_matching_retry_decision():
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=_Bus(),
        submit_attempt=lambda *_args: None,
        connect_bus=False,
        debounce_ms=0,
    )
    target = SpoolTarget.create("Result", {}, "C:/out/a.xlsx", "C:/out/.spool")
    controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()
    rebuild = controller.model.active_rebuild(target.key)

    class _HostileFailure:
        job_id = rebuild.job_id
        attempt_id = rebuild.attempt_id

        @property
        def failures(self):
            raise SystemExit("hostile rebuild terminal")

    assert controller.handle_worker_failed(_HostileFailure()) is False
    assert controller.handle_retry_requested(
        RetryExportRequested(rebuild.job_id, rebuild.attempt_id)
    )



def test_controller_owns_detached_thread_handles_per_instance_not_module_global():
    source = Path(
        SequenceExportController.__module__.replace(".", "/") + ".py"
    )
    if not source.exists():
        source = (
            Path(__file__).resolve().parents[2]
            / "ui"
            / "sequence"
            / "sequence_export_controller.py"
        )
    text = source.read_text(encoding="utf-8")
    assert "_DETACHED_EXPORT_THREADS" not in text


def test_controller_fifo_retry_ignore_and_stale_worker_terminals():
    bus = _Bus()
    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )

    assert controller.handle_export_requested(_request("job-1", "record-1")) is True
    assert controller.handle_export_requested(_request("job-2", "record-2")) is True
    first_job, first_attempt = submissions.pop(0)
    controller.handle_worker_failed(
        SimpleNamespace(
            job_id=first_job.job_id,
            attempt_id=first_attempt.attempt_id,
            record_id=first_job.record_id,
            failures=(("excel", "locked"),),
            target_results=(),
            dirty_targets=(),
        )
    )
    assert len(bus.events.export_failed.values) == 1
    assert controller.handle_retry_requested(
        RetryExportRequested(first_job.job_id, first_attempt.attempt_id)
    )
    retried_job, retried_attempt = submissions.pop(0)
    assert retried_attempt.attempt_id.startswith("job-1-a2::")
    assert retried_attempt.attempt_number == 2
    assert controller.handle_worker_completed(
        SimpleNamespace(
            job_id=first_job.job_id,
            attempt_id=first_attempt.attempt_id,
            record_id=first_job.record_id,
            failures=(),
            target_results=(),
            dirty_targets=(),
        )
    ) is False
    controller.handle_worker_failed(
        SimpleNamespace(
            job_id=retried_job.job_id,
            attempt_id=retried_attempt.attempt_id,
            record_id=retried_job.record_id,
            failures=(("excel", "still locked"),),
            target_results=(),
            dirty_targets=(),
        )
    )
    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested(retried_job.job_id, retried_attempt.attempt_id)
    )
    assert isinstance(bus.events.export_completed.values[-1], ExportCompleted)
    assert submissions[-1][0].job_id == "job-2"


def test_record_attempt_failure_seals_first_terminal_until_retry_or_ignore():
    bus = _Bus()
    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_request("job-1", "record-1"))
    job, attempt = submissions.pop()
    failure = SimpleNamespace(
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        record_id=job.record_id,
        failures=(("excel", "locked"),),
        target_results=(),
        dirty_targets=(),
    )

    assert controller.handle_worker_failed(failure) is True
    assert controller.handle_worker_failed(failure) is False
    assert controller.handle_worker_completed(
        SimpleNamespace(
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            record_id=job.record_id,
            failures=(),
            target_results=(),
            dirty_targets=(),
        )
    ) is False
    assert len(bus.events.export_failed.values) == 1
    assert bus.events.export_completed.values == []
    assert controller.model.active_record_job.job_id == job.job_id


def test_rebuild_failure_decision_is_not_hidden_by_new_record_progress():
    bus = _Bus()
    view = _View()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
        debounce_ms=0,
    )
    target = SpoolTarget.create("Result", {}, "C:/out/a.xlsx", "C:/out/.spool")
    controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()
    rebuild, _attempt = submissions.pop()
    controller.handle_worker_failed(
        SimpleNamespace(
            job_id=rebuild.job_id,
            attempt_id=rebuild.attempt_id,
            failures=(("Result", "locked"),),
            target_results=(),
            dirty_targets=(),
        )
    )
    failure_identity = view.failures[-1][:2]
    submissions.clear()

    assert controller.handle_export_requested(_request("job-2", "record-2"))
    assert submissions == []
    assert view.failures[-1][:2] == failure_identity
    assert view.started[-1] == failure_identity

    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested(*failure_identity)
    )
    assert submissions[-1][0].job_id == "job-2"
    assert view.started[-1][0] == "job-2"


def test_record_retry_resumes_after_completed_mes_without_repeating_side_effect():
    calls = []
    excel_attempts = iter((ExportResult(False, "locked"), ExportResult(True, "saved")))
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: (
            calls.append("MES") or SimpleNamespace(ok=True, message="mes")
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: (
            calls.append("Excel") or next(excel_attempts)
        ),
    )
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=_Bus(),
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    first_job, first_attempt = submissions.pop()
    first_outcome = service.execute_record_job(
        first_job, first_attempt.attempt_id
    )
    assert first_outcome.ok is False
    controller.handle_worker_failed(first_outcome)
    controller.handle_retry_requested(
        RetryExportRequested(first_job.job_id, first_attempt.attempt_id)
    )
    retried_job, retried_attempt = submissions.pop()

    second_outcome = service.execute_record_job(
        retried_job, retried_attempt.attempt_id
    )

    assert second_outcome.ok is True
    assert calls == ["MES", "Excel", "Excel"]


def test_record_retry_resumes_across_multiple_direct_and_spool_targets_in_order():
    calls = []
    spool_results = iter(
        (ExportResult(False, "locked"), ExportResult(True, "spooled"))
    )
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: (
            calls.append("MES") or SimpleNamespace(ok=True, message="mes")
        ),
        output_path_resolver=lambda cfg, **_kwargs: f"C:/out/{cfg['name']}.xlsx",
        spool_dir_resolver=lambda cfg, **_kwargs: f"C:/spool/{cfg['name']}",
        excel_exporter=lambda cfg, **_kwargs: (
            calls.append(cfg["name"]) or ExportResult(True, "saved")
        ),
        csv_exporter=lambda cfg, **_kwargs: (
            calls.append(cfg["name"]) or next(spool_results)
        ),
    )
    request = _request()
    request = ExportRequested(
        request.job_id,
        request.record_id,
        request.result_snapshot,
        (
            request.target_configurations[0],
            {
                "type": "excel",
                "config_name": "Direct-1",
                "configuration": {"fast_mode": False, "name": "Direct-1"},
            },
            {
                "type": "excel",
                "config_name": "Spool",
                "configuration": {"fast_mode": True, "name": "Spool"},
            },
            {
                "type": "excel",
                "config_name": "Direct-2",
                "configuration": {"fast_mode": False, "name": "Direct-2"},
            },
        ),
    )
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=_Bus(),
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(request)
    work, attempt = submissions.pop()
    failure = service.execute_record_job(work, attempt.attempt_id)
    assert failure.ok is False
    controller.handle_worker_failed(failure)
    controller.handle_retry_requested(
        RetryExportRequested(work.job_id, attempt.attempt_id)
    )
    retry_work, retry_attempt = submissions.pop()

    success = service.execute_record_job(
        retry_work, retry_attempt.attempt_id
    )

    assert success.ok is True
    assert calls == ["MES", "Direct-1", "Spool", "Direct-2", "Spool"]
    assert len(success.dirty_targets) == 1


def test_controller_presents_rebuild_failure_and_routes_retry_ignore_commands():
    bus = _Bus()
    view = _View()
    submissions = []
    model = SequenceExportModel()
    controller = SequenceExportController(
        model,
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
        debounce_ms=0,
    )
    target = SpoolTarget.create("Result", {}, "C:/out/a.xlsx", "C:/out/.spool")
    assert controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()
    first, _attempt = submissions.pop()
    failure = SimpleNamespace(
        ok=False,
        job_id=first.job_id,
        attempt_id=first.attempt_id,
        record_id=first.target.file_path,
        failures=(("Result", "locked"),),
        target_results=(),
        dirty_targets=(),
    )

    assert controller.handle_worker_failed(failure)
    assert view.failures[-1][:2] == (first.job_id, first.attempt_id)
    assert controller.handle_retry_requested(
        RetryExportRequested(first.job_id, first.attempt_id)
    )
    second, retry_attempt = submissions.pop()
    assert second.job_id == first.job_id
    assert second.attempt_id != first.attempt_id
    assert retry_attempt.attempt_number == 2
    failure.attempt_id = second.attempt_id
    assert controller.handle_worker_failed(failure)
    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested(second.job_id, second.attempt_id)
    )
    assert bus.events.export_completed.values == []


def test_default_controller_path_runs_blocking_service_on_qthread():
    bus = SequenceEventBus()
    view = _View()
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: SimpleNamespace(ok=True, message="mes"),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: ExportResult(True, "csv"),
    )
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        service=service,
    )
    completed = QSignalSpy(bus.events.export_completed)

    bus.commands.export_requested.emit(_request())

    assert completed.wait(3000)
    event = completed[0][0]
    assert isinstance(event, ExportCompleted)
    assert (event.job_id, event.record_id) == ("job-1", "record-1")
    _QAPP.processEvents()
    thread = controller._worker_thread
    if thread is not None:
        assert thread.wait(3000)
    controller.disconnect()


def test_real_qthread_terminal_uses_guarded_workflow_port_before_label_continuation():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.phase = WorkflowPhase.RESULT_EXPORTING
    workflow_model.active_job_id = "job-1"
    workflow_model.export_record_id = "record-1"
    workflow_model.export_continuation = ExportContinuation.LABEL_COMMIT
    workflow_model.active_label_command_id = "label-1"
    workflow_model.active_label_record_id = "record-1"
    workflow_model.active_label = "OK"
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        export_decision_requires_terminal=True,
    )
    service = SequenceExportService(
        mes_validator=lambda _cfg: (True, ""),
        mes_writer=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, message="mes"
        ),
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: ExportResult(True, "csv"),
    )
    export = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=service
    )
    commits = []
    bus.register_workflow_continuation_recipient(
        "workflow-state", "export-test-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "export-test-label",
        lambda message: commits.append(message) or True,
    )
    hostile_raw_calls = []

    def hostile_raw(_message):
        hostile_raw_calls.append(True)
        raise SystemExit("raw Qt terminal must not be canonical")

    bus.events.export_completed.connect(hostile_raw)
    bus.commands.export_requested.emit(_request())

    deadline = time.monotonic() + 3
    while not commits and time.monotonic() < deadline:
        _QAPP.processEvents()
    assert commits
    assert workflow_model.phase is WorkflowPhase.LABEL_COMMITTING
    assert hostile_raw_calls == []
    thread = export._worker_thread
    if thread is not None:
        assert thread.wait(3000)
    export.disconnect()
    workflow.disconnect()


def test_real_qthread_freezes_hostile_outcome_and_exits_with_failed_decision():
    class _HostileOutcomeService:
        def execute_record_job(self, job, attempt_id):
            return type(
                "HostileOutcome",
                (),
                {
                    "job_id": job.job_id,
                    "attempt_id": attempt_id,
                    "ok": property(
                        lambda self: (_ for _ in ()).throw(
                            SystemExit("hostile qthread outcome")
                        )
                    ),
                },
            )()

    bus = SequenceEventBus()
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=_HostileOutcomeService(),
    )
    failed = QSignalSpy(bus.events.export_failed)

    bus.commands.export_requested.emit(_request())

    assert failed.wait(3000)
    assert controller.model.record_failure is not None
    thread = controller._worker_thread
    if thread is not None:
        assert thread.wait(3000)
    controller.disconnect()


def test_disconnected_real_qthread_is_owned_per_controller_until_true_finish():
    entered = threading.Event()
    release = threading.Event()

    class _BlockingService:
        def execute_record_job(self, job, attempt_id):
            entered.set()
            release.wait(5)
            return SimpleNamespace(
                ok=True,
                job_id=job.job_id,
                attempt_id=attempt_id,
                record_id=job.record_id,
                target_results=(),
                failures=(),
                dirty_targets=(),
                completed_target_indices=job.target_indices,
            )

    first = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=SequenceEventBus(),
        service=_BlockingService(),
    )
    second = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=SequenceEventBus(),
    )
    try:
        assert first.handle_export_requested(_request())
        assert entered.wait(3)
        thread = first._worker_thread
        finished = QSignalSpy(thread.finished)

        first.disconnect()

        assert len(first._owned_thread_handles) == 1
        assert len(second._owned_thread_handles) == 0
        release.set()
        assert finished.wait(3000)
        _QAPP.processEvents()
        assert len(first._owned_thread_handles) == 0
        assert len(second._owned_thread_handles) == 0
    finally:
        release.set()
        thread = first._worker_thread
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        first.disconnect()
        second.disconnect()


def test_thread_finish_before_queued_terminal_does_not_start_pending_work():
    threads = []
    workers = []

    def thread_factory():
        thread = _SetupThread("ok")
        threads.append(thread)
        return thread

    def worker_factory(*_args, **_kwargs):
        worker = _SetupWorker("ok")
        workers.append(worker)
        return worker

    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=_Bus(),
        thread_factory=thread_factory,
        worker_factory=worker_factory,
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
        debounce_ms=0,
    )
    controller.handle_export_requested(_request())
    attempt = controller.model.active_record_attempt
    target = SpoolTarget.create(
        "Result", {}, "C:/out/a.xlsx", "C:/out/.spool"
    )
    controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()

    controller._worker_finished(threads[0], workers[0])

    assert len(threads) == 1
    assert len(controller._pending_worker_jobs) == 1

    assert controller.handle_worker_completed(
        SimpleNamespace(
            ok=True,
            job_id="job-1",
            attempt_id=attempt.attempt_id,
            record_id="record-1",
            target_results=(),
            failures=(),
            dirty_targets=(),
            completed_target_indices=(0,),
        )
    )
    assert len(threads) == 2
    assert controller._active_worker_identity[0].startswith("spool-rebuild-")


def test_formal_ignore_terminal_reaches_workflow_only_after_export_controller():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.phase = WorkflowPhase.ANALYZING
    workflow_model.active_analysis_id = "analysis-1"
    workflow_model.analysis_source_id = "record-1"
    workflow_model.analysis_record_id = "record-1"
    workflow_model.retained_record_id = "record-1"
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        job_id_factory=lambda: "job-1",
        export_decision_requires_terminal=True,
    )
    submissions = []
    export = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
    )
    snapshot = mutable_export_value(_request().result_snapshot)
    snapshot["export_handoff"]["analysis_config"] = {
        "Result": {"type": "Excel", "fast_mode": True}
    }

    assert workflow.handle_analysis_completed(
        AnalysisCompleted("analysis-1", "record-1", snapshot)
    )
    for _ in range(3):
        _QAPP.processEvents()
    job, attempt = submissions.pop()
    export.handle_worker_failed(
        SimpleNamespace(
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            record_id=job.record_id,
            failures=(("excel", "locked"),),
            target_results=(),
            dirty_targets=(),
        )
    )
    for _ in range(3):
        _QAPP.processEvents()
    assert workflow_model.export_failure_pending is True

    bus.commands.ignore_export_failure_requested.emit(
        IgnoreExportFailureRequested(job.job_id, attempt.attempt_id)
    )
    for _ in range(5):
        _QAPP.processEvents()

    assert workflow_model.phase is WorkflowPhase.IDLE
    assert workflow_model.active_job_id is None
    export.disconnect()
    workflow.disconnect()


def test_manual_label_commit_is_not_published_before_export_terminal():
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    workflow_model.awaiting_label = True
    labeled_snapshot = mutable_export_value(_request().result_snapshot)
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        job_id_factory=lambda: "job-1",
        export_decision_requires_terminal=True,
    )
    submissions = []
    export = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
    )
    labeled_snapshot["export_handoff"]["analysis_config"] = {
        "Result": {"type": "Excel", "fast_mode": True}
    }
    assert export.retain_result_snapshot("record-1", labeled_snapshot)
    commits = []
    bus.register_workflow_continuation_recipient(
        "workflow-state", "manual-label-test-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "manual-label-test",
        lambda message: commits.append(message) or True,
    )

    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("label-1", "record-1", "NG")
    )
    for _ in range(4):
        _QAPP.processEvents()
    assert commits == []
    job, attempt = submissions.pop()

    export.handle_worker_completed(
        SimpleNamespace(
            ok=True,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            record_id=job.record_id,
            target_results=(("excel", "saved"),),
            failures=(),
            dirty_targets=(),
        )
    )
    for _ in range(4):
        _QAPP.processEvents()

    assert len(commits) == 1
    assert (commits[0].record_id, commits[0].label) == ("record-1", "NG")
    export.disconnect()
    workflow.disconnect()


@pytest.mark.parametrize(
    ("analysis_config", "expected_phase", "expected_export_count", "expected_commit_count"),
    [
        (
            {"Result": {"type": "Excel", "fast_mode": True}},
            "RESULT_EXPORTING",
            1,
            0,
        ),
        ({}, "LABEL_COMMITTING", 0, 1),
    ],
    ids=["prepared-with-export", "prepared-without-export"],
)
def test_manual_label_preparation_keeps_idle_until_exact_prepared_terminal(
    analysis_config,
    expected_phase,
    expected_export_count,
    expected_commit_count,
):
    bus = SequenceEventBus()
    states = []
    commits = []
    capture_state = lambda message: states.append(message.new_phase) or True
    bus.register_workflow_continuation_recipient(
        "workflow-state",
        "manual-preparation-phase-state",
        capture_state,
    )
    bus.events.workflow_state_changed.connect(capture_state)
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "manual-preparation-phase-label",
        lambda message: commits.append(message) or True,
    )
    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus, service=SequenceExportService()
    )
    retained = mutable_export_value(_request().result_snapshot)
    retained["export_handoff"]["analysis_config"] = analysis_config
    assert export_owner.retain_result_snapshot("record-1", retained)
    model = SequenceWorkflowModel()
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    exports = QSignalSpy(bus.commands.export_requested)
    workflow = SequenceWorkflowController(
        model,
        bus,
        job_id_factory=lambda: "manual-job-1",
        preparation_id_factory=lambda: "manual-prepare-1",
    )

    assert workflow.handle_manual_label(
        ManualLabelRequested("manual-label-1", "record-1", "OK")
    )

    assert states == [expected_phase]
    assert model.phase.name == expected_phase
    assert len(exports) == expected_export_count
    assert len(commits) == expected_commit_count
    export_owner.disconnect()
    workflow.disconnect()


def test_manual_label_pending_is_idle_but_blocks_admission_and_cancels_formally():
    bus = SequenceEventBus()
    states = []
    capture_state = lambda message: states.append(message.new_phase) or True
    bus.register_workflow_continuation_recipient(
        "workflow-state",
        "manual-pending-cancel-state",
        capture_state,
    )
    bus.events.workflow_state_changed.connect(capture_state)
    bus.register_workflow_continuation_recipient(
        "manual-label-export-prepare",
        "manual-pending-holder",
        lambda _message: False,
    )
    model = SequenceWorkflowModel()
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    workflow = SequenceWorkflowController(
        model, bus, preparation_id_factory=lambda: "manual-prepare-pending"
    )

    assert workflow.handle_manual_label(
        ManualLabelRequested("manual-label-1", "record-1", "NG")
    )
    request = workflow._pending_export_preparation
    assert type(request) is PrepareManualLabelExportRequested
    assert model.phase is WorkflowPhase.IDLE
    assert states == []
    assert workflow.handle_manual_label(
        ManualLabelRequested("manual-label-2", "record-1", "OK")
    ) is False
    assert workflow._pending_export_preparation is request

    export_owner = SequenceExportController(
        SequenceExportModel(), _View(), bus=bus
    )
    assert workflow.handle_cancel_workflow(
        CancelWorkflowRequested("cancel-manual-preparation", 1, "cancel")
    )

    assert model.phase is WorkflowPhase.IDLE
    assert workflow._pending_export_preparation is None
    assert states == ["CANCELLING", "IDLE"]
    stale = ManualLabelExportPreparationFailed(
        request.request_id,
        request.command_id,
        request.record_id,
        request.label,
        request.workflow_generation,
        "late",
    )
    assert workflow.handle_manual_label_export_preparation_failed(stale) is False
    assert model.phase is WorkflowPhase.IDLE
    export_owner.disconnect()
    workflow.disconnect()


def test_facade_manual_label_click_only_posts_frozen_workflow_command():
    window = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(window)
    window.configuration_model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(store_wave_data=[1.0])
    )
    window.recording_model = RecordingModel()
    window.sequence_config = []
    window.count_board = SimpleNamespace(
        ok_btn=QPushButton("OK", window),
        ng_btn=QPushButton("NG", window),
    )
    window.workflow_model = SimpleNamespace(retained_record_id="record-1")
    window.recorded_signal_info = {"file_path": "record-1", "labels": "not_labeled"}
    window.recorded_path = "record-1"
    window.sequence_event_bus = SequenceEventBus(window)
    request_service = RecordingManualLabelRequestService(
        data_provider=lambda: window.configuration_model.data_struct.store_wave_data,
        sequence_config_provider=lambda: window.sequence_config,
        retained_record_id_provider=lambda: window.workflow_model.retained_record_id,
        recorded_signal_info_provider=lambda: window.recorded_signal_info,
        recorded_path_provider=lambda: window.recorded_path,
        ok_button=window.count_board.ok_btn,
        ng_button=window.count_board.ng_btn,
        publish=window.sequence_event_bus.commands.manual_label_requested.emit,
        present_warning=lambda _title, _text: None,
        command_id_factory=lambda: "manual-label-test",
    )
    window.recording_controller = SequenceRecordingController(
        window.recording_model,
        window.sequence_event_bus,
        manual_label_request_service=request_service,
        connect_queued=False,
    )
    commands = QSignalSpy(window.sequence_event_bus.commands.manual_label_requested)
    window.count_board.ok_btn.clicked.connect(window.clicked_ok_or_ng)

    window.count_board.ok_btn.click()

    assert len(commands) == 1
    command = commands[0][0]
    assert isinstance(command, ManualLabelRequested)
    assert (command.record_id, command.label) == ("record-1", "OK")
    assert window.recorded_signal_info["labels"] == "not_labeled"


def test_manual_label_for_new_record_does_not_rewrite_stale_analysis_snapshot():
    stale = mutable_export_value(_request("job-a", "record-A").result_snapshot)
    stale["export_targets"] = mutable_export_value(
        _request("job-a", "record-A").target_configurations
    )
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-B"
    workflow_model.awaiting_label = True
    workflow_model.analysis_result_snapshot = stale
    bus = SequenceEventBus()
    export_owner = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        service=SequenceExportService(),
    )
    assert export_owner.retain_result_snapshot("record-A", stale)
    commits = []
    bus.register_workflow_continuation_recipient(
        "workflow-state", "new-record-test-state", lambda _message: True
    )
    bus.register_workflow_continuation_recipient(
        "label-commit",
        "new-record-test-label",
        lambda message: commits.append(message) or True,
    )
    exports = QSignalSpy(bus.commands.export_requested)
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
    )

    frozen = export_owner.build_labeled_result("record-B", "OK")
    assert frozen["export_targets"] == ()
    assert "analysis_items_data" not in frozen["export_handoff"]

    assert workflow.handle_manual_label(
        ManualLabelRequested("label-B", "record-B", "OK")
    )
    assert len(exports) == 0
    assert len(commits) == 1
    assert workflow_model.labeled_result_snapshot["record_id"] == "record-B"
    assert "analysis_items_data" not in workflow_model.labeled_result_snapshot[
        "export_handoff"
    ]
    export_owner.disconnect()
    workflow.disconnect()


def test_manual_label_current_snapshot_freezes_only_excel_targets_never_mes():
    current = mutable_export_value(
        _request("job-current", "record-current").result_snapshot
    )
    analysis_config = {
        "display_sequence": ("Result",),
        "Result": {
            "type": "Excel",
            "save_mes_enabled": True,
            "fast_mode": True,
        },
    }
    current["export_handoff"]["analysis_config"] = analysis_config
    export_owner = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=SequenceEventBus(),
        service=SequenceExportService(),
    )
    assert export_owner.retain_result_snapshot("record-current", current)

    snapshot = export_owner.build_labeled_result("record-current", "NG")

    assert snapshot["record_id"] == "record-current"
    assert snapshot["export_handoff"]["record_id"] == "record-current"
    assert [target["type"] for target in snapshot["export_targets"]] == [
        "excel"
    ]
    export_owner.disconnect()


def test_view_uses_non_blocking_window_modal_open(monkeypatch):
    opened = []

    class _Dialog:
        def __init__(self, *_args, **_kwargs):
            self.finished = SimpleNamespace(connect=lambda callback: None)

        def setWindowTitle(self, _value):
            pass

        def setWindowModality(self, value):
            assert value == Qt.WindowModal

        def setLabelText(self, _value):
            pass

        def setCancelButton(self, _value):
            pass

        def setRange(self, _minimum, _maximum):
            pass

        def open(self):
            opened.append(True)

        def close(self):
            pass

        def deleteLater(self):
            pass

    view = SequenceExportView(progress_dialog_factory=_Dialog)
    view.show_progress("job-1", "attempt-1")

    assert opened == [True]


def test_failure_view_falls_back_nonblocking_and_preserves_recovery_identity():
    opened = []

    class _DecisionDialog:
        Warning = 1
        AcceptRole = 2
        RejectRole = 3

        def __init__(self, *_args):
            self.buttonClicked = _ConnectableSignal()
            self.buttons = []

        def setWindowModality(self, value):
            assert value == Qt.WindowModal

        def setIcon(self, _value):
            pass

        def setWindowTitle(self, _value):
            pass

        def setText(self, _value):
            pass

        def setInformativeText(self, _value):
            pass

        def addButton(self, text, _role):
            button = object()
            self.buttons.append((text, button))
            return button

        def setDefaultButton(self, _button):
            pass

        def open(self):
            opened.append(True)

        def close(self):
            pass

        def deleteLater(self):
            pass

    fallback = _DecisionDialog()
    view = SequenceExportView(
        progress_dialog_factory=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("progress")
        ),
        failure_dialog_factory=lambda *_args: (_ for _ in ()).throw(
            KeyboardInterrupt("primary")
        ),
        fallback_failure_dialog_factory=lambda *_args: fallback,
    )
    retries = []
    view.retry_requested.connect(retries.append)

    assert view.show_progress("job-1", "attempt-1") is False
    assert view.show_failure(
        "job-1",
        "attempt-1",
        (ExportTargetFailure("excel", "Result", "locked"),),
    )
    assert opened == [True]
    fallback.buttonClicked.emit(fallback.buttons[0][1])
    assert (retries[0].job_id, retries[0].attempt_id) == (
        "job-1",
        "attempt-1",
    )


def test_failure_view_keeps_recovery_pending_after_open_failure_and_destroy():
    calls = {"count": 0}

    def failing_factory(*_args):
        calls["count"] += 1
        raise SystemExit("open failed")

    view = SequenceExportView(
        progress_dialog_factory=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("progress")
        ),
        failure_dialog_factory=failing_factory,
        fallback_failure_dialog_factory=failing_factory,
    )
    view.show_progress("job-1", "attempt-1")

    assert view.show_failure("job-1", "attempt-1", ()) is False
    assert view.recovery_pending_identity == ("job-1", "attempt-1")
    assert view.show_failure("job-1", "attempt-1", ()) is False
    view._mark_destroyed()
    assert view.recovery_pending_identity == ("job-1", "attempt-1")


class _FailOnceSignal:
    def __init__(self, error):
        self.error = error
        self.calls = 0
        self.values = []

    def emit(self, value):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        self.values.append(value)


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("observer"),
        KeyboardInterrupt("observer"),
        SystemExit("observer"),
    ),
)
def test_completed_publication_failure_pauses_fifo_until_explicit_retry(error):
    bus = _Bus()
    bus.events.export_completed = _FailOnceSignal(error)
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_request("job-1", "record-1"))
    work, attempt = submissions.pop()
    controller.handle_export_requested(_request("job-2", "record-2"))
    outcome = ExportExecutionOutcome(
        True,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (
            ExportTargetResult("mes", "Result", "mes-ok"),
            ExportTargetResult("excel", "Result", "excel-ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )

    assert controller.handle_worker_completed(outcome) is False
    assert controller.pending_terminal_publication_identity == (
        "job-1",
        attempt.attempt_id,
    )
    assert controller.model.active_record_job.job_id == "job-1"
    assert submissions == []

    assert controller.retry_pending_terminal_publication(
        "job-1", attempt.attempt_id
    )
    assert bus.events.export_completed.calls == 2
    assert len(bus.events.export_completed.values) == 1
    assert controller.model.active_record_job.job_id == "job-2"
    assert submissions[-1][0].job_id == "job-2"


def test_worker_finish_cannot_resume_pending_rebuild_before_publication_retry():
    threads = []
    workers = []

    def thread_factory():
        thread = _SetupThread("ok")
        threads.append(thread)
        return thread

    def worker_factory(*_args, **_kwargs):
        worker = _SetupWorker("ok")
        workers.append(worker)
        return worker

    bus = _Bus()
    bus.events.export_completed = _FailOnceSignal(RuntimeError("observer"))
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        thread_factory=thread_factory,
        worker_factory=worker_factory,
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
        debounce_ms=0,
    )
    controller.handle_export_requested(_request())
    attempt = controller.model.active_record_attempt
    target = SpoolTarget.create(
        "Result", {}, "C:/out/result.xlsx", "C:/out/.spool"
    )
    controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()
    outcome = ExportExecutionOutcome(
        True,
        "job-1",
        attempt.attempt_id,
        "record-1",
        (
            ExportTargetResult("mes", "Result", "mes-ok"),
            ExportTargetResult("excel", "Result", "excel-ok"),
        ),
        (),
        (),
        (0, 1),
        (),
    )
    assert controller.handle_worker_completed(outcome) is False

    controller._worker_finished(threads[0], workers[0])

    assert len(threads) == 1
    assert controller.retry_pending_terminal_publication(
        "job-1", attempt.attempt_id
    )
    assert len(threads) == 2
    assert controller._active_worker_identity[0].startswith("spool-rebuild-")


def test_failed_publication_failure_keeps_exact_decision_until_republished():
    bus = _Bus()
    bus.events.export_failed = _FailOnceSignal(KeyboardInterrupt("observer"))
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _View(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
    )
    controller.handle_export_requested(_request())
    work, attempt = submissions.pop()
    outcome = ExportExecutionOutcome(
        False,
        work.job_id,
        attempt.attempt_id,
        work.record_id,
        (ExportTargetResult("mes", "Result", "mes-ok"),),
        (ExportTargetFailure("excel", "Result", "locked"),),
        (),
        (0,),
        (1,),
    )

    assert controller.handle_worker_failed(outcome) is False
    assert controller.model.record_failure is not None
    assert controller.handle_retry_requested(
        RetryExportRequested(work.job_id, attempt.attempt_id)
    )
    assert bus.events.export_failed.calls == 2
    assert controller.model.record_attempt_state.name == "FAILED_AWAITING_DECISION"
    assert submissions == []


def test_record_failure_decision_defers_dirty_rebuild_until_ignore_terminal():
    request = _request("job-spool", "record-spool")
    request = ExportRequested(
        request.job_id,
        request.record_id,
        request.result_snapshot,
        (
            {
                "type": "excel",
                "config_name": "Spool",
                "configuration": {"fast_mode": True},
            },
            {
                "type": "excel",
                "config_name": "Direct",
                "configuration": {"fast_mode": False},
            },
        ),
    )
    service = SequenceExportService(
        output_path_resolver=lambda *_args, **_kwargs: "C:/out/result.xlsx",
        spool_dir_resolver=lambda *_args, **_kwargs: "C:/out/.spool",
        csv_exporter=lambda *_args, **_kwargs: ExportResult(True, "spooled"),
        excel_exporter=lambda *_args, **_kwargs: ExportResult(False, "locked"),
    )
    submissions = []
    view = _View()
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=_Bus(),
        service=service,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        attempt_id_factory=lambda job_id, number: f"{job_id}-a{number}",
        connect_bus=False,
        debounce_ms=0,
    )
    controller.handle_export_requested(request)
    work, attempt = submissions.pop()
    outcome = service.execute_record_job(work, attempt.attempt_id)
    assert controller.handle_worker_failed(outcome)
    progress_count = len(view.started)

    controller.handle_rebuild_debounce()

    assert submissions == []
    assert len(view.started) == progress_count
    assert controller.handle_ignore_requested(
        IgnoreExportFailureRequested(work.job_id, attempt.attempt_id)
    )
    assert len(submissions) == 1
    assert submissions[0][0].kind is ExportJobKind.REBUILD


def test_sync_flush_calls_service_directly_and_returns_every_failure():
    calls = []

    class _Service:
        def flush_spool_targets(self, targets, *, analysis_config, product_model, on_close):
            calls.append((tuple(targets), analysis_config, product_model, on_close))
            return [("one", "locked"), ("two", "denied")]

    facade = SimpleNamespace(
        export_service=_Service(),
        export_model=SimpleNamespace(tracked_spool_targets=lambda: ("target",)),
        analysis_config={"Excel": {}},
        lineedit_type=SimpleNamespace(text=lambda: "MODEL"),
    )

    failures = SequenceWindow.flush_excel_spool_build(facade, on_close=True)

    assert failures == [("one", "locked"), ("two", "denied")]
    assert calls == [(('target',), {"Excel": {}}, "MODEL", True)]


def test_pure_sync_flush_collects_failure_and_exception_for_every_target():
    calls = []

    def build(_configuration, *, file_path, spool_dir):
        calls.append((file_path, spool_dir))
        if file_path.endswith("two.xlsx"):
            raise PermissionError("denied")
        return ExportResult(False, "locked")

    service = SequenceExportService(spool_builder=build)
    targets = (
        SpoolTarget.create("one", {}, "C:/out/one.xlsx", "C:/out/one"),
        SpoolTarget.create("two", {}, "C:/out/two.xlsx", "C:/out/two"),
    )

    failures = service.flush_spool_targets(
        targets,
        analysis_config={},
        product_model="",
        on_close=True,
    )

    assert failures == [("one", "locked"), ("two", "denied")]
    assert len(calls) == 2


def test_sync_flush_reports_path_builder_false_and_builder_exception_in_config_order():
    calls = []

    def output_path(configuration, *, product_model):
        name = configuration["name"]
        if name == "path":
            raise PermissionError("bad output path")
        return f"C:/out/{name}.xlsx"

    def spool_dir(configuration, *, file_path):
        if configuration["name"] == "spool":
            raise OSError("bad spool path")
        return f"{file_path}.spool"

    def build(configuration, *, file_path, spool_dir):
        name = configuration["name"]
        calls.append(name)
        if name == "false":
            return ExportResult(False, "locked")
        if name == "raise":
            raise PermissionError("denied")
        return ExportResult(True, "saved")

    analysis_config = {
        name: {"type": "Excel", "fast_mode": True, "name": name}
        for name in ("path", "false", "raise", "success", "spool")
    }
    service = SequenceExportService(
        output_path_resolver=output_path,
        spool_dir_resolver=spool_dir,
        spool_builder=build,
    )

    failures = service.flush_spool_targets(
        (),
        analysis_config=analysis_config,
        product_model="MODEL",
    )

    assert failures == [
        ("path", "bad output path"),
        ("false", "locked"),
        ("raise", "denied"),
        ("spool", "bad spool path"),
    ]
    assert calls == ["false", "raise", "success"]


def test_model_evicts_only_clean_target_state_and_never_pending_or_failed_work():
    model = SequenceExportModel()
    pending = SpoolTarget.create("pending", {}, "C:/out/pending.xlsx", "C:/spool/pending")
    failed = SpoolTarget.create("failed", {"version": 1}, "C:/out/failed.xlsx", "C:/spool/v1")
    clean = SpoolTarget.create("clean", {}, "C:/out/clean.xlsx", "C:/spool/clean")
    model.mark_target_dirty(pending)
    model.mark_target_dirty(failed)
    failed_job = model.begin_rebuild(failed.key)
    model.complete_rebuild(
        failed_job.job_id,
        failed_job.attempt_id,
        succeeded=False,
        failure=(("failed", "locked"),),
    )
    model.mark_target_dirty(clean)
    clean_job = model.begin_rebuild(clean.key)

    model.complete_rebuild(clean_job.job_id, clean_job.attempt_id, succeeded=True)

    assert model.target_state_count == 2
    assert model.has_active_work() is True
    assert set(model.dirty_target_keys()) == {pending.key}
    assert model.active_rebuild(failed.key) is None
    assert model.retry_rebuild(failed_job.job_id, failed_job.attempt_id) is not None


def test_model_target_metadata_remains_bounded_after_many_unique_clean_builds():
    model = SequenceExportModel()

    for index in range(200):
        target = SpoolTarget.create(
            f"Result-{index}",
            {},
            f"C:/out/{index}.xlsx",
            f"C:/spool/{index}",
        )
        model.mark_target_dirty(target)
        job = model.begin_rebuild(target.key)
        model.complete_rebuild(job.job_id, job.attempt_id, succeeded=True)

    assert model.target_state_count == 0
    assert model.tracked_spool_targets() == ()
    assert model.has_active_work() is False


def test_export_mvc_production_code_contains_no_nested_retry_or_event_pump():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "ui" / "sequence" / name
        for name in (
            "sequence_export_controller.py",
            "sequence_export_model.py",
            "sequence_export_view.py",
            "sequence_export_service.py",
            "sequence_export_worker.py",
        )
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.While) for node in ast.walk(tree)), path.name
        assert "processEvents" not in source
        assert ".exec_(" not in source
        assert ".exec(" not in source


def test_formal_export_controller_replaces_task7_legacy_adapter():
    source = Path(SequenceWindow.__module__.replace(".", "/") + ".py")
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / "ui" / "sequence" / "sequence_widget.py"
    text = source.read_text(encoding="utf-8")
    assert "self.export_controller = SequenceExportController(" in text
    assert 'if getattr(self, "export_controller", None) is None:' not in text
    assert "_handle_legacy_analysis_export_requested" not in text
    assert "export_decision_requires_terminal=True" in text


def test_real_sequence_window_event_bus_and_qthread_export_are_composed(monkeypatch):
    monkeypatch.setattr(
        LoadUiConfig,
        "get_tcp_config",
        staticmethod(lambda: ("127.0.0.1", 0)),
    )
    window = SequenceWindow()

    class _SuccessService:
        def execute_record_job(self, job, attempt_id):
            return ExportExecutionOutcome(
                True,
                job.job_id,
                attempt_id,
                job.record_id,
                tuple(
                    ExportTargetResult(
                        str(job.logical_job.target_configurations[index]["type"]),
                        str(
                            job.logical_job.target_configurations[index][
                                "config_name"
                            ]
                        ),
                        "ok",
                    )
                    for index in job.target_indices
                ),
                (),
                (),
                job.target_indices,
                (),
            )

    completed = []
    window.sequence_event_bus.unregister_export_terminal_recipient(
        window.workflow_controller._export_terminal_recipient_name
    )
    window.sequence_event_bus.register_export_terminal_recipient(
        "composition-test",
        lambda message: completed.append(message) or True,
        critical=True,
    )
    window.export_controller.service = _SuccessService()
    try:
        window.sequence_event_bus.commands.export_requested.emit(_request())

        deadline = time.monotonic() + 3
        while not completed and time.monotonic() < deadline:
            _QAPP.processEvents()
        assert completed
        assert isinstance(window.export_controller, SequenceExportController)
        assert isinstance(window.sequence_event_bus, SequenceEventBus)
        thread = window.export_controller._worker_thread
        if thread is not None:
            assert thread.wait(3000)
    finally:
        window.hide()
        _complete_window_resource_shutdown(window)
        window.close()
        _QAPP.processEvents()


@pytest.mark.parametrize(
    "first_outcome",
    [
        False,
        RuntimeError("sender failed"),
        KeyboardInterrupt("sender interrupted"),
        SystemExit("sender exited"),
    ],
    ids=["false", "ordinary", "keyboard-interrupt", "system-exit"],
)
def test_real_qthread_window_teardown_does_not_veto_next_transport_retry(
    monkeypatch, first_outcome
):
    from ui.sequence.sequence_widget import _CANONICAL_TCP_MIRROR_STATE

    monkeypatch.setattr(
        LoadUiConfig,
        "get_tcp_config",
        staticmethod(lambda: ("127.0.0.1", 0)),
    )
    baseline_owner_count = _CANONICAL_TCP_MIRROR_STATE.owner_count
    old = SequenceWindow()
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline_owner_count + 1

    class _SuccessService:
        def execute_record_job(self, job, attempt_id):
            return ExportExecutionOutcome(
                True,
                job.job_id,
                attempt_id,
                job.record_id,
                tuple(
                    ExportTargetResult(
                        str(job.logical_job.target_configurations[index]["type"]),
                        str(
                            job.logical_job.target_configurations[index][
                                "config_name"
                            ]
                        ),
                        "ok",
                    )
                    for index in job.target_indices
                ),
                (),
                (),
                job.target_indices,
                (),
            )

    old.export_controller.service = _SuccessService()
    old.sequence_event_bus.unregister_export_terminal_recipient(
        old.workflow_controller._export_terminal_recipient_name
    )
    old.sequence_event_bus.register_export_terminal_recipient(
        "loop4-qthread-terminal", lambda _message: True, critical=True
    )
    new = None
    transport_owner = None
    try:
        assert old.export_controller.handle_export_requested(_request())
        thread = old.export_controller._worker_thread
        assert thread is not None and thread.wait(3000)
        _QAPP.processEvents()
        _complete_window_resource_shutdown(old)
        old.close()
        assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline_owner_count

        new = SequenceWindow()
        assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline_owner_count + 1
        outcomes = iter((first_outcome, True))
        sends = []

        def send(message):
            sends.append(message)
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        server = SimpleNamespace(
            send_to_current_client=send,
            stop=lambda: True,
            client_address=None,
        )
        assert new._set_tcp_mirror_identity(server) is True
        bus = SequenceEventBus()
        model = SequenceWorkflowModel()
        bus.register_workflow_continuation_lifecycle_owner(new)
        transport_owner = SequenceAnalysisTransportController(
            bus=bus,
            authorization_provider=model.is_analysis_transport_authorized,
            authorization_consumer=model.consume_analysis_transport,
            tcp_enabled_provider=lambda: True,
            tcp_server_provider=new._get_tcp_mirror_identity,
        )
        event = AnalysisTransportReady(
            "analysis-next", "source-next", "record-next", 1, {"Label": "OK"}
        )
        assert model.authorize_analysis_transport(event)
        delivery_id = (
            "analysis-transport",
            event.analysis_id,
            event.source_id,
            event.record_id,
            event.workflow_generation,
        )

        assert bus.deliver_workflow_continuation(
            delivery_id, "analysis-transport", event, owner=new
        ) is False
        assert model.is_analysis_transport_authorized(event)
        assert bus.deliver_workflow_continuation(
            delivery_id, "analysis-transport", event, owner=new
        ) is True
        assert not model.is_analysis_transport_authorized(event)
        assert len(sends) == 2
    finally:
        if transport_owner is not None:
            transport_owner.disconnect()
        if new is not None:
            _complete_window_resource_shutdown(new)
            new.close()
        _complete_window_resource_shutdown(old)
        old.close()
        _QAPP.processEvents()
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline_owner_count
