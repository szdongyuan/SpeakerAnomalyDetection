from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication

from consts import error_code
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import SequenceExportModel
from ui.sequence.sequence_messages import (
    CommitRecordingLabelRequested,
    ManualLabelRequested,
)
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_service import (
    RecordingLabelContext,
    RecordingLabelService,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase


_QAPP = QApplication.instance() or QApplication([])


class _ExportView:
    pass


def _run_one_event_turn() -> None:
    loop = QEventLoop()
    QTimer.singleShot(20, loop.quit)
    loop.exec_()


def _build_recovering_label_flow(
    tmp_path: Path,
    failures: list[object],
    *,
    fail_second_forward: bool = False,
):
    original = tmp_path / "not_labeled" / "sample.wav"
    target = tmp_path / "OK" / original.name
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": "record-1", "labels": "not_labeled"}
    database = dict(info)
    calls: list[tuple[str, object]] = []

    class Projection:
        apply_attempts = 0
        restore_attempts = 0

        def capture_label_projection(self, _command):
            return {"projection": "old"}

        def apply_label_projection(self, *_args):
            self.apply_attempts += 1
            calls.append(("projection-forward", self.apply_attempts))
            if self.apply_attempts == 1:
                raise RuntimeError("force recovery")
            return True

        def restore_label_projection(self, _checkpoint, _error):
            self.restore_attempts += 1
            calls.append(("projection-restore", self.restore_attempts))
            if failures:
                failure = failures.pop(0)
                if failure is False:
                    return False
                raise failure
            return True

    def move(path, label):
        destination = tmp_path / label / Path(path).name
        destination.parent.mkdir(exist_ok=True)
        forward_attempt = 1 + len(
            [call for call in calls if call[0] == "file-forward"]
        )
        calls.append(("file-forward", forward_attempt))
        if fail_second_forward and forward_attempt == 2:
            raise RuntimeError("second forward failed")
        os.replace(path, destination)
        return str(destination)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        calls.append(("database", updated["labels"]))
        return error_code.OK, "updated"

    real_service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        root_dir="",
    )
    delivered_commands = []

    class TracingService:
        def commit(self, command, projection):
            delivered_commands.append(command)
            return real_service.commit(command, projection)

    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "test-view", lambda _event: True
    )
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    workflow_model.awaiting_label = True
    export = SequenceExportController(
        SequenceExportModel(), _ExportView(), bus=bus
    )
    workflow = SequenceWorkflowController(workflow_model, bus, connect_bus=True)
    workflow._continuation_retry_base_delay_ms = 100
    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=TracingService(),
        connect_queued=True,
    )
    return {
        "bus": bus,
        "workflow": workflow,
        "workflow_model": workflow_model,
        "export": export,
        "recording": recording,
        "original": original,
        "target": target,
        "info": info,
        "database": database,
        "calls": calls,
        "delivered_commands": delivered_commands,
    }


@pytest.mark.parametrize(
    "failures",
    [
        [False],
        [RuntimeError("restore-1"), RuntimeError("restore-2")],
        [KeyboardInterrupt("restore-1"), KeyboardInterrupt("restore-2")],
        [SystemExit("restore-1"), SystemExit("restore-2"), SystemExit("restore-3")],
    ],
    ids=("false-once", "ordinary-several", "keyboard-several", "system-exit-several"),
)
def test_recovery_pending_nacks_formal_delivery_until_automatic_exact_retry_succeeds(
    tmp_path, failures
):
    flow = _build_recovering_label_flow(tmp_path, list(failures))
    bus = flow["bus"]
    workflow = flow["workflow"]
    model = flow["workflow_model"]
    committed = QSignalSpy(bus.events.recording_label_committed)
    failed = QSignalSpy(bus.events.recording_label_commit_failed)

    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("manual-label", "record-1", "OK")
    )
    _run_one_event_turn()

    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.pending_continuation_publication_ids == (
        ("label-commit", "manual-label", model.workflow_generation),
    )
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert workflow._continuation_retry_timer.isSingleShot()
    assert workflow._continuation_retry_timer.isActive()
    assert (
        0
        < workflow.continuation_retry_delay_ms
        <= workflow.continuation_retry_max_delay_ms
    )
    assert len(committed) == 0 and len(failed) == 0

    assert committed.wait(5_000)
    workflow_state = QSignalSpy(bus.events.workflow_state_changed)
    if model.phase is not WorkflowPhase.IDLE:
        assert workflow_state.wait(1_000)

    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id is None
    assert model.awaiting_label is False
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 4
    assert workflow._continuation_retry_timer.isActive() is False
    assert len(committed) == 1 and len(failed) == 0
    assert all(
        command is flow["delivered_commands"][0]
        for command in flow["delivered_commands"]
    )
    assert flow["target"].read_bytes() == b"audio"
    assert flow["info"]["labels"] == "OK"
    assert flow["database"]["labels"] == "OK"
    assert flow["calls"].count(("database", "not_labeled")) == 1
    assert len(
        [call for call in flow["calls"] if call[0] == "file-forward"]
    ) == 2
    completed_delivery_id = (
        "label-commit",
        "manual-label",
        model.workflow_generation,
    )
    assert bus.deliver_workflow_continuation(
        completed_delivery_id,
        "label-commit",
        flow["delivered_commands"][0],
        owner=workflow,
    ) is True
    assert len(committed) == 1 and len(failed) == 0
    workflow.disconnect()
    flow["recording"].disconnect()


def test_recovery_convergence_can_ack_one_failed_terminal_then_admit_next_command(
    tmp_path,
):
    flow = _build_recovering_label_flow(
        tmp_path, [False], fail_second_forward=True
    )
    bus = flow["bus"]
    workflow = flow["workflow"]
    model = flow["workflow_model"]
    committed = QSignalSpy(bus.events.recording_label_committed)
    failed = QSignalSpy(bus.events.recording_label_commit_failed)
    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("manual-label", "record-1", "OK")
    )
    _run_one_event_turn()

    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert len(committed) == 0 and len(failed) == 0
    assert failed.wait(2_000)
    workflow_state = QSignalSpy(bus.events.workflow_state_changed)
    if model.phase is not WorkflowPhase.IDLE:
        assert workflow_state.wait(1_000)

    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id == "record-1"
    assert model.awaiting_label is True
    assert workflow.pending_continuation_publication_ids == ()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert len(committed) == 0 and len(failed) == 1
    first_command = flow["delivered_commands"][0]
    assert flow["delivered_commands"] == [first_command, first_command]

    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("next-label", "record-1", "OK")
    )
    assert committed.wait(1_000)
    next_state = QSignalSpy(bus.events.workflow_state_changed)
    if model.phase is not WorkflowPhase.IDLE:
        assert next_state.wait(1_000)

    assert model.phase is WorkflowPhase.IDLE
    assert model.retained_record_id is None
    assert model.awaiting_label is False
    assert len(committed) == 1 and len(failed) == 1
    assert flow["delivered_commands"][-1].command_id == "next-label"
    workflow.disconnect()
    flow["recording"].disconnect()


def test_modified_same_id_payload_is_rejected_without_touching_pending_recovery(
    tmp_path,
):
    flow = _build_recovering_label_flow(tmp_path, [False, False])
    bus = flow["bus"]
    workflow = flow["workflow"]
    model = flow["workflow_model"]
    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("manual-label", "record-1", "OK")
    )
    _run_one_event_turn()
    delivery_id = workflow.pending_continuation_publication_ids[0]
    before_calls = tuple(flow["calls"])
    before_deliveries = tuple(flow["delivered_commands"])

    assert bus.deliver_workflow_continuation(
        delivery_id,
        "label-commit",
        CommitRecordingLabelRequested("manual-label", "record-1", "NG", ()),
        owner=workflow,
    ) is False

    assert tuple(flow["calls"]) == before_calls
    assert tuple(flow["delivered_commands"]) == before_deliveries
    assert model.phase is WorkflowPhase.LABEL_COMMITTING
    assert workflow.pending_continuation_publication_ids == (delivery_id,)
    workflow.disconnect()
    flow["recording"].disconnect()


def test_disconnect_abandons_pending_label_delivery_and_stops_bounded_retry(
    tmp_path,
):
    flow = _build_recovering_label_flow(
        tmp_path, [SystemExit("still pending")] * 8
    )
    bus = flow["bus"]
    workflow = flow["workflow"]
    committed = QSignalSpy(bus.events.recording_label_committed)
    failed = QSignalSpy(bus.events.recording_label_commit_failed)
    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("manual-label", "record-1", "OK")
    )
    _run_one_event_turn()

    assert workflow._continuation_retry_timer.isSingleShot()
    assert workflow._continuation_retry_timer.isActive()
    workflow.disconnect()

    assert workflow.pending_continuation_publication_ids == ()
    assert workflow._continuation_retry_timer.isActive() is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 3
    assert bus.abandoned_workflow_continuation_delivery_count == 1
    assert len(committed) == 0 and len(failed) == 0
    flow["recording"].disconnect()
