from __future__ import annotations

import ast
import json
import os
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication

from consts import error_code
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import SequenceExportModel
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    CommitRecordingLabelRequested,
    ConfigurationSnapshot,
    ManualLabelRequested,
    StartTestRequested,
)
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_model import (
    RecordingCounterStatistics,
    RecordingModel,
    RecordingStatisticsSnapshot,
    freeze_recording_session_value,
    thaw_recording_session_value,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase


_QAPP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[2]
OPERATION_DATE = date(2042, 2, 3).isoformat()
STALE_OPERATION_DATE = (
    date.fromisoformat(OPERATION_DATE) - timedelta(days=1)
).isoformat()


def _statistics_model(*, mark_total=4, mark_ok=3, mark_ng=1):
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=0,
            ok=0,
            ng=0,
            ok_percent="0%",
            datatime=OPERATION_DATE,
        ),
        mark=RecordingCounterStatistics(
            total=mark_total,
            ok=mark_ok,
            ng=mark_ng,
            not_labels=0,
            datatime=OPERATION_DATE,
        ),
    )
    return model


def _mark_statistics_payload(
    *, total=4, ok=3, ng=1, operation_date=OPERATION_DATE
):
    return json.dumps(
        {
            "total": total,
            "ok": ok,
            "ng": ng,
            "not_labels": 0,
            "datatime": operation_date,
        },
        indent=4,
    ).encode("utf-8")


def _process_events(limit: int = 24) -> None:
    for _ in range(limit):
        _QAPP.processEvents()


def test_sequence_window_contains_no_recording_owner_algorithms():
    source_path = ROOT / "ui" / "sequence" / "sequence_widget.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name
        for node in facade.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert methods.isdisjoint(
        {
            "update_recorded_signal_info_to_db",
            "_persist_recording_count_snapshot",
            "_workflow_recording_session_snapshot",
            "_cancel_recording_adapter",
            "_finalize_test_run",
            "_handle_legacy_analysis_label_commit_requested",
            "_apply_committed_manual_label",
            "_capture_legacy_replay_target",
            "_workflow_replay_readiness",
            "_handle_replay_workflow_rejection",
        }
    )
    for concrete_owner in (
        "get_recorded_info",
        "save_recorded_data_to_json",
        "FileOps.move_wav_to_dir",
        "RecordingManager()",
        "_delete_failed_streaming_outputs",
    ):
        assert concrete_owner not in source


def test_sequence_window_recording_compatibility_state_is_model_owned():
    source_path = ROOT / "ui" / "sequence" / "sequence_widget.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    properties = {}
    for node in facade.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [ast.unparse(item) for item in node.decorator_list]
        if "property" in decorators or any(item.endswith(".setter") for item in decorators):
            properties.setdefault(node.name, []).append(node)

    for name in (
        "recorded_path",
        "recorded_signal_info",
        "current_recorded_count",
        "last_play_count",
    ):
        accessors = properties[name]
        assert len(accessors) == 2
        assert all(
            "recording_model" in {
                item.attr for item in ast.walk(accessor) if isinstance(item, ast.Attribute)
            }
            for accessor in accessors
        )
    methods = {
        node.name
        for node in facade.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_commit_recording_identity" not in methods
    assert "_commit_recording_count_value" not in methods


def test_recording_count_projection_rolls_back_owner_and_line_edit_base_exception():
    from ui.sequence.sequence_recording_model import RecordingModel
    from ui.sequence.sequence_recording_view import RecordingCountProjection

    class LineEdit:
        def __init__(self):
            self.value = "4"
            self.fail = True

        def text(self):
            return self.value

        def setText(self, value):
            self.value = str(value)
            if self.fail and self.value == "5":
                raise KeyboardInterrupt("count line edit interrupted")

    model = RecordingModel()
    model.current_recorded_count = 4
    model.last_play_count = 4
    line_edit = LineEdit()
    projection = RecordingCountProjection(model, line_edit)

    with pytest.raises(KeyboardInterrupt, match="count line edit interrupted"):
        projection(5)

    assert model.current_recorded_count == 4
    assert model.last_play_count == 4
    assert line_edit.text() == "4"
    line_edit.fail = False
    projection(5)
    assert model.current_recorded_count == 5
    assert model.last_play_count == 5
    assert line_edit.text() == "5"


def test_analysis_legacy_test_mode_posts_shared_label_command_only():
    source_path = ROOT / "ui" / "sequence" / "sequence_analysis_controller.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceAnalysisController"
    )
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_complete_legacy_test_mode"
    )
    names = {
        node.id for node in ast.walk(method) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
    }

    assert "ManualLabelRequested" in names
    assert "manual_label_requested" in attributes
    assert "_finalize_test_run" not in attributes
    assert "set_test_result_file" not in attributes
    assert "set_test_text" not in attributes


def test_recording_admission_service_owns_normal_and_replay_snapshots(tmp_path):
    from ui.sequence.sequence_recording_service import (
        RecordingAdmissionInputs,
        RecordingAdmissionService,
    )

    state = RecordingAdmissionInputs(
        configuration_generation=11,
        product_model="MODEL",
        serial_number="SN-7",
        scanner_enabled=True,
        current_recorded_count=4,
        last_play_count=4,
        recorded_path=str(tmp_path / "previous.wav"),
        recorded_signal_info={"file_path": "previous-id", "labels": "not_labeled"},
        stimulus_data=[0.1, 0.2],
        stimulus_info={"repeat_times": 2},
        alignment_sample_count=2,
    )
    paths = []

    def recorded_info(product, count, serial, label):
        paths.append((product, count, serial, label))
        path = str(tmp_path / f"{product}-{count}-{serial}.wav")
        return path, {"file_path": path, "labels": label, "barcode": serial}

    service = RecordingAdmissionService(
        raw_inputs=lambda: state,
        recorded_info_factory=recorded_info,
        command_id_factory=lambda: "replay-command",
    )
    configuration = ConfigurationSnapshot([], {})
    start = StartTestRequested("start", "manual", "not_labeled", False, 11)
    normal = service.session_snapshot(start, configuration)

    assert paths == [("MODEL", 5, "SN-7", "not_labeled")]
    assert normal["configuration_generation"] == 11
    assert normal["pending_count"] == 5
    assert normal["record_id"] == normal["recorded_path"]
    assert normal["product_model"] == "MODEL"
    assert normal["serial_number"] == "SN-7"
    assert normal["scanner_enabled"] is True
    assert normal["stimulus_data"] == [0.1, 0.2]
    assert normal["stimulus_info"] == {"repeat_times": 2}
    assert normal["alignment_sample_count"] == 2

    replay = service.create_replay_request()
    assert replay is not None
    assert replay.command_id == "replay-command"
    assert replay.record_id == "previous-id"
    assert service.replay_readiness(replay, configuration) == (True, "")

    state.product_model = "CHANGED"
    state.serial_number = "SN-CHANGED"
    state.scanner_enabled = False
    state.stimulus_data.append(0.3)
    state.stimulus_info["repeat_times"] = 99
    state.alignment_sample_count = 99
    state.recorded_signal_info["barcode"] = "changed"
    replay_snapshot = service.session_snapshot(replay, configuration)
    assert replay_snapshot["record_id"] == "previous-id"
    assert replay_snapshot["recorded_path"] == str(tmp_path / "previous.wav")
    assert replay_snapshot["pending_count"] is None
    assert replay_snapshot["product_model"] == "MODEL"
    assert replay_snapshot["serial_number"] == "SN-7"
    assert replay_snapshot["scanner_enabled"] is True
    assert replay_snapshot["stimulus_data"] == (0.1, 0.2)
    assert replay_snapshot["stimulus_info"] == {"repeat_times": 2}
    assert replay_snapshot["alignment_sample_count"] == 2
    assert replay_snapshot["recorded_signal_info"]["file_path"] == str(
        tmp_path / "previous.wav"
    )
    assert "barcode" not in replay_snapshot["recorded_signal_info"]
    admitted = BeginRecordingRequested(
        replay.command_id,
        "session-replay",
        True,
        {**replay_snapshot, "workflow_generation": 3},
    )
    state.stimulus_data.append(0.4)
    assert admitted.session_snapshot["stimulus_data"] == (0.1, 0.2)
    assert service.pending_replay_count == 0


def test_recording_persistence_service_owns_database_and_count_ports():
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    calls = []
    manager = SimpleNamespace(
        save_signal_info_to_db=lambda info, stimulus: calls.append(
            ("database", info, stimulus)
        )
    )
    service = RecordingPersistenceService(
        recording_manager_factory=lambda: manager,
        count_saver=lambda product, count, serial, scanner: calls.append(
            ("count", product, count, serial, scanner)
        ),
    )

    service.save_recording_database({"file_path": "one.wav"}, {"name": "stim"})
    service.persist_count(
        8,
        {
            "product_model": "MODEL",
            "serial_number": "SN-8",
            "scanner_enabled": True,
        },
    )
    service.persist_view_count("MODEL", "9", "SN-9", False)

    assert calls == [
        ("database", {"file_path": "one.wav"}, {"name": "stim"}),
        ("count", "MODEL", 8, "SN-8", True),
        ("count", "MODEL", "9", "SN-9", False),
    ]


def test_recording_persistence_thaws_detached_plain_database_payloads():
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    source_signal = {
        "file_path": "one.wav",
        "channels": [0, 1],
        "metadata": {"tags": {"left", "right"}},
    }
    source_stimulus = {
        "stimulus_method": "chirp",
        "repeat_times": 2,
        "nested": {"levels": [0.25, 0.5]},
    }
    frozen_signal = freeze_recording_session_value(source_signal)
    frozen_stimulus = freeze_recording_session_value(source_stimulus)
    received = {}

    class MutatingManager:
        def save_signal_info_to_db(self, signal_info, stimulus_info):
            assert type(signal_info) is dict
            assert type(stimulus_info) is dict
            received["signal"] = signal_info
            received["stimulus"] = stimulus_info
            signal_info["channels"].append(2)
            signal_info["metadata"]["tags"].add("mutated")
            stimulus_info["nested"]["levels"].append(0.75)
            return "saved"

    service = RecordingPersistenceService(recording_manager_factory=MutatingManager)

    assert service.save_recording_database(frozen_signal, frozen_stimulus) == "saved"
    assert thaw_recording_session_value(frozen_signal) == source_signal
    assert thaw_recording_session_value(frozen_stimulus) == source_stimulus
    assert received["signal"]["channels"] == [0, 1, 2]
    assert received["stimulus"]["nested"]["levels"] == [0.25, 0.5, 0.75]


def test_recording_persistence_preserves_none_stimulus():
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    calls = []
    manager = SimpleNamespace(
        save_signal_info_to_db=lambda signal, stimulus: calls.append(
            (signal, stimulus)
        )
    )
    service = RecordingPersistenceService(
        recording_manager_factory=lambda: manager
    )

    service.save_recording_database(
        freeze_recording_session_value({"file_path": "one.wav"}), None
    )

    assert calls == [({"file_path": "one.wav"}, None)]


def test_recording_persistence_rejects_non_mapping_signal_before_manager():
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    manager_constructions = []
    service = RecordingPersistenceService(
        recording_manager_factory=lambda: manager_constructions.append(True)
    )

    with pytest.raises(TypeError, match="signal info must be a plain dictionary"):
        service.save_recording_database(("not", "a", "mapping"), None)

    assert manager_constructions == []


def test_recording_persistence_rejects_non_mapping_stimulus_before_manager():
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    manager_constructions = []
    service = RecordingPersistenceService(
        recording_manager_factory=lambda: manager_constructions.append(True)
    )

    with pytest.raises(TypeError, match="stimulus info must be a plain dictionary"):
        service.save_recording_database(
            freeze_recording_session_value({"file_path": "one.wav"}),
            ("not", "a", "mapping"),
        )

    assert manager_constructions == []


def test_recording_persistence_supplies_plain_mapping_to_legacy_stimulus_copy():
    from base.recording_management import RecordingManager
    from ui.sequence.sequence_recording_service import RecordingPersistenceService

    class FakeDatabase:
        def query_matching_data(self, *_args, **_kwargs):
            return [("existing-stimulus",)]

    class LegacyManager:
        def save_signal_info_to_db(self, _signal_info, stimulus_info):
            return RecordingManager.get_stimulus_info_to_db(
                stimulus_info, FakeDatabase()
            )

    frozen_stimulus = freeze_recording_session_value(
        {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "repeat_times": 1,
            "start_freq": 80,
            "stop_freq": 2000,
            "sample_rate": 48000,
            "total_time": 1,
        }
    )
    service = RecordingPersistenceService(recording_manager_factory=LegacyManager)

    stimulus_data, inserted = service.save_recording_database(
        freeze_recording_session_value({"file_path": "one.wav"}),
        frozen_stimulus,
    )

    assert stimulus_data == [("existing-stimulus",)]
    assert inserted is False


def test_recording_cancellation_service_is_owner_idempotent_and_never_uses_facade():
    from ui.sequence.sequence_recording_service import RecordingCancellationService

    calls = []
    prepared = SimpleNamespace(snapshot=SimpleNamespace(session_id="session"))
    blocking = SimpleNamespace(
        cancel=lambda value, reason: calls.append((value, reason))
        or {"restored": True, "errors": ()}
    )
    service = RecordingCancellationService(blocking)

    assert service.cancel(prepared, "stop") == {"restored": True, "errors": ()}
    assert service.cancel(prepared, "stop") == {"restored": True, "errors": ()}
    assert calls == [(prepared, "stop")]


def test_recording_label_owner_rolls_back_base_exception_then_accepts_next_command(
    tmp_path,
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = []

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    def update_label(updated, old_path):
        database.append((dict(updated), old_path))
        return error_code.OK, "updated"

    projection_calls = []

    def projection(command, receipt):
        projection_calls.append((command.command_id, receipt.new_path))
        if command.command_id == "label-base":
            raise KeyboardInterrupt("projection interrupted")

    label_service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update_label,
        root_dir="",
    )
    bus = SequenceEventBus()
    failed = []
    completed = []
    bus.events.recording_label_commit_failed.connect(failed.append)
    bus.events.recording_label_committed.connect(completed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=projection),
        label_service=label_service,
        connect_queued=False,
    )

    first = CommitRecordingLabelRequested("label-base", str(original), "OK", ())
    assert controller.handle_commit_recording_label(first) is True
    assert controller.handle_commit_recording_label(first) is False
    assert len(failed) == 1
    assert completed == []
    assert original.exists()
    assert info == {"file_path": str(original), "labels": "not_labeled"}

    second = CommitRecordingLabelRequested("label-next", str(original), "NG", ())
    assert controller.handle_commit_recording_label(second) is True
    assert len(failed) == 1
    assert len(completed) == 1
    assert completed[0].command_id == "label-next"
    assert Path(projection_calls[-1][1]).exists()
    assert info["labels"] == "NG"
    assert len(database) == 3  # failed write, rollback write, successful write


@pytest.mark.parametrize(
    "failure_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_label_transaction_restores_count_file_text_and_runtime_then_retries(
    tmp_path,
    failure_type,
):
    from ui.sequence.sequence_recording_service import (
        RecordingCountBoardPersistence,
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    count_path = tmp_path / "mark-result.json"
    old_count_payload = _mark_statistics_payload()
    count_path.write_bytes(old_count_payload)
    info = {"file_path": str(original), "labels": "not_labeled"}
    ui = {"text": "old", "plot": "old", "scanner": "old", "player": "old"}
    database = []

    class Projection:
        def capture_label_projection(self, _command):
            return dict(ui)

        def apply_label_projection(self, _command, _receipt, _checkpoint):
            ui.update(text="new", plot="empty", scanner="new", player="new")
            raise failure_type("projection failed after count file commit")

        def restore_label_projection(self, checkpoint, _error):
            ui.clear()
            ui.update(checkpoint)
            return True

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    def update_label(updated, old_path):
        database.append((dict(updated), old_path))
        return error_code.OK, "updated"

    statistics_model = _statistics_model()
    old_statistics = statistics_model.statistics
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "test-result.dat",
        mark_path_provider=lambda: count_path,
        statistics_model=statistics_model,
        date_provider=lambda: OPERATION_DATE,
    )
    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update_label,
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested("label-first", str(original), "OK", ())

    with pytest.raises(failure_type, match="projection failed"):
        service.commit(command, Projection())

    assert original.exists()
    assert count_path.read_bytes() == old_count_payload
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert statistics_model.statistics == old_statistics
    assert ui == {"text": "old", "plot": "old", "scanner": "old", "player": "old"}
    assert len(database) == 2


def test_recording_count_board_persistence_owns_atomic_file_algorithm(tmp_path):
    from ui.sequence.sequence_recording_service import RecordingCountBoardPersistence

    count_path = tmp_path / "mark-result.json"
    count_path.write_bytes(_mark_statistics_payload())
    statistics_model = _statistics_model()
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "test-result.dat",
        mark_path_provider=lambda: count_path,
        statistics_model=statistics_model,
        date_provider=lambda: OPERATION_DATE,
    )
    command = CommitRecordingLabelRequested("count-owner", "record", "NG", ())
    checkpoint = persistence.checkpoint(command)

    persistence.commit(command, checkpoint)

    payload = count_path.read_text(encoding="utf-8")
    assert '"total": 5' in payload
    assert '"ok": 3' in payload
    assert '"ng": 2' in payload


def test_recording_count_board_fixture_still_rejects_stale_operation_date(tmp_path):
    from ui.sequence.sequence_recording_service import RecordingCountBoardPersistence

    count_path = tmp_path / "mark-result.json"
    count_path.write_bytes(
        _mark_statistics_payload(operation_date=STALE_OPERATION_DATE)
    )
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "test-result.dat",
        mark_path_provider=lambda: count_path,
        statistics_model=_statistics_model(),
        date_provider=lambda: OPERATION_DATE,
    )
    command = CommitRecordingLabelRequested("count-stale", "record", "OK", ())

    checkpoint = persistence.checkpoint(command)

    with pytest.raises(ValueError, match="date does not match operation"):
        persistence.reconcile(checkpoint)


def test_label_transaction_attempts_every_restore_and_holds_incomplete_rollback(
    tmp_path,
):
    from ui.sequence.sequence_recording_service import (
        RecordingCountBoardPersistence,
        RecordingLabelContext,
        RecordingLabelRecoveryPending,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    count_path = tmp_path / "mark-result.json"
    old_count_payload = _mark_statistics_payload()
    count_path.write_bytes(old_count_payload)
    info = {"file_path": str(original), "labels": "not_labeled"}
    restore_calls = []
    apply_calls = []

    class Projection:
        def capture_label_projection(self, _command):
            return "checkpoint"

        def apply_label_projection(self, *_args):
            apply_calls.append("view")
            if len(apply_calls) == 1:
                raise KeyboardInterrupt("apply interrupted")
            return True

        def restore_label_projection(self, *_args):
            restore_calls.append("view")
            if restore_calls.count("view") == 1:
                raise SystemExit("view restore failed")
            return True

    statistics_model = _statistics_model()
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "test-result.dat",
        mark_path_provider=lambda: count_path,
        statistics_model=statistics_model,
        date_provider=lambda: OPERATION_DATE,
    )
    original_restore = persistence.restore

    def tracked_restore(checkpoint, error):
        restore_calls.append("count-file")
        return original_restore(checkpoint, error)

    persistence.restore = tracked_restore

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda *_args: (error_code.OK, "updated"),
        count_board_persistence=persistence,
        root_dir="",
    )

    command = CommitRecordingLabelRequested(
        "label-restore", str(original), "NG", ()
    )
    with pytest.raises(RecordingLabelRecoveryPending) as captured:
        service.commit(
            command,
            Projection(),
        )

    assert restore_calls == ["view", "count-file"]
    assert count_path.read_bytes() == old_count_payload
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert captured.value.rollback_outcome.pending == ("view",)
    with pytest.raises(RecordingLabelRecoveryPending):
        service.commit(
            CommitRecordingLabelRequested("label-next", str(original), "OK", ()),
            Projection(),
        )

    outcome = service.commit(command, Projection())
    assert outcome["database_updated"] is True
    assert info["labels"] == "NG"
    assert restore_calls == ["view", "count-file", "view"]


@pytest.mark.parametrize(
    ("failure_stage", "failure_type"),
    [
        ("window", KeyboardInterrupt),
        ("count-text", RuntimeError),
        ("plot", SystemExit),
        ("scanner", RuntimeError),
        ("player", KeyboardInterrupt),
        ("player-false", RuntimeError),
    ],
)
def test_label_view_projection_restores_every_mutated_surface(
    failure_stage,
    failure_type,
):
    from ui.sequence.sequence_recording_service import RecordingLabelCommitReceipt
    from ui.sequence.sequence_recording_view import SequenceRecordingLabelProjection

    class Field:
        def __init__(self, value, *, enabled=True):
            self.value = str(value)
            self.enabled = enabled
            self.read_only = False
            self.icon_value = "old-icon"
            self.icon_size_value = "old-size"

        def text(self):
            return self.value

        def setText(self, value):
            self.value = str(value)

        def clear(self):
            self.value = ""

        def isEnabled(self):
            return self.enabled

        def setEnabled(self, value):
            self.enabled = bool(value)

        def isReadOnly(self):
            return self.read_only

        def setReadOnly(self, value):
            self.read_only = bool(value)

        def icon(self):
            return self.icon_value

        def setIcon(self, value):
            self.icon_value = value

        def iconSize(self):
            return self.icon_size_value

        def setIconSize(self, value):
            self.icon_size_value = value

    class Window:
        def __init__(self, name):
            self.name = name
            self.visible = True

        def isVisible(self):
            return self.visible

        def show(self):
            self.visible = True

        def hide(self):
            self.visible = False

    analysis_window = Window("analysis")
    summary_window = Window("summary")
    feedback_window = Window("feedback")
    analysis_model = SimpleNamespace(
        analysis_instances=[analysis_window],
        analysis_registry={"analysis": analysis_window},
    )
    analysis_view = SimpleNamespace(
        model=analysis_model,
        summary_window=summary_window,
        feedback_dialogs=[feedback_window],
        window_keys={analysis_window: "analysis"},
    )
    count_board = SimpleNamespace(
        total_line_edit=Field("10"),
        ok_line_edit=Field("8"),
        ng_line_edit=Field("2"),
        yield_line_edit=Field("80%"),
        datatime_line_edit=Field("today"),
        mark_total_edit=Field("4"),
        mark_ok_edit=Field("3"),
        mark_ng_edit=Field("1"),
    )

    original_mark_ok_set_text = count_board.mark_ok_edit.setText
    count_text_failed = False

    def set_mark_ok_text(value):
        nonlocal count_text_failed
        original_mark_ok_set_text(value)
        if failure_stage == "count-text" and not count_text_failed:
            count_text_failed = True
            raise failure_type("count text failure")

    count_board.mark_ok_edit.setText = set_mark_ok_text
    player_btn = Field("", enabled=True)
    serial = Field("SN-OLD", enabled=True)
    data_struct = SimpleNamespace(
        store_wave_data="mono-old",
        store_wave_data_multi="multi-old",
        wav_calibration_metadata={"old": True},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    plot_state = {"value": "plot-old"}
    runtime = SimpleNamespace(
        recorded_path="old.wav",
        signal_info={"old": True},
        lineedit_s_or_n=serial,
        replayer_btn=Field("", enabled=True),
        data_btn=Field("", enabled=True),
        player_btn=player_btn,
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: True),
        count_board=count_board,
        data_struct=data_struct,
        analysis_view=analysis_view,
        _sn_clear_on_next_scan=True,
    )

    def close_windows():
        analysis_window.hide()
        summary_window.hide()
        feedback_window.hide()
        analysis_model.analysis_instances.clear()
        analysis_model.analysis_registry.clear()
        analysis_view.summary_window = None
        analysis_view.feedback_dialogs.clear()
        analysis_view.window_keys.clear()
        if failure_stage == "window":
            raise failure_type("window failure")

    def clear_plot():
        plot_state["value"] = "empty"
        if failure_stage == "plot":
            raise failure_type("plot failure")

    def clicked_scanner():
        serial.setEnabled(False)
        serial.setReadOnly(True)
        if failure_stage == "scanner":
            raise failure_type("scanner failure")

    def update_player():
        player_btn.setEnabled(False)
        player_btn.setIcon("new-icon")
        player_btn.setIconSize("new-size")
        if failure_stage == "player":
            raise failure_type("player failure")
        if failure_stage == "player-false":
            return False

    runtime._close_analysis_windows = close_windows
    runtime._capture_plot_projection_state = lambda: plot_state["value"]
    runtime._restore_plot_projection_state = lambda state: plot_state.update(value=state)
    runtime._clear_plot_area = clear_plot
    runtime.clicked_scanner = clicked_scanner
    runtime.update_player_btn_is_paused = update_player
    projection = SequenceRecordingLabelProjection(runtime)
    command = CommitRecordingLabelRequested("label-view", "old.wav", "OK", ())
    receipt = RecordingLabelCommitReceipt(
        "old.wav",
        "new.wav",
        {"file_path": "old.wav"},
        {"file_path": "new.wav"},
        _statistics_model().statistics,
    )
    checkpoint = projection.capture_label_projection(command)

    with pytest.raises(failure_type):
        projection.apply_label_projection(command, receipt, checkpoint)
    assert projection.restore_label_projection(checkpoint, failure_type("failed"))

    assert runtime.recorded_path == "old.wav"
    assert runtime.signal_info == {"old": True}
    assert serial.text() == "SN-OLD"
    assert serial.isEnabled() is True
    assert serial.isReadOnly() is False
    assert runtime.replayer_btn.isEnabled() is True
    assert runtime.data_btn.isEnabled() is True
    assert player_btn.isEnabled() is True
    assert player_btn.icon() == "old-icon"
    assert player_btn.iconSize() == "old-size"
    assert count_board.mark_total_edit.text() == "4"
    assert count_board.mark_ok_edit.text() == "3"
    assert count_board.mark_ng_edit.text() == "1"
    assert data_struct.store_wave_data == "mono-old"
    assert data_struct.store_wave_data_multi == "multi-old"
    assert plot_state["value"] == "plot-old"
    assert analysis_model.analysis_instances == [analysis_window]
    assert analysis_model.analysis_registry == {"analysis": analysis_window}
    assert analysis_view.summary_window is summary_window
    assert analysis_view.feedback_dialogs == [feedback_window]
    assert analysis_view.window_keys == {analysis_window: "analysis"}
    assert analysis_window.visible and summary_window.visible and feedback_window.visible


def test_formal_workflow_label_command_is_consumed_by_recording_owner(tmp_path):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": "record-1", "labels": "not_labeled"}
    side_effects = []

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        side_effects.append(("move", label))
        return str(target)

    bus = SequenceEventBus()
    bus.register_workflow_continuation_recipient(
        "workflow-state", "test-view", lambda _event: True
    )
    workflow_model = SequenceWorkflowModel()
    workflow_model.retained_record_id = "record-1"
    workflow_model.awaiting_label = True
    export = SequenceExportController(
        SequenceExportModel(), SimpleNamespace(), bus=bus
    )
    workflow = SequenceWorkflowController(workflow_model, bus, connect_bus=True)
    label_service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda updated, old: side_effects.append(
            ("database", updated["labels"], old)
        )
        or (error_code.OK, "updated"),
        root_dir="",
    )
    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(
            commit_label_projection=lambda command, _receipt: side_effects.append(
                ("view", command.label)
            )
        ),
        label_service=label_service,
        connect_queued=True,
    )

    bus.commands.manual_label_requested.emit(
        ManualLabelRequested("manual-label", "record-1", "OK")
    )
    _process_events()

    assert workflow_model.phase is WorkflowPhase.IDLE
    assert workflow_model.retained_record_id is None
    assert workflow_model.awaiting_label is False
    assert [item[0] for item in side_effects] == ["move", "database", "view"]
    assert export is not None and workflow is not None and recording is not None


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_label_move_mutate_then_failure_restores_exact_file_before_terminal(
    tmp_path, failure
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    target = tmp_path / "OK" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"original-audio")
    info = {"file_path": str(original), "labels": "not_labeled"}

    def move(path, _label):
        target.parent.mkdir()
        os.replace(path, target)
        if failure is False:
            return False
        raise failure("move failed after mutation")

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda *_args: pytest.fail("database must not be attempted"),
        root_dir="",
    )
    command = CommitRecordingLabelRequested(
        "move-ambiguous", str(original), "OK", ()
    )

    expected = RuntimeError if failure is False else failure
    with pytest.raises(expected):
        service.commit(command, SequenceRecordingView())

    assert original.read_bytes() == b"original-audio"
    assert not target.exists()
    assert info == {"file_path": str(original), "labels": "not_labeled"}


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_label_database_mutate_then_failure_is_always_compensated(tmp_path, failure):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)
    calls = []

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    def update(updated, old_path):
        calls.append((dict(updated), old_path))
        database.clear()
        database.update(updated)
        if updated["labels"] == "not_labeled":
            return error_code.OK, "restored"
        if failure is False:
            return False
        raise failure("database failed after mutation")

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        root_dir="",
    )
    command = CommitRecordingLabelRequested(
        "database-ambiguous", str(original), "OK", ()
    )

    expected = RuntimeError if failure is False else failure
    with pytest.raises(expected):
        service.commit(command, SequenceRecordingView())

    assert len(calls) == 2
    assert database == {"file_path": str(original), "labels": "not_labeled"}
    assert original.exists()
    assert info == database


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_label_count_forward_mutate_then_failure_rolls_back_every_layer(
    tmp_path, failure
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    count_state = {"value": "old"}

    class CountPersistence:
        def checkpoint(self, _command):
            return "old"

        def reconcile(self, checkpoint):
            return checkpoint

        def commit(self, _command, _checkpoint):
            count_state["value"] = "new"
            if failure is False:
                return False
            raise failure("count failed after mutation")

        def restore(self, checkpoint, _error):
            count_state["value"] = checkpoint
            return True

        def fail_closed(self, _checkpoint, _error):
            pytest.fail("exact count restore must succeed")

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    database = []

    def update(updated, old_path):
        database.append((dict(updated), old_path))
        return error_code.OK, "updated"

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=CountPersistence(),
        root_dir="",
    )
    command = CommitRecordingLabelRequested("count-ambiguous", str(original), "OK", ())

    expected = RuntimeError if failure is False else failure
    with pytest.raises(expected):
        service.commit(command, SequenceRecordingView())

    assert count_state == {"value": "old"}
    assert original.exists()
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert len(database) == 2


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_label_mapping_mutate_then_failure_rolls_back_exact_state(tmp_path, failure):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")

    class AmbiguousMapping(dict):
        failed = False

        def update(self, values=(), **kwargs):
            normalized = dict(values, **kwargs)
            if normalized.get("labels") == "OK" and not self.failed:
                self.failed = True
                super().update({"labels": "OK"})
                if failure is False:
                    return False
                raise failure("mapping failed after mutation")
            return super().update(normalized)

    info = AmbiguousMapping(
        {"file_path": str(original), "labels": "not_labeled", "stable": True}
    )

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    database = []

    def update(updated, old_path):
        database.append((dict(updated), old_path))
        return error_code.OK, "updated"

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        root_dir="",
    )
    command = CommitRecordingLabelRequested("mapping-ambiguous", str(original), "OK", ())

    expected = RuntimeError if failure is False else failure
    with pytest.raises(expected):
        service.commit(command, SequenceRecordingView())

    assert dict(info) == {
        "file_path": str(original),
        "labels": "not_labeled",
        "stable": True,
    }
    assert original.exists()
    assert len(database) == 2


@pytest.mark.parametrize(
    "restore_failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_unconverged_label_rollback_stays_pending_and_same_command_retries(
    tmp_path, restore_failure
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    ui = {"value": "old"}
    apply_attempts = []
    restore_attempts = []
    fail_closed_attempts = []

    class Projection:
        def capture_label_projection(self, _command):
            return dict(ui)

        def apply_label_projection(self, command, _receipt, _checkpoint):
            apply_attempts.append(command.command_id)
            ui["value"] = "new"
            if len(apply_attempts) == 1:
                raise RuntimeError("projection failed")
            return True

        def restore_label_projection(self, checkpoint, _error):
            restore_attempts.append(len(restore_attempts) + 1)
            if len(restore_attempts) == 1:
                if restore_failure is False:
                    return False
                raise restore_failure("restore failed")
            ui.clear()
            ui.update(checkpoint)
            return True

        def fail_closed_label_projection(self, _checkpoint, _error):
            fail_closed_attempts.append(len(fail_closed_attempts) + 1)
            return False

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    database = []

    def update(updated, old_path):
        database.append((dict(updated), old_path))
        return error_code.OK, "updated"

    projection = Projection()
    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        root_dir="",
    )
    bus = SequenceEventBus()
    committed = []
    failed = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=projection),
        label_service=service,
        connect_queued=False,
    )
    command = CommitRecordingLabelRequested("pending-label", str(original), "OK", ())
    other = CommitRecordingLabelRequested("other-label", str(original), "NG", ())

    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    assert controller.handle_commit_recording_label(other) is False
    assert committed == [] and failed == []

    assert controller.handle_commit_recording_label(command) is True
    assert len(committed) == 1 and failed == []
    assert committed[0].command_id == command.command_id
    assert controller.handle_commit_recording_label(command) is False
    assert restore_attempts == [1, 2]
    assert fail_closed_attempts == []
    assert apply_attempts == [command.command_id, command.command_id]
    assert info["labels"] == "OK"
    assert ui == {"value": "new"}


@pytest.mark.parametrize(
    "restore_failure",
    [False, RuntimeError, KeyboardInterrupt, SystemExit],
)
def test_ambiguous_database_compensation_blocks_terminal_until_retry(
    tmp_path, restore_failure
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)
    compensation_attempts = []
    projection_attempts = []

    class Projection:
        def capture_label_projection(self, _command):
            return {"value": "old"}

        def apply_label_projection(self, _command, _receipt, _checkpoint):
            projection_attempts.append(len(projection_attempts) + 1)
            if len(projection_attempts) == 1:
                raise RuntimeError("force rollback")
            return True

        def restore_label_projection(self, _checkpoint, _error):
            return True

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        if updated["labels"] == "not_labeled":
            compensation_attempts.append(len(compensation_attempts) + 1)
            if len(compensation_attempts) == 1:
                if restore_failure is False:
                    return False
                raise restore_failure("compensation failed after mutation")
        return error_code.OK, "updated"

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        root_dir="",
    )
    bus = SequenceEventBus()
    committed = []
    failed = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=service,
        connect_queued=False,
    )
    command = CommitRecordingLabelRequested("database-recovery", str(original), "OK", ())

    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    assert controller.handle_commit_recording_label(command) is True
    assert len(committed) == 1 and failed == []
    assert compensation_attempts == [1, 2]
    assert projection_attempts == [1, 2]
    assert database["labels"] == "OK"
    assert controller.handle_commit_recording_label(command) is False


def test_label_rollback_attempts_all_layers_when_multiple_restores_fail(tmp_path):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    calls = []

    class Projection:
        def capture_label_projection(self, _command):
            return "view-checkpoint"

        def apply_label_projection(self, *_args):
            calls.append("view-forward")
            raise RuntimeError("forward failed")

        def restore_label_projection(self, *_args):
            calls.append("view-restore")
            raise KeyboardInterrupt("view restore failed")

        def fail_closed_label_projection(self, *_args):
            calls.append("view-fail-close")
            return False

    class CountPersistence:
        def checkpoint(self, _command):
            return "count-checkpoint"

        def reconcile(self, checkpoint):
            return checkpoint

        def commit(self, *_args):
            calls.append("count-forward")
            return {"count_file_updated": True}

        def restore(self, *_args):
            calls.append("count-restore")
            raise SystemExit("count restore failed")

        def fail_closed(self, *_args):
            calls.append("count-fail-close")
            return False

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        if updated["labels"] == "not_labeled":
            calls.append("database-restore")
            raise RuntimeError("database restore failed")
        calls.append("database-forward")
        return error_code.OK, "updated"

    def replace_file(_source, _target):
        calls.append("file-restore")
        raise RuntimeError("file restore failed")

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=CountPersistence(),
        replace_file=replace_file,
        root_dir="",
    )
    bus = SequenceEventBus()
    committed = []
    failed = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=service,
        connect_queued=False,
    )
    command = CommitRecordingLabelRequested("multi-recovery", str(original), "OK", ())

    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    for expected in (
        "view-restore",
        "count-restore",
        "database-restore",
        "file-restore",
    ):
        assert expected in calls
    assert "view-fail-close" not in calls
    assert "count-fail-close" not in calls
    recovery_files = list(tmp_path.rglob("*.recovery"))
    assert len(recovery_files) == 1
    assert recovery_files[0].read_bytes() == b"audio"


def test_label_controller_reentrant_same_command_has_one_first_terminal():
    command = CommitRecordingLabelRequested("reentrant-label", "record.wav", "OK", ())
    bus = SequenceEventBus()
    committed = []
    failed = []
    nested_results = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)

    class ReentrantService:
        calls = 0

        def commit(self, nested_command, _view):
            self.calls += 1
            if self.calls == 1:
                nested_results.append(
                    controller.handle_commit_recording_label(nested_command)
                )
            return {"database_updated": True, "view_projected": True}

    service = ReentrantService()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        label_service=service,
        connect_queued=False,
    )

    assert controller.handle_commit_recording_label(command) is True
    assert nested_results == [False]
    assert service.calls == 1
    assert len(committed) == 1 and failed == []
    assert controller.handle_commit_recording_label(command) is False


def test_file_rollback_failure_preserves_audio_and_holds_cross_layer_recovery(
    tmp_path,
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    target = tmp_path / "OK" / "sample.wav"
    original.parent.mkdir()
    target.parent.mkdir()
    original.write_bytes(b"original-audio")
    target.write_bytes(b"prior-target")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)
    replace_enabled = {"value": False}
    apply_attempts = []

    class Projection:
        def capture_label_projection(self, _command):
            return {"state": "old"}

        def apply_label_projection(self, *_args):
            apply_attempts.append(len(apply_attempts) + 1)
            if len(apply_attempts) == 1:
                raise RuntimeError("force file rollback")
            return True

        def restore_label_projection(self, *_args):
            return True

        def enter_label_recovery(self, descriptor, _checkpoint, _error):
            assert descriptor["state"] == "RECOVERY_PENDING"
            return True

    def move(path, _label):
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        return error_code.OK, "updated"

    def replace_file(source, destination):
        if not replace_enabled["value"]:
            return False
        os.replace(source, destination)

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        replace_file=replace_file,
        root_dir="",
    )
    bus = SequenceEventBus()
    committed = []
    failed = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=service,
        connect_queued=False,
    )
    command = CommitRecordingLabelRequested(
        "file-recovery", str(original), "OK", {"nested": [{"attempt": 1}]}
    )

    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert database == info
    recovery_files = list(tmp_path.rglob("*.recovery"))
    assert len(recovery_files) == 1
    recovery_path = recovery_files[0]
    assert recovery_path.exists()
    assert recovery_path.read_bytes() == b"original-audio"
    assert not (not original.exists() and not target.exists() and not recovery_path.exists())

    replace_enabled["value"] = True
    assert controller.handle_commit_recording_label(command) is True
    assert len(committed) == 1 and failed == []
    assert info["labels"] == "OK"
    assert target.read_bytes() == b"original-audio"


def test_pending_label_retry_requires_exact_safe_canonical_command_identity(tmp_path):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    apply_attempts = []
    restore_attempts = []

    class Projection:
        def capture_label_projection(self, _command):
            return "checkpoint"

        def apply_label_projection(self, *_args):
            apply_attempts.append(len(apply_attempts) + 1)
            if len(apply_attempts) == 1:
                raise RuntimeError("force pending recovery")
            return True

        def restore_label_projection(self, *_args):
            restore_attempts.append(len(restore_attempts) + 1)
            return len(restore_attempts) > 1

        def fail_closed_label_projection(self, *_args):
            return False

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda *_args: (error_code.OK, "updated"),
        root_dir="",
    )
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=service,
        connect_queued=False,
    )
    original_command = CommitRecordingLabelRequested(
        "canonical-label",
        str(original),
        "OK",
        {"outer": [{"value": 1}], "order": ["a", "b"]},
    )
    altered = CommitRecordingLabelRequested(
        "canonical-label",
        str(original),
        "OK",
        {"outer": [{"value": 2}], "order": ["a", "b"]},
    )
    equivalent = CommitRecordingLabelRequested(
        "canonical-label",
        str(original),
        "OK",
        {"order": ["a", "b"], "outer": [{"value": 1}]},
    )

    assert controller.handle_commit_recording_label(original_command) is False
    assert controller.handle_commit_recording_label(altered) is False
    assert restore_attempts == [1]
    assert apply_attempts == [1]
    assert controller.handle_commit_recording_label(equivalent) is True
    assert restore_attempts == [1, 2]
    assert apply_attempts == [1, 2]


def test_label_identity_rejects_hostile_and_oversized_payload_without_service_call():
    hostile_calls = []

    class Hostile:
        def __eq__(self, _other):
            hostile_calls.append("eq")
            return False

        def __hash__(self):
            hostile_calls.append("hash")
            return 1

        def __str__(self):
            hostile_calls.append("str")
            return "hostile"

        def __repr__(self):
            hostile_calls.append("repr")
            return "Hostile()"

    calls = []
    service = SimpleNamespace(
        commit=lambda *_args: calls.append("commit") or {"ok": True}
    )
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        view=SequenceRecordingView(),
        label_service=service,
        connect_queued=False,
    )
    hostile = object.__new__(CommitRecordingLabelRequested)
    for name, value in (
        ("command_id", "hostile-label"),
        ("record_id", "record.wav"),
        ("label", "OK"),
        ("export_outcome", Hostile()),
    ):
        object.__setattr__(hostile, name, value)
    oversized = CommitRecordingLabelRequested(
        "oversized-label", "record.wav", "OK", {"payload": "x" * 20_000}
    )

    assert controller.handle_commit_recording_label(hostile) is False
    assert controller.handle_commit_recording_label(oversized) is False
    assert calls == []
    assert hostile_calls == []


def test_label_recovery_never_clears_checkpoint_and_retries_only_unconfirmed_layers(
    tmp_path,
):
    from ui.sequence.sequence_recording_service import (
        RecordingLabelContext,
        RecordingLabelService,
    )

    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    count_path = tmp_path / "count.dat"
    count_path.write_bytes(b"old-count")
    events = []
    destructive_fallbacks = []

    class RecoverableMapping(dict):
        restore_attempts = 0

        def update(self, values=(), **kwargs):
            normalized = dict(values, **kwargs)
            if normalized.get("labels") == "not_labeled" and events:
                self.restore_attempts += 1
                events.append(f"mapping-restore-{self.restore_attempts}")
                if self.restore_attempts <= 3:
                    return False
            return super().update(normalized)

        def clear(self):
            if events and self.get("labels") == "OK":
                destructive_fallbacks.append("mapping-clear")
            return super().clear()

    info = RecoverableMapping(
        {"file_path": str(original), "labels": "not_labeled", "stable": True}
    )
    view_state = {"value": "old", "plot": "old"}

    class CountPersistence:
        restore_attempts = 0

        def checkpoint(self, _command):
            return count_path.read_bytes()

        def reconcile(self, checkpoint):
            return checkpoint

        def commit(self, *_args):
            count_path.write_bytes(b"new-count")
            events.append("count-forward")
            return {"count_file_updated": True}

        def restore(self, checkpoint, _error):
            self.restore_attempts += 1
            events.append(f"count-restore-{self.restore_attempts}")
            if self.restore_attempts <= 3:
                return False
            count_path.write_bytes(checkpoint)
            return True

        def fail_closed(self, *_args):
            destructive_fallbacks.append("count-delete")
            count_path.unlink(missing_ok=True)
            return True

    class Projection:
        restore_attempts = 0
        apply_attempts = 0

        def capture_label_projection(self, _command):
            return dict(view_state)

        def apply_label_projection(self, *_args):
            self.apply_attempts += 1
            events.append(f"view-forward-{self.apply_attempts}")
            if self.apply_attempts == 2:
                assert count_path.read_bytes() == b"new-count"
                assert info["labels"] == "OK"
            view_state.update(value="new", plot="new")
            if self.apply_attempts == 1:
                raise RuntimeError("force recovery")
            return True

        def restore_label_projection(self, checkpoint, _error):
            self.restore_attempts += 1
            events.append(f"view-restore-{self.restore_attempts}")
            if self.restore_attempts <= 3:
                return False
            view_state.clear()
            view_state.update(checkpoint)
            return True

        def fail_closed_label_projection(self, *_args):
            destructive_fallbacks.append("view-clear")
            view_state.clear()
            return True

    database_restores = []

    def update(updated, _old_path):
        if updated["labels"] == "not_labeled":
            database_restores.append(dict(updated))
            events.append("database-restore")
        else:
            events.append("database-forward")
        return error_code.OK, "updated"

    file_rollbacks = []

    def replace_file(source, destination):
        file_rollbacks.append((source, destination))
        os.replace(source, destination)

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        events.append("file-forward")
        return str(target)

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=CountPersistence(),
        replace_file=replace_file,
        root_dir="",
    )
    bus = SequenceEventBus()
    committed = []
    failed = []
    bus.events.recording_label_committed.connect(committed.append)
    bus.events.recording_label_commit_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_label_projection=Projection()),
        label_service=service,
        connect_queued=False,
    )
    command = CommitRecordingLabelRequested(
        "non-destructive-recovery",
        str(original),
        "OK",
        {"nested": {"attempt": 1}},
    )
    modified = CommitRecordingLabelRequested(
        "non-destructive-recovery",
        str(original),
        "NG",
        {"nested": {"attempt": 1}},
    )

    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    assert destructive_fallbacks == []
    assert count_path.read_bytes() == b"new-count"
    assert dict(info)["labels"] == "OK"
    assert view_state == {"value": "new", "plot": "new"}
    assert controller.handle_commit_recording_label(modified) is False

    assert controller.handle_commit_recording_label(command) is False
    assert controller.handle_commit_recording_label(command) is False
    assert committed == [] and failed == []
    assert controller.handle_commit_recording_label(command) is True

    assert len(committed) == 1 and failed == []
    assert destructive_fallbacks == []
    assert len(database_restores) == 1
    assert len(file_rollbacks) == 1
    assert events.count("view-restore-1") == 1
    assert events.count("view-restore-4") == 1
    assert events.count("count-restore-4") == 1
    assert events.count("mapping-restore-4") == 1
    assert controller.handle_commit_recording_label(command) is False
