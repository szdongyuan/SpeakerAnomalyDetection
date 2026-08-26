from __future__ import annotations

import ast
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui.sequence.sequence_recording_model import (
    RecordingCounterStatistics,
    RecordingModel,
    RecordingStatisticsSnapshot,
)
from ui.sequence.sequence_recording_service import (
    RecordingCountBoardPersistence,
    RecordingLabelContext,
    RecordingLabelRecoveryPending,
    RecordingLabelService,
    RecordingStatisticsService,
)
from ui.sequence.sequence_recording_view import (
    SequenceRecordingStatisticsProjection,
)
from ui.sequence.sequence_messages import CommitRecordingLabelRequested


ROOT = Path(__file__).resolve().parents[2]
TODAY = "2026-08-21"
TEST_PAYLOAD = (
    "total: 0\n"
    "ok: 0\n"
    "ng: 0\n"
    "ok_percent: 0%\n"
    f"datatime: {TODAY}\n"
).replace("\n", os.linesep).encode("utf-8")
MARK_PAYLOAD = json.dumps(
    {
        "total": 0,
        "ok": 0,
        "ng": 0,
        "not_labels": 0,
        "datatime": TODAY,
    },
    indent=4,
).replace("\n", os.linesep).encode("utf-8")


def _persistence(tmp_path: Path, *, replace_file=os.replace):
    test_path = tmp_path / "log" / "test_result_log" / f"{TODAY}.dat"
    mark_path = tmp_path / "ui" / "ui_config" / "mark_result.json"
    return (
        RecordingCountBoardPersistence(
            test_path_provider=lambda: test_path,
            mark_path_provider=lambda: mark_path,
            replace_file=replace_file,
            date_provider=lambda: TODAY,
        ),
        test_path,
        mark_path,
    )


def test_statistics_persistence_initializes_exact_legacy_payloads_atomically(tmp_path):
    replacements = []

    def replace_file(source, target):
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == target_path.parent
        replacements.append((source_path.name, target_path.name))
        os.replace(source, target)

    persistence, test_path, mark_path = _persistence(
        tmp_path, replace_file=replace_file
    )

    snapshot = persistence.initialize_statistics()

    assert test_path.read_bytes() == TEST_PAYLOAD
    assert mark_path.read_bytes() == MARK_PAYLOAD
    assert snapshot.test.total == snapshot.test.ok == snapshot.test.ng == 0
    assert snapshot.test.ok_percent == "0%"
    assert snapshot.test.datatime == TODAY
    assert snapshot.mark.total == snapshot.mark.ok == snapshot.mark.ng == 0
    assert snapshot.mark.not_labels == 0
    assert len(replacements) == 2

    current_payload = json.dumps(
        {"total": 7, "ok": 5, "ng": 2, "not_labels": 3, "datatime": TODAY},
        indent=4,
    ).encode("utf-8")
    mark_path.write_bytes(current_payload)
    persistence.initialize_statistics()
    assert mark_path.read_bytes() == current_payload

    stale = json.loads(current_payload)
    stale["datatime"] = "2026-08-20"
    mark_path.write_text(json.dumps(stale, indent=4), encoding="utf-8")
    persistence.initialize_statistics()
    assert mark_path.read_bytes() == MARK_PAYLOAD


def test_statistics_startup_reads_each_owner_file_once_and_projects_one_snapshot(
    tmp_path, monkeypatch
):
    persistence, test_path, mark_path = _persistence(tmp_path)
    original_test, _original_mark = _seed_nonzero_files(test_path, mark_path)
    externally_mutated_test = _dated_test_payload(TODAY, total=99)
    original_read_bytes = Path.read_bytes
    original_write_bytes = Path.write_bytes
    read_calls = {test_path.resolve(): 0, mark_path.resolve(): 0}

    def changing_read_bytes(path):
        resolved = path.resolve()
        payload = original_read_bytes(path)
        if resolved in read_calls:
            read_calls[resolved] += 1
        if resolved == test_path.resolve() and read_calls[resolved] == 1:
            original_write_bytes(path, externally_mutated_test)
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    model = RecordingModel()
    projection = _Projection()
    service = RecordingStatisticsService(model, persistence, projection)

    assert service.initialize_statistics() is True

    assert read_calls == {test_path.resolve(): 1, mark_path.resolve(): 1}
    assert model.statistics.test.total == 9
    assert projection.last_snapshot is model.statistics
    assert projection.last_snapshot.test.total == 9
    assert original_read_bytes(test_path) == externally_mutated_test
    assert original_test != externally_mutated_test


class _Projection:
    def __init__(self):
        self.value = "old-view"
        self.fail_with = None
        self.service = None
        self.nested_result = None
        self.last_snapshot = None

    def capture_statistics_projection(self):
        return self.value

    def apply_statistics_projection(self, _snapshot, *, reset_runtime):
        self.last_snapshot = _snapshot
        self.value = "reset-view" if reset_runtime else "initialized-view"
        if self.service is not None:
            self.nested_result = self.service.reset_statistics()
            self.service = None
        if self.fail_with is False:
            return False
        if isinstance(self.fail_with, BaseException):
            error = self.fail_with
            self.fail_with = None
            raise error
        return True

    def restore_statistics_projection(self, checkpoint, _error):
        self.value = checkpoint
        return True


class _StatisticsProjectionWidget:
    def __init__(self, text="serial", *, enabled=True, read_only=False):
        self._text = text
        self._enabled = enabled
        self._read_only = read_only

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, value):
        self._enabled = value

    def setDisabled(self, value):
        self._enabled = not value

    def isReadOnly(self):
        return self._read_only

    def setReadOnly(self, value):
        self._read_only = value


class _HostileStatisticsRuntime:
    def __getattribute__(self, name):
        if name == "clicked_player_flag":
            raise AssertionError("statistics projection accessed dead player state")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "clicked_player_flag":
            raise AssertionError("statistics projection recreated dead player state")
        object.__setattr__(self, name, value)


def test_statistics_projection_never_accesses_dead_clicked_player_state():
    runtime = _HostileStatisticsRuntime()
    runtime.recorded_path = "recorded.wav"
    runtime.signal_info = {"sample_rate": 48_000}
    runtime.lineedit_s_or_n = _StatisticsProjectionWidget()
    runtime.replayer_btn = _StatisticsProjectionWidget()
    runtime.data_btn = _StatisticsProjectionWidget()
    runtime._sn_clear_on_next_scan = True
    runtime.data_struct = SimpleNamespace(
        store_wave_data=[1.0],
        store_wave_data_multi=[[1.0]],
        wav_calibration_metadata={"v2pa": 1.0},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    runtime.update_player_btn_is_paused = lambda: None
    projection = SequenceRecordingStatisticsProjection(runtime)

    checkpoint = projection.capture_statistics_projection()
    assert set(checkpoint) == {"runtime"}
    assert projection.apply_statistics_projection(None, reset_runtime=True)
    assert runtime.data_struct.store_wave_data is None
    assert runtime.data_struct.store_wave_data_multi is None
    assert runtime.data_struct.wav_calibration_metadata is None
    assert runtime.data_struct.wav_calibration_metadata_authoritative is False
    assert runtime.data_struct.wav_calibration_warning_shown is False
    assert runtime._sn_clear_on_next_scan is False
    assert runtime.replayer_btn.isEnabled() is False
    assert runtime.data_btn.isEnabled() is False

    assert projection.restore_statistics_projection(checkpoint, RuntimeError("undo"))
    assert runtime.data_struct.store_wave_data == [1.0]
    assert runtime.data_struct.store_wave_data_multi == [[1.0]]
    assert runtime.data_struct.wav_calibration_metadata == {"v2pa": 1.0}
    assert runtime.data_struct.wav_calibration_metadata_authoritative is True
    assert runtime.data_struct.wav_calibration_warning_shown is True
    assert runtime._sn_clear_on_next_scan is True
    assert runtime.replayer_btn.isEnabled() is True
    assert runtime.data_btn.isEnabled() is True


def test_dead_clicked_player_state_is_absent_from_production_facade_and_projection():
    occurrences = []
    for relative_path in (
        "ui/sequence/sequence_widget.py",
        "ui/sequence/sequence_recording_view.py",
    ):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "clicked_player_flag"
            ) or (
                isinstance(node, ast.Constant)
                and node.value == "clicked_player_flag"
            ):
                occurrences.append((relative_path, node.lineno))

    assert occurrences == []


def _seed_nonzero_files(test_path: Path, mark_path: Path) -> tuple[bytes, bytes]:
    test_path.parent.mkdir(parents=True, exist_ok=True)
    mark_path.parent.mkdir(parents=True)
    test_bytes = (
        "total: 9\nok: 6\nng: 3\nok_percent: 66.67%\n"
        f"datatime: {TODAY}\n"
    ).encode("utf-8")
    mark_bytes = json.dumps(
        {"total": 4, "ok": 3, "ng": 1, "not_labels": 2, "datatime": TODAY},
        indent=4,
    ).encode("utf-8")
    test_path.write_bytes(test_bytes)
    mark_path.write_bytes(mark_bytes)
    return test_bytes, mark_bytes


class _ChangingStatisticsInputs:
    def __init__(self, tmp_path: Path, count: int = 4):
        self.test_paths = [
            tmp_path / f"operation-{index}" / "test.dat"
            for index in range(count)
        ]
        self.mark_paths = [
            tmp_path / f"operation-{index}" / "mark.json"
            for index in range(count)
        ]
        self.dates = [f"2026-08-{21 + index:02d}" for index in range(count)]
        self.test_calls = 0
        self.mark_calls = 0
        self.date_calls = 0
        self.allow_resolution = True

    def _resolved(self, values, name):
        if not self.allow_resolution:
            raise AssertionError(f"{name} provider was called after operation capture")
        call_name = f"{name}_calls"
        index = getattr(self, call_name)
        setattr(self, call_name, index + 1)
        return values[index]

    def test_path(self):
        return self._resolved(self.test_paths, "test")

    def mark_path(self):
        return self._resolved(self.mark_paths, "mark")

    def today(self):
        return self._resolved(self.dates, "date")


def _changing_persistence(inputs, *, replace_file=os.replace):
    return RecordingCountBoardPersistence(
        test_path_provider=inputs.test_path,
        mark_path_provider=inputs.mark_path,
        date_provider=inputs.today,
        replace_file=replace_file,
    )


def _dated_test_payload(date: str, *, total: int = 0) -> bytes:
    return (
        f"total: {total}\nok: 0\nng: {total}\nok_percent: 0%\n"
        f"datatime: {date}\n"
    ).encode("utf-8")


def _dated_mark_payload(date: str, *, total: int = 0) -> bytes:
    return json.dumps(
        {
            "total": total,
            "ok": 0,
            "ng": total,
            "not_labels": 0,
            "datatime": date,
        },
        indent=4,
    ).encode("utf-8")


def _seed_changing_first_pair(inputs):
    test_path = inputs.test_paths[0]
    mark_path = inputs.mark_paths[0]
    test_path.parent.mkdir(parents=True)
    test_bytes = _dated_test_payload(inputs.dates[0], total=7)
    mark_bytes = _dated_mark_payload("2026-08-20", total=5)
    test_path.write_bytes(test_bytes)
    mark_path.write_bytes(mark_bytes)
    return test_bytes, mark_bytes


def test_default_test_path_uses_the_single_frozen_operation_date():
    dates = iter(("2031-01-02", "2031-01-03"))
    persistence = RecordingCountBoardPersistence(date_provider=lambda: next(dates))

    paths = persistence.resolve_operation_paths()

    assert paths.datatime == "2031-01-02"
    assert paths.test.name == "2031-01-02.dat"
    assert next(dates) == "2031-01-03"


@pytest.mark.parametrize(
    "invalid_date",
    [
        "2026-8-1",
        "2026-02-30",
        " 2026-08-21",
        "2026-08-21 ",
        "2026/08/21",
        "../2026-08-21",
    ],
)
def test_operation_paths_reject_ambiguous_or_invalid_calendar_date(
    tmp_path, invalid_date
):
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "test.dat",
        mark_path_provider=lambda: tmp_path / "mark.json",
        date_provider=lambda: invalid_date,
    )

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        persistence.resolve_operation_paths()

    assert list(tmp_path.iterdir()) == []


def test_operation_paths_are_absolute_canonical_and_distinct(tmp_path):
    test_raw = tmp_path / "nested" / ".." / "test.dat"
    mark_raw = tmp_path / "." / "mark.json"
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: test_raw,
        mark_path_provider=lambda: mark_raw,
        date_provider=lambda: TODAY,
    )

    paths = persistence.resolve_operation_paths()

    assert paths.test.is_absolute() and paths.mark.is_absolute()
    assert str(paths.test) == os.path.normcase(str(test_raw.resolve()))
    assert str(paths.mark) == os.path.normcase(str(mark_raw.resolve()))
    assert ".." not in paths.test.parts


@pytest.mark.parametrize("alias_kind", ["same", "case", "hardlink"])
def test_operation_paths_reject_aliasing_test_and_mark_targets(tmp_path, alias_kind):
    first = tmp_path / "Count.dat"
    if alias_kind == "same":
        second = first
    elif alias_kind == "case":
        second = tmp_path / "count.DAT"
    else:
        first.write_bytes(b"counts")
        second = tmp_path / "hardlink.json"
        os.link(first, second)
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: first,
        mark_path_provider=lambda: second,
        date_provider=lambda: TODAY,
    )

    with pytest.raises(ValueError, match="distinct"):
        persistence.resolve_operation_paths()


@pytest.mark.parametrize("invalid_target", ["", "   ", "bad\x00target"])
def test_operation_paths_reject_invalid_target_text_before_checkpoint(
    tmp_path, invalid_target
):
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: invalid_target,
        mark_path_provider=lambda: tmp_path / "mark.json",
        date_provider=lambda: TODAY,
    )

    with pytest.raises((TypeError, ValueError, OSError)):
        persistence.resolve_operation_paths()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("section", ["test", "mark"])
def test_operation_paths_reject_existing_directory_target(tmp_path, section):
    directory = tmp_path / "counts"
    directory.mkdir()
    test_target = directory if section == "test" else tmp_path / "test.dat"
    mark_target = directory if section == "mark" else tmp_path / "mark.json"
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: test_target,
        mark_path_provider=lambda: mark_target,
        date_provider=lambda: TODAY,
    )

    with pytest.raises(ValueError, match="regular file"):
        persistence.resolve_operation_paths()

    assert list(directory.iterdir()) == []


def test_operation_paths_reject_non_directory_existing_ancestor(tmp_path):
    ancestor = tmp_path / "not-a-directory"
    ancestor.write_bytes(b"file")
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: ancestor / "nested" / "test.dat",
        mark_path_provider=lambda: tmp_path / "mark.json",
        date_provider=lambda: TODAY,
    )

    with pytest.raises(ValueError, match="ancestor"):
        persistence.resolve_operation_paths()

    assert ancestor.read_bytes() == b"file"


def test_operation_paths_reject_windows_reserved_target_name(tmp_path):
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: tmp_path / "CON.dat",
        mark_path_provider=lambda: tmp_path / "mark.json",
        date_provider=lambda: TODAY,
    )

    with pytest.raises(ValueError, match="invalid"):
        persistence.resolve_operation_paths()


@pytest.mark.parametrize("provider", ["date", "test", "mark"])
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("provider failed"), KeyboardInterrupt("provider failed"), SystemExit("provider failed")],
)
def test_operation_path_provider_baseexception_precedes_checkpoint_and_mutation(
    tmp_path, provider, failure
):
    def value(name, result):
        if provider == name:
            raise failure
        return result

    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: value("test", tmp_path / "test.dat"),
        mark_path_provider=lambda: value("mark", tmp_path / "mark.json"),
        date_provider=lambda: value("date", TODAY),
    )

    with pytest.raises(type(failure), match="provider failed"):
        persistence.checkpoint_statistics()

    assert list(tmp_path.iterdir()) == []


def _assert_statistics_failure(service, operation, failure):
    method = getattr(service, f"{operation}_statistics")
    if isinstance(failure, Exception):
        assert method() is False
    else:
        with pytest.raises(type(failure), match="operation failed"):
            method()


@pytest.mark.parametrize("operation", ["initialize", "reset"])
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("operation failed"),
        KeyboardInterrupt("operation failed"),
        SystemExit("operation failed"),
    ],
)
def test_statistics_projection_failure_uses_one_frozen_path_pair_and_date(
    tmp_path, operation, failure
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    old_test, old_mark = _seed_changing_first_pair(inputs)
    persistence = _changing_persistence(inputs)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime="2026-08-20", not_labels=0
        ),
    )
    projection = _Projection()
    projection.fail_with = failure
    service = RecordingStatisticsService(model, persistence, projection)

    _assert_statistics_failure(service, operation, failure)

    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 1
    assert inputs.test_paths[0].read_bytes() == old_test
    assert inputs.mark_paths[0].read_bytes() == old_mark
    assert not inputs.test_paths[1].exists()
    assert not inputs.mark_paths[1].exists()


@pytest.mark.parametrize("operation", ["initialize", "reset"])
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("operation failed"),
        KeyboardInterrupt("operation failed"),
        SystemExit("operation failed"),
    ],
)
def test_statistics_file_failure_restores_the_exact_frozen_path_pair(
    tmp_path, operation, failure
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    old_test, old_mark = _seed_changing_first_pair(inputs)
    attempts = 0

    def replace_then_fail(source, target):
        nonlocal attempts
        attempts += 1
        os.replace(source, target)
        if attempts == 1:
            raise failure

    persistence = _changing_persistence(inputs, replace_file=replace_then_fail)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime="2026-08-20", not_labels=0
        ),
    )
    service = RecordingStatisticsService(model, persistence, _Projection())

    _assert_statistics_failure(service, operation, failure)

    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 1
    assert inputs.test_paths[0].read_bytes() == old_test
    assert inputs.mark_paths[0].read_bytes() == old_mark
    assert not inputs.test_paths[1].exists()
    assert not inputs.mark_paths[1].exists()
    assert not list(tmp_path.rglob(".*.initialize"))
    assert not list(tmp_path.rglob(".*.reset"))


@pytest.mark.parametrize("operation", ["initialize", "reset"])
@pytest.mark.parametrize("behavior", ["no-op", "wrong", "missing", "unreadable"])
def test_statistics_forward_replace_requires_exact_readback_before_projection(
    tmp_path, monkeypatch, operation, behavior
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    old_test, old_mark = _seed_changing_first_pair(inputs)
    target = inputs.mark_paths[0] if operation == "initialize" else inputs.test_paths[0]
    original_read_bytes = Path.read_bytes
    corrupt_once = True
    unreadable_armed = False

    def read_bytes(path):
        nonlocal unreadable_armed
        if unreadable_armed and path == target:
            unreadable_armed = False
            raise PermissionError("destination is unreadable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    def replace_file(source, destination):
        nonlocal corrupt_once, unreadable_armed
        if not corrupt_once:
            return os.replace(source, destination)
        corrupt_once = False
        if behavior == "no-op":
            return None
        os.replace(source, destination)
        destination_path = Path(destination)
        if behavior == "wrong":
            wrong = (
                _dated_mark_payload(inputs.dates[0], total=91)
                if operation == "initialize"
                else _dated_test_payload(inputs.dates[0], total=91)
            )
            destination_path.write_bytes(wrong)
        elif behavior == "missing":
            destination_path.unlink()
        else:
            unreadable_armed = True

    persistence = _changing_persistence(inputs, replace_file=replace_file)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime="2026-08-20", not_labels=0
        ),
    )
    projection = _Projection()
    service = RecordingStatisticsService(model, persistence, projection)

    assert getattr(service, f"{operation}_statistics")() is False

    assert projection.value == "old-view"
    assert inputs.test_paths[0].read_bytes() == old_test
    assert inputs.mark_paths[0].read_bytes() == old_mark
    assert not list(tmp_path.rglob(".*.initialize"))
    assert not list(tmp_path.rglob(".*.reset"))
    assert not list(tmp_path.rglob(".*.rollback"))


@pytest.mark.parametrize("behavior", ["no-op", "wrong", "missing", "unreadable"])
def test_label_forward_count_replace_requires_exact_readback_and_rolls_back(
    tmp_path, monkeypatch, behavior
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    test_path = inputs.test_paths[0]
    count_path = inputs.mark_paths[0]
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(_dated_test_payload(inputs.dates[0], total=7))
    old_count = _dated_mark_payload(inputs.dates[0], total=5)
    count_path.write_bytes(old_count)
    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)
    original_read_bytes = Path.read_bytes
    corrupt_once = True
    unreadable_armed = False

    def read_bytes(path):
        nonlocal unreadable_armed
        if unreadable_armed and path == count_path:
            unreadable_armed = False
            raise PermissionError("destination is unreadable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    def replace_file(source, destination):
        nonlocal corrupt_once, unreadable_armed
        if not corrupt_once:
            return os.replace(source, destination)
        corrupt_once = False
        if behavior == "no-op":
            return None
        os.replace(source, destination)
        destination_path = Path(destination)
        if behavior == "wrong":
            destination_path.write_bytes(
                _dated_mark_payload(inputs.dates[0], total=91)
            )
        elif behavior == "missing":
            destination_path.unlink()
        else:
            unreadable_armed = True

    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime=inputs.dates[0], not_labels=0
        ),
    )
    persistence = _changing_persistence(inputs, replace_file=replace_file)
    persistence.statistics_model = model

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        return 0, "ok"

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested("verify-count", str(original), "OK", ())

    with pytest.raises((RuntimeError, OSError)):
        service.commit(command, SimpleNamespace(
            capture_label_projection=lambda _command: "view",
            apply_label_projection=lambda *_args: True,
            restore_label_projection=lambda *_args: True,
        ))

    assert original.read_bytes() == b"audio"
    assert count_path.read_bytes() == old_count
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert database == info
    assert model.statistics.mark.total == 5
    assert not list(tmp_path.rglob(".*.commit"))
    assert not list(tmp_path.rglob(".*.rollback"))


def test_label_count_rollback_noop_replace_stays_recovery_pending_until_verified(
    tmp_path
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    test_path = inputs.test_paths[0]
    count_path = inputs.mark_paths[0]
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(_dated_test_payload(inputs.dates[0], total=7))
    old_count = _dated_mark_payload(inputs.dates[0], total=5)
    count_path.write_bytes(old_count)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime=inputs.dates[0], not_labels=0
        ),
    )
    replacements = 0

    def replace_file(source, destination):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            return None
        return os.replace(source, destination)

    persistence = _changing_persistence(inputs, replace_file=replace_file)
    persistence.statistics_model = model
    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    class Projection:
        attempts = 0

        def capture_label_projection(self, _command):
            return "view"

        def apply_label_projection(self, *_args):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("projection failed")
            return True

        def restore_label_projection(self, *_args):
            return True

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda *_args: (0, "ok"),
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested("rollback-readback", str(original), "OK", ())
    projection = Projection()

    with pytest.raises(RecordingLabelRecoveryPending) as captured:
        service.commit(command, projection)

    assert captured.value.rollback_outcome.pending == ("count-board",)
    assert count_path.read_bytes() != old_count
    assert not list(tmp_path.rglob(".*.rollback"))

    outcome = service.commit(command, projection)

    assert outcome["view_projected"] is True
    assert model.statistics.mark.total == 1
    assert model.statistics.mark.datatime == inputs.dates[1]
    assert not list(tmp_path.rglob(".*.rollback"))


@pytest.mark.parametrize("operation", ["initialize", "reset"])
def test_statistics_success_uses_first_pair_and_retry_resolves_next_pair(
    tmp_path, operation
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    next_test = next_mark = None
    if operation == "reset":
        for index in range(2):
            inputs.test_paths[index].parent.mkdir(parents=True)
            inputs.test_paths[index].write_bytes(
                _dated_test_payload(inputs.dates[index], total=7)
            )
            inputs.mark_paths[index].write_bytes(
                _dated_mark_payload(inputs.dates[index], total=5)
            )
        next_test = inputs.test_paths[1].read_bytes()
        next_mark = inputs.mark_paths[1].read_bytes()
    persistence = _changing_persistence(inputs)
    service = RecordingStatisticsService(RecordingModel(), persistence, _Projection())
    method = getattr(service, f"{operation}_statistics")

    assert method() is True
    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 1
    assert inputs.test_paths[0].exists()
    assert inputs.mark_paths[0].exists()
    if operation == "initialize":
        assert not inputs.test_paths[1].exists()
        assert not inputs.mark_paths[1].exists()
    else:
        assert inputs.test_paths[1].read_bytes() == next_test
        assert inputs.mark_paths[1].read_bytes() == next_mark

    assert method() is True
    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 2
    assert inputs.test_paths[1].exists()
    assert inputs.mark_paths[1].exists()


@pytest.mark.parametrize("operation", ["initialize", "reset"])
def test_statistics_retry_after_rollback_resolves_a_new_operation_pair(
    tmp_path, operation
):
    inputs = _ChangingStatisticsInputs(tmp_path)
    old_test, old_mark = _seed_changing_first_pair(inputs)
    if operation == "reset":
        inputs.test_paths[1].parent.mkdir(parents=True)
        inputs.test_paths[1].write_bytes(
            _dated_test_payload(inputs.dates[1], total=8)
        )
        inputs.mark_paths[1].write_bytes(
            _dated_mark_payload(inputs.dates[1], total=6)
        )
    persistence = _changing_persistence(inputs)
    projection = _Projection()
    projection.fail_with = False
    service = RecordingStatisticsService(RecordingModel(), persistence, projection)
    method = getattr(service, f"{operation}_statistics")

    assert method() is False
    assert inputs.test_paths[0].read_bytes() == old_test
    assert inputs.mark_paths[0].read_bytes() == old_mark
    projection.fail_with = None

    assert method() is True
    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 2
    assert inputs.test_paths[1].exists()
    assert inputs.mark_paths[1].exists()


@pytest.mark.parametrize("section", ["test", "mark"])
@pytest.mark.parametrize("existing", [False, True])
def test_label_reconciles_selected_section_from_frozen_file_or_empty_daily_base(
    tmp_path, section, existing
):
    operation_date = "2026-08-22"
    test_path = tmp_path / "custom-test-name.counts"
    mark_path = tmp_path / "custom-mark-name.counts"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        if section == "test":
            test_path.write_bytes(_dated_test_payload(operation_date, total=4))
        else:
            mark_path.write_bytes(_dated_mark_payload(operation_date, total=4))
    model = RecordingModel()
    old_snapshot = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=9, ok=6, ng=3, datatime="2026-08-21", ok_percent="66.67%"
        ),
        mark=RecordingCounterStatistics(
            total=8, ok=5, ng=3, datatime="2026-08-21", not_labels=0
        ),
    )
    model.statistics = old_snapshot
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: test_path,
        mark_path_provider=lambda: mark_path,
        date_provider=lambda: operation_date,
        statistics_model=model,
    )
    command = SimpleNamespace(
        export_outcome=(
            {"continuation": "post-analysis"} if section == "test" else {}
        ),
        label="OK",
    )

    checkpoint = persistence.checkpoint(command)
    persistence.reconcile(checkpoint)
    persistence.commit(command, checkpoint)

    expected_total = 5 if existing else 1
    selected = model.statistics.test if section == "test" else model.statistics.mark
    assert selected.total == expected_total
    assert selected.datatime == operation_date
    if section == "test":
        assert model.statistics.mark == old_snapshot.mark
        assert f"datatime: {operation_date}" in test_path.read_text(encoding="utf-8")
    else:
        assert model.statistics.test == old_snapshot.test
        assert json.loads(mark_path.read_text(encoding="utf-8"))["datatime"] == operation_date


@pytest.mark.parametrize(
    "payload",
    [
        _dated_test_payload("2026-08-20", total=4),
        b"total: 4\nok: 5\nng: 0\nok_percent: 125%\ndatatime: 2026-08-22\n",
        b"total: 0\ndatatime: 2026-08-22\n",
    ],
)
def test_label_reconcile_rejects_invalid_frozen_persisted_counts_or_date(
    tmp_path, payload
):
    test_path = tmp_path / "custom-test.counts"
    mark_path = tmp_path / "custom-mark.counts"
    test_path.write_bytes(payload)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=9, ok=6, ng=3, datatime="2026-08-21", ok_percent="66.67%"
        ),
        mark=RecordingCounterStatistics(
            total=8, ok=5, ng=3, datatime="2026-08-21", not_labels=0
        ),
    )
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: test_path,
        mark_path_provider=lambda: mark_path,
        date_provider=lambda: "2026-08-22",
        statistics_model=model,
    )
    command = SimpleNamespace(
        export_outcome={"continuation": "post-analysis"}, label="OK"
    )
    checkpoint = persistence.checkpoint(command)

    with pytest.raises(ValueError, match="persisted"):
        persistence.reconcile(checkpoint)

    assert model.statistics.test.total == 9
    assert test_path.read_bytes() == payload


def test_label_cross_date_projection_failure_rolls_back_file_model_and_view(
    tmp_path
):
    operation_date = "2026-08-22"
    test_path = tmp_path / "daily-custom.counts"
    mark_path = tmp_path / "mark-custom.counts"
    old_snapshot = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=9, ok=6, ng=3, datatime="2026-08-21", ok_percent="66.67%"
        ),
        mark=RecordingCounterStatistics(
            total=8, ok=5, ng=3, datatime="2026-08-21", not_labels=0
        ),
    )
    model = RecordingModel()
    model.statistics = old_snapshot
    persistence = RecordingCountBoardPersistence(
        test_path_provider=lambda: test_path,
        mark_path_provider=lambda: mark_path,
        date_provider=lambda: operation_date,
        statistics_model=model,
    )
    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)
    view = {"statistics": old_snapshot}
    projected = []

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        return 0, "ok"

    class Projection:
        def capture_label_projection(self, _command):
            return dict(view)

        def apply_label_projection(self, _command, receipt, _checkpoint):
            projected.append(receipt.statistics_snapshot)
            view["statistics"] = receipt.statistics_snapshot
            raise RuntimeError("projection failed")

        def restore_label_projection(self, checkpoint, _error):
            view.clear()
            view.update(checkpoint)
            return True

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested(
        "cross-date",
        str(original),
        "OK",
        {"continuation": "post-analysis"},
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        service.commit(command, Projection())

    assert not test_path.exists()
    assert model.statistics == old_snapshot
    assert view == {"statistics": old_snapshot}
    assert projected[0].test.total == 1
    assert projected[0].test.datatime == operation_date
    assert original.read_bytes() == b"audio"
    assert database == info


def test_label_cross_date_retry_resolves_new_paths_and_uses_new_empty_base(tmp_path):
    inputs = _ChangingStatisticsInputs(tmp_path, count=2)
    old_snapshot = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=9, ok=6, ng=3, datatime="2026-08-20", ok_percent="66.67%"
        ),
        mark=RecordingCounterStatistics(
            total=8, ok=5, ng=3, datatime="2026-08-20", not_labels=0
        ),
    )
    model = RecordingModel()
    model.statistics = old_snapshot
    persistence = _changing_persistence(inputs)
    persistence.statistics_model = model
    original = tmp_path / "not_labeled" / "retry.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    database = dict(info)

    def move(path, label):
        target = tmp_path / label / Path(path).name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
        return str(target)

    def update(updated, _old_path):
        database.clear()
        database.update(updated)
        return 0, "ok"

    class Projection:
        attempts = 0

        def capture_label_projection(self, _command):
            return old_snapshot

        def apply_label_projection(self, _command, _receipt, _checkpoint):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("first projection failed")
            return True

        def restore_label_projection(self, _checkpoint, _error):
            return True

    projection = Projection()
    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=update,
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested(
        "cross-date-retry",
        str(original),
        "NG",
        {"continuation": "post-analysis"},
    )

    with pytest.raises(RuntimeError, match="first projection failed"):
        service.commit(command, projection)

    assert not inputs.test_paths[0].exists()
    assert model.statistics == old_snapshot
    assert original.exists()

    outcome = service.commit(command, projection)

    assert outcome["view_projected"] is True
    assert not inputs.test_paths[0].exists()
    assert inputs.test_paths[1].exists()
    assert model.statistics.test.total == 1
    assert model.statistics.test.ng == 1
    assert model.statistics.test.datatime == inputs.dates[1]
    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 2


def test_label_checkpoint_freezes_both_paths_and_date_before_commit(tmp_path):
    inputs = _ChangingStatisticsInputs(tmp_path)
    test_path = inputs.test_paths[0]
    mark_path = inputs.mark_paths[0]
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(_dated_test_payload(inputs.dates[0], total=7))
    mark_path.write_bytes(_dated_mark_payload(inputs.dates[0], total=5))
    persistence = _changing_persistence(inputs)
    model = RecordingModel()
    model.statistics = RecordingStatisticsSnapshot(
        test=RecordingCounterStatistics(
            total=7, ok=0, ng=7, datatime=inputs.dates[0], ok_percent="0%"
        ),
        mark=RecordingCounterStatistics(
            total=5, ok=0, ng=5, datatime=inputs.dates[0], not_labels=0
        ),
    )
    persistence.statistics_model = model
    command = SimpleNamespace(export_outcome={}, label="OK")

    checkpoint = persistence.checkpoint(command)
    command.export_outcome = {"continuation": "post-analysis"}
    inputs.allow_resolution = False
    persistence.commit(command, checkpoint)

    assert inputs.test_calls == inputs.mark_calls == inputs.date_calls == 1
    assert json.loads(mark_path.read_text(encoding="utf-8")) == {
        "total": 6,
        "ng": 5,
        "ok": 1,
        "datatime": inputs.dates[0],
    }
    assert persistence.restore(checkpoint, RuntimeError("projection failed"))
    assert mark_path.read_bytes() == _dated_mark_payload(inputs.dates[0], total=5)


def test_reset_statistics_rolls_back_files_model_and_view_then_allows_retry(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    old_test, old_mark = _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    model.recorded_path = "retained.wav"
    model.recorded_signal_info = {"file_path": "retained.wav"}
    model.current_recorded_count = 11
    model.last_play_count = 11
    old_model = model.statistics
    projection = _Projection()
    projection.fail_with = False
    service = RecordingStatisticsService(model, persistence, projection)

    assert service.reset_statistics() is False
    assert test_path.read_bytes() == old_test
    assert mark_path.read_bytes() == old_mark
    assert model.statistics == old_model
    assert model.last_play_count == 11
    assert projection.value == "old-view"
    assert service.operation_active is False

    projection.fail_with = None
    assert service.reset_statistics() is True
    assert test_path.read_bytes() == TEST_PAYLOAD
    assert mark_path.read_bytes() == old_mark
    assert model.statistics.test.total == 0
    assert model.statistics.mark.total == 4
    assert model.recorded_path == "retained.wav"
    assert model.recorded_signal_info == {"file_path": "retained.wav"}
    assert model.current_recorded_count == 11
    assert model.last_play_count is None
    assert projection.value == "reset-view"


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt("projection interrupted"), SystemExit("projection interrupted")],
)
def test_reset_statistics_rejects_reentry_and_recovers_before_reraising_baseexception(
    tmp_path, failure
):
    persistence, test_path, mark_path = _persistence(tmp_path)
    old_test, old_mark = _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    projection = _Projection()
    service = RecordingStatisticsService(model, persistence, projection)
    projection.service = service
    projection.fail_with = failure

    with pytest.raises(type(failure), match="projection interrupted"):
        service.reset_statistics()

    assert projection.nested_result is False
    assert test_path.read_bytes() == old_test
    assert mark_path.read_bytes() == old_mark
    assert projection.value == "old-view"
    assert service.operation_active is False
    assert service.reset_statistics() is True


def test_statistics_file_replace_rejection_restores_and_allows_next_retry(tmp_path):
    attempts = 0

    def replace_file(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return False
        os.replace(source, target)

    persistence, test_path, mark_path = _persistence(
        tmp_path, replace_file=replace_file
    )
    old_test, old_mark = _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    projection = _Projection()
    service = RecordingStatisticsService(model, persistence, projection)

    assert service.reset_statistics() is False
    assert test_path.read_bytes() == old_test
    assert mark_path.read_bytes() == old_mark
    assert projection.value == "old-view"
    assert service.reset_statistics() is True
    assert test_path.read_bytes() == TEST_PAYLOAD


def test_label_count_commit_and_restore_keep_recording_model_statistics_canonical(
    tmp_path,
):
    persistence, test_path, mark_path = _persistence(tmp_path)
    _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    persistence.statistics_model = model
    model.statistics = persistence.read_statistics()
    old_statistics = model.statistics
    # Deliberately divergent UI-like values must never be consulted.
    persistence.count_board = SimpleNamespace(
        total_line_edit=SimpleNamespace(text=lambda: "999"),
        ok_line_edit=SimpleNamespace(text=lambda: "998"),
        ng_line_edit=SimpleNamespace(text=lambda: "1"),
    )
    command = SimpleNamespace(
        export_outcome={"continuation": "post-analysis"}, label="OK"
    )
    checkpoint = persistence.checkpoint(command)

    persistence.commit(command, checkpoint)

    assert model.statistics.test.total == 10
    assert model.statistics.test.ok == 7
    assert model.statistics.test.ng == 3
    assert model.statistics.mark == old_statistics.mark
    assert persistence.restore(checkpoint, RuntimeError("later projection failed"))
    assert model.statistics == old_statistics


def test_statistics_snapshot_is_externally_immutable(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    _seed_nonzero_files(test_path, mark_path)
    snapshot = persistence.read_statistics()

    with pytest.raises(FrozenInstanceError):
        snapshot.test.total = 500


def test_label_commit_rejects_external_model_replacement_after_checkpoint(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    old_test, _old_mark = _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    persistence.statistics_model = model
    model.statistics = persistence.read_statistics()
    old_statistics = model.statistics
    command = SimpleNamespace(
        export_outcome={"continuation": "post-analysis"}, label="OK"
    )
    checkpoint = persistence.checkpoint(command)
    model.statistics = type(old_statistics)(
        test=type(old_statistics.test)(
            total=99,
            ok=99,
            ng=0,
            datatime=TODAY,
            ok_percent="100%",
        ),
        mark=old_statistics.mark,
    )

    with pytest.raises(RuntimeError, match="changed after label checkpoint"):
        persistence.commit(command, checkpoint)

    assert test_path.read_bytes() == old_test
    assert persistence.restore(checkpoint, RuntimeError("abort"))
    assert model.statistics == old_statistics


def test_stale_mark_initialization_updates_model_and_view_through_transaction(
    tmp_path,
):
    persistence, test_path, mark_path = _persistence(tmp_path)
    test_path.parent.mkdir(parents=True)
    mark_path.parent.mkdir(parents=True)
    test_path.write_bytes(TEST_PAYLOAD)
    stale = {"total": 8, "ok": 7, "ng": 1, "not_labels": 4, "datatime": "2026-08-20"}
    mark_path.write_text(json.dumps(stale, indent=4), encoding="utf-8")
    model = RecordingModel()
    projection = _Projection()
    service = RecordingStatisticsService(model, persistence, projection)

    assert service.initialize_statistics() is True

    assert model.statistics.mark.total == 0
    assert model.statistics.mark.datatime == TODAY
    assert projection.last_snapshot == model.statistics
    assert projection.value == "initialized-view"


def test_stale_mark_view_failure_restores_file_model_and_view(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    test_path.parent.mkdir(parents=True)
    mark_path.parent.mkdir(parents=True)
    test_path.write_bytes(TEST_PAYLOAD)
    stale = {"total": 8, "ok": 7, "ng": 1, "not_labels": 4, "datatime": "2026-08-20"}
    stale_bytes = json.dumps(stale, indent=4).encode("utf-8")
    mark_path.write_bytes(stale_bytes)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    old_statistics = model.statistics
    projection = _Projection()
    projection.fail_with = False
    service = RecordingStatisticsService(model, persistence, projection)

    assert service.initialize_statistics() is False

    assert mark_path.read_bytes() == stale_bytes
    assert model.statistics == old_statistics
    assert projection.value == "old-view"
    projection.fail_with = None
    assert service.initialize_statistics() is True


def test_reset_rejects_concurrent_label_statistics_identity_without_mutation(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    old_test, old_mark = _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    service = RecordingStatisticsService(model, persistence, _Projection())
    label_identity = ("label", "command-1")

    assert persistence.try_begin_operation(label_identity)
    assert service.reset_statistics() is False
    assert test_path.read_bytes() == old_test
    assert mark_path.read_bytes() == old_mark
    assert persistence.end_operation(label_identity)
    assert service.reset_statistics() is True


def test_label_rejects_active_reset_identity_before_any_side_effect(tmp_path):
    persistence, test_path, mark_path = _persistence(tmp_path)
    _seed_nonzero_files(test_path, mark_path)
    model = RecordingModel()
    model.statistics = persistence.read_statistics()
    persistence.statistics_model = model
    original = tmp_path / "not_labeled" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=lambda *_args: pytest.fail("busy label must not move a file"),
        count_board_persistence=persistence,
        root_dir="",
    )
    command = CommitRecordingLabelRequested("busy-label", str(original), "OK", ())
    reset_identity = ("statistics", 1)

    assert persistence.try_begin_operation(reset_identity)
    with pytest.raises(RuntimeError, match="statistics operation is already active"):
        service.commit(command, SimpleNamespace())

    assert original.read_bytes() == b"audio"
    assert info == {"file_path": str(original), "labels": "not_labeled"}
    assert persistence.end_operation(reset_identity)


def test_duplicate_label_reentry_is_rejected_while_first_transaction_completes(
    tmp_path,
):
    original = tmp_path / "not_labeled" / "sample.wav"
    target = tmp_path / "OK" / "sample.wav"
    original.parent.mkdir()
    original.write_bytes(b"audio")
    info = {"file_path": str(original), "labels": "not_labeled"}
    updates = []

    def move(path, _label):
        target.parent.mkdir()
        os.replace(path, target)
        return str(target)

    service = RecordingLabelService(
        context_provider=lambda: RecordingLabelContext(str(original), info),
        move_wav=move,
        update_label=lambda updated, old: updates.append((dict(updated), old)) or (0, "ok"),
        root_dir="",
    )
    command = CommitRecordingLabelRequested("duplicate-label", str(original), "OK", ())

    class Projection:
        nested_attempted = False

        def capture_label_projection(self, _command):
            if not self.nested_attempted:
                self.nested_attempted = True
                with pytest.raises(RuntimeError, match="label operation is already active"):
                    service.commit(command, self)
            return "checkpoint"

        def apply_label_projection(self, *_args):
            return True

        def restore_label_projection(self, *_args):
            return True

    outcome = service.commit(command, Projection())

    assert outcome["database_updated"] is True
    assert target.read_bytes() == b"audio"
    assert len(updates) == 1


def test_sequence_window_statistics_methods_are_owner_delegates_without_file_io():
    source_path = ROOT / "ui" / "sequence" / "sequence_widget.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in facade.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name in (
        "init_result_files",
        "reset_test_reord",
        "on_reset_statistics_clicked",
    ):
        method = methods[name]
        names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
        }
        assert names.isdisjoint({"open", "json", "os", "datetime", "DEFAULT_DIR"})
        assert attributes.isdisjoint(
            {
                "set_test_text",
                "set_mark_text",
                "store_wave_data",
                "store_wave_data_multi",
                "last_play_count",
            }
        )
        assert "recording_statistics_service" in attributes

    assert "init_mark_result_file" not in methods

    connect_method = methods["set_member_connect"]
    connect_source = ast.unparse(connect_method)
    assert "reset_btn.clicked.connect(self.recording_statistics_service.reset_statistics)" in connect_source


def test_count_board_is_a_pure_view_without_statistics_file_algorithms():
    source_path = ROOT / "ui" / "sequence" / "sequencement_count_board.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    count_board = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceCountBoard"
    )
    method_names = {
        node.name
        for node in count_board.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert method_names.isdisjoint(
        {
            "_parse_test_log",
            "set_test_text",
            "set_mark_text",
            "set_test_result_file",
            "set_mark_result_file",
        }
    )
    assert not any("file" in name.lower() for name in method_names)
    assert names.isdisjoint(
        {"open", "Path", "json", "datetime", "DEFAULT_DIR", "ensure_test_result_file"}
    )
    assert attributes.isdisjoint(
        {
            "open",
            "read_bytes",
            "read_text",
            "replace",
            "write_bytes",
            "write_text",
        }
    )


def test_sequence_window_constructs_count_board_before_single_owner_initialization():
    source_path = ROOT / "ui" / "sequence" / "sequence_widget.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    constructor = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    board_statement_indexes = []
    initialize_statement_indexes = []
    for index, statement in enumerate(constructor.body):
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            function = statement.value.func
            if isinstance(function, ast.Name) and function.id == "SequenceCountBoard":
                board_statement_indexes.append(index)
        calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)]
        if any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "init_result_files"
            for call in calls
        ):
            initialize_statement_indexes.append(index)

    assert len(board_statement_indexes) == 1
    assert len(initialize_statement_indexes) == 1
    assert board_statement_indexes[0] < initialize_statement_indexes[0]


def test_label_count_persistence_has_no_widget_text_fallback():
    source_path = ROOT / "ui" / "sequence" / "sequence_recording_service.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RecordingCountBoardPersistence"
    )
    commit = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "commit"
    )
    attributes = {
        node.attr for node in ast.walk(commit) if isinstance(node, ast.Attribute)
    }

    assert "text" not in attributes
    assert "_count_board_provider" not in attributes
    assert "current_statistics" in attributes
