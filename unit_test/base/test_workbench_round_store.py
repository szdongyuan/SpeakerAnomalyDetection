import json
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pytest

import base.workbench_round_store as round_store_module
from base.workbench_round_store import (
    RoundSnapshot,
    WorkbenchRoundStateError,
    WorkbenchRoundStore,
)
from consts.running_consts import DEFAULT_DIR


def test_round_snapshot_round_trips(tmp_path):
    store = WorkbenchRoundStore(tmp_path / "round.json")
    assert store.load() is None
    assert not store.path.exists()

    snapshot = RoundSnapshot(date(2026, 11, 12), 10000)
    store.save(snapshot)

    assert store.load() == snapshot
    assert json.loads(store.path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "date": "2026-11-12",
        "round_number": 10000,
    }


def test_round_snapshot_is_immutable():
    snapshot = RoundSnapshot(date(2026, 11, 12), 1)
    with pytest.raises(FrozenInstanceError):
        snapshot.number = 2


def test_default_path_is_anchored_to_application_root(tmp_path, monkeypatch):
    expected = Path(DEFAULT_DIR) / "ui/ui_config/workbench_round_state.json"
    before = WorkbenchRoundStore().path
    monkeypatch.chdir(tmp_path)

    assert WorkbenchRoundStore.default_path() == expected
    assert WorkbenchRoundStore().path == before == expected
    assert before.is_absolute()


def test_store_creates_parent_directory_and_uses_utf8(tmp_path):
    store = WorkbenchRoundStore(tmp_path / "轮次" / "状态.json")
    snapshot = RoundSnapshot(date(2026, 11, 12), 1)

    store.save(snapshot)

    assert store.load() == snapshot
    assert json.loads(store.path.read_bytes().decode("utf-8"))["round_number"] == 1


@pytest.mark.parametrize(
    "content",
    [
        "{invalid JSON",
        "null",
        "[]",
        '"text"',
        "{}",
        '{"schema_version": 2, "date": "2026-11-12", "round_number": 1}',
        '{"schema_version": true, "date": "2026-11-12", "round_number": 1}',
        '{"schema_version": 1.0, "date": "2026-11-12", "round_number": 1}',
        '{"date": "2026-11-12", "round_number": 1}',
        '{"schema_version": 1, "round_number": 1}',
        '{"schema_version": 1, "date": "2026-11-12"}',
        '{"schema_version": 1, "date": "2026-02-30", "round_number": 1}',
        '{"schema_version": 1, "date": "20261112", "round_number": 1}',
        '{"schema_version": 1, "date": "2026-W46-4", "round_number": 1}',
        '{"schema_version": 1, "date": 20261112, "round_number": 1}',
        '{"schema_version": 1, "date": null, "round_number": 1}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": true}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": false}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": 0}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": -1}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": 1.0}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": "1"}',
        '{"schema_version": 1, "date": "2026-11-12", "round_number": null}',
    ],
)
def test_invalid_existing_state_is_reported_and_preserved(tmp_path, content):
    path = tmp_path / "round.json"
    path.write_text(content, encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(WorkbenchRoundStateError) as error:
        WorkbenchRoundStore(path).load()

    assert str(path) in str(error.value)
    assert "load" in str(error.value).lower()
    assert error.value.__cause__ is not None
    assert path.read_bytes() == original


@pytest.mark.parametrize("number", [True, False, 0, -1, 1.0, "1", None])
def test_save_rejects_invalid_round_number_before_writing(tmp_path, number):
    store = WorkbenchRoundStore(tmp_path / "round.json")

    with pytest.raises(WorkbenchRoundStateError) as error:
        store.save(RoundSnapshot(date(2026, 11, 12), number))

    assert "save" in str(error.value).lower()
    assert str(store.path) in str(error.value)
    assert not store.path.exists()


@pytest.mark.parametrize("day", ["2026-11-12", None, datetime(2026, 11, 12)])
def test_save_rejects_invalid_day_before_writing(tmp_path, day):
    store = WorkbenchRoundStore(tmp_path / "round.json")

    with pytest.raises(WorkbenchRoundStateError):
        store.save(RoundSnapshot(day, 1))

    assert not store.path.exists()


def test_replace_failure_preserves_old_state_and_allows_retry(tmp_path, monkeypatch):
    store = WorkbenchRoundStore(tmp_path / "round.json")
    old = RoundSnapshot(date(2026, 11, 12), 8)
    new = RoundSnapshot(date(2026, 11, 12), 9)
    store.save(old)
    original = store.path.read_bytes()
    real_replace = round_store_module.os.replace
    attempts = []

    def fail_once(source, destination):
        attempts.append((Path(source), Path(destination)))
        assert Path(source).parent == store.path.parent
        assert json.loads(Path(source).read_text(encoding="utf-8"))["round_number"] == 9
        if len(attempts) == 1:
            raise PermissionError("injected replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(round_store_module.os, "replace", fail_once)

    with pytest.raises(WorkbenchRoundStateError) as error:
        store.save(new)

    assert "save" in str(error.value).lower()
    assert str(store.path) in str(error.value)
    assert isinstance(error.value.__cause__, PermissionError)
    assert store.path.read_bytes() == original
    assert store.load() == old
    assert list(tmp_path.iterdir()) == [store.path]

    store.save(new)

    assert len(attempts) == 2
    assert store.load() == new
    assert list(tmp_path.iterdir()) == [store.path]


def test_fsync_failure_preserves_old_file_and_cleans_temporary_file(tmp_path, monkeypatch):
    store = WorkbenchRoundStore(tmp_path / "round.json")
    store.save(RoundSnapshot(date(2026, 11, 12), 8))
    original = store.path.read_bytes()

    def fail_fsync(fd):
        raise OSError("injected disk sync failure")

    monkeypatch.setattr(round_store_module.os, "fsync", fail_fsync)

    with pytest.raises(WorkbenchRoundStateError) as error:
        store.save(RoundSnapshot(date(2026, 11, 12), 9))

    assert str(store.path) in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert store.path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [store.path]


def test_directory_in_place_of_state_file_is_reported_and_preserved(tmp_path):
    path = tmp_path / "round.json"
    path.mkdir()
    marker = path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    store = WorkbenchRoundStore(path)

    with pytest.raises(WorkbenchRoundStateError) as load_error:
        store.load()
    with pytest.raises(WorkbenchRoundStateError) as save_error:
        store.save(RoundSnapshot(date(2026, 11, 12), 1))

    assert "load" in str(load_error.value).lower()
    assert str(path) in str(load_error.value)
    assert "save" in str(save_error.value).lower()
    assert str(path) in str(save_error.value)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_utf8_is_reported_and_preserved(tmp_path):
    path = tmp_path / "round.json"
    original = b"\xff\xfe\x00"
    path.write_bytes(original)

    with pytest.raises(WorkbenchRoundStateError) as error:
        WorkbenchRoundStore(path).load()

    assert "load" in str(error.value).lower()
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, UnicodeError)
    assert path.read_bytes() == original


def test_parent_creation_failure_is_reported_without_modifying_existing_path(tmp_path):
    parent = tmp_path / "parent"
    parent.write_text("keep", encoding="utf-8")
    store = WorkbenchRoundStore(parent / "round.json")

    with pytest.raises(WorkbenchRoundStateError) as error:
        store.save(RoundSnapshot(date(2026, 11, 12), 1))

    assert "save" in str(error.value).lower()
    assert str(store.path) in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert parent.read_text(encoding="utf-8") == "keep"
