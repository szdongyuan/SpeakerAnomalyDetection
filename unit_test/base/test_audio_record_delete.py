from pathlib import Path
import sqlite3

import pytest

from base.audio_record_delete import (
    count_audio_deletion_files, delete_audio_recordings, plan_audio_record_deletion,
)
from base.recording_management import RecordingManager
from consts import error_code
from unit_test.base.test_audio_record_package import make_recording


@pytest.fixture
def recordings(tmp_path):
    first = make_recording(tmp_path / "results")
    second = make_recording(tmp_path / "results", project="ProjectB")
    rows = [("one", str(first["wav"])), ("two", str(second["wav"]))]
    database = tmp_path / "audio.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.executemany("INSERT INTO audio_data_table VALUES (?, ?)", rows)
    manager = RecordingManager()
    manager.db_path = str(database)
    return rows, (first, second), manager


def database_ids(manager):
    with sqlite3.connect(manager.db_path) as connection:
        return [row[0] for row in connection.execute("SELECT audio_data_id FROM audio_data_table ORDER BY audio_data_id")]


def test_confirmation_plan_is_read_only_and_delete_is_exact(recordings):
    rows, (first, second), manager = recordings
    unrelated = first["images"].with_name("notes.txt")
    unrelated.write_text("retain")
    plans = plan_audio_record_deletion(rows[:1])
    assert count_audio_deletion_files(plans) == dict.fromkeys(first, 1)
    assert all(path.exists() for path in first.values())
    assert database_ids(manager) == ["one", "two"]
    deleted, errors = delete_audio_recordings(plans, manager)
    assert deleted == ["one"]
    assert errors == []
    assert all(not path.exists() for path in first.values())
    assert all(path.exists() for path in second.values())
    assert unrelated.exists()
    assert first["images"].parent.is_dir()
    assert not first["analysis_csv"].parent.exists()
    assert database_ids(manager) == ["two"]


def test_missing_wav_still_deletes_companions_and_database(recordings):
    rows, (first, _), manager = recordings
    first["wav"].unlink()
    plans = plan_audio_record_deletion(rows[:1])
    assert count_audio_deletion_files(plans)["wav"] == 0
    assert delete_audio_recordings(plans, manager) == (["one"], [])
    assert all(not path.exists() for path in first.values())


def test_file_failure_keeps_wav_and_record_other_record_completes_then_retry(recordings, monkeypatch):
    rows, (first, second), manager = recordings
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == first["analysis_csv"]:
            raise PermissionError("file in use")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        deleted, errors = delete_audio_recordings(plan_audio_record_deletion(rows), manager)
    assert deleted == ["two"]
    assert len(errors) == 1 and "file in use" in errors[0]
    assert first["wav"].exists()
    assert first["analysis_csv"].exists()
    assert all(not path.exists() for path in second.values())
    assert database_ids(manager) == ["one"]
    assert delete_audio_recordings(plan_audio_record_deletion(rows[:1]), manager) == (["one"], [])
    assert database_ids(manager) == []


def test_real_database_failure_is_reported_and_row_survives(recordings):
    rows, (first, _), manager = recordings
    with sqlite3.connect(manager.db_path) as connection:
        connection.execute("CREATE TRIGGER block_delete BEFORE DELETE ON audio_data_table BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    deleted, errors = delete_audio_recordings(plan_audio_record_deletion(rows[:1]), manager)
    assert deleted == []
    assert len(errors) == 1 and "数据库记录删除失败" in errors[0]
    assert database_ids(manager) == ["one", "two"]
    assert all(not path.exists() for path in first.values())


def test_confirmed_snapshot_does_not_expand_to_new_files(recordings):
    rows, (first, _), manager = recordings
    plans = plan_audio_record_deletion(rows[:1])
    later = first["images"].with_name("created_after_confirmation.png")
    later.write_bytes(b"new result")
    assert delete_audio_recordings(plans, manager) == (["one"], [])
    assert later.exists()
    assert later.parent.is_dir()


def test_companion_directory_redirect_is_rejected_before_delete(recordings, monkeypatch, tmp_path):
    rows, (first, _), manager = recordings
    plans = plan_audio_record_deletion(rows[:1])
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == first["raw_csv"].parent:
            return tmp_path / "outside"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    deleted, errors = delete_audio_recordings(plans, manager)
    assert deleted == []
    assert "超出录音样本目录" in errors[0]
    assert all(path.exists() for path in first.values())
    assert database_ids(manager) == ["one", "two"]


@pytest.mark.parametrize("already_empty", [False, True])
def test_removes_only_recording_result_directories(recordings, already_empty):
    rows, (first, second), manager = recordings
    result_directories = [first[kind].parent for kind in ("images", "analysis_csv")]
    shared_directories = [first[kind].parent for kind in ("wav", "raw_csv")]
    shared_directories.extend(directory.parent for directory in result_directories)
    sibling = result_directories[0].with_name("another_recording")
    sibling.mkdir()
    if already_empty:
        for path in first.values():
            path.unlink()

    plans = plan_audio_record_deletion(rows[:1])
    assert all(directory.is_dir() for directory in result_directories)
    assert delete_audio_recordings(plans, manager) == (["one"], [])
    assert all(not directory.exists() for directory in result_directories)
    assert all(directory.is_dir() for directory in shared_directories)
    assert sibling.is_dir()
    assert all(path.exists() for path in second.values())


def test_directory_cleanup_failure_keeps_record_and_can_retry(recordings, monkeypatch):
    rows, (first, _), manager = recordings
    blocked = first["analysis_csv"].parent
    original = Path.rmdir

    def rmdir(path):
        if path == blocked:
            raise PermissionError("directory in use")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", rmdir)
        deleted, errors = delete_audio_recordings(plan_audio_record_deletion(rows[:1]), manager)
    assert deleted == []
    assert len(errors) == 1 and "directory in use" in errors[0]
    assert blocked.is_dir()
    assert database_ids(manager) == ["one", "two"]
    assert delete_audio_recordings(plan_audio_record_deletion(rows[:1]), manager) == (["one"], [])
    assert not blocked.exists()


def test_directory_becoming_nonempty_is_preserved(recordings, monkeypatch):
    rows, (first, _), manager = recordings
    directory = first["images"].parent
    added = directory / "new_result.png"
    original = Path.rmdir

    def rmdir(path):
        if path == directory:
            added.write_bytes(b"created after confirmation")
        return original(path)

    monkeypatch.setattr(Path, "rmdir", rmdir)
    assert delete_audio_recordings(plan_audio_record_deletion(rows[:1]), manager) == (["one"], [])
    assert added.read_bytes() == b"created after confirmation"


def test_empty_result_directory_redirect_is_rejected(recordings, monkeypatch, tmp_path):
    rows, (first, _), manager = recordings
    for path in first.values():
        path.unlink()
    plans = plan_audio_record_deletion(rows[:1])
    redirected = first["images"].parent
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == redirected:
            return tmp_path / "outside"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    deleted, errors = delete_audio_recordings(plans, manager)
    assert deleted == []
    assert len(errors) == 1 and "超出录音样本目录" in errors[0]
    assert redirected.is_dir()
    assert database_ids(manager) == ["one", "two"]
