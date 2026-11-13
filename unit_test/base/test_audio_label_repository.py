"""Real SQLite label-only transactions over existing archived WAV records."""

import sqlite3
from contextlib import closing
from datetime import date

import pytest

from base.audio_label_repository import (
    AudioLabelPersistenceError, AudioLabelRepository, AudioLabelUpdate,
)
from base.db_manager import DataSave
from base.recording_management import RecordingManager
from base.workbench_recording_archive import WorkbenchRecordingArchive
from base.workbench_recording_protocol import StationWav, StationWavChannel, WorkbenchRecordingCompletion
from consts import error_code, running_consts


def _read_rows(path):
    with closing(sqlite3.connect(path)) as database:
        return database.execute(
            "SELECT audio_data_id, file_path, product_model, sample_rate, record_date, "
            "labels, barcode, stimulus_id FROM audio_data_table ORDER BY file_path"
        ).fetchall()


@pytest.fixture
def archived(tmp_path, monkeypatch):
    application = tmp_path / "application"
    monkeypatch.setattr(running_consts, "DEFAULT_DIR", str(application))
    database_path = tmp_path / "audio.db"
    database = DataSave(database_path)
    try:
        code, message = database.create_table()
        assert code == error_code.OK, message
    finally:
        database.close()
    channel = StationWavChannel("ch", "声音", "audio", "default", 0, 0)
    paths = []
    for direction in ("forward", "reverse"):
        stations = tuple(
            StationWav(station, str(root / "recordings" / direction / f"{station}.wav"), (channel,))
            for station, root in (("station-a", application), ("station-b", tmp_path / "external"))
        )
        completion = WorkbenchRecordingCompletion(
            f"request-{direction}", "project-1", direction, 48000, 64, stations,
        )
        WorkbenchRecordingArchive(database_path).save(completion, "电机型号 A", date(2026, 11, 13))
        paths.extend(station.path for station in stations)
    with closing(sqlite3.connect(database_path)) as database, database:
        database.execute("UPDATE audio_data_table SET barcode = 'SN-001'")
    return database_path, paths


@pytest.mark.parametrize("label", ["OK", "NG", "not_labeled"])
@pytest.mark.parametrize("target_index", [0, 1, 2, 3])
def test_update_changes_only_target_label_and_preserves_schema(archived, label, target_index):
    database_path, paths = archived
    target = paths[target_index]
    file_path = RecordingManager.normalize_audio_path_for_db(target)
    before = _read_rows(database_path)
    with closing(sqlite3.connect(database_path)) as database:
        schema = database.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
    # Start with a different label so all three storage labels exercise a write.
    with closing(sqlite3.connect(database_path)) as database, database:
        database.execute(
            "UPDATE audio_data_table SET labels = ? WHERE file_path = ?",
            ("OK" if label == "NG" else "NG", file_path),
        )

    update = AudioLabelRepository(database_path).update(target, label)

    expected = [row[:5] + (label,) + row[6:] if row[1] == file_path else row for row in before]
    assert _read_rows(database_path) == expected
    target_row = next(row for row in expected if row[1] == file_path)
    assert update == AudioLabelUpdate(target_row[0], file_path, label)
    if target_index % 2 == 0:
        assert file_path.startswith("recordings/")
    else:
        assert file_path == target.replace("\\", "/")
    with closing(sqlite3.connect(database_path)) as database:
        assert database.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == schema


def test_repeated_label_write_is_successful_and_update_is_immutable(archived):
    database_path, paths = archived
    repository = AudioLabelRepository(database_path)
    first = repository.update(paths[0], "OK")
    before = _read_rows(database_path)

    assert repository.update(paths[0], "OK") == first
    assert _read_rows(database_path) == before
    with pytest.raises(AttributeError):
        first.label = "NG"


def test_missing_row_reports_path_without_inserting(archived, tmp_path):
    database_path, _ = archived
    missing = tmp_path / "missing.wav"
    before = _read_rows(database_path)

    with pytest.raises(AudioLabelPersistenceError) as failure:
        AudioLabelRepository(database_path).update(missing, "OK")

    assert RecordingManager.normalize_audio_path_for_db(missing) in str(failure.value)
    assert failure.value.__cause__ is None
    assert _read_rows(database_path) == before


def test_lock_timeout_reports_cause_and_leaves_original_labels(archived):
    database_path, paths = archived
    before = _read_rows(database_path)
    repository = AudioLabelRepository(database_path, timeout=0.01)
    with closing(sqlite3.connect(database_path)) as writer, writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(AudioLabelPersistenceError, match="database is locked") as failure:
            repository.update(paths[0], "OK")
        assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
        assert RecordingManager.normalize_audio_path_for_db(paths[0]) in str(failure.value)
        assert _read_rows(database_path) == before

    assert repository.update(paths[0], "OK").label == "OK"


def test_trigger_failure_rolls_back_update_and_preserves_error_cause(archived):
    database_path, paths = archived
    before = _read_rows(database_path)
    with closing(sqlite3.connect(database_path)) as database, database:
        database.execute(
            "CREATE TRIGGER reject_label AFTER UPDATE OF labels ON audio_data_table "
            "BEGIN SELECT RAISE(FAIL, 'label rejected'); END"
        )

    with pytest.raises(AudioLabelPersistenceError, match="label rejected") as failure:
        AudioLabelRepository(database_path).update(paths[0], "OK")

    assert isinstance(failure.value.__cause__, sqlite3.IntegrityError)
    assert RecordingManager.normalize_audio_path_for_db(paths[0]) in str(failure.value)
    assert _read_rows(database_path) == before


def test_connection_failure_is_reported_with_wav_path(tmp_path):
    wav_path = tmp_path / "recording.wav"

    with pytest.raises(AudioLabelPersistenceError, match="unable to open database file") as failure:
        AudioLabelRepository(tmp_path / "missing" / "audio.db").update(wav_path, "NG")

    assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
    assert RecordingManager.normalize_audio_path_for_db(wav_path) in str(failure.value)
