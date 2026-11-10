import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date
from threading import Barrier
from uuid import UUID

import pytest

from base.db_manager import DataSave
from base.recording_management import RecordingManager
from base.workbench_recording_archive import WorkbenchRecordingArchive, WorkbenchRecordingArchiveError
from base.workbench_recording_protocol import StationWav, StationWavChannel, WorkbenchRecordingCompletion
from consts import error_code, running_consts


def _create_database(path):
    database = DataSave(path)
    try:
        code, message = database.create_table()
        assert code == error_code.OK, message
    finally:
        database.close()
    return path


def _read_rows(path):
    with closing(sqlite3.connect(path)) as database:
        return database.execute(
            "SELECT audio_data_id, file_path, product_model, sample_rate, record_date, "
            "labels, barcode, stimulus_id FROM audio_data_table ORDER BY file_path"
        ).fetchall()


@pytest.fixture
def database_path(tmp_path):
    return _create_database(tmp_path / "audio_data.db")


@pytest.fixture
def recording(tmp_path, monkeypatch):
    application = tmp_path / "application"
    monkeypatch.setattr(running_consts, "DEFAULT_DIR", str(application))
    channel = StationWavChannel("ch", "声音", "audio", "default", 0, 0)
    completion = WorkbenchRecordingCompletion(
        "request-1", "project-1", "condition-1", 48000, 64,
        (
            StationWav("station-a", str(application / "recordings" / "station-a.wav"), (channel,)),
            StationWav("station-b", str(tmp_path / "external" / "station-b.wav"), (channel,)),
        ),
    )
    return completion, "电机型号 A", date(2026, 11, 10)


def test_archive_writes_each_station_with_existing_schema(database_path, recording, tmp_path):
    WorkbenchRecordingArchive(database_path).save(*recording)

    rows = _read_rows(database_path)
    external_path = (tmp_path / "external" / "station-b.wav").as_posix()
    assert {row[1] for row in rows} == {"recordings/station-a.wav", external_path}
    assert len({row[0] for row in rows}) == len(rows) == 2
    for audio_id, _, product, sample_rate, recorded_on, label, barcode, stimulus_id in rows:
        assert UUID(audio_id).version == 1
        assert (product, sample_rate, recorded_on) == ("电机型号 A", 48000, "2026-11-10")
        assert (label, barcode, stimulus_id) == ("not_labeled", None, None)

    manager = RecordingManager()
    manager.db_path = database_path
    code, archive_rows = manager.get_record_audio_data()
    assert code == error_code.OK
    assert sorted(archive_rows) == sorted(rows)


def test_repeat_and_concurrent_archive_preserve_existing_id_and_label(database_path, recording, tmp_path):
    archive = WorkbenchRecordingArchive(database_path)
    archive.save(*recording)
    with closing(sqlite3.connect(database_path)) as database, database:
        database.execute(
            "UPDATE audio_data_table SET labels = 'OK', barcode = 'SN-001' WHERE file_path = ?",
            ("recordings/station-a.wav",),
        )
    labeled_rows = _read_rows(database_path)
    archive.save(*recording)
    assert _read_rows(database_path) == labeled_rows

    concurrent_path = _create_database(tmp_path / "concurrent.db")
    concurrent_archive = WorkbenchRecordingArchive(concurrent_path)
    start = Barrier(2, timeout=5)

    def save_together():
        start.wait()
        concurrent_archive.save(*recording)

    with ThreadPoolExecutor(max_workers=2) as executor:
        submissions = [executor.submit(save_together) for _ in range(2)]
        for submission in submissions:
            submission.result(timeout=10)

    concurrent_rows = _read_rows(concurrent_path)
    assert [row[1] for row in concurrent_rows] == [row[1] for row in labeled_rows]


@pytest.mark.parametrize("failure", ["metadata-conflict", "second-insert"])
def test_conflict_or_second_insert_failure_rolls_back_new_rows(database_path, recording, failure):
    completion, product, recorded_on = recording
    second_path = completion.stations[1].path.replace("\\", "/")
    existing_path = second_path if failure == "metadata-conflict" else "existing.wav"
    with closing(sqlite3.connect(database_path)) as database, database:
        database.execute(
            "INSERT INTO audio_data_table VALUES (?, ?, ?, ?, ?, 'OK', 'SN-002', NULL)",
            ("existing-id", existing_path, product, 44100, recorded_on.isoformat()),
        )
        if failure == "second-insert":
            database.execute(
                "CREATE TRIGGER reject_second_station BEFORE INSERT ON audio_data_table "
                "WHEN NEW.file_path <> 'recordings/station-a.wav' "
                "BEGIN SELECT RAISE(ABORT, 'second station rejected'); END"
            )
    existing_rows = _read_rows(database_path)

    with pytest.raises(WorkbenchRecordingArchiveError) as failure_result:
        WorkbenchRecordingArchive(database_path).save(*recording)

    if failure == "metadata-conflict":
        assert second_path in str(failure_result.value)
    else:
        assert str(failure_result.value) == "second station rejected"
        assert isinstance(failure_result.value.__cause__, sqlite3.IntegrityError)
    assert _read_rows(database_path) == existing_rows


def test_commit_failure_rolls_back_new_rows(database_path, recording):
    with closing(sqlite3.connect(database_path)) as database, database:
        database.executescript(
            """
            CREATE TABLE deferred_archive_check (
                stimulus_id TEXT REFERENCES stimulus_signal_table(stimulus_id) DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TRIGGER reject_archive_at_commit AFTER INSERT ON audio_data_table
            BEGIN INSERT INTO deferred_archive_check VALUES ('missing-stimulus'); END;
            """
        )

    with pytest.raises(WorkbenchRecordingArchiveError, match="FOREIGN KEY constraint failed") as failure_result:
        WorkbenchRecordingArchive(database_path).save(*recording)

    assert isinstance(failure_result.value.__cause__, sqlite3.IntegrityError)
    assert _read_rows(database_path) == []


def test_writer_lock_is_reported_without_partial_rows(database_path, recording):
    archive = WorkbenchRecordingArchive(database_path, timeout=0.01)
    with closing(sqlite3.connect(database_path)) as writer, writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(WorkbenchRecordingArchiveError, match="database is locked") as failure_result:
            archive.save(*recording)
        assert isinstance(failure_result.value.__cause__, sqlite3.OperationalError)
        assert _read_rows(database_path) == []

    archive.save(*recording)
    assert len(_read_rows(database_path)) == 2


def test_unavailable_database_directory_is_reported(tmp_path, recording):
    archive = WorkbenchRecordingArchive(tmp_path / "missing" / "audio_data.db")

    with pytest.raises(WorkbenchRecordingArchiveError, match="unable to open database file") as failure_result:
        archive.save(*recording)

    assert isinstance(failure_result.value.__cause__, sqlite3.OperationalError)
