"""Register published workbench station WAVs in the existing audio archive."""

import sqlite3
from contextlib import closing
from datetime import date
from uuid import uuid1

from base.recording_management import RecordingManager
from base.workbench_recording_protocol import WorkbenchRecordingCompletion


class WorkbenchRecordingArchiveError(RuntimeError):
    """The published WAV group could not be registered in the audio table."""


class WorkbenchRecordingArchive:
    def __init__(self, database_path, *, timeout=5.0):
        self._database_path = database_path
        self._timeout = timeout

    def save(
        self, completion: WorkbenchRecordingCompletion, product_name: str, recorded_on: date,
    ) -> None:
        metadata = (product_name, completion.sample_rate, recorded_on.isoformat())
        try:
            with closing(sqlite3.connect(self._database_path, timeout=self._timeout)) as database, database:
                database.execute("PRAGMA foreign_keys = ON")
                for station in completion.stations:
                    file_path = RecordingManager.normalize_audio_path_for_db(station.path)
                    insert = database.execute(
                        "INSERT INTO audio_data_table "
                        "(audio_data_id, file_path, product_model, sample_rate, record_date, "
                        "labels, barcode, stimulus_id) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL) "
                        "ON CONFLICT(file_path) DO NOTHING",
                        (str(uuid1()), file_path, *metadata, "not_labeled"),
                    )
                    if insert.rowcount == 0:
                        existing = database.execute(
                            "SELECT product_model, sample_rate, record_date "
                            "FROM audio_data_table WHERE file_path = ?",
                            (file_path,),
                        ).fetchone()
                        if existing != metadata:
                            raise WorkbenchRecordingArchiveError(f"已归档路径的录制信息冲突：{file_path}")
        except sqlite3.Error as error:
            raise WorkbenchRecordingArchiveError(str(error)) from error
