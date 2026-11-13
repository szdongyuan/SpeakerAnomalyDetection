"""Persist labels of existing archived WAVs without changing their metadata."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass

from base.recording_management import RecordingManager


@dataclass(frozen=True)
class AudioLabelUpdate:
    audio_data_id: str
    file_path: str
    label: str


class AudioLabelPersistenceError(RuntimeError):
    """An existing WAV record's label could not be committed."""


class AudioLabelRepository:
    def __init__(self, database_path, *, timeout=5.0):
        self._database_path = database_path
        self._timeout = timeout

    def update(self, wav_path, label) -> AudioLabelUpdate:
        file_path = RecordingManager.normalize_audio_path_for_db(wav_path)
        try:
            with closing(sqlite3.connect(self._database_path, timeout=self._timeout)) as database, database:
                updated = database.execute(
                    "UPDATE audio_data_table SET labels = ? WHERE file_path = ?",
                    (label, file_path),
                )
                if updated.rowcount == 0:
                    raise AudioLabelPersistenceError(f"Audio record not found: {file_path}")
                row = database.execute(
                    "SELECT audio_data_id, file_path, labels FROM audio_data_table WHERE file_path = ?",
                    (file_path,),
                ).fetchone()
        except sqlite3.Error as error:
            raise AudioLabelPersistenceError(f"Audio label update failed for {file_path}: {error}") from error
        return AudioLabelUpdate(*row)
