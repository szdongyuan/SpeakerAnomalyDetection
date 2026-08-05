import os
import sqlite3
from pathlib import Path

import numpy as np

from base import recording_management
from base.db_manager import DataSave
from base.recording_management import RecordingManager
from consts import error_code, model_consts
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _DatabaseContext:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        return self.database

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _InsertDatabase:
    def __init__(self, insert_result):
        self.insert_result = insert_result
        self.inserted_rows = []

    def query_matching_data(self, *_args, **_kwargs):
        return []

    def insert_data_into_db(self, table_name, columns, data):
        self.inserted_rows.append((table_name, columns, data))
        return self.insert_result


class _UpdateDatabase:
    def __init__(self, update_result):
        self.update_result = update_result

    def update_table_data(self, *_args, **_kwargs):
        return self.update_result


class _RelabelWidget(SequenceWidgetStreamingOpsMixin):
    pass


def _audio_info(file_path):
    return {
        "file_path": str(file_path),
        "product_model": "S004-1",
        "sample_rate": 44100,
        "record_date": "2026-08-05",
        "labels": "not_labeled",
        "barcode": "",
    }


def test_normalize_audio_path_for_db_uses_relative_path_inside_application_root(
    tmp_path,
    monkeypatch,
):
    application_root = tmp_path / "application"
    file_path = application_root / "audio_data" / "stored_data" / "S004-1" / "not_labeled" / "sample.wav"
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))

    normalized = RecordingManager.normalize_audio_path_for_db(str(file_path))

    assert normalized == "audio_data/stored_data/S004-1/not_labeled/sample.wav"


def test_normalize_audio_path_for_db_keeps_external_path_absolute(tmp_path, monkeypatch):
    application_root = tmp_path / "application"
    external_path = tmp_path / "factory_audio" / "S004-1" / "not_labeled" / "sample.wav"
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))

    normalized = RecordingManager.normalize_audio_path_for_db(str(external_path))

    assert normalized == os.path.abspath(external_path).replace("\\", "/")


def test_get_audio_info_to_db_does_not_mutate_source_metadata(tmp_path, monkeypatch):
    application_root = tmp_path / "application"
    file_path = application_root / "audio_data" / "stored_data" / "S004-1" / "not_labeled" / "sample.wav"
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))
    database = _InsertDatabase((error_code.OK, "ok"))
    audio_info = _audio_info(file_path)
    original_file_path = audio_info["file_path"]

    audio_data = RecordingManager.get_audio_info_to_db(audio_info, None, database)

    assert audio_info["file_path"] == original_file_path
    assert audio_data[1] == "audio_data/stored_data/S004-1/not_labeled/sample.wav"


def test_save_signal_info_to_db_propagates_insert_failure(tmp_path, monkeypatch):
    database = _InsertDatabase((error_code.INVALID_INSERT, "insert failed"))
    monkeypatch.setattr(
        recording_management,
        "DataSave",
        lambda _db_path: _DatabaseContext(database),
    )

    result = RecordingManager().save_signal_info_to_db(
        _audio_info(tmp_path / "external" / "sample.wav"),
        None,
    )

    assert result == (error_code.INVALID_INSERT, "insert failed")


def test_save_recording_to_wav_propagates_database_failure(tmp_path, monkeypatch):
    manager = RecordingManager()
    monkeypatch.setattr(recording_management, "save_audio_simple", lambda *_args: None)
    monkeypatch.setattr(
        manager,
        "save_signal_info_to_db",
        lambda *_args: (error_code.INVALID_INSERT, "insert failed"),
    )
    audio_info = _audio_info(tmp_path / "sample.wav")
    audio_info["recorded_signal"] = np.zeros(16, dtype=np.float32)

    result = manager.save_recording_to_wav(audio_info, None)

    assert result == (error_code.INVALID_INSERT, "insert failed")


def test_update_audio_label_propagates_no_match_failure(monkeypatch):
    database = _UpdateDatabase((error_code.INVALID_UPDATE, "No data has been updated"))
    monkeypatch.setattr(
        recording_management,
        "DataSave",
        lambda _db_path: _DatabaseContext(database),
    )

    result = RecordingManager().update_audio_label(
        {"file_path": "new.wav", "labels": "OK"},
        "old.wav",
    )

    assert result == (error_code.INVALID_UPDATE, "No data has been updated")


def test_external_audio_path_round_trips_through_sqlite(tmp_path):
    database_path = tmp_path / "audio_data.db"
    database = DataSave(str(database_path))
    create_code, _create_message = database.create_table()
    database.close()
    assert create_code == error_code.OK

    manager = RecordingManager()
    manager.db_path = str(database_path)
    source_path = tmp_path / "factory_audio" / "S004-1" / "not_labeled" / "sample.wav"
    save_code, _save_message = manager.save_signal_info_to_db(
        _audio_info(source_path),
        None,
    )

    expected_source_path = os.path.abspath(source_path).replace("\\", "/")
    assert save_code == error_code.OK
    with sqlite3.connect(database_path) as connection:
        stored_path, stored_label = connection.execute(
            "SELECT file_path, labels FROM audio_data_table"
        ).fetchone()
    assert (stored_path, stored_label) == (expected_source_path, "not_labeled")

    target_path = tmp_path / "factory_audio" / "S004-1" / "OK" / "sample.wav"
    updated_info = _audio_info(target_path)
    updated_info["labels"] = "OK"
    update_code, _update_message = manager.update_audio_label(
        updated_info,
        expected_source_path,
    )

    assert update_code == error_code.OK
    with sqlite3.connect(database_path) as connection:
        stored_path, stored_label = connection.execute(
            "SELECT file_path, labels FROM audio_data_table"
        ).fetchone()
    assert stored_path == os.path.abspath(target_path).replace("\\", "/")
    assert stored_label == "OK"


def test_relabel_rolls_file_back_when_database_update_fails(tmp_path, monkeypatch):
    root = tmp_path / "factory_audio"
    source_dir = root / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample_6000_107c610bb999.wav"
    source_path.write_bytes(b"RIFF")
    monkeypatch.setattr(
        RecordingManager,
        "update_audio_label",
        lambda *_args, **_kwargs: (
            error_code.INVALID_UPDATE,
            "No data has been updated",
        ),
    )
    recorded_info = _audio_info(source_path)
    recorded_info[model_consts.RECORDING_ROOT_CONFIG_KEY] = str(root)

    code, _message, final_path, updated_info = _RelabelWidget()._relabel_stored_audio_record(
        str(source_path),
        recorded_info,
        "OK",
    )

    assert code == error_code.INVALID_UPDATE
    assert Path(final_path) == source_path
    assert source_path.is_file()
    assert not (root / "S004-1" / "OK" / source_path.name).exists()
    assert updated_info["labels"] == "not_labeled"
    assert Path(updated_info["file_path"]) == source_path


def test_relabel_tries_original_mixed_separator_database_path(tmp_path, monkeypatch):
    root = tmp_path / "factory_audio"
    source_dir = root / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample_6000_107c610bb999.wav"
    source_path.write_bytes(b"RIFF")
    mixed_db_path = r"stored_data\S004-1/not_labeled/sample_6000_107c610bb999.wav"
    attempted_paths = []

    def update_audio_label(_manager, _updated_info, old_file_path):
        attempted_paths.append(old_file_path)
        if old_file_path == mixed_db_path:
            return error_code.OK, "updated"
        return error_code.INVALID_UPDATE, "not found"

    monkeypatch.setattr(RecordingManager, "update_audio_label", update_audio_label)
    recorded_info = _audio_info(source_path)
    recorded_info["file_path"] = mixed_db_path
    recorded_info[model_consts.RECORDING_ROOT_CONFIG_KEY] = str(root)

    code, _message, final_path, updated_info = _RelabelWidget()._relabel_stored_audio_record(
        str(source_path),
        recorded_info,
        "OK",
    )

    assert code == error_code.OK
    assert attempted_paths[0] == mixed_db_path
    assert Path(final_path) == root / "S004-1" / "OK" / source_path.name
    assert updated_info["labels"] == "OK"
