"""Actual SQLite return contracts for already-written recording persistence."""
import copy
import sqlite3

import pytest

from base.db_manager import DataSave
from base.recording_management import RecordingManager
from consts import error_code


@pytest.fixture
def persistence(tmp_path):
    path = tmp_path / 'recordings.db'
    db = DataSave(str(path))
    assert db.create_audio_tables()[0] == error_code.OK
    db.close()
    manager = RecordingManager.__new__(RecordingManager)
    manager.db_path = str(path)
    info = dict(file_path=str(tmp_path / 'saved.wav'), product_model='input', sample_rate=51200,
                record_date='2026-11-02', labels='not_labeled', barcode=None)
    return manager, path, info


def test_actual_audio_insert_rejection_is_not_success(persistence):
    manager, path, info = persistence
    with sqlite3.connect(path) as db:
        db.execute("CREATE TRIGGER reject_audio BEFORE INSERT ON audio_data_table "
                   "BEGIN SELECT RAISE(FAIL, 'audio insert rejected'); END")
    code, message = manager.save_signal_info_to_db(info, None)
    assert code != error_code.OK and 'audio insert rejected' in message
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM audio_data_table').fetchone()[0] == 0


def test_actual_stimulus_insert_rejection_does_not_attempt_audio(persistence):
    manager, path, info = persistence
    with sqlite3.connect(path) as db:
        db.execute("CREATE TRIGGER reject_stimulus BEFORE INSERT ON stimulus_signal_table "
                   "BEGIN SELECT RAISE(FAIL, 'stimulus insert rejected'); END")
    stimulus = dict(stimulus_method='chirp', stimulus_type='linear', repeat_times=1,
                    start_freq=80, stop_freq=2000, sample_rate=48000, total_time=3, num_steps=None)
    code, message = manager.save_signal_info_to_db(info, stimulus)
    assert code != error_code.OK and 'stimulus insert rejected' in message
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM stimulus_signal_table').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM audio_data_table').fetchone()[0] == 0


@pytest.mark.parametrize('barcode', [None, 'barcode'])
def test_identical_existing_recording_is_idempotent_without_retrying_insert(persistence, monkeypatch, barcode):
    manager, path, info = persistence
    info['barcode'] = barcode
    assert manager.save_signal_info_to_db(copy.deepcopy(info), None)[0] == error_code.OK
    statements = []
    original = DataSave.connect
    def connect(database):
        result = original(database)
        if database.connection is not None: database.connection.set_trace_callback(statements.append)
        return result
    monkeypatch.setattr(DataSave, 'connect', connect)
    code, message = manager.save_signal_info_to_db(copy.deepcopy(info), None)
    assert code == error_code.OK, message
    assert not any(sql.lstrip().upper().startswith('INSERT') for sql in statements)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM audio_data_table').fetchone()[0] == 1


def test_existing_path_with_conflicting_metadata_reports_failure(persistence):
    manager, path, info = persistence
    assert manager.save_signal_info_to_db(copy.deepcopy(info), None)[0] == error_code.OK
    changed = dict(info, product_model='different')
    code, message = manager.save_signal_info_to_db(changed, None)
    assert code != error_code.OK
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT product_model FROM audio_data_table').fetchall() == [('input',)]


@pytest.mark.parametrize('changes', [{'sample_rate': 64000}, {'labels': 'OK'}])
def test_explicit_replay_updates_metadata_and_preserves_id(persistence, changes):
    manager, path, info = persistence
    assert manager.save_signal_info_to_db(copy.deepcopy(info), None)[0] == error_code.OK
    with sqlite3.connect(path) as db:
        original_id = db.execute('SELECT audio_data_id FROM audio_data_table').fetchone()[0]
    changed = dict(info, **changes)
    code, message = manager.save_signal_info_to_db(changed, None, replace_existing=True)
    assert code == error_code.OK, message
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT audio_data_id, sample_rate, labels FROM audio_data_table').fetchall() == [
            (original_id, changed['sample_rate'], changed['labels'])]


def test_actual_replay_update_rejection_preserves_prior_row(persistence):
    manager, path, info = persistence
    assert manager.save_signal_info_to_db(copy.deepcopy(info), None)[0] == error_code.OK
    with sqlite3.connect(path) as db:
        prior = db.execute('SELECT * FROM audio_data_table').fetchall()
        db.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON audio_data_table "
                   "BEGIN SELECT RAISE(FAIL, 'recording update rejected'); END")
    code, message = manager.save_signal_info_to_db(dict(info, sample_rate=64000), None, replace_existing=True)
    assert code != error_code.OK and 'recording update rejected' in message
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM audio_data_table').fetchall() == prior
