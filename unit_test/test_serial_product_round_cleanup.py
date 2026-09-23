import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from base import recording_management
from base.recording_management import RecordingManager
from consts import error_code
from ui.sequence import sequence_widget_serial_trigger_ops as serial_ops_module
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class _RecentPanel:
    def __init__(self):
        self.removed = []

    def remove_session(self, session_id):
        self.removed.append(session_id)


def test_discard_recent_session_group_removes_only_target_round_and_recordings(monkeypatch):
    deleted_paths = []

    class _RecordingManager:
        def delete_audio(self, file_path):
            deleted_paths.append(file_path)
            return error_code.OK, "deleted"

    monkeypatch.setattr(serial_ops_module, "RecordingManager", _RecordingManager)
    host = SimpleNamespace(
        recent_test_sessions=["r1-1", "r1-2", "r2-1"],
        recent_test_session_by_id={
            "r1-1": {"group_id": "round-1", "recorded_path": "one.wav"},
            "r1-2": {
                "group_id": "round-1",
                "recorded_path": "",
                "recorded_signal_info": {"file_path": "two.wav"},
            },
            "r1-map-only": {
                "group_id": "round-1",
                "recorded_path": "map-only.wav",
            },
            "r2-1": {"group_id": "round-2", "recorded_path": "other.wav"},
        },
        _current_recent_session_id="r1-2",
        _pending_recent_session_append=True,
        recent_session_panel=_RecentPanel(),
        default_logger=_Logger(),
    )

    removed_count = SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(
        host,
        "round-1",
    )

    assert removed_count == 3
    assert deleted_paths == ["one.wav", "two.wav", "map-only.wav"]
    assert host.recent_test_sessions == ["r2-1"]
    assert set(host.recent_test_session_by_id) == {"r2-1"}
    assert host.recent_session_panel.removed == ["r1-1", "r1-2", "r1-map-only"]
    assert host._current_recent_session_id is None
    assert host._pending_recent_session_append is False


def test_delete_audio_uses_the_normalized_database_path(tmp_path, monkeypatch):
    application_root = tmp_path / "application"
    audio_path = application_root / "audio_data" / "round.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"wav")
    database = Mock()
    data_save = MagicMock()
    data_save.return_value.__enter__.return_value = database
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))
    monkeypatch.setattr(recording_management, "DataSave", data_save)

    code, _message = RecordingManager().delete_audio(str(audio_path))

    assert code == error_code.OK
    database.delete_with_condition.assert_called_once_with(
        "audio_data_table",
        {"file_path": "audio_data/round.wav"},
    )
    assert not audio_path.exists()


def test_delete_audio_removes_database_record_when_wav_is_already_missing(
    tmp_path,
    monkeypatch,
):
    application_root = tmp_path / "application"
    missing_audio_path = application_root / "audio_data" / "missing.wav"
    database = Mock()
    data_save = MagicMock()
    data_save.return_value.__enter__.return_value = database
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))
    monkeypatch.setattr(recording_management, "DataSave", data_save)

    code, _message = RecordingManager().delete_audio(str(missing_audio_path))

    assert code == error_code.OK
    database.delete_with_condition.assert_called_once_with(
        "audio_data_table",
        {"file_path": "audio_data/missing.wav"},
    )


@pytest.mark.parametrize('release_order', ['recording_first', 'csv_first', 'recording_failed'])
def test_serial_cleanup_waits_for_both_owners_and_preserves_old_scope(monkeypatch, tmp_path, release_order):
    from base.raw_audio_csv_tasks import CsvTaskLedger
    from base.raw_audio_csv_protocol import CsvExportRequest
    ledger = CsvTaskLedger()
    old = tmp_path / 'old.wav'
    other = tmp_path / 'new.wav'
    csv = tmp_path / 'old.csv'
    for path in (old, other, csv):
        path.write_bytes(b'keep')
    leased = {str(old)}
    recording_service = SimpleNamespace(is_path_leased=lambda path: path in leased,
        defer_path_cleanup=lambda path, callback: True)
    token = ledger.reserve('old').reservation
    assert ledger.commit(token, CsvExportRequest('task', 'old', str(old), str(csv), (0,), 'old-round', 'old-record')) == 'accepted'
    ledger.dispatch_next(1)
    deleted = []
    class Manager:
        def delete_audio(self, path):
            assert ledger.try_acquire_mutation((path,)) is None
            assert not leased
            deleted.append(path)
            old.unlink()
            return error_code.OK, 'deleted'
    monkeypatch.setattr(serial_ops_module, 'RecordingManager', Manager)
    host = SimpleNamespace(raw_audio_csv_service=ledger, recording_bridge=SimpleNamespace(service=recording_service),
        recent_test_sessions=['old'], recent_test_session_by_id={'old': {'group_id': 'old-round', 'recorded_path': str(old)}},
        default_logger=_Logger(), recent_session_panel=None)
    assert SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(host, 'old-round') == 1
    host.recent_test_session_by_id['new'] = {'group_id': 'new-round', 'recorded_path': str(other)}
    if release_order == 'recording_first':
        leased.clear()
        assert ledger.run_deferred_mutations() == 0
    ledger.mark_terminal('task', 1, 'succeeded')
    ledger.release_task('task', 1)
    if release_order != 'recording_first':
        assert ledger.run_deferred_mutations() == 0
    assert ledger.try_acquire_mutation((str(old),)) is None
    if release_order == 'recording_failed':
        assert old.exists() and not deleted
        return
    leased.clear()
    assert ledger.run_deferred_mutations() == 1
    assert ledger.run_deferred_mutations() == 0
    assert deleted == [str(old)]
    assert other.exists() and csv.exists()
    assert not ledger.paths_busy((str(old),))


def test_serial_cleanup_failure_is_logged_without_ui_consumer_and_service_recovers(
        monkeypatch, tmp_path, caplog):
    import logging
    from base.raw_audio_csv_service import RawAudioCsvService
    from unit_test.base.test_raw_audio_csv_service import eventually, submit

    old = tmp_path / 'old.wav'
    old.write_bytes(b'preserved recording')
    attempts = []

    class Manager:
        def delete_audio(self, path):
            attempts.append(path)
            return error_code.INVALID_DELETE, 'disk locked ' + 'x' * 2000

    monkeypatch.setattr(serial_ops_module, 'RecordingManager', Manager)
    service = RawAudioCsvService()
    host = SimpleNamespace(
        raw_audio_csv_service=service,
        recent_test_sessions=['old'],
        recent_test_session_by_id={'old': {'group_id': 'old-round', 'recorded_path': str(old)}},
        default_logger=_Logger(), recent_session_panel=None)
    try:
        with caplog.at_level(logging.WARNING, logger='base.raw_audio_csv_service'):
            assert SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(host, 'old-round') == 1
            eventually(lambda: len(attempts) == 1 and not service.paths_busy((str(old),)))
        diagnostics = [record.getMessage() for record in caplog.records
                       if record.name == 'base.raw_audio_csv_service']
        assert len(diagnostics) == 1
        assert str(old) in diagnostics[0]
        assert 'OSError' in diagnostics[0] and 'disk locked' in diagnostics[0]
        assert len(diagnostics[0]) < len(str(old)) + 1200
        assert old.read_bytes() == b'preserved recording'
        assert service.snapshot().phase == 'open'
        permit = service.try_acquire_mutation((str(old),))
        assert permit is not None
        assert service.release_mutation(permit)
        request = submit(service, tmp_path, 'next-export')
        eventually(lambda: service.snapshot().outstanding == 0)
        from pathlib import Path
        assert Path(request.csv_path + '.zip').is_file()
        assert attempts == [str(old)]
    finally:
        service.begin_shutdown()
        assert service.closed.wait(5)


@pytest.mark.parametrize('stage', ['zip_write', 'zip_verify', 'zip_publish', 'csv_cleanup'])
def test_serial_cleanup_waits_real_zip_then_recording_lease_and_only_deletes_old_wav(tmp_path, monkeypatch, stage):
    import multiprocessing
    from pathlib import Path
    from base.raw_audio_csv_service import RawAudioCsvService
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    from unit_test.base.test_raw_audio_csv_service import eventually
    from unit_test.base.test_raw_audio_csv_worker import make_command
    context = multiprocessing.get_context('spawn')
    entered, gate = context.Event(), context.Event()
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(stage, entered, gate, 'warning'))
    request = make_command(tmp_path, 'old').request
    old = Path(request.wav_path)
    new = tmp_path / 'new.wav'
    new.write_bytes(b'new round')
    leased = {str(old)}
    deleted = []
    class Manager:
        def delete_audio(self, path):
            assert not leased
            assert service.try_acquire_mutation((path,)) is None
            deleted.append(path)
            Path(path).unlink()
            return error_code.OK, 'deleted'
    monkeypatch.setattr(serial_ops_module, 'RecordingManager', Manager)
    host = SimpleNamespace(raw_audio_csv_service=service,
        recording_bridge=SimpleNamespace(service=SimpleNamespace(is_path_leased=lambda path: path in leased)),
        recent_test_sessions=['old'], recent_test_session_by_id={'old': {'group_id': 'old-round', 'recorded_path': str(old)}},
        default_logger=_Logger(), recent_session_panel=None)
    try:
        assert service.commit(service.reserve(request.recording_id).reservation, request) == 'accepted'
        assert entered.wait(10)
        assert SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(host, 'old-round') == 1
        host.recent_test_session_by_id['new'] = {'group_id': 'new-round', 'recorded_path': str(new)}
        assert not deleted and old.exists()
        gate.set()
        eventually(lambda: service.snapshot().outstanding == 0)
        assert not deleted and old.exists()
        assert service.try_acquire_mutation((str(old),)) is None
        leased.clear()
        eventually(lambda: len(deleted) == 1 and not service.paths_busy((str(old),)))
        assert deleted == [str(old)]
        assert new.read_bytes() == b'new round'
        assert Path(request.csv_path).exists() and Path(request.csv_path + '.zip').exists()
    finally:
        gate.set()
        leased.clear()
        service.begin_shutdown()
        assert service.closed.wait(10)
