from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.raw_audio_csv_protocol import CsvExportRequest
from base.raw_audio_csv_tasks import CsvTaskLedger
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from ui.sequence.sequence_widget_round_reset_ops import SequenceWidgetRoundResetOpsMixin


def request(task='task'):
    return CsvExportRequest(task, 'recording', 'recording.wav', 'recording.csv', (0,), 'old', 'old')


@pytest.mark.parametrize('state', ['queued', 'running', 'unconfirmed'])
def test_all_unreleased_states_deny_both_file_mutations(state):
    ledger = CsvTaskLedger()
    assert ledger.commit(ledger.reserve('recording').reservation, request()) == 'accepted'
    if state != 'queued':
        ledger.dispatch_next(1)
        ledger.mark_running('task', 1)
    if state == 'unconfirmed':
        ledger.mark_terminal('task', 1, 'failed')
    for path in ('recording.wav', 'recording.csv'):
        assert ledger.try_acquire_mutation((path,)) is None


def test_round_reset_is_busy_with_only_a_csv_reservation():
    ledger = CsvTaskLedger()
    ledger.reserve('recording')
    host = SimpleNamespace(raw_audio_csv_service=ledger,
        _analysis_has_pending_tasks=lambda: False, _round_data_records={}, _round_reset_group_id='old')
    assert SequenceWidgetRoundResetOpsMixin._round_reset_busy(host)


def test_recording_context_claim_survives_until_atomic_commit(tmp_path):
    from unit_test.ui.test_raw_audio_csv_recording_integration import Host
    ledger = CsvTaskLedger()
    host = Host(tmp_path, ledger)
    permit = ledger.try_acquire_mutation((host.recorded_path,))
    host.context.csv_path_permit = permit
    assert host._schedule_raw_audio_csv_export((0, 2))
    host._release_raw_audio_csv_context(host.context)
    assert ledger.paths_busy((host.recorded_path,))
    assert ledger.snapshot().queued == 1


def test_main_archive_entry_injects_shared_services(monkeypatch):
    import main_window as module
    host = SimpleNamespace(raw_audio_csv_service=CsvTaskLedger(), recording_bridge=SimpleNamespace(service=object()))
    factory = Mock()
    monkeypatch.setattr(module, 'ArchiveAudioDataDialog', factory)
    module.MainWindow.on_audio_manager_init(host)
    assert factory.call_args.kwargs['raw_audio_csv_service'] is host.raw_audio_csv_service
    assert factory.call_args.kwargs['recording_service'] is host.recording_bridge.service


@pytest.mark.parametrize('blocked', [False, True])
def test_actual_recording_start_claims_path_before_bridge_start(ui_qapp, tmp_path, blocked):
    from unit_test.ui.test_recording_process_integration import main_host
    from base.recording_service import RecordingSession
    ledger = CsvTaskLedger()
    service = SimpleNamespace(is_path_leased=lambda path: False)
    host = main_host(service, tmp_path)
    host.raw_audio_csv_service = ledger
    host._discard_current_recent_session = Mock()
    if blocked:
        ledger.try_acquire_mutation((host.recorded_path,))
    calls = []
    def start(request, callbacks):
        calls.append(request)
        assert ledger.try_acquire_mutation((request.path,)) is None
        return RecordingSession(service, request, callbacks)
    host.recording_bridge.start = start
    recorded, rate = host.reset_work_pram()
    if blocked:
        with pytest.raises(RuntimeError, match='CSV'):
            host._start_process_recording(recorded, rate)
        assert not calls
    else:
        host._start_process_recording(recorded, rate)
        context = next(iter(host._recording_contexts().values()))
        assert context.csv_path_permit is not None
        SequenceWidgetRawCsvOpsMixin._release_raw_audio_csv_context(host, context)
        assert not ledger.paths_busy((host.recorded_path,))


@pytest.mark.parametrize('product', [True, False])
def test_database_only_label_allowed_but_legacy_move_denied(ui_qapp, tmp_path, monkeypatch, product):
    from unit_test.ui.test_recording_process_integration import main_host
    from base.recording_management import RecordingManager
    from consts import model_consts
    host = main_host(SimpleNamespace(is_path_leased=lambda path: False), tmp_path)
    source = tmp_path / 'not_labeled' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'original')
    host.raw_audio_csv_service = CsvTaskLedger()
    host.raw_audio_csv_service.try_acquire_mutation((str(source),))
    update = Mock(return_value=(0, 'updated'))
    monkeypatch.setattr(RecordingManager, 'update_audio_label', update)
    info = {'file_path': str(source), 'labels': 'not_labeled', model_consts.RECORDING_ROOT_CONFIG_KEY: str(tmp_path)}
    if product:
        info['analysis_storage'] = {}
    result = host._relabel_stored_audio_record(str(source), info, 'OK')
    assert (result[0] == 0) is product
    assert source.read_bytes() == b'original'
    assert update.call_count == int(product)


@pytest.mark.parametrize('csv_finishes', [False, True])
def test_context_cleanup_keeps_recording_claim_until_lease_released(tmp_path, csv_finishes):
    from unit_test.ui.test_raw_audio_csv_recording_integration import Host
    ledger = CsvTaskLedger()
    host = Host(tmp_path, ledger, enabled=csv_finishes)
    leased = {host.recorded_path}
    host.recording_bridge = SimpleNamespace(service=SimpleNamespace(is_path_leased=lambda path: path in leased))
    host.context.csv_path_permit = ledger.try_acquire_mutation((host.recorded_path,))
    if csv_finishes:
        assert host._schedule_raw_audio_csv_export((0, 2))
        task = ledger.dispatch_next(1)
        ledger.mark_terminal(task.request.task_id, 1, 'succeeded')
        ledger.release_task(task.request.task_id, 1)
    host._release_raw_audio_csv_context(host.context)
    assert ledger.try_acquire_mutation((host.recorded_path,)) is None
    assert ledger.run_deferred_mutations() == 0
    leased.clear()
    ledger.run_deferred_mutations()
    permit = ledger.try_acquire_mutation((host.recorded_path,))
    assert permit is not None
    host._release_raw_audio_csv_context(host.context)
    assert ledger.release_mutation(permit)


@pytest.mark.parametrize('leased', [False, True])
def test_wav_only_publication_releases_completed_claim_before_legacy_label(tmp_path, leased):
    from unit_test.ui.test_raw_audio_csv_recording_integration import Host
    ledger = CsvTaskLedger()
    host = Host(tmp_path, ledger, enabled=False)
    host.recording_bridge = SimpleNamespace(service=SimpleNamespace(is_path_leased=lambda path: leased))
    host.context.csv_path_permit = ledger.try_acquire_mutation((host.recorded_path,))
    assert host._schedule_raw_audio_csv_export((0, 2)) is False
    assert ledger.paths_busy((host.recorded_path,)) is leased


def test_legacy_relabel_claim_includes_possible_rollback_destination(ui_qapp, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host
    from base.recording_management import RecordingManager
    from consts import model_consts
    host = main_host(SimpleNamespace(is_path_leased=lambda path: False), tmp_path)
    source = tmp_path / 'misc' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'original')
    ledger = host.raw_audio_csv_service = CsvTaskLedger()
    ledger.try_acquire_mutation((str(tmp_path / 'not_labeled' / source.name),))
    update = Mock(return_value=(0, 'updated'))
    monkeypatch.setattr(RecordingManager, 'update_audio_label', update)
    info = {'file_path': str(source), 'labels': 'not_labeled', model_consts.RECORDING_ROOT_CONFIG_KEY: str(tmp_path)}
    result = host._relabel_stored_audio_record(str(source), info, 'OK')
    assert result[0] != 0
    assert source.read_bytes() == b'original'
    update.assert_not_called()


def test_service_retains_recording_claim_and_runs_cleanup_only_on_supervisor(tmp_path):
    from threading import get_ident
    from base.raw_audio_csv_service import RawAudioCsvService
    from unit_test.base.test_raw_audio_csv_service import eventually
    service = RawAudioCsvService()
    caller = get_ident()
    released = Event()
    checked = Event()
    deleted = Event()
    path = str(tmp_path / 'recording.wav')
    permit = service.try_acquire_mutation((path,))
    def ready():
        assert get_ident() != caller
        checked.set()
        return released.is_set()
    def cleanup():
        assert get_ident() != caller
        assert service.try_acquire_mutation((path,)) is None
        deleted.set()
    try:
        assert service.release_mutation(permit, ready=ready)
        assert service.defer_mutation((path,), cleanup, ready=ready)
        assert checked.wait(2)
        assert service.try_acquire_mutation((path,)) is None
        assert not deleted.is_set()
        released.set()
        assert deleted.wait(2)
        eventually(lambda: not service.paths_busy((path,)))
    finally:
        released.set()
        service.begin_shutdown()
        assert service.closed.wait(3)


@pytest.mark.parametrize('stage', ['zip_write', 'zip_verify', 'zip_publish', 'csv_cleanup'])
def test_real_zip_task_blocks_round_reset_and_legacy_move_but_allows_db_label(ui_qapp, tmp_path, monkeypatch, stage):
    import multiprocessing
    from pathlib import Path
    from base.raw_audio_csv_service import RawAudioCsvService
    from base.test_round_data import RoundDataRecord
    from base.recording_management import RecordingManager
    from consts import model_consts
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    from unit_test.base.test_raw_audio_csv_worker import make_command
    from unit_test.ui.test_recording_process_integration import main_host
    context = multiprocessing.get_context('spawn')
    entered, gate = context.Event(), context.Event()
    request = make_command(tmp_path, 'owned').request
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(stage, entered, gate))
    host = main_host(SimpleNamespace(is_path_leased=lambda path: False), tmp_path)
    host.raw_audio_csv_service = service
    host._round_data_records = {'old': {'record': RoundDataRecord(request.wav_path,
        {request.wav_path, request.csv_path, request.csv_path + '.zip'},
        raw_csv_files={request.csv_path, request.csv_path + '.zip'})}}
    update = Mock(return_value=(0, 'updated'))
    monkeypatch.setattr(RecordingManager, 'update_audio_label', update)
    try:
        assert service.commit(service.reserve(request.recording_id).reservation, request) == 'accepted'
        assert entered.wait(10)
        original = Path(request.wav_path).read_bytes()
        assert SequenceWidgetRoundResetOpsMixin._delete_round_generated_data(host, 'old') is None
        info = {'file_path': request.wav_path, 'labels': 'not_labeled', model_consts.RECORDING_ROOT_CONFIG_KEY: str(tmp_path)}
        assert host._relabel_stored_audio_record(request.wav_path, info, 'OK')[0] != 0
        update.assert_not_called()
        info['analysis_storage'] = {}
        assert host._relabel_stored_audio_record(request.wav_path, info, 'OK')[0] == 0
        update.assert_called_once()
        assert Path(request.wav_path).read_bytes() == original
        for path in (request.wav_path, request.csv_path, request.csv_path + '.zip'):
            assert service.try_acquire_mutation((path,)) is None
    finally:
        gate.set()
        service.begin_shutdown()
        assert service.closed.wait(10)
