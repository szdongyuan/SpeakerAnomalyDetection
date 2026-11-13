from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.raw_audio_csv_protocol import CsvExportRequest
from base.raw_audio_csv_tasks import CsvTaskLedger


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


def test_main_archive_entry_injects_shared_services(monkeypatch):
    import main_window as module
    host = SimpleNamespace(raw_audio_csv_service=CsvTaskLedger(), recording_bridge=SimpleNamespace(service=object()))
    factory = Mock()
    monkeypatch.setattr(module, 'ArchiveAudioDataDialog', factory)
    module.MainWindow.on_audio_manager_init(host)
    assert factory.call_args.kwargs['raw_audio_csv_service'] is host.raw_audio_csv_service
    assert factory.call_args.kwargs['recording_service'] is host.recording_bridge.service


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
