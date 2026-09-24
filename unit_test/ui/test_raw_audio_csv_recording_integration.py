from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from base.analysis_artifact_paths import AnalysisStorageContext
from base.raw_audio_csv_service import RawAudioCsvService
from base.raw_audio_csv_tasks import CsvTaskLedger
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_streaming_ops as streaming_ops_module
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from ui.sequence.recording_process_context import RecordingProcessContext


class Host(SequenceWidgetRawCsvOpsMixin, SequenceWidgetStreamingOpsMixin):
    def __init__(self, tmp_path, service, enabled=True):
        self.raw_audio_csv_service = service
        self.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: enabled}
        self.default_logger = Mock()
        self._on_raw_audio_csv_export_failed = Mock()
        self._on_raw_audio_csv_export_succeeded = Mock()
        self._owned_raw_audio_csv_tasks = set()
        storage = AnalysisStorageContext(str(tmp_path.resolve()), 'project', 'model',
            'sample', 1, 'port', 'condition', datetime(2026, 9, 4, 10, 11, 12))
        self.recorded_path = str(tmp_path / 'recording.wav')
        sf.write(self.recorded_path, np.zeros((3, 2), dtype=np.float32), 8000)
        self.recorded_signal_info = {'analysis_storage': storage.to_metadata(),
            'round_data_group_id': 'original-group', 'round_data_record_key': 'original-record'}
        admission = self._reserve_raw_audio_csv_recording()
        self.context = RecordingProcessContext(
            request=SimpleNamespace(request_id=admission.recording_id, path=self.recorded_path),
            direction='', preview_enabled=False,
            recorded_signal_info=dict(self.recorded_signal_info),
            csv_reservation=admission.csv_reservation,
            csv_enabled_snapshot=admission.csv_enabled_snapshot)
        self._pending_raw_audio_csv_recording = None
        self._publishing_raw_audio_csv_context = self.context
        self.registrations = []
        self._register_round_file = lambda info, path, **kw: self.registrations.append((info, path))


def test_wav_only_snapshot_cannot_introduce_csv_later(tmp_path):
    host = Host(tmp_path, CsvTaskLedger(), enabled=False)
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = True
    assert host._schedule_raw_audio_csv_export((0, 2)) is False
    assert host.raw_audio_csv_service.snapshot().outstanding == 0


def test_submission_keeps_capacity_and_original_ownership_after_toggle(tmp_path):
    service = CsvTaskLedger()
    host = Host(tmp_path, service)
    for i in range(15):
        service.reserve(str(i))
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = False
    host.recorded_signal_info = {'round_data_group_id': 'new-group'}
    original_commit = service.commit
    seen = []
    def commit(token, request):
        assert len(host.registrations) == 2
        assert [entry[1] for entry in host.registrations] == [request.csv_path, request.csv_path + '.zip']
        assert all(entry[0]['round_data_group_id'] == 'original-group' for entry in host.registrations)
        assert host.registrations[0][0] is host.registrations[1][0]
        seen.append(request)
        return original_commit(token, request)
    service.commit = commit
    assert host._schedule_raw_audio_csv_export((0, 2))
    assert seen[0].raw_channels == (0, 2)
    assert seen[0].owner_group == 'original-group'
    assert seen[0].owner_record == 'original-record'
    assert seen[0].csv_path.endswith('recording.csv')
    assert service.snapshot().outstanding == 16
    assert service.snapshot().reserved == 15
    assert host.context.csv_reservation is None
    host._release_raw_audio_csv_context(host.context)
    assert service.snapshot().outstanding == 16


def test_rejected_submission_releases_token_and_keeps_wav(tmp_path):
    service = CsvTaskLedger()
    host = Host(tmp_path, service)
    permit = service.try_acquire_mutation((host.recorded_path,))
    assert host._schedule_raw_audio_csv_export((0, 2)) is False
    assert service.snapshot().outstanding == 0
    assert host.context.csv_reservation is None
    assert (tmp_path / 'recording.wav').exists()
    host._on_raw_audio_csv_export_failed.assert_called_once()
    service.release_mutation(permit)


def test_old_context_cleanup_cannot_release_new_pending_token(tmp_path):
    host = Host(tmp_path, CsvTaskLedger())
    newer = host._reserve_raw_audio_csv_recording()
    host._release_raw_audio_csv_context(host.context)
    host._release_raw_audio_csv_context(host.context)
    assert host.raw_audio_csv_service.snapshot().reserved == 1
    assert host._pending_raw_audio_csv_recording is newer
    host._release_raw_audio_csv_recording(newer)


def test_parent_never_formats_csv_and_draining_accepts_reserved_recording(tmp_path, monkeypatch):
    from base import raw_audio_csv_exporter
    def parent_export_forbidden(*args, **kwargs):
        raise AssertionError('parent formatted CSV')
    monkeypatch.setattr(raw_audio_csv_exporter, 'export_raw_audio_csv', parent_export_forbidden)
    service = RawAudioCsvService()
    events = []
    service.subscribe(events.append)
    try:
        host = Host(tmp_path, service)
        service.begin_shutdown()
        assert host._schedule_raw_audio_csv_export((0, 2))
        assert service.closed.wait(15)
        terminal = [event for event in events if event.kind == 'terminal']
        assert len(terminal) == 1
        assert terminal[0].result.frames == 3
        assert terminal[0].result.worker_pid != __import__('os').getpid()
        csv_path = host.registrations[0][1]
        from zipfile import ZipFile
        assert not Path(csv_path).exists()
        with ZipFile(csv_path + '.zip') as archive:
            assert archive.read(Path(csv_path).name).decode('utf-8-sig').splitlines()[0] == 'time_s,CH1,CH3'
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)


def test_shared_service_terminal_is_only_presented_by_submitting_window(tmp_path):
    from base.raw_audio_csv_protocol import CsvServiceEvent, CsvTaskSnapshot, CsvFailure
    service = CsvTaskLedger()
    host = Host(tmp_path, service)
    other = Host(tmp_path, service)
    requests = []
    original = service.commit
    def commit(token, request):
        requests.append(request)
        return original(token, request)
    service.commit = commit
    host._schedule_raw_audio_csv_export((0, 2))
    request = requests[0]
    event = CsvServiceEvent('terminal', service.snapshot(), CsvTaskSnapshot(request, 'failed', 1),
        CsvFailure(request.task_id, 1, 'export', 'OSError', 'locked'))
    for consumer in (host, other, host):
        consumer._on_raw_audio_csv_service_event(event)
    host._on_raw_audio_csv_export_failed.assert_called_once()
    other._on_raw_audio_csv_export_failed.assert_not_called()


def test_csv_failure_handler_logs_and_shows_explicit_wav_safe_message(
    monkeypatch,
):
    messages = []
    logs = []
    host = SimpleNamespace(
        default_logger=SimpleNamespace(error=logs.append),
    )
    monkeypatch.setattr(
        streaming_ops_module.QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    SequenceWidgetStreamingOpsMixin._on_raw_audio_csv_export_failed(
        host,
        "D:/results/recording.wav",
        "target is locked",
    )

    assert "raw_audio_csv_export_failed" in logs[0]
    assert messages == [
        (
            "原始音频 CSV 保存失败",
            "WAV 已保存，但原始 CSV 保存失败。\ntarget is locked",
        )
    ]


# Use the established real recording process fixture, keeping hardware and DB
# inside the same temporary root as the CSV ownership assertions.
from unit_test.ui.test_recording_process_integration import service, workflow_host, pump


def recording_host_with_csv(service, tmp_path, monkeypatch):
    from types import MethodType
    host, save = workflow_host(service, tmp_path, monkeypatch)
    for name, method in SequenceWidgetRawCsvOpsMixin.__dict__.items():
        if callable(method):
            setattr(host, name, MethodType(method, host))
    host.raw_audio_csv_service = CsvTaskLedger()
    host._owned_raw_audio_csv_tasks = set()
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: True}
    host._csv_admission_notice = Mock()
    host._on_raw_audio_csv_export_failed = Mock()
    storage = AnalysisStorageContext(str(tmp_path.resolve()), 'project', 'model',
        'sample', 1, 'port', 'condition', datetime(2026, 9, 4, 10, 11, 12))
    host.recorded_signal_info.update(analysis_storage=storage.to_metadata(),
        round_data_group_id='original-group', round_data_record_key='original-record')
    host._register_round_file = Mock()
    host._discard_current_recent_session = Mock()
    return host, save


@pytest.mark.parametrize('outcome', ['metadata', 'initialization', 'start', 'cancel', 'invalid', 'invalid_result'])
def test_recording_failure_boundaries_release_exact_csv_reservation(
        ui_qapp, service, tmp_path, monkeypatch, outcome):
    host, save = recording_host_with_csv(service, tmp_path, monkeypatch)
    if outcome == 'metadata':
        host._begin_test_round_metadata = lambda: False
    elif outcome == 'initialization':
        host.reset_work_pram = Mock(side_effect=ValueError('invalid configuration'))
    elif outcome == 'start':
        host.recording_bridge.start = Mock(side_effect=RuntimeError('start failed'))
    elif outcome == 'invalid':
        service._backend_options['fail_write'] = True
        host.run = Mock()
    elif outcome == 'invalid_result':
        from dataclasses import replace
        receive = host._on_process_recording_result
        def receive_invalid_result(session, audio):
            receive(session, replace(audio, descriptor=replace(
                audio.descriptor, sample_rate=audio.descriptor.sample_rate + 1)))
        host._on_process_recording_result = receive_invalid_result
    else:
        service._backend_options['manual'] = True
    host.start_this_play()
    session = getattr(host, '_recording_process_session', None)
    if session is not None:
        if outcome == 'cancel':
            session.cancel()
        pump(ui_qapp, lambda: session.released.is_set() and not host._recording_contexts())
        host._on_process_recording_released(session)
    assert host.raw_audio_csv_service.snapshot().outstanding == 0
    assert host._pending_raw_audio_csv_recording is None
    save.assert_not_called()


def test_recording_transfers_stable_id_and_submission_survives_later_ui_exception(
        ui_qapp, service, tmp_path, monkeypatch):
    host, save = recording_host_with_csv(service, tmp_path, monkeypatch)
    service._backend_options['manual'] = True
    token = host._reserve_raw_audio_csv_recording()
    token_id = token.recording_id
    host.start_this_play()
    session = host._recording_process_session
    context = host._recording_contexts()[session.request.request_id]
    assert session.request.request_id == token_id
    assert context.csv_reservation.owner_id == token_id
    assert context.csv_enabled_snapshot
    assert host._pending_raw_audio_csv_recording is None
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = False
    host._cache_condition_record = Mock(side_effect=ValueError('late GUI exception'))
    for index in range(3):
        (tmp_path / f'feed-{index}').touch()
    pump(ui_qapp, lambda: session.released.is_set() and not host._recording_contexts())
    assert context.csv_reservation is None
    assert host.raw_audio_csv_service.snapshot().queued == 1
    host._on_process_recording_released(session)
    assert host.raw_audio_csv_service.snapshot().queued == 1
    assert host._register_round_file.call_count == 2
    assert Path(session.request.path).exists()


def test_qt_reentrant_busy_start_preserves_original_csv_through_submission(
        ui_qapp, service, tmp_path, monkeypatch):
    from PyQt5.QtCore import QTimer
    host, save = recording_host_with_csv(service, tmp_path, monkeypatch)
    service._backend_options['manual'] = True
    for index in range(15):
        host.raw_audio_csv_service.reserve(f'other-window-{index}')
    admission = host._reserve_raw_audio_csv_recording()
    original_id = admission.recording_id
    reentered = []
    def nested_start():
        reentered.append(host._record_workflow_busy)
        host.start_this_play()
    QTimer.singleShot(0, nested_start)
    host.start_this_play()
    session = host._recording_process_session
    context = host._recording_contexts()[session.request.request_id]
    assert reentered == [True]
    assert session.request.request_id == original_id
    assert context.csv_enabled_snapshot
    assert context.csv_reservation is not None
    assert host.raw_audio_csv_service.snapshot().outstanding == 16
    for index in range(3):
        (tmp_path / f'feed-{index}').touch()
    pump(ui_qapp, lambda: session.released.is_set() and not host._recording_contexts())
    assert host.raw_audio_csv_service.snapshot().reserved == 15
    assert host.raw_audio_csv_service.snapshot().queued == 1
    assert host._register_round_file.call_count == 2
    save.assert_called_once()
    timings = list(host._recording_stage_timings)
    assert {'csv_submission', 'waveform_projection', 'database_save', 'history_update',
            'automatic_analysis_enqueue', 'gui_complete'} <= {item['stage'] for item in timings}
    assert all(item['request_id'] == original_id and item['seconds'] >= 0 for item in timings)


@pytest.mark.parametrize('stage', ['export', 'zip_write', 'zip_verify', 'zip_publish', 'warning', 'success'])
def test_terminal_reports_actual_artifact_and_stage_once(tmp_path, monkeypatch, stage):
    from base.raw_audio_csv_protocol import CsvResult, CsvFailure, CsvServiceEvent, CsvTaskSnapshot
    host = Host(tmp_path, CsvTaskLedger())
    del host._on_raw_audio_csv_export_failed
    del host._on_raw_audio_csv_export_succeeded
    messages = []
    monkeypatch.setattr(streaming_ops_module.QMessageBox, 'warning', lambda *args: messages.append(args[1:]))
    assert host._schedule_raw_audio_csv_export((0, 2))
    task = host.raw_audio_csv_service.dispatch_next(1)
    req = task.request
    if stage in ('warning', 'success'):
        result = CsvResult(req.task_id, 1, req.csv_path, 123, .1, 3, 60,
            archive_path=req.csv_path + '.zip', archive_bytes=30,
            csv_retained=stage == 'warning', cleanup_diagnostics=('locked',) if stage == 'warning' else (),
            csv_export_seconds=.01, zip_write_seconds=.01, zip_verify_seconds=.01,
            zip_publish_seconds=.01, csv_cleanup_seconds=.01)
    else:
        result = CsvFailure(req.task_id, 1, stage, 'OSError', 'locked')
    event = CsvServiceEvent('terminal', host.raw_audio_csv_service.snapshot(), task, result)
    host.recorded_signal_info = {'round_data_group_id': 'new'}
    host._on_raw_audio_csv_service_event(event)
    host._on_raw_audio_csv_service_event(event)
    host._on_raw_audio_csv_service_event(CsvServiceEvent('released', event.snapshot, task, result))
    host._on_raw_audio_csv_service_event(event)
    if stage == 'success':
        assert not messages
    else:
        assert len(messages) == 1
        expected = 'ZIP 已保存但 CSV 清理失败' if stage == 'warning' else (
            'CSV 已保存' if stage.startswith('zip_') else '原始 CSV 保存失败')
        assert expected in messages[0][1]
        if stage.startswith('zip_') or stage == 'warning':
            assert req.csv_path in messages[0][1]
    if stage in ('warning', 'success'):
        assert req.csv_path + '.zip' in host.default_logger.info.call_args.args[0]


def test_scheduled_pcm24_csv_child_keeps_saved_samples_and_channel_labels(tmp_path):
    from base.save_data import save_audio_simple
    from zipfile import ZipFile
    import csv
    import io
    service = RawAudioCsvService()
    events = []
    service.subscribe(events.append)
    try:
        host = Host(tmp_path, service)
        source = np.array([[0.5, 2.5], [2**-25, -2**-25],
                           [0.001953125, -3]], dtype=np.float32)
        save_audio_simple(host.recorded_path, source, 8000)
        assert host._schedule_raw_audio_csv_export((7, 1))
        service.begin_shutdown()
        assert service.closed.wait(15)
        terminal = [event for event in events if event.kind == 'terminal']
        assert len(terminal) == 1
        assert terminal[0].result.frames == 3
        archive_path = host.registrations[1][1]
        with ZipFile(archive_path) as archive:
            rows = list(csv.reader(io.StringIO(archive.read('recording.csv').decode('utf-8-sig'))))
        assert rows[0] == ['time_s', 'CH8', 'CH2']
        assert [row[1:] for row in rows[1:]] == [
            ['0.5', '0.999999881'], ['0', '-1.1920929e-07'], ['0.001953125', '-1']]
        saved, _ = sf.read(host.recorded_path, dtype='float32', always_2d=True)
        np.testing.assert_array_equal(np.array([row[1:] for row in rows[1:]], dtype=np.float32), saved)
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)
