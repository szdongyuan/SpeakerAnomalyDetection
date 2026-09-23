import logging
import os
import threading
import time

import numpy as np
import soundfile as sf
import pytest

from base.raw_audio_csv_protocol import CsvExportRequest
from base.raw_audio_csv_service import RawAudioCsvService
from unit_test.ui.conftest import ui_qapp


def test_correlated_parent_timing_survives_disabled_logging_and_delayed_qt(tmp_path, ui_qapp):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    source = tmp_path / 'source.wav'
    sf.write(source, np.ones((32, 2), dtype=np.float32), 8000, subtype='FLOAT')
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    events, delivered, threads = [], [], []
    service.subscribe(lambda event: (events.append(event), threads.append(threading.current_thread().name)))
    bridge.subscribe(delivered.append)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        request = CsvExportRequest('task', 'recording', str(source), str(tmp_path / 'out.csv'), (0, 2), 'g', 'r')
        token = service.reserve('recording').reservation
        assert service.commit(token, request) == 'accepted'
        service.begin_shutdown()
        assert service.closed.wait(15)

        time.sleep(.05)  # Deliberately leave Qt delivery queued after resources close.
        ui_qapp.processEvents()
        stages = {e.timing.stage: e for e in events if e.timing is not None}
        assert {'reserve', 'submit', 'dispatch', 'ready', 'terminal_received', 'released'} <= stages.keys()
        assert stages['reserve'].timing.recording_id == 'recording'
        assert stages['submit'].timing.token_id == token.token_id
        for stage in ('dispatch', 'terminal_received', 'released'):
            timing = stages[stage].timing
            assert timing.task_id == 'task'
            assert timing.generation == 1
            assert timing.worker_pid != os.getpid()
            assert timing.worker_pid > 0
        terminal = next(e for e in delivered if e.kind == 'terminal')
        assert terminal.qt_delivery_seconds >= .05
        assert terminal.result.elapsed_seconds == stages['terminal_received'].result.elapsed_seconds
        assert terminal.result.export_end_seconds - terminal.result.export_begin_seconds == pytest.approx(terminal.result.csv_export_seconds)
        phases = [terminal.result.csv_export_seconds, terminal.result.zip_write_seconds,
                  terminal.result.zip_verify_seconds, terminal.result.zip_publish_seconds,
                  terminal.result.csv_cleanup_seconds]
        assert all(value >= 0 for value in phases)
        assert terminal.result.elapsed_seconds >= sum(phases)
        assert service.snapshot().outstanding == 0
        assert all(name in ('raw-csv-supervisor', 'raw-csv-completion') for name in threads)
    finally:
        logging.disable(previous)
        bridge.close_delivery()
        service.begin_shutdown()
        assert service.closed.wait(15)


def test_reused_worker_retains_original_ready_observation(tmp_path):
    from tools.benchmark_raw_audio_csv_process import CsvRun
    source = tmp_path / 'source.wav'
    sf.write(source, np.ones((20, 1), dtype=np.float32), 8000, subtype='FLOAT')
    service = RawAudioCsvService()
    reports = []
    try:
        for index in range(2):
            run = CsvRun('process', service)
            run.start(source, tmp_path / f'{index}.csv', (0,))
            assert run.done.wait(15)
            run.close()
            reports.append(run.evidence())
        dispatches = [next(e['timing'] for e in report['events'] if e['kind'] == 'dispatch') for report in reports]
        assert dispatches[0]['worker_ready_seconds'] == dispatches[1]['worker_ready_seconds']
        assert dispatches[0]['worker_ready_seconds'] <= dispatches[0]['parent_seconds']
        assert dispatches[0]['worker_pid'] == dispatches[1]['worker_pid']
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)


def test_submit_timestamp_precedes_dispatch_while_commit_return_is_delayed(tmp_path, monkeypatch):
    from tools.benchmark_raw_audio_csv_process import CsvRun
    # Drive one supervisor dispatch deterministically after ledger publication,
    # before commit returns. No process/file work is needed for this interleaving.
    monkeypatch.setattr(RawAudioCsvService, '_run', lambda self: None)
    service = RawAudioCsvService()
    service._thread.join()
    run = CsvRun('process', service)
    published, dispatch_done = threading.Event(), threading.Event()
    original_commit = service._ledger.commit
    failures = []
    def delayed_commit(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        published.set()
        assert dispatch_done.wait(5)
        return result
    monkeypatch.setattr(service._ledger, 'commit', delayed_commit)
    def submit():
        try:
            run.start(tmp_path / 'source.wav', tmp_path / 'output.csv', (0,))
        except Exception as error:
            failures.append(error)
    caller = threading.Thread(target=submit)
    caller.start()
    try:
        assert published.wait(5)
        task = service._ledger.dispatch_next(1)
        assert task is not None
        service._emit('dispatch', task=task)
    finally:
        dispatch_done.set()
        caller.join(5)
    assert not caller.is_alive()
    assert not failures
    service._drain_observations()
    report = run.evidence()
    run.close()
    events = [e for e in report['events'] if e['kind'] in ('submit', 'dispatch')]
    assert [e['kind'] for e in events] == ['dispatch', 'submit']
    dispatch, submit = [e['timing']['parent_seconds'] for e in events]
    assert submit <= dispatch
    assert report['queue_seconds'] == dispatch - submit >= 0


def test_malformed_commit_diagnostics_preserve_invalid_contract():
    service = RawAudioCsvService()
    try:
        token = service.reserve('invalid-recording').reservation
        assert service.commit(token, None) == 'invalid'
        assert service.snapshot().outstanding == 0
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)


def test_benchmark_refuses_existing_work_or_report_and_hash_mismatch(tmp_path):
    from tools.benchmark_raw_audio_csv_process import prepare_run, check_content, csv_evidence
    source = tmp_path / 'source.wav'
    source.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path, tmp_path / 'report.json', source=source)
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path / 'fresh', source, source=source)
    output = tmp_path / 'output.csv'
    output.write_bytes(b'\xef\xbb\xbftime_s,CH1\n0.000000000,1\n')
    evidence = csv_evidence(output)
    assert evidence['csv_data_rows'] == 1
    assert check_content(evidence, {**evidence, 'sha256': 'wrong'}) == 'fail'
    assert check_content(evidence, evidence) == 'pass'
    assert source.read_bytes() == b'original'


def test_missing_and_failed_hard_gates_never_pass():
    from tools.benchmark_raw_audio_csv_process import validate_recording, validate_analysis, validate_resources, percentiles
    recording = dict(expected_frames=100, actual_frames=99, expected_sample_rate=8000,
        sample_rate=8000, expected_channels=2, channels=2, trim_frames=0,
        raw_frames=100, target_frames=100, written_frames=99, drop_count=0)
    assert validate_recording(recording) == 'fail'
    recording.update(actual_frames=100, written_frames=100, drop_count=None)
    assert validate_recording(recording) == 'unverified'
    recording['drop_count'] = 0
    assert validate_recording(recording) == 'pass'
    analysis = dict(execution_status='分析完成', expected_instance_count=2,
        completed_instance_count=2, failure_count=1, judgement={'overall': 'NG'})
    assert validate_analysis(analysis, analysis['judgement']) == 'fail'
    analysis['failure_count'] = 0
    assert validate_analysis(analysis, {'overall': 'OK'}) == 'fail'
    assert validate_analysis(analysis, None) == 'unverified'
    assert validate_analysis(analysis, analysis['judgement']) == 'pass'
    assert validate_resources({'samples': [], 'processes': {}}) == 'unverified'
    assert validate_resources({'samples': [{'pid': 1}], 'expected_roles': ['parent', 'recording'],
        'processes': {'parent:1': {'role': 'parent', 'cpu_seconds': 1, 'peak_working_set_bytes': 1,
                                 'incomplete': False}}}) == 'unverified'
    assert percentiles([1, 2, 3]) == {'p50': 2.0, 'p95': 2.9, 'max': 3.0}


@pytest.mark.parametrize('csv_mode', ['off', 'thread', 'process'])
def test_real_concurrency_load_produces_gui_recording_analysis_evidence(tmp_path, csv_mode):
    import json
    import subprocess
    import sys
    source = tmp_path / 'source.wav'
    sf.write(source, np.tile(np.linspace(-.1, .1, 8000, dtype=np.float32)[:, None], (1, 2)),
             8000, subtype='FLOAT')
    work = tmp_path / 'concurrency'
    result = subprocess.run([sys.executable, 'tools/benchmark_raw_audio_csv_process.py',
        '--mode', 'concurrency', '--source-wav', str(source), '--work-dir', str(work),
        '--report', str(work / 'report.json'), '--conditions', '200', '--csv-mode', csv_mode],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
    report = json.loads((work / 'report.json').read_text(encoding='utf-8'))
    assert result.returncode == 0, report
    run = report['runs'][0]
    assert run['gui']['history_conditions'] == 200
    assert run['gui']['physical_plots'] == 2
    assert run['gui']['database_rows'] == 1
    assert len(run['gui']['stages']) >= 6
    assert run['recording']['validation_status'] == 'pass', run['recording']
    assert run['recording']['capture_observed_seconds'] is not None
    assert run['recording']['qt_started_seconds'] >= run['recording']['capture_observed_seconds']
    startup = run['recording']['startup_timing']
    assert startup['request_id'] == run['recording']['request_id']
    assert startup['generation'] >= 1 and startup['worker_pid'] != report['environment']['parent_pid']
    assert run['recording']['capture_observed_seconds'] == pytest.approx(
        startup['capture_observed_seconds'] - startup['accepted_seconds'])
    assert run['recording']['qt_started_seconds'] == pytest.approx(
        startup['qt_started_seconds'] - startup['accepted_seconds'])
    assert run['analysis']['execution_status'] == '分析完成'
    assert run['analysis']['completed_instance_count'] == 2
    assert run['analysis']['failure_count'] == 0
    assert run['analysis']['validation_status'] == 'unverified'  # No supplied judgement baseline.
    if csv_mode == 'process':
        assert run['csv']['result']['worker_pid'] != report['environment']['parent_pid']
        assert run['csv']['content']['member_name'] == 'raw.csv'
        assert run['csv']['content']['output_bytes'] == run['csv']['result']['bytes_written']
        assert run['csv']['content']['archive_bytes'] == run['csv']['result']['archive_bytes']
        assert not (work / 'run-1' / 'raw.csv').exists()
    elif csv_mode == 'thread':
        assert run['csv']['result']['worker_pid'] == report['environment']['parent_pid']
    else:
        assert run['csv']['result'] is None
        assert not (work / 'run-1' / 'raw.csv').exists()
    assert run['resources']['validation_status'] == 'pass', run['resources']
    assert report['status'] == 'unverified'
    assert report['clean_exit'] is True


@pytest.mark.parametrize('stage', ['database', 'gui'])
def test_concurrency_setup_failure_preserves_error_and_releases_resources(tmp_path, ui_qapp, monkeypatch, stage):
    from types import SimpleNamespace
    from tools import raw_audio_csv_benchmark_load as load
    from consts import model_consts
    from base.db_manager import DataSave
    original_database = model_consts.DATABASE_PATH
    samplers = []
    original_sampler = load.ResourceSampler
    def sampler():
        instance = original_sampler()
        samplers.append(instance)
        return instance
    def fail(*args, **kwargs):
        raise ValueError('injected setup failure')
    monkeypatch.setattr(load, 'ResourceSampler', sampler)
    if stage == 'database':
        monkeypatch.setattr(DataSave, 'create_table', fail)
    else:
        monkeypatch.setattr(load, 'build_host', fail)
    with pytest.raises(ValueError, match='injected setup failure'):
        load.one_trial(ui_qapp, SimpleNamespace(csv_mode='off'), tmp_path / 'run', tmp_path / 'source.wav',
                       (0,), [], None, None, None, None)
    assert model_consts.DATABASE_PATH == original_database
    assert not samplers[0].thread.is_alive()
    assert samplers[0].handles == {}


def test_thread_and_off_modes_do_not_construct_product_csv_service(tmp_path, monkeypatch):
    from tools import benchmark_raw_audio_csv_process as tool
    calls = []
    monkeypatch.setattr(tool, 'export_raw_audio_csv', lambda *args: calls.append(threading.get_ident()))
    caller = threading.get_ident()
    for mode in ('off', 'thread'):
        run = tool.CsvRun(mode)
        run.start(tmp_path / 'source.wav', tmp_path / f'{mode}.csv', (0,))
        assert run.done.wait(3)
        run.close()
        assert run.service is None
    assert len(calls) == 1 and calls[0] != caller


def test_analysis_baseline_requires_same_input_and_configuration():
    from tools.raw_audio_csv_benchmark_load import resolve_analysis_baseline
    config = {'type': 'SPL'}
    judgement = {'overall': 'NG'}
    baseline = dict(source_sha256='input', analysis_config=config, runs=[{'analysis': {
        'execution_status': '分析完成', 'expected_instance_count': 1, 'completed_instance_count': 1,
        'failure_count': 0, 'judgement': judgement}}])
    assert resolve_analysis_baseline(baseline, 'input', config) == judgement
    with pytest.raises(ValueError, match='input'):
        resolve_analysis_baseline(baseline, 'different', config)
    with pytest.raises(ValueError, match='configuration'):
        resolve_analysis_baseline(baseline, 'input', {'type': 'FFT'})
    assert resolve_analysis_baseline({}, 'input', config) is None


@pytest.mark.parametrize('payload,header,last,rows', [
    (b'', '', '', 0), (b'\xef\xbb\xbf', '', '', 0),
    (b'time_s,CH1', 'time_s,CH1', 'time_s,CH1', 0),
    (b'\xef\xbb\xbftime_s,CH1\n', 'time_s,CH1', 'time_s,CH1', 0),
    (b'time_s,CH1\n0,1', 'time_s,CH1', '0,1', 1),
    (b'time_s,CH1\n0,1\n', 'time_s,CH1', '0,1', 1),
    (('time_s,CH1\n' + '中' * (1024 * 1024) + '\n1,2').encode(),
     'time_s,CH1', '1,2', 2),
    (('time_s,CH1\n' + '中' * 5000).encode(), 'time_s,CH1', '中' * 5000, 1),
], ids=['empty', 'bom', 'header', 'bom-header', 'no-final-lf', 'final-lf', 'cross-block', 'long-tail'])
def test_zip_evidence_streams_exact_csv_content(tmp_path, monkeypatch, payload, header, last, rows):
    import hashlib
    import zipfile
    from tools.benchmark_raw_audio_csv_process import zip_csv_evidence
    path = tmp_path / '中文 example.v2.csv.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('中文 example.v2.csv', payload)
    original = zipfile.ZipExtFile.read
    reads = []
    def bounded_read(self, size=-1):
        assert 0 < size <= 1024 * 1024
        reads.append(size)
        return original(self, size)
    monkeypatch.setattr(zipfile.ZipExtFile, 'read', bounded_read)
    actual = zip_csv_evidence(path)
    assert reads
    assert actual['sha256'] == hashlib.sha256(payload).hexdigest()
    assert actual['output_bytes'] == len(payload)
    assert actual['archive_bytes'] == path.stat().st_size
    assert actual['member_name'] == '中文 example.v2.csv'
    assert actual['header'] == header
    assert actual['last_row'] == last
    assert actual['csv_data_rows'] == rows
    assert actual['compression_ratio_formula'] == 'archive_bytes / output_bytes'
    assert actual['compression_ratio'] == (path.stat().st_size / len(payload) if payload else None)


@pytest.mark.parametrize('members', [[], ['wrong.csv'], ['sample.csv', 'extra.csv'], ['sample.csv/']])
def test_zip_evidence_rejects_invalid_members(tmp_path, members):
    import zipfile
    from tools.benchmark_raw_audio_csv_process import zip_csv_evidence
    path = tmp_path / 'sample.csv.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for member in members:
            archive.writestr(member, b'header\n')
    with pytest.raises(ValueError, match='member'):
        zip_csv_evidence(path)


def test_zip_evidence_reads_to_eof_and_rejects_crc_corruption(tmp_path):
    import zipfile
    from tools.benchmark_raw_audio_csv_process import zip_csv_evidence
    path = tmp_path / 'sample.csv.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('sample.csv', b'header\n0,123456789\n')
    data = path.read_bytes().replace(b'0,123456789', b'0,923456789')
    path.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile, match='CRC'):
        zip_csv_evidence(path)


def test_process_self_test_uses_confirmed_archive_and_removes_csv(tmp_path):
    from pathlib import Path
    from tools.benchmark_raw_audio_csv_process import self_test
    report = self_test(tmp_path)
    assert report['status'] == 'pass'
    assert report['bytes_equal'] and report['csv']['clean_exit']
    assert report['csv_absent'] and report['temporary_files'] == []
    assert Path(report['csv']['result']['archive_path']).is_file()
    assert report['content']['output_bytes'] == report['csv']['result']['bytes_written']
    assert report['content']['archive_bytes'] == report['csv']['result']['archive_bytes']


def test_disk_budget_covers_expansion_old_archive_and_accumulated_outputs():
    from tools.benchmark_raw_audio_csv_process import output_disk_budget
    csv_bytes = 1000000
    one = output_disk_budget(csv_bytes, repetitions=1, old_archive_bytes=700000)
    three = output_disk_budget(csv_bytes, repetitions=3, old_archive_bytes=700000)
    assert one > csv_bytes * 2 + 700000
    assert three > one + csv_bytes * 2
    assert output_disk_budget(0, repetitions=1) > 0
