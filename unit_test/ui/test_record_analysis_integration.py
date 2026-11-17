"""Real resident workers through MainWindow's sole poll timer and native details."""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import multiprocessing
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

import main_window
from base.analysis.service import AnalysisService
from base.record_analysis_reference import RecordAnalysisReference
from base.wav_calibration_metadata import append_wav_calibration_metadata
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_config_values import WorkbenchChannel, WorkbenchHardwareChannel
from base.workbench_recording_archive import WorkbenchRecordingArchive
from base.workbench_recording_protocol import StationWav, StationWavChannel, WorkbenchRecordingCompletion
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.sequence.station_plot_cell import StationPlotCell
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import startup
from unit_test.ui.test_workbench_analysis_lifecycle import wait_until


def write_calibrated_wav(path, rate=44100, duration=.4):
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(rate * duration)) / rate
    signal = .02 * np.sin(2 * np.pi * 1000 * time)
    sf.write(path, np.column_stack((signal, signal * 2)), rate, subtype='FLOAT')
    assert append_wav_calibration_metadata(path, {'recorded_channels': [
        {'wav_channel_index': index, 'physical_input_channel': index,
         'calibrated': True, 'v2pa_factor': 1., 'standard_spl': 94.}
        for index in (0, 1)]})


def analysis_queue(*, only_spl=False, weighting='A'):
    values = {
        'SPL': {'type': 'SPL', 'analysis_channels': [0, 1], 'weighting': weighting,
                'limit_checked': True, 'limit_metric': 'overall_spl',
                'scalar_upper_enabled': True, 'scalar_upper_value': 50.,
                'scalar_lower_enabled': False},
        'FFT': {'type': 'FFT', 'analysis_channels': [0, 1], 'n_fft': 1024},
        'FBA': {'type': 'FBA', 'analysis_channels': [0, 1]},
        'Spec': {'type': 'Spec', 'analysis_channels': [0, 1]},
        'LOUD': {'type': 'LOUD', 'analysis_channels': [0, 1],
                 'method': 'time_varying_iso532_1',
                 'advanced': {'curve_y_unit': 'phon', 'show_specific_loudness_heatmap': True},
                 'display': {'curves': ['loudness_time', 'specific_loudness_profile'],
                             'heatmaps': ['specific_loudness'],
                             'summary_metrics': ['steady_state_average_loudness', 'max_transient_loudness']}},
        'unsupported': {'type': 'AI', 'analysis_channels': [0, 1]},
    }
    if only_spl:
        values = {'SPL': values['SPL']}
    return AnalysisTestQueue('wav', {'display_sequence': list(values), **values})


def flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


def click_view(details):
    item = details.table.item(0, 5)
    details.table.scrollToItem(item)
    QApplication.processEvents()
    QTest.mouseClick(details.table.viewport(), Qt.LeftButton,
                     pos=details.table.visualItemRect(item).center())
    view = details.findChild(RecordAnalysisDialog)
    assert view is not None and view.isVisible()
    return view


@pytest.fixture
def real_runtime(startup, recording_workbench, monkeypatch, tmp_path):
    """Fake only acquisition/login/config; keep host, service, archive and labels real."""
    recording = recording_workbench
    channels = tuple(WorkbenchChannel(f'mic-{i}', name, 'sound_pressure',
                     WorkbenchHardwareChannel('offline-device', i))
                     for i, name in enumerate(('近端麦克风', '远端麦克风')))
    station = replace(recording.config.analysis.context.stations[0], channels=channels)
    condition = replace(recording.config.analysis.context.conditions[0],
                        applicable_station_ids=(station.station_id,))
    reverse = replace(condition, condition_id='reverse', condition_name='反转')
    config = configuration_with(recording.config, stations=(station,), conditions=(condition, reverse),
        test_queues={condition.test_queue: analysis_queue()},
        parameters={condition.test_queue: {'total_time': .4, 'sample_rate': 44100}})
    repository = Mock(source_identity='independent-integration-product.json')
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = config
    monkeypatch.setattr(main_window, 'MotorWorkbenchConfigRepository', lambda: repository)
    service = AnalysisService()
    window = startup.launch(lambda _: (None, None), analysis_service=service)
    controller = window.workbench_controller
    # Acquisition ports are deterministic; the host retains its real archive and UI paths.
    controller.bind_recording(replace(recording.controller._recording,
        submit_archive=controller._recording.submit_archive,
        completed=controller._recording.completed))
    summary = window.sequence_window.left_panel.dashboard_panel.summary_panel
    panel = window.sequence_window.left_panel.station_panel
    history = tmp_path / 'history'
    history.mkdir()
    (history / 'previous.csv').write_bytes(b'frequency,value\r\n1000,42\r\n')
    path = history / ('中文长文件名_' + 'historical-record-' * 6 + '.wav')
    write_calibrated_wav(path)
    wav_channels = tuple(StationWavChannel(ch.channel_id, ch.channel_name, ch.signal_type,
                        'recorded-device', i, i) for i, ch in enumerate(channels))
    reference = RecordAnalysisReference('historical-request', station.station_id,
                                        condition.condition_id, str(path), wav_channels)
    completion = WorkbenchRecordingCompletion(reference.request_id, 'product', condition.condition_id,
                     44100, 17640, (StationWav(station.station_id, str(path), wav_channels),))
    WorkbenchRecordingArchive(str(startup.database_path)).save(completion, '电机', datetime.now().date())
    now = datetime.now()
    summary.set_daily_wav_records([dict(completed_at=now, station_id=1, file_name=path.name,
        condition='原记录工况', result='NG', analysis_reference=reference)], now.date())
    summary._open_details()
    received, batches, submissions, changed = [], [], [], []
    window.workbench_analysis_event.connect(received.append)
    window.audio_label_sync_bridge.changed.connect(changed.append)
    poll = service.poll_events
    submit = service.submit_batch

    def observe_poll():
        batch = poll()
        batches.append(batch)
        return batch

    def observe_submit(payload):
        receipt = submit(payload)
        submissions.append((deepcopy(payload), receipt))
        return receipt

    monkeypatch.setattr(service, 'poll_events', observe_poll)
    monkeypatch.setattr(service, 'submit_batch', observe_submit)
    pids = set()
    state = SimpleNamespace(window=window, controller=controller, service=service,
        repository=repository, config=config, summary=summary, panel=panel,
        details=summary._details_dialog, history=history, database=startup.database_path,
        received=received, batches=batches, submissions=submissions, changed=changed,
        recording=recording, pids=pids)
    try:
        wait_until(lambda: service.station_states and all(
            value['state'] == 'ready' for value in service.station_states.values()), timeout=60000)
        pids.update(value['pid'] for value in service.station_states.values())
        assert len(pids) == 1 and None not in pids
        yield state
    finally:
        window.close()
        wait_until(lambda: getattr(window, '_exit_cleanup_complete', False), timeout=45000)
        details = summary._details_dialog
        if details is not None and not sip.isdeleted(details):
            details.deleteLater()
        flush_deletes()
        assert service.closed and window.audio_label_sync_bridge.is_closed
        assert window.audio_label_sync_bridge.pending == 0
        assert not window._analysis_timer.isActive()
        assert not (pids & {child.pid for child in multiprocessing.active_children()})


def database_rows(runtime):
    with closing(sqlite3.connect(runtime.database)) as database:
        return database.execute('SELECT * FROM audio_data_table ORDER BY file_path').fetchall()


def history_bytes(runtime):
    return {str(path.relative_to(runtime.history)): None if path.is_dir() else path.read_bytes()
            for path in runtime.history.rglob('*')}


def presentation_state(runtime):
    controller = runtime.controller
    grid = runtime.window.sequence_window.plot_grid
    return (deepcopy(runtime.panel.get_station_data(1)),
            deepcopy(controller._wav_state.records_for(datetime.now().date())),
            controller._current_round_id, controller._recording_round,
            tuple(getattr(runtime.summary, name + '_label').text()
                  for name in ('total', 'ok', 'ng', 'unmarked', 'rate')),
            tuple((id(cell), len(cell.plot_widget.getPlotItem().listDataItems()))
                  for cell in grid.findChildren(StationPlotCell)))


def current_session(runtime):
    return next(iter(runtime.controller._record_analysis._sessions.values()))


def start_production(runtime):
    previous = len(runtime.submissions)
    runtime.controller.start_recording()
    assert runtime.controller.recording_busy
    request = runtime.recording.requests[-1]
    for station in request.station_group.stations:
        write_calibrated_wav(Path(request.station_group.final_path) / station.filename,
                             request.sample_rate, .4)
    runtime.recording.complete()
    wait_until(lambda: len(runtime.submissions) > previous)
    assert runtime.submissions[-1][0]['tasks'][0].get('persist_artifacts', True)
    return runtime.submissions[-1][0]['tasks'][0]


def test_real_worker_details_all_types_are_read_only_and_production_still_persists(real_runtime):
    r = real_runtime
    before = history_bytes(r), database_rows(r), presentation_state(r)
    timer = r.window._analysis_timer
    timers = tuple(r.window.findChildren(QTimer))
    view = click_view(r.details)
    assert QApplication.activeModalWidget() is view
    assert view.windowModality() == Qt.WindowModal and view.parentWidget() is r.details
    assert view.navigation.currentRow() == 0
    assert r.controller.open_record_analysis(r.details.table.item(0, 5).data(Qt.UserRole), r.details) is view
    assert len(r.submissions) == 1 and r.submissions[0][1]['accepted']
    task = r.submissions[0][0]['tasks'][0]
    assert task['persist_artifacts'] is False and task['analysis_result_directory'] is None
    session = current_session(r)
    wait_until(lambda: bool(session.presenter.results), timeout=60000)
    assert list(session.presenter.results) == ['SPL', 'FFT', 'FBA', 'Spec', 'LOUD', 'unsupported']
    for index, key in enumerate(session.presenter.results):
        view.navigation.setCurrentRow(index)
        result = session.presenter.results[key]
        assert [channel.channel_index for channel in result.channels] == [0, 1]
        if key == 'unsupported':
            assert not view.content.plots and view.content.diagnostic_text()
            assert all(channel.judgement is None for channel in result.channels)
        else:
            assert all(channel.status == '分析完成' and channel.panels for channel in result.channels)
            assert view.content.plots
            if key in ('Spec', 'LOUD'):
                assert view.channel_selector.count() == 2 and len(view.content.images) == 1
                view.channel_selector.setCurrentIndex(1)
                assert session.presenter.selected_channel == 1
            if key == 'LOUD':
                profile = next(panel for panel in result.channels[0].panels
                               if panel.key == 'specific_loudness_profile')
                assert (profile.x_unit, profile.y_unit) == ('Bark', 'sone/Bark')
                assert len(profile.series[0]['x']) > 100
            if key == 'SPL':
                assert all(channel.metrics['calibration_source'] == 'wav' for channel in result.channels)
                assert all(channel.judgement == 'NG' for channel in result.channels)
    assert len(r.submissions) == 1
    assert (history_bytes(r), database_rows(r), presentation_state(r)) == before
    assert r.changed == [] and not any(event['event_type'] == 'task_result' for event in r.received)
    terminals = [event for batch in r.batches for event in batch
                 if event['event_type'] == 'task_result']
    assert len(terminals) == 1 and all(not item['artifacts'] for item in terminals[0]['payload']['instance_results'])
    assert r.window._analysis_timer is timer and tuple(r.window.findChildren(QTimer)) == timers
    assert {value['pid'] for value in r.service.station_states.values()} == r.pids
    view.close()
    flush_deletes()
    # Ordinary production uses its original default-persist, archive and result routes.
    r.config = configuration_with(r.config, test_queues={'queue-a': analysis_queue(only_spl=True)})
    r.repository.load_active.return_value = r.config
    r.controller.refresh_configuration()
    task = start_production(r)
    wait_until(lambda: len(r.changed) == 1, timeout=60000)
    event, = [event for event in r.received if event['event_type'] == 'task_result']
    assert event['task_id'] == task['task_id'] and event['is_current']
    items = event['payload']['instance_results']
    assert len(items) == 2 and all(item['judgement'] == 'NG' for item in items)
    artifacts = [artifact for item in items for artifact in item['artifacts']]
    assert artifacts and all(artifact['status'] == 'saved' and Path(artifact['path']).stat().st_size > 0
                             for artifact in artifacts)
    assert r.panel.get_station_data(1)['forward'] == 'NG'
    reverse = start_production(r)
    assert reverse['condition_id'] != task['condition_id']
    wait_until(lambda: len(r.changed) == 2, timeout=60000)
    assert r.panel.get_station_data(1)['result'] == 'NG'
    assert r.summary.total_label.text() == r.summary.ng_label.text() == '1'
    assert len(r.controller._result_state.records_for(datetime.now().date())) == 1
    assert len(r.controller._wav_state.records_for(datetime.now().date())) == 2
    rows = database_rows(r)
    assert len(rows) == 3 and [row[5] for row in rows].count('NG') == 2
    assert [row[5] for row in rows].count('not_labeled') == 1
    assert history_bytes(r) == before[0]
    assert {value['pid'] for value in r.service.station_states.values()} == r.pids


def test_real_late_results_snapshot_reopen_parent_destruction_and_shutdown_barrier(real_runtime, monkeypatch):
    r = real_runtime
    view = click_view(r.details)
    old = current_session(r)
    old_task = r.submissions[-1][0]['tasks'][0]['task_id']
    # Apply a valid new current config while this session owns its original snapshot.
    newer = configuration_with(r.config, test_queues={'queue-a': analysis_queue(only_spl=True, weighting='Z')})
    r.repository.load_active.return_value = newer
    r.controller.refresh_configuration()
    assert old.analysis is r.config.analysis and len(old.items) == 6
    assert r.controller.get_current_analysis_snapshot()[0] is newer.analysis
    view.close()  # Before a Qt tick can deliver the accepted worker terminal.
    assert old.presenter.disposed and not old.presenter.results
    assert r.controller._record_analysis._tasks[old_task].session_id is None
    flush_deletes()
    reopened = click_view(r.details)
    session = current_session(r)
    assert len(session.items) == 1 and session.analysis is newer.analysis
    assert session.items[0].parameters['weighting'] == 'Z'
    assert r.submissions[-1][0]['tasks'][0]['task_id'] != old_task
    wait_until(lambda: bool(session.presenter.results), timeout=60000)
    assert not r.controller._record_analysis._tasks and not old.presenter.results
    assert list(session.presenter.results) == ['SPL']
    reopened.close()
    flush_deletes()
    orphan = click_view(r.details)
    presenter = current_session(r).presenter
    r.details.deleteLater()
    flush_deletes()
    assert sip.isdeleted(orphan) and presenter.disposed and not presenter.results
    wait_until(lambda: not r.controller._record_analysis._tasks, timeout=60000)
    assert not any(event['event_type'] == 'task_result' for event in r.received)
    # Keep real worker results at the service boundary until the real shutdown
    # notification arrives, emulating one coalesced terminal poll batch.
    r.summary._open_details()
    r.details = r.summary._details_dialog
    r.config = newer
    poll = r.service.poll_events
    held, final_batches, barriers = [], [], []

    def coalesced_poll():
        events = poll()
        held.extend(events)
        if any(event['event_type'] == 'service_closed' for event in held):
            batch = list(held)
            held.clear()
            # Exercise notification before terminals, with the same real events.
            batch.sort(key=lambda event: event['event_type'] != 'service_closed')
            final_batches.append(batch)
            return batch
        return ()

    monkeypatch.setattr(r.service, 'poll_events', coalesced_poll)
    production = start_production(r)
    # A recording update refreshes details using the actual WAV ledger.
    view = click_view(r.details)
    wait_until(lambda: sum(event['event_type'] == 'task_result' for event in held) == 2, timeout=60000)
    assert all(event['payload']['execution_status'] == '分析完成'
               for event in held if event['event_type'] == 'task_result')
    r.window.workbench_analysis_event.connect(lambda event: barriers.append(
        (event['event_type'], r.window.audio_label_sync_bridge._closing,
         getattr(r.window, '_exit_cleanup_complete', False))))
    r.window.close()
    wait_until(lambda: getattr(r.window, '_exit_cleanup_complete', False), timeout=45000)
    assert final_batches and final_batches[-1][0]['event_type'] == 'service_closed'
    terminals = [event for batch in final_batches for event in batch if event['event_type'] == 'task_result']
    assert {event['task_id'] for event in terminals} == {
        production['task_id'], r.submissions[-1][0]['tasks'][0]['task_id']}
    assert all(not closing and not exited for _, closing, exited in barriers)
    assert [event['event_type'] for event in r.received].count('task_result') == 1
    assert r.window.audio_label_sync_bridge.is_closed and r.window.audio_label_sync_bridge.pending == 0
    assert len(r.changed) == 1 and r.changed[0].label == 'NG'
    assert [row[5] for row in database_rows(r)].count('NG') == 1
    assert not r.controller._record_analysis._tasks
