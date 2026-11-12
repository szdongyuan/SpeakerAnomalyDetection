from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest

from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_plot_config import parse_plot_templates
from unit_test.base.workbench_plot_fixtures import template_layout
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import AnalysisServiceStub


@pytest.fixture
def plot_setup(recording_workbench):
    w = recording_workbench
    original = w.config.analysis.context.stations[0]
    stations = tuple(replace(original, station_id=f's{i}', station_name=f'工位{i + 1}', channels=tuple(
        replace(original.channels[0], channel_id=f'ch{i}-{j}', channel_name=f'声音{j + 1}',
                signal_type='sound_pressure', hardware_channel=replace(
                    original.channels[0].hardware_channel, physical_channel_index=j)) for j in range(2))) for i in range(4))
    conditions = tuple(replace(c, applicable_station_ids=tuple(s.station_id for s in stations),
                               segmented_analysis={'mode': 'none'}) for c in w.config.analysis.context.conditions)
    queues = {c.test_queue: AnalysisTestQueue('wav', {
        'display_sequence': ['声压级', '频谱'],
        '声压级': {'type': 'SPL', 'analysis_channels': [0, 1], 'weighting': 'Z',
                  'limit_checked': True, 'limit_metric': 'curve_y', 'limit_mode': 'manual',
                  'manual_input_mode': 'constant', 'constant_upper_enabled': True,
                  'constant_lower_enabled': False, 'constant_upper_value': 80},
        '频谱': {'type': 'FFT', 'analysis_channels': [0, 1], 'weighting': 'Z', 'n_fft': 2048},
    }) for c in conditions}
    layout = template_layout()
    layout['rows'] = 2
    layout['default_condition_display'].update(x_axis_mode='condition_local', condition_gap_px=16)
    layout['plot_templates']['spl']['y_axis'].update(empty_data_range_min=0, empty_data_range_max=100)
    fft = deepcopy(layout['plot_templates']['spl'])
    fft.update(title='频谱', analysis_item_name='频谱', analysis_type='FFT', x_axis={
        'label': '频率', 'scale': 'log', 'auto_range': False, 'range_min': 20, 'range_max': 20000})
    layout['plot_templates']['fft'] = fft
    layout['cells'].append([{'plot_template': 'fft'} for _ in range(4)])
    w.config = configuration_with(w.config, stations=stations, conditions=conditions,
                                  test_queues=queues, plot_layout=parse_plot_templates(layout))
    w.repository.result = w.config
    w.controller.refresh_configuration()
    w.grid = w.controller._plot_grid
    w.grid.resize(1540, 850)
    w.grid.show()
    return w


@pytest.fixture
def plotted(plot_setup):
    w = plot_setup
    w.analysis = AnalysisServiceStub()
    w.events = []
    w.controller.bind_analysis_service(w.analysis, w.events.append)
    w.controller.activate_analysis()
    return w


def start(w):
    w.controller.start_recording()
    w.callbacks[-1].started(SimpleNamespace(request=w.requests[-1]))


def result_event(task, value=60):
    items = []
    for key, kind in [('声压级', 'SPL'), ('频谱', 'FFT')]:
        for channel in (0, 1):
            items.append(dict(config_key=key, analysis_type=kind, channel_index=channel,
                              execution_status='分析完成', judgement_status='completed', judgement='OK',
                              metrics={}, error=None, artifacts=[], display_payload={
                                  'kind': 'curve', 'x': [0.2, 0.3, 0.4] if kind == 'SPL' else [100, 1000, 10000],
                                  'y': [value + channel, None, value + channel + 1],
                                  'upper': [80, None, 80], 'lower': [],
                                  'x_unit': 's' if kind == 'SPL' else 'Hz', 'y_unit': 'dB',
                              }))
    return dict(event_type='task_result', task_id=task['task_id'], station_id=task['station_id'], payload={
        'task_id': task['task_id'], 'station_id': task['station_id'], 'condition_id': task['condition_id'],
        'execution_status': '分析完成', 'instance_results': items, 'error': None,
    })


def deliver(w, *events):
    w.analysis.events.extend(events)
    w.controller.poll_analysis_events()


def deliver_completed_batch(w, event):
    tasks = next(batch['tasks'] for batch in w.analysis.submissions
                 if any(task['task_id'] == event['task_id'] for task in batch['tasks']))
    deliver(w, *(result_event(task) for task in tasks if task['task_id'] != event['task_id']), event)


def test_staggered_results_wait_for_all_stations_including_failure(plotted):
    w = plotted
    start(w); w.complete()
    tasks = w.analysis.submissions[0]['tasks']
    for task in tasks[:-1]:
        deliver(w, result_event(task))
        assert all(not w.grid._cells[0, col]._condition_series for col in range(4))
        column = int(task['station_id'][1:])
        assert w.grid._cells[0, column]._condition_states['warm'][0] == '分析中'
        # The controller still delivers each result to its other subscribers immediately.
        assert w.events[-1]['task_id'] == task['task_id']
    failed = result_event(tasks[-1])
    failed['payload'].update(execution_status='分析失败', instance_results=[], error={'message': '测试失败'})
    deliver(w, failed)
    assert all('warm' in w.grid._cells[0, col]._condition_series for col in range(3))
    assert w.grid._cells[0, 3]._condition_states['warm'][0] == '分析失败'
    assert w.grid.updatesEnabled()
    assert not w.controller._plot_presenter._batches


@pytest.mark.parametrize('station_count', [1, 2])
def test_display_batch_waits_only_for_participating_stations(plotted, station_count):
    w = plotted
    conditions = tuple(replace(c, applicable_station_ids=tuple(f's{i}' for i in range(station_count)))
                       for c in w.config.analysis.context.conditions)
    w.repository.result = configuration_with(w.config, conditions=conditions)
    w.controller.refresh_configuration()
    start(w); w.complete()
    tasks = w.analysis.submissions[0]['tasks']
    assert len(tasks) == station_count
    deliver(w, *(result_event(task) for task in tasks))
    assert all('warm' in w.grid._cells[0, col]._condition_series for col in range(station_count))


def test_completed_condition_does_not_wait_for_another_condition(plotted):
    w = plotted
    start(w); w.complete()
    warm = w.analysis.submissions[0]['tasks']
    deliver(w, result_event(warm[0]))
    start(w); w.complete()
    cold = w.analysis.submissions[1]['tasks']
    deliver(w, *(result_event(task) for task in cold))
    assert all(set(w.grid._cells[0, col]._condition_series) == {'cold'} for col in range(4))
    deliver(w, *(result_event(task) for task in warm[1:]))
    assert all(set(w.grid._cells[0, col]._condition_series) == {'warm', 'cold'} for col in range(4))


def test_demo_follows_completed_batch_only_and_is_not_submitted(plotted, qt_app):
    w = plotted
    layout = template_layout()
    layout['rows'] = 2
    rms = deepcopy(layout['plot_templates']['spl'])
    rms.update(analysis_type='VIBRATION_RMS', title='振动 RMS', value_unit='m/s²',
               data_source='simulated', channel_display={
                   'show_all_analyzed_channels': False, 'display_wav_channels': [1, 2, 3]})
    layout['plot_templates']['rms'] = rms
    layout['cells'].append([{'plot_template': 'rms'} for _ in range(4)])
    w.repository.result = configuration_with(w.config, plot_layout=parse_plot_templates(layout))
    w.controller.refresh_configuration()
    demo = w.grid._cells[1, 0]
    assert not demo._condition_series
    start(w)
    assert not demo._condition_series
    assert demo._condition_states['warm'][0] == '采集中'
    w.complete()
    for queue in w.analysis.submissions[0]['test_queues'].values():
        assert all(item.get('type') != 'VIBRATION_RMS' for item in queue['analysis_config'].values()
                   if isinstance(item, dict))
    tasks = w.analysis.submissions[0]['tasks']
    deliver(w, result_event(tasks[0]))
    assert not demo._condition_series
    deliver(w, *(result_event(task) for task in tasks[1:]))
    assert len(demo._condition_series['warm']) == 3
    assert 'cold' not in demo._condition_series
    assert demo._condition_states['warm'][0] == '模拟数据'
    assert not demo._condition_summaries['warm']
    demo.open_detail()
    qt_app.processEvents()
    assert '模拟数据' in demo._detail_dialog.cell.display.title
    assert demo._detail_dialog.cell._condition_states['warm'][0] == '模拟数据'
    start(w); w.complete(outcome='cancelled')
    assert 'cold' not in demo._condition_series
    assert demo._condition_states['cold'][0] == '已取消'
    start(w); w.complete()
    failed = []
    for task in w.analysis.submissions[1]['tasks']:
        event = result_event(task)
        event['payload'].update(execution_status='分析失败', instance_results=[], error={'message': '测试失败'})
        failed.append(event)
    deliver(w, *failed)
    assert 'cold' not in demo._condition_series
    assert demo._condition_states['cold'][0] == '未生成模拟数据'
    start(w)
    assert not demo._condition_series
    assert not demo._detail_dialog.cell._condition_series
    demo.close_detail()


def test_four_station_conditions_channels_limits_and_detail_from_memory(plotted, qt_app):
    w = plotted
    cell = w.grid._cells[0, 0]
    start(w)
    assert cell.empty_labels['warm'].text() == '采集中'
    assert cell.empty_labels['cold'].text() == '待采集'
    w.complete()
    assert cell.empty_labels['warm'].text() == '待分析'
    tasks = w.analysis.submissions[0]['tasks']
    deliver(w, dict(event_type='task_started', task_id=tasks[0]['task_id'], payload={}))
    assert cell.empty_labels['warm'].text() == '分析中'
    for index, task in enumerate(tasks):
        deliver(w, result_event(task, 50 + index * 10))
    cell.open_detail()
    start(w)
    assert 'warm' in cell._condition_series
    assert cell.empty_labels['cold'].text() == '采集中'
    w.complete()
    for index, task in enumerate(w.analysis.submissions[1]['tasks']):
        deliver(w, result_event(task, 55 + index * 10))
    qt_app.processEvents()
    for row in range(2):
        for column in range(4):
            current = w.grid._cells[row, column]
            for condition, base in [('warm', 50), ('cold', 55)]:
                curves = current._condition_series[condition]
                assert [(c['wav_channel'], c['role']) for c in curves] == [(1, 'curve'), (1, 'upper'), (2, 'curve'), (2, 'upper')]
                assert curves[0]['y'][0] == base + column * 10
                assert np.isnan(curves[0]['y'][1])
                assert current._condition_states[condition] == ('', '')
    assert set(cell._detail_dialog.cell._condition_series) == {'warm', 'cold'}
    cell.close_detail()


def test_new_round_invalidates_all_old_conditions_before_submission(plotted):
    w = plotted
    start(w); w.complete()
    old_warm = result_event(w.analysis.submissions[0]['tasks'][0])
    deliver(w, old_warm)
    start(w); w.complete()
    old_cold = result_event(w.analysis.submissions[1]['tasks'][0])
    start(w)  # New warm is recording, not submitted yet.
    cell = w.grid._cells[0, 0]
    assert not cell._condition_series
    deliver(w, old_cold, old_warm)
    assert not cell._condition_series
    assert cell.empty_labels['warm'].text() == '采集中'
    w.complete()
    latest = result_event(w.analysis.submissions[2]['tasks'][0], 70)
    deliver_completed_batch(w, latest)
    assert cell._condition_series['warm'][0]['y'][0] == 70
    deliver(w, old_cold, old_warm, latest)  # Duplicate/retired tasks cannot redraw.
    assert cell._condition_series['warm'][0]['y'][0] == 70
    assert 'cold' not in cell._condition_series


def test_config_switch_unknown_task_and_mismatched_identity_cannot_draw(plotted):
    w = plotted
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    unknown = deepcopy(event)
    unknown['task_id'] = 'unknown'
    deliver(w, unknown)
    assert not w.grid._cells[0, 0]._condition_series
    mismatch = deepcopy(event)
    mismatch['payload']['station_id'] = 's3'
    deliver_completed_batch(w, mismatch)
    assert w.grid._cells[0, 0].empty_labels['warm'].text() == '结果不匹配'
    start(w); w.complete()
    old = result_event(w.analysis.submissions[1]['tasks'][1])
    deliver(w, result_event(w.analysis.submissions[1]['tasks'][0]))
    w.repository.result = configuration_with(w.config, project_name='另一产品')
    w.controller.refresh_configuration()
    assert not w.controller._plot_presenter._batches
    deliver(w, old)
    assert not w.events[-1]['is_current']
    assert not w.grid._cells[0, 1]._condition_series


@pytest.mark.parametrize('problem', ['failed', 'unit', 'length', 'duplicate', 'missing'])
def test_bad_channel_does_not_discard_other_channel_or_analysis(plotted, problem):
    w = plotted
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    items = event['payload']['instance_results']
    if problem == 'failed':
        items[0].update(execution_status='分析失败', error={'message': '测试失败'})
    elif problem == 'unit':
        items[0]['display_payload']['y_unit'] = 'dBA'
    elif problem == 'length':
        items[0]['display_payload']['y'] = [1]
    elif problem == 'duplicate':
        items.append(deepcopy(items[0]))
    else:
        items.pop(0)
    deliver_completed_batch(w, event)
    cell = w.grid._cells[0, 0]
    assert {c['wav_channel'] for c in cell._condition_series['warm']} == {2}
    assert cell._condition_states['warm'][0] == '部分结果不可用'
    assert len(w.grid._cells[1, 0]._condition_series['warm']) == 4


def test_failed_task_and_rejected_submission_show_reason(plotted):
    w = plotted
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    event['payload'].update(execution_status='分析失败', instance_results=[], error={'message': 'WAV 无法读取'})
    deliver_completed_batch(w, event)
    cell = w.grid._cells[0, 0]
    assert cell.empty_labels['warm'].text() == '分析失败'
    assert 'WAV 无法读取' in cell.empty_labels['warm'].toolTip()
    w.analysis.submit_batch = lambda _: {'accepted': False, 'issues': [{'message': '队列已满'}]}
    start(w); w.complete()
    assert cell.empty_labels['cold'].text() == '分析未提交'
    assert '队列已满' in cell.empty_labels['cold'].toolTip()


def test_saving_failure_keeps_curves_and_scalar_limit_is_not_invented(plotted):
    w = plotted
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    item = event['payload']['instance_results'][0]
    item.update(metrics={'overall_upper_limit': 65}, artifacts=[{'status': 'failed', 'path': 'missing.csv'}])
    item['display_payload']['upper'] = []
    deliver_completed_batch(w, event)
    curves = w.grid._cells[0, 0]._condition_series['warm']
    assert [(c['wav_channel'], c['role']) for c in curves] == [(1, 'curve'), (2, 'curve'), (2, 'upper')]


def test_selected_channel_exact_item_and_unit_alias(plotted):
    w = plotted
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template['channel_display'] = {'show_all_analyzed_channels': False, 'display_wav_channels': [2]}
    queues = {name: AnalysisTestQueue('wav', {**queue.analysis_config,
              '声压级': {**queue.analysis_config['声压级'], 'weighting': 'A'}})
              for name, queue in w.config.analysis.test_queues.items()}
    w.repository.result = configuration_with(w.config, plot_layout=parse_plot_templates(layout), test_queues=queues)
    w.controller.refresh_configuration()
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    event['payload']['instance_results'][1]['display_payload']['y_unit'] = 'dBA'
    unrelated = deepcopy(event['payload']['instance_results'][1])
    unrelated['config_key'] = '另一个声压级'
    unrelated['display_payload']['y'] = [999, 999, 999]
    event['payload']['instance_results'].append(unrelated)
    deliver_completed_batch(w, event)
    cell = w.grid._cells[0, 0]
    assert cell._condition_states['warm'] == ('', '')
    assert {curve['wav_channel'] for curve in cell._condition_series['warm']} == {2}
    assert cell._condition_series['warm'][0]['y'][0] == 61


def test_recording_failure_cancellation_and_retry_update_only_current_region(plotted):
    w = plotted
    start(w); w.complete()
    deliver_completed_batch(w, result_event(w.analysis.submissions[0]['tasks'][0]))
    start(w); w.complete(outcome='failure')
    cell = w.grid._cells[0, 0]
    assert cell.empty_labels['cold'].text() == '采集失败'
    assert 'warm' in cell._condition_series
    start(w); w.complete(outcome='cancelled')
    assert cell.empty_labels['cold'].text() == '已取消'
    start(w)
    assert cell.empty_labels['cold'].text() == '采集中'
    assert 'warm' in cell._condition_series


def test_invalid_limit_keeps_measured_curve_and_reports_detail(plotted):
    w = plotted
    start(w); w.complete()
    event = result_event(w.analysis.submissions[0]['tasks'][0])
    event['payload']['instance_results'][0]['display_payload']['upper'] = [80]
    deliver_completed_batch(w, event)
    cell = w.grid._cells[0, 0]
    assert [(c['wav_channel'], c['role']) for c in cell._condition_series['warm']] == [(1, 'curve'), (2, 'curve'), (2, 'upper')]
    assert cell._condition_states['warm'][0] == '部分结果不可用'
    assert 'upper' in cell._condition_states['warm'][1]


def test_real_resident_workers_feed_four_station_plots_from_closed_wavs(plot_setup, qt_app):
    import soundfile as sf
    from base.analysis.service import AnalysisService
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    w = plot_setup
    service, events = AnalysisService(), []
    w.controller.bind_analysis_service(service, events.append)
    w.controller.activate_analysis()
    timer = QTimer()
    timer.timeout.connect(w.controller.poll_analysis_events)
    timer.start(20)

    def wait(predicate):
        end = monotonic() + 40
        while not predicate() and monotonic() < end:
            QTest.qWait(10)
        assert predicate(), service.station_states

    try:
        wait(lambda: all(state['state'] == 'ready' for state in service.station_states.values()))
        pids = {key: state['pid'] for key, state in service.station_states.items()}
        for _ in range(2):
            start(w)
            group = w.requests[-1].station_group
            Path(group.final_path).mkdir(parents=True, exist_ok=True)
            for index, station in enumerate(group.stations):
                path = Path(group.final_path) / station.filename
                time = np.arange(3 * 44100) / 44100
                tone = np.sin(2 * np.pi * 1000 * time) * .01 * (index + 1)
                sf.write(path, np.column_stack((tone, tone * 2)), 44100, subtype='PCM_24')
                assert append_wav_calibration_metadata(path, {'recorded_channels': [
                    {'wav_channel_index': i, 'physical_input_channel': i, 'calibrated': True,
                     'v2pa_factor': 2., 'standard_spl': 94.} for i in (0, 1)]})
            w.complete()
        wait(lambda: sum(e['event_type'] == 'task_result' for e in events) == 8)
        for column in range(4):
            for row in range(2):
                cell = w.grid._cells[row, column]
                assert set(cell._condition_series) == {'warm', 'cold'}
                assert all(state == ('', '') for state in cell._condition_states.values())
                for condition in ('warm', 'cold'):
                    assert {c['wav_channel'] for c in cell._condition_series[condition]} == {1, 2}
                    assert all(np.isfinite(c['y']).any() for c in cell._condition_series[condition])
        assert {key: state['pid'] for key, state in service.station_states.items()} == pids
        w.grid._cells[0, 0].open_detail()
        qt_app.processEvents()
        assert set(w.grid._cells[0, 0]._detail_dialog.cell._condition_series) == {'warm', 'cold'}
        w.grid._cells[0, 0].close_detail()
    finally:
        w.controller.begin_analysis_shutdown()
        wait(lambda: service.closed)
        timer.stop()
