"""Real compute-only results become detached, complete record-view data."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.record_analysis_request import RecordAnalysisItem
from ui.sequence.record_analysis_results import build_record_analysis_results


def test_import_and_numeric_mapping_do_not_load_qt_in_fresh_process():
    script = '''
import sys
from ui.sequence.record_analysis_results import build_record_analysis_results

def assert_no_qt(stage):
    loaded = [name for name in sys.modules
              if name == 'PyQt5' or name.startswith('PyQt5.')
              or name == 'pyqtgraph' or name.startswith('pyqtgraph.')]
    assert not loaded, (stage, loaded)

assert_no_qt('import')
from base.record_analysis_request import RecordAnalysisItem
item = RecordAnalysisItem('pressure', 'SPL', (0,), {})
result = {
    'execution_status': '分析完成', 'error': None,
    'instance_results': [{
        'config_key': 'pressure', 'analysis_type': 'SPL', 'channel_index': 0,
        'execution_status': '分析完成', 'judgement_status': 'completed', 'judgement': 'NG',
        'metrics': {'unit': 'dBA', 'overall_spl': 80.0}, 'error': None,
        'violations': [{'x_start': 1.0, 'x_end': 1.0, 'x_unit': 's'}],
        'display_payload': {'kind': 'curve', 'x': [0.0, 1.0], 'y': [70.0, 80.0],
                            'upper': [75.0, 75.0], 'lower': [],
                            'x_unit': 's', 'y_unit': 'dBA'},
    }],
}
channel, = build_record_analysis_results((item,), result)['pressure'].channels
assert channel.status == '分析完成', channel.diagnostic
assert [value['role'] for value in channel.panels[0].series] == ['curve', 'upper', 'exceedance']
assert_no_qt('numeric mapping')
'''
    completed = subprocess.run([sys.executable, '-c', script],
                               cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope='module')
def real_results(tmp_path_factory):
    directory = tmp_path_factory.mktemp('record-view-results')
    rate = 44100
    time = np.arange(int(0.4 * rate)) / rate
    tone = 0.02 * np.sin(2 * np.pi * 1000 * time)
    path = directory / 'two-channels.wav'
    sf.write(path, np.column_stack((tone, 2 * tone)), rate, subtype='FLOAT')
    task = AnalysisTask(
        'view', 'station', '工位', 'condition', '工况', str(path), None,
        (AnalysisChannel(0, '近端', 'sound_pressure', v2pa_factor=2.0),
         AnalysisChannel(1, '远端', 'sound_pressure', v2pa_factor=3.0)),
        (), {'mode': 'none'}, persist_artifacts=False,
    )
    configs = {}
    for kind in ('SPL', 'FFT', 'FBA', 'Spec', 'LOUD'):
        config = {'weighting': 'A'}
        if kind != 'Spec':
            config.update(limit_checked=True, limit_metric='curve_y', limit_mode='manual',
                          manual_input_mode='constant', constant_upper_enabled=True,
                          constant_upper_value=0.001, constant_lower_enabled=False)
        if kind == 'LOUD':
            config.update(method='time_varying_iso532_1', advanced={
                'curve_y_unit': 'phon', 'show_specific_loudness_heatmap': True},
                display={'curves': ['loudness_time', 'specific_loudness_profile'],
                         'heatmaps': ['specific_loudness'],
                         'summary_metrics': ['steady_state_average_loudness',
                                             'max_transient_loudness']})
        configs[kind] = config
    instances = tuple(AnalysisInstance(kind, kind, index, deepcopy(config))
                      for kind, config in configs.items() for index in (0, 1))
    task = replace(task, instances=instances)
    result = execute_task(task)
    assert result['execution_status'] == '分析完成', result
    items = tuple(RecordAnalysisItem(kind, kind, (0, 1), config)
                  for kind, config in configs.items())
    return task, items, result


@pytest.mark.parametrize('kind', ['SPL', 'FFT', 'FBA', 'Spec', 'LOUD'])
def test_real_full_coordinates_units_metrics_and_judgement(real_results, kind):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result, channel_names={0: '近端', 1: '远端'})
    assert list(mapped) == [item.config_key for item in items]
    view = mapped[kind]
    assert view.config_key == kind and view.analysis_type == kind
    assert view.status == '分析完成'
    assert [channel.wav_channel for channel in view.channels] == [1, 2]
    assert [channel.name for channel in view.channels] == ['近端', '远端']
    for channel in view.channels:
        source = next(value for value in result['instance_results']
                      if value['config_key'] == kind and value['channel_index'] == channel.channel_index)
        payload = source['display_payload']
        assert channel.execution_status == source['execution_status']
        assert channel.judgement_status == source['judgement_status']
        assert channel.judgement == source['judgement']
        assert channel.metrics == source['metrics']
        assert channel.violations == source['violations']
        assert channel.status == '分析完成' and not channel.diagnostic
        if kind == 'Spec':
            panel, = channel.panels
            assert (panel.kind, panel.x_unit, panel.y_unit) == ('heatmap', 's', 'Hz')
            assert panel.value_unit == source['metrics']['unit']
            np.testing.assert_array_equal(panel.heatmap['x'], payload['time_s'])
            np.testing.assert_array_equal(panel.heatmap['y'], payload['frequency_hz'])
            np.testing.assert_array_equal(panel.heatmap['z'], payload['values_db'])
            continue
        panel = channel.panels[0]
        assert panel.x_unit == payload['x_unit'] and panel.y_unit == payload['y_unit']
        curve, = [series for series in panel.series if series['role'] == 'curve']
        assert curve['name'] == f'CH{channel.wav_channel} · {channel.name}'
        np.testing.assert_array_equal(curve['x'], payload['x'])
        np.testing.assert_allclose(curve['y'], payload['y'])
        for side in ('upper', 'lower'):
            # Disabled bounds serialize as null arrays, not displayed limits.
            if np.isfinite(np.asarray(payload[side], dtype=float)).any():
                limit, = [series for series in panel.series if series['role'] == side]
                np.testing.assert_allclose(limit['y'], payload[side])
        assert any(series['role'] == 'exceedance' for series in panel.series)
        if kind == 'FBA':
            assert panel.kind == 'bands'
            assert panel.band_labels == tuple(payload['band_labels'])
        if kind == 'SPL':
            assert channel.metrics['overall_spl'] == source['metrics']['overall_spl']


def test_loudness_preserves_every_selected_composite_panel_and_summary(real_results):
    _, items, result = real_results
    view = build_record_analysis_results(items, result)['LOUD']
    for channel in view.channels:
        source = result['instance_results'][8 + channel.channel_index]['display_payload']['loudness_display']
        assert [panel.key for panel in channel.panels] == [
            curve['key'] for curve in source['curves']] + [heatmap['key'] for heatmap in source['heatmaps']]
        assert channel.summary_cards == tuple(source['summary_cards'])
        time, profile, heatmap = channel.panels
        assert (time.x_unit, time.y_unit) == ('s', 'phon')
        assert (profile.x_unit, profile.y_unit) == ('Bark', 'sone/Bark')
        assert (heatmap.x_unit, heatmap.y_unit, heatmap.value_unit) == ('s', 'Bark', 'sone/Bark')
        assert profile.x_label == source['curves'][1]['x_label']
        assert profile.y_label == source['curves'][1]['y_label']
        np.testing.assert_array_equal(profile.series[0]['x'], source['curves'][1]['x'])
        np.testing.assert_array_equal(profile.series[0]['y'], source['curves'][1]['y'])
        np.testing.assert_array_equal(heatmap.heatmap['z'], source['heatmaps'][0]['z'])


@pytest.mark.parametrize('kind', ['SPL', 'FFT', 'FBA', 'Spec', 'LOUD'])
def test_real_channel_failure_preserves_other_channels_and_items(real_results, kind):
    task, items, _ = real_results
    task = replace(task, channels=(replace(task.channels[0], v2pa_factor=None), task.channels[1]))
    result = execute_task(task)
    assert result['execution_status'] == '结果不完整'
    view = build_record_analysis_results(items, result)[kind]
    failed, good = view.channels
    assert failed.status == '分析失败' and not failed.panels
    assert failed.judgement is None and failed.judgement_status == 'not_executed'
    assert kind in failed.diagnostic and 'CH1' in failed.diagnostic
    assert result['instance_results'][2 * list(v.config_key for v in items).index(kind)]['error']['message'] in failed.diagnostic
    assert good.status == '分析完成' and good.panels
    assert view.status == '部分结果不可用'


@pytest.mark.parametrize('change,expected', [
    ('duplicate', '重复'), ('missing', '未返回'), ('wrong_type', '未返回'),
    ('wrong_channel_type', '未返回'), ('wrong_key', '未返回'),
])
def test_matching_is_exact_and_ambiguous_channel_is_never_selected(real_results, change, expected):
    _, items, source = real_results
    result = deepcopy(source)
    instance = result['instance_results'][0]
    if change == 'duplicate':
        result['instance_results'].append(deepcopy(instance))
    elif change == 'missing':
        result['instance_results'].pop(0)
    elif change == 'wrong_type':
        instance['analysis_type'] = 'FFT'
    elif change == 'wrong_key':
        instance['config_key'] = 'SPL other'
    else:
        instance['channel_index'] = False
    mapped = build_record_analysis_results(items, result)
    failed, good = mapped['SPL'].channels
    assert expected in failed.diagnostic and not failed.panels
    assert failed.judgement is None and failed.judgement_status is None
    assert good.status == '分析完成'
    assert mapped['FFT'].status == '分析完成'


def test_task_failure_overrides_stale_successes_for_every_item(real_results):
    _, items, source = real_results
    result = dict(source, execution_status='分析失败', error={'message': '无法读取所选 WAV'})
    for view in build_record_analysis_results(items, result).values():
        assert view.status == '结果不可用'
        for channel in view.channels:
            assert '无法读取所选 WAV' in channel.diagnostic
            assert not channel.panels and channel.judgement is None


def test_unsupported_types_retain_backend_diagnostic(real_results):
    task, _, _ = real_results
    task = replace(task, instances=tuple(AnalysisInstance(kind, kind, index, {})
                                        for kind in ('AI', 'VIBRATION_RMS') for index in (0, 1)))
    result = execute_task(task)
    items = tuple(RecordAnalysisItem(kind, kind, (0, 1), {}) for kind in ('AI', 'VIBRATION_RMS'))
    views = build_record_analysis_results(items, result)
    for view in views.values():
        for channel in view.channels:
            assert channel.status == '分析失败' and channel.judgement is None
            source = next(value for value in result['instance_results']
                          if value['config_key'] == view.config_key
                          and value['channel_index'] == channel.channel_index)
            assert source['error']['message'] in channel.diagnostic and not channel.panels


@pytest.mark.parametrize('kind,field,value,diagnostic', [
    ('SPL', 'x', [0.0], 'same length'),
    ('FFT', 'y_unit', None, 'y_unit'),
    ('FBA', 'band_labels', ['only one'], 'band_labels'),
    ('FBA', 'band_labels', None, 'band_labels'),
    ('FBA', 'band_labels', [None] * 10, 'band_labels'),
    ('Spec', 'matrix_axes', ['time', 'frequency'], '轴顺序'),
    ('Spec', 'values_db', [[1.0]], '形状'),
    ('LOUD', 'loudness_display', {'curves': {}, 'heatmaps': [], 'summary_cards': []}, 'curves'),
])
def test_corrupt_payload_is_local_to_its_item_channel(real_results, kind, field, value, diagnostic):
    _, items, source = real_results
    result = deepcopy(source)
    instance = next(entry for entry in result['instance_results']
                    if entry['config_key'] == kind and entry['channel_index'] == 0)
    if kind == 'FBA' and field == 'band_labels' and isinstance(value, list) and value[0] is None:
        value = [None] * len(instance['display_payload']['x'])
    instance['display_payload'][field] = value
    views = build_record_analysis_results(items, result)
    bad, good = views[kind].channels
    assert bad.status == '内容不可用' and not bad.panels
    assert kind in bad.diagnostic and 'CH1' in bad.diagnostic and diagnostic in bad.diagnostic
    assert good.status == '分析完成' and good.panels
    assert all(view.status == '分析完成' for key, view in views.items() if key != kind)


@pytest.mark.parametrize('curves,summary_metrics,expected_panels', [
    ([], [], []),
    (['specific_loudness_profile'], [], ['specific_loudness_profile']),
    ([], ['max_transient_loudness'], []),
])
def test_loudness_honors_actual_selection_without_inventing_time_curve(
        real_results, curves, summary_metrics, expected_panels):
    task, _, _ = real_results
    original = next(value for value in task.instances if value.analysis_type == 'LOUD')
    config = deepcopy(original.parameters)
    config['display'] = dict(curves=curves, summary_metrics=summary_metrics, heatmaps=[])
    task = replace(task, instances=(replace(original, parameters=config),))
    result = execute_task(task)
    assert result['execution_status'] == '分析完成'
    item = RecordAnalysisItem('LOUD', 'LOUD', (0,), config)
    view = build_record_analysis_results((item,), result)['LOUD']
    channel, = view.channels
    assert channel.status == '分析完成'
    assert [panel.key for panel in channel.panels] == expected_panels
    assert len(channel.summary_cards) == len(summary_metrics)


def test_mapping_copies_all_arrays_metrics_cards_and_does_not_rejudge(real_results):
    _, items, source = real_results
    result = deepcopy(source)
    parameters_before = [deepcopy(item.parameters) for item in items]
    # Keep an intentionally authoritative backend outcome, irrespective of points.
    result['instance_results'][0].update(judgement='OK', violations=[])
    before = deepcopy(result)
    views = build_record_analysis_results(items, result)
    assert views['SPL'].channels[0].judgement == 'OK'
    assert not any(series['role'] == 'exceedance' for series in views['SPL'].channels[0].panels[0].series)
    assert result == before
    assert [item.parameters for item in items] == parameters_before
    views['SPL'].channels[0].panels[0].series[0]['y'][0] = 999
    views['SPL'].channels[0].metrics['overall_spl'] = 999
    views['FBA'].channels[0].panels[0].series[0]['x'][0] = 999
    views['Spec'].channels[0].panels[0].heatmap['z'][0, 0] = 999
    views['LOUD'].channels[0].panels[2].heatmap['z'][0, 0] = 999
    views['LOUD'].channels[0].summary_cards[0]['value'] = 999
    views['LOUD'].channels[0].violations.clear()
    assert result == before
    second = build_record_analysis_results(items, result)
    assert second['SPL'].channels[0].metrics['overall_spl'] != 999
    assert views['SPL'].channels[1].panels[0].series[0]['y'][0] != 999


def test_stationary_loudness_heatmap_preserves_its_single_real_time_frame(real_results):
    task, _, _ = real_results
    original = next(value for value in task.instances if value.analysis_type == 'LOUD')
    config = deepcopy(original.parameters)
    config['method'] = 'per_segment'
    task = replace(task, instances=(replace(original, parameters=config),))
    result = execute_task(task)
    assert result['execution_status'] == '分析完成', result
    item = RecordAnalysisItem('LOUD', 'LOUD', (0,), config)
    channel, = build_record_analysis_results((item,), result)['LOUD'].channels
    assert channel.status == '分析完成', channel.diagnostic
    source = result['instance_results'][0]['display_payload']['loudness_display']['heatmaps'][0]
    panel = next(value for value in channel.panels if value.key == source['key'])
    np.testing.assert_array_equal(panel.heatmap['x'], source['x'])
    np.testing.assert_array_equal(panel.heatmap['y'], source['y'])
    np.testing.assert_array_equal(panel.heatmap['z'].ravel(), np.asarray(source['z']).ravel())
    assert panel.heatmap['z'].shape == (len(source['y']), len(source['x']))


def test_failed_judgement_keeps_real_curves_and_never_displays_ok(real_results):
    task, _, _ = real_results
    originals = tuple(value for value in task.instances if value.analysis_type == 'LOUD')
    config = deepcopy(originals[0].parameters)
    config.update(limit_metric='max_transient', scalar_upper_enabled=True,
                  scalar_upper_value=1, scalar_lower_enabled=True, scalar_lower_value=100)
    task = replace(task, instances=(replace(originals[0], parameters=config), originals[1]))
    result = execute_task(task)
    assert result['execution_status'] == '结果不完整'
    item = RecordAnalysisItem('LOUD', 'LOUD', (0, 1), config)
    failed, good = build_record_analysis_results((item,), result)['LOUD'].channels
    assert failed.judgement_status == 'failed' and failed.judgement is None
    assert failed.status == '部分结果不可用' and failed.panels and failed.metrics
    assert result['instance_results'][0]['error']['message'] in failed.diagnostic
    assert failed.diagnostic.count(result['instance_results'][0]['error']['message']) == 1
    assert good.status == '分析完成' and good.panels


def test_overall_spl_limits_are_scalar_reference_lines_without_point_rejudgement(real_results):
    task, _, _ = real_results
    original = task.instances[0]
    config = dict(weighting='A', limit_checked=True, limit_metric='overall_spl',
                  scalar_upper_enabled=True, scalar_upper_value=0.001, scalar_lower_enabled=False)
    result = execute_task(replace(task, instances=(replace(original, parameters=config),)))
    before = deepcopy(result)
    source = result['instance_results'][0]
    payload = source['display_payload']
    item = RecordAnalysisItem('SPL', 'SPL', (0,), config)
    channel, = build_record_analysis_results((item,), result)['SPL'].channels
    assert channel.status == '分析完成' and not channel.diagnostic
    assert channel.judgement == source['judgement'] == 'NG'
    assert channel.judgement_status == source['judgement_status'] == 'completed'
    assert channel.metrics == source['metrics']
    assert channel.metrics['overall_upper_limit'] == 0.001
    assert channel.violations == source['violations']
    assert source['violations']
    assert all(value['x_start'] is None and value['x_end'] is None
               for value in source['violations'])
    assert not any(np.isfinite(np.asarray(payload[side], dtype=float)).any()
                   for side in ('upper', 'lower'))
    # Scalar NG and points above the reference must not invent point highlights.
    assert [series['role'] for series in channel.panels[0].series] == ['curve', 'upper']
    curve, reference = channel.panels[0].series
    np.testing.assert_array_equal(curve['x'], payload['x'])
    np.testing.assert_array_equal(curve['y'], payload['y'])
    assert np.any(curve['y'] > channel.metrics['overall_upper_limit'])
    assert reference['limit_scope'] == 'scalar'
    np.testing.assert_array_equal(reference['x'], payload['x'])
    np.testing.assert_array_equal(reference['y'], np.full(
        len(payload['x']), channel.metrics['overall_upper_limit']))
    assert result == before


def test_same_type_different_keys_keep_navigation_order_and_independent_data(real_results):
    _, items, source = real_results
    result = deepcopy(source)
    duplicate_config = [dict(deepcopy(value), config_key='second SPL')
                        for value in result['instance_results'] if value['config_key'] == 'SPL']
    result['instance_results'].extend(duplicate_config)
    second = RecordAnalysisItem('second SPL', 'SPL', (1, 0), items[0].parameters)
    views = build_record_analysis_results((second, items[0]), result)
    assert list(views) == ['second SPL', 'SPL']
    assert [value.channel_index for value in views['second SPL'].channels] == [1, 0]
    views['second SPL'].channels[1].panels[0].series[0]['y'][0] = 999
    assert views['SPL'].channels[0].panels[0].series[0]['y'][0] != 999
