"""Cross-surface regressions for curves, limits and calibrated matrices."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent

from base.analysis.contracts import AnalysisInstance
from base.analysis.executor import _judge
from base.analysis.results import calculation_payload
from base.analysis_plot_geometry import loudness_time_curve
from base.workbench_plot_config import parse_plot_templates
from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.workbench_plot_display import (
    ConditionSegmentDisplay, ConditionPlotDisplay, WorkbenchPlotCellDisplay,
)
from ui.sequence.workbench_plot_results import build_condition_result
from unit_test.ui.test_shared_plot_pipeline import station_layout


@pytest.fixture(autouse=True)
def app(qt_app):
    return qt_app


def close(widget):
    if hasattr(widget, 'dispose'):
        widget.dispose()
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def segment(kind, unit, x_unit='s'):
    return ConditionSegmentDisplay('c1', '正转', duration_seconds=1., channels=((1, 'CH1'),),
                                   x_unit=x_unit, y_unit=unit, analysis_type=kind)


def adapt(kind, unit, item, x_unit='s'):
    return build_condition_result(SimpleNamespace(analysis_item='item', analysis_type=kind),
                                   segment(kind, unit, x_unit), {'instance_results': [item]})


def loud_result(x, y, method='time_varying_iso532_1'):
    config = dict(method=method, advanced={'curve_y_unit': 'sone'}, limit_checked=True,
                  limit_metric='curve_y', limit_mode='manual', manual_input_mode='constant',
                  constant_upper_enabled=True, constant_upper_value=2., constant_lower_enabled=False)
    px, py = loudness_time_curve(x, y, method=method, start_s=0., end_s=1.)
    calculation = dict(metrics={'unit': 'sone'}, curve={'x': x, 'y': y},
                       loudness_curves={'sone': y},
                       plot_curve={'x': px, 'y': py, 'upper': [], 'lower': []},
                       plot_loudness_curves={'sone': py}, plot={})
    instance = AnalysisInstance('item', 'LOUD', 0, config)
    metrics, payload = calculation_payload(instance, calculation, 0.)
    item = dict(config_key='item', analysis_type='LOUD', channel_index=0,
                execution_status='分析完成', metrics=metrics, display_payload=payload)
    _judge(instance, calculation, item)
    return config, item


def test_fft_delta_should_not_overlay_absolute_baseline():
    instance = AnalysisInstance('item', 'FFT', 0, {'baseline_display_mode': 'delta'})
    calculation = dict(metrics={'display_mode': 'delta', 'weighting': 'Z'},
                       curve={'x': np.array([100., 1000., 10000.]), 'y': np.array([0., 1., 0.])},
                       plot={'baseline': np.array([40., 60., 40.])})
    metrics, payload = calculation_payload(instance, calculation, 0.)
    item = dict(config_key='item', analysis_type='FFT', channel_index=0,
                execution_status='分析完成', judgement_status='not_enabled', metrics=metrics,
                display_payload=payload)
    result = adapt('FFT', 'dB', item, 'Hz')
    assert not result.status
    assert metrics['display_mode'] == 'delta'
    assert [s['role'] for s in result.series] == ['curve'], 'delta mode unexpectedly includes absolute baseline'




def test_loudness_highlight_should_lie_on_displayed_curve():
    x, y = np.array([.1, .3, .5]), np.array([9., 1., 1.])
    _, item = loud_result(x, y, 'per_segment')
    result = adapt('LOUD', 'sone', item)
    series = {s['role']: s for s in result.series}
    main, red = series['curve'], series['exceedance']
    valid = np.isfinite(red['y'])
    displayed_y = np.interp(red['x'][valid], main['x'], main['y'])
    np.testing.assert_allclose(red['y'][valid], displayed_y,
                                err_msg='raw violation is detached from transformed main curve')




def test_fixed_viewport_should_clip_line_segments_not_remove_them(qt_app):
    layout = station_layout()
    layout['plot_templates']['spl']['x_axis'].update(auto_range=False, range_min=.4, range_max=.6)
    settings = parse_plot_templates(layout).cells[0][0].display_settings
    cell = ConditionPlotCell(WorkbenchPlotCellDisplay(title='SPL', condition_plot=ConditionPlotDisplay(
        settings, (segment('SPL', 'dB(Z)'),))))
    try:
        cell.set_condition_curve_data('c1', [dict(name='upper', wav_channel=1, role='upper',
                                                 limit_scope='scalar', x=[0., 1.], y=[40., 40.])])
        line, = cell.plot_widget.listDataItems()
        np.testing.assert_array_equal(line.yData, [40., 40.])
        np.testing.assert_array_equal(cell._condition_series['c1'][0]['x'], [0., 1.])
        cell.open_detail()
        qt_app.processEvents()
        detail_line, = cell._detail_dialog.cell.plot_widget.listDataItems()
        np.testing.assert_array_equal(detail_line.yData, line.yData)
        cell.set_condition_state('c1', '', clear=True)
        assert not cell.plot_widget.listDataItems()
        assert not cell._detail_dialog.cell.plot_widget.listDataItems()
    finally:
        close(cell)






@pytest.mark.parametrize('judgement', ['OK', 'NG'])
def test_control_overall_limits_do_not_rejudge_instantaneous_peaks(judgement):
    item = dict(config_key='item', analysis_type='SPL', channel_index=0,
                execution_status='分析完成', judgement_status='completed', judgement=judgement,
                metrics={'unit': 'dB(Z)', 'overall_upper_limit': 40., 'overall_lower_limit': 0.},
                display_payload={'kind': 'curve', 'x': [0., .5, 1.], 'y': [20., 80., 20.],
                                 'upper': [], 'lower': [], 'x_unit': 's', 'y_unit': 'dB'},
                violations=[{'x_start': None, 'x_end': None}])
    result = adapt('SPL', 'dB(Z)', item)
    assert [s['role'] for s in result.series] == ['curve', 'upper', 'lower']
    np.testing.assert_equal(result.series[1]['y'], [40., 40., 40.])
    np.testing.assert_equal(result.series[2]['y'], [0., 0., 0.])


def test_control_fft_overlay_keeps_baseline():
    item = dict(config_key='item', analysis_type='FFT', channel_index=0,
                execution_status='分析完成', judgement_status='not_enabled',
                metrics={'display_mode': 'overlay'},
                display_payload={'kind': 'curve', 'x': [100., 1000.], 'y': [40., 60.],
                                 'baseline': [30., 50.], 'x_unit': 'Hz', 'y_unit': 'dB'})
    result = adapt('FFT', 'dB(Z)', item, 'Hz')
    assert [s['role'] for s in result.series] == ['curve', 'baseline']


def test_loudness_unit_mismatch_does_not_draw_phon_values_on_sone_axis():
    _, item = loud_result(np.array([0., .5, 1.]), np.array([1., 3., 1.]))
    item['metrics']['limit_unit'] = 'phon'
    payload = item['display_payload']
    payload.update(limit_unit='phon', upper=[], lower=[])
    payload['plot_curve'].update(upper=[], lower=[])
    before = deepcopy(item)
    result = adapt('LOUD', 'sone', item)
    assert [s['role'] for s in result.series] == ['curve']
    assert result.summaries[0]['judgement'] == 'NG'
    np.testing.assert_equal(item, before)


@pytest.mark.parametrize('method', ['per_segment', 'time_varying_iso532_1'])
def test_loudness_display_keeps_raw_judgement_and_input_samples(method):
    x, y = np.array([.1, .3, .5]), np.array([9., 1., 1.])
    before_x, before_y = x.copy(), y.copy()
    config, item = loud_result(x, y, method)
    before = deepcopy(item)
    result = adapt('LOUD', 'sone', item)
    assert item['judgement'] == 'NG'
    assert [(v['x_start'], v['x_end']) for v in item['violations']] == [(.1, .1)]
    assert result.summaries[0]['judgement'] == 'NG'
    np.testing.assert_equal(item, before)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(item['display_payload']['x'], x)
    np.testing.assert_array_equal(item['display_payload']['y'], y)
