"""Analysis-item thresholds reach the workbench without a second judgement."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_plot_config import parse_plot_templates
from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.workbench_plot_presenter import WorkbenchPlotPresenter
from ui.sequence.workbench_plot_results import build_condition_result
from ui.sequence.workbench_ui_projection import WorkbenchPlotBinding, build_ui_projection
from unit_test.base.workbench_plot_fixtures import template_layout
from unit_test.ui.test_condition_plot_layout import configuration, grid, show


def analysis_configuration(configuration, kind, parameters):
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type=kind, title=kind)
    template['x_axis'] = dict(label='频率' if kind == 'FFT' else '时间',
                              scale='linear', auto_range=True)
    layout['default_condition_display']['x_axis_mode'] = 'condition_local'
    queues = {name: AnalysisTestQueue('wav', {
        'display_sequence': ['声压级'],
        '声压级': dict(parameters, type=kind, analysis_channels=[0]),
    }) for name in configuration.analysis.test_queues}
    template['channel_display'] = {'show_all_analyzed_channels': True}
    return replace(configuration,
                   analysis=replace(configuration.analysis, test_queues=queues),
                   ui=replace(configuration.ui, layout=parse_plot_templates(layout)))


@pytest.mark.parametrize(('kind', 'unit'), [('FFT', 'dB(Z)'), ('LOUD', 'sone'), ('LOUD', 'phon')])
@pytest.mark.parametrize('limit_mode', ['disabled', 'manual', 'csv'])
@pytest.mark.parametrize('visible', [False, True])
def test_real_analysis_thresholds_reach_main_and_detail(
        configuration, grid, qt_app, tmp_path, kind, unit, limit_mode, visible):
    upper = .01 if unit == 'sone' else 40.
    lower = .001 if unit == 'sone' else -100.
    end = 20000. if kind == 'FFT' else .5
    parameters = dict(
        display={'show_threshold_lines': visible},
        weighting='Z', n_fft=2048, method='per_segment',
        advanced={'curve_y_unit': unit} if kind == 'LOUD' else {},
        limit_checked=limit_mode != 'disabled', limit_metric='curve_y',
        curve_limit_unit=unit, limit_mode='csv' if limit_mode == 'csv' else 'manual',
        manual_input_mode='constant', constant_upper_enabled=True,
        constant_upper_value=upper, constant_lower_enabled=True, constant_lower_value=lower,
        limit_data=[[0, end], [upper, upper], [lower, lower]],
    )
    before = deepcopy(parameters)
    config = analysis_configuration(configuration, kind, parameters)
    projection = show(grid, config, qt_app)
    cell = grid._cells[0, 0]
    binding = projection.bindings[0][0][0]
    assert binding is not None
    assert next(iter(cell.conditions.values())).y_unit == unit
    rate = 48000
    time = np.arange(rate // 2) / rate
    wav = tmp_path / 'signal.wav'
    sf.write(wav, .1 * np.sin(2 * np.pi * 1000 * time), rate, subtype='PCM_24')
    task = AnalysisTask('threshold-test', binding.station_id, '工位1', binding.condition_id,
                        '工况1', str(wav), str(tmp_path),
                        (AnalysisChannel(0, 'CH1', 'sound_pressure', v2pa_factor=1.),),
                        (AnalysisInstance(binding.analysis_item, kind, 0, parameters),), {'mode': 'none'})
    result = execute_task(task)
    instance = result['instance_results'][0]
    assert instance['execution_status'] == '分析完成', instance['error']
    assert instance['judgement_status'] == ('not_enabled' if limit_mode == 'disabled' else 'completed')
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    presenter.analysis_event(dict(event_type='task_result', is_current=True, payload=result,
                                  task_id=task.task_id, station_id=task.station_id,
                                  condition_id=task.condition_id))
    assert cell._condition_states[binding.condition_id] == ('', '')
    series = cell._condition_series[binding.condition_id]
    if limit_mode == 'disabled':
        assert [s['role'] for s in series] == ['curve']
    else:
        assert instance['judgement'] == 'NG'
        curves = {s['role']: s for s in series}
        assert set(curves) == {'curve', 'upper', 'lower', 'exceedance'}
        payload = instance['display_payload']
        presentation = payload.get('plot_curve', payload)
        for side in ('upper', 'lower'):
            np.testing.assert_equal(curves[side]['y'], presentation[side])
        expected = ((np.asarray(payload['y']) > np.asarray(payload['upper']))
                    | (np.asarray(payload['y']) < np.asarray(payload['lower'])))
        np.testing.assert_array_equal(np.isfinite(curves['exceedance']['y']), expected)
    cell.open_detail()
    qt_app.processEvents()
    detail = cell._detail_dialog.cell
    expected_count = len(series) - (2 if not visible and limit_mode != 'disabled' else 0)
    assert len(cell.plot_widget.listDataItems()) == expected_count
    assert len(detail.plot_widget.listDataItems()) == expected_count
    if limit_mode != 'disabled':
        for view in (cell, detail):
            highlights = [item for item in view.plot_widget.listDataItems()
                          if item.opts['symbol'] == 'o']
            assert len(highlights) == 1
            assert np.isfinite(highlights[0].yData).any()
    for original, enlarged in zip(cell.plot_widget.listDataItems(), detail.plot_widget.listDataItems()):
        assert original.opts['pen'].color() == enlarged.opts['pen'].color()
    for original, enlarged in zip(series, detail._condition_series[binding.condition_id]):
        np.testing.assert_equal(original['y'], enlarged['y'])
    cell.set_condition_state(binding.condition_id, '', clear=True)
    assert not cell.plot_widget.listDataItems()
    assert not detail.plot_widget.listDataItems()
    cell.close_detail()
    assert parameters == before


def test_loudness_conditions_with_different_display_units_are_not_combined(configuration):
    config = analysis_configuration(configuration, 'LOUD', {'advanced': {'curve_y_unit': 'sone'}})
    queues = dict(config.analysis.test_queues)
    second = config.analysis.context.conditions[1].test_queue
    queues[second] = AnalysisTestQueue('wav', {
        'display_sequence': ['声压级'],
        '声压级': {'type': 'LOUD', 'analysis_channels': [0], 'advanced': {'curve_y_unit': 'phon'}},
    })
    config = replace(config, analysis=replace(config.analysis, test_queues=queues))
    projection = build_ui_projection(config.ui, config.analysis)
    assert '单位不一致' in projection.cells[0][0].warning_detail
    assert all(binding is None for binding in projection.bindings[0][0])


@pytest.mark.parametrize(('kind', 'unit'), [
    ('FFT', 'dB(Z)'), ('LOUD', 'sone'), ('ROUGH', 'asper'), ('SHRP', 'acum'),
])
def test_shared_threshold_renderer_uses_payload_and_backend_intervals(configuration, qt_app, kind, unit):
    # ROUGH/SHRP here test the shared view contract, not an implemented executor.
    display = build_ui_projection(configuration.ui, configuration.analysis).cells[0][0]
    segment = replace(display.condition_plot.conditions[0], channels=((1, 'CH1'),),
                      x_unit='Hz' if kind == 'FFT' else 's', y_unit=unit)
    display = replace(display, title=kind, condition_plot=replace(
        display.condition_plot, conditions=(segment,), summary_rows=0))
    binding = WorkbenchPlotBinding('s1', segment.condition_id, 'metric', kind, (1,))
    payload = dict(kind='curve', x=[0, 1, 2], y=[70, 50, 10],
                   upper=[40, 40, 40], lower=[20, 20, 20], x_unit=segment.x_unit, y_unit=unit)
    result = {'instance_results': [dict(
        config_key='metric', analysis_type=kind, channel_index=0,
        execution_status='分析完成', judgement_status='completed', judgement='NG',
        display_payload=payload,
        violations=[dict(x_start=1, x_end=1, x_unit=segment.x_unit),
                    dict(x_start=2, x_end=2, x_unit=segment.x_unit)],
    )]}
    update = build_condition_result(binding, segment, result)
    assert not update.status
    assert [s['role'] for s in update.series] == ['curve', 'upper', 'lower', 'exceedance']
    # Only the backend's intervals are colored; the view must not judge x=0 itself.
    np.testing.assert_equal(update.series[-1]['y'], [np.nan, 50, 10])
    cell = ConditionPlotCell(display)
    try:
        cell.resize(700, 420)
        cell.show()
        cell.set_condition_curve_data(segment.condition_id, update.series, summaries=update.summaries)
        qt_app.processEvents()
        lines = cell.plot_widget.listDataItems()
        assert len(lines) == 4
        assert cell.plot_widget.getPlotItem().getAxis('left').labelText == unit
        assert lines[-1].opts['pen'].color().name().lower() == (
            cell._settings['plot_style']['exceedance_highlight']['color'].lower())
        np.testing.assert_equal(lines[-1].yData, [np.nan, 50, 10])
    finally:
        cell.dispose()
        cell.close()
        cell.deleteLater()
