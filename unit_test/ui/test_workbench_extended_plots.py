"""New workbench rendering; fixtures never modify saved user configurations."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QImage, QPainter
import pyqtgraph as pg

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_plot_config import parse_plot_templates
from ui.sequence.condition_heatmap_cell import ConditionHeatmapCell
from ui.sequence.spectrogram_data import validate_spectrogram
from ui.sequence.spectrogram_image import SpectrogramImage
from ui.sequence.workbench_plot_presenter import WorkbenchPlotPresenter
from ui.sequence.workbench_plot_results import build_condition_result
from ui.sequence.workbench_ui_projection import build_ui_projection
from unit_test.base.workbench_plot_fixtures import template_layout
from unit_test.ui.test_condition_plot_layout import configuration, grid, show


@pytest.mark.parametrize('kind', ['SPL', 'Spec', 'VIBRATION_RMS'])
def test_small_window_scrolls_without_squashing_data_area(grid, configuration, qt_app, kind):
    config = configuration if kind == 'SPL' else extended_configuration(configuration, kind)
    projection = build_ui_projection(config.ui, config.analysis)
    grid.show_workbench_layout(projection.column_names, projection.cells * 3)

    def settle():
        for _ in range(8):
            qt_app.processEvents()

    for _ in range(2):
        grid.resize(660, 500)
        settle()
        assert grid._scroll.verticalScrollBar().maximum() > 0
        assert grid._scroll.horizontalScrollBar().maximum() > 0
        for cell in grid._cells.values():
            assert cell.plot_widget.getViewBox().height() >= 120
        grid._scroll.verticalScrollBar().setValue(grid._scroll.verticalScrollBar().maximum())
        grid.resize(1600, 1100)
        settle()
        assert grid._scroll.verticalScrollBar().maximum() == 0
        assert grid._scroll.horizontalScrollBar().maximum() == 0
        assert grid._cells[0, 0].plot_widget.getViewBox().height() > 120

    cell = grid._cells[0, 0]
    cell.open_detail()
    dialog = cell._detail_dialog
    dialog.resize(320, 180)
    settle()
    assert dialog.cell.plot_widget.getViewBox().height() >= 120
    cell.close_detail()


def extended_configuration(configuration, kind='Spec', *, scale='linear', auto_colors=False):
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type=kind, title='时频图' if kind == 'Spec' else '振动 RMS',
                    chart_type='heatmap' if kind == 'Spec' else 'line')
    layout['default_condition_display'].update(x_axis_mode='condition_local', condition_gap_px=16)
    if kind == 'Spec':
        template.update(channel_display={'show_all_analyzed_channels': False, 'display_wav_channels': [1]},
                        spectrogram={'color_map': 'viridis', 'show_color_bar': True,
                                     'color_range': {'mode': 'fixed', 'min_db': 0, 'max_db': 100}},
                        y_axis={'label': '频率', 'scale': scale, 'auto_range': False,
                                'range_min': 20 if scale == 'log' else 0, 'range_max': 20000})
        if auto_colors:
            template['spectrogram']['color_range'] = {'mode': 'auto'}
    else:
        template['value_unit'] = 'm/s²'
        template['y_axis'] = dict(label='RMS', scale='linear', auto_range=True,
                                  empty_data_range_min=0, empty_data_range_max=1)
    queues = {name: AnalysisTestQueue('wav', {'display_sequence': ['声压级'],
              '声压级': {'type': kind, 'analysis_channels': [0, 1]}})
              for name in configuration.analysis.test_queues}
    context = configuration.analysis.context
    if kind == 'VIBRATION_RMS':
        context = replace(context, stations=tuple(replace(s, channels=tuple(
            replace(ch, signal_type='acceleration') for ch in s.channels)) for s in context.stations))
    return replace(configuration, analysis=replace(configuration.analysis, context=context, test_queues=queues),
                   ui=replace(configuration.ui, layout=parse_plot_templates(layout)))


def instance(binding, payload, *, judgement=None, violations=None, metrics=None):
    return dict(config_key=binding.analysis_item, analysis_type=binding.analysis_type,
                channel_index=binding.wav_channels[0] - 1, execution_status='分析完成',
                judgement_status='completed' if judgement else 'not_enabled',
                judgement=judgement, violations=violations, metrics=metrics or {}, display_payload=payload)


def publish(projection, presenter, items, condition='c1', station='s1'):
    result = dict(task_id='task', station_id=station, condition_id=condition, instance_results=items)
    presenter.analysis_event(dict(event_type='task_result', is_current=True, payload=result,
                                  task_id='task', station_id=station, condition_id=condition))


def matrix_payload(offset=0):
    return dict(kind='spectrogram', matrix_axes=['frequency', 'time'], time_s=[0, 1, 2],
                frequency_hz=[0, 100, 1000, 10000],
                values_db=(np.arange(12).reshape(4, 3) + offset).tolist())


def test_spectrogram_regions_common_scale_full_matrix_and_detail(grid, configuration, qt_app):
    config = extended_configuration(configuration)
    projection = show(grid, config, qt_app)
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    cell = grid._cells[0, 0]
    assert isinstance(cell, ConditionHeatmapCell)
    assert cell.kind == 'heatmap'
    for index, binding in enumerate(projection.bindings[0][0]):
        publish(projection, presenter, [instance(binding, matrix_payload(index * 40))],
                condition=binding.condition_id, station=binding.station_id)
    for _ in range(5):
        qt_app.processEvents()
    assert len(cell._images) == 2
    assert all(im.levels == (0, 100) for im in cell._images)
    first, second = cell._images
    assert first.boundingRect().right() < second.boundingRect().left()
    assert first.image.width() == 3 and first.image.height() == 4
    assert not first._uniform  # Nonuniform CQT-style frequency rows retain actual positions.
    cell.open_detail()
    qt_app.processEvents()
    detail = cell._detail_dialog.cell
    assert isinstance(detail, ConditionHeatmapCell)
    assert len(detail._images) == 2
    np.testing.assert_array_equal(detail._condition_heatmaps['c1']['values_db'], np.arange(12).reshape(4, 3))
    assert detail._condition_heatmaps['c1']['values_db'] is not cell._condition_heatmaps['c1']['values_db']
    cell.set_condition_state('c2', '采集中', clear=True)
    assert len(detail._images) == 1
    assert detail.empty_labels['c2'].text() == '采集中'
    cell.clear_plot()
    assert not detail.has_result_data()
    cell.close_detail()


def test_spectrogram_auto_colors_recolor_all_conditions_and_log_dc(grid, configuration, qt_app):
    config = extended_configuration(configuration, scale='log', auto_colors=True)
    projection = show(grid, config, qt_app)
    cell = grid._cells[0, 0]
    for key, offset in [('c1', 0), ('c2', 40)]:
        data = validate_spectrogram(matrix_payload(offset))
        cell.set_condition_heatmap_data(key, dict(data, wav_channel=1))
    assert all(image.levels == (0, 51) for image in cell._images)
    assert cell._images[0].image.height() == 3  # zero Hz is excluded on log display only
    assert cell._condition_heatmaps['c1']['values_db'].shape == (4, 3)
    assert np.isfinite(cell._images[0].y_edges).all()
    data = validate_spectrogram(dict(matrix_payload(), frequency_hz=[0, 1000], values_db=[[0, 1, 2], [3, 4, 5]]))
    cell.set_condition_heatmap_data('c1', dict(data, wav_channel=1))
    assert cell._images[0].image.height() == 1


@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_spectrogram_khz_ticks_preserve_hz_data_and_detail(grid, configuration, qt_app, scale):
    config = extended_configuration(configuration, scale=scale)
    show(grid, config, qt_app)
    cell = grid._cells[0, 0]
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    source = validate_spectrogram(matrix_payload())
    cell.set_condition_heatmap_data('c1', dict(source, wav_channel=1))
    cell.open_detail()
    qt_app.processEvents()
    for target in (cell, cell._detail_dialog.cell):
        axis = target.plot_widget.getPlotItem().getAxis('left')
        assert axis.labelText == 'kHz'
        assert not axis.autoSIPrefix
        if scale == 'linear':
            assert axis.tickStrings([0, 10000, 20000], axis.scale, 10000) == ['0', '10', '20']
        else:
            assert axis.tickStrings([2, 3, 4], axis.scale, 1) == ['0.1', '1', '10¹']
        np.testing.assert_array_equal(target._condition_heatmaps['c1']['frequency_hz'], source['frequency_hz'])
        assert target._configuration['y_range'][1] == 20000
    cell.clear_plot()
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    cell.close_detail()


@pytest.mark.parametrize('change', [
    {'matrix_axes': ['time', 'frequency']}, {'values_db': [[1, 2]]},
    {'frequency_hz': [0, 100, 100, 1000]}, {'time_s': [0, 1, 3]},
    {'values_db': [[float('nan')] * 3] * 4},
])
def test_invalid_spectrogram_reports_error_without_crashing(change, configuration):
    config = extended_configuration(configuration)
    projection = build_ui_projection(config.ui, config.analysis)
    binding = projection.bindings[0][0][0]
    segment = projection.cells[0][0].condition_plot.conditions[0]
    result = build_condition_result(binding, segment, {'instance_results': [instance(binding, {**matrix_payload(), **change})]})
    assert result.heatmap is None and result.status == '分析失败'


def test_rms_only_acceleration_channels_and_no_db_conversion(grid, configuration, qt_app):
    config = extended_configuration(configuration, 'VIBRATION_RMS')
    projection = show(grid, config, qt_app)
    binding = projection.bindings[0][0][0]
    segment = projection.cells[0][0].condition_plot.conditions[0]
    payload = dict(kind='curve', x=[0, 1, 2], y=[.01, .03, .02], x_unit='s', y_unit='m/s²', upper=[], lower=[])
    items = [dict(instance(binding, deepcopy(payload)), channel_index=i) for i in (0, 1)]
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    publish(projection, presenter, items, condition=binding.condition_id, station=binding.station_id)
    cell = grid._cells[0, 0]
    np.testing.assert_array_equal(cell._condition_series['c1'][0]['y'], [.01, .03, .02])
    assert cell._configuration['y_label'] == 'm/s²'
    wrong = deepcopy(items)
    wrong[0]['display_payload']['y_unit'] = 'g'
    update = build_condition_result(binding, segment, {'instance_results': wrong})
    assert update.status == '部分结果不可用' and len(update.series) == 1
    wrong_context = replace(config.analysis.context, stations=configuration.analysis.context.stations)
    bad = build_ui_projection(config.ui, replace(config.analysis, context=wrong_context))
    assert bad.bindings[0][0][0] is None
    assert '加速度' in bad.cells[0][0].warning_detail


def test_single_sample_violation_survives_stride_and_scalar_has_no_false_line(grid, configuration, qt_app):
    projection = show(grid, configuration, qt_app)
    binding = projection.bindings[0][0][0]
    segment = projection.cells[0][0].condition_plot.conditions[0]
    x = np.linspace(0, 3, 301)
    y = np.full(301, 60.)
    y[137] = 95
    payload = dict(kind='curve', x=x, y=y, upper=np.full(301, 80.), lower=[], x_unit='s', y_unit='dBA')
    violation = dict(x_start=x[137], x_end=x[137], x_unit='s', side='above_upper')
    items = [instance(binding, payload, judgement='NG', violations=[violation]),
             dict(instance(binding, payload), channel_index=1)]
    update = build_condition_result(binding, segment, {'instance_results': items})
    cell = grid._cells[0, 0]
    cell._display_downsampling = {'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': 10}
    cell.set_condition_curve_data('c1', update.series, summaries=update.summaries)
    lines = cell.plot_widget.listDataItems()
    assert np.nanmax(lines[0].yData) == 60  # deliberate display stride misses spike
    assert np.nanmax(lines[2].yData) == 95  # independent unsampled backend highlight
    assert lines[2].opts['symbol'] == 'o'
    assert 'NG' in cell.condition_labels['c1'].text()
    scalar = instance(binding, dict(payload, upper=[]), judgement='NG',
                      violations=[dict(x_start=None, x_end=None)],
                      metrics={'overall_spl': 88, 'overall_upper_limit': 80, 'unit': 'dBA'})
    update = build_condition_result(binding, segment, {'instance_results': [scalar, items[1]]})
    assert not any(s['role'] == 'exceedance' for s in update.series)
    cell.set_condition_curve_data('c1', update.series, summaries=update.summaries)
    assert '总体 SPL 88' in cell.condition_labels['c1'].toolTip()
    assert 'CH1 88.0 dB(A)' in cell.overall_labels['c1'].text()
    assert cell._overall_spl_items[0][1].line().y1() == 88  # measured value, not the limit 80
    assert not cell.condition_labels['c1'].testAttribute(Qt.WA_TransparentForMouseEvents)


@pytest.mark.parametrize('channels', [[1], [1, 2]])
def test_overall_spl_references_use_backend_values_and_follow_detail_and_clear(grid, configuration, qt_app, channels):
    layout = template_layout()
    layout['plot_templates']['spl']['channel_display'] = {
        'show_all_analyzed_channels': False, 'display_wav_channels': channels}
    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(layout)))
    projection = show(grid, config, qt_app)
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    cell = grid._cells[0, 0]
    binding = projection.bindings[0][0][0]
    payload = dict(kind='curve', x=[0, 1, 2], y=[40, 40, 40], upper=[], lower=[], x_unit='s', y_unit='dBA')
    items = [dict(instance(binding, payload, metrics={'overall_spl': 65.46 + i, 'unit': 'dBA'}), channel_index=i)
             for i in range(len(channels))]
    publish(projection, presenter, items)
    qt_app.processEvents()
    for i in range(len(channels)):
        prefix = f'CH{i + 1} ' if len(channels) > 1 else ''
        key, line, text, value = cell._overall_spl_items[i]
        assert text == f'{prefix}{65.46 + i:.1f} dB(A)'
        assert text in cell.overall_labels['c1'].text()
        assert key == 'c1'
        assert line.line().y1() == line.line().y2() == value == 65.46 + i
        assert (line.line().x1(), line.line().x2()) == cell._spans['c1']
        assert line.pen().style() == Qt.DashLine
        assert line.zValue() < cell.plot_widget.listDataItems()[0].zValue()
    assert len(cell._overall_spl_items) == len(channels)
    assert '总体' not in cell.condition_labels['c2'].text()
    rect = cell.plot_widget.mapFromScene(cell.plot_widget.getViewBox().sceneBoundingRect()).boundingRect()
    assert cell.condition_labels['c1'].geometry().bottom() < rect.top()
    cell.open_detail()
    qt_app.processEvents()
    detail = cell._detail_dialog.cell
    assert [text for _, _, text, _ in detail._overall_spl_items] == [
        text for _, _, text, _ in cell._overall_spl_items]
    for target in (cell, detail):
        rect = target.plot_widget.mapFromScene(target.plot_widget.getViewBox().sceneBoundingRect()).boundingRect()
        assert target.condition_labels['c1'].geometry().bottom() < rect.top()
        overall = target.overall_labels['c1']
        assert '总体' not in overall.text()
        assert overall.geometry().top() - target.condition_labels['c1'].geometry().bottom() >= 12
        assert overall.geometry().bottom() < rect.top()
        assert overall.alignment() & Qt.AlignHCenter
        assert overall.geometry().center().x() == target.condition_labels['c1'].geometry().center().x()
        assert not any(isinstance(item, pg.TextItem) for item in target.plot_widget.getPlotItem().items)
    cell.set_condition_state('c1', '采集中', clear=True)
    qt_app.processEvents()
    assert '总体' not in cell.condition_labels['c1'].text()
    assert '总体' not in cell._detail_dialog.cell.condition_labels['c1'].text()
    assert not cell._overall_spl_items
    assert not cell._detail_dialog.cell._overall_spl_items
    assert not cell.overall_labels['c1'].isVisible()
    assert not cell._detail_dialog.cell.overall_labels['c1'].isVisible()
    cell.close_detail()


def test_overall_spl_references_separate_conditions_and_ignore_invalid_values(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    series = [dict(wav_channel=1, name='CH1', x=[0, 1, 2], y=[40, 50, 40])]
    def summary(value):
        return dict(wav_channel=1, channel='CH1', metrics={'overall_spl': value, 'unit': 'dBZ'})
    cell.set_condition_curve_data('c1', series, summaries=[summary(45)])
    cell.set_condition_curve_data('c2', series, summaries=[summary(47)])
    qt_app.processEvents()
    assert len(cell._overall_spl_items) == 2
    for key, line, text, value in cell._overall_spl_items:
        assert (line.line().x1(), line.line().x2()) == cell._spans[key]
        assert text.endswith('dB(Z)')
    cell.set_condition_curve_data('c1', series, summaries=[summary(None), summary(float('nan'))])
    assert [key for key, _, _, _ in cell._overall_spl_items] == ['c2']
    cell.clear_plot()
    assert not cell._overall_spl_items


@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_real_wav_spectrogram_backend_to_canvas(grid, configuration, qt_app, tmp_path, scale):
    config = extended_configuration(configuration, scale=scale)
    projection = show(grid, config, qt_app)
    binding = projection.bindings[0][0][0]
    rate = 44100
    time = np.arange(rate * 3) / rate
    wav = tmp_path / 'tone.wav'
    sf.write(wav, .02 * np.sin(2 * np.pi * 1000 * time), rate, subtype='PCM_24')
    task = AnalysisTask('real-spec', binding.station_id, '工位1', binding.condition_id, '工况1',
                        str(wav), str(tmp_path), (AnalysisChannel(0, 'CH1', 'sound_pressure', v2pa_factor=1.),),
                        (AnalysisInstance(binding.analysis_item, 'Spec', 0,
                         {'n_fft': 2048, 'hop_length': 256, 'freq_scale_type': scale}),), {'mode': 'none'})
    result = execute_task(task)
    assert result['instance_results'][0]['execution_status'] == '分析完成', result
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    presenter.analysis_event(dict(event_type='task_result', is_current=True, payload=result,
                                  task_id=task.task_id, station_id=task.station_id, condition_id=task.condition_id))
    cell = grid._cells[0, 0]
    assert cell._condition_states['c1'] == ('', '')
    heatmap = cell._condition_heatmaps['c1']
    peak = np.argmax(np.mean(heatmap['values_db'], axis=1))
    assert heatmap['frequency_hz'][peak] == pytest.approx(1000, rel=.08)
    qt_app.processEvents()
    assert not cell.grab().isNull()


def test_real_spl_limit_interpolation_and_judgement_are_preserved(grid, configuration, qt_app, tmp_path):
    projection = show(grid, configuration, qt_app)
    binding = projection.bindings[0][0][0]
    rate = 44100
    time = np.arange(rate * 3) / rate
    wav = tmp_path / 'limits.wav'
    tone = .1 * np.sin(2 * np.pi * 1000 * time)
    sf.write(wav, np.column_stack((tone, tone)), rate, subtype='PCM_24')
    parameters = dict(weighting='A', limit_checked=True, limit_metric='curve_y', limit_mode='csv',
                      limit_data=[[0, 3], [60, 100], [None, None]])
    task = AnalysisTask('limits', binding.station_id, '工位1', binding.condition_id, '工况1', str(wav), str(tmp_path),
                        tuple(AnalysisChannel(i, f'CH{i + 1}', 'sound_pressure', v2pa_factor=1.) for i in range(2)),
                        tuple(AnalysisInstance(binding.analysis_item, 'SPL', i, parameters) for i in range(2)), {'mode': 'none'})
    result = execute_task(task)
    item = result['instance_results'][0]
    assert item['judgement'] == 'NG', item
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    presenter.analysis_event(dict(event_type='task_result', is_current=True, payload=result,
                                  task_id=task.task_id, station_id=task.station_id, condition_id=task.condition_id))
    curves = grid._cells[0, 0]._condition_series['c1']
    upper = next(s for s in curves if s['role'] == 'upper' and s['wav_channel'] == 1)
    np.testing.assert_array_equal(upper['y'], item['display_payload']['upper'])
    np.testing.assert_allclose(upper['y'], 60 + upper['x'] / 3 * 40)
    highlight = next(s for s in curves if s['role'] == 'exceedance' and s['wav_channel'] == 1)
    expected = np.asarray(item['display_payload']['y']) > upper['y']
    np.testing.assert_array_equal(np.isfinite(highlight['y']), expected)


def test_spectrogram_multiple_channels_require_explicit_selection(configuration):
    config = extended_configuration(configuration)
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type='Spec', chart_type='heatmap')
    config = replace(config, ui=replace(config.ui, layout=parse_plot_templates(layout)))
    projection = build_ui_projection(config.ui, config.analysis)
    assert projection.bindings[0][0][0] is None
    assert '只显示一个' in projection.cells[0][0].warning_detail


@pytest.mark.parametrize('settings', [
    {'color_map': 'unknown'}, {'show_color_bar': 1},
    {'color_range': {'mode': 'fixed', 'min_db': 100, 'max_db': 0}},
    {'color_range': {'mode': 'auto', 'min_db': 0}},
])
def test_invalid_heatmap_config_is_rejected_before_ui(settings):
    layout = template_layout()
    layout['plot_templates']['spl'].update(analysis_type='Spec', chart_type='heatmap', spectrogram=settings)
    with pytest.raises(ValueError, match='spectrogram'):
        parse_plot_templates(layout)


def test_heatmap_preserves_view_clip_and_frequency_row_orientation(qt_app):
    item = SpectrogramImage(np.array([[0., 0.], [100., 100.]]), [0, 50, 100], [0, 50, 100],
                            (0, 100), pg.colormap.get('viridis'), (0, 100))
    canvas = QImage(100, 100, QImage.Format_ARGB32)
    canvas.fill(0xffffffff)
    painter = QPainter(canvas)
    painter.setClipRect(QRectF(0, 20, 100, 60))
    item.paint(painter, None)
    painter.end()
    assert canvas.pixelColor(50, 10).name() == '#ffffff'
    assert canvas.pixelColor(50, 90).name() == '#ffffff'
    assert canvas.pixelColor(50, 30).blue() > canvas.pixelColor(50, 30).red()  # low frequency purple
    assert canvas.pixelColor(50, 70).red() > canvas.pixelColor(50, 70).blue()  # high frequency yellow
