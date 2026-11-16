"""Regression contracts for shared rendering and product-driven workbench columns."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QRectF, Qt

from base.analysis_curve_style import resolve_curve_colors
from base.workbench_configuration_loader import load_workbench_configuration
from base.workbench_plot_config import parse_plot_templates
from ui.plotting.heatmap import create_image, update_image
from ui.plotting.renderers import (
    analysis_curve_pen, draw_bars, draw_curve, draw_exceedance, draw_fill,
    draw_scalar_limits, draw_scatter, draw_text,
)
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.sequence.workbench_plot_display import ConditionSegmentDisplay
from ui.sequence.workbench_plot_results import build_condition_result
from ui.sequence.workbench_ui_projection import build_ui_projection
from unit_test.base.workbench_config_fixtures import write_json, write_workbench_files
from unit_test.base.workbench_plot_fixtures import template_layout


def station_layout():
    layout = template_layout()
    layout.pop('rows')
    layout.pop('columns')
    cells = layout.pop('cells')
    layout.update(schema_version=2, column_source='product_stations',
                  row_templates=[row[0] for row in cells])
    layout['default_plot_style'].pop('channel_curve_styles')
    layout['default_condition_display']['x_axis_mode'] = 'condition_local'
    return layout


def close_widget(widget):
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_columns_follow_product_order_and_condition_analysis_colors(tmp_path, qt_app, count):
    files = write_workbench_files(tmp_path, all_stations=True)
    product = json.loads(files['product'].read_text(encoding='utf-8'))
    product['stations'] = list(reversed(product['stations'][:count]))
    expected_ids = [s['station_id'] for s in product['stations']]
    for condition in product['conditions']:
        condition['applicable_station_ids'] = expected_ids
    write_json(files['product'], product)
    for index, name in enumerate(('测试队列A', '测试队列B')):
        queue = json.loads(files[name].read_text(encoding='utf-8'))
        item = queue[0]['seq1']['analysis_list']['声压级']
        item['weighting'] = 'A'
        item['display'].update(main_curve_color=('#123456', '#654321')[index],
                               upper_limit_color='#C06020', lower_limit_color='#6020C0')
        write_json(files[name], queue)
    write_json(files['layout'], station_layout())
    configuration = load_workbench_configuration(files['programs'], files['queues'], files['layout'])
    before = configuration.analysis.describe()
    projection = build_ui_projection(configuration.ui, configuration.analysis)
    assert projection.column_names == tuple(s['station_name'] for s in product['stations'])
    assert len(projection.cells[0]) == count
    assert [bindings[0].station_id for bindings in projection.bindings[0]] == expected_ids
    grid = StationPlotGrid()
    try:
        grid.show_workbench_layout(projection.column_names, projection.cells)
        assert len(grid._cells) == count
        cell = grid._cells[0, 0]
        for segment in projection.cells[0][0].condition_plot.conditions:
            colors = dict(segment.curve_colors)
            assert colors['main_curve_color'] == ('#123456' if segment.condition_id == 'c1' else '#654321')
            cell.set_condition_curve_data(segment.condition_id, [dict(
                name='CH1', wav_channel=segment.channels[0][0], role=role,
                x=[0, 1, 2], y=[40, 50, 40] if role == 'curve' else [60, 60, 60],
            ) for role in ('curve', 'upper')])
        items = cell.plot_widget.listDataItems()
        assert len(items) == 4
        assert [item.opts['pen'].color().name() for item in items] == [
            '#123456', '#c06020', '#654321', '#c06020']
        assert all(item.opts['pen'].widthF() == 2 for item in items)
        assert items[1].opts['pen'].style() == Qt.DashLine
        cell.open_detail()
        qt_app.processEvents()
        assert len(cell._detail_dialog.cell.plot_widget.listDataItems()) == 4
        cell.close_detail()
    finally:
        for cell in grid._cells.values():
            cell.dispose()
        close_widget(grid)
    assert configuration.analysis.describe() == before


@pytest.mark.parametrize('key,value', [
    ('plot_templates', None), ('plot_templates', []), ('plot_templates', {}),
    ('schema_version', True), ('schema_version', 3), ('row_templates', []),
    ('column_source', 'fixed'), ('columns', 4),
])
def test_v2_rejects_invalid_and_competing_column_definitions(key, value):
    layout = station_layout()
    layout[key] = value
    with pytest.raises(ValueError):
        parse_plot_templates(layout)


@pytest.mark.parametrize('location', ['default', 'template', 'override'])
def test_v2_does_not_accept_second_threshold_style_source(location):
    layout = station_layout()
    style = {'limit_lines': {'visible': False}}
    if location == 'default':
        layout['default_plot_style'].update(style)
    elif location == 'template':
        layout['plot_templates']['spl']['plot_style'].update(style)
    else:
        layout['row_templates'][0]['overrides'] = {'plot_style': style}
    with pytest.raises(ValueError, match='v2'):
        parse_plot_templates(layout)


@pytest.mark.parametrize('kind,unit,metric', [
    ('SPL', 'dB(A)', 'overall_spl'), ('LOUD', 'sone', 'steady_state_average'),
    ('LOUD', 'phon', 'max_transient'),
])
@pytest.mark.parametrize('judgement', ['OK', 'NG'])
def test_scalar_limits_use_backend_values_without_peak_rejudgement(kind, unit, metric, judgement):
    segment = ConditionSegmentDisplay('c1', '正转', channels=((1, 'CH1'),), x_unit='s', y_unit=unit)
    item = dict(config_key='item', analysis_type=kind, channel_index=0,
                execution_status='分析完成', judgement_status='completed', judgement=judgement,
                metrics={'overall_upper_limit': 40., 'overall_lower_limit': 0.,
                         'limit_metric': metric, 'limit_unit': unit},
                display_payload={'kind': 'curve', 'x': [0, 1, 2], 'y': [20, 80, 20],
                                 'upper': [], 'lower': [], 'x_unit': 's', 'y_unit': unit},
                violations=[{'x_start': None, 'x_end': None}])
    before = deepcopy(item)
    result = build_condition_result(SimpleNamespace(analysis_item='item', analysis_type=kind),
                                    segment, {'instance_results': [item]})
    assert not result.status
    assert [s['role'] for s in result.series] == ['curve', 'upper', 'lower']
    np.testing.assert_array_equal(result.series[0]['y'], [20, 80, 20])
    np.testing.assert_array_equal(result.series[1]['y'], [40, 40, 40])
    np.testing.assert_array_equal(result.series[2]['y'], [0, 0, 0])
    assert result.series[1]['limit_scope'] == 'scalar'
    assert item == before


@pytest.mark.parametrize('metric,limit_unit,expected_status', [
    ('specific_loudness_summed_exceedance', 'sone', ''),
    ('steady_state_average', 'phon', '部分结果不可用'),
])
def test_scalar_limits_are_not_drawn_on_a_different_quantity(metric, limit_unit, expected_status):
    segment = ConditionSegmentDisplay('c1', '正转', channels=((1, 'CH1'),), x_unit='s', y_unit='sone')
    item = dict(config_key='item', analysis_type='LOUD', channel_index=0,
                execution_status='分析完成', judgement_status='completed', judgement='NG',
                metrics={'overall_upper_limit': 40., 'limit_metric': metric, 'limit_unit': limit_unit},
                display_payload={'kind': 'curve', 'x': [0, 1], 'y': [1, 2],
                                 'upper': [], 'lower': [], 'x_unit': 's', 'y_unit': 'sone'})
    result = build_condition_result(SimpleNamespace(analysis_item='item', analysis_type='LOUD'),
                                    segment, {'instance_results': [item]})
    assert [s['role'] for s in result.series] == ['curve']
    assert result.status == expected_status


def test_shared_primitives_keep_data_geometry_and_existing_items(qt_app):
    plot = pg.PlotWidget()
    try:
        x, y = np.arange(5.), np.array([1., 3., 2., 5., 0.])
        curve = draw_curve(plot, x, y, pen='g')
        out = draw_exceedance(plot, x, y, [False, True, False, True, True])
        np.testing.assert_equal(out.xData, [1, np.nan, 3, 4])
        np.testing.assert_equal(out.yData, [3, np.nan, 5, 0])
        colors = resolve_curve_colors({})
        limits = draw_scalar_limits(plot, [0, 4], 4, 0, colors=colors)
        assert len(limits) == 2
        assert limits[0].opts['pen'] == analysis_curve_pen('upper', colors)
        bars = draw_bars(plot, x=[1, 2], height=[3, 4], width=.8, brushes=['g', 'r'])
        np.testing.assert_equal(bars.opts['height'], [3, 4])
        scatter = draw_scatter(plot, x=[1, 3], y=[3, 5], size=8)
        np.testing.assert_equal(scatter.getData()[1], [3, 5])
        text = draw_text(plot, 'NG', (3, 5), color='r')
        assert text.pos().x() == 3
        fill = draw_fill(plot, x, y, np.zeros(5), brush=(255, 0, 0, 50))
        np.testing.assert_equal(fill.curves[0].yData, y)
        image = create_image(plot, axisOrder='row-major')
        matrix = np.arange(12).reshape(3, 4)
        update_image(image, matrix, rect=QRectF(0, 10, 2, 30), levels=(0, 11),
                     color_map=pg.colormap.get('viridis'))
        np.testing.assert_equal(image.image, matrix)
        assert image.mapRectToParent(image.boundingRect()) == QRectF(0, 10, 2, 30)
        assert curve in plot.listDataItems()
        np.testing.assert_equal(curve.yData, y)
    finally:
        close_widget(plot)


def test_fft_reference_is_a_separate_gray_layer_and_survives_detail(qt_app):
    from ui.sequence.condition_plot_cell import ConditionPlotCell
    from ui.sequence.workbench_plot_display import ConditionPlotDisplay, WorkbenchPlotCellDisplay

    layout = station_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type='FFT', analysis_item_name='fft')
    template['x_axis'] = dict(label='频率', scale='log', auto_range=False, range_min=20, range_max=20000)
    settings = parse_plot_templates(layout).cells[0][0].display_settings
    segment = ConditionSegmentDisplay('c1', '正转', channels=((1, 'CH1'),),
                                      x_unit='Hz', y_unit='dB(Z)', analysis_type='FFT',
                                      curve_colors=tuple(resolve_curve_colors({}).items()))
    item = dict(config_key='fft', analysis_type='FFT', channel_index=0,
                execution_status='分析完成', judgement_status='not_enabled',
                display_payload=dict(kind='curve', x=[100, 1000, 10000], y=[30, 60, 20],
                                     baseline=[25, 50, 15], x_unit='Hz', y_unit='dB'))
    data = build_condition_result(SimpleNamespace(analysis_item='fft', analysis_type='FFT'),
                                  segment, {'instance_results': [item]})
    assert not data.status
    assert [s['role'] for s in data.series] == ['curve', 'baseline']
    cell = ConditionPlotCell(WorkbenchPlotCellDisplay(title='FFT', condition_plot=ConditionPlotDisplay(
        settings, (segment,))))
    try:
        cell.set_condition_curve_data('c1', data.series)
        reference = cell.plot_widget.listDataItems()[1]
        assert reference.opts['pen'].color().name() == '#808080'
        assert reference.opts['pen'].widthF() == 2
        np.testing.assert_equal(reference.yData, [25, 50, 15])
        cell.open_detail()
        qt_app.processEvents()
        assert cell._detail_dialog.cell.plot_widget.listDataItems()[1].opts['pen'] == reference.opts['pen']
        cell.set_condition_state('c1', '', clear=True)
        assert not cell._detail_dialog.cell.plot_widget.listDataItems()
    finally:
        cell.dispose()
        close_widget(cell)
