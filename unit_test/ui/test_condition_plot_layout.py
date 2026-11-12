import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtGui import QImage, QPainter

from base.workbench_configuration_loader import load_workbench_configuration
from base.workbench_plot_config import parse_plot_templates
from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.motor_workbench_controller import MotorWorkbenchController
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.sequence.workbench_ui_projection import build_ui_projection
from unit_test.base.workbench_config_fixtures import write_json, write_workbench_files
from unit_test.base.workbench_plot_fixtures import template_layout


@pytest.fixture
def configuration(tmp_path):
    files = write_workbench_files(tmp_path, all_stations=True)
    # Two known microphone curves, with consistent units, in each condition.
    for key in ('测试队列A', '测试队列B'):
        queue = json.loads(files[key].read_text(encoding='utf-8'))
        item = queue[0]['seq1']['analysis_list']['声压级']
        item.update(analysis_channels=[0, 1], weighting='A')
        write_json(files[key], queue)
    write_json(files['layout'], template_layout())
    return load_workbench_configuration(files['programs'], files['queues'], files['layout'])


@pytest.fixture
def grid(qt_app):
    widget = StationPlotGrid()
    widget.resize(1600, 650)
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def show(grid, configuration, qt_app):
    projection = build_ui_projection(configuration.ui, configuration.analysis)
    grid.show_workbench_layout(projection.column_names, projection.cells)
    qt_app.processEvents()
    return projection


def curve(channel=1, low=45, high=65, role='curve'):
    return {'wav_channel': channel, 'role': role, 'name': f'CH{channel}', 'x': [0, 1, 2], 'y': [low, high, low]}


def test_dense_curves_keep_full_sources_limits_gaps_and_independent_detail(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    source = grid._cells[0, 0]
    x = np.linspace(0, 3, 30001)
    y = np.full(x.size, 55.)
    y[12345] = 95
    y[15000:15003] = np.nan
    source.set_condition_curve_data('c1', [
        dict(curve(), x=x, y=y),
        dict(curve(role='upper'), x=x, y=np.full(x.size, 80.)),
    ])
    source.set_condition_curve_data('c2', [dict(curve(channel=2), x=x, y=y + 1)])
    qt_app.processEvents()
    lines = source.plot_widget.listDataItems()
    assert len(lines[0].xData) < 3000
    assert np.nanmax(lines[0].yData) == 95
    assert np.isnan(lines[0].yData).sum() == 2
    assert lines[0].opts['connect'] == 'finite'
    assert len(lines[1].xData) == len(x)  # Limits keep every turn.
    assert lines[0].xData.max() <= 3
    assert lines[2].xData.min() >= 3
    np.testing.assert_array_equal(source._condition_series['c1'][0]['y'], y)
    source.open_detail()
    qt_app.processEvents()
    detail = source._detail_dialog.cell
    view = detail.plot_widget.getViewBox()
    view.setXRange(1.23, 1.24, padding=0)
    view.sigRangeChangedManually.emit([True, False])
    qt_app.processEvents()
    detail_line = detail.plot_widget.listDataItems()[0]
    assert np.count_nonzero((detail_line.xData >= 1.23) & (detail_line.xData <= 1.24)) == 101
    assert source.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 6])
    source._detail_dialog.resize(1150, 700)
    for _ in range(3):
        qt_app.processEvents()
    assert view.viewRange()[0] == pytest.approx([1.23, 1.24])
    source.close_detail()


def test_result_null_is_a_curve_gap_but_infinity_remains_invalid(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    cell.set_condition_curve_data('c1', [dict(curve(), y=[45, None, 65])])
    assert np.isnan(cell.plot_widget.listDataItems()[0].yData[1])
    with pytest.raises(ValueError, match='finite'):
        cell.set_condition_curve_data('c1', [dict(curve(), y=[45, float('inf'), 65])])


def test_detail_initial_resize_keeps_complete_canvas_with_pixel_gutter(grid, configuration, qt_app):
    data = template_layout()
    data['default_condition_display'].update(x_axis_mode='condition_local', condition_gap_px=16)
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    show(grid, configuration, qt_app)
    source = grid._cells[0, 0]
    source.set_condition_curve_data('c1', [curve()])
    source.open_detail()
    detail = source._detail_dialog.cell
    for _ in range(5):
        qt_app.processEvents()
    assert detail.plot_widget.getViewBox().viewRange()[0] == pytest.approx(detail._configuration['x_range'])
    source._detail_dialog.resize(1200, 750)
    for _ in range(5):
        qt_app.processEvents()
    assert detail.plot_widget.getViewBox().viewRange()[0] == pytest.approx(detail._configuration['x_range'])
    source.close_detail()


def test_detail_window_reuses_memory_updates_and_closes_on_configuration_change(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    source = grid._cells[0, 0]
    source.set_condition_curve_data('c1', [curve()])
    source.expand_button.click()
    qt_app.processEvents()
    detail = source._detail_dialog
    assert detail.isVisible()
    assert '工位1' in detail.windowTitle()
    assert not hasattr(detail.cell, 'expand_button')
    for cell in (source, detail.cell):
        axis = cell.plot_widget.getPlotItem().getAxis('left')
        assert axis.labelText == 'dB(A)'
        assert axis.labelUnits == ''
    source.expand_button.click()
    assert source._detail_dialog is detail
    assert detail.cell._condition_series['c1'][0]['x'] is not source._condition_series['c1'][0]['x']
    np.testing.assert_array_equal(detail.cell._condition_series['c1'][0]['y'], [45, 65, 45])
    view = detail.cell.plot_widget.getViewBox()
    assert all(view.state['mouseEnabled'])
    view.setRange(xRange=(0.5, 2), yRange=(48, 60), padding=0)
    view.sigRangeChangedManually.emit([True, True])
    source.set_condition_curve_data('c2', [curve(low=50)])
    assert set(detail.cell._condition_series) == {'c1', 'c2'}
    assert view.viewRange()[0] == pytest.approx([0.5, 2])
    assert source.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 6])
    source.clear_plot()
    assert not detail.cell.plot_widget.listDataItems()
    detail.reject()  # Escape and the close button must release the same resources.
    assert source._detail_dialog is None
    source.set_condition_curve_data('c1', [curve()])
    source.open_detail()
    reopened = source._detail_dialog
    assert reopened is not detail
    assert reopened.cell.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 6])
    show(grid, configuration, qt_app)
    assert source._detail_dialog is None
    assert not reopened.isVisible()


def test_detail_auto_range_follows_later_condition_after_resize(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    source = grid._cells[0, 0]
    source.set_condition_curve_data('c1', [curve(low=30, high=40)])
    source.open_detail()
    dialog = source._detail_dialog
    for _ in range(5):
        qt_app.processEvents()
    dialog.resize(1150, 700)
    for _ in range(5):
        qt_app.processEvents()
    source.set_condition_curve_data('c2', [curve(low=80, high=90)])
    for _ in range(5):
        qt_app.processEvents()
    bounds = dialog.cell.plot_widget.getViewBox().viewRange()[1]
    assert bounds[0] <= 30 and bounds[1] >= 90
    source.close_detail()


@pytest.mark.parametrize('manual_axes', [[True, False], [True, True]])
def test_detail_preserves_only_manually_changed_axes_and_resets_next_round(
        grid, configuration, qt_app, manual_axes):
    show(grid, configuration, qt_app)
    source = grid._cells[0, 0]
    source.set_condition_curve_data('c1', [curve(low=30, high=40)])
    source.open_detail()
    for _ in range(5):
        qt_app.processEvents()
    view = source._detail_dialog.cell.plot_widget.getViewBox()
    ranges = dict(xRange=(0.5, 2))
    if manual_axes[1]:
        ranges['yRange'] = (32, 38)
    view.setRange(**ranges, padding=0)
    view.sigRangeChangedManually.emit(manual_axes)
    source.set_condition_curve_data('c2', [curve(low=80, high=90)])
    assert view.viewRange()[0] == pytest.approx([0.5, 2])
    if manual_axes[1]:
        assert view.viewRange()[1] == pytest.approx([32, 38])
    else:
        assert view.viewRange()[1][1] >= 90
    source.clear_plot()
    source.set_condition_curve_data('c1', [curve(low=100, high=110)])
    assert view.viewRange()[0] == pytest.approx([0, 6])
    assert view.viewRange()[1][1] >= 110
    source.close_detail()


def test_existing_controller_builds_one_canvas_per_chart(grid, configuration, qt_app):
    stations = []
    controller = MotorWorkbenchController(
        repository=SimpleNamespace(load_active=lambda: configuration, source_identity='isolated.json'),
        station_panel=SimpleNamespace(show_workbench_stations=stations.append),
        plot_grid=grid, set_active_product=lambda _: None, show_video_settings=lambda: None,
    )
    controller.refresh_configuration()
    qt_app.processEvents()
    cell = grid._cells[0, 0]
    assert len(stations[-1]) == 4
    assert isinstance(cell, ConditionPlotCell)
    assert len(cell.findChildren(pg.PlotWidget)) == 1
    assert list(cell.conditions) == ['c1', 'c2']
    assert [label.text() for label in cell.condition_labels.values()] == ['工况1', '工况2']
    assert cell.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 6])
    assert len(cell.separators) == 1
    assert all(not label.text() for label in cell.empty_labels.values())
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    assert controller.get_plot_bindings(0, 0)[0].wav_channels == (1, 2)
    for visible in (False, True):
        grid.set_mark_buttons_visible(visible)
        for header in grid._headers.values():
            assert header.name.isVisible()
            assert all(control.isHidden() for control in (header.combo, header.play, header.ok, header.ng))
    assert grid._header_scroll.height() == 36
    assert cell.expand_button.iconSize().width() == 16
    assert cell.expand_button.width() == 24
    assert not cell.expand_button.icon().isNull()


def test_curves_are_offset_not_joined_with_shared_axes_and_configured_styles(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    source = curve(1, 80, 95)
    grid.set_condition_curve_data(0, 0, 'c1', [curve(1), curve(2), curve(1, 70, 70, 'upper')])
    grid.set_condition_curve_data(0, 0, 'c2', [source])
    cell = grid._cells[0, 0]
    plot = cell.plot_widget
    lines = plot.listDataItems()
    assert len(lines) == 4
    np.testing.assert_allclose(lines[0].xData, [0, 1, 2])
    np.testing.assert_allclose(lines[3].xData, [3, 4, 5])
    assert source['x'] == [0, 1, 2]
    assert lines[0].opts['pen'].color().name() == '#33c44d'
    assert lines[0].opts['pen'].widthF() == 2
    assert lines[1].opts['pen'].style() == Qt.DashLine
    assert lines[2].opts['pen'].color().name() == '#d9534f'
    assert lines[2].opts['pen'].style() == Qt.DashLine
    assert plot.getPlotItem().legend is None
    assert len(cell.legend_bar.entries) == 3
    assert plot.getPlotItem().getAxis('left').grid_pen.style() == Qt.DotLine
    assert plot.getViewBox().viewRange()[1][1] >= 95
    assert not cell.empty_labels['c1'].isVisible()
    grid.clear_plots()
    assert all(not label.text() for label in cell.empty_labels.values())
    assert not cell.empty_labels['c1'].isVisible()
    assert not plot.listDataItems()
    assert len(cell.separators) == 1


def test_configuration_switch_uses_applicable_conditions_in_product_order(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    grid.set_condition_curve_data(0, 0, 'c1', [curve()])
    context = configuration.analysis.context
    first = replace(context.conditions[1], condition_id='actual-uuid', condition_name='6000rpm', applicable_station_ids=('s1',))
    others = replace(context.conditions[0], applicable_station_ids=('s2',))
    analysis = replace(configuration.analysis, context=replace(context, conditions=(first, others)))
    changed = replace(configuration, analysis=analysis)
    projection = show(grid, changed, qt_app)
    cell = grid._cells[0, 0]
    assert list(cell.conditions) == ['actual-uuid']
    assert cell.condition_labels['actual-uuid'].text() == '6000rpm'
    assert len(cell.findChildren(pg.PlotWidget)) == 1
    assert not cell.separators
    assert cell.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 3])
    assert not cell.plot_widget.listDataItems()
    assert projection.bindings[0][0][0].condition_id == 'actual-uuid'
    assert list(grid._cells[0, 1].conditions) == ['c1']
    assert not grid._cells[0, 2].conditions
    assert grid._cells[0, 2]._no_conditions.isVisible()


def test_auto_time_axis_uses_recording_duration_and_more_than_two_conditions(grid, configuration, qt_app):
    data = template_layout()
    data['plot_templates']['spl']['x_axis'] = {'label': '累计时间', 'scale': 'linear', 'auto_range': True}
    context = configuration.analysis.context
    third = replace(context.conditions[0], condition_id='c3', condition_name='工况3')
    analysis = replace(configuration.analysis, context=replace(context, conditions=(*context.conditions, third)))
    ui = replace(configuration.ui, layout=parse_plot_templates(data),
                 condition_durations={**configuration.ui.condition_durations, 'c3': 4})
    projection = show(grid, replace(configuration, analysis=analysis, ui=ui), qt_app)
    cell = grid._cells[0, 0]
    assert len(projection.bindings[0][0]) == 3
    assert len(cell.findChildren(pg.PlotWidget)) == 1
    assert [line.value() for line in cell.separators] == [2.5, 6]
    assert cell.plot_widget.getViewBox().viewRange()[0] == pytest.approx([0, 10])


@pytest.mark.parametrize(('change', 'expected'), [
    ('channel', '未配置'), ('item', '缺少分析项'), ('type', '类型不符'),
])
def test_mismatches_have_short_labels_and_detailed_tooltips(grid, configuration, qt_app, change, expected):
    data = template_layout()
    if change == 'channel':
        data['plot_templates']['spl']['channel_display'] = {'show_all_analyzed_channels': False, 'display_wav_channels': [9]}
    elif change == 'item':
        data['plot_templates']['spl']['analysis_item_name'] = '不存在的项目'
    else:
        data['plot_templates']['spl']['analysis_type'] = 'FFT'
        data['default_condition_display']['x_axis_mode'] = 'condition_local'
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    projection = show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    assert next(iter(cell.empty_labels.values())).text() == '配置不匹配'
    assert expected in cell.header.right.toolTip()
    assert projection.bindings[0][0][0] is None


def test_log_frequency_restarts_per_condition_in_same_canvas(grid, configuration, qt_app):
    data = template_layout()
    template = data['plot_templates']['spl']
    template.update(analysis_type='FFT', analysis_item_name='频谱')
    template['x_axis'] = {'label': '频率', 'scale': 'log', 'auto_range': False, 'range_min': 20, 'range_max': 20000}
    data['default_condition_display']['x_axis_mode'] = 'condition_local'
    data['cells'][0][0]['overrides'] = {'plot_style': {
        'legend': {'visible': False, 'position': 'top_left'},
        'limit_lines': {'visible': False,
                        'upper': {'color': '#000000', 'line_style': 'solid', 'line_width': 1},
                        'lower': {'color': '#000000', 'line_style': 'solid', 'line_width': 1}},
    }}
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    show(grid, configuration, qt_app)
    grid.set_condition_curve_data(0, 0, 'c1', [dict(curve(2), x=[20, 200, 20000]), dict(curve(2, role='upper'), x=[20, 200, 20000])])
    grid.set_condition_curve_data(0, 0, 'c2', [dict(curve(1), x=[20, 200, 20000])])
    cell = grid._cells[0, 0]
    plot = cell.plot_widget.getPlotItem()
    assert len(plot.listDataItems()) == 2
    assert plot.legend is None
    np.testing.assert_allclose(plot.listDataItems()[0].xData, [0, 1 / 3, 1])
    np.testing.assert_allclose(plot.listDataItems()[1].xData, [1, 4 / 3, 2])
    assert plot.getViewBox().viewRange()[0] == pytest.approx([0, 2])
    assert [text for _, text in plot.getAxis('bottom')._tickLevels[0]] == ['0.1', '1', '10'] * 2
    assert len(cell._axis_breaks) == 1
    assert plot.getAxis('left').labelText == next(iter(cell.conditions.values())).y_unit
    assert plot.getAxis('left').labelUnits == ''
    axis_gap = cell._axis_breaks[0]
    assert axis_gap.text() == ''
    assert not axis_gap.isVisible()


@pytest.mark.parametrize('gap_pixels', [12, 16, 20])
@pytest.mark.parametrize('analysis_type', ['SPL', 'FFT'])
def test_local_regions_keep_pixel_gap_on_resize(grid, configuration, qt_app, gap_pixels, analysis_type):
    data = template_layout()
    data['default_condition_display'].update(x_axis_mode='condition_local', condition_gap_px=gap_pixels)
    if analysis_type == 'FFT':
        data['plot_templates']['spl'].update(analysis_type='FFT', analysis_item_name='频谱')
        data['plot_templates']['spl']['x_axis'] = {
            'label': '频率', 'scale': 'log', 'auto_range': False, 'range_min': 20, 'range_max': 20000}
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    show(grid, configuration, qt_app)
    source = curve()
    if analysis_type == 'FFT':
        source.update(x=[20, 200, 20000], wav_channel=2, name='CH2')
    grid.set_condition_curve_data(0, 0, 'c1', [source])
    grid.set_condition_curve_data(0, 0, 'c2', [dict(source, wav_channel=1, name='CH1')])
    cell = grid._cells[0, 0]
    for width in (1600, 1100, 1900):
        grid.resize(width, 650)
        for _ in range(4):
            qt_app.processEvents()
        view = cell.plot_widget.getViewBox()
        left_end = view.mapViewToScene(pg.Point(cell._spans['c1'][1], 0)).x()
        right_start = view.mapViewToScene(pg.Point(cell._spans['c2'][0], 0)).x()
        assert right_start - left_end == pytest.approx(gap_pixels, abs=0.5)
        assert cell.separators[0].value() == pytest.approx(sum(cell._gaps()[0]) / 2)
        assert cell._axis_breaks[0].width() == pytest.approx(gap_pixels, abs=1)
        ticks = cell.plot_widget.getPlotItem().getAxis('bottom')._tickLevels[0]
        expected = ['0', '1.5', '3'] if analysis_type == 'SPL' else ['0.1', '1', '10']
        assert [text for _, text in ticks] == expected * 2
        if analysis_type == 'SPL':
            axis = cell.plot_widget.getPlotItem().getAxis('bottom')
            image = QImage(1000, 100, QImage.Format_ARGB32)
            painter = QPainter(image)
            painter.setFont(axis.style['tickFont'])
            try:
                _, _, texts = axis.generateDrawSpecs(painter)
            finally:
                painter.end()
            # Endpoint labels occupy their own region, not the gutter.
            anchors = [axis.mapFromScene(view.mapViewToScene(pg.Point(ticks[i][0], 0))).x()
                       for i in (2, 3)]
            end_rect = min((rect for rect, _, text in texts if text == expected[-1]),
                           key=lambda rect: abs(rect.right() - anchors[0]))
            start_rect = min((rect for rect, _, text in texts if text == expected[0]),
                             key=lambda rect: abs(rect.left() - anchors[1]))
            assert end_rect.right() == pytest.approx(anchors[0], abs=0.5)
            assert start_rect.left() == pytest.approx(anchors[1], abs=0.5)
            assert start_rect.left() - end_rect.right() >= gap_pixels - 0.5
        lines = cell.plot_widget.listDataItems()
        assert max(lines[0].xData) <= cell._spans['c1'][1]
        assert min(lines[1].xData) >= cell._spans['c2'][0]
        assert len(cell.findChildren(pg.PlotWidget)) == 1
    assert source['x'] == ([0, 1, 2] if analysis_type == 'SPL' else [20, 200, 20000])


def test_selected_channels_and_invalid_updates_preserve_last_good_curves(grid, configuration, qt_app):
    data = template_layout()
    data['plot_templates']['spl']['channel_display'] = {'show_all_analyzed_channels': False, 'display_wav_channels': [2]}
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    projection = show(grid, configuration, qt_app)
    assert projection.bindings[0][0][0].wav_channels == (2,)
    grid.set_condition_curve_data(0, 0, 'c1', [curve(2)])
    cell = grid._cells[0, 0]
    before = cell.plot_widget.listDataItems()[0]
    for invalid in (curve(1), curve(2, role='bad'), dict(curve(2), y=[1])):
        with pytest.raises(ValueError):
            grid.set_condition_curve_data(0, 0, 'c1', [invalid])
        assert cell.plot_widget.listDataItems()[0] is before
    assert not cell.empty_labels['c1'].isVisible()
    assert not cell.empty_labels['c2'].isVisible()
    assert len(cell.separators) == 1


def test_reconfiguration_returns_to_legacy_layout(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    grid.set_condition_curve_data(0, 0, 'c1', [curve()])
    grid.show_workbench_columns(['工位1'])
    grid.configure_plot(0, 0, kind='curve')
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    assert len(grid._cells[0, 0].plot_widget.listDataItems()) == 1
    grid.show_workbench_layout((), ())
    qt_app.processEvents()
    assert not grid._cells


def test_different_units_cannot_be_plotted_on_one_ordinate(grid, configuration, qt_app):
    queues = dict(configuration.analysis.test_queues)
    queue = queues['测试队列B']
    analysis_config = dict(queue.analysis_config)
    analysis_config['声压级'] = {**analysis_config['声压级'], 'weighting': 'C'}
    queues['测试队列B'] = replace(queue, analysis_config=analysis_config)
    configuration = replace(configuration, analysis=replace(configuration.analysis, test_queues=queues))
    projection = show(grid, configuration, qt_app)
    assert '单位不一致' in projection.cells[0][0].warning_detail
    assert all(binding is None for binding in projection.bindings[0][0])
    with pytest.raises(ValueError, match='单位不一致'):
        grid.set_condition_curve_data(0, 0, 'c1', [curve(1)])


def test_main_window_styles_do_not_enlarge_condition_or_empty_labels(grid, configuration, qt_app):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
    from consts.ui_style_const import main_window_base_style

    host = QWidget()
    host.setStyleSheet(main_window_base_style)
    QVBoxLayout(host).addWidget(grid)
    host.resize(1600, 850)
    host.show()
    try:
        show(grid, configuration, qt_app)
        cell = grid._cells[0, 0]
        assert cell.header.left.font().pixelSize() == 14
        assert not cell.header.left.font().bold()
        for label in (*cell.condition_labels.values(), *cell.empty_labels.values()):
            assert label.font().pixelSize() == 13
            assert label.font().family() == cell.header.left.font().family()
        assert cell.empty_labels['c1'].text() == ''
    finally:
        grid.setParent(None)
        host.close()
        host.deleteLater()


def test_mouse_gestures_cannot_change_overview_ranges(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    grid.set_condition_curve_data(0, 0, 'c1', [curve()])
    plot = grid._cells[0, 0].plot_widget.getPlotItem()
    view = plot.getViewBox()
    before = np.array(view.viewRange())
    for button in (Qt.LeftButton, Qt.RightButton):
        event = SimpleNamespace(
            accept=lambda: None, scenePos=lambda: pg.Point(30, 30),
            lastScenePos=lambda: pg.Point(0, 0), button=lambda: button,
            screenPos=lambda: pg.Point(30, 30), lastScreenPos=lambda: pg.Point(0, 0),
            buttonDownPos=lambda _: pg.Point(0, 0),
        )
        view.mouseDragEvent(event)
    view.wheelEvent(SimpleNamespace(accept=lambda: None, delta=lambda: 120, pos=lambda: pg.Point(50, 50)))
    np.testing.assert_allclose(view.viewRange(), before)
    assert not plot.menuEnabled()
    assert not view.menuEnabled()
    assert not plot.autoBtn.isVisible()




def test_configured_empty_y_range_shows_ticks_then_auto_ranges_and_restores(grid, configuration, qt_app):
    data = template_layout()
    data['plot_templates']['spl']['y_axis'].update(empty_data_range_min=0, empty_data_range_max=100)
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    axis = cell.plot_widget.getPlotItem().getAxis('left')
    view = cell.plot_widget.getViewBox()
    assert axis.style['showValues']
    assert axis.style['tickLength'] == -5
    assert view.viewRange()[1] == pytest.approx([0, 100])
    cell.open_detail()
    qt_app.processEvents()
    detail = cell._detail_dialog.cell
    assert detail.plot_widget.getPlotItem().getAxis('left').style['showValues']
    assert detail.plot_widget.getViewBox().viewRange()[1] == pytest.approx([0, 100])
    cell.set_condition_curve_data('c1', [curve(low=150, high=200)])
    assert view.viewRange()[1][0] > 100
    assert view.viewRange()[1][1] >= 200
    assert detail.plot_widget.getViewBox().viewRange()[1][1] >= 200
    cell.clear_plot()
    assert view.viewRange()[1] == pytest.approx([0, 100])
    assert detail.plot_widget.getViewBox().viewRange()[1] == pytest.approx([0, 100])
    assert axis.style['showValues']
    cell.set_condition_curve_data('c1', [dict(curve(), y=[None, None, None])])
    assert view.viewRange()[1] == pytest.approx([0, 100])
    cell.close_detail()


def test_auto_axis_empty_ticks_restore_when_results_arrive(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    axis = cell.plot_widget.getPlotItem().getAxis('left')
    assert not axis.style['showValues']
    assert axis.style['tickLength'] == 0
    grid.set_condition_curve_data(0, 0, 'c1', [curve()])
    assert axis.style['showValues']
    assert axis.style['tickLength'] == -5
    cell.clear_plot()
    assert not axis.style['showValues']
    assert axis.style['tickLength'] == 0


def test_empty_canvas_stays_clear_but_runtime_states_remain_visible(grid, configuration, qt_app):
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    assert all(not label.text() for label in cell.empty_labels.values())
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    for key in ('c1', 'c2'):
        grid.set_condition_curve_data(0, 0, key, [curve(1), curve(2)])
    assert len(cell.legend_bar.entries) == 2
    assert cell.plot_widget.getPlotItem().legend is None
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    grid.set_condition_curve_data(0, 0, 'c2', [])
    assert not cell.empty_labels['c2'].isVisible()
    for status in ('待采集', '采集中', '分析中', '分析失败'):
        cell.set_condition_state('c2', status)
        assert cell.empty_labels['c2'].isVisible()
        assert cell.empty_labels['c2'].text() == status
    cell.open_detail()
    qt_app.processEvents()
    assert cell._detail_dialog.cell.empty_labels['c2'].text() == '分析失败'
    cell.clear_plot()
    assert not any(label.isVisible() for label in cell.empty_labels.values())
    assert not any(label.isVisible() for label in cell._detail_dialog.cell.empty_labels.values())
    cell.close_detail()
    assert cell.legend_bar.entries == []


@pytest.mark.parametrize('channels', [[1], [1, 2]])
@pytest.mark.parametrize('legend_position', ['top_right', 'bottom_left'])
def test_result_arrival_and_clear_keep_plot_geometry(grid, configuration, qt_app, channels, legend_position):
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template['channel_display'] = {'show_all_analyzed_channels': False, 'display_wav_channels': channels}
    template['plot_style']['legend'] = {'visible': True, 'position': legend_position}
    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(layout)))
    show(grid, config, qt_app)
    cell = grid._cells[0, 0]
    cell.open_detail()
    targets = (cell, cell._detail_dialog.cell)

    def settle():
        for _ in range(6):
            qt_app.processEvents()

    def geometry(target):
        rect = target.plot_widget.getViewBox().sceneBoundingRect()
        return rect.x(), rect.y(), rect.width(), rect.height()

    settle()
    original = [geometry(target) for target in targets]
    for key in ('c1', 'c2'):
        cell.set_condition_curve_data(key, [curve(channel) for channel in channels], summaries=[
            dict(wav_channel=channel, channel=f'CH{channel}', metrics={'overall_spl': 60., 'unit': 'dBA'})
            for channel in channels
        ])
        settle()
        for target, expected in zip(targets, original):
            assert geometry(target) == pytest.approx(expected, abs=0.5)
            rect = target.plot_widget.mapFromScene(target.plot_widget.getViewBox().sceneBoundingRect()).boundingRect()
            assert target.overall_labels[key].geometry().bottom() < rect.top()
            if legend_position.startswith('top'):
                assert target.legend_bar.parentWidget() is target.header
                assert target.legend_bar.geometry().left() > target.header.left.geometry().right()

    cell.clear_plot()
    settle()
    for target, expected in zip(targets, original):
        assert geometry(target) == pytest.approx(expected, abs=0.5)
        assert not any(label.isVisible() for label in target.overall_labels.values())
    cell.close_detail()


def test_legend_overflow_exposes_all_entries_in_menu(qt_app):
    from ui.sequence.station_plot_legend import StationPlotLegend
    legend = StationPlotLegend(None)
    legend.resize(180, 24)
    entries = [(f'CH{i}', pg.mkPen('#123456', width=2)) for i in range(1, 9)]
    legend.set_entries(entries)
    legend.show()
    qt_app.processEvents()
    try:
        assert legend.more.isVisible()
        assert [action.text() for action in legend.menu.actions()] == [name for name, _ in entries]
        legend.resize(1000, 24)
        qt_app.processEvents()
        assert not legend.more.isVisible()
        legend.set_entries([])
        assert not legend.menu.actions()
    finally:
        legend.close()
        legend.deleteLater()


@pytest.mark.parametrize('channels', [[1], [1, 2]])
def test_line_rows_align_headings(grid, configuration, qt_app, channels):
    from copy import deepcopy

    layout = template_layout()
    spl = layout['plot_templates']['spl']
    spl['channel_display'] = {'show_all_analyzed_channels': False, 'display_wav_channels': channels}
    rms = deepcopy(spl)
    rms.update(title='振动 RMS', analysis_type='VIBRATION_RMS',
               data_source='simulated', value_unit='m/s²')
    rms['y_axis'] = dict(label='RMS', scale='linear', auto_range=True,
                         empty_data_range_min=0, empty_data_range_max=0.15)
    layout['plot_templates']['rms'] = rms
    layout['rows'] = 2
    layout['cells'].append([{'plot_template': 'rms'} for _ in range(4)])
    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(layout)))
    show(grid, config, qt_app)
    spl_cell, rms_cell = grid._cells[0, 0], grid._cells[1, 0]

    def settle():
        for _ in range(8):
            qt_app.processEvents()

    def geometry(cell):
        rect = cell.plot_widget.mapFromScene(cell.plot_widget.getViewBox().sceneBoundingRect()).boundingRect()
        return rect.top(), rect.height(), cell.condition_labels['c1'].y()

    settle()
    original = geometry(spl_cell)
    rms_geometry = geometry(rms_cell)
    assert (rms_geometry[0], rms_geometry[2]) == pytest.approx((original[0], original[2]), abs=0.5)
    # Qt distributes an odd available height between rows with a one-pixel difference.
    assert abs(rms_geometry[1] - original[1]) <= 1
    for cell in (spl_cell, rms_cell):
        original = geometry(cell)
        cell.set_condition_curve_data('c1', [curve(ch) for ch in channels])
        settle()
        assert geometry(cell) == pytest.approx(original, abs=0.5)
        cell.open_detail()
        settle()
        detail = cell._detail_dialog.cell
        assert detail._heading_height == cell._heading_height
        cell.clear_plot()
        settle()
        assert geometry(cell) == pytest.approx(original, abs=0.5)
        cell.close_detail()


def test_line_chart_hides_only_top_grid_line(grid, configuration, qt_app):
    from PyQt5.QtCore import QPointF

    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    cell.open_detail()
    for target in (cell, cell._detail_dialog.cell):
        axis = target.plot_widget.getPlotItem().getAxis('left')
        axis.setStyle(showValues=True)
        view = target.plot_widget.getViewBox()
        view.setYRange(0, 100, padding=0)
        axis.setTicks([[(value, str(value)) for value in range(0, 101, 20)]])
        qt_app.processEvents()
        image = QImage(800, 600, QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            _, ticks, texts = axis.generateDrawSpecs(painter)
        finally:
            painter.end()
        top = axis.mapFromScene(view.mapViewToScene(QPointF(0, 100))).y()
        assert ticks  # Internal grid lines remain.
        assert all(abs(start.y() - top) > 0.5 for _, start, _ in ticks)
        assert any(text == '100' for _, _, text in texts)
    cell.close_detail()


def test_channel_color_is_stable_when_only_second_condition_arrives_first(grid, configuration, qt_app):
    data = template_layout()
    data['default_plot_style']['channel_curve_styles'] = []
    configuration = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(data)))
    show(grid, configuration, qt_app)
    cell = grid._cells[0, 0]
    grid.set_condition_curve_data(0, 0, 'c2', [curve(2)])
    color = cell.plot_widget.listDataItems()[0].opts['pen'].color()
    grid.set_condition_curve_data(0, 0, 'c1', [curve(1), curve(2)])
    lines = cell.plot_widget.listDataItems()
    assert lines[1].opts['pen'].color() == color == lines[2].opts['pen'].color()
    assert len(cell.legend_bar.entries) == 2
