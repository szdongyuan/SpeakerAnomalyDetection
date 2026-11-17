from dataclasses import replace

import pytest

from base.workbench_plot_config import parse_plot_templates
from ui.sequence.workbench_plot_presenter import WorkbenchPlotPresenter
from unit_test.base.workbench_plot_fixtures import template_layout
from unit_test.ui.test_condition_plot_layout import configuration, grid, show


def demo_layout():
    layout = template_layout()
    template = layout['plot_templates']['spl']
    template.update(title='振动 RMS', analysis_type='VIBRATION_RMS', value_unit='m/s²',
                    data_source='simulated', channel_display={
                        'show_all_analyzed_channels': False, 'display_wav_channels': [1, 2, 3]})
    return layout


def test_three_simulated_channels_fit_in_separate_legend_row(
        configuration, grid, qt_app, main_ui_font):
    from ui.sequence.workbench_plot_demo import simulated_rms_series

    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(demo_layout())))
    grid.resize(1528, 650)
    show(grid, config, qt_app)
    cell = grid._cells[0, 0]
    first = next(iter(cell.conditions))
    cell.set_condition_curve_data(first, simulated_rms_series(
        cell.conditions[first], cell._settings, 0, 0))
    cell.open_detail()
    for _ in range(6):
        qt_app.processEvents()
    for target in (cell, cell._detail_dialog.cell):
        legend = target.legend_bar
        assert [name for name, _ in legend.entries] == ['X', 'Y', 'Z']
        assert legend._visible_count == 3
        assert not legend.more.isVisible()
        assert legend.geometry().top() > target.header.geometry().bottom()
        assert legend.width() == target.plot_widget.width()
    assert cell.expand_button.parentWidget() is cell.header
    cell.close_detail()


def test_demo_needs_no_analysis_item_or_hardware_and_cannot_receive_real_results(configuration, grid, qt_app):
    queues_before = dict(configuration.analysis.test_queues)
    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(demo_layout())))
    projection = show(grid, config, qt_app)
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    cell = grid._cells[0, 0]
    assert '模拟数据' in cell.display.title
    assert not cell.display.warning
    assert all(binding is None for row in projection.bindings for bindings in row for binding in bindings)
    assert not presenter._targets
    assert configuration.analysis.test_queues == queues_before
    first = next(iter(cell.conditions))
    assert not cell._condition_series
    presenter.analysis_event(dict(is_current=True, event_type='task_result', station_id='s1',
                                  condition_id=first, task_id='real', payload={}))
    assert not cell._condition_series
    assert not cell._condition_summaries
    cell.open_detail()
    qt_app.processEvents()
    assert '模拟数据' in cell._detail_dialog.cell.display.title
    assert not cell._detail_dialog.cell._condition_series
    cell.close_detail()


@pytest.mark.parametrize('change', [dict(data_source='unknown'), dict(analysis_type='SPL'),
                                  dict(channel_display={'show_all_analyzed_channels': True})])
def test_simulation_is_explicit_and_limited_to_rms(change):
    layout = demo_layout()
    layout['plot_templates']['spl'].update(change)
    with pytest.raises(ValueError):
        parse_plot_templates(layout)


def test_omitted_data_source_keeps_real_binding_validation(configuration, grid, qt_app):
    layout = demo_layout()
    del layout['plot_templates']['spl']['data_source']
    config = replace(configuration, ui=replace(configuration.ui, layout=parse_plot_templates(layout)))
    projection = show(grid, config, qt_app)
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    assert grid._cells[0, 0].display.warning == '配置不匹配'
    assert not grid._cells[0, 0]._condition_series
