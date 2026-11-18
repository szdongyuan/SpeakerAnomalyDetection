"""Shared external legends compared with the actual main-window chart."""
from dataclasses import replace

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QDialog

from base.analysis_curve_style import resolve_curve_colors
from base.workbench_plot_config import parse_plot_templates
from ui.plotting.renderers import analysis_curve_pen
from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.sequence.record_analysis_renderers import RecordAnalysisRenderer
from ui.sequence.record_analysis_results import (
    RecordAnalysisChannelResult, RecordAnalysisItemResult, RecordAnalysisPanel,
)
from ui.sequence.station_plot_legend import StationPlotLegend
from ui.sequence.workbench_plot_display import (
    ConditionPlotDisplay, ConditionSegmentDisplay, WorkbenchPlotCellDisplay,
)
from unit_test.ui.test_shared_plot_pipeline import close_widget, station_layout


def result(kind='FFT', names=('CH1', 'far'), roles=('curve', 'baseline', 'upper', 'lower', 'exceedance'), scalar=False):
    channels = []
    for number, name in enumerate(names, 1):
        label = f'CH{number}' if name == f'CH{number}' else f'CH{number} · {name}'
        series = tuple(dict(wav_channel=number, name=label, role=role,
                            x=np.array([1., 2., 3.]), y=np.array([4., 6., 5.]),
                            limit_scope='scalar' if scalar else 'curve') for role in roles)
        panel = RecordAnalysisPanel(kind, 'bands' if kind == 'FBA' else 'curve', 'Hz', 'dB',
                                    series=series, band_labels=('100', '200', '400') if kind == 'FBA' else ())
        channels.append(RecordAnalysisChannelResult(number - 1, number, name, '分析完成', panels=(panel,)))
    return RecordAnalysisItemResult(kind, kind, tuple(channels), '分析完成')


def main_cell(mapped, parameters):
    layout = station_layout()
    layout['plot_templates']['spl']['analysis_type'] = mapped.analysis_type
    settings = parse_plot_templates(layout).cells[0][0].display_settings
    segment = ConditionSegmentDisplay('condition', '正转', duration_seconds=3,
        channels=tuple((ch.wav_channel, ch.name) for ch in mapped.channels), x_unit='Hz', y_unit='dB',
        analysis_type=mapped.analysis_type, curve_colors=tuple(resolve_curve_colors(parameters).items()),
        show_threshold_lines=parameters['display']['show_threshold_lines'])
    cell = ConditionPlotCell(WorkbenchPlotCellDisplay(title=mapped.analysis_type,
        condition_plot=ConditionPlotDisplay(settings, (segment,))))
    cell.set_condition_curve_data('condition', [series for ch in mapped.channels for series in ch.panels[0].series])
    return cell


@pytest.mark.parametrize('kind', ['SPL', 'FFT'])
@pytest.mark.parametrize('visible', [True, False])
@pytest.mark.parametrize('scalar', [True, False])
def test_legend_matches_actual_main_names_font_and_pens(qt_app, kind, visible, scalar):
    mapped = result(kind, scalar=scalar)
    parameters = {'display': {'show_threshold_lines': visible, 'main_curve_color': '#126798',
        'upper_limit_color': '#A26914', 'lower_limit_color': '#723DA1'}}
    content = RecordAnalysisRenderer().render(mapped, parameters)
    cell = main_cell(mapped, parameters)
    try:
        legend, = content.legends
        assert type(legend) is StationPlotLegend
        assert legend.entries == cell.legend_bar.entries
        assert legend.font() == cell.legend_bar.font()
        assert legend.height() == cell.legend_bar.height() == 24
        for name in ('left', 'bottom'):
            actual = content.plots[0].getAxis(name)
            main = cell.plot_widget.getAxis(name)
            assert actual.style['tickFont'] == main.style['tickFont']
            assert actual.label.font() == main.label.font()
            assert actual.pen() == main.pen()
            assert actual.textPen() == main.textPen()
        names = [name for name, _ in legend.entries]
        assert names[0] == 'CH1'
        assert 'CH2 · far' in names
        assert 'CH1 参考' in names and 'CH1 超限' in names
        if visible:
            assert ('CH1 总体上限' if scalar else 'CH1 上限') in names
            assert ('CH1 总体下限' if scalar else 'CH1 下限') in names
        else:
            assert all('上限' not in name and '下限' not in name for name in names)
        assert all(plot.plotItem.legend is None for plot in content.plots)
        for (_, pen), curve in zip(legend.entries, content.curves):
            assert pen == curve.opts['pen']
    finally:
        content.dispose()
        close_widget(content)
        cell.dispose()
        close_widget(cell)


def test_external_row_full_width_overflow_restores_without_covering_canvas(qt_app):
    content = RecordAnalysisRenderer().render(result(names=('long channel name ' * 3, 'second channel')), {})
    try:
        content.resize(400, 600)
        content.show()
        qt_app.processEvents()
        legend, = content.legends
        plot, = content.plots
        assert not content.title_labels
        assert not plot.plotItem.titleLabel.text
        assert legend.geometry().bottom() < plot.geometry().top()
        assert content.content_layout.indexOf(legend) + 1 == content.content_layout.indexOf(plot)
        assert legend.align_right
        assert legend.width() == plot.width()
        assert legend.height() == 24
        assert legend.more.isVisible()
        assert [action.text() for action in legend.menu.actions()] == [name for name, _ in legend.entries]
        legend.menu.popup(legend.mapToGlobal(legend.rect().bottomRight()))
        qt_app.processEvents()
        assert legend.menu.isVisible()
        legend.menu.hide()
        content.resize(4000, 600)
        qt_app.processEvents()
        assert not legend.more.isVisible()
        assert legend._visible_count == len(legend.entries)
        assert legend.height() == 24
        # The shared legend paints the fitted entries at the right edge.
        pixels = legend.grab().toImage()
        background = pixels.pixelColor(0, 0)
        assert all(pixels.pixelColor(x, y) == background for x in range(10, 100)
                   for y in range(5, 20))
        assert any(pixels.pixelColor(x, 12) != background
                   for x in range(legend.width() - 200, legend.width() - 10))
    finally:
        content.dispose()
        close_widget(content)


@pytest.mark.parametrize('visible', [True, False])
def test_fba_legend_includes_actual_visible_roles_and_shared_bar_pens(qt_app, visible):
    mapped = result('FBA', roles=('curve', 'upper', 'lower', 'exceedance'))
    parameters = {'display': {'show_threshold_lines': visible}}
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        legend, = content.legends
        names = ['CH1', 'CH1 上限', 'CH1 下限', 'CH1 超限', 'CH2 · far', 'CH2 上限', 'CH2 下限', 'CH2 超限']
        if not visible:
            names = [name for name in names if '上限' not in name and '下限' not in name]
        assert [name for name, _ in legend.entries] == names
        colors = resolve_curve_colors(parameters)
        roles = ['curve', 'upper', 'lower', 'exceedance'] if visible else ['curve', 'exceedance']
        assert [pen for _, pen in legend.entries] == [analysis_curve_pen(role, colors) for _ in range(2) for role in roles]
        assert all(plot.plotItem.legend is None for plot in content.plots)
    finally:
        content.dispose()
        close_widget(content)


def test_legend_deduplicates_channel_role_and_pen(qt_app):
    mapped = result(names=('CH1',))
    channel, = mapped.channels
    panel, = channel.panels
    mapped = replace(mapped, channels=(replace(channel, panels=(replace(panel, series=panel.series * 2),)),))
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert len(content.curves) == 10
        assert len(content.legends[0].entries) == 5
    finally:
        content.dispose()
        close_widget(content)


def test_loud_panels_have_independent_legends_and_heatmap_only_colorbar(qt_app):
    mapped = result('LOUD', roles=('curve',))
    channels = []
    for channel in mapped.channels:
        panel, = channel.panels
        matrix = RecordAnalysisPanel('specific_loudness', 'heatmap', 's', 'Bark', value_unit='sone/Bark',
            heatmap={'x': np.array([1., 2.]), 'y': np.array([1., 2.]), 'z': np.ones((2, 2))})
        channels.append(replace(channel, panels=(replace(panel, key='loudness_time'),
            replace(panel, key='specific_loudness_profile'), matrix)))
    content = RecordAnalysisRenderer().render(replace(mapped, channels=tuple(channels)), {})
    try:
        assert len(content.plots) == 3 and len(content.legends) == 2 and len(content.colorbars) == 1
        assert content.legends[0] is not content.legends[1]
        assert content.legends[0].entries == content.legends[1].entries
        assert all(plot.plotItem.legend is None for plot in content.plots)
    finally:
        content.dispose()
        close_widget(content)


@pytest.mark.parametrize('release', ['switch', 'close', 'parent'])
def test_legend_entries_menus_and_references_released_immediately(qt_app, release):
    parent = QDialog()
    dialog = RecordAnalysisDialog(parent)
    content = RecordAnalysisRenderer().render(result(), {})
    dialog.set_content(content)
    legends, titles = list(content.legends), list(content.title_labels)
    cleared = []
    # This callback runs before Qt destroys each legend, so no stale-wrapper reads.
    for legend in legends:
        legend.destroyed.connect(lambda _, legend=legend: cleared.append((legend.entries, content.legends[:], content.title_labels[:])))
    if release == 'switch':
        dialog.set_content(RecordAnalysisRenderer().render(result('SPL'), {}))
    elif release == 'close':
        dialog.close()
    else:
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert content.disposed and not content.legends and not content.title_labels
    if release != 'parent':
        assert all(not legend.entries and not legend.menu.actions() for legend in legends)
        assert all(title.isHidden() for title in titles)
        dialog.close()
        parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert cleared and all(entry == ([], [], []) for entry in cleared)
