"""Compare actual GUI surfaces and forward spies through public drawing APIs."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from base.analysis_curve_style import resolve_curve_colors, show_threshold_lines
from base.record_analysis_request import RecordAnalysisItem
from base.workbench_plot_config import parse_plot_templates
from ui.plotting import heatmap, renderers
from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.record_analysis_renderers import RecordAnalysisRenderer
from ui.sequence.record_analysis_results import build_record_analysis_results
from ui.sequence.spectrogram_data import bin_edges
from ui.sequence.workbench_plot_display import (
    ConditionPlotDisplay, ConditionSegmentDisplay, WorkbenchPlotCellDisplay,
)
from ui.sequence.workbench_plot_results import build_condition_result
from unit_test.ui.test_record_analysis_results import real_results
from unit_test.ui.test_shared_plot_pipeline import close_widget, station_layout


def main_cell(kind, parameters, panel, source):
    layout = station_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type=kind)
    template['x_axis'] = dict(label=panel.x_unit, auto_range=True, scale='linear')
    settings = parse_plot_templates(layout).cells[0][0].display_settings
    segment = ConditionSegmentDisplay(
        'condition', '正转', duration_seconds=.4, channels=((1, 'near'), (2, 'far')),
        x_unit=panel.x_unit, y_unit=panel.y_unit, analysis_type=kind,
        curve_colors=tuple(resolve_curve_colors(parameters).items()),
        show_threshold_lines=show_threshold_lines(parameters))
    mapped = build_condition_result(SimpleNamespace(analysis_item=kind, analysis_type=kind),
                                    segment, source)
    assert not mapped.status, mapped.detail
    cell = ConditionPlotCell(WorkbenchPlotCellDisplay(title=kind, condition_plot=ConditionPlotDisplay(
        settings, (segment,))))
    cell.set_condition_curve_data(segment.condition_id, mapped.series)
    roles = [series['role'] for series in mapped.series
             if show_threshold_lines(parameters) or series['role'] not in ('upper', 'lower')]
    return cell, roles


@pytest.mark.parametrize('display', [
    {},
    {'main_curve_color': '#126798', 'upper_limit_color': '#A26914', 'lower_limit_color': '#723DA1'},
    {'main_curve_color': 'invalid', 'upper_limit_color': None, 'lower_limit_color': '#12'},
], ids=['default', 'custom', 'invalid-fallback'])
@pytest.mark.parametrize('visible', [True, False])
@pytest.mark.parametrize('kind', ['SPL', 'FFT', 'LOUD'])
def test_view_pens_match_actual_main_gui(qt_app, real_results, kind, display, visible):
    _, items, original = real_results
    source = deepcopy(original)
    # The FFT overlay and lower limits exercise roles absent from the WAV fixture.
    for instance in source['instance_results']:
        if instance['config_key'] == kind:
            payload = instance['display_payload']
            payload['lower'] = np.full(len(payload['x']), -80.)
            if payload.get('plot_curve'):
                payload['plot_curve']['lower'] = np.full(len(payload['plot_curve']['x']), -80.)
            if kind == 'FFT':
                payload['baseline'] = np.asarray(payload['y']) - 10.
                instance['metrics']['display_mode'] = 'overlay'
    item = next(item for item in items if item.config_key == kind)
    parameters = dict(item.parameters, display=dict(item.parameters.get('display', {}),
                      **display, show_threshold_lines=visible))
    mapped = build_record_analysis_results((replace(item, parameters=parameters),), source,
                                            channel_names={0: 'near', 1: 'far'})[kind]
    before = deepcopy(source)
    content = RecordAnalysisRenderer().render(mapped, parameters)
    panel = mapped.channels[0].panels[0]
    cell, roles = main_cell(kind, parameters, panel, source)
    try:
        view = [curve for curve in content.curves if curve in content.plots[0].listDataItems()]
        assert [curve.record_role for curve in view] == roles
        main = cell.plot_widget.listDataItems()
        assert len(main) == len(view)
        for expected, actual in zip(main, view):
            assert actual.opts['pen'] == expected.opts['pen']
        mains = [curve for curve in view if curve.record_role == 'curve']
        assert [curve.name() for curve in mains] == ['CH1 · near', 'CH2 · far']
        assert mains[0].opts['pen'] == mains[1].opts['pen']
        assert any(curve.record_role == 'exceedance' for curve in view)
        if kind == 'FFT':
            assert any(curve.record_role == 'baseline' for curve in view)
            assert next(curve for curve in view if curve.record_role == 'exceedance').opts['pen'].widthF() == 3
        assert any(role == 'upper' for role in roles) == visible
        np.testing.assert_equal(source, before)
    finally:
        content.dispose()
        close_widget(content)
        cell.dispose()
        close_widget(cell)


@pytest.mark.parametrize('kind', ['SPL', 'LOUD'])
def test_hidden_scalar_reference_lines_keep_data_and_judgement(qt_app, kind):
    payload = dict(kind='curve', x=[0., .2, .4], y=[20., 80., 20.],
                   upper=[], lower=[], x_unit='s', y_unit='sone' if kind == 'LOUD' else 'dB')
    if kind == 'LOUD':
        payload['loudness_display'] = dict(curves=[dict(payload, key='loudness_time')],
                                           heatmaps=[], summary_cards=[])
    source = dict(instance_results=[dict(config_key=kind, analysis_type=kind, channel_index=0,
        execution_status='分析完成', judgement_status='completed', judgement='NG',
        metrics=dict(unit=payload['y_unit'], overall_upper_limit=40., overall_lower_limit=0.),
        display_payload=payload, violations=[dict(x_start=None, x_end=None)])])
    item = RecordAnalysisItem(kind, kind, (0,), {'display': {'show_threshold_lines': False}})
    mapped = build_record_analysis_results((item,), source)[kind]
    assert {s['role'] for s in mapped.channels[0].panels[0].series} == {'curve', 'upper', 'lower'}
    content = RecordAnalysisRenderer().render(mapped, item.parameters)
    try:
        assert [curve.record_role for curve in content.curves] == ['curve']
        assert mapped.channels[0].judgement == 'NG'
        np.testing.assert_equal(content.curves[0].full_y, payload['y'])
    finally:
        content.dispose()
        close_widget(content)


def forwarding_spy(monkeypatch, module, name):
    calls = []
    original = getattr(module, name)

    def draw(*args, **kwargs):
        item = original(*args, **kwargs)
        calls.append((args, kwargs, item))
        return item

    monkeypatch.setattr(module, name, draw)
    return calls


@pytest.mark.parametrize('kind', ['SPL', 'FFT', 'LOUD'])
def test_real_curves_delegate_and_retain_complete_data(qt_app, real_results, monkeypatch, kind):
    _, items, source = real_results
    mapped = build_record_analysis_results(items, source)[kind]
    before = deepcopy(source)
    calls = forwarding_spy(monkeypatch, renderers, 'draw_curve')
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        expected = [s for first in mapped.channels[0].panels if first.kind == 'curve'
                    for channel in mapped.channels for panel in channel.panels
                    if panel.key == first.key for s in panel.series]
        assert len(calls) == len(expected) == len(content.curves)
        for series, curve in zip(expected, content.curves):
            assert curve in [call[2] for call in calls]
            np.testing.assert_equal(curve.full_x, series['x'])
            np.testing.assert_equal(curve.full_y, series['y'])
            assert curve.opts['connect'] == 'finite'
        assert all(call[1]['sampling'] is not None for call in calls)
        np.testing.assert_equal(source, before)
        curves = list(content.curves)
    finally:
        content.dispose()
        close_widget(content)
    assert all(curve.full_x.size == curve.full_y.size == 0 for curve in curves)


@pytest.mark.parametrize('visible', [True, False])
def test_real_bands_delegate_values_markers_and_annotations(qt_app, real_results, monkeypatch, visible):
    _, items, source = real_results
    mapped = build_record_analysis_results(items, source)['FBA']
    before = deepcopy(source)
    bars = forwarding_spy(monkeypatch, renderers, 'draw_bars')
    curves = forwarding_spy(monkeypatch, renderers, 'draw_curve')
    texts = forwarding_spy(monkeypatch, renderers, 'draw_text')
    parameters = {'display': dict(main_curve_color='#126798', upper_limit_color='#A26914',
                                 lower_limit_color='#723DA1', show_threshold_lines=visible)}
    content = RecordAnalysisRenderer().render(mapped, parameters)
    colors = resolve_curve_colors(parameters)
    try:
        assert [call[2] for call in bars] == content.bars
        assert len(bars) == 2
        assert len(curves) == len(content.curves) == (2 if visible else 0)
        assert [call[2] for call in texts] == content.annotations
        assert texts
        for index, (channel, bar) in enumerate(zip(mapped.channels, content.bars)):
            panel, = channel.panels
            series = {s['role']: s for s in panel.series}
            main = series['curve']
            positions = np.arange(len(panel.band_labels)) - .35 + .35 * (index + .5)
            np.testing.assert_allclose(bar.opts['x'], positions)
            np.testing.assert_equal(bar.opts['height'], main['y'])
            assert bar.opts['width'] == .35
            flags = np.isfinite(series['exceedance']['y'])
            for brush, flag in zip(bar.opts['brushes'], flags):
                assert brush.color() == renderers.analysis_curve_pen('exceedance' if flag else 'curve', colors).color()
            annotations = [text for text in content.annotations if text.record_channel == channel.wav_channel]
            np.testing.assert_allclose([text.pos().x() for text in annotations], positions[flags])
            np.testing.assert_equal([text.pos().y() for text in annotations], main['y'][flags])
            assert all(text.toPlainText() == 'NG' for text in annotations)
        for _, _, curve in curves:
            assert curve.opts['pen'] == renderers.analysis_curve_pen(curve.record_role, colors)
            assert curve.opts['symbol'] == 'o'
        np.testing.assert_equal(source, before)
    finally:
        content.dispose()
        close_widget(content)


@pytest.mark.parametrize('kind,scale', [('Spec', 'linear'), ('Spec', 'log'), ('LOUD', 'linear')])
def test_real_heatmaps_delegate_matrix_edges_levels_and_units(qt_app, real_results, monkeypatch, kind, scale):
    _, items, source = real_results
    mapped = build_record_analysis_results(items, source)[kind]
    panel = next(panel for panel in mapped.channels[0].panels if panel.kind == 'heatmap')
    mapped = replace(mapped, channels=(replace(mapped.channels[0], panels=(panel,)),))
    data = panel.heatmap
    before = deepcopy(source)
    calls = forwarding_spy(monkeypatch, heatmap, 'draw_binned_image')
    parameters = dict(freq_scale_type=scale, custom_limit=True, bottom_limit=40., top_limit=80.)
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        assert len(calls) == 1
        args, _, image = calls[0]
        assert content.images == [image]
        x_edges, y_edges = bin_edges(data['x']), bin_edges(data['y'])
        z = data['z']
        if kind == 'Spec' and scale == 'log':
            first = int(data['y'][0] == 0)
            y_edges, z = y_edges[first:], z[first:]
            y_edges[0] = max(data['y'][first] / 2, y_edges[0])
            y_edges = np.log10(y_edges)
        elif kind == 'Spec':
            y_edges[0] = max(0, y_edges[0])
        np.testing.assert_equal(args[1], z)
        np.testing.assert_equal(image.values, z)
        np.testing.assert_equal(image.x_edges, x_edges)
        np.testing.assert_equal(image.y_edges, y_edges)
        assert image.levels == ((20., 100.) if kind == 'Spec' else (40., 80.))
        assert content.colorbars[0].axis.labelText == (panel.value_label or panel.value_unit)
        assert content.plots[0].record_y_unit == panel.y_unit
        assert args[-1] == (x_edges[0], x_edges[-1])
        np.testing.assert_equal(source, before)
    finally:
        content.dispose()
        close_widget(content)
