"""Render actual mapped backend results with native graphics."""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from base.analysis.contracts import AnalysisInstance
from base.analysis.executor import execute_task
from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.record_analysis_request import RecordAnalysisItem
from ui.sequence.record_analysis_renderers import RecordAnalysisRenderer
from ui.sequence.record_analysis_results import build_record_analysis_results
from unit_test.ui.test_record_analysis_results import real_results


@pytest.mark.parametrize('kind', ['SPL', 'FFT'])
def test_real_multichannel_curves_limits_and_disposal(qt_app, real_results, kind):
    _, items, result = real_results
    item = next(value for value in items if value.config_key == kind)
    mapped = build_record_analysis_results(items, result, channel_names={0: 'near', 1: 'far'})[kind]
    content = RecordAnalysisRenderer().render(mapped, item.parameters)
    assert len(content.plots) == 1
    curves = [curve for curve in content.curves if curve.record_role == 'curve']
    assert [curve.name() for curve in curves] == ['CH1 · near', 'CH2 · far']
    assert len(content.plots[0].plotItem.legend.items) == 2
    assert any(curve.record_role == 'upper' for curve in content.curves)
    assert any(curve.record_role == 'exceedance' for curve in content.curves)
    for curve, channel in zip(curves, mapped.channels):
        np.testing.assert_array_equal(curve.full_x, channel.panels[0].series[0]['x'])
    content.show()
    qt_app.processEvents()
    assert content.plots[0].getViewBox().height() >= 120
    content.dispose()
    assert not content.curves and not content.images and not content.plots
    assert all(curve.full_x.size == 0 for curve in curves)


def test_spec_native_nonuniform_bins_colorbar_and_log_dc(qt_app, real_results):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['Spec']
    channel = mapped.channels[0]
    # Use the real matrix with nonuniform physical frequency rows.
    panel = channel.panels[0]
    data = dict(panel.heatmap, y=np.geomspace(20, 20000, len(panel.heatmap['y'])))
    data['y'][0] = 0
    channel = replace(channel, panels=(replace(panel, heatmap=data),))
    mapped = replace(mapped, channels=(channel, mapped.channels[1]))
    original = data['z'].copy()
    content = RecordAnalysisRenderer().render(mapped, {'freq_scale_type': 'log'}, channel_index=0)
    image, = content.images
    assert not image._uniform
    assert image.image.height() == len(data['y']) - 1
    assert len(content.colorbars) == 1
    assert content.colorbars[0].axis.labelText == panel.value_unit
    assert content.plots[0].plotItem.getAxis('left').scale == .001
    np.testing.assert_array_equal(data['z'], original)
    content.dispose()


def test_loud_selected_composite_uses_independent_axes_and_metadata(qt_app, real_results):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['LOUD']
    content = RecordAnalysisRenderer().render(mapped, {}, channel_index=1)
    assert [plot.record_panel_key for plot in content.plots] == [
        'loudness_time', 'specific_loudness_profile', 'specific_loudness']
    assert [(plot.record_x_unit, plot.record_y_unit) for plot in content.plots] == [
        ('s', 'phon'), ('Bark', 'sone/Bark'), ('s', 'Bark')]
    assert len(content.images) == 1
    assert len(content.summary_labels) == 2
    assert all('本次判定' in label.text() for label in content.summary_labels)
    assert all(card['unit'] in label.text() for card, label in zip(
        [card for ch in mapped.channels for card in ch.summary_cards], content.card_labels))
    assert content.plots[0].record_metadata['zero_based'] is True
    assert content.plots[2].record_metadata['color_map'] == 'viridis'
    assert content.plots[0].viewRange()[1][0] == 0
    # A selected display without a main time curve must remain without it.
    channels = tuple(replace(ch, panels=ch.panels[1:]) for ch in mapped.channels)
    profile_only = RecordAnalysisRenderer().render(replace(mapped, channels=channels), {}, channel_index=0)
    assert [plot.record_panel_key for plot in profile_only.plots] == [
        'specific_loudness_profile', 'specific_loudness']
    content.dispose()
    profile_only.dispose()


def test_axis_pair_grouping_never_overlays_different_units(qt_app, real_results):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['FFT']
    second = mapped.channels[1]
    mapped = replace(mapped, channels=(mapped.channels[0], replace(second, panels=(
        replace(second.panels[0], y_unit='Pa'),))))
    content = RecordAnalysisRenderer().render(mapped, {})
    assert len(content.plots) == 2
    content.dispose()


def test_fft_configured_log_axis_and_delta_unit_are_preserved(qt_app, real_results):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['FFT']
    channels = tuple(replace(ch, metrics=dict(ch.metrics, unit='dB', display_mode='delta'),
                             panels=(replace(ch.panels[0], y_unit='dB'),)) for ch in mapped.channels)
    content = RecordAnalysisRenderer().render(replace(mapped, channels=channels), {'x_axis_scale': 'log'})
    assert content.plots[0].plotItem.getAxis('bottom').logMode
    assert '幅值单位：dB' in content.summary_labels[0].text()
    assert 'dB(A)' not in content.summary_labels[0].text()
    assert 'Hz' in content.summary_labels[0].text()
    content.dispose()


def test_loud_cards_only_and_bad_color_map_have_explicit_content(qt_app, real_results):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['LOUD']
    cards_only = replace(mapped, channels=tuple(replace(ch, panels=()) for ch in mapped.channels))
    content = RecordAnalysisRenderer().render(cards_only, {})
    assert not content.plots and len(content.card_labels) == 4
    content.dispose()
    selected = mapped.channels[0]
    panel = replace(selected.panels[-1], metadata={'color_map': 'not-a-local-map'})
    broken = replace(mapped, channels=(replace(selected, panels=(panel,)),))
    content = RecordAnalysisRenderer().render(broken, {})
    assert not content.plots
    assert 'not-a-local-map' in content.diagnostic_text()
    content.dispose()


@pytest.mark.parametrize('focus_enabled', [False, True])
@pytest.mark.parametrize('axis_settings, expected_log', [
    ({}, True),
    ({'x_axis_scale': None}, True),
    ({'x_axis_scale': ''}, True),
    ({'x_axis_scale': 'unknown'}, True),
    ({'x_axis_scale': 'LOG'}, True),
    ({'x_axis_scale': 'LINEAR'}, False),
    ({'x_axis_scale': 'log'}, True),
    ({'x_axis_scale': 'linear'}, False),
])
def test_real_fft_supported_axis_defaults_match_calculation(qt_app, real_results, axis_settings, expected_log, focus_enabled):
    task, _, _ = real_results
    parameters = dict(axis_settings, focus_range_enabled=focus_enabled)
    item = RecordAnalysisItem('normalized-fft', 'FFT', (0,), parameters)
    task = replace(task, instances=(AnalysisInstance(item.config_key, 'FFT', 0, parameters),))
    result = execute_task(task)
    mapped = build_record_analysis_results((item,), result)[item.config_key]
    assert mapped.status == '分析完成', mapped.diagnostic
    source_x = mapped.channels[0].panels[0].series[0]['x']
    if focus_enabled:
        assert source_x[0] >= 100
    else:
        assert (source_x[0] > 0) == expected_log
    content = RecordAnalysisRenderer().render(mapped, item.parameters)
    try:
        assert content.plots[0].plotItem.getAxis('bottom').logMode == expected_log
        np.testing.assert_array_equal(content.curves[0].full_x, source_x)
    finally:
        content.dispose()


@pytest.mark.parametrize('kind', ['SPL', 'FBA', 'LOUD'])
def test_fft_default_does_not_change_other_panel_axes(qt_app, real_results, kind):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)[kind]
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert all(not plot.plotItem.getAxis('bottom').logMode for plot in content.plots)
    finally:
        content.dispose()


@pytest.mark.parametrize('limit_settings', [
    {},
    {'bottom_limit': 55},
    {'top_limit': 80},
    {'bottom_limit': 50, 'top_limit': 70},
    {'bottom_limit': '45.5', 'top_limit': '82.5'},
    {'bottom_limit': 40.5, 'top_limit': 90.5},
])
def test_real_spec_supported_color_defaults_match_calculation(qt_app, real_results, limit_settings):
    task, _, _ = real_results
    parameters = dict(limit_settings, custom_limit=True)
    item = RecordAnalysisItem('normalized-spec', 'Spec', (0,), parameters)
    task = replace(task, instances=(AnalysisInstance(item.config_key, 'Spec', 0, parameters),))
    samples, sample_rate = sf.read(task.wav_path, always_2d=True)
    calculation = calculate_analysis_instance(
        'Spec', samples[:, 0], sample_rate, parameters, task.channels[0].v2pa_factor,
        source='station_analysis', sequence_snapshot={}, evaluate_limits=False, full_spectrogram=True)
    result = execute_task(task)
    mapped = build_record_analysis_results((item,), result)[item.config_key]
    assert mapped.status == '分析完成', mapped.diagnostic
    matrix = mapped.channels[0].panels[0].heatmap['z']
    original = matrix.copy()
    content = RecordAnalysisRenderer().render(mapped, item.parameters)
    try:
        image, = content.images
        assert image.levels == calculation['plot']['levels']
        np.testing.assert_array_equal(matrix, original)
        assert not content.diagnostic_text()
    finally:
        content.dispose()


@pytest.mark.parametrize('limit_settings', [
    {'bottom_limit': 'not-a-number'},
    {'top_limit': None},
    {'bottom_limit': float('nan')},
    {'bottom_limit': 90, 'top_limit': 60},
])
def test_spec_invalid_color_settings_remain_diagnostic(qt_app, real_results, limit_settings):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['Spec']
    content = RecordAnalysisRenderer().render(mapped, dict(limit_settings, custom_limit=True))
    try:
        assert not content.plots
        assert 'Spec 内容不可用' in content.diagnostic_text()
        assert content.summary_labels
    finally:
        content.dispose()


@pytest.mark.parametrize('color_settings', [
    pytest.param({}, id='map-omitted'),
    pytest.param({'color_map': None}, id='map-none'),
    pytest.param({'color_map': ''}, id='map-empty'),
    pytest.param({'color_map': 'viridis'}, id='map-viridis'),
    pytest.param({'color_map': 'magma'}, id='map-magma'),
    pytest.param({'color_map': 'unknown-colormap'}, id='map-invalid'),
])
@pytest.mark.parametrize('axis_settings', [
    pytest.param({}, id='scale-omitted'),
    pytest.param({'freq_scale_type': None}, id='scale-none'),
    pytest.param({'freq_scale_type': ''}, id='scale-empty'),
    pytest.param({'freq_scale_type': 'linear'}, id='scale-linear'),
    pytest.param({'freq_scale_type': 'LINEAR'}, id='scale-upper-linear'),
    pytest.param({'freq_scale_type': 'log'}, id='scale-log'),
    pytest.param({'freq_scale_type': 'LOG'}, id='scale-upper-log'),
    pytest.param({'freq_scale_type': 'Log'}, id='scale-mixed-log'),
    pytest.param({'freq_scale_type': 'unknown'}, id='scale-unknown'),
])
@pytest.mark.parametrize('limit_settings', [
    pytest.param({'custom_limit': False}, id='auto-colors'),
    pytest.param({'custom_limit': True, 'bottom_limit': '50', 'top_limit': '70'}, id='fixed-colors'),
])
def test_real_spec_display_setting_combinations_match_calculation(
        qt_app, real_results, color_settings, axis_settings, limit_settings):
    task, _, _ = real_results
    parameters = dict(color_settings, **axis_settings, **limit_settings)
    item = RecordAnalysisItem('display-spec', 'Spec', (0,), parameters)
    task = replace(task, instances=(AnalysisInstance(item.config_key, 'Spec', 0, parameters),))
    samples, sample_rate = sf.read(task.wav_path, always_2d=True)
    calculation = calculate_analysis_instance(
        'Spec', samples[:, 0], sample_rate, parameters, task.channels[0].v2pa_factor,
        source='station_analysis', sequence_snapshot={}, evaluate_limits=False, full_spectrogram=True)
    result = execute_task(task)
    mapped = build_record_analysis_results((item,), result)[item.config_key]
    assert mapped.status == '分析完成', mapped.diagnostic
    matrix = mapped.channels[0].panels[0].heatmap['z']
    original = matrix.copy()
    content = RecordAnalysisRenderer().render(mapped, item.parameters)
    try:
        if color_settings.get('color_map') == 'unknown-colormap':
            assert not content.plots
            assert 'unknown-colormap' in content.diagnostic_text()
        else:
            image, = content.images
            expected_log = mapped.channels[0].metrics['transform'] == 'CQT'
            assert content.plots[0].plotItem.getAxis('left').logMode == expected_log
            assert content.colorbars[0].colorMap().name == calculation['plot']['color_map']
            assert content.plots[0].record_y_unit == 'Hz'
            expected_levels = calculation['plot']['levels']
            if expected_levels is None:
                finite = matrix[np.isfinite(matrix)]
                expected_levels = (float(finite.min()), float(finite.max()))
            assert image.levels == expected_levels
            assert not content.diagnostic_text()
        np.testing.assert_array_equal(matrix, original)
    finally:
        content.dispose()


@pytest.mark.parametrize('metadata_color', ['magma', None, ''])
def test_spec_normalization_preserves_loud_metadata_and_bark_axes(qt_app, real_results, metadata_color):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['LOUD']
    selected = mapped.channels[0]
    heatmap = replace(selected.panels[-1], metadata={'color_map': metadata_color})
    mapped = replace(mapped, channels=(replace(selected, panels=(heatmap,)),))
    content = RecordAnalysisRenderer().render(mapped, {'color_map': 'viridis', 'freq_scale_type': 'LOG'})
    try:
        if metadata_color == 'magma':
            assert content.colorbars[0].colorMap().name == 'magma'
            assert not content.plots[0].plotItem.getAxis('left').logMode
            assert content.plots[0].record_y_unit == 'Bark'
        else:
            assert not content.plots
            assert '内容不可用' in content.diagnostic_text()
    finally:
        content.dispose()


@pytest.mark.parametrize('kind', ['SPL', 'FFT', 'FBA', 'Spec', 'LOUD'])
def test_human_summary_keeps_measurements_without_backend_metadata(qt_app, real_results, kind):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)[kind]
    metrics = dict(mapped.channels[0].metrics)
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        text = '\n'.join(label.text() for label in content.summary_labels + content.card_labels)
        for internal in ('sample_rate', 'wav_duration_sec', 'analysis_start_time_sec',
                         'analysis_end_time_sec', 'v2pa_factor', 'overall_upper_limit',
                         'overall_lower_limit', 'peak_value', 'peak_frequency_hz'):
            assert internal not in text
        assert 'CH1' in text and 'CH2' in text and '本次判定' in text
        if kind == 'SPL':
            assert '总体声压级' in text
        if kind == 'FFT':
            assert '峰值频率' in text and 'Hz' in text
        if kind == 'LOUD':
            assert all(card['label'] in text for channel in mapped.channels for card in channel.summary_cards)
        assert mapped.channels[0].metrics == metrics
    finally:
        content.dispose()


def test_loud_titles_are_readable_while_internal_panel_identity_is_preserved(qt_app, real_results):
    _, items, result = real_results
    content = RecordAnalysisRenderer().render(build_record_analysis_results(items, result)['LOUD'], {})
    try:
        assert [plot.record_panel_key for plot in content.plots] == [
            'loudness_time', 'specific_loudness_profile', 'specific_loudness']
        titles = [plot.plotItem.titleLabel.text for plot in content.plots]
        assert 'specific_loudness_profile' not in titles
        assert 'loudness_time' not in titles
        assert any('特征响度' in title for title in titles)
    finally:
        content.dispose()


def test_default_stereo_single_plot_axes_fit_initial_viewport(qt_app, real_results):
    from PyQt5.QtCore import QCoreApplication, QEvent, QRect
    from PyQt5.QtTest import QTest
    from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
    from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter

    _, items, result = real_results
    dialog = RecordAnalysisDialog()
    presenter = RecordAnalysisPresenter(dialog)
    try:
        presenter.set_items(items)
        presenter.show_result(build_record_analysis_results(items, result))
        dialog.resize(960, 680)
        dialog.show()
        QTest.qWait(30)
        viewport = dialog.content_scroll.viewport()
        plot, = dialog.content.plots
        rect = QRect(plot.mapTo(viewport, plot.rect().topLeft()), plot.size())
        assert viewport.rect().contains(rect), (viewport.rect(), rect)
        assert dialog.content_scroll.verticalScrollBar().value() == 0
        assert dialog.content_scroll.verticalScrollBar().maximum() == 0
        assert plot.getViewBox().height() >= 120
        assert len([curve for curve in dialog.content.curves if curve.record_role == 'curve']) == 2
        low, high = plot.viewRange()[1]
        for curve in dialog.content.curves:
            finite = curve.full_y[np.isfinite(curve.full_y)]
            assert finite.min() >= low and finite.max() <= high
    finally:
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()


def test_failed_channel_rows_remain_compact_at_top(qt_app, real_results):
    from PyQt5.QtTest import QTest
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['SPL']
    failed = replace(mapped, channels=tuple(replace(channel, panels=(), metrics={},
        diagnostic='校准信息缺失', status='分析失败', judgement=None,
        judgement_status='not_executed') for channel in mapped.channels))
    content = RecordAnalysisRenderer().render(failed, {})
    try:
        content.resize(700, 500)
        content.show()
        QTest.qWait(20)
        assert content.summary_labels[0].y() <= 12
        bottom = content.diagnostics[-1].geometry().bottom()
        assert bottom < 200
        assert not content.plots
    finally:
        content.dispose()
        content.close()


@pytest.mark.parametrize('metric', ['steady_state_average', 'max_transient'])
@pytest.mark.parametrize('display_unit,limit_unit,upper,lower', [
    ('sone', 'phon', 200., 10.),
    ('phon', 'sone', 20., .1),
])
def test_real_loud_scalar_bounds_use_backend_limit_units_without_converting_results(
        qt_app, real_results, metric, display_unit, limit_unit, upper, lower):
    from copy import deepcopy

    task, _, _ = real_results
    parameters = {
        'method': 'time_varying_iso532_1',
        'advanced': {'curve_y_unit': display_unit},
        'display': {'curves': ['loudness_time'], 'heatmaps': [],
                    'summary_metrics': ['steady_state_average_loudness', 'max_transient_loudness']},
        'curve_limit_unit': limit_unit, 'limit_checked': True, 'limit_metric': metric,
        'scalar_upper_enabled': True, 'scalar_upper_value': upper,
        'scalar_lower_enabled': True, 'scalar_lower_value': lower,
    }
    item = RecordAnalysisItem('scalar-loud', 'LOUD', (0,), parameters)
    source = execute_task(replace(task, instances=(AnalysisInstance(
        item.config_key, item.analysis_type, 0, parameters),)))
    original = deepcopy(source)
    instance, = source['instance_results']
    assert instance['execution_status'] == '分析完成'
    assert instance['judgement_status'] == 'completed'
    assert instance['metrics']['unit'] == display_unit
    assert instance['metrics']['limit_unit'] == limit_unit
    assert instance['metrics']['overall_upper_limit'] == upper
    assert instance['metrics']['overall_lower_limit'] == lower
    mapped = build_record_analysis_results((item,), source)[item.config_key]
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        text = content.summary_labels[0].text()
        assert f'上限：{upper:.2f} {limit_unit}' in text
        assert f'下限：{lower:.2f} {limit_unit}' in text
        assert f'上限：{upper:.2f} {display_unit}' not in text
        assert f'下限：{lower:.2f} {display_unit}' not in text
        assert content.plots[0].record_y_unit == display_unit
        for card, label in zip(mapped.channels[0].summary_cards, content.card_labels):
            assert card['unit'] == display_unit
            assert f'{card["value"]:.2f} {display_unit}' in label.text()
        np.testing.assert_equal(source, original)
        np.testing.assert_equal(mapped.channels[0].metrics, instance['metrics'])
        np.testing.assert_equal(content.curves[0].full_y, instance['display_payload']['y'])
        assert mapped.channels[0].judgement == instance['judgement']
    finally:
        content.dispose()


def test_real_spl_scalar_bounds_keep_canonical_weighted_unit_without_limit_unit(qt_app, real_results):
    task, _, _ = real_results
    parameters = {'weighting': 'A', 'limit_checked': True, 'limit_metric': 'overall_spl',
                  'scalar_upper_enabled': True, 'scalar_upper_value': 65.,
                  'scalar_lower_enabled': True, 'scalar_lower_value': 30.}
    item = RecordAnalysisItem('scalar-spl', 'SPL', (0,), parameters)
    source = execute_task(replace(task, instances=(AnalysisInstance('scalar-spl', 'SPL', 0, parameters),)))
    instance, = source['instance_results']
    assert instance['judgement_status'] == 'completed'
    assert 'limit_unit' not in instance['metrics']
    mapped = build_record_analysis_results((item,), source)[item.config_key]
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        text = content.summary_labels[0].text()
        assert '上限：65.00 dB(A)' in text
        assert '下限：30.00 dB(A)' in text
        assert '总体声压级' in text and '幅值单位：dB(A)' in text
    finally:
        content.dispose()

@pytest.mark.parametrize('channel_count', [1, 2])
def test_real_fba_grouped_bars_preserve_values_limits_labels_and_ng(qt_app, real_results, channel_count):
    from copy import deepcopy
    import pyqtgraph as pg

    task, items, _ = real_results
    original_item = next(item for item in items if item.config_key == 'FBA')
    parameters = dict(original_item.parameters, constant_lower_enabled=True, constant_lower_value=-80.)
    item = replace(original_item, channel_indices=tuple(range(channel_count)), parameters=parameters)
    source = execute_task(replace(task, instances=tuple(
        AnalysisInstance('FBA', 'FBA', i, parameters) for i in item.channel_indices)))
    source_before = deepcopy(source)
    mapped = build_record_analysis_results((item,), source, channel_names={0: 'near', 1: 'far'})['FBA']
    mapped_before = deepcopy(mapped)
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        assert len(content.bars) == channel_count
        plot, = content.plots
        assert plot.record_band_labels == mapped.channels[0].panels[0].band_labels
        assert plot.plotItem.getAxis('bottom')._tickLevels == [
            list(enumerate(plot.record_band_labels))]
        assert not plot.plotItem.ctrl.xGridCheck.isChecked()
        assert plot.plotItem.ctrl.yGridCheck.isChecked()
        assert len(plot.plotItem.legend.items) == channel_count
        assert [label.text for _, label in plot.plotItem.legend.items] == [
            'CH1 · near', 'CH2 · far'][:channel_count]
        for index, (bar, channel) in enumerate(zip(content.bars, mapped.channels)):
            assert isinstance(bar, pg.BarGraphItem)
            panel, = channel.panels
            series = {s['role']: s for s in panel.series}
            main = series['curve']
            x = np.arange(len(panel.band_labels)) - .35 + (.7 / channel_count) * (index + .5)
            np.testing.assert_allclose(bar.opts['x'], x)
            np.testing.assert_array_equal(bar.opts['height'], main['y'])
            assert bar.opts['width'] == pytest.approx(.7 / channel_count)
            assert bar.opts['pen'].color().getRgb()[:3] == (255, 255, 255)
            assert bar.opts['pen'].widthF() == .5
            exceeded = np.isfinite(series['exceedance']['y'])
            assert exceeded.any()
            assert [brush.color().getRgb()[:3] for brush in bar.opts['brushes']] == [
                (244, 67, 54) if flag else (76, 175, 80) for flag in exceeded]
            for role, color, symbol in [('upper', (0, 188, 212), 'o'), ('lower', (63, 81, 181), 't')]:
                line, = [curve for curve in content.curves
                         if curve.record_role == role and curve.record_channel == channel.wav_channel]
                np.testing.assert_allclose(line.full_x, x)
                np.testing.assert_array_equal(line.full_y, series[role]['y'])
                assert line.opts['pen'].color().getRgb()[:3] == color
                assert line.opts['symbol'] == symbol
            annotations = [text for text in content.annotations if text.record_channel == channel.wav_channel]
            assert len(annotations) == int(exceeded.sum())
            np.testing.assert_allclose([text.pos().x() for text in annotations], x[exceeded])
            np.testing.assert_array_equal([text.pos().y() for text in annotations], main['y'][exceeded])
            assert all(text.toPlainText() == 'NG' for text in annotations)
            backend = next(instance for instance in source['instance_results']
                           if instance['channel_index'] == channel.channel_index)
            np.testing.assert_array_equal(bar.opts['height'], backend['display_payload']['y'])
            assert plot.record_y_unit == backend['display_payload']['y_unit']
            assert channel.judgement == backend['judgement']
        np.testing.assert_equal(source, source_before)
        for channel, before in zip(mapped.channels, mapped_before.channels):
            for actual, expected in zip(channel.panels[0].series, before.panels[0].series):
                np.testing.assert_equal(actual, expected)
        bars, annotations = list(content.bars), list(content.annotations)
    finally:
        content.dispose()
    assert not content.bars and not content.annotations and not content.curves and not content.plots
    assert all(bar.scene() is None for bar in bars)
    assert all(text.scene() is None for text in annotations)

@pytest.mark.parametrize('difference', ['labels', 'centres'])
def test_fba_separates_different_band_domains(qt_app, real_results, difference):
    _, items, result = real_results
    mapped = build_record_analysis_results(items, result)['FBA']
    second = mapped.channels[1]
    panel, = second.panels
    if difference == 'labels':
        panel = replace(panel, band_labels=tuple('other-' + label for label in panel.band_labels))
    else:
        panel = replace(panel, series=tuple(dict(series, x=series['x'] * 1.01) for series in panel.series))
    mapped = replace(mapped, channels=(mapped.channels[0], replace(second, panels=(panel,))))
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert len(content.plots) == 2
        assert len(content.bars) == 2
        for plot, bar, channel in zip(content.plots, content.bars, mapped.channels):
            assert plot.plotItem.getAxis('bottom')._tickLevels == [list(enumerate(channel.panels[0].band_labels))]
            np.testing.assert_array_equal(bar.opts['x'], np.arange(len(channel.panels[0].band_labels)))
            assert bar.opts['width'] == .7
    finally:
        content.dispose()


@pytest.mark.parametrize('gap', [None, float('nan')])
def test_fba_gaps_are_gray_zero_without_rejudging_or_mutating(qt_app, real_results, gap):
    from copy import deepcopy

    _, items, result = real_results
    source = deepcopy(result)
    instance = next(value for value in source['instance_results'] if value['config_key'] == 'FBA')
    instance['display_payload']['y'][0] = gap
    instance['display_payload']['upper'][1] = float('nan')
    before = deepcopy(source)
    mapped = build_record_analysis_results(items, source)['FBA']
    channel = mapped.channels[0]
    panel, = channel.panels
    # Deliberately disagree with numerical thresholds: only the returned
    # exceedance series can identify NG. Even a marked missing point stays gray.
    exceedance = np.full(len(panel.band_labels), np.nan)
    exceedance[:2] = 1.
    series = tuple(dict(value, y=exceedance) if value['role'] == 'exceedance' else value
                   for value in panel.series)
    mapped = replace(mapped, channels=(replace(channel, panels=(replace(panel, series=series),)),))
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        bar, = content.bars
        assert bar.opts['height'][0] == 0
        assert [brush.color().getRgb()[:3] for brush in bar.opts['brushes'][:3]] == [
            (189, 189, 189), (244, 67, 54), (76, 175, 80)]
        text, = content.annotations
        assert text.pos().x() == 1.
        upper, = content.curves
        assert np.isnan(upper.full_y[1])
        assert upper.opts['connect'] == 'finite'
        np.testing.assert_equal(source, before)
        assert np.isnan(mapped.channels[0].panels[0].series[0]['y'][0])
    finally:
        content.dispose()


@pytest.mark.parametrize('release', ['switch', 'close', 'parent'])
def test_fba_graphics_released_on_selection_and_window_lifecycle(qt_app, real_results, release):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QDialog
    from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
    from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter

    _, items, result = real_results
    parent = QDialog()
    dialog = RecordAnalysisDialog(parent)
    presenter = RecordAnalysisPresenter(dialog)
    presenter.set_items(items)
    presenter.show_result(build_record_analysis_results(items, result))
    dialog.navigation.setCurrentRow(2)
    old = dialog.content
    assert len(old.bars) == 2 and old.annotations
    if release == 'switch':
        dialog.navigation.setCurrentRow(0)
    elif release == 'close':
        dialog.close()
    else:
        parent.deleteLater()
        qt_app.sendPostedEvents(None, QEvent.DeferredDelete)
    assert old.disposed
    assert not old.bars and not old.annotations and not old.curves and not old.plots
    if release != 'parent':
        dialog.close()
        parent.deleteLater()
    qt_app.sendPostedEvents(None, QEvent.DeferredDelete)

@pytest.mark.parametrize('invalid', [float('inf'), float('-inf'), float('nan')])
def test_fba_invalid_amplitudes_keep_mapping_diagnostics(qt_app, real_results, invalid):
    from copy import deepcopy

    _, items, result = real_results
    source = deepcopy(result)
    instance = next(value for value in source['instance_results'] if value['config_key'] == 'FBA')
    values = instance['display_payload']['y']
    if np.isnan(invalid):
        values[:] = [invalid] * len(values)
    else:
        values[0] = invalid
    mapped = build_record_analysis_results(items, source)['FBA']
    assert not mapped.channels[0].panels
    assert mapped.channels[0].diagnostic
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert len(content.bars) == 1
        assert mapped.channels[0].diagnostic in content.diagnostic_text()
    finally:
        content.dispose()
