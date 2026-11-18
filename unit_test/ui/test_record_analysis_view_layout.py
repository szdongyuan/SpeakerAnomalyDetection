"""Deterministic presentation contracts for the simplified saved-record view."""
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextDocument
from PyQt5.QtWidgets import QLabel

from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_request import RecordAnalysisItem
from consts.ui_style_const import (
    MAIN_UI_SMALL_FONT_FAMILY_NAME, MOTOR_STATION_STATUS_COLORS,
    MOTOR_WORKBENCH_SCROLLBAR_STYLE,
)
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter
from ui.sequence.record_analysis_renderers import RecordAnalysisRenderer
from ui.sequence.record_analysis_results import (
    RecordAnalysisChannelResult, RecordAnalysisItemResult, RecordAnalysisPanel,
)
from ui.sequence.station_plot_cell import StationPlotCell
from unit_test.ui.test_station_plot_cell import axis_draw_specs


def document(label):
    doc = QTextDocument()
    doc.setHtml(label.text())
    return doc


def text(label):
    return document(label).toPlainText()


def token_color(label, token):
    doc = document(label)
    return doc.find(token).charFormat().foreground().color().name()


def result(kind='SPL', *, metrics=None, unit='dBA', channels=1):
    panel = RecordAnalysisPanel(kind, 'curve', 's', unit,
        series=(dict(role='curve', wav_channel=1, name='CH1',
                     x=np.array([0., 1., 2.]), y=np.array([20., 21., 22.])),),
        metadata={'original': 'unchanged'})
    first = RecordAnalysisChannelResult(0, 1, 'CH1', '分析完成',
        judgement_status='completed', judgement='NG',
        metrics=metrics if metrics is not None else {'overall_spl': 21., 'unit': unit},
        panels=(panel,))
    values = (first,) if channels == 1 else (first, replace(first, channel_index=1,
        wav_channel=2, name='far', judgement='OK', panels=(replace(panel, series=(dict(
            panel.series[0], wav_channel=2, name='CH2 · far'),)),)))
    return RecordAnalysisItemResult(kind, kind, values, '分析完成')


def test_header_removes_duplicate_widgets_and_stacks_independent_colored_results(qt_app):
    dialog = RecordAnalysisDialog()
    presenter = RecordAnalysisPresenter(dialog)
    try:
        dialog.show_record(DailyWavRecord(datetime(2026, 11, 18), 7, 'saved.wav', '正转', 'NG'))
        presenter.set_items((RecordAnalysisItem('SPL', 'SPL', (0,), {}),))
        current = result()
        presenter.show_result({'SPL': replace(current, channels=(replace(current.channels[0], judgement='OK'),))})
        dialog.show()
        qt_app.processEvents()
        assert not hasattr(dialog, 'record_info')
        assert not hasattr(dialog, 'item_title')
        assert text(dialog.original_result) == '原记录结果：NG'
        assert text(dialog.current_result) == '本次判定：OK'
        assert token_color(dialog.original_result, 'NG') == MOTOR_STATION_STATUS_COLORS['NG']
        assert token_color(dialog.current_result, 'OK') == MOTOR_STATION_STATUS_COLORS['OK']
        assert token_color(dialog.current_result, '本次判定') != MOTOR_STATION_STATUS_COLORS['OK']
        assert dialog.original_result.mapTo(dialog, dialog.original_result.rect().bottomLeft()).y() < dialog.current_result.mapTo(dialog, dialog.current_result.rect().topLeft()).y()
        assert dialog.original_result.geometry().right() == dialog.current_result.geometry().right()
        assert not dialog.content.title_labels
        labels = [label.text() for label in dialog.content.findChildren(QLabel)]
        assert not any('CH1' in value or '分析完成' in value or '本次判定' in value for value in labels)
        assert '总体声压级：21.00' in dialog.content.summary_labels[0].text()
        bar = dialog.content_scroll.verticalScrollBar()
        assert bar.objectName() == 'workbenchStationVerticalScrollBar'
        assert bar.styleSheet() == MOTOR_WORKBENCH_SCROLLBAR_STYLE
        assert dialog.content_scroll.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    finally:
        dialog.close()


def test_judgement_switch_multi_failure_unsupported_wait_and_disposal(qt_app):
    dialog = RecordAnalysisDialog()
    presenter = RecordAnalysisPresenter(dialog)
    try:
        items = tuple(RecordAnalysisItem(key, key, (0, 1), {}) for key in ('SPL', 'FFT', 'AI', 'missing'))
        presenter.set_items(items)
        multi = result(channels=2)
        failed = RecordAnalysisItemResult('FFT', 'FFT', (replace(multi.channels[0],
            judgement=None, judgement_status='failed', status='分析失败', diagnostic='bad calibration', panels=()),), '分析失败')
        unsupported = RecordAnalysisItemResult('AI', 'AI', (), '不支持', 'unsupported')
        presenter.show_result({'SPL': multi, 'FFT': failed, 'AI': unsupported})
        assert text(dialog.current_result) == '本次判定：CH1 NG · CH2 OK'
        assert token_color(dialog.current_result, 'NG') == MOTOR_STATION_STATUS_COLORS['NG']
        assert token_color(dialog.current_result, 'OK') == MOTOR_STATION_STATUS_COLORS['OK']
        assert all(label.text().startswith(f'CH{i}') for i, label in enumerate(dialog.content.summary_labels, 1))
        old = dialog.content
        dialog.navigation.setCurrentRow(1)
        assert old.disposed
        assert text(dialog.current_result) == '本次判定：判定失败'
        assert 'bad calibration' in dialog.content.diagnostic_text()
        dialog.navigation.setCurrentRow(2)
        assert text(dialog.current_result) == '本次判定：未判定'
        assert 'unsupported' in dialog.content.diagnostic_text()
        dialog.navigation.setCurrentRow(3)
        assert text(dialog.current_result) == '本次判定：--'
        presenter.show_status('分析中')
        assert text(dialog.current_result) == '本次判定：--'
        assert dialog.status.text() == '分析中'
        assert dialog.content is None
    finally:
        dialog.close()
    assert presenter.disposed


@pytest.mark.parametrize('metrics,unit,parameters,expected', [
    ({'weighting': 'A', 'unit': 'dBC'}, 'dBC', {'weighting': 'Z'}, 'dB(A)'),
    ({'weighting': 'C'}, 'dBA', {}, 'dB(C)'),
    ({'weighting': 'Z'}, 'dBA', {}, 'dB(Z)'),
    ({'weighting': None}, 'dBA', {'weighting': 'A'}, 'dB(Z)'),
    ({'weighting': 'None'}, 'dBA', {}, 'dB(Z)'),
    ({'unit': 'dBC'}, 'dBA', {'weighting': 'A'}, 'dB(C)'),
    ({}, 'dBA', {'weighting': 'C'}, 'dB(A)'),
    ({}, 'dB', {'weighting': 'C'}, 'dB(C)'),
    ({}, 'dB', {}, 'dB(Z)'),
])
def test_spl_axis_weighting_precedence_fonts_and_immutable_units(qt_app, metrics, unit, parameters, expected):
    mapped = result(metrics=metrics, unit=unit)
    panel = mapped.channels[0].panels[0]
    original = panel.series[0]['y'].copy()
    content = RecordAnalysisRenderer().render(mapped, parameters)
    try:
        plot = content.plots[0]
        assert plot.getAxis('left').labelText == expected
        assert plot.getAxis('bottom').labelText == '时间（s）'
        for name in ('left', 'bottom'):
            axis = plot.getAxis(name)
            assert axis.style['tickFont'].family() == MAIN_UI_SMALL_FONT_FAMILY_NAME
            assert axis.style['tickFont'].pixelSize() == 16
            assert axis.label.font() == axis.style['tickFont']
            assert axis.pen().color().name() == '#afc0d6'
            assert axis.textPen().color().name() == '#64748b'
        assert panel.y_unit == plot.record_y_unit == unit
        assert panel.metadata == plot.record_metadata == {'original': 'unchanged'}
        np.testing.assert_array_equal(panel.series[0]['y'], original)
        np.testing.assert_array_equal(content.curves[0].full_y, original)
        assert mapped.channels[0].metrics == metrics
    finally:
        content.dispose()
        content.close()


@pytest.mark.parametrize('kind,unit', [('FFT', 'dB'), ('FFT', 'dBA'), ('FBA', 'dBC'), ('LOUD', 'phon'), ('LOUD', 'sone')])
def test_non_spl_axis_semantics_without_title_rows(qt_app, kind, unit):
    mapped = result(kind, unit=unit)
    content = RecordAnalysisRenderer().render(mapped, {'weighting': 'A'})
    try:
        expected = {'dBA': 'dB(A)', 'dBC': 'dB(C)'}.get(unit, unit)
        assert content.plots[0].getAxis('left').labelText == expected
        assert content.plots[0].getAxis('bottom').labelText == '时间（s）'
        assert not content.title_labels
        assert content.content_layout.indexOf(content.legends[0]) + 1 == content.content_layout.indexOf(content.plots[0])
        assert content.legends[0].align_right
    finally:
        content.dispose()
        content.close()


def test_heatmap_keeps_matrix_coordinates_colorbar_and_bark(qt_app):
    panel = RecordAnalysisPanel('specific_loudness', 'heatmap', 's', 'Bark',
        x_label='Time / s', y_label='Bark', value_unit='sone/Bark',
        heatmap={'x': np.array([1., 2.]), 'y': np.array([1., 3.]), 'z': np.array([[1., 2.], [3., 4.]])})
    mapped = result('LOUD')
    mapped = replace(mapped, channels=(replace(mapped.channels[0], panels=(panel,)),))
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert content.plots[0].getAxis('bottom').labelText == '时间（s）'
        assert content.plots[0].getAxis('left').labelText == 'Bark'
        assert content.colorbars[0].axis.labelText == 'sone/Bark'
        np.testing.assert_array_equal(content.images[0].values, panel.heatmap['z'])
        np.testing.assert_array_equal(content.images[0].x_edges, [.5, 1.5, 2.5])
        np.testing.assert_array_equal(content.images[0].y_edges, [0., 2., 4.])
        np.testing.assert_array_equal(panel.heatmap['x'], [1., 2.])
        np.testing.assert_array_equal(panel.heatmap['y'], [1., 3.])
        assert not content.title_labels
    finally:
        content.dispose()
        content.close()


def test_failed_spl_missing_unit_preserves_diagnostic_without_empty_summary(qt_app):
    channel = RecordAnalysisChannelResult(0, 1, 'CH1', '分析失败',
        diagnostic='单位不可用', execution_status='分析失败', metrics={'unit': None})
    mapped = RecordAnalysisItemResult('SPL', 'SPL', (channel,), '分析失败')
    content = RecordAnalysisRenderer().render(mapped, {})
    try:
        assert content.diagnostic_text() == '单位不可用'
        assert not content.summary_labels
        assert not content.plots
        assert channel.metrics == {'unit': None}
    finally:
        content.dispose()
        content.close()


def test_axes_share_main_classes_margins_and_keep_endpoint_text_inside(qt_app):
    content = RecordAnalysisRenderer().render(result(), {})
    main = StationPlotCell()
    try:
        content.resize(420, 400)
        content.show()
        plot = content.plots[0]
        plot.setXRange(0, .4, padding=0)
        axis = plot.getAxis('bottom')
        axis.setTicks([[(0, '0'), (.2, '0.2'), (.4, '0.4')]])
        for _ in range(3):
            qt_app.processEvents()
        _, ticks, labels = axis_draw_specs(axis)
        _, native_ticks, _ = axis_draw_specs(axis, base=True)
        assert ticks == native_ticks
        assert '0.4' in {value for _, _, value in labels}
        viewport = plot.mapToScene(plot.viewport().rect()).boundingRect()
        for rect, _, value in labels:
            assert viewport.contains(axis.mapRectToScene(rect)), (value, rect, viewport)
        for name in ('left', 'bottom'):
            assert type(plot.getAxis(name)) is type(main.plot_widget.getAxis(name))
        assert plot.plotItem.layout.getContentsMargins() == main.plot_widget.plotItem.layout.getContentsMargins()
        np.testing.assert_array_equal(content.curves[0].full_x, [0., 1., 2.])
    finally:
        content.dispose()
        content.close()
        main.dispose()
        main.close()
