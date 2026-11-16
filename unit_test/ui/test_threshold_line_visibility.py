"""Limit visibility is presentation-only, shared across result viewing surfaces."""

import json
from copy import deepcopy

import pytest
from PyQt5.QtCore import Qt

from base.workbench_configuration_loader import load_workbench_configuration
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.sequence.workbench_ui_projection import build_ui_projection
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow
from ui.ui_analysis_config.loudness_config_dialog import LoudnessConfigWindow
from ui.ui_analysis_config.reference_spectrum_config_dialog import ReferenceSpectrumConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from unit_test.ui.test_plot_surface_consistency import close
from unit_test.ui.test_shared_plot_pipeline import station_layout
from unit_test.ui.test_spl_spec_config_dialog import _ConfigManager
from unit_test.base.workbench_config_fixtures import write_workbench_files, write_json


def checkbox(dialog):
    if isinstance(dialog, LoudnessConfigWindow):
        return dialog.panel.show_threshold_lines_box
    if isinstance(dialog, ReferenceSpectrumConfigWindow):
        return dialog.show_threshold_lines_box
    return dialog.curve_color_widget.show_threshold_lines_box


@pytest.mark.parametrize('kind,window', [
    ('SPL', SplConfigWindow), ('FFT', FftConfigWindow), ('FBA', FbaConfigWindow),
    ('LOUD', LoudnessConfigWindow), ('RSC', ReferenceSpectrumConfigWindow),
])
@pytest.mark.parametrize('display', [{}, {'show_threshold_lines': True}, {'show_threshold_lines': False}])
def test_dialog_saves_and_reloads_visibility_without_changing_limits(qt_app, kind, window, display):
    manager = _ConfigManager({kind: {'display': display, 'limit_checked': True}})
    dialog = window(manager, kind)
    try:
        box = checkbox(dialog)
        assert box.text() == '显示阈值线'
        assert box.isChecked() == display.get('show_threshold_lines', True)
        before = dialog.get_default_config()
        box.setChecked(not box.isChecked())
        after = dialog.get_default_config()
        assert after['display']['show_threshold_lines'] == box.isChecked()
        # Everything except this display field, including thresholds, is unchanged.
        expected = deepcopy(before)
        expected['display']['show_threshold_lines'] = box.isChecked()
        assert after == expected
        reopened = window(_ConfigManager({kind: after}), kind)
        try:
            assert checkbox(reopened).isChecked() == box.isChecked()
        finally:
            close(reopened)
    finally:
        close(dialog)


def test_main_and_detail_respect_each_condition_and_keep_highlights(tmp_path, qt_app):
    files = write_workbench_files(tmp_path, all_stations=True)
    for visible, name in zip((False, True), ('测试队列A', '测试队列B')):
        queue = json.loads(files[name].read_text(encoding='utf-8'))
        queue[0]['seq1']['analysis_list']['声压级']['weighting'] = 'A'
        queue[0]['seq1']['analysis_list']['声压级']['display']['show_threshold_lines'] = visible
        write_json(files[name], queue)
    write_json(files['layout'], station_layout())
    configuration = load_workbench_configuration(files['programs'], files['queues'], files['layout'])
    projection = build_ui_projection(configuration.ui, configuration.analysis)
    grid = StationPlotGrid()
    try:
        grid.show_workbench_layout(projection.column_names, projection.cells)
        cell = grid._cells[0, 0]
        segments = projection.cells[0][0].condition_plot.conditions
        assert [s.show_threshold_lines for s in segments] == [False, True]
        for segment in segments:
            cell.set_condition_curve_data(segment.condition_id, [dict(
                name='CH1', wav_channel=segment.channels[0][0], role=role,
                x=[0, 1, 2], y=[1, 3, 1] if role == 'curve' else [2, 2, 2],
            ) for role in ('curve', 'upper', 'lower', 'exceedance')])
        assert all(len(v) == 4 for v in cell._condition_series.values())
        cell.open_detail()
        qt_app.processEvents()
        for view in (cell, cell._detail_dialog.cell):
            items = view.plot_widget.listDataItems()
            assert len(items) == 6  # hidden condition: curve + highlight; visible: all four
            assert sum(item.opts['pen'].style() == Qt.DashLine for item in items) == 2
            assert sum(item.opts['symbol'] == 'o' for item in items) == 2
    finally:
        for cell in grid._cells.values():
            cell.dispose()
        close(grid)
