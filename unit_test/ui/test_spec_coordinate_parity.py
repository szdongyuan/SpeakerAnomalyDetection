"""Main/detail plots preserve the backend's physical time/frequency bins."""
import numpy as np
import pytest
from base.analysis_algorithm_adapters import _calculate_spec
from base.workbench_plot_config import parse_plot_templates
from ui.plotting.geometry import bin_edges, frequency_bin_geometry
from ui.sequence.condition_heatmap_cell import ConditionHeatmapCell
from ui.sequence.workbench_plot_display import (
    ConditionPlotDisplay, ConditionSegmentDisplay, WorkbenchPlotCellDisplay,
)
from unit_test.ui.test_plot_surface_consistency import close
from unit_test.ui.test_shared_plot_pipeline import station_layout


def make_main_cell(scale):
    layout = station_layout()
    template = layout['plot_templates']['spl']
    template.update(analysis_type='Spec', chart_type='heatmap',
                    channel_display={'show_all_analyzed_channels': False, 'display_wav_channels': [1]},
                    x_axis={'label': 'Time', 'scale': 'linear', 'auto_range': True},
                    y_axis={'label': 'Frequency', 'scale': scale, 'auto_range': True},
                    spectrogram={'color_range': {'mode': 'auto'}})
    settings = parse_plot_templates(layout).cells[0][0].display_settings
    segment = ConditionSegmentDisplay('c1', 'condition', duration_seconds=1.,
                                      channels=((1, 'CH1'),), x_unit='s', y_unit='Hz', analysis_type='Spec')
    return ConditionHeatmapCell(WorkbenchPlotCellDisplay(
        title='Spec', condition_plot=ConditionPlotDisplay(settings, (segment,))))


@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('factor', [1., 2.])
def test_main_and_detail_preserve_calibrated_matrix_and_bin_centres(qt_app, scale, factor):
    rate = 48000
    time = np.arange(rate) / rate
    signal = (.02 * np.sin(2 * np.pi * 3000 * time)).astype(np.float32)
    config = dict(n_fft=256, hop_length=64, window_func='hann', freq_scale_type=scale)
    result = _calculate_spec(signal, rate, config, factor, full_spectrogram=True)['plot']
    main = make_main_cell(scale)
    try:
        main.set_condition_heatmap_data('c1', dict(time_s=result['x'], frequency_hz=result['y'],
                                                   values_db=result['z'], wav_channel=1))
        first, expected_edges = frequency_bin_geometry(result['y'], logarithmic=scale == 'log')
        if scale == 'log':
            expected_edges = np.log10(expected_edges)
        main.open_detail()
        qt_app.processEvents()
        for view in (main, main._detail_dialog.cell):
            image = view._images[0]
            np.testing.assert_array_equal(image.values, result['z'][first:])
            np.testing.assert_allclose(image.y_edges, expected_edges)
            np.testing.assert_allclose(image.x_edges, view._map_x('c1', bin_edges(result['x'])))
            if scale == 'linear':
                row = int(np.flatnonzero(result['y'] == 3000.)[0])
                assert (image.y_edges[row] + image.y_edges[row + 1]) / 2 == pytest.approx(3000.)
        main.set_condition_state('c1', '', clear=True)
        assert not main._images and not main._detail_dialog.cell._images
    finally:
        close(main)
