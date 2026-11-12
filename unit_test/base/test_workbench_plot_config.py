from copy import deepcopy
import subprocess
import sys

import pytest

from base.workbench_plot_config import parse_plot_templates
from base.workbench_configuration_loader import WorkbenchConfigurationError, load_workbench_configuration
from unit_test.base.workbench_config_fixtures import write_json, write_workbench_files
from unit_test.base.workbench_plot_fixtures import template_layout


@pytest.mark.parametrize('setting', [
    None, {}, {'method': 'bad'}, {'method': 'none', 'pixel_bucket_width': 1},
    {'method': 'preserve_min_max'}, {'method': 'preserve_min_max', 'pixel_bucket_width': True},
    {'method': 'preserve_min_max', 'pixel_bucket_width': 0},
    {'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': 1.5},
    {'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': -1},
])
def test_downsampling_rejects_invalid_or_ambiguous_settings(setting):
    data = template_layout()
    data['plot_templates']['spl']['display_downsampling'] = setting
    with pytest.raises(ValueError, match='display_downsampling'):
        parse_plot_templates(data)


def test_downsampling_default_and_cell_override_are_independent():
    data = template_layout()
    data['cells'][0][1]['overrides'] = {'display_downsampling': {
        'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': 10,
    }}
    data['cells'][0][2]['overrides'] = {'display_downsampling': {'method': 'none'}}
    result = parse_plot_templates(data)
    assert result.cells[0][0].display_settings['display_downsampling'] == {
        'method': 'preserve_min_max', 'pixel_bucket_width': 1,
    }
    assert result.cells[0][1].display_settings['display_downsampling']['keep_one_point_every_n_points'] == 10
    assert result.cells[0][2].display_settings['display_downsampling'] == {'method': 'none'}




@pytest.mark.parametrize('values', [
    {'empty_data_range_min': 0},
    {'empty_data_range_min': 10, 'empty_data_range_max': 0},
    {'empty_data_range_min': True, 'empty_data_range_max': 100},
    {'empty_data_range_min': 0, 'empty_data_range_max': float('inf')},
    {'empty_data_range_min': 0, 'empty_data_range_max': 100, 'scale': 'log'},
    {'empty_data_range_min': 0, 'empty_data_range_max': 100, 'auto_range': False},
])
def test_invalid_empty_y_range_is_rejected(values):
    data = template_layout()
    data['plot_templates']['spl']['y_axis'].update(values)
    with pytest.raises(ValueError, match='y_axis'):
        parse_plot_templates(data)


def test_templates_and_overrides_resolve_without_mutation_or_shared_mutable_state():
    data = template_layout()
    data['cells'][0][1]['overrides'] = {
        'title': '指定工位',
        'channel_display': {'show_all_analyzed_channels': False, 'display_wav_channels': [2]},
        'plot_style': {'curve_line_style': 'dotted', 'channel_curve_styles': [
            {'wav_channel': 2, 'curve_color': '#FF6600'},
        ]},
    }
    before = deepcopy(data)
    result = parse_plot_templates(data)
    first, second = result.cells[0][:2]
    assert first.display_settings['plot_style']['curve_line_width'] == 2
    assert first.display_settings['plot_style']['curve_line_style'] == 'solid'
    assert second.title == '指定工位'
    assert second.display_settings['plot_style']['curve_line_width'] == 2
    assert second.display_settings['plot_style']['curve_line_style'] == 'dotted'
    assert len(second.display_settings['plot_style']['channel_curve_styles']) == 1
    assert second.display_settings['channel_display']['display_wav_channels'] == (2,)
    assert data == before
    data['default_condition_display']['show_condition_names'] = False
    assert first.display_settings['condition_display']['show_condition_names'] is True
    with pytest.raises(TypeError):
        first.display_settings['plot_style']['curve_line_width'] = 8


@pytest.mark.parametrize(('path', 'value', 'message'), [
    (('cells', 0, 0, 'plot_template'), 'missing', '模板'),
    (('cells', 0, 0, 'overrides'), {'plot_style': None}, 'plot_style'),
    (('default_plot_style', 'curve_line_width'), -1, 'curve_line_width'),
    (('default_plot_style', 'curve_line_width'), float('nan'), 'curve_line_width'),
    (('default_plot_style', 'curve_line_style'), 'dash', 'curve_line_style'),
    (('default_plot_style', 'background_color'), 'bad', 'background_color'),
    (('default_plot_style', 'grid', 'visible'), 1, 'visible'),
    (('default_plot_style', 'grid', 'typo'), True, 'typo'),
    (('default_condition_display', 'layout'), 'overlay', 'layout'),
    (('default_condition_display', 'condition_source'), 'manual', 'condition_source'),
    (('default_condition_display', 'x_axis_mode'), 'unknown', 'x_axis_mode'),
    (('default_condition_display', 'show_condition_names'), 1, 'show_condition_names'),
    (('default_condition_display', 'show_condition_separator'), 'yes', 'show_condition_separator'),
    (('default_condition_display', 'condition_gap_px'), -1, 'condition_gap_px'),
    (('default_condition_display', 'condition_gap_px'), True, 'condition_gap_px'),
    (('plot_templates', 'spl', 'channel_display'), {'show_all_analyzed_channels': False, 'display_wav_channels': [0]}, 'display_wav_channels'),
    (('plot_templates', 'spl', 'channel_display'), {'show_all_analyzed_channels': False, 'display_wav_channels': [1, 1]}, '通道重复'),
    (('plot_templates', 'spl', 'channel_display'), {'show_all_analyzed_channels': True, 'display_wav_channels': [1]}, '全部通道'),
    (('plot_templates', 'spl', 'x_axis', 'scale'), 'log', '对数轴'),
    (('plot_templates', 'spl', 'x_axis', 'range_max'), 0, '范围'),
    (('plot_templates', 'spl', 'x_axis', 'auto_range'), True, '自动范围'),
    (('plot_templates', 'spl', 'chart_type'), 'heatmap', 'chart_type'),
    (('plot_templates', 'spl', 'unknown'), 1, 'unknown'),
    (('rows',), True, 'rows'),
    (('cells', 0), [], 'cells'),
])
def test_invalid_config_has_a_specific_error(path, value, message):
    data = template_layout()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        parse_plot_templates(data)


def test_condition_gap_defaults_to_sixteen_pixels_without_changing_input():
    data = template_layout()
    del data['default_condition_display']['condition_gap_px']
    result = parse_plot_templates(data)
    assert result.cells[0][0].display_settings['condition_display']['condition_gap_px'] == 16
    assert 'condition_gap_px' not in data['default_condition_display']


def test_existing_loader_reads_templates_and_reports_file_path(tmp_path):
    files = write_workbench_files(tmp_path)
    data = template_layout()
    write_json(files['layout'], data)
    configuration = load_workbench_configuration(files['programs'], files['queues'], files['layout'])
    assert configuration.ui.layout.cells[0][0].display_settings['condition_display']['layout'] == 'single_canvas'
    data['cells'][0][0]['plot_template'] = 'missing'
    write_json(files['layout'], data)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load_workbench_configuration(files['programs'], files['queues'], files['layout'])
    assert str(files['layout']) in str(caught.value)
    assert 'cells[0][0]' in str(caught.value)


def test_parser_does_not_import_qt_or_plotting():
    code = "import sys; from base.workbench_plot_config import parse_plot_templates; assert not any(k.startswith(('PyQt5', 'pyqtgraph')) for k in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
