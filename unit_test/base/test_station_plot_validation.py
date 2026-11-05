import numpy as np
import pytest

from ui.sequence.station_plot_validation import (
    validate_curve_series,
    validate_heatmap_data,
    validate_plot_config,
)


def test_curve_snapshot_copies_arrays_and_discards_extra_fields():
    x = np.array([2., 0., 1.])
    y = [3., 4., 5.]
    result = validate_curve_series(
        [{'x': x, 'y': y, 'ignored': object()}, {'x': [9], 'y': [8]}],
        context='set_curve_data cell (1, 2)',
    )
    x[:] = 100
    y[0] = 100
    np.testing.assert_array_equal(result[0]['x'], [2, 0, 1])
    np.testing.assert_array_equal(result[0]['y'], [3, 4, 5])
    assert set(result[0]) == {'x', 'y', 'name', 'color'}
    assert result[0]['name'] == '曲线 1'
    assert result[1]['name'] == '曲线 2'


@pytest.mark.parametrize('value,error', [
    ('123', TypeError), ([True, 1], TypeError), ([1j], TypeError),
    (['1'], TypeError), (np.array([False]), TypeError),
    (np.array([1], dtype=object), TypeError), ([[1]], ValueError),
    ([float('nan')], ValueError), ([float('inf')], ValueError),
])
def test_curve_rejects_invalid_real_vector_with_field_diagnostic(value, error):
    with pytest.raises(error, match=r'set_curve_data.*\(1, 2\).*series\[0\].x'):
        validate_curve_series([{'x': value, 'y': [1]}], context='set_curve_data cell (1, 2)')


@pytest.mark.parametrize('series,error,field', [
    (None, TypeError, 'series'), ([[]], TypeError, r'series\[0\]'),
    ([{'x': [1]}], ValueError, 'y'),
    ([{'x': [1], 'y': []}], ValueError, 'y'),
    ([{'x': [], 'y': [], 'name': 1}], TypeError, 'name'),
    ([{'x': [], 'y': [], 'color': '#not-a-color'}], ValueError, 'color'),
    ([{'x': [], 'y': [], 'color': None}], TypeError, 'color'),
])
def test_curve_structure_and_optional_fields(series, error, field):
    with pytest.raises(error, match=field):
        validate_curve_series(series, context='set_curve_data cell (0, 0)')


@pytest.mark.parametrize('color', ['r', '#1234', 'transparent', '#12345678'])
def test_curve_accepts_pyqtgraph_color_strings(color):
    result = validate_curve_series([{'x': [1], 'y': [2], 'color': color}],
                                   context='set_curve_data cell (0, 0)')
    assert result[0]['color'] == color


@pytest.mark.parametrize('field,value,error', [
    ('kind', 'bars', ValueError), ('kind', True, TypeError),
    ('title', None, TypeError), ('right_text', None, TypeError),
    ('right_text', 12, TypeError), ('x_unit', 12, TypeError),
    ('x_range', [1, 1], ValueError), ('y_range', [3, 2], ValueError),
    ('x_range', [0, float('inf')], ValueError), ('x_range', [0, True], TypeError),
    ('x_range', [0, 1, 2], ValueError),
])
def test_config_rejects_invalid_fields(field, value, error):
    params = {'kind': 'curve', field: value}
    with pytest.raises(error, match=rf'configure_plot.*\(0, 3\).*{field}'):
        validate_plot_config(**params, context='configure_plot cell (0, 3)')


def test_config_corner_text_is_literal_and_defaults_empty():
    context = 'configure_plot cell (0, 3)'
    assert validate_plot_config(kind='curve', context=context)['right_text'] == ''
    config = validate_plot_config(kind='heatmap', title='<b>SPL & 趋势</b>',
                                  right_text='MIC 1', context=context)
    assert config['title'] == '<b>SPL & 趋势</b>' and config['right_text'] == 'MIC 1'


def test_heatmap_snapshot_extent_and_constant_levels():
    values = np.full((2, 3), 7.)
    result = validate_heatmap_data(values, context='set_heatmap_data cell (0, 0)')
    values[:] = 100
    np.testing.assert_array_equal(result['values'], np.full((2, 3), 7.))
    assert result['extent'] == (0, 3, 0, 2)
    assert result['levels'] == pytest.approx((6.93, 7.07))
    zero = validate_heatmap_data([[0]], context='set_heatmap_data cell (0, 0)')
    assert zero['levels'] == (-1e-6, 1e-6)


@pytest.mark.parametrize('value,error', [
    ([[1, 2], [3]], ValueError), ([1, 2], ValueError),
    ([[True]], TypeError), ([['1']], TypeError), ([[1j]], TypeError),
    ([[float('nan')]], ValueError), (np.ones((1, 2, 3)), ValueError),
])
def test_heatmap_rejects_invalid_matrix(value, error):
    with pytest.raises(error, match=r'set_heatmap_data.*\(2, 3\).*values'):
        validate_heatmap_data(value, context='set_heatmap_data cell (2, 3)')


@pytest.mark.parametrize('kwargs,field', [
    ({'extent': (0, 0, 0, 1)}, 'extent'),
    ({'extent': (0, 1, 2)}, 'extent'),
    ({'levels': (2, 1)}, 'levels'),
    ({'levels': (0, float('inf'))}, 'levels'),
    ({'colormap': 'not-a-local-map'}, 'colormap'),
    ({'colormap': '../viridis'}, 'colormap'),
])
def test_heatmap_rejects_mapping_before_commit(kwargs, field):
    with pytest.raises(ValueError, match=field):
        validate_heatmap_data([[1, 2]], context='set_heatmap_data cell (0, 0)', **kwargs)


@pytest.mark.parametrize('values', [None, np.empty((0, 2)), [[], []]])
def test_empty_heatmap_is_explicit_clear(values):
    result = validate_heatmap_data(values, context='set_heatmap_data cell (0, 0)')
    assert result['values'] is None


@pytest.mark.parametrize('field', ['subheader_left_text', 'subheader_first_half_text', 'subheader_second_half_text'])
def test_subheader_config_defaults_and_type_diagnostics(field):
    context = 'configure_plot cell (1, 2)'
    assert validate_plot_config(kind='curve', context=context)[field] == ''
    assert validate_plot_config(kind='heatmap', context=context, **{field: '<b>A & B</b>'})[field] == '<b>A & B</b>'
    with pytest.raises(TypeError, match=rf'configure_plot.*\(1, 2\).*{field}'):
        validate_plot_config(kind='heatmap', context=context, **{field: None})
