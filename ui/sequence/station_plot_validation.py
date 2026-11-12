"""Validate complete plot submissions and return independent numeric snapshots."""

from numbers import Real

import numpy as np
import pyqtgraph as pg


def _real_array(value, *, dimensions, context, field, allow_gaps=False):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f'{context}: {field} must be a list, tuple or real NumPy array')
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'iuf':
            raise TypeError(f'{context}: {field} must contain real numbers, excluding bool')
    else:
        # Check before conversion: NumPy would silently coerce mixed bool/number
        # or string/number lists and lose the caller's original value types.
        def check_numbers(items):
            for item in items:
                if isinstance(item, (list, tuple)):
                    check_numbers(item)
                elif isinstance(item, np.ndarray):
                    if item.dtype.kind not in 'iuf':
                        raise TypeError(f'{context}: {field} must contain real numbers, excluding bool')
                elif item is None and allow_gaps:
                    continue
                elif isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
                    raise TypeError(f'{context}: {field} must contain real numbers, excluding bool')
        check_numbers(value)
    try:
        array = np.array(value, dtype=float, copy=True)
    except (ValueError, OverflowError) as error:
        # Conversion is the external numeric boundary for ragged arrays and
        # real Python integers too large for the display's float representation.
        raise ValueError(f'{context}: {field} must be a rectangular finite real array') from error
    if array.ndim != dimensions:
        raise ValueError(f'{context}: {field} must have {dimensions} dimension(s)')
    if np.isinf(array).any() or (not allow_gaps and np.isnan(array).any()):
        raise ValueError(f'{context}: {field} must contain only finite values')
    return array


def validate_range(value, *, context, field):
    """Return a finite increasing two-endpoint range (or automatic None)."""
    if value is None:
        return None
    array = _real_array(value, dimensions=1, context=context, field=field)
    if array.size != 2 or not array[0] < array[1]:
        raise ValueError(f'{context}: {field} must contain two strictly increasing endpoints')
    return tuple(array)


def validate_plot_header(left_text, right_text, *, context):
    for field, value in (('left_text', left_text), ('right_text', right_text)):
        if not isinstance(value, str):
            raise TypeError(f'{context}: {field} must be a string')


def validate_plot_subheader(left_text, first_half_text, second_half_text, *, context):
    for field, value in (('left_text', left_text), ('first_half_text', first_half_text),
                         ('second_half_text', second_half_text)):
        if not isinstance(value, str):
            raise TypeError(f'{context}: {field} must be a string')


def validate_plot_config(*, kind, title='', right_text='', x_label='', y_label='', x_unit='',
                         y_unit='', x_range=None, y_range=None, subheader_left_text='',
                         subheader_first_half_text='', subheader_second_half_text='', context):
    config = dict(kind=kind, title=title, right_text=right_text, x_label=x_label, y_label=y_label,
                  x_unit=x_unit, y_unit=y_unit, subheader_left_text=subheader_left_text,
                  subheader_first_half_text=subheader_first_half_text,
                  subheader_second_half_text=subheader_second_half_text)
    for field, value in config.items():
        if not isinstance(value, str):
            raise TypeError(f'{context}: {field} must be a string')
    if kind not in ('curve', 'heatmap'):
        raise ValueError(f'{context}: kind must be curve or heatmap')
    config['x_range'] = validate_range(x_range, context=context, field='x_range')
    config['y_range'] = validate_range(y_range, context=context, field='y_range')
    return config


def validate_curve_series(series, *, context, allow_gaps=False):
    if not isinstance(series, (list, tuple)):
        raise TypeError(f'{context}: series must be a list or tuple')
    result = []
    for index, item in enumerate(series):
        field = f'series[{index}]'
        if not isinstance(item, dict):
            raise TypeError(f'{context}: {field} must be a dict')
        for key in ('x', 'y'):
            if key not in item:
                raise ValueError(f'{context}: {field}.{key} is required')
        x = _real_array(item['x'], dimensions=1, context=context, field=f'{field}.x')
        y = _real_array(item['y'], dimensions=1, context=context, field=f'{field}.y', allow_gaps=allow_gaps)
        if x.size != y.size:
            raise ValueError(f'{context}: {field}.y must have the same length as x')
        name = item.get('name', f'曲线 {index + 1}')
        if not isinstance(name, str):
            raise TypeError(f'{context}: {field}.name must be a string')
        color = item.get('color')
        if 'color' in item:
            if not isinstance(color, str):
                raise TypeError(f'{context}: {field}.color must be a color string')
            if not color:
                raise ValueError(f'{context}: {field}.color is not a valid color string')
            try:
                pg.mkColor(color)
            except ValueError as error:
                # Match the actual plotting parser (including #RGBA and 'r'),
                # while adding the submission location to its input error.
                raise ValueError(f'{context}: {field}.color is not a valid color string') from error
        result.append(dict(x=x, y=y, name=name, color=color))
    return result


def validate_heatmap_data(values, *, extent=None, levels=None, colormap='viridis', context):
    array = None if values is None else _real_array(
        values, dimensions=2, context=context, field='values')
    if array is not None and array.size == 0:
        array = None
    if extent is not None:
        bounds = _real_array(extent, dimensions=1, context=context, field='extent')
        if bounds.size != 4 or not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            raise ValueError(f'{context}: extent must contain four increasing axis endpoints')
        extent = tuple(bounds)
    elif array is not None:
        extent = (0, array.shape[1], 0, array.shape[0])
    levels = validate_range(levels, context=context, field='levels')
    if levels is None and array is not None:
        low, high = float(array.min()), float(array.max())
        if low == high:
            delta = max(abs(low) * .01, 1e-6)
            low, high = low - delta, high + delta
        levels = (low, high)
    if not isinstance(colormap, str):
        raise TypeError(f'{context}: colormap must be a local color map name')
    if colormap not in pg.colormap.listMaps():
        raise ValueError(f'{context}: colormap {colormap!r} is not locally available')
    # Skip the library cache so each cell owns an unshared ColorMap instance.
    color_map = pg.colormap.get(colormap, skipCache=True)
    return dict(values=array, extent=extent, levels=levels, colormap=color_map)
