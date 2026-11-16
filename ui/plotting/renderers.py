"""Render prepared data without owning a window, judgement, or configuration.

Coordinates and data are caller supplied. In particular, a caller that maps a
condition onto a shared canvas must map all its curves before rendering them.
These functions never clear a canvas or change its axes/range.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt

from ui.plotting.sampled_curve import SampledPlotCurve


def analysis_curve_pen(role, colors, analysis_type=''):
    """Use the analysis-window pens; FFT has a distinct exceedance emphasis."""
    if role == 'baseline':
        return pg.mkPen('#808080', width=2)
    if role == 'exceedance':
        return pg.mkPen('#F44336' if analysis_type == 'FFT' else 'r',
                        width=3 if analysis_type == 'FFT' else 2)
    field = 'main_curve_color' if role == 'curve' else role + '_limit_color'
    return pg.mkPen(colors[field], width=2,
                    style=Qt.SolidLine if role == 'curve' else Qt.DashLine)


def draw_curve(plot, *data, sampling=None, **options):
    """Draw a curve, optionally retaining full data for viewport sampling."""
    if sampling is None:
        return plot.plot(*data, **options)
    item = SampledPlotCurve(*data, settings=sampling, **options)
    plot.addItem(item)
    return item


def draw_exceedance(plot, x, y, mask, *, color='r', width=2,
                    marker_size=None):
    """Draw an existing verdict mask; no threshold comparisons occur here.

    NaN-separated runs preserve the analysis window's original line topology.
    Optional markers make isolated offending samples visible on small canvases.
    """
    if not np.any(mask):
        return None
    x, y = np.asarray(x), np.asarray(y)
    changes = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    parts_x, parts_y = [], []
    for start, end in zip(starts, ends):
        parts_x.extend(x[start:end].tolist())
        parts_y.extend(y[start:end].tolist())
        parts_x.append(np.nan)
        parts_y.append(np.nan)
    options = {}
    if marker_size is not None:
        options.update(symbol='o', symbolSize=marker_size, symbolPen=None,
                       symbolBrush=color)
    return draw_curve(plot, np.asarray(parts_x[:-1]), np.asarray(parts_y[:-1]),
                      pen=pg.mkPen(color=color, width=width), connect='finite', **options)


def draw_scalar_limits(plot, x_bounds, upper=None, lower=None, *, colors, visible=True):
    """Draw scalar limits over one region, without treating peaks as violations."""
    items = []
    if not visible:
        return items
    for value, side in ((upper, 'upper'), (lower, 'lower')):
        if value is not None and np.isfinite(value):
            items.append(draw_curve(
                plot, x_bounds, [value, value],
                pen=analysis_curve_pen(side, colors),
                name='Overall ' + side + ' limit'))
    return items


def draw_bars(plot, **options):
    item = pg.BarGraphItem(**options)
    plot.addItem(item)
    return item


def draw_scatter(plot, **options):
    item = pg.ScatterPlotItem(**options)
    plot.addItem(item)
    return item


def draw_text(plot, text, position, **options):
    item = pg.TextItem(text, **options)
    item.setPos(*position)
    plot.addItem(item)
    return item


def draw_fill(plot, x, top, bottom, **options):
    upper = pg.PlotDataItem(x, top)
    lower = pg.PlotDataItem(x, bottom)
    item = pg.FillBetweenItem(upper, lower, **options)
    plot.addItem(item)
    return item
