"""Uniform image rendering and legacy standalone heatmap composition."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from ui.plotting.spectrogram import SpectrogramImage


def create_image(plot, **options):
    item = pg.ImageItem(**options)
    plot.addItem(item)
    return item


def update_image(item, values, *, rect=None, levels=None, color_map=None,
                 **image_options):
    """Preserve caller-supplied orientation and physical coordinates."""
    item.setImage(values, **image_options)
    if rect is not None:
        item.setRect(rect)
    if color_map is not None:
        item.setLookupTable(color_map.getLookupTable(nPts=256))
    if levels is not None:
        item.setLevels(levels)
    return item


def draw_binned_image(plot, values, x_edges, y_edges, levels, color_map, clip_x):
    item = SpectrogramImage(values, x_edges, y_edges, levels, color_map, clip_x)
    plot.addItem(item)
    return item

def plot_2d_image(
    x,
    y,
    z,
    title="2D Plot",
    xlabel="X",
    ylabel="Y",
    colormap="viridis",
    x_range=None,
    y_range=None,
    z_range=None,
    y_ticks=None,
    background_color=None,
    x_padding=0,
    y_padding=0,
):
    """
    Plot a 2D image with a colorbar in PyQt.

    Parameters:
    -----------
    x : numpy.ndarray
        1D array of x coordinates
    y : numpy.ndarray
        1D array of y coordinates
    z : numpy.ndarray
        2D array of z values where z[i,j] = f(x[i], y[j])
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    colormap : str
        Name of the colormap to use
    x_range : tuple or None
        (min, max) range for x-axis. If None, uses data range.
    y_range : tuple or None
        (min, max) range for y-axis. If None, uses data range.
    z_range : tuple or None
        (min, max) range for colormap. If None, uses data range.
    y_ticks : list or None
        Custom ticks for the y-axis in the format expected by AxisItem.setTicks.
    background_color : str or None
        Background color for the plot widget. If None, uses default.

    Returns:
    --------
    QWidget
        Widget containing the plot, can be added to a PyQt layout
    """
    widget = QWidget()
    layout = QHBoxLayout()
    widget.setLayout(layout)

    plot_widget = pg.PlotWidget(title=title)
    if background_color is not None:
        plot_widget.setBackground(background_color)
    plot_widget.setLabel("bottom", xlabel)
    plot_widget.setLabel("left", ylabel)
    layout.addWidget(plot_widget)

    img = create_image(plot_widget)

    view_box = plot_widget.getViewBox()
    if view_box:
        view_box.setDefaultPadding(0.0)

    update_image(img, z)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    img.setRect(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))

    if z_range is None:
        z_min, z_max = z.min(), z.max()
    else:
        z_min, z_max = z_range

    if z_min == z_max:
        z_max = z_min + 1  # Add a small offset to prevent single color

    pos = np.linspace(0.0, 1.0, 256)
    colors = pg.colormap.get(colormap).getLookupTable(nPts=256)
    cmap = pg.ColorMap(pos, colors)

    lut = cmap.getLookupTable(nPts=256)
    img.setLookupTable(lut)

    img.setLevels([z_min, z_max])

    colorbar = pg.ColorBarItem(values=(z_min, z_max), colorMap=cmap)
    colorbar.setImageItem(img, insert_in=plot_widget.getPlotItem())

    if x_range is not None:
        plot_widget.setXRange(x_range[0], x_range[1])
    else:
        plot_widget.setXRange(x_min, x_max, padding=x_padding)

    if y_range is not None:
        plot_widget.getViewBox().disableAutoRange(axis=pg.ViewBox.YAxis)
        plot_widget.getViewBox().setYRange(y_range[0], y_range[1], padding=y_padding)
    else:
        plot_widget.getViewBox().setYRange(y_min, y_max, padding=y_padding)

    if y_ticks is not None:
        left_axis = plot_widget.getAxis("left")
        try:
            major, minor = y_ticks
            mapped_major = []
            for freq, label in major:
                if y_min <= freq <= y_max:
                    idx = np.interp(freq, y, np.arange(len(y)))
                    pos = y_min + (idx + 0.5) * (y_max - y_min) / len(y)
                    mapped_major.append((pos, label))
            left_axis.setTicks([mapped_major, minor])
        except Exception as e:
            left_axis.setTicks(y_ticks)

    return widget, colorbar
