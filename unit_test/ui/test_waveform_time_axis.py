import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QFont, QPainter, QPicture

from consts.recording_preview_consts import (
    PLOT_PRESENTATION_COMPLETE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from ui.sequence.analysis_waveform_panel import AnalysisWaveformRow
from ui.sequence.channel_plot_workspace import ChannelPlotSubWindow


@pytest.fixture(params=[AnalysisWaveformRow, ChannelPlotSubWindow])
def waveform_window(request, ui_qapp):
    window_class = request.param
    args = (None, 0, "Front") if window_class is AnalysisWaveformRow else (None, 0)
    window = window_class(*args)
    yield window
    window.close()
    window.deleteLater()
    ui_qapp.processEvents()


def _draw_specs(axis, native=False):
    picture = QPicture()
    painter = QPainter(picture)
    try:
        if axis.style["tickFont"] is not None:
            painter.setFont(axis.style["tickFont"])
        if native:
            return pg.AxisItem.generateDrawSpecs(axis, painter)
        return axis.generateDrawSpecs(painter)
    finally:
        painter.end()


def _assert_endpoints_visible(window):
    plot = window.plot_widget
    axis = plot.getAxis("bottom")
    _, _, text_specs = _draw_specs(axis)
    endpoints = {text: rect for rect, _, text in text_specs if text in ("-10", "0")}
    assert endpoints.keys() == {"-10", "0"}
    clip = QRectF(axis.boundingRect().toAlignedRect())
    viewport = QRectF(plot.viewport().rect())
    transform = axis.deviceTransform(plot.viewportTransform())
    for text, rect in endpoints.items():
        assert rect.left() >= clip.left(), (text, rect, clip)
        assert rect.right() <= clip.right(), (text, rect, clip)
        assert viewport.contains(transform.mapRect(rect)), (text, rect, viewport)


@pytest.mark.parametrize("width", [480, 800, 1340])
@pytest.mark.parametrize("font_pixels", [14, 24])
def test_latest_ten_endpoint_labels_fit_clip_and_viewport(
    waveform_window, ui_qapp, width, font_pixels
):
    window = waveform_window
    window.resize(width, 220)
    font = QFont()
    font.setPixelSize(font_pixels)
    window.plot_widget.getAxis("bottom").setTickFont(font)
    x = np.linspace(-10, 0, 1000)
    y = np.sin(x * 3)
    window.set_relative_preview_data(x, y)
    window.show()
    ui_qapp.processEvents()
    window.grab()  # Exercise the actual Qt painter and settle axis layout.
    ui_qapp.processEvents()

    assert window.plot_widget.viewRange()[0] == [-10, 0]
    np.testing.assert_array_equal(window.plot_item.xData, x)
    np.testing.assert_array_equal(window.plot_item.yData, y)
    assert window.plot_widget.getAxis("bottom").grid
    assert window.plot_widget.getAxis("left").grid
    _assert_endpoints_visible(window)


def test_axis_preserves_native_lines_grid_and_fitting_text(waveform_window, ui_qapp):
    window = waveform_window
    window.resize(800, 220)
    x = np.linspace(-10, 0, 1000)
    window.set_relative_preview_data(x, np.sin(x * 3))
    window.show()
    ui_qapp.processEvents()
    window.grab()
    ui_qapp.processEvents()

    plot = window.plot_widget
    axis = plot.getAxis("bottom")
    native_axis, native_ticks, native_text = _draw_specs(axis, native=True)
    adjusted_axis, adjusted_ticks, adjusted_text = _draw_specs(axis)
    assert adjusted_axis == native_axis
    assert adjusted_ticks == native_ticks
    assert len(adjusted_text) == len(native_text)
    bounds = axis.mapRectFromParent(axis.geometry())
    transform = axis.deviceTransform(plot.viewportTransform())
    viewport = QRectF(plot.viewport().rect())
    fitting_labels = []
    moved_labels = []
    for native, adjusted in zip(native_text, adjusted_text):
        native_rect, native_flags, native_label = native
        rect, flags, label = adjusted
        assert (flags, label) == (native_flags, native_label)
        assert rect.size() == native_rect.size()
        assert rect.top() == native_rect.top()
        if bounds.contains(native_rect) and viewport.contains(transform.mapRect(native_rect)):
            assert rect == native_rect
            fitting_labels.append(label)
        if rect != native_rect:
            moved_labels.append(label)
    assert fitting_labels
    assert set(moved_labels) == {"-10", "0"}


def test_resize_updates_modes_and_snapshot_restore(waveform_window, ui_qapp):
    window = waveform_window
    window.show()
    x = np.linspace(-10, 0, 1000)
    for index, width in enumerate((480, 1340, 800)):
        window.resize(width, 220)
        y = np.sin(x * (index + 1))
        window.set_relative_preview_data(x, y)
        ui_qapp.processEvents()
        window.grab()
        ui_qapp.processEvents()
        assert window.plot_widget.viewRange()[0] == [-10, 0]
        np.testing.assert_array_equal(window.plot_item.xData, x)
        np.testing.assert_array_equal(window.plot_item.yData, y)
        _assert_endpoints_visible(window)

    snapshot = window.snapshot_plot_state()
    elapsed = np.linspace(0, 25, 1000)
    for setter, mode in (
        (window.set_cumulative_preview_data, PREVIEW_TIME_MODE_CUMULATIVE),
        (window.set_data, PLOT_PRESENTATION_COMPLETE),
    ):
        setter(elapsed, y)
        ui_qapp.processEvents()
        assert window.presentation_mode == mode
        assert window.plot_widget.getViewBox().state["autoRange"][0]
        low, high = window.plot_widget.viewRange()[0]
        assert low <= 0 and high >= 25
        np.testing.assert_array_equal(window.plot_item.xData, elapsed)
        np.testing.assert_array_equal(window.plot_item.yData, y)

    window.restore_plot_state(snapshot)
    ui_qapp.processEvents()
    window.grab()
    ui_qapp.processEvents()
    assert window.presentation_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
    assert window.is_live_preview
    assert window.plot_widget.viewRange()[0] == [-10, 0]
    assert not window.plot_widget.getViewBox().state["autoRange"][0]
    np.testing.assert_array_equal(window.plot_item.xData, x)
    np.testing.assert_array_equal(window.plot_item.yData, y)
    _assert_endpoints_visible(window)

    # Explicit pan/zoom still takes ownership after snapshot restoration.
    window.plot_widget.setXRange(-8, -2, padding=0)
    ui_qapp.processEvents()
    assert window.plot_widget.viewRange()[0] == [-8, -2]
