import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

from ui.adaptive_waveform import AdaptiveWaveformItem, bandlimited_values


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("frequency", [0.05, 0.25, 0.4])
@pytest.mark.parametrize("phase", [0, np.pi/4, 1.2])
def test_bandlimited_sine_reconstruction(frequency, phase):
    x = np.arange(256)
    y = np.sin(2*np.pi*frequency*x + phase)
    positions = np.linspace(64, 192, 4097)
    result = bandlimited_values(y, positions)
    expected = np.sin(2*np.pi*frequency*positions + phase)
    assert np.max(np.abs(result - expected)) < 0.005
    np.testing.assert_allclose(bandlimited_values(y, x), y, atol=1e-12)


def test_bandlimited_constant_and_file_boundaries():
    positions = np.linspace(-40, 43, 100)
    np.testing.assert_allclose(bandlimited_values(np.ones(4), positions), 1, atol=1e-12)
    assert np.isfinite(bandlimited_values([1, -1], np.linspace(0, 1, 100))).all()


def test_reconstructed_peak_is_in_auto_range(qapp):
    widget = pg.PlotWidget()
    widget.resize(800, 400)
    widget.show()
    x = np.arange(256, dtype=float)
    y = np.sin(2*np.pi*0.25*x + np.pi/4)
    item = AdaptiveWaveformItem(x, y)
    widget.addItem(item)
    widget.setXRange(100, 116, padding=0)
    qapp.processEvents()
    qapp.processEvents()
    assert item._smooth
    bounds = item.dataBounds(1)
    assert bounds[0] < -0.99 and bounds[1] > 0.99
    bottom, top = widget.getViewBox().viewRange()[1]
    assert bottom < -0.99 and top > 0.99
    widget.close()


def test_clear_hides_children_and_readd_recovers(qapp):
    widget = pg.PlotWidget()
    widget.show()
    item = AdaptiveWaveformItem(np.arange(256), np.sin(np.arange(256)))
    widget.addItem(item)
    widget.setXRange(100, 110, padding=0)
    qapp.processEvents()
    assert item._smooth
    item.clear()
    assert all(not child.isVisible() for child in
               [item.curve, item.scatter, item._markers, item._reconstruction])
    widget.removeItem(item)
    widget.addItem(item)
    item.setData(np.arange(256), np.sin(np.arange(256)), streaming=False)
    qapp.processEvents()
    assert item._smooth
    item.setData([], [])
    assert all(not child.isVisible() for child in
               [item.curve, item.scatter, item._markers, item._reconstruction])
    widget.close()


def test_zoom_streaming_and_original_data(qapp, monkeypatch):
    widget = pg.PlotWidget()
    widget.getAxis("left").setWidth(60)
    widget.resize(800, 400)
    widget.show()
    x = np.arange(48000) / 48000
    y = np.sin(2 * np.pi * 1000 * x)
    item = AdaptiveWaveformItem(x, y, pen="k")
    widget.addItem(item)
    vb = widget.getViewBox()
    vb.setRange(xRange=(0, 1), yRange=(-1.2, 1.2), padding=0)
    qapp.processEvents()
    assert not item._smooth
    dt = x[1] - x[0]
    for spacing in (3.5, 4 - 1e-6, 4, 4.5, 6, 12.5, 3.5, 6):
        # Starting at zero avoids cancellation at the exact 4 px boundary.
        vb.setXRange(0, dt * vb.width() / spacing, padding=0)
        qapp.processEvents()
        left, right = vb.viewRange()[0]
        actual_spacing = dt * vb.width() / (right - left)
        assert actual_spacing == pytest.approx(spacing, abs=1e-9)
        if spacing == 4:
            assert actual_spacing == 4
        assert item._smooth
        assert item._markers.isVisible() == (spacing >= 4)
        if spacing >= 4:
            marker_x, marker_y = item._markers.getData()
            indices = np.searchsorted(x, marker_x)
            np.testing.assert_array_equal(marker_x, x[indices])
            np.testing.assert_array_equal(marker_y, y[indices])
            visible = (marker_x >= left) & (marker_x <= right)
            np.testing.assert_array_equal(marker_x[visible], x[(x >= left) & (x <= right)])
    assert item._smooth
    assert item._markers.isVisible()
    assert item._markers.scene() is widget.scene()
    assert not item.curve.isVisible()
    assert item._reconstruction.path().elementCount() < 9000
    np.testing.assert_array_equal(item.xData, x)
    np.testing.assert_array_equal(item.yData, y)
    original_path = item._reconstruction.path()
    item._refresh_smoothing()
    assert item._reconstruction.path() == original_path

    for width, expected in ((450, False), (800, True)):
        widget.resize(width, 400)
        qapp.processEvents()
        left, right = vb.viewRange()[0]
        assert (dt * vb.width() / (right - left) >= 4) == expected
        assert item._smooth
        assert item._markers.isVisible() == expected

    def forbidden(*args):
        pytest.fail("Streaming must never evaluate sinc interpolation")

    with monkeypatch.context() as patch:
        patch.setattr("ui.adaptive_waveform.bandlimited_values", forbidden)
        item.setData(x, y, streaming=True)
        vb.setXRange(0.2, 0.2 + dt * vb.width() / 6, padding=0)
        qapp.processEvents()
        assert not item._smooth
        assert not item._markers.isVisible()
        assert item.curve.isVisible()
    item.setData(x, y, streaming=False)
    assert item._smooth
    assert item._markers.isVisible()
    vb.setXRange(0, 1, padding=0)
    qapp.processEvents()
    assert not item._smooth
    assert not item._markers.isVisible()
    widget.close()


@pytest.mark.parametrize("values", [[], [1], [0, np.nan, 1], [0, np.inf, 1]])
def test_short_and_nonfinite_data_fall_back(qapp, values):
    widget = pg.PlotWidget()
    item = AdaptiveWaveformItem(np.arange(len(values)), np.asarray(values))
    widget.addItem(item)
    widget.setXRange(0, 3, padding=0)
    item._refresh_smoothing()
    assert not item._smooth
    widget.close()


def test_dense_view_preserves_impulse_and_resize_reselects_mode(qapp):
    widget = pg.PlotWidget()
    widget.resize(900, 400)
    widget.show()
    x = np.arange(48000) / 48000
    y = np.zeros(len(x))
    y[12345] = 7
    item = AdaptiveWaveformItem(x, y)
    widget.addItem(item)
    widget.setXRange(0, 1, padding=0)
    qapp.processEvents()
    assert item.getData()[1].max() == 7
    widget.setXRange(0.2, 0.2 + 300/48000, padding=0)
    qapp.processEvents()
    assert item._smooth
    widget.resize(300, 400)
    qapp.processEvents()
    assert not item._smooth
    widget.close()


@pytest.mark.parametrize("length", [48001, 48005])
@pytest.mark.parametrize("position", [0, 24000, -1, -3])
@pytest.mark.parametrize("amplitude", [-7, 7])
def test_dense_peak_buckets_preserve_all_impulses(qapp, length, position, amplitude):
    widget = pg.PlotWidget()
    widget.resize(800, 400)
    widget.show()
    x = np.arange(length, dtype=float)
    y = np.zeros(length)
    y[position] = amplitude
    item = AdaptiveWaveformItem(x, y, streaming=True)
    widget.addItem(item)
    item.setDownsampling(ds=12, auto=False, method="peak")
    for left, right in [(0, length), (length - 100, length)]:
        widget.setXRange(left, right, padding=0)
        qapp.processEvents()
        if left <= x[position] <= right:
            assert amplitude in item.getData()[1]
            assert amplitude in item.curve.getData()[1]
            low, high = item.dataBounds(1)
            assert low <= amplitude <= high
            bottom, top = widget.getViewBox().viewRange()[1]
            assert bottom <= amplitude <= top
    np.testing.assert_array_equal(item.xData, x)
    np.testing.assert_array_equal(item.yData, y)
    raw_x, raw_y = item.getOriginalDataset()
    np.testing.assert_array_equal(raw_x, x)
    np.testing.assert_array_equal(raw_y, y)
    widget.close()


def test_cache_and_local_reconstruction_are_duration_independent(qapp, monkeypatch):
    calls = []

    def spy(y, positions):
        calls.append((len(y), len(positions)))
        return bandlimited_values(y, positions)

    monkeypatch.setattr("ui.adaptive_waveform.bandlimited_values", spy)
    widget = pg.PlotWidget()
    widget.resize(1600, 400)
    widget.show()
    vb = widget.getViewBox()
    vb.setRange(xRange=(1, 1.001), yRange=(-2, 2), padding=0)
    qapp.processEvents()
    item = AdaptiveWaveformItem()
    widget.addItem(item)
    sizes = []
    for seconds in (10, 60):
        x = np.arange(seconds * 48000) / 48000
        y = np.sin(2 * np.pi * 1000 * x)
        calls.clear()
        item.setData(x, y)
        qapp.processEvents()
        assert calls
        assert all(n <= 4096 + 64 and m <= 12288 for n, m in calls)
        sizes.append(calls[-1])
        calls.clear()
        for _ in range(3):
            item._refresh_smoothing()
            qapp.processEvents()
        assert calls == []
    assert sizes[0] == sizes[1]
    vb.setXRange(1.1, 1.101, padding=0)
    qapp.processEvents()
    assert calls
    calls.clear()
    widget.resize(800, 400)
    qapp.processEvents()
    assert calls
    widget.close()


@pytest.mark.parametrize("mode", ["fft", "log", "derivative", "phasemap", "nonuniform", "remote_nonuniform"])
def test_unsupported_data_keeps_native_display(qapp, monkeypatch, mode):
    widget = pg.PlotWidget()
    widget.show()
    x = np.arange(256, dtype=float) + 1
    y = 2 + np.sin(x)
    if mode == "nonuniform":
        x[100] += 0.2
    elif mode == "remote_nonuniform":
        x[240] += 0.2
    item = AdaptiveWaveformItem(x, y, streaming=True)
    widget.addItem(item)
    widget.setXRange(95, 110, padding=0)
    if mode == "fft":
        item.setFftMode(True)
    elif mode == "log":
        item.setLogMode(False, True)
    elif mode == "derivative":
        item.setDerivativeMode(True)
    elif mode == "phasemap":
        item.setPhasemapMode(True)

    def forbidden(*args):
        pytest.fail("Unsupported sampling/mapping must not interpolate")

    monkeypatch.setattr("ui.adaptive_waveform.bandlimited_values", forbidden)
    item.setData(x, y, streaming=False)
    qapp.processEvents()
    assert not item._smooth
    assert item.curve.isVisible()
    widget.close()


@pytest.mark.parametrize("gap", [np.nan, np.inf, -np.inf])
def test_dense_nonfinite_gaps_preserve_finite_connect(qapp, gap):
    widget = pg.PlotWidget()
    x = np.arange(101, dtype=float)
    y = np.ones(101)
    y[50] = gap
    y[-1] = -7
    item = AdaptiveWaveformItem(x, y)
    widget.addItem(item)
    item.setDownsampling(ds=12, auto=False, method="peak")
    widget.setXRange(0, 101, padding=0)
    qapp.processEvents()
    display_x, display_y = item.curve.getData()
    np.testing.assert_array_equal(display_x, x)
    np.testing.assert_array_equal(display_y, y)
    assert item.curve.opts["connect"] == "finite"
    widget.close()


@pytest.mark.parametrize("gap", [np.inf, -np.inf])
def test_zoomed_y_dynamic_limit_preserves_nonfinite_gap(qapp, gap):
    widget = pg.PlotWidget()
    widget.resize(800, 400)
    widget.setDownsampling(ds=12, auto=False, mode="peak")
    widget.setClipToView(True)
    widget.show()
    x = np.arange(101, dtype=float)
    y = np.zeros(101)
    y[50] = gap
    y[60] = 7
    item = AdaptiveWaveformItem(x, y, streaming=True)
    widget.addItem(item)
    widget.getViewBox().setRange(xRange=(0, 100), yRange=(-1e-7, 1e-7), padding=0)
    qapp.processEvents()
    for displayed in (item.getData(), item.curve.getData()):
        display_x, display_y = displayed
        assert not np.isfinite(display_y[display_x == 50]).any()
        assert display_y[display_x == 50].size == 1
    assert item.curve.opts["connect"] == "finite"
    np.testing.assert_array_equal(item.yData, y)
    assert item.opts["dynamicRangeLimit"] == 1e6
    widget.close()


def test_hysteresis_and_streaming_resize_never_reconstruct(qapp, monkeypatch):
    widget = pg.PlotWidget()
    widget.resize(800, 400)
    widget.show()
    vb = widget.getViewBox()
    vb.setRange(xRange=(0, 10000), yRange=(-2, 2), padding=0)
    item = AdaptiveWaveformItem(np.arange(10000), np.sin(np.arange(10000)))
    widget.addItem(item)
    qapp.processEvents()
    for spacing, expected in [(1.8, False), (2.1, True), (1.8, True), (1.4, False)]:
        vb.setXRange(1000, 1000 + vb.width() / spacing, padding=0)
        qapp.processEvents()
        assert item._smooth == expected
    x, y = item.xData.copy(), item.yData.copy()

    def forbidden(*args):
        pytest.fail("Streaming zoom/resize must not interpolate")

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr("ui.adaptive_waveform.bandlimited_values", forbidden)
            item.setData(x, y, streaming=True)
            vb.setXRange(1000, 1020, padding=0)
            widget.resize(900, 400)
            qapp.processEvents()
            item._refresh_smoothing()
            assert not item._smooth
        item.setData(x, y, streaming=False)
        qapp.processEvents()
        assert item._smooth
        widget.resize(800, 400)
        qapp.processEvents()
    widget.close()


def test_dynamic_range_cache_does_not_reduce_twice(qapp):
    widget = pg.PlotWidget()
    widget.show()
    vb = widget.getViewBox()
    vb.setRange(xRange=(0, 400), yRange=(-1, 1), padding=0)
    item = AdaptiveWaveformItem(np.arange(401), np.arange(401) * 1e9, streaming=True)
    widget.addItem(item)
    item.setDownsampling(ds=12, auto=False, method="peak")
    item.setDynamicRangeLimit(2)
    qapp.processEvents()
    expected_size = len(item.getData()[0])
    assert expected_size == 68
    for extent in (1.1, 1.2, 1.3):
        vb.setYRange(-extent, extent, padding=0)
        qapp.processEvents()
        assert len(item.getData()[0]) == expected_size
    widget.close()


@pytest.mark.parametrize("amplitude", [-7, 7])
def test_automatic_peak_reduction_and_view_clipping(qapp, amplitude):
    widget = pg.PlotWidget()
    # PlotItem applies its own controls to every item on addItem.
    widget.setDownsampling(auto=True, mode="peak")
    widget.setClipToView(True)
    widget.resize(800, 400)
    widget.show()
    x = np.arange(48005, dtype=float)
    y = np.zeros(len(x))
    y[-3] = amplitude
    item = AdaptiveWaveformItem(x, y, streaming=True)
    widget.addItem(item)
    widget.setXRange(0, len(x), padding=0)
    qapp.processEvents()
    assert amplitude in item.curve.getData()[1]
    assert len(item.getData()[0]) < len(x) // 2
    widget.setXRange(47000, len(x), padding=0)
    qapp.processEvents()
    display_x, display_y = item.getData()
    assert amplitude in display_y
    assert display_x[0] >= 46990
    assert len(display_x) < 1100
    widget.close()
