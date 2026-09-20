import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

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
    positions = np.linspace(0, 3, 100)
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


def test_zoom_streaming_and_original_data(qapp, monkeypatch):
    widget = pg.PlotWidget()
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
    vb.setXRange(0.1, 0.1005, padding=0)
    qapp.processEvents()
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

    def forbidden(*args):
        pytest.fail("Streaming must never evaluate sinc interpolation")

    with monkeypatch.context() as patch:
        patch.setattr("ui.adaptive_waveform.bandlimited_values", forbidden)
        item.setData(x, y, streaming=True)
        vb.setXRange(0.2, 0.2005, padding=0)
        qapp.processEvents()
        assert not item._smooth
        assert item.curve.isVisible()
    item.setData(x, y, streaming=False)
    assert item._smooth
    vb.setXRange(0, 1, padding=0)
    qapp.processEvents()
    assert not item._smooth
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
