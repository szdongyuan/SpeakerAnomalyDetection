"""Display-only, zoom-dependent smoothing for uniformly sampled audio."""

import numpy as np
from PyQt5.QtWidgets import QGraphicsPathItem
import pyqtgraph as pg
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataset


SINC_RADIUS = 32


def bandlimited_values(y, positions):
    """Evaluate a normalized 64-tap Lanczos-windowed sinc in sample coordinates.

    Endpoint samples are extended constantly outside the file. This is a finite
    approximation to bandlimited reconstruction, not a claim about true peaks.
    """
    positions = np.asarray(positions, dtype=float)
    indices = np.floor(positions).astype(np.int64)[:, None] + np.arange(-SINC_RADIUS + 1, SINC_RADIUS + 1)
    distance = positions[:, None] - indices
    weights = np.sinc(distance) * np.sinc(distance / SINC_RADIUS)
    weights[np.abs(distance) >= SINC_RADIUS] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(np.asarray(y)[np.clip(indices, 0, len(y) - 1)] * weights, axis=1)


class AdaptiveWaveformItem(pg.PlotDataItem):
    """Keep original data for bounds/export and smooth only sparse visible samples.

    Audio time axes must be increasing and uniformly sampled. Streaming bypasses
    interpolation entirely. Dense views include every partial peak bucket.
    """

    def __init__(self, *args, streaming=False, **kwargs):
        self.streaming = streaming
        self._smooth = False
        self._path_key = None
        self._reconstruction = None
        self._peak_display = None
        self._peak_source = None
        self._uniform_sampling = None
        kwargs.setdefault("clipToView", True)
        kwargs.setdefault("autoDownsample", True)
        kwargs.setdefault("downsampleMethod", "peak")
        kwargs.setdefault("connect", "finite")
        super().__init__(*args, **kwargs)
        self._reconstruction = QGraphicsPathItem(self)
        self._markers = pg.ScatterPlotItem(size=4, pen=None)
        self._markers.setParentItem(self)
        self._refresh_smoothing()

    def setData(self, *args, **kwargs):
        self.streaming = kwargs.pop("streaming", getattr(self, "streaming", False))
        self._path_key = None
        self._uniform_sampling = None
        super().setData(*args, **kwargs)

    def clear(self):
        super().clear()
        self._path_key = None
        self._peak_display = None
        self._peak_source = None
        self._smooth = False
        self.curve.hide()
        self.scatter.hide()
        if self._reconstruction is not None:
            self._reconstruction.hide()
            self._markers.hide()

    def updateItems(self, styleUpdate=True):
        super().updateItems(styleUpdate=styleUpdate)
        self._refresh_smoothing()

    def _mapped_mode(self):
        return (self.opts.get("fftMode", False)
                or any(self.opts.get("logMode", (False, False)))
                or self.opts.get("derivativeMode", False)
                or self.opts.get("phasemapMode", False))

    def _getDisplayDataset(self):
        """Adapt the 0.13.x reduction hook without duplicating its pipeline.

        An unrecognized method skips only reduction in PlotDataItem; its normal
        mapping, density calculation, clipping and dynamic-range limiting still
        run. Restore the public option even if the dependency raises. Keep the
        dependency cache unreduced: its dynamic-range hysteresis reuses it.
        Cache our result by source identity to follow its normal invalidation.
        Mapped plots retain the dependency's native behavior. Nonfinite data
        retains every clipped sample, preserving finite-connect gaps.
        """
        if self.opts["downsampleMethod"] != "peak" or self._mapped_mode():
            return super()._getDisplayDataset()
        # Test the original values before the dependency's range limiter can
        # turn +/-Inf into finite values and erase a finite-connect break.
        # PlotDataset owns this flag, so the scan happens only once per dataset.
        if self._dataset is not None and self._dataset.yAllFinite is None:
            self._dataset.yAllFinite = bool(np.isfinite(self._dataset.y).all())
        range_limit = self.opts["dynamicRangeLimit"]
        if self._dataset is not None and not self._dataset.yAllFinite:
            self.opts["dynamicRangeLimit"] = None
        self.opts["downsampleMethod"] = "_adaptive_peak"
        try:
            dataset = super()._getDisplayDataset()
        finally:
            self.opts["downsampleMethod"] = "peak"
            self.opts["dynamicRangeLimit"] = range_limit
        if dataset is None:
            return None
        if dataset is self._peak_source:
            return self._peak_display
        self._peak_source = dataset
        ds = self._adsLastValue if self.opts["autoDownsample"] else self.opts["downsample"]
        x, y = dataset.x, dataset.y
        if (isinstance(ds, int) and ds > 1 and len(y)
                and np.isfinite(x).all() and np.isfinite(y).all()):
            starts = np.arange(0, len(y), ds)
            ends = np.minimum(starts + ds, len(y))
            centers = (starts + ends - 1) // 2
            peaks = np.maximum.reduceat(y, starts)
            troughs = np.minimum.reduceat(y, starts)
            dataset = PlotDataset(np.repeat(x[centers], 2),
                                  np.column_stack((peaks, troughs)).ravel(), True, True)
        self._peak_display = dataset
        return dataset

    def viewRangeChanged(self, vb=None, ranges=None, changed=None):
        # Older PyQtGraph versions briefly expose the PlotWidget during reparenting.
        view = self.getViewBox()
        if view is not None and not isinstance(view, pg.ViewBox):
            return
        super().viewRangeChanged(vb, ranges, changed)
        self._refresh_smoothing()

    def viewTransformChanged(self):
        super().viewTransformChanged()
        self._refresh_smoothing()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        bounds = super().dataBounds(ax, frac=frac, orthoRange=orthoRange)
        if self._smooth:
            # PlotDataItem excludes its hidden straight-line child from bounds.
            bounds = self.curve.dataBounds(ax, frac=frac, orthoRange=orthoRange)
        if ax == 1 and self._smooth and self._path_key is not None and frac == 1.0:
            rect = self._reconstruction.path().boundingRect()
            if bounds[0] is not None and bounds[1] is not None:
                return min(bounds[0], rect.top()), max(bounds[1], rect.bottom())
        return bounds

    def _refresh_smoothing(self):
        if self._reconstruction is None:
            return
        x, y = self.xData, self.yData
        vb = self.getViewBox()
        smooth = False
        spacing = 0.0
        mapped = self._mapped_mode()
        if not self.streaming and not mapped and x is not None and len(x) >= 2:
            if self._uniform_sampling is None:
                # Validate once per static data replacement, never per pan or
                # reconstruction. A remote irregularity also rules out sinc.
                dt = float(x[1]) - float(x[0])
                self._uniform_sampling = (dt > 0 and np.isfinite(x).all()
                                          and np.allclose(np.diff(x), dt, rtol=1e-5, atol=0))
        if (not self.streaming and not mapped and self._uniform_sampling
                and isinstance(vb, pg.ViewBox) and x is not None and len(x) >= 2):
            left, right = vb.viewRange()[0]
            width = vb.width()
            if right > left and width > 0:
                spacing = (x[1] - x[0]) * width / (right - left)
                threshold = 1.5 if self._smooth else 2.0
                if spacing >= threshold:
                    start = max(0, int(np.searchsorted(x, left)) - 1)
                    stop = min(len(x), int(np.searchsorted(x, right, side="right")) + 1)
                    xs, ys = x[start:stop], y[start:stop]
                    halo_start = max(0, start - SINC_RADIUS)
                    halo_stop = min(len(x), stop + SINC_RADIUS)
                    halo_x, halo_y = x[halo_start:halo_stop], y[halo_start:halo_stop]
                    smooth = (2 <= len(xs) <= 4096 and np.isfinite(xs).all()
                              and np.isfinite(halo_y).all()
                              and np.allclose(np.diff(halo_x), x[1] - x[0], rtol=1e-5, atol=0))
        self._smooth = bool(smooth)
        self._reconstruction.setVisible(self._smooth)
        self._markers.setVisible(bool(self._smooth and spacing >= 12))
        self.curve.setVisible(not self._smooth and x is not None and len(x) > 0)
        if not self._smooth:
            return
        pen = pg.mkPen(self.opts["pen"])
        self._reconstruction.setPen(pen)
        key = (id(x), id(y), start, stop, left, right, width, pen.color().rgba())
        if key != self._path_key:
            dt = x[1] - x[0]
            # Bound work by screen width, including original sample positions.
            lo = max(float(x[0]), left - (right - left) / width)
            hi = min(float(x[-1]), right + (right - left) / width)
            times = np.linspace(lo, hi, min(8192, max(2, int(width * 2))))
            times = np.unique(np.concatenate((times, xs)))
            positions = (times - x[halo_start]) / dt
            values = bandlimited_values(halo_y, positions)
            self._reconstruction.setPath(pg.arrayToQPath(times, values, connect="all"))
            self._markers.setData(xs, ys, brush=pen.color())
            self._path_key = key
            self.informViewBoundsChanged()
