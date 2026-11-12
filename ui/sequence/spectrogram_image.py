"""Paint native spectrogram bins, including nonuniform CQT frequencies."""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt


class SpectrogramImage(pg.GraphicsObject):
    """A cached color image with physical frequency-row placement.

    STFT linear axes use one image draw. Nonuniform/log frequency axes use
    one strip per frequency bin instead of falsely stretching the whole matrix.
    No transform, interpolation, threshold calculation or matrix decimation.
    """

    def __init__(self, values, x_edges, y_edges, levels, color_map, clip_x):
        super().__init__()
        self.x_edges = np.array(x_edges, copy=True)
        self.y_edges = np.array(y_edges, copy=True)
        self.levels = tuple(levels)
        pixels, _ = pg.makeARGB(values, lut=color_map.getLookupTable(nPts=256), levels=levels)
        self.image = pg.makeQImage(pixels, transpose=False)
        self._bounds = QRectF(x_edges[0], y_edges[0], x_edges[-1] - x_edges[0], y_edges[-1] - y_edges[0])
        self._clip = QRectF(clip_x[0], y_edges[0], clip_x[1] - clip_x[0], y_edges[-1] - y_edges[0])
        self._uniform = np.allclose(np.diff(y_edges), y_edges[1] - y_edges[0])

    def boundingRect(self):
        return self._bounds.intersected(self._clip)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setClipRect(self.boundingRect(), Qt.IntersectClip)
        if self._uniform:
            painter.drawImage(self._bounds, self.image)
        else:
            for row, (bottom, top) in enumerate(zip(self.y_edges[:-1], self.y_edges[1:])):
                painter.drawImage(QRectF(self._bounds.left(), bottom, self._bounds.width(), top - bottom),
                                  self.image, QRectF(0, row, self.image.width(), 1))
        painter.restore()
