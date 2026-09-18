import pyqtgraph as pg
from PyQt5.QtCore import QRectF


class WaveformTimeAxis(pg.AxisItem):
    """Keep endpoint text inside the waveform axis without moving its ticks."""

    def __init__(self):
        super().__init__(orientation="bottom")

    def generateDrawSpecs(self, painter):
        specs = super().generateDrawSpecs(painter)
        if specs is None:
            return None
        axis_spec, tick_specs, text_specs = specs
        # Grid-enabled axes clip text to this horizontal drawing area.
        bounds = self.mapRectFromParent(self.geometry())
        # A styled PlotWidget frame can also trim the outer viewport edge.
        view = self.getViewWidget()
        viewport_bounds = self.mapRectFromScene(
            view.mapToScene(view.viewport().rect()).boundingRect()
        )
        bounds = bounds.intersected(viewport_bounds)
        adjusted = []
        for rect, flags, text in text_specs:
            rect = QRectF(rect)
            if rect.left() < bounds.left():
                rect.moveLeft(bounds.left())
            elif rect.right() > bounds.right():
                rect.moveRight(bounds.right())
            adjusted.append((rect, flags, text))
        return axis_spec, tick_specs, adjusted
