"""Display a reduced curve while keeping full-data bounds and zoom detail."""

import numpy as np
import pyqtgraph as pg

from ui.sequence.curve_sampling import select_display_indices


class SampledPlotCurve(pg.PlotDataItem):
    def __init__(self, x, y, settings, **kwargs):
        self.full_x, self.full_y = x, y
        self.settings = settings
        self._selected = None
        super().__init__(x, y, connect='finite', **kwargs)

    def refresh_display(self, x_range, width):
        selected = select_display_indices(self.full_x, self.full_y, self.settings, x_range, width)
        if self._selected is None or not np.array_equal(self._selected, selected):
            self._selected = selected
            self.setData(self.full_x[selected], self.full_y[selected], connect='finite')

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        # Auto-ranging must not depend on points omitted by the display policy.
        values, other = (self.full_x, self.full_y) if ax == 0 else (self.full_y, self.full_x)
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.opts['logMode'][ax]:
                values = np.log10(values)
            if self.opts['logMode'][1 - ax]:
                other = np.log10(other)
        valid = np.isfinite(values) & np.isfinite(other)
        if orthoRange is not None:
            valid &= (other >= orthoRange[0]) & (other <= orthoRange[1])
        values = values[valid]
        if not values.size:
            return None, None
        if frac < 1:
            return tuple(np.percentile(values, [50 * (1 - frac), 50 * (1 + frac)]))
        return float(values.min()), float(values.max())
