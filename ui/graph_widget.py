import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout


def custom_log_tick_strings(values, scale, spacing):
    estrings = ["%0.1g" % x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
    convdict = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    dstrings = []
    for i, e in enumerate(estrings):
        if "e" in e:
            v, p = e.split("e")
            sign = "⁻" if p[0] == "-" else ""
            pot = "".join([convdict[pp] for pp in p[1:].lstrip("0")])
            if v == "1":
                v = ""
                dstrings.append(v + "10" + sign + pot)
            elif v == "2" or v == "5":
                v = v + "·"
                dstrings.append(v + "10" + sign + pot)
            else:
                dstrings.append("")
        else:
            dstrings.append(e)
    return dstrings


class QmyFigureCanvas(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.resize(300, 100)

        layout = QHBoxLayout()
        label_graph = QLabel("Show graph here...")
        label_graph.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_graph)

        self.setLayout(layout)


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

    img = pg.ImageItem()
    plot_widget.addItem(img)

    view_box = plot_widget.getViewBox()
    if view_box:
        view_box.setDefaultPadding(0.0)

    img.setImage(z)

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


class ColorBarItem(pg.GraphicsObject):
    def __init__(self, cmap, width=20, height=200, label_side="right"):
        pg.GraphicsObject.__init__(self)
        self.cmap = cmap
        self.width = width
        self.height = height
        self.label_side = label_side

        self.bar = pg.ImageItem()
        self.bar.setImage(np.linspace(0, 1, 256).reshape(256, 1))

        self.bar.setLookupTable(self.cmap.getLookupTable(nPts=256))

        self.axis = pg.AxisItem(orientation="right")
        self.axis.setLabel("Value")

        self.bar.setFixedWidth(self.width)
        self.axis.setFixedWidth(50)

    def paint(self, p, *args):
        pass

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width + 50, self.height)


class DraggablePlotWidget(pg.PlotWidget):
    sigSelectionCancelled = pyqtSignal()

    def __init__(self, region_item, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.region = region_item
        self.is_creating_new_region = False
        self.drag_start_pos = None

        self.region_len = 0
        self.drag_mode = "click_drag"

    def mousePressEvent(self, event):
        if self.drag_mode == "click_drag":
            self._mouse_press_event_click_drag(event)
        elif self.drag_mode == "click":
            self._mouse_press_event_click(event)
            self.is_creating_new_region = False
        else:
            super().mousePressEvent(event)

    def _mouse_press_event_click_drag(self, event):
        if (event.button() == Qt.LeftButton and
                self.getPlotItem().getViewBox().sceneBoundingRect().contains(event.pos())):
            items_under_cursor = self.scene().items(event.pos())
            region_parts = [self.region] + self.region.lines
            is_on_region = any(item in items_under_cursor for item in region_parts)
            if is_on_region:
                self.is_creating_new_region = False
                super().mousePressEvent(event)
            else:
                self.is_creating_new_region = True
                self.drag_start_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
                self.region.hide()
                event.accept()
        else:
            super().mousePressEvent(event)

    def _mouse_press_event_click(self, event):
        if (event.button() == Qt.LeftButton and
                self.getPlotItem().getViewBox().sceneBoundingRect().contains(event.pos())):
            self.region.hide()
            current_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
            self.region.setRegion((current_pos.x(), current_pos.x() + self.region_len))
            self.region.show()
            self.region.sigRegionChanged.emit(current_pos)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_creating_new_region:
            if not self.region.isVisible():
                self.region.show()
            current_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
            self.region.setRegion(sorted([self.drag_start_pos.x(), current_pos.x()]))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_creating_new_region:
            self.is_creating_new_region = False
            current_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
            self.region.setRegion(sorted([self.drag_start_pos.x(), current_pos.x()]))
            start, end = self.region.getRegion()
            if abs(start - end) < 1e-9:
                self.region.hide()
                self.sigSelectionCancelled.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class LimitPlotUtils:
    """
    Limit Check and Plot Utility Class

    Contains 4 static methods for unified limit comparison, out-of-limit segment plotting,
    plot setup, and interpolation check.
    Used by all plotting functions (THD, SPL time-domain, SPLF, FR).
    """

    @staticmethod
    def compare_with_limits(
        plot_y: np.ndarray,
        upper_limits: np.ndarray,
        lower_limits: np.ndarray,
        valid_mask: np.ndarray = None,
    ) -> tuple:
        """
        Common limit comparison (used by SPL time-domain, SPLF, FR).

        Args:
            plot_y: Y values to compare
            upper_limits: Upper limits for each point (same length as plot_y)
            lower_limits: Lower limits for each point (same length as plot_y)
            valid_mask: Validity mask (optional, defaults to all valid)

        Returns:
            out_mask: Out-of-limit mask (True = out of limit)
            deviation: Deviation value (max exceedance if out, min margin if ok)
            is_ok: Whether all points are within limits
        """
        n = len(plot_y)
        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)

        u_ok = np.isfinite(upper_limits)
        l_ok = np.isfinite(lower_limits)

        # Check out-of-limit
        out_mask = valid_mask & (
            (u_ok & (plot_y > upper_limits)) |
            (l_ok & (plot_y < lower_limits))
        )

        # Calculate deviation
        deviation = 0.0
        is_ok = True
        if np.any(out_mask):
            is_ok = False
            dev_upper = np.where(out_mask & u_ok, plot_y - upper_limits, 0.0)
            dev_lower = np.where(out_mask & l_ok, lower_limits - plot_y, 0.0)
            deviation = float(np.nanmax(np.maximum(dev_upper, dev_lower)))
        else:
            # Calculate minimum margin when within limits
            in_range = valid_mask & np.isfinite(plot_y)
            if np.any(in_range):
                margin_u = np.where(u_ok[in_range], upper_limits[in_range] - plot_y[in_range], np.inf)
                margin_l = np.where(l_ok[in_range], plot_y[in_range] - lower_limits[in_range], np.inf)
                margins = np.minimum(margin_u, margin_l)
                margins = margins[np.isfinite(margins)]
                if margins.size > 0:
                    deviation = float(np.min(margins))

        return out_mask, round(deviation, 2), is_ok

    @staticmethod
    def plot_out_segments(
        plot_widget,
        x_data: np.ndarray,
        y_data: np.ndarray,
        out_mask: np.ndarray,
        pen_color: str = "r",
        pen_width: int = 2,
    ):
        """
        Plot out-of-limit segments (used by all 4 functions).

        Uses NaN separation + single plot call for performance.

        Args:
            plot_widget: pyqtgraph plot widget
            x_data: X coordinate array
            y_data: Y coordinate array
            out_mask: Out-of-limit mask (True = out of limit)
            pen_color: Pen color
            pen_width: Pen width
        """
        if not np.any(out_mask):
            return

        # Vectorized find start/end indices of out-of-limit segments
        out_int = out_mask.astype(np.int8)
        diff = np.diff(np.concatenate([[0], out_int, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Use NaN to separate segments
        out_x_all, out_y_all = [], []
        for s, e in zip(starts, ends):
            out_x_all.extend(x_data[s:e].tolist())
            out_x_all.append(np.nan)
            out_y_all.extend(y_data[s:e].tolist())
            out_y_all.append(np.nan)

        if out_x_all:
            plot_widget.plot(
                np.array(out_x_all[:-1]),
                np.array(out_y_all[:-1]),
                pen=mkPen(color=pen_color, width=pen_width),
                connect="finite",
            )

    @staticmethod
    def setup_limit_plot(
        plot_widget,
        data_x: np.ndarray,
        data_y: np.ndarray,
        csv_x: np.ndarray,
        csv_upper: np.ndarray,
        csv_lower: np.ndarray,
        x_label: str = "X",
        y_label: str = "Y",
        log_x: bool = False,
        curve_color: tuple = (51, 196, 77),
        curve_width: int = 2,
        curve_name: str = None,
        curve_colors: dict = None,
    ):
        """
        Common plot setup function (used by all 4 functions).

        Includes: clear canvas, draw main curve, draw limit curves, set axes, show grid.

        Args:
            plot_widget: pyqtgraph plot widget
            data_x: Main curve X coordinates
            data_y: Main curve Y coordinates
            csv_x: Limit curve X coordinates
            csv_upper: Upper limit values
            csv_lower: Lower limit values
            x_label: X axis label
            y_label: Y axis label
            log_x: Whether to use logarithmic X axis
            curve_color: Main curve color, default green (51, 196, 77)
            curve_width: Main curve width
            curve_name: Main curve name (optional, THD uses "THD")
            curve_colors: Optional main/upper/lower curve color mapping
        """
        plot_widget.clear()
        configured_colors = curve_colors or {}
        main_curve_color = configured_colors.get(
            "main_curve_color",
            curve_color,
        )
        upper_limit_color = configured_colors.get(
            "upper_limit_color",
            (128, 0, 128),
        )
        lower_limit_color = configured_colors.get(
            "lower_limit_color",
            (128, 0, 128),
        )

        # 1. Draw main curve (supports name parameter for legend)
        plot_widget.plot(
            data_x,
            data_y,
            pen=mkPen(color=main_curve_color, width=curve_width),
            name=curve_name,
        )

        # 2. Draw limit curves
        upper_pen = mkPen(
            color=upper_limit_color,
            width=2,
            style=Qt.DashLine,
        )
        lower_pen = mkPen(
            color=lower_limit_color,
            width=2,
            style=Qt.DashLine,
        )
        plot_widget.plot(csv_x, csv_upper, pen=upper_pen)
        plot_widget.plot(csv_x, csv_lower, pen=lower_pen)

        # 3. Set axis labels
        plot_widget.setLabel("left", y_label)
        plot_widget.setLabel("bottom", x_label)

        # 4. Set log scale (frequency domain uses it, time domain does not)
        plot_widget.setLogMode(x=log_x, y=False)

        # 5. Show grid
        plot_widget.showGrid(x=True, y=True)

    @staticmethod
    def check_interp_limits(
        data_x: np.ndarray,
        data_y: np.ndarray,
        csv_x: np.ndarray,
        csv_upper: np.ndarray,
        csv_lower: np.ndarray,
    ) -> tuple:
        """
        Complete interpolation limit check (used by SPLF and FR).

        Interpolates CSV limits to original data points, so out-of-limit segments
        are plotted on the original curve (not on CSV frequency grid).

        Args:
            data_x: Original data X (frequency)
            data_y: Original data Y (SPL/FR values)
            csv_x: CSV frequency list
            csv_upper: CSV upper limit list
            csv_lower: CSV lower limit list

        Returns:
            out_mask: Out-of-limit mask (on original data points)
            plot_x: X for plotting (= original data_x, filtered)
            plot_y: Y for plotting (= original data_y, filtered)
            deviation: Deviation value
            is_ok: Whether all points are within limits
        """
        # === 1. Preprocessing original data ===
        mask = np.isfinite(data_x) & np.isfinite(data_y) & (data_x > 0)
        freq = data_x[mask]
        mag = data_y[mask]

        if freq.size < 2:
            return (
                np.zeros(len(data_x), dtype=bool),
                data_x,
                data_y,
                0.0,
                True,
            )

        sort_idx = np.argsort(freq)
        freq, mag = freq[sort_idx], mag[sort_idx]

        # === 2. Interpolate CSV limits to original data points ===
        # Only compare within CSV frequency range
        csv_min, csv_max = csv_x.min(), csv_x.max()
        in_band = (freq >= csv_min) & (freq <= csv_max)

        # Interpolate upper/lower limits to original frequency points
        upper_at_freq = np.full(freq.shape, np.nan)
        lower_at_freq = np.full(freq.shape, np.nan)
        if np.any(in_band):
            upper_at_freq[in_band] = np.interp(freq[in_band], csv_x, csv_upper)
            lower_at_freq[in_band] = np.interp(freq[in_band], csv_x, csv_lower)

        # === 3. Compare using common function ===
        out_mask, deviation, is_ok = LimitPlotUtils.compare_with_limits(
            mag, upper_at_freq, lower_at_freq, valid_mask=in_band
        )

        return out_mask, freq, mag, deviation, is_ok


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QmyFigureCanvas()
    window.show()
    sys.exit(app.exec_())
