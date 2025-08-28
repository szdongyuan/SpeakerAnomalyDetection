import sys
import numpy as np
import pyqtgraph as pg
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

    def mousePressEvent(self, event):
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QmyFigureCanvas()
    window.show()
    sys.exit(app.exec_())
