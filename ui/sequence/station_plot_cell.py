"""One independent, caller-fed curve or heatmap plot."""

import pyqtgraph as pg
from PyQt5.QtCore import QPoint, QPointF, QRectF
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY_NAME
from ui.sequence.station_plot_header import StationPlotHeader, StationPlotSubheader
from ui.sequence.station_plot_validation import (
    validate_curve_series,
    validate_heatmap_data,
    validate_plot_config,
    validate_plot_header,
    validate_plot_subheader,
)


class _BottomAxis(pg.AxisItem):
    def generateDrawSpecs(self, painter):
        specs = super().generateDrawSpecs(painter)
        if specs is None:
            return None
        axis_spec, tick_specs, text_specs = specs

        # Shift only text that would cross the cell edge or AxisItem's paint clip.
        plot_bounds = self.mapRectFromParent(self.parentItem().rect())
        bounds = plot_bounds.intersected(self.boundingRect()).adjusted(1, 0, -1, 0)
        for spec in text_specs[:]:
            rect = spec[0]
            at_right = rect.right() > bounds.right()
            if not at_right and rect.left() >= bounds.left():
                continue

            clamped = QRectF(rect)
            if at_right:
                clamped.moveRight(bounds.right())
            else:
                clamped.moveLeft(bounds.left())
            others = [other for other, _, _ in text_specs if other is not rect]
            collisions = [other for other in others if clamped.intersects(other)]
            if bounds.contains(clamped) and (
                    not collisions or self._make_room(clamped, at_right, bounds, others, collisions)):
                rect.moveLeft(clamped.left())
            else:
                text_specs.remove(spec)

        # Keep the first label in pyqtgraph's priority order when labels overlap.
        readable_text_specs = []
        for rect, flags, text in text_specs:
            if not any(rect.intersects(other) for other, _, _ in readable_text_specs):
                readable_text_specs.append((rect, flags, text))
        return axis_spec, tick_specs, readable_text_specs

    @staticmethod
    def _make_room(clamped, at_right, bounds, others, collisions):
        # Keep the endpoint and its neighbor when there is spare space on the inside.
        if len(collisions) != 1:
            return False
        neighbor = collisions[0]
        if not bounds.contains(neighbor):
            return False
        shifted = QRectF(neighbor)
        if at_right:
            shifted.moveRight(clamped.left() - 1)
        else:
            shifted.moveLeft(clamped.right() + 1)
        if not bounds.contains(shifted) or any(
                shifted.intersects(other) for other in others if other is not neighbor):
            return False
        neighbor.moveLeft(shifted.left())
        return True


class StationPlotCell(QWidget):
    def __init__(self, parent=None, *, row=0, column=0):
        super().__init__(parent)
        self.row = row
        self.column = column
        self.kind = 'curve'
        self._configuration = validate_plot_config(kind='curve', context=self._context('configure_plot'))
        self._colorbar = None
        self._image = None
        self._series = []
        self._heatmap = None
        self._curve_colors = ('#2478C8', '#E68A2E', '#2C9C74', '#9A63C3', '#D5586B', '#499BAA')
        self.setMinimumSize(280, 180)
        self.plot_widget = pg.PlotWidget(
            parent=self, background='white', axisItems={'bottom': _BottomAxis('bottom')})
        plot = self.plot_widget.getPlotItem()
        # Keep endpoint tick text inside the viewport and away from grid borders.
        # This is internal plot padding; adjacent grid cells still touch.
        plot.layout.setContentsMargins(7, 8, 7, 5)
        plot.setTitle(None)
        plot.getViewBox().setDefaultPadding(0)
        axis_font = QFont(MAIN_UI_SMALL_FONT_FAMILY_NAME)
        axis_font.setPixelSize(16)
        for name in ('left', 'bottom'):
            axis = plot.getAxis(name)
            axis.setPen('#AFC0D6')
            axis.label.setFont(axis_font)
            axis.setTextPen('#64748B')
            axis.setTickFont(axis_font)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = StationPlotHeader(self)
        layout.addWidget(self.header)
        self.subheader = StationPlotSubheader(self)
        layout.addWidget(self.subheader)
        layout.addWidget(self.plot_widget, 1)
        plot.getViewBox().geometryChanged.connect(self._sync_subheader_geometry)
        self.subheader.resized.connect(self._sync_subheader_geometry)
        self._sync_subheader_geometry()

    def _context(self, method):
        return f'{method} cell ({self.row}, {self.column})'

    def configure_plot(self, *, kind, title='', right_text='', x_label='', y_label='', x_unit='',
                       y_unit='', x_range=None, y_range=None, subheader_left_text='',
                       subheader_first_half_text='', subheader_second_half_text=''):
        config = validate_plot_config(
            kind=kind, title=title, right_text=right_text, x_label=x_label, y_label=y_label, x_unit=x_unit,
            y_unit=y_unit, x_range=x_range, y_range=y_range,
            subheader_left_text=subheader_left_text, subheader_first_half_text=subheader_first_half_text,
            subheader_second_half_text=subheader_second_half_text,
            context=self._context('configure_plot'))
        if kind != self.kind:
            self._clear_data()
        self.kind = kind
        self._configuration = config
        plot = self.plot_widget.getPlotItem()
        self.header.set_texts(title, right_text)
        self.subheader.set_texts(subheader_left_text, subheader_first_half_text, subheader_second_half_text)
        plot.setLabel('bottom', text=x_label, units=x_unit)
        plot.setLabel('left', text=y_label, units=y_unit)
        self._apply_ranges()

    def set_plot_header(self, left_text='', right_text=''):
        validate_plot_header(left_text, right_text, context=self._context('set_plot_header'))
        self._configuration = dict(self._configuration, title=left_text, right_text=right_text)
        self.header.set_texts(left_text, right_text)

    def set_plot_subheader(self, left_text='', first_half_text='', second_half_text=''):
        validate_plot_subheader(left_text, first_half_text, second_half_text,
                                context=self._context('set_plot_subheader'))
        self._configuration = dict(self._configuration, subheader_left_text=left_text,
                                   subheader_first_half_text=first_half_text,
                                   subheader_second_half_text=second_half_text)
        self.subheader.set_texts(left_text, first_half_text, second_half_text)

    def _sync_subheader_geometry(self):
        view = self.plot_widget.getViewBox()
        bounds = view.mapRectToScene(view.rect())
        origin = self.plot_widget.viewport().mapTo(self, QPoint()).x() - self.subheader.x()
        left = self.plot_widget.mapFromScene(QPointF(bounds.left(), bounds.top())).x() + origin
        right = self.plot_widget.mapFromScene(QPointF(bounds.right(), bounds.top())).x() + origin
        self.subheader.set_plot_span(left, right)

    def set_curve_data(self, series):
        context = self._context('set_curve_data')
        if self.kind != 'curve':
            raise ValueError(f'{context}: kind must be curve')
        snapshot = validate_curve_series(series, context=context)
        self._clear_data()
        self._series = snapshot
        plot = self.plot_widget.getPlotItem()
        if sum(item['x'].size > 0 for item in snapshot) > 1:
            plot.addLegend(labelTextColor='#64748B')
        for index, item in enumerate(snapshot):
            if item['x'].size:
                color = item['color'] or self._curve_colors[index % len(self._curve_colors)]
                plot.plot(item['x'], item['y'], name=item['name'], pen=pg.mkPen(color, width=1.5))
        self._apply_ranges()

    def set_heatmap_data(self, values, *, extent=None, levels=None, colormap='viridis'):
        context = self._context('set_heatmap_data')
        if self.kind != 'heatmap':
            raise ValueError(f'{context}: kind must be heatmap')
        snapshot = validate_heatmap_data(
            values, extent=extent, levels=levels, colormap=colormap, context=context)
        self._clear_data()
        if snapshot['values'] is not None:
            self._heatmap = snapshot
            plot = self.plot_widget.getPlotItem()
            self._image = pg.ImageItem(axisOrder='row-major')
            self._image.setImage(snapshot['values'], autoLevels=False, levels=snapshot['levels'])
            x_min, x_max, y_min, y_max = snapshot['extent']
            self._image.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
            plot.addItem(self._image)
            self._colorbar = pg.ColorBarItem(
                values=snapshot['levels'], colorMap=snapshot['colormap'], interactive=False)
            self._colorbar.axis.setPen('#AFC0D6')
            self._colorbar.axis.setTextPen('#64748B')
            self._colorbar.setImageItem(self._image, insert_in=plot)
        self._apply_ranges()

    def clear_plot(self):
        self._clear_data()
        self._apply_ranges()

    def _clear_data(self):
        plot = self.plot_widget.getPlotItem()
        if self._colorbar is not None:
            bar = self._colorbar
            bar.setImageItem([])
            plot.layout.removeItem(bar)
            # ColorBarItem.close needs its scene while releasing axes/ViewBox.
            scene = bar.scene()
            bar.close()
            bar.setParentItem(None)
            scene.removeItem(bar)
            bar.deleteLater()
            plot.layout.setColumnFixedWidth(4, 0)
            self._colorbar = None
        plot.clear()
        if plot.legend is not None:
            legend = plot.legend
            legend.clear()
            scene = legend.scene()
            # QGraphicsScene removes the parent relationship too; calling the
            # LegendItem override with None would try to anchor to that parent.
            scene.removeItem(legend)
            legend.deleteLater()
            plot.legend = None
        self._image = None
        self._series = []
        self._heatmap = None

    def _apply_ranges(self):
        view = self.plot_widget.getViewBox()
        has_data = self._image is not None or any(item['x'].size for item in self._series)
        for axis, field in enumerate(('x_range', 'y_range')):
            configured = self._configuration[field]
            if configured is not None:
                view.setRange(**{('xRange', 'yRange')[axis]: configured}, padding=0)
            else:
                if not has_data:
                    view.setRange(**{('xRange', 'yRange')[axis]: (0, 1)}, padding=0)
                view.enableAutoRange(axis=axis, enable=True)
        view.updateAutoRange()
