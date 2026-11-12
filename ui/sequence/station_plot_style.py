"""Translate validated display styles into pyqtgraph pens and axes."""

import pyqtgraph as pg
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from ui.sequence.sampled_plot_curve import SampledPlotCurve


def expand_plot_icon():
    """Two outward arrows, drawn at high DPI without an external image asset."""
    image = QPixmap(32, 32)
    image.setDevicePixelRatio(2)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor('#5D7691'), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    for x1, y1, x2, y2 in ((9, 7, 13, 3), (8, 3, 13, 3), (13, 3, 13, 8),
                            (7, 9, 3, 13), (3, 8, 3, 13), (3, 13, 8, 13)):
        painter.drawLine(x1, y1, x2, y2)
    painter.end()
    return QIcon(image)


PEN_STYLES = {
    'solid': Qt.SolidLine, 'dashed': Qt.DashLine,
    'dotted': Qt.DotLine, 'dash_dot': Qt.DashDotLine,
}


class StyledGridAxis(pg.AxisItem):
    grid_pen = None
    hide_top_grid_line = False

    def generateDrawSpecs(self, painter):
        specs = super().generateDrawSpecs(painter)
        if specs is not None:
            axis, ticks, text = specs
            if self.grid and self.orientation == 'left' and self.hide_top_grid_line:
                view = self.linkedView()
                top = self.mapFromScene(view.mapViewToScene(QPointF(0, view.viewRange()[1][1]))).y()
                ticks = [(p, start, end) for p, start, end in ticks
                         if abs(start.y() - top) > 0.5 or abs(end.y() - top) > 0.5]
            if self.grid and self.grid_pen is not None:
                styled_ticks = []
                for native_pen, start, end in ticks:
                    grid_pen = QPen(self.grid_pen)
                    color = grid_pen.color()
                    # Retain pyqtgraph's density/level fading instead of making
                    # every minor grid line as prominent as a major line.
                    color.setAlpha(native_pen.color().alpha())
                    grid_pen.setColor(color)
                    styled_ticks.append((grid_pen, start, end))
                ticks = styled_ticks
            if self.orientation == 'left':
                # Fixed ranges put endpoint labels exactly on the plot edge.
                # Keep those labels inside the axis paint clip in narrow panels.
                bounds = self.mapRectFromParent(self.parentItem().rect()).intersected(self.boundingRect())
                for rect, _, _ in text:
                    if rect.top() < bounds.top():
                        rect.moveTop(bounds.top() + 1)
                    if rect.bottom() > bounds.bottom():
                        rect.moveBottom(bounds.bottom() - 1)
            return axis, ticks, text
        return specs


def pen(settings):
    return pg.mkPen(settings['color'], width=settings['line_width'],
                    style=PEN_STYLES[settings['line_style']])


def apply_style(widget, style):
    widget.setBackground(style['background_color'])
    plot = widget.getPlotItem()
    grid = style['grid']
    for axis_name in ('left', 'bottom'):
        axis = plot.getAxis(axis_name)
        axis.grid_pen = pen(grid)
        axis.picture = None
        axis.update()
    plot.showGrid(x=grid['visible'], y=grid['visible'], alpha=0.45)


def draw_curves(cell, raw_series, snapshots):
    style = cell._display_style
    plot = cell.plot_widget.getPlotItem()
    channel_styles = {item['wav_channel']: item for item in style['channel_curve_styles']}
    legend = style['legend']
    if legend['visible'] and any(item['x'].size for item in snapshots):
        item = plot.addLegend(labelTextColor='#64748B')
        right = legend['position'].endswith('right')
        bottom = legend['position'].startswith('bottom')
        anchor = (int(right), int(bottom))
        item.anchor(anchor, anchor, offset=(-10 if right else 10, -10 if bottom else 10))
    for index, (raw, data) in enumerate(zip(raw_series, snapshots)):
        if not data['x'].size:
            continue
        role = raw.get('role', 'curve')
        if role == 'exceedance':
            highlight = style['exceedance_highlight']
            if not highlight['visible']:
                continue
            curve_pen = pen(highlight)
        elif role != 'curve':
            if not style['limit_lines']['visible']:
                continue
            curve_pen = pen(style['limit_lines'][role])
        else:
            channel = channel_styles.get(raw.get('wav_channel'), {})
            curve_pen = pen({
                'color': channel.get('curve_color', data['color'] or cell._curve_colors[(raw.get('wav_channel', index + 1) - 1) % len(cell._curve_colors)]),
                'line_style': channel.get('curve_line_style', style['curve_line_style']),
                'line_width': channel.get('curve_line_width', style['curve_line_width']),
            })
        sampling = getattr(cell, '_display_downsampling', None)
        if sampling is not None and role == 'curve':
            plot.addItem(SampledPlotCurve(data['x'], data['y'], sampling,
                                         name=data['name'], pen=curve_pen))
        else:
            options = {}
            if role == 'exceedance':
                # A violation can consist of one sample, which has no line segment.
                options = dict(symbol='o', symbolSize=highlight['marker_size'],
                               symbolPen=None, symbolBrush=highlight['color'])
            plot.plot(data['x'], data['y'], name=data['name'], pen=curve_pen, connect='finite', **options)
