"""Native curve, frequency-band and matrix strategies; no computation or I/O."""

from numbers import Real

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.sequence.sampled_plot_curve import SampledPlotCurve
from ui.sequence.spectrogram_data import bin_edges
from ui.sequence.spectrogram_image import SpectrogramImage
from ui.sequence.station_plot_validation import validate_curve_series, validate_heatmap_data
from ui.sequence.workbench_units import format_workbench_spl_unit


class RecordAnalysisContent(QWidget):
    """Own all graphics and their full data for exactly one selection."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet('background:white;')
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(10, 8, 10, 8)
        self.content_layout.setSpacing(6)
        self.plots, self.curves, self.images, self.colorbars = [], [], [], []
        self.bars, self.annotations = [], []
        self.summary_labels, self.card_labels, self.diagnostics = [], [], []
        self._connections = []
        self.disposed = False

    def label(self, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content_layout.addWidget(label)
        return label

    def diagnostic_text(self):
        return '\n'.join(label.text() for label in self.diagnostics)

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        for signal, callback in self._connections:
            signal.disconnect(callback)
        self._connections.clear()
        for curve in self.curves:
            curve.full_x = curve.full_y = np.array([], dtype=float)
            curve._selected = None
            curve.clear()
        for image in self.images:
            image.image = None
            image.x_edges = image.y_edges = np.array([], dtype=float)
        for widget in self.plots:
            widget.record_metadata.clear()
            widget.clear()
            widget.hide()
            widget.deleteLater()
        self.curves.clear()
        self.images.clear()
        self.colorbars.clear()
        self.bars.clear()
        self.annotations.clear()
        self.plots.clear()


def _plot(content, panel):
    widget = pg.PlotWidget(background='w')
    widget.setMinimumSize(240, 220)
    # Let a single canvas use the viewport instead of imposing PlotWidget's
    # large preferred height through word-wrapped labels' height-for-width.
    # Multiple canvases retain their minimum height and scroll normally.
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
    widget.record_panel_key = panel.key
    widget.record_x_unit, widget.record_y_unit = panel.x_unit, panel.y_unit
    widget.record_metadata = dict(panel.metadata or {})
    plot = widget.getPlotItem()
    title = {'SPL': '声压级', 'FFT': '频谱', 'FBA': '分频程', 'Spec': '时频图',
             'loudness_time': '响度随时间', 'specific_loudness_profile': '特征响度',
             'specific_loudness': '特征响度随时间'}.get(panel.key, '')
    plot.setTitle(title, color='#46617d', size='11pt')
    plot.showGrid(x=True, y=True, alpha=.15)
    for name in ('left', 'bottom'):
        axis = plot.getAxis(name)
        axis.setPen('#afc0d6')
        axis.setTextPen('#64748b')
        axis.enableAutoSIPrefix(False)
    plot.setLabel('bottom', panel.x_label or panel.x_unit)
    plot.setLabel('left', panel.y_label or panel.y_unit)
    content.plots.append(widget)
    content.content_layout.addWidget(widget, 1)
    return widget


class CurveRenderer:
    def render(self, content, panels, parameters):
        panel = panels[0]
        widget = _plot(content, panel)
        plot = widget.getPlotItem()
        plot.addLegend(labelTextColor='#64748b')
        if panel.key == 'FFT':
            # Match the FFT adapter's normalization, including old queue configs.
            scale = str(parameters.get('x_axis_scale', 'log') or 'log').lower()
            if scale not in {'linear', 'log'}:
                scale = 'log'
            x_log = scale == 'log'
        else:
            x_log = parameters.get('x_axis_scale', parameters.get('x_axis', {}).get('scale')) == 'log'
        y_log = parameters.get('y_axis', {}).get('scale') == 'log'
        plot.setLogMode(x=x_log, y=y_log)
        owned = []
        for current in panels:
            checked = validate_curve_series(current.series, context=panel.key, allow_gaps=True)
            for raw, data in zip(current.series, checked):
                role = raw.get('role', 'curve')
                channel = raw['wav_channel']
                color = pg.intColor(channel - 1, hues=8, values=1, maxValue=190)
                options = {}
                if role == 'exceedance':
                    color = '#dc3545'
                    options.update(symbol='o', symbolSize=5, symbolPen=None, symbolBrush=color)
                elif role in ('upper', 'lower'):
                    color = '#d99a39'
                curve_pen = pg.mkPen(color, width=2 if role == 'curve' else 1.5,
                                     style=Qt.DashLine if role in ('upper', 'lower') else Qt.SolidLine)
                sampling = parameters.get('display_downsampling', {'method': 'preserve_min_max', 'pixel_bucket_width': 1})
                if role != 'curve' or x_log:
                    # Log-domain viewport sampling needs transformed X; retain all original points.
                    sampling = {'method': 'none'}
                curve = SampledPlotCurve(data['x'], data['y'], sampling,
                                         name=data['name'] if role == 'curve' else None,
                                         pen=curve_pen, **options)
                curve.record_role = role
                plot.addItem(curve)
                content.curves.append(curve)
                owned.append(curve)

        def refresh(*_):
            view = widget.getViewBox()
            for curve in owned:
                curve.refresh_display(view.viewRange()[0], max(1, view.width()))

        for signal in (widget.getViewBox().sigXRangeChanged, widget.getViewBox().sigResized):
            signal.connect(refresh)
            content._connections.append((signal, refresh))
        if (panel.metadata or {}).get('zero_based'):
            high = max((np.nanmax(curve.full_y) for curve in owned if curve.full_y.size), default=1)
            plot.setYRange(0, max(float(high) * 1.05, 1e-6), padding=0)
        refresh()


class BandRenderer:
    """Place returned band values and limits on equal-spaced channel groups."""

    def render(self, content, panels, parameters):
        panel = panels[0]
        widget = _plot(content, panel)
        plot = widget.getPlotItem()
        legend = plot.addLegend(labelTextColor='#64748b')
        plot.showGrid(x=False, y=True, alpha=.15)
        widget.record_band_labels = panel.band_labels
        positions = np.arange(len(panel.band_labels), dtype=float)
        plot.getAxis('bottom').setTicks([list(enumerate(panel.band_labels))])
        width = .7 / len(panels)
        normal, exceeded_color, missing = (76, 175, 80), (244, 67, 54), (189, 189, 189)
        for index, current in enumerate(panels):
            checked = validate_curve_series(current.series, context=panel.key, allow_gaps=True)
            series = {raw.get('role', 'curve'): data for raw, data in zip(current.series, checked)}
            main = series['curve']
            channel = next(raw['wav_channel'] for raw in current.series if raw.get('role') == 'curve')
            centers = positions - .35 + width * (index + .5)
            finite = np.isfinite(main['y'])
            exceeded = np.zeros(positions.size, dtype=bool)
            if 'exceedance' in series:
                exceeded = np.isfinite(series['exceedance']['y']) & finite
            brushes = [pg.mkBrush(missing if not valid else exceeded_color if ng else normal)
                       for valid, ng in zip(finite, exceeded)]
            bar = pg.BarGraphItem(x=centers, height=np.where(finite, main['y'], 0.), width=width,
                                  brush=pg.mkBrush(normal), brushes=brushes, pen=pg.mkPen('w', width=.5))
            bar.setToolTip(main['name'])
            plot.addItem(bar)
            legend.addItem(bar, main['name'])
            content.bars.append(bar)
            for role, color, symbol, size in (
                    ('upper', (0, 188, 212), 'o', 4), ('lower', (63, 81, 181), 't', 6)):
                if role not in series:
                    continue
                curve = SampledPlotCurve(centers, series[role]['y'], {'method': 'none'},
                                         pen=pg.mkPen(color, width=2), symbol=symbol,
                                         symbolSize=size, symbolBrush=color)
                curve.record_role, curve.record_channel = role, channel
                plot.addItem(curve)
                content.curves.append(curve)
            for x, y in zip(centers[exceeded], main['y'][exceeded]):
                text = pg.TextItem('NG', color=exceeded_color, anchor=(.5, 1.))
                text.record_channel = channel
                text.setPos(x, y)
                plot.addItem(text)
                content.annotations.append(text)


class HeatmapRenderer:
    def render(self, content, panels, parameters):
        panel, = panels
        data = panel.heatmap
        x, y, z = data['x'], data['y'], data['z']
        metadata = panel.metadata or {}
        if panel.key == 'Spec':
            name = str(parameters.get('color_map', 'viridis') or 'viridis')
        else:
            name = metadata.get('color_map', parameters.get('color_map', 'viridis'))
        # Validate colormap/levels through the shared native plotting boundary.
        levels = None
        finite = z[np.isfinite(z)]
        try:
            if parameters.get('custom_limit'):
                if panel.key == 'Spec':
                    # Match _calculate_spec's queue defaults, numeric conversion,
                    # and effective color range without changing the matrix.
                    bottom = float(parameters.get('bottom_limit', 50))
                    top = float(parameters.get('top_limit', 70))
                    middle = (top - bottom) / 2.0
                    levels = (bottom - middle, top + middle)
                else:
                    levels = (parameters['bottom_limit'], parameters['top_limit'])
            settings = validate_heatmap_data(None, levels=levels, colormap=name, context=panel.key)
        except (TypeError, ValueError) as error:
            # Color interpretation requires Qt. Numeric/style settings can be
            # malformed independently of the mapped matrix. Preserve summaries
            # and other panels while exposing the bad display setting.
            content.diagnostics.append(content.label(f'{panel.key} 内容不可用：{error}'))
            return
        if levels is None:
            low, high = float(finite.min()), float(finite.max())
            levels = (low, high) if low < high else (low - 1, high + 1)
        widget = _plot(content, panel)
        plot = widget.getPlotItem()

        def edges(axis):
            return bin_edges(axis) if len(axis) > 1 else np.array([axis[0] - .5, axis[0] + .5])

        x_edges, y_edges = edges(x), edges(y)
        scale = parameters.get('freq_scale_type')
        if panel.key == 'Spec':
            scale = str(scale or 'linear').lower()
        log_y = scale == 'log' and panel.y_unit == 'Hz'
        if log_y:
            first = 1 if y[0] == 0 else 0
            y, z, y_edges = y[first:], z[first:], y_edges[first:]
            y_edges[0] = max(y[0] / 2, y_edges[0])
            y_edges = np.log10(y_edges)
            plot.setLogMode(y=True)
        elif panel.y_unit == 'Hz':
            y_edges[0] = max(0, y_edges[0])
        if panel.y_unit == 'Hz':
            axis = plot.getAxis('left')
            axis.setScale(.001)
            axis.setLabel('kHz')
        image = SpectrogramImage(z, x_edges, y_edges, levels, settings['colormap'],
                                 (x_edges[0], x_edges[-1]))
        plot.addItem(image)
        content.images.append(image)
        plot.setRange(xRange=(x_edges[0], x_edges[-1]), yRange=(y_edges[0], y_edges[-1]), padding=0)
        bar = pg.ColorBarItem(values=levels, colorMap=settings['colormap'], interactive=False, width=12)
        bar.axis.setLabel(panel.value_label or panel.value_unit)
        bar.axis.setTextPen('#64748b')
        bar.setImageItem([], insert_in=plot)
        content.colorbars.append(bar)


class RecordAnalysisRenderer:
    def __init__(self):
        self.strategies = {'curve': CurveRenderer(), 'bands': BandRenderer(), 'heatmap': HeatmapRenderer()}

    def render(self, result, parameters, *, channel_index=None):
        content = RecordAnalysisContent()
        if channel_index is None and any(panel.kind == 'heatmap' for ch in result.channels for panel in ch.panels):
            channel_index = result.channels[0].channel_index
        selected = next((ch for ch in result.channels if ch.channel_index == channel_index), None)
        if selected is not None and not selected.panels and not selected.summary_cards:
            content.diagnostics.append(content.label(selected.diagnostic or selected.status))
            content.content_layout.addStretch(1)
            return content
        for channel in result.channels:
            judgement = (channel.judgement if channel.judgement_status == 'completed'
                         else '判定失败' if channel.judgement_status == 'failed' else '未判定')
            heading = f'CH{channel.wav_channel} · {channel.name}  本次判定：{judgement or "未判定"}  {channel.status}'
            metrics = channel.metrics or {}
            unit = (format_workbench_spl_unit(metrics['weighting'])
                    if result.analysis_type == 'SPL' and 'weighting' in metrics else metrics.get('unit', ''))
            values = []
            measurements = {
                'SPL': (('overall_spl', '总体声压级', unit),),
                'FFT': (('peak_value', '峰值', unit), ('peak_frequency_hz', '峰值频率', 'Hz')),
                'FBA': (('overall_weighted_db', '总体频段声级', unit),),
            }.get(result.analysis_type, ())
            bound_unit = metrics.get('limit_unit') or unit
            bounds = (('overall_upper_limit', '上限', bound_unit),
                      ('overall_lower_limit', '下限', bound_unit))
            for key, label, measurement_unit in (*measurements, *bounds):
                value = metrics.get(key)
                if isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value):
                    values.append(f'{label}：{value:.2f}' +
                                  (f' {measurement_unit}' if measurement_unit else ''))
            if unit and result.analysis_type in ('SPL', 'FFT', 'FBA'):
                values.insert(0, f'幅值单位：{unit}')
            content.summary_labels.append(content.label(heading + ('\n' + '  ·  '.join(values) if values else '')))
            if channel.diagnostic:
                content.diagnostics.append(content.label(channel.diagnostic))
            for card in channel.summary_cards:
                content.card_labels.append(content.label(
                    f'CH{channel.wav_channel}  {card["label"]}：{card["value"]:.2f} {card["unit"]}'))
        grouped = {}
        for channel in result.channels:
            for panel in channel.panels:
                if panel.kind == 'heatmap' and channel is not selected:
                    continue
                key = (panel.key, panel.kind, panel.x_unit, panel.y_unit, panel.value_unit)
                if panel.kind == 'bands':
                    main = next(series for series in panel.series if series['role'] == 'curve')
                    key += (panel.band_labels, tuple(main['x']))
                grouped.setdefault(key, []).append(panel)
        for panels in grouped.values():
            self.strategies[panels[0].kind].render(content, panels, parameters)
        if not grouped and not content.card_labels and not content.diagnostics:
            content.diagnostics.append(content.label(result.diagnostic or result.status or '结果不可用'))
        if not content.plots:
            content.content_layout.addStretch(1)
        return content
