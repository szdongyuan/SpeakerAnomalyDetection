"""One spectrogram canvas, multiple condition regions, one shared color scale."""

from copy import deepcopy

import numpy as np
import pyqtgraph as pg

from ui.sequence.condition_plot_cell import ConditionPlotCell
from ui.sequence.spectrogram_data import bin_edges
from ui.sequence.spectrogram_image import SpectrogramImage


class ConditionHeatmapCell(ConditionPlotCell):
    def __init__(self, *args, **kwargs):
        self._condition_heatmaps = {}
        self._images = []
        super().__init__(*args, **kwargs)
        self.legend_bar.hide()
        # Put time ticks outside the heatmap, with space before their labels.
        self.plot_widget.getPlotItem().getAxis('bottom').setStyle(tickLength=5, tickTextOffset=4)
        # Scale tick text only; matrices, frequency coordinates and configured ranges stay in Hz.
        axis = self.plot_widget.getPlotItem().getAxis('left')
        axis.enableAutoSIPrefix(False)
        axis.setScale(0.001)
        axis.setLabel(text='kHz', units='')
        self._configuration = dict(self._configuration, y_label='kHz')
        self._update_minimum_height()

    def set_condition_heatmap_data(self, condition_id, heatmap, *, status='', detail=''):
        condition = self.conditions[condition_id]
        if condition.warning:
            raise ValueError(condition.warning)
        if heatmap is not None:
            if heatmap['wav_channel'] not in {ch for ch, _ in condition.channels}:
                raise ValueError('时频图 WAV 通道不在当前工况显示配置中')
            self._condition_heatmaps[condition_id] = deepcopy(heatmap)
        else:
            self._condition_heatmaps.pop(condition_id, None)
        self._condition_states[condition_id] = (status, detail)
        self._redraw()
        self.results_changed.emit()

    def _has_condition_data(self, condition_id):
        return condition_id in self._condition_heatmaps

    def _clear_condition_data(self, condition_id):
        super()._clear_condition_data(condition_id)
        self._condition_heatmaps.pop(condition_id, None)

    def _condition_range(self, key):
        bounds = super()._condition_range(key)
        if bounds is None and key in self._condition_heatmaps:
            edges = bin_edges(self._condition_heatmaps[key]['time_s'])
            return max(0, edges[0]), edges[-1]
        return bounds

    def copy_results_to(self, target):
        target._condition_heatmaps = deepcopy(self._condition_heatmaps)
        super().copy_results_to(target)

    def clear_plot(self):
        self._condition_heatmaps.clear()
        super().clear_plot()

    def _redraw(self):
        self._clear_data()
        self._layout_axis()
        self._images = []
        plot = self.plot_widget.getPlotItem()
        settings = self._settings['spectrogram']
        color_map = pg.colormap.get(settings['color_map'])
        values = [data['values_db'] for data in self._condition_heatmaps.values()]
        color_range = settings['color_range']
        if color_range['mode'] == 'fixed':
            levels = (color_range['min_db'], color_range['max_db'])
        elif values:
            low, high = min(np.nanmin(v) for v in values), max(np.nanmax(v) for v in values)
            levels = (low, high) if low < high else (low - 1, high + 1)
        else:
            levels = (0, 100)
        y_bounds = []
        for key, data in self._condition_heatmaps.items():
            frequency, matrix = data['frequency_hz'], data['values_db']
            edges = bin_edges(frequency)
            if self._settings['y_axis']['scale'] == 'log':
                first = 1 if frequency[0] == 0 else 0  # DC cannot be represented on a log axis.
                frequency, matrix, edges = frequency[first:], matrix[first:], edges[first:]
            edges[0] = max(frequency[0] / 2 if self._log_axes[1] else 0, edges[0])
            y_bounds.append((edges[0], edges[-1]))
            if self._log_axes[1]:
                edges = np.log10(edges)
            image = SpectrogramImage(matrix, self._map_x(key, bin_edges(data['time_s'])), edges,
                                     levels, color_map, self._spans[key])
            plot.addItem(image)
            self._images.append(image)
        self._image = self._images[0] if self._images else None
        axis = self._settings['y_axis']
        y_range = self._fixed_range(axis)
        if axis['auto_range']:
            if y_bounds:
                y_range = (min(v[0] for v in y_bounds), max(v[1] for v in y_bounds))
            elif 'empty_data_range_min' in axis:
                y_range = (axis['empty_data_range_min'], axis['empty_data_range_max'])
        self._configuration = dict(self._configuration, y_range=y_range)
        if settings['show_color_bar']:
            self._colorbar = pg.ColorBarItem(values=levels, colorMap=color_map, interactive=False, width=10)
            self._colorbar.axis.setPen('#AFC0D6')
            self._colorbar.axis.setTextPen('#64748B')
            self._colorbar.axis.setLabel('dB')
            self._colorbar.setToolTip('dB re 20 µPa，变换幅度；各工况使用相同颜色范围，不表示判定限值')
            self._colorbar.setImageItem([], insert_in=plot)
        self._apply_ranges()
        self._draw_condition_overlays()
