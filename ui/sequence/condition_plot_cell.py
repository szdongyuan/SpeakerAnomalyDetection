"""Render all applicable conditions in one canvas with one shared ordinate."""

import math
from copy import deepcopy
from numbers import Real

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtWidgets import QLabel, QToolButton

from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY
from ui.sequence.station_plot_cell import StationPlotCell
from ui.sequence.station_plot_legend import StationPlotLegend
from ui.sequence.station_plot_style import expand_plot_icon
from ui.sequence.station_plot_validation import validate_curve_series
from ui.sequence.sampled_plot_curve import SampledPlotCurve
from ui.plotting.geometry import clip_curve_x


class ConditionPlotCell(StationPlotCell):
    results_changed = pyqtSignal()

    def __init__(self, display, parent=None, *, row=0, column=0, station_name='', detail_view=False):
        super().__init__(parent, row=row, column=column)
        self.station_name = station_name
        self._manual_view_axes = [False, False]
        self._detail_dialog = None
        self.display = display
        self._settings = display.condition_plot.settings
        self._display_downsampling = self._settings['display_downsampling']
        self.conditions = {item.condition_id: item for item in display.condition_plot.conditions}
        self._condition_series = {}
        self._condition_states = {}
        self._condition_summaries = {}
        # Reserve summary rows from the display binding, not arriving results.
        value_rows = display.condition_plot.summary_rows
        self._heading_height = 20
        if value_rows:
            self._heading_height += 12 + 18 * value_rows
        self._spans = {}
        self._source_ranges = {}
        self.condition_labels = {}
        self.overall_labels = {}
        self.empty_labels = {}
        for key, condition in self.conditions.items():
            self.condition_labels[key] = self._label(condition.name)
            self.condition_labels[key].setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            # Condition headings sit outside the data region and expose result tooltips.
            self.condition_labels[key].setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.overall_labels[key] = self._label('')
            self.overall_labels[key].setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.empty_labels[key] = self._label('配置不匹配' if condition.warning else '', muted=not condition.warning)
            self.empty_labels[key].setToolTip(condition.warning)
        self._axis_breaks = []
        for _ in list(self.conditions)[1:]:
            label = self._label('')
            label.setStyleSheet(label.styleSheet() +
                               f'QLabel {{ background: {self._settings["plot_style"]["background_color"]}; }}')
            self._axis_breaks.append(label)
        self._layout_width = 0
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._redraw_preserving_view)
        self._sampling_timer = QTimer(self)
        self._sampling_timer.setSingleShot(True)
        self._sampling_timer.timeout.connect(self._refresh_display_curves)
        self._no_conditions = self._label('无适用工况')
        self._no_conditions.setVisible(not self.conditions)
        settings = self._settings
        first = next(iter(self.conditions.values()), None)
        self.configure_plot(
            kind='heatmap' if settings['chart_type'] == 'heatmap' else 'curve', title=display.title, right_text=display.warning,
            right_text_detail=display.warning_detail,
            x_label=settings['x_axis']['label'], x_unit=('kHz' if first.x_unit == 'Hz' else first.x_unit) if first else '',
            # The chart title names the quantity; the ordinate shows only its unit.
            y_label=first.y_unit if first else '', y_unit='',
            x_range=(0, 1), y_range=self._fixed_range(settings['y_axis']),
        )
        self.subheader.hide()
        self.header.left.setStyleSheet(self.header.left.styleSheet() +
                                      'QLabel { font-size: 14px; font-weight: normal; color: #3F5870; }')
        self.header.layout().setContentsMargins(10, 7, 10, 7)
        if not detail_view:
            self.expand_button = QToolButton(self.header)
            self.expand_button.setAutoRaise(True)
            self.expand_button.setIcon(expand_plot_icon())
            self.expand_button.setIconSize(QSize(16, 16))
            self.expand_button.setToolTip('放大查看')
            self.expand_button.setAccessibleName('放大查看')
            self.expand_button.setFixedSize(24, 24)
            self.header.layout().addWidget(self.expand_button)
            self.expand_button.clicked.connect(self.open_detail)
        self.legend_bar = StationPlotLegend(self)
        legend = settings['plot_style']['legend']
        if legend['position'].startswith('top'):
            # Reserve a full-width row even before results arrive, keeping the canvas stable.
            self.layout().insertWidget(1, self.legend_bar)
        else:
            self.layout().addWidget(self.legend_bar)
        self.legend_bar.align_right = legend['position'].endswith('right')
        self.legend_bar.setVisible(legend['visible'])
        # X is mapped into condition regions. Tick text retains physical units;
        # pyqtgraph must not apply a second logarithm or SI scale to this axis.
        self.set_display_style({**settings['plot_style'], 'legend': {**legend, 'visible': False}},
                               {'scale': 'linear'}, settings['y_axis'])
        plot = self.plot_widget.getPlotItem()
        plot.getViewBox().setMouseEnabled(x=True, y=True)
        plot.getViewBox().sigRangeChangedManually.connect(self._mark_manual_view_axes)
        plot.setMenuEnabled(True)
        plot.showButtons()
        # Native auto-fit omits empty conditions; restore the configured canvas instead.
        plot.autoBtn.clicked.disconnect(plot.autoBtnClicked)
        plot.autoBtn.clicked.connect(self.reset_view)
        plot.autoBtn.setToolTip('还原视图')
        plot.getAxis('bottom').enableAutoSIPrefix(False)
        if settings['condition_display']['x_axis_mode'] == 'cumulative_time':
            plot.getAxis('bottom').label.setToolTip('各工况显示时间首尾拼接，不包含工况切换或等待时间')
        plot.getAxis('left').setWidth(52)
        plot.getAxis('left').hide_top_grid_line = settings['chart_type'] == 'line'
        plot.layout.setContentsMargins(7, self._heading_height + 8, 7, 5)
        plot.getViewBox().sigRangeChanged.connect(self._position_labels)
        plot.getViewBox().sigXRangeChanged.connect(self._schedule_sampling)
        plot.getViewBox().geometryChanged.connect(self._schedule_sampling)
        plot.getViewBox().geometryChanged.connect(self._position_labels)
        plot.getViewBox().geometryChanged.connect(self._resize_regions)
        self._redraw()
        self._update_minimum_height()

    def _update_minimum_height(self):
        # Include headings, axes and any external legend above the minimum data area.
        plot = self.plot_widget.getPlotItem()
        plot.getViewBox().setMinimumHeight(120)
        self.plot_widget.setMinimumHeight(math.ceil(plot.effectiveSizeHint(Qt.MinimumSize).height()))
        self.setMinimumHeight(self.layout().minimumSize().height())

    def _schedule_sampling(self, *_):
        self._sampling_timer.start(0)

    def _refresh_display_curves(self):
        view = self.plot_widget.getViewBox()
        for curve in self.plot_widget.listDataItems():
            if isinstance(curve, SampledPlotCurve):
                curve.refresh_display(view.viewRange()[0], view.width())

    def _resize_regions(self):
        if abs(self.plot_widget.getViewBox().width() - self._layout_width) > 0.5:
            self._resize_timer.start(0)

    def reset_view(self):
        self._manual_view_axes = [False, False]
        self._redraw()

    def _redraw_preserving_view(self):
        previous = self._manual_view_ranges()
        self._redraw()
        self._restore_manual_view_ranges(previous)

    def _mark_manual_view_axes(self, axes):
        self._manual_view_axes = [old or changed for old, changed in zip(self._manual_view_axes, axes)]

    def _manual_view_ranges(self):
        if not self.has_result_data():
            return {}
        ranges = self.plot_widget.getViewBox().viewRange()
        return {name: ranges[index] for index, name in enumerate(('xRange', 'yRange'))
                if self._manual_view_axes[index]}

    def _restore_manual_view_ranges(self, ranges):
        if not self.has_result_data():
            self._manual_view_axes = [False, False]
        elif ranges:
            # Preserve user gestures only; untouched axes follow incoming results.
            self.plot_widget.getViewBox().setRange(**ranges, padding=0)

    def _label(self, text, *, muted=False):
        label = QLabel(text, self.plot_widget.viewport())
        label.setTextFormat(Qt.PlainText)
        label.setAlignment(Qt.AlignCenter)
        label.setAttribute(Qt.WA_TransparentForMouseEvents)
        # QSS must specify the font too: the main window's QLabel rule otherwise
        # overrides setFont() and enlarges these labels to 20px.
        label.setStyleSheet(
            f'QLabel {{ color: {"#8896A6" if muted else "#64748B"}; background: transparent; border: 0; '
            f'font-family: {MAIN_UI_SMALL_FONT_FAMILY}; font-size: 13px; }}')
        font = self.font()
        font.setPixelSize(13)
        label.setFont(font)
        return label

    @staticmethod
    def _fixed_range(axis):
        return None if axis['auto_range'] else (axis['range_min'], axis['range_max'])

    def _condition_range(self, key):
        axis = self._settings['x_axis']
        fixed = self._fixed_range(axis)
        if fixed is not None:
            return fixed
        if self.conditions[key].x_unit == 's':
            duration = self.conditions[key].duration_seconds
            if duration is not None:
                return (0, duration)
        values = [item['x'] for item in self._condition_series.get(key, ()) if item['x'].size]
        if not values:
            return None
        x = np.concatenate(values)
        if axis['scale'] == 'log':
            x = x[x > 0]
        if x.size and x.min() < x.max():
            return (float(x.min()), float(x.max()))
        return None

    def _map_x(self, key, values):
        low, high = self._source_ranges[key]
        start, end = self._spans[key]
        if self._settings['x_axis']['scale'] == 'log':
            values, low, high = np.log10(values), math.log10(low), math.log10(high)
        return start + (values - low) / (high - low) * (end - start)

    def _layout_axis(self):
        cumulative = self._settings['condition_display']['x_axis_mode'] == 'cumulative_time'
        self._source_ranges = {key: self._condition_range(key) for key in self.conditions}
        widths = [bounds[1] - bounds[0] if cumulative and bounds else 1.0
                  for bounds in self._source_ranges.values()]
        self._layout_width = max(1, self.plot_widget.getViewBox().width())
        count = len(self.conditions)
        # Reserve a fixed logical-pixel gutter; very narrow views retain room for data.
        pixels = min(self._settings['condition_display']['condition_gap_px'],
                     self._layout_width / (2 * max(1, count)))
        gap = sum(widths) * pixels / (self._layout_width - max(0, count - 1) * pixels)
        ticks, offset, elapsed = [], 0.0, 0.0
        alignments = []
        for index, (key, width) in enumerate(zip(self.conditions, widths)):
            first_tick = len(ticks)
            bounds = self._source_ranges[key]
            self._spans[key] = (offset, offset + width)
            if bounds:
                if cumulative:
                    ticks.extend((offset + fraction * width, f'{elapsed + fraction * width:g}')
                                 for fraction in (0, 0.5))
                else:
                    low, high = bounds
                    if self._settings['x_axis']['scale'] == 'log':
                        values = [10.0 ** power for power in range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1)]
                    else:
                        values = list(np.linspace(low, high, 3))
                    # Leave space around the separator, where frequency restarts.
                    if len(self.conditions) > 1 and self._settings['x_axis']['scale'] == 'log':
                        values = [value for value in values if low < value < high]
                    ticks.extend((float(self._map_x(key, value)), f'{value / (1000 if self.conditions[key].x_unit == "Hz" else 1):g}') for value in values)
            if not cumulative and len(ticks) > first_tick:
                if index > 0 and math.isclose(ticks[first_tick][0], offset):
                    alignments.append((ticks[first_tick][0], 'left'))
                if index < count - 1 and math.isclose(ticks[-1][0], offset + width):
                    alignments.append((ticks[-1][0], 'right'))
            offset += width
            elapsed += width
            if index < count - 1:
                offset += gap
        if cumulative and self.conditions:
            ticks.append((offset, f'{elapsed:g}'))
        self._configuration = dict(self._configuration, x_range=(0, offset or 1))
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        axis.setTicks([ticks])
        axis.set_tick_text_alignments(alignments)

    def _gaps(self):
        spans = list(self._spans.values())
        return [(left[1], right[0]) for left, right in zip(spans, spans[1:])]

    def set_condition_curve_data(self, condition_id, series, *, status='', detail='', summaries=()):
        """Accept local x coordinates; offsets affect display only, never source data."""
        condition = self.conditions[condition_id]
        if condition.warning:
            raise ValueError(condition.warning)
        snapshot = validate_curve_series(series, context=self._context('set_condition_curve_data'), allow_gaps=True)
        allowed = {number for number, _ in condition.channels}
        for raw, item in zip(series, snapshot):
            if type(raw.get('wav_channel')) is not int or raw['wav_channel'] not in allowed:
                raise ValueError('曲线的 WAV 通道不在当前工况显示配置中')
            if raw.get('role', 'curve') not in ('curve', 'baseline', 'upper', 'lower', 'exceedance'):
                raise ValueError('curve role must be curve, baseline, upper, lower or exceedance')
            item.update(wav_channel=raw['wav_channel'], role=raw.get('role', 'curve'))
            if raw.get('limit_scope') == 'scalar':
                item['limit_scope'] = 'scalar'
        self._condition_series[condition_id] = snapshot
        self._condition_states[condition_id] = (status, detail)
        self._condition_summaries[condition_id] = deepcopy(summaries)
        self._redraw_preserving_view()
        self.results_changed.emit()

    def set_condition_state(self, condition_id, status, detail='', *, clear=False):
        if condition_id not in self.conditions:
            raise ValueError('未知工况')
        self._condition_states[condition_id] = (status, detail)
        if clear:
            self._clear_condition_data(condition_id)
            self._redraw_preserving_view()
        else:
            self._position_labels()
        self.results_changed.emit()

    def open_detail(self):
        from ui.sequence.plot_detail_dialog import PlotDetailDialog

        if self._detail_dialog is None:
            self._detail_dialog = PlotDetailDialog(self)
            self._detail_dialog.finished.connect(self._detail_closed)
        if self._detail_dialog.isMinimized():
            self._detail_dialog.showNormal()
        else:
            self._detail_dialog.show()
        self._detail_dialog.raise_()
        self._detail_dialog.activateWindow()

    def _detail_closed(self):
        self._detail_dialog = None

    def close_detail(self):
        if self._detail_dialog is not None:
            self._detail_dialog.close()

    def copy_results_to(self, target):
        """Copy the in-memory local-coordinate data, never read files or run analysis."""
        previous = target._manual_view_ranges()
        target._condition_series = deepcopy(self._condition_series)
        target._condition_states = dict(self._condition_states)
        target._condition_summaries = deepcopy(self._condition_summaries)
        target._redraw()
        target._restore_manual_view_ranges(previous)

    def _clear_condition_data(self, condition_id):
        self._condition_series.pop(condition_id, None)
        self._condition_summaries.pop(condition_id, None)

    def _has_condition_data(self, condition_id):
        return any(item['x'].size for item in self._condition_series.get(condition_id, ()))

    def has_result_data(self):
        return any(self._has_condition_data(key) for key in self.conditions)

    def _role_visible(self, role):
        style = self._settings['plot_style']
        if role == 'exceedance':
            return style['exceedance_highlight']['visible']
        return role in ('curve', 'baseline') or style['limit_lines']['visible']

    def dispose(self):
        self.close_detail()
        self._resize_timer.stop()
        self._sampling_timer.stop()
        super().dispose()

    def _redraw(self):
        self._layout_axis()
        curves = []
        for key, condition in self.conditions.items():
            bounds = self._source_ranges[key]
            for item in self._condition_series.get(key, ()):
                if bounds is None:
                    continue
                if item['role'] in ('upper', 'lower') and not condition.show_threshold_lines:
                    continue
                visible_x, visible_y = clip_curve_x(
                    item['x'], item['y'], bounds,
                    log_x=self._settings['x_axis']['scale'] == 'log',
                    log_y=self._settings['y_axis']['scale'] == 'log')
                name = item['name'] or f'CH{item["wav_channel"]}'
                if item['role'] != 'curve':
                    name = f'CH{item["wav_channel"]} ' + {'upper': '上限', 'lower': '下限', 'exceedance': '超限', 'baseline': '参考'}[item['role']]
                    if item.get('limit_scope') == 'scalar':
                        name = name.replace('上限', '总体上限').replace('下限', '总体下限')
                curve = dict(item, name=name, x=self._map_x(key, visible_x), y=visible_y)
                if condition.curve_colors:
                    curve['analysis_colors'] = dict(condition.curve_colors)
                    curve['analysis_type'] = condition.analysis_type
                if curve['color'] is None:
                    del curve['color']
                curves.append(curve)
        axis = self._settings['y_axis']
        y_range = self._fixed_range(axis)
        has_values = any(np.isfinite(item['y']).any() for item in curves if self._role_visible(item['role']))
        if axis['auto_range'] and not has_values and 'empty_data_range_min' in axis:
            y_range = (axis['empty_data_range_min'], axis['empty_data_range_max'])
        self._configuration = dict(self._configuration, y_range=y_range)
        super().set_curve_data(curves, allow_gaps=True)
        self._refresh_display_curves()
        visible = [item for item in curves if item['x'].size and self._role_visible(item['role'])]
        entries, seen = [], set()
        for item, line in zip(visible, self.plot_widget.listDataItems()):
            pen = line.opts['pen']
            key = (item['wav_channel'], item['role'], pen.color().rgba(), pen.widthF(), pen.style())
            if key not in seen:
                entries.append((item['name'], pen))
                seen.add(key)
        self.legend_bar.set_entries(entries)
        # Configured empty-data ranges keep a complete axis while waiting.
        has_data = any(item['x'].size for item in self._series)
        self.plot_widget.getPlotItem().getAxis('left').setStyle(
            tickLength=0 if y_range is None and not has_data else -5)
        self._draw_condition_overlays()

    def _overall_spl_values(self, key):
        condition = self.conditions[key]
        for summary in self._condition_summaries.get(key, ()):
            metrics = summary['metrics']
            value = metrics.get('overall_spl')
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
                continue
            if self._log_axes[1] and value <= 0:
                continue
            unit = metrics.get('unit') or condition.y_unit
            if unit in ('dBA', 'dBB', 'dBC', 'dBD', 'dBZ'):
                unit = f'dB({unit[-1]})'
            elif unit == 'dB' and condition.y_unit == 'dB(Z)':
                unit = 'dB(Z)'
            channel = f'CH{summary["wav_channel"]} ' if len(condition.channels) > 1 else ''
            yield value, f'{channel}{value:.1f} {unit}'

    def _draw_condition_overlays(self):
        self.separators = []
        if self._settings['condition_display']['show_condition_separator']:
            for start, end in self._gaps():
                line = pg.InfiniteLine((start + end) / 2, angle=90, movable=False,
                                       pen=pg.mkPen('#B6C6D9', width=1, style=Qt.DashLine))
                self.plot_widget.addItem(line, ignoreBounds=True)
                self.separators.append(line)
        self._position_labels()

    def _condition_heading(self, key):
        status, _ = self._condition_states.get(key, ('', ''))
        lines = [self.conditions[key].name + (f' · {status}' if self._has_condition_data(key) and status else '')]
        return lines

    def _position_labels(self, *_):
        view = self.plot_widget.getViewBox()
        rect = self.plot_widget.mapFromScene(view.sceneBoundingRect()).boundingRect()
        x_min, x_max = view.viewRange()[0]
        for label, (start, end) in zip(self._axis_breaks, self._gaps()):
            left = rect.left() + (start - x_min) / (x_max - x_min) * rect.width()
            right = rect.left() + (end - x_min) / (x_max - x_min) * rect.width()
            # Curves and ticks already avoid the gutter; interrupt only the baseline.
            label.setGeometry(round(left), rect.bottom() - 2, round(right) - round(left), 4)
            label.setVisible(end > start)
            label.raise_()
        for key, (start, end) in self._spans.items():
            left = rect.left() + (start - x_min) / (x_max - x_min) * rect.width()
            right = rect.left() + (end - x_min) / (x_max - x_min) * rect.width()
            left, right = max(rect.left(), left), min(rect.right(), right)
            width = max(0, int(right - left))
            label = self.condition_labels[key]
            status, detail = self._condition_states.get(key, ('', ''))
            has_data = self._has_condition_data(key)
            summaries = self._condition_summaries.get(key, ())
            label.setText('\n'.join(self._condition_heading(key)))
            label.setGeometry(int(left), rect.top() - self._heading_height - 2, width, 20)
            label.setVisible(self._settings['condition_display']['show_condition_names'] and width > 0)
            values = [text for _, text in self._overall_spl_values(key)]
            overall = self.overall_labels[key]
            overall.setText('\n'.join(values))
            overall.setGeometry(int(left), label.geometry().bottom() + 13, width, 18 * len(values))
            overall.setVisible(bool(values) and width > 0)
            summary_text = []
            for summary in summaries:
                metrics = summary['metrics']
                text = summary['channel'] + '：' + str(summary.get('judgement') or '未判定')
                if metrics.get('overall_spl') is not None:
                    text += f'，总体 SPL {metrics["overall_spl"]:g} {metrics.get("unit", "")}'
                for field, caption in (('overall_upper_limit', '总体上限'), ('overall_lower_limit', '总体下限')):
                    if metrics.get(field) is not None:
                        text += f'，{caption} {metrics[field]:g}'
                summary_text.append(text)
            label.setToolTip('\n'.join([self.conditions[key].name, *summary_text, detail]).strip())
            empty = self.empty_labels[key]
            empty.setText('配置不匹配' if self.conditions[key].warning else status)
            empty.setToolTip(self.conditions[key].warning or detail)
            empty.setGeometry(int(left), rect.center().y() - 12, width, 24)
            empty.setVisible(bool(empty.text()) and not has_data and width > 0)
        self._no_conditions.setGeometry(rect)

    def clear_plot(self):
        self._manual_view_axes = [False, False]
        self._condition_series.clear()
        self._condition_states.clear()
        self._condition_summaries.clear()
        self._redraw()
        self.results_changed.emit()
