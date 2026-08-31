from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QSizePolicy, QVBoxLayout, QWidget,
)

from base.channel_layout import load_channel_layout, save_channel_layout
from consts import ui_style_const
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel


class AnalysisWaveformRow(QFrame):
    """One display-only waveform row for a fixed acquisition channel."""

    direction_label_changed = pyqtSignal(str, str)

    def __init__(self, canvas, channel_index, direction_label):
        super().__init__(canvas)
        self.channel_index = int(channel_index)
        channel_label = f'CH{self.channel_index + 1}'
        self.channel_label = str(channel_label or "")
        self.direction_label = str(direction_label or "")
        self.plot_item = None

        self.setObjectName("fiveChannelWaveformRow")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QFrame#fiveChannelWaveformRow {"
            "background:#FFFFFF; border:none; border-top:1px solid #D7E1EA;"
            "}"
        )

        label_panel = QWidget(self)
        label_panel.setFixedWidth(76)
        label_panel.setStyleSheet("QWidget { background:transparent; border:none; }")
        label_layout = QVBoxLayout(label_panel)
        label_layout.setContentsMargins(12, 8, 6, 8)
        label_layout.setSpacing(4)
        label_layout.addStretch(1)

        channel = self.channel_caption = QLabel(self.channel_label)
        channel.setObjectName("waveformChannelLabel")
        channel.setStyleSheet(self._label_style("#25364A", 16, False))
        self.direction_editor = QLineEdit(self.direction_label, label_panel)
        self.direction_editor.setObjectName("waveformDirectionEditor")
        self.direction_editor.setFrame(False)
        self.direction_editor.setMaximumWidth(52)
        self.direction_editor.setToolTip("点击编辑通道位置，按 Enter 保存")
        self.direction_editor.setStyleSheet(self._direction_editor_style())
        self.direction_editor.editingFinished.connect(
            self._commit_direction_label
        )
        self.direction_editor.returnPressed.connect(
            self.direction_editor.clearFocus
        )
        label_layout.addWidget(channel)
        label_layout.addWidget(self.direction_editor)
        label_layout.addStretch(1)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setObjectName("fiveChannelPlot")
        self.plot_widget.setMinimumHeight(72)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setStyleSheet(
            "QWidget#fiveChannelPlot { background:#FBFDFF; "
            "border:1px solid #D7E1EA; border-radius:3px; }"
        )
        self._setup_plot()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 12, 8)
        layout.setSpacing(4)
        layout.addWidget(label_panel)
        layout.addWidget(self.plot_widget, stretch=1)

    @staticmethod
    def _label_style(color: str, font_size: int, bold: bool) -> str:
        weight = "font-weight:600;" if bold else ""
        return (
            "QLabel { background:transparent; border:none; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            f"font-size:{font_size}px; {weight} }}"
        )

    @staticmethod
    def _direction_editor_style() -> str:
        return (
            "QLineEdit#waveformDirectionEditor { background:transparent; "
            "border:1px solid transparent; border-radius:2px; padding:0; "
            f"color:{ui_style_const.COLOR_PRIMARY}; "
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            "font-size:14px; font-weight:600; }"
            "QLineEdit#waveformDirectionEditor:focus { background:#FFFFFF; "
            f"border:1px solid {ui_style_const.COLOR_PRIMARY}; }}"
        )

    def _commit_direction_label(self) -> None:
        direction_label = self.direction_editor.text().strip()
        if not direction_label:
            self.direction_editor.setText(self.direction_label)
            return
        self.direction_editor.setText(direction_label)
        if direction_label == self.direction_label:
            return
        self.direction_label_changed.emit(self.channel_label, direction_label)

    def _setup_plot(self) -> None:
        self.plot_widget.setBackground("#FBFDFF")
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.showAxis("bottom")
        self.plot_widget.showAxis("left")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)

    def clear_plot(self) -> None:
        self.plot_widget.clear()
        self.plot_item = None

    def set_data(self, x, y) -> bool:
        x_values = np.asarray(x)
        y_values = np.asarray(y)
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape[0] != y_values.shape[0]:
            return False
        if x_values.shape[0] == 0:
            self.clear_plot()
            return False

        if self.plot_item is None:
            self.plot_widget.getViewBox().enableAutoRange(
                axis=pg.ViewBox.XYAxes,
                enable=True,
            )
            self.plot_item = self.plot_widget.plot(
                x_values,
                y_values,
                pen=pg.mkPen(ui_style_const.COLOR_WAVEFORM, width=1.4),
            )
        else:
            self.plot_item.setData(x_values, y_values)
        return True


    def set_title(self, title):
        self.channel_caption.setText(str(title))

    def snapshot_plot_state(self):
        if self.plot_item is None:
            return None
        x_data, y_data = self.plot_item.getData()
        return np.asarray(x_data).copy(), np.asarray(y_data).copy()

    def restore_plot_state(self, state):
        if state is None:
            self.clear_plot()
        else:
            self.set_data(*state)


class AnalysisWaveformPanel(ChannelPlotWorkspace):
    """Stashed task layout with dynamic physical-channel plots."""

    channels_changed = pyqtSignal(list)

    def __init__(self, parent=None, condition_configs=None, channel_layout_path=None):
        super().__init__(parent)
        self._channel_layout_path = channel_layout_path
        self._channel_layout = load_channel_layout(channel_layout_path)
        self._conditions = {}
        self._condition_contexts = {}
        self._active_condition_key = ""
        self._audio_paths = {}
        self._mode = "test"
        self.scroll.viewport().installEventFilter(self)
        self.set_forced_columns(1)
        self.set_preserve_positions(False)
        self.setMinimumWidth(480)
        self.canvas.setStyleSheet("background:white;")

        header = QFrame(self)
        header.setObjectName("analysisWaveformHeader")
        header.setFixedHeight(36)
        header.setStyleSheet(
            f"QFrame#analysisWaveformHeader {{ background:{ui_style_const.COLOR_PRIMARY}; "
            "border:none; border-top-left-radius:4px; border-top-right-radius:4px; }"
            "QLabel { background:transparent; color:white; border:none; "
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            "font-size:16px; font-weight:bold; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)
        self.title_label = QLabel(header)
        self.status_label = QLabel("同步待机", header)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.status_label)

        meta = QFrame(self)
        meta.setObjectName("analysisWaveformMeta")
        meta.setFixedHeight(46)
        meta.setStyleSheet(
            "QFrame#analysisWaveformMeta { background:white; border:none; "
            "border-bottom:1px solid #D7E1EA; }"
            "QLabel { background:transparent; border:none; color:#526477; "
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; font-size:13px; }}"
        )
        meta_layout = QHBoxLayout(meta)
        meta_layout.setContentsMargins(12, 0, 12, 0)
        meta_layout.setSpacing(26)
        self.current_condition_label = QLabel(meta)
        self.duration_label = QLabel(meta)
        self.test_items_label = QLabel(meta)
        for caption, label in (
            ("当前档位：", self.current_condition_label),
            ("录制时长：", self.duration_label),
            ("测试内容：", self.test_items_label),
        ):
            field = QHBoxLayout()
            field.setContentsMargins(0, 0, 0, 0)
            field.setSpacing(8)
            caption_label = QLabel(caption, meta)
            caption_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            field.addWidget(caption_label)
            field.addWidget(label)
            meta_layout.addLayout(field, 1 if label is self.test_items_label else 0)
        self.layout().insertWidget(0, header)
        self.layout().insertWidget(1, meta)
        self.set_conditions(condition_configs)

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._tile_subwindows)
        return super().eventFilter(watched, event)

    def set_channels(self, channel_indices):
        channels = [int(channel) for channel in channel_indices]
        if channels == self._channel_indices and self._subwins:
            return
        for window in self._subwins:
            window.hide()
            window.deleteLater()
        self.canvas.set_windows([])
        self._channel_indices = channels
        self._subwins = []
        for channel in channels:
            label = f"CH{channel + 1}"
            row = AnalysisWaveformRow(self.canvas, channel, self._channel_layout.get(label, ""))
            row.direction_label_changed.connect(
                lambda label, _value, row=row: self._save_channel_alias(label, row)
            )
            row.show()
            self._subwins.append(row)
        self.canvas.set_windows(self._subwins)
        QTimer.singleShot(0, self._tile_subwindows)
        self._refresh_context_labels()
        self.channels_changed.emit(channels)

    def _save_channel_alias(self, channel_label, row):
        value = row.direction_editor.text().strip()
        previous = self._channel_layout.get(channel_label, "")
        if not value or value == previous:
            row.direction_editor.setText(previous)
            return
        updated_layout = {**self._channel_layout, channel_label: value}
        if save_channel_layout(updated_layout, self._channel_layout_path):
            self._channel_layout = updated_layout
            row.direction_label = value
        else:
            row.direction_editor.setText(previous)
            row.direction_editor.setToolTip("通道显示名称保存失败，请检查配置目录权限")

    def _calc_tile_spec(self, viewport_w, viewport_h, num_windows):
        spec = super()._calc_tile_spec(viewport_w, viewport_h, num_windows)
        return replace(
            spec, cols=1, win_w=max(160, viewport_w),
            win_h=max(90, viewport_h // max(1, num_windows)), gap=0, pad=0,
        )

    def set_conditions(self, condition_configs):
        self._conditions = {
            item["key"]: item
            for item in DirectionWaveformPanel._normalize_conditions(condition_configs)
        }
        key = self._active_condition_key
        if key not in self._conditions:
            key = next(iter(self._conditions), "")
        self._condition_contexts = {
            key: context for key, context in self._condition_contexts.items()
            if key in self._conditions
        }
        self._audio_paths = {key: path for key, path in self._audio_paths.items() if key in self._conditions}
        self.set_active_condition(key)

    def condition_keys(self):
        return list(self._conditions)

    def set_active_condition(self, condition_key):
        key = str(condition_key or "")
        if key and key not in self._conditions:
            return False
        if key != self._active_condition_key:
            # Never present the previous gear's waveform under a new gear name.
            super().clear_plots()
        self._active_condition_key = key
        self._refresh_context_labels()
        return True

    def set_condition_context(self, condition_key, **context):
        key = str(condition_key or "")
        if key not in self._conditions:
            return
        self._condition_contexts.setdefault(key, {}).update(context)
        if key == self._active_condition_key:
            self._refresh_context_labels()

    def clear_direction(self, direction):
        self._audio_paths.pop(str(direction or ""), None)
        if str(direction or "") == self._active_condition_key:
            super().clear_plots()

    def clear_plots(self):
        super().clear_plots()
        self._audio_paths.clear()

    def set_condition_audio_path(self, condition_key, path):
        if path:
            self._audio_paths[condition_key] = str(path)
        else:
            self._audio_paths.pop(condition_key, None)

    def set_mode(self, mode):
        self._mode = str(mode or "test")

    def _refresh_context_labels(self):
        condition = self._conditions.get(self._active_condition_key, {})
        context = self._condition_contexts.get(self._active_condition_key, {})
        name = condition.get("display_name") or condition.get("name") or ""
        title = f"{len(self._channel_indices)}通道同步录制"
        self.title_label.setText(title)
        self.current_condition_label.setText(name or "--")
        duration = context.get("recording_duration")
        duration_text = f"{duration:g}秒" if isinstance(duration, (int, float)) else "--"
        self.duration_label.setText(duration_text)
        items = context.get("test_items") or []
        item_text = "、".join(items) if isinstance(items, (list, tuple)) else str(items)
        self.test_items_label.setText(item_text or "--")
        self.status_label.setText(str(context.get("status") or "同步待机"))
