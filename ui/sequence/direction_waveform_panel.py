from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const


class DirectionWaveformCard(QFrame):
    def __init__(self, key: str, title: str, parent=None, on_play=None, on_mark=None):
        super().__init__(parent)
        self.key = str(key or "")
        self.on_play = on_play
        self.on_mark = on_mark
        self.plot_item = None
        self._last_x = None
        self._last_y = None
        self._audio_path = ""
        self._result_label = ""
        self.setObjectName("directionWaveformCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(ui_style_const.waveform_frame_style)

        header = QWidget(self)
        header.setObjectName("directionWaveformHeader")
        header.setFixedHeight(28)
        header.setStyleSheet(ui_style_const.waveform_title_bar_style)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 3, 8, 3)
        header_layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("QLabel { background: transparent; color: white; font-weight: 600; }")
        header_layout.addWidget(self.title_label, stretch=1)

        self.play_btn = self._create_action_button("播放", "播放录音", "conditionPlayButton")
        self.play_btn.clicked.connect(lambda: self._emit_play())
        header_layout.addWidget(self.play_btn, stretch=0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_plot_style(self.plot_widget)

        self.mark_panel = QWidget(self)
        self.mark_panel.setFixedWidth(74)
        self.mark_panel.setStyleSheet("QWidget { background: transparent; border: none; }")
        mark_layout = QVBoxLayout(self.mark_panel)
        mark_layout.setContentsMargins(8, 10, 8, 10)
        mark_layout.setSpacing(12)
        mark_layout.addStretch()
        self.ok_btn = self._create_mark_button("OK", "标记 OK", "conditionOkButton")
        self.ng_btn = self._create_mark_button("NG", "标记 NG", "conditionNgButton")
        self.ok_btn.clicked.connect(lambda: self._emit_mark("OK"))
        self.ng_btn.clicked.connect(lambda: self._emit_mark("NG"))
        mark_layout.addWidget(self.ok_btn)
        mark_layout.addWidget(self.ng_btn)
        mark_layout.addStretch()

        body = QWidget(self)
        body.setObjectName("directionWaveformBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.plot_widget, stretch=1)
        body_layout.addWidget(self.mark_panel, stretch=0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(body, stretch=1)
        self.set_mode("test")

    @staticmethod
    def _create_action_button(text: str, tooltip: str, object_name: str = "") -> QPushButton:
        button = QPushButton(text)
        if object_name:
            button.setObjectName(object_name)
        button.setFixedHeight(22)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setStyleSheet(ui_style_const.waveform_action_button_style)
        return button

    @staticmethod
    def _create_mark_button(text: str, tooltip: str, object_name: str = "") -> QPushButton:
        button = DirectionWaveformCard._create_action_button(text, tooltip, object_name)
        button.setFixedSize(58, 42)
        button.setStyleSheet(ui_style_const.waveform_mark_button_style)
        return button

    def _emit_play(self) -> None:
        if callable(self.on_play):
            self.on_play(self.key)

    def _emit_mark(self, label: str) -> None:
        if callable(self.on_mark):
            self.on_mark(self.key, label)

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title or ""))

    def set_meta(self, text: str) -> None:
        self.play_btn.setToolTip(str(text or "") or "播放录音")

    def set_mode(self, mode: str) -> None:
        is_mark = str(mode or "").lower() == "mark"
        self.mark_panel.setVisible(is_mark)

    def set_audio_path(self, path: str | None) -> None:
        self._audio_path = str(path or "")
        self.play_btn.setToolTip(self._audio_path or "当前工况暂无可播放录音")

    def set_result_label(self, label: str | None) -> None:
        self._result_label = str(label or "").upper()
        self.ok_btn.setEnabled(self._result_label != "OK")
        self.ng_btn.setEnabled(self._result_label != "NG")

    def clear_record_state(self) -> None:
        self.set_audio_path("")
        self.set_result_label("")

    @staticmethod
    def _setup_plot_style(plot_widget: pg.PlotWidget) -> None:
        plot_widget.setBackground("#FFFFFF")
        plot_widget.showGrid(x=True, y=True, alpha=0.25)
        plot_widget.setMouseEnabled(x=False, y=False)

        font = QFont()
        font.setPixelSize(12)
        for axis_name in ("bottom", "left"):
            axis = plot_widget.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setTextPen("#64748B")

    def clear_plot(self) -> None:
        self.plot_widget.clear()
        self.plot_item = None
        self._last_x = None
        self._last_y = None
        self.clear_record_state()

    def set_data(self, x, y) -> None:
        self._last_x, self._last_y = x, y
        if self.plot_item is None:
            self.plot_item = self.plot_widget.plot(x, y, pen=pg.mkPen("#3B82F6", width=1.4))
        else:
            self.plot_item.setData(x, y)

class DirectionWaveformPanel(QWidget):
    def __init__(self, parent=None, condition_configs=None, on_play_condition=None, on_mark_condition=None):
        super().__init__(parent)
        self._cards = {}
        self._audio_paths = {}
        self._result_labels = {}
        self._mode = "test"
        self._grid_cols = 0
        self._grid_rows = 0
        self.on_play_condition = on_play_condition
        self.on_mark_condition = on_mark_condition
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(ui_style_const.waveform_canvas_style)

        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(8, 0, 8, 8)
        self.grid.setSpacing(12)
        self.set_conditions(condition_configs)

    def set_conditions(self, condition_configs) -> None:
        self._clear_grid()
        self._reset_grid_stretches()
        self._cards = {}
        conditions = self._normalize_conditions(condition_configs)

        # Keep layout simple: choose 2~4 columns based on number of rpm points.
        # (No heavy responsive math; just a stable grid.)
        n = len(conditions)
        cols = 2 if n <= 4 else (3 if n <= 9 else 4)
        rows = max(1, math.ceil(n / cols)) if n else 1
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
        for r in range(rows):
            self.grid.setRowStretch(r, 1)
        self._grid_cols = cols
        self._grid_rows = rows

        for index, item in enumerate(conditions):
            name = str(item["name"] or "")
            card = DirectionWaveformCard(
                item["key"],
                f"{name} 波形",
                self,
                on_play=self._handle_play_condition,
                on_mark=self._handle_mark_condition,
            )
            card.set_mode(self._mode)
            card.set_audio_path(self._audio_paths.get(item["key"], ""))
            card.set_result_label(self._result_labels.get(item["key"], ""))
            self._cards[item["key"]] = card
            self.grid.addWidget(card, index // cols, index % cols)

        if not conditions:
            empty = QLabel("暂无工况配置")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(ui_style_const.motor_final_result_title_style)
            self.grid.addWidget(empty, 0, 0)
        self.grid.invalidate()
        self.updateGeometry()

    def set_direction_titles(self, titles) -> None:
        for key, title in (titles or {}).items():
            card = self._cards.get(str(key or ""))
            if card is not None:
                card.set_title(title)

    def clear_direction(self, direction: str) -> None:
        key = str(direction or "")
        card = self._cards.get(key)
        if card is not None:
            card.clear_plot()
        self._audio_paths.pop(key, None)
        self._result_labels.pop(key, None)

    def clear_plots(self) -> None:
        for card in self._cards.values():
            card.clear_plot()
        self._audio_paths = {}
        self._result_labels = {}

    def set_direction_data(self, direction: str, x, y) -> None:
        card = self._cards.get(str(direction or ""))
        if card is not None:
            card.set_data(x, y)

    def condition_keys(self):
        return list(self._cards.keys())

    def set_mode(self, mode: str) -> None:
        self._mode = "mark" if str(mode or "").lower() == "mark" else "test"
        for card in self._cards.values():
            card.set_mode(self._mode)

    def set_condition_audio_path(self, condition_key: str, path: str | None) -> None:
        key = str(condition_key or "")
        if not key:
            return
        if path:
            self._audio_paths[key] = str(path)
        else:
            self._audio_paths.pop(key, None)
        card = self._cards.get(key)
        if card is not None:
            card.set_audio_path(self._audio_paths.get(key, ""))

    def set_condition_result(self, condition_key: str, label: str | None) -> None:
        key = str(condition_key or "")
        if not key:
            return
        normalized = str(label or "").strip().upper()
        if normalized in ("OK", "NG"):
            self._result_labels[key] = normalized
        else:
            self._result_labels.pop(key, None)
        card = self._cards.get(key)
        if card is not None:
            card.set_result_label(self._result_labels.get(key, ""))

    def _handle_play_condition(self, condition_key: str) -> None:
        if callable(self.on_play_condition):
            self.on_play_condition(str(condition_key or ""))

    def _handle_mark_condition(self, condition_key: str, label: str) -> None:
        if callable(self.on_mark_condition):
            self.on_mark_condition(str(condition_key or ""), str(label or ""))

    @staticmethod
    def _normalize_conditions(condition_configs):
        result = []
        used_keys = set()
        for index, item in enumerate(condition_configs or []):
            if not isinstance(item, dict):
                continue
            name = str(
                item.get("display_name")
                or item.get("condition_name")
                or item.get("name")
                or item.get("test_queue")
                or ""
            ).strip()
            if not name:
                continue
            base_key = str(
                item.get("key")
                or item.get("trigger_state")
                or item.get("test_queue")
                or index
            ).strip()
            key = base_key
            if key in used_keys:
                key = f"{base_key}#{index + 1}"
            used_keys.add(key)
            result.append({"key": key, "name": name})
        return result

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _reset_grid_stretches(self):
        for c in range(self._grid_cols):
            self.grid.setColumnStretch(c, 0)
            self.grid.setColumnMinimumWidth(c, 0)
        for r in range(self._grid_rows):
            self.grid.setRowStretch(r, 0)
            self.grid.setRowMinimumHeight(r, 0)
