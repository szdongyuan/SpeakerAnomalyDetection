from __future__ import annotations

from typing import Dict

import pyqtgraph as pg
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


class DirectionWaveformCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.plot_item = None

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QFrame{background: white; border: 1px solid #9aa7bd; border-radius: 2px;}")

        self.title_label = QLabel(title)
        self.title_label.setFixedHeight(26)
        self.title_label.setStyleSheet("background-color: #2f5aa8; color: white; font-weight: 600; padding-left: 8px;")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_plot_style(self.plot_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.plot_widget, stretch=1)
        self.setLayout(layout)

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title or ""))

    @staticmethod
    def _setup_plot_style(plot_widget: pg.PlotWidget) -> None:
        plot_widget.setBackground("white")
        plot_widget.setLabel("left", "Amplitude(V)", **{"font-size": "18px"})
        plot_widget.setLabel("bottom", "Time(s)", **{"font-size": "18px"})
        plot_widget.showGrid(x=True, y=True)

        font = QFont()
        font.setPixelSize(18)
        bottom_axis = plot_widget.getAxis("bottom")
        left_axis = plot_widget.getAxis("left")
        bottom_axis.setTickFont(font)
        left_axis.setTickFont(font)
        bottom_axis.setTextPen("black")
        left_axis.setTextPen("black")

    def clear_plot(self) -> None:
        self.plot_widget.clear()
        self.plot_item = None

    def set_data(self, x, y) -> None:
        if self.plot_item is None:
            self.plot_item = self.plot_widget.plot(x, y, pen="k")
        else:
            self.plot_item.setData(x, y)


class DirectionWaveformPanel(QWidget):
    """
    Fixed two-panel waveform display for forward/reverse motor runs.

    Unlike ChannelPlotWorkspace, this panel is managed by Qt layouts and does not
    support dragging, overlap checks, or manual fixed-size tiling.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #f6f8fc; border: 1px solid #c9d3e3;")

        self._cards: Dict[str, DirectionWaveformCard] = {
            "forward": DirectionWaveformCard("正转波形", self),
            "reverse": DirectionWaveformCard("反转波形", self),
        }

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        layout.addWidget(self._cards["forward"], stretch=1)
        layout.addWidget(self._cards["reverse"], stretch=1)
        self.setLayout(layout)

    def set_direction_titles(self, titles: Dict[str, str]) -> None:
        for direction, title in (titles or {}).items():
            card = self._cards.get(str(direction or ""))
            if card is not None:
                card.set_title(title)

    def clear_direction(self, direction: str) -> None:
        card = self._cards.get(str(direction or ""))
        if card is not None:
            card.clear_plot()

    def clear_plots(self) -> None:
        for card in self._cards.values():
            card.clear_plot()

    def set_direction_data(self, direction: str, x, y) -> None:
        card = self._cards.get(str(direction or ""))
        if card is not None:
            card.set_data(x, y)
