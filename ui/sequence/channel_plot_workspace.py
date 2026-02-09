from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pyqtgraph as pg
from PyQt5.QtCore import QPoint, QRect, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class _TileSpec:
    cols: int
    win_w: int
    win_h: int
    gap: int
    pad: int


class ChannelPlotTitleBar(QWidget):
    def __init__(self, parent_window: "ChannelPlotSubWindow", title: str):
        super().__init__(parent_window)
        self._parent_window = parent_window
        self._drag_active = False
        self._drag_start_global: Optional[QPoint] = None
        self._drag_start_pos: Optional[QPoint] = None

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: white; font-weight: 600;")
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.setLayout(layout)
        self.setFixedHeight(26)
        self.setCursor(Qt.SizeAllCursor)
        self.setStyleSheet("background-color: #2f5aa8;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_start_global = event.globalPos()
            self._drag_start_pos = self._parent_window.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and self._drag_start_global is not None and self._drag_start_pos is not None:
            delta = event.globalPos() - self._drag_start_global
            new_pos = self._drag_start_pos + delta
            self._parent_window.request_move(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_active:
            self._drag_active = False
            self._drag_start_global = None
            self._drag_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ChannelPlotSubWindow(QFrame):
    def __init__(self, canvas: "ChannelPlotCanvas", channel_index: int):
        super().__init__(canvas)
        self._canvas = canvas
        self.channel_index = int(channel_index)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = None

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame{background: white; border: 1px solid #9aa7bd; border-radius: 2px;}")

        title = f"In{self.channel_index + 1}"
        self.title_bar = ChannelPlotTitleBar(self, title)

        self._setup_plot_style(self.plot_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_bar)
        layout.addWidget(self.plot_widget, stretch=1)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    @staticmethod
    def _setup_plot_style(plot_widget: pg.PlotWidget) -> None:
        plot_widget.setBackground("white")
        plot_widget.setLabel("left", "Amplitude(V)", **{"font-size": "18px"})
        plot_widget.setLabel("bottom", "Time(s)", **{"font-size": "18px"})
        plot_widget.showGrid(x=True, y=True)

        font = QFont()
        font.setPixelSize(18)
        b_axis = plot_widget.getAxis("bottom")
        l_axis = plot_widget.getAxis("left")
        b_axis.setTickFont(font)
        l_axis.setTickFont(font)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")

    def request_move(self, new_top_left: QPoint) -> bool:
        return self._canvas.try_move(self, new_top_left)

    def clear_plot(self) -> None:
        self.plot_widget.clear()
        self.plot_item = None

    def set_data(self, x, y) -> None:
        if self.plot_item is None:
            self.plot_item = self.plot_widget.plot(x, y, pen="k")
        else:
            self.plot_item.setData(x, y)


class ChannelPlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._windows: List[ChannelPlotSubWindow] = []
        # Make the workspace area visually distinct from the surrounding UI.
        self.setStyleSheet("background-color: #f6f8fc; border: 1px solid #c9d3e3;")

    def set_windows(self, windows: List[ChannelPlotSubWindow]) -> None:
        self._windows = list(windows)

    def windows(self) -> List[ChannelPlotSubWindow]:
        return list(self._windows)

    def try_move(self, win: ChannelPlotSubWindow, new_top_left: QPoint) -> bool:
        if win is None:
            return False

        bound = self.rect()
        w = win.width()
        h = win.height()

        x = max(0, min(int(new_top_left.x()), max(0, bound.width() - w)))
        y = max(0, min(int(new_top_left.y()), max(0, bound.height() - h)))
        new_rect = QRect(x, y, w, h)

        for other in self._windows:
            if other is win:
                continue
            if new_rect.intersects(other.geometry()):
                return False

        win.move(x, y)
        return True


class ChannelPlotWorkspace(QWidget):
    """
    Scrollable workspace that hosts draggable, non-overlapping plot subwindows.

    - Only the plot area is affected (used as a replacement of the old single PlotWidget).
    - Subwindows can be dragged within the workspace bounds, cannot overlap.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll = QScrollArea(self)
        # Resizable is important so the canvas width tracks the viewport; otherwise during early init
        # viewport width may be 0/1 and child windows get fully clipped.
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        try:
            self.scroll.viewport().setStyleSheet("background-color: #e9eef6;")
        except Exception:
            pass

        self.canvas = ChannelPlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.scroll.setWidget(self.canvas)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll)
        self.setLayout(layout)

        self._channel_indices: List[int] = []
        self._subwins: List[ChannelPlotSubWindow] = []

    def set_channels(self, channel_indices: List[int]) -> None:
        channel_indices = [int(i) for i in (channel_indices or [])]
        if not channel_indices:
            channel_indices = [0]

        if channel_indices == self._channel_indices and self._subwins:
            return

        for w in self._subwins:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass

        self._channel_indices = list(channel_indices)
        self._subwins = [ChannelPlotSubWindow(self.canvas, ch) for ch in self._channel_indices]
        for w in self._subwins:
            try:
                w.show()
            except Exception:
                pass
        self.canvas.set_windows(self._subwins)

        # Tile after the event loop gets a chance to layout the scroll viewport (width becomes valid).
        QTimer.singleShot(0, self._tile_subwindows)

    def subwindows(self) -> List[ChannelPlotSubWindow]:
        return list(self._subwins)

    def clear_plots(self) -> None:
        for w in self._subwins:
            w.clear_plot()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep canvas width in sync with the viewport, and ensure all windows remain in-bounds.
        self._tile_subwindows(keep_positions=True)
        if not self._layout_is_valid():
            # If resizing caused overlaps/out-of-bounds, fall back to a clean tiling layout.
            self._tile_subwindows(keep_positions=False)

    def _layout_is_valid(self) -> bool:
        if not self._subwins:
            return True
        bound = self.canvas.rect()
        for i, w in enumerate(self._subwins):
            if not bound.contains(w.geometry()):
                return False
            for j in range(i + 1, len(self._subwins)):
                if w.geometry().intersects(self._subwins[j].geometry()):
                    return False
        return True

    def _calc_tile_spec(self, viewport_w: int, viewport_h: int, num_windows: int) -> _TileSpec:
        gap = 12
        pad = 8
        min_win_w = 320
        min_win_h = 220

        usable_w = max(1, viewport_w - 2 * pad)
        cols = 2 if usable_w >= (min_win_w * 2 + gap) else 1

        win_w = int((usable_w - gap * (cols - 1)) / cols)
        win_w = max(220, min(win_w, usable_w))

        # Default behavior: keep a stable minimum height and let the scroll area handle overflow.
        usable_h = max(1, viewport_h - 2 * pad)
        rows = 2 if usable_h >= (min_win_h * 2 + gap) else 1

        win_h = int((usable_h - gap * (rows - 1)) / rows)
        win_h = max(220, min(win_h, usable_h))

        # Special-case: when showing 4 channels, prefer a 2x2 tiling that fills the viewport height.
        # If the viewport is short, the windows will shrink accordingly (no forced minimum in this mode).
        if int(num_windows) == 4 and viewport_h > 0:
            rows = (num_windows + cols - 1) // cols
            if rows == 2:
                usable_h = max(1, viewport_h - 2 * pad)
                filled_h = int((usable_h - gap * (rows - 1)) / rows)
                if filled_h >= 1:
                    win_h = filled_h

        return _TileSpec(cols=cols, win_w=win_w, win_h=win_h, gap=gap, pad=pad)

    def _tile_subwindows(self, keep_positions: bool = False) -> None:
        if not self._subwins:
            self.canvas.setMinimumHeight(max(1, int(self.height() or 1)))
            return

        viewport_w = int(self.scroll.viewport().width() or self.width() or 0)
        viewport_h = int(self.scroll.viewport().height() or self.height() or 0)
        if viewport_w < 50:
            # Layout not ready yet; retry shortly.
            QTimer.singleShot(0, lambda: self._tile_subwindows(keep_positions=keep_positions))
            return
        spec = self._calc_tile_spec(viewport_w, viewport_h, len(self._subwins))

        cols = spec.cols
        gap = spec.gap
        pad = spec.pad
        win_w = spec.win_w
        win_h = spec.win_h

        for w in self._subwins:
            w.setFixedSize(win_w, win_h)

        if keep_positions:
            # Only adjust canvas size and clamp positions; do not reflow windows.
            rows = (len(self._subwins) + cols - 1) // cols
        else:
            rows = (len(self._subwins) + cols - 1) // cols
            for idx, w in enumerate(self._subwins):
                r = idx // cols
                c = idx % cols
                x = pad + c * (win_w + gap)
                y = pad + r * (win_h + gap)
                w.move(x, y)

        canvas_h = pad + rows * win_h + (rows - 1) * gap + pad
        # With widgetResizable=True, width will follow the viewport; keep enough height for vertical scroll.
        self.canvas.setMinimumHeight(canvas_h)

        # Clamp all windows within new bounds.
        for w in self._subwins:
            self.canvas.try_move(w, w.pos())
