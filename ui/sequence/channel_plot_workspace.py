from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pyqtgraph as pg
from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from ui.custom_ui_widget.widgets import PushButton, Label
from ui.sequence.channel_plot_workspace_controller import ChannelPlotWorkspaceController
from ui.sequence.channel_plot_workspace_model import ChannelPlotWorkspaceModel
from ui.ui_src import ui_resources

_RETRY_DELAY_MS = 16


@dataclass(frozen=True)
class _TileSpec:
    cols: int
    height_rows: int
    win_w: int
    win_h: int
    gap: int
    pad: int


class ChannelPlotTitleBar(QWidget):
    close_requested = pyqtSignal()

    def __init__(self, parent_window: "ChannelPlotSubWindow", title: str):
        super().__init__(parent_window)
        self.setObjectName("ChannelPlotTitleBar")
        self._parent_window = parent_window
        self._drag_active = False
        self._drag_start_global: Optional[QPoint] = None
        self._drag_start_pos: Optional[QPoint] = None

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 0, 2)
        layout.setSpacing(8)

        self.title_label = Label(title)
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.close_btn = PushButton()
        self.close_btn.setIcon(QIcon(":/ui/icon/fork.png"))
        self.close_btn.setCursor(Qt.ArrowCursor)
        self.close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_btn)

        self.setLayout(layout)
        self.setFixedHeight(26)
        self.setCursor(Qt.SizeAllCursor)

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
    hide_requested = pyqtSignal(int)

    def __init__(self, canvas: "ChannelPlotCanvas", channel_index: int):
        super().__init__(canvas)
        self._canvas = canvas
        self.channel_index = int(channel_index)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = None

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("ChannelPlotSubWindow")

        title = f"In{self.channel_index + 1}"
        self.title_bar = ChannelPlotTitleBar(self, title)
        self.title_bar.close_requested.connect(self._emit_hide_requested)

        self._setup_plot_style(self.plot_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_bar)
        layout.addWidget(self.plot_widget, stretch=1)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    @staticmethod
    def _setup_plot_style(plot_widget: pg.PlotWidget) -> None:
        font_size = ui_style_const.scale_size_px(18)
        plot_widget.setMenuEnabled(True)
        plot_widget.setBackground("white")
        plot_widget.setLabel("left", "Amplitude(V)", **{"font-size": f"{font_size}px"})
        plot_widget.setLabel("bottom", "Time(s)", **{"font-size": f"{font_size}px"})
        plot_widget.showGrid(x=True, y=True)

        font = QFont()
        font.setPixelSize(font_size)
        b_axis = plot_widget.getAxis("bottom")
        l_axis = plot_widget.getAxis("left")
        b_axis.setTickFont(font)
        l_axis.setTickFont(font)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")

    def request_move(self, new_top_left: QPoint) -> bool:
        return self._canvas.try_move(self, new_top_left)

    def _emit_hide_requested(self) -> None:
        self.hide_requested.emit(self.channel_index)

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
        self.setObjectName("ChannelPlotCanvas")
        self._windows: List[ChannelPlotSubWindow] = []

    def set_windows(self, windows: List[ChannelPlotSubWindow]) -> None:
        self._windows = list(windows)

    def windows(self) -> List[ChannelPlotSubWindow]:
        return list(self._windows)

    def _bounded_rect(self, win: ChannelPlotSubWindow, new_top_left: QPoint) -> QRect:
        bound = self.rect()
        w = win.width()
        h = win.height()

        x = max(0, min(int(new_top_left.x()), max(0, bound.width() - w)))
        y = max(0, min(int(new_top_left.y()), max(0, bound.height() - h)))
        return QRect(x, y, w, h)

    def can_place(self, win: ChannelPlotSubWindow, new_top_left: QPoint) -> bool:
        if win is None:
            return False

        bound = self.rect()
        new_rect = self._bounded_rect(win, new_top_left)
        if not bound.contains(new_rect):
            return False

        for other in self._windows:
            if other is win or not other.isVisible():
                continue
            if new_rect.intersects(other.geometry()):
                return False
        return True

    def try_move(self, win: ChannelPlotSubWindow, new_top_left: QPoint) -> bool:
        if not self.can_place(win, new_top_left):
            return False

        win.move(self._bounded_rect(win, new_top_left).topLeft())
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
        self.scroll.setObjectName("ChannelPlotWorkspaceScrollArea")
        # Resizable is important so the canvas width tracks the viewport; otherwise during early init
        # viewport width may be 0/1 and child windows get fully clipped.
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.canvas = ChannelPlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.scroll.setWidget(self.canvas)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll)
        self.setLayout(layout)

        self._model = ChannelPlotWorkspaceModel(parent=self)
        self._controller = ChannelPlotWorkspaceController(self._model, menu_parent=self)
        self._subwins_by_channel: Dict[int, ChannelPlotSubWindow] = {}
        self._single_canvas_mode = False
        self._single_canvas_visibility_guard = False
        self._last_tile_cols: Optional[int] = None
        self._last_tile_height_rows: Optional[int] = None
        self._tile_retry_keep_positions = False
        self._tile_retry_timer = QTimer(self)
        self._tile_retry_timer.setSingleShot(True)
        self._tile_retry_timer.timeout.connect(self._run_tile_retry)
        self._restore_retry_channels: set[int] = set()
        self._restore_retry_timer = QTimer(self)
        self._restore_retry_timer.setSingleShot(True)
        self._restore_retry_timer.timeout.connect(self._run_restore_retries)

        self._bind_context_menu(self)
        self._bind_context_menu(self.canvas)

        self._model.channels_reset.connect(self._rebuild_subwindows)
        self._model.visibility_changed.connect(self._apply_channel_visibility)

    def set_channels(self, channel_indices: List[int]) -> None:
        self._single_canvas_mode = False
        self._cancel_restore_retries()
        self._controller.reset_channels(channel_indices)

    def set_single_canvas_mode(self, channel_index: int = 0) -> None:
        self._single_canvas_mode = True
        self._cancel_restore_retries()
        self._controller.reset_channels([int(channel_index)])
        self._apply_single_canvas_layout()

    def subwindows(self) -> List[ChannelPlotSubWindow]:
        return self._visible_subwindows()

    def subwindow(self, channel_index: int) -> Optional[ChannelPlotSubWindow]:
        return self._subwins_by_channel.get(int(channel_index))

    def model(self) -> ChannelPlotWorkspaceModel:
        return self._model

    def controller(self) -> ChannelPlotWorkspaceController:
        return self._controller

    def clear_plots(self) -> None:
        for w in self._subwins_by_channel.values():
            w.clear_plot()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep canvas width in sync with the viewport, and ensure all windows remain in-bounds.
        self._tile_subwindows(keep_positions=True)
        if not self._layout_is_valid():
            # If resizing caused overlaps/out-of-bounds, fall back to a clean tiling layout.
            self._tile_subwindows(keep_positions=False)
        self._resume_pending_restore_retries()

    def showEvent(self, event):
        super().showEvent(event)
        if self._single_canvas_mode:
            self._apply_single_canvas_layout()
            QTimer.singleShot(0, self._apply_single_canvas_layout)
        self._resume_pending_restore_retries()

    def _layout_is_valid(self) -> bool:
        visible_windows = self._visible_subwindows()
        if not visible_windows:
            return True
        bound = self.canvas.rect()
        for i, w in enumerate(visible_windows):
            if not bound.contains(w.geometry()):
                return False
            for j in range(i + 1, len(visible_windows)):
                if w.geometry().intersects(visible_windows[j].geometry()):
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
        win_w = min(usable_w, max(220, win_w))

        # Default behavior: keep a stable minimum height and let the scroll area handle overflow.
        usable_h = max(1, viewport_h - 2 * pad)
        height_rows = 2 if usable_h >= (min_win_h * 2 + gap) else 1

        win_h = int((usable_h - gap * (height_rows - 1)) / height_rows)
        win_h = max(220, min(win_h, usable_h))

        # Special-case: when showing 4 channels, prefer a 2x2 tiling that fills the viewport height
        # only after the independent height threshold also says two rows fit.
        if int(num_windows) == 4 and viewport_h > 0:
            rows = (num_windows + cols - 1) // cols
            if rows == 2 and height_rows == 2:
                usable_h = max(1, viewport_h - 2 * pad)
                filled_h = int((usable_h - gap * (rows - 1)) / rows)
                if filled_h >= 1:
                    win_h = filled_h

        return _TileSpec(
            cols=cols,
            height_rows=height_rows,
            win_w=win_w,
            win_h=win_h,
            gap=gap,
            pad=pad,
        )

    def _tile_subwindows(self, keep_positions: bool = False) -> None:
        if self._single_canvas_mode:
            self._apply_single_canvas_layout()
            return

        visible_windows = self._sync_canvas_windows()

        if not visible_windows:
            self._last_tile_cols = None
            self._last_tile_height_rows = None
            self._update_canvas_height(visible_windows)
            return

        viewport_w = int(self.scroll.viewport().width() or self.width() or 0)
        viewport_h = int(self.scroll.viewport().height() or self.height() or 0)
        if viewport_w < 50:
            # Layout is not ready yet; coalesce retries so hidden pages do not spin the event loop.
            self._schedule_tile_retry(keep_positions=keep_positions)
            return
        spec = self._calc_tile_spec(viewport_w, viewport_h, len(visible_windows))
        if (
            keep_positions
            and self._last_tile_cols is not None
            and self._last_tile_height_rows is not None
            and (spec.cols != self._last_tile_cols or spec.height_rows != self._last_tile_height_rows)
        ):
            keep_positions = False
        self._last_tile_cols = spec.cols
        self._last_tile_height_rows = spec.height_rows

        cols = spec.cols
        gap = spec.gap
        pad = spec.pad
        win_w = spec.win_w
        win_h = spec.win_h

        for w in visible_windows:
            w.setFixedSize(win_w, win_h)

        if keep_positions:
            # Only adjust canvas size and clamp positions; do not reflow windows.
            rows = (len(visible_windows) + cols - 1) // cols
        else:
            rows = (len(visible_windows) + cols - 1) // cols
            for idx, w in enumerate(visible_windows):
                r = idx // cols
                c = idx % cols
                x = pad + c * (win_w + gap)
                y = pad + r * (win_h + gap)
                w.move(x, y)

        self._update_canvas_height(
            visible_windows,
            tile_rows=rows,
            win_h=win_h,
            gap=gap,
            pad=pad,
        )
        self._resume_pending_restore_retries()

        # Clamp all windows within new bounds.
        for w in visible_windows:
            self.canvas.try_move(w, w.pos())
        self._update_canvas_height(
            visible_windows,
            tile_rows=rows,
            win_h=win_h,
            gap=gap,
            pad=pad,
        )

    def _apply_single_canvas_layout(self) -> None:
        channel_indices = self._model.channel_indices()
        if not channel_indices:
            self._update_canvas_height([])
            return

        channel_index = channel_indices[0]
        try:
            if not self._model.is_visible(channel_index):
                if self._single_canvas_visibility_guard:
                    self._update_canvas_height([])
                    return
                self._single_canvas_visibility_guard = True
                try:
                    self._model.set_visible(channel_index, True)
                finally:
                    self._single_canvas_visibility_guard = False
        except KeyError:
            self._update_canvas_height([])
            return

        subwindow = self._subwins_by_channel.get(channel_index)
        if subwindow is None:
            self._update_canvas_height([])
            return

        for window in self._subwins_by_channel.values():
            if window is subwindow:
                window.title_bar.hide()
            else:
                window.hide()

        if self.isVisible():
            viewport_w = int(self.scroll.viewport().width() or self.width() or 1)
            viewport_h = int(self.scroll.viewport().height() or self.height() or 1)
        else:
            viewport_w = int(self.width() or self.scroll.viewport().width() or 1)
            viewport_h = int(self.height() or self.scroll.viewport().height() or 1)
        width = max(1, viewport_w)
        height = max(1, viewport_h)
        self.canvas.setMinimumHeight(height)
        self.canvas.resize(width, height)
        subwindow.setFixedSize(width, height)
        subwindow.move(0, 0)
        subwindow.show()
        self.canvas.set_windows([subwindow])

    def _schedule_tile_retry(self, keep_positions: bool) -> None:
        keep_positions = bool(keep_positions)
        if self._tile_retry_timer.isActive():
            self._tile_retry_keep_positions = self._tile_retry_keep_positions and keep_positions
            return

        self._tile_retry_keep_positions = keep_positions
        self._tile_retry_timer.start(_RETRY_DELAY_MS)

    def _run_tile_retry(self) -> None:
        keep_positions = self._tile_retry_keep_positions
        self._tile_retry_keep_positions = False
        self._tile_subwindows(keep_positions=keep_positions)

    def _visible_subwindows(self) -> List[ChannelPlotSubWindow]:
        return [
            self._subwins_by_channel[channel_index]
            for channel_index in self._model.visible_channels()
            if channel_index in self._subwins_by_channel
        ]

    def all_subwindows(self) -> List[ChannelPlotSubWindow]:
        return [
            self._subwins_by_channel[channel_index]
            for channel_index in self._model.channel_indices()
            if channel_index in self._subwins_by_channel
        ]

    def _sync_canvas_windows(self) -> List[ChannelPlotSubWindow]:
        visible_windows = self._visible_subwindows()
        self.canvas.set_windows(visible_windows)
        return visible_windows

    def _update_canvas_height(
        self,
        visible_windows: Optional[List[ChannelPlotSubWindow]] = None,
        tile_rows: Optional[int] = None,
        win_h: Optional[int] = None,
        gap: int = 12,
        pad: int = 8,
    ) -> None:
        if visible_windows is None:
            visible_windows = self._visible_subwindows()

        canvas_h = max(1, int(self.scroll.viewport().height() or self.height() or 1))
        if visible_windows:
            bottom_edge = max(w.geometry().bottom() + 1 for w in visible_windows)
            canvas_h = max(canvas_h, bottom_edge + pad)

        if tile_rows is not None and win_h is not None:
            tiled_h = pad + tile_rows * win_h + max(0, tile_rows - 1) * gap + pad
            canvas_h = max(canvas_h, tiled_h)

        self.canvas.setMinimumHeight(canvas_h)

    def _ensure_canvas_height(self, canvas_h: int) -> None:
        self.canvas.setMinimumHeight(max(self.canvas.minimumHeight(), int(canvas_h)))

    def _restore_layout_is_ready(self) -> bool:
        if not self.isVisible():
            return False

        viewport = self.scroll.viewport()
        viewport_w = int(viewport.width() or self.width() or 0)
        viewport_h = int(viewport.height() or self.height() or 0)
        return viewport_w >= 50 and viewport_h > 0

    def _resume_pending_restore_retries(self) -> None:
        if not self._restore_retry_channels:
            return
        if not self._restore_layout_is_ready():
            return
        if not self._restore_retry_timer.isActive():
            self._restore_retry_timer.start(_RETRY_DELAY_MS)

    def _restore_subwindow_position(self, subwindow: ChannelPlotSubWindow) -> bool:
        visible_windows = self._visible_subwindows()
        viewport_w = int(self.scroll.viewport().width() or self.width() or 0)
        viewport_h = int(self.scroll.viewport().height() or self.height() or 0)
        if not self._restore_layout_is_ready():
            return False

        spec = self._calc_tile_spec(viewport_w, viewport_h, len(visible_windows))
        rows = max(1, (len(visible_windows) + spec.cols - 1) // spec.cols)
        subwindow.setFixedSize(spec.win_w, spec.win_h)
        shown_windows = [w for w in visible_windows if w is not subwindow and w.isVisible()]
        self._update_canvas_height(
            shown_windows,
            tile_rows=rows,
            win_h=spec.win_h,
            gap=spec.gap,
            pad=spec.pad,
        )

        if self.canvas.try_move(subwindow, subwindow.pos()):
            return True

        for idx in range(rows * spec.cols):
            row = idx // spec.cols
            col = idx % spec.cols
            candidate = QPoint(
                spec.pad + col * (spec.win_w + spec.gap),
                spec.pad + row * (spec.win_h + spec.gap),
            )
            if self.canvas.try_move(subwindow, candidate):
                return True

        fallback_y = spec.pad
        if shown_windows:
            fallback_y = max(w.geometry().bottom() + 1 for w in shown_windows) + spec.gap
        self._ensure_canvas_height(fallback_y + spec.win_h + spec.pad)
        return self.canvas.try_move(subwindow, QPoint(spec.pad, fallback_y))

    def _show_subwindow_when_ready(self, channel_index: int) -> None:
        channel_index = int(channel_index)
        try:
            if not self._model.is_visible(channel_index):
                return
        except KeyError:
            return

        subwindow = self._subwins_by_channel.get(channel_index)
        if subwindow is None or subwindow.isVisible():
            return

        self._sync_canvas_windows()
        if not self._restore_layout_is_ready():
            self._schedule_restore_retry(channel_index)
            return
        if self._restore_subwindow_position(subwindow):
            subwindow.show()
            self._sync_canvas_windows()
            self._update_canvas_height()
            return

        self._schedule_restore_retry(channel_index)

    def _schedule_restore_retry(self, channel_index: int) -> None:
        self._restore_retry_channels.add(int(channel_index))
        if self._restore_layout_is_ready() and not self._restore_retry_timer.isActive():
            self._restore_retry_timer.start(_RETRY_DELAY_MS)

    def _run_restore_retries(self) -> None:
        pending_channels = sorted(self._restore_retry_channels)
        self._restore_retry_channels.clear()
        for channel_index in pending_channels:
            self._show_subwindow_when_ready(channel_index)

    def _cancel_restore_retries(self) -> None:
        self._restore_retry_channels.clear()
        if self._restore_retry_timer.isActive():
            self._restore_retry_timer.stop()

    def _rebuild_subwindows(self) -> None:
        self._cancel_restore_retries()
        for w in self._subwins_by_channel.values():
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass

        self._subwins_by_channel = {}
        for channel_index in self._model.channel_indices():
            subwindow = ChannelPlotSubWindow(self.canvas, channel_index)
            subwindow.hide_requested.connect(self._controller.hide_channel)
            self._subwins_by_channel[channel_index] = subwindow

        if self._single_canvas_mode:
            self._apply_single_canvas_layout()
            return

        for subwindow in self._subwins_by_channel.values():
            subwindow.title_bar.show()

        for channel_index in self._model.channel_indices():
            self._apply_channel_visibility(channel_index, self._model.is_visible(channel_index), retile=False)

        # Tile immediately when the viewport is already ready; otherwise `_tile_subwindows()`
        # will keep the coalesced positive-delay retry path for not-ready layouts.
        self._tile_subwindows(keep_positions=False)

    def _apply_channel_visibility(self, channel_index: int, visible: bool, retile: bool = True) -> None:
        subwindow = self._subwins_by_channel.get(int(channel_index))
        if subwindow is None:
            return

        if self._single_canvas_mode:
            if not visible and not self._single_canvas_visibility_guard:
                self._single_canvas_visibility_guard = True
                try:
                    self._model.set_visible(int(channel_index), True)
                finally:
                    self._single_canvas_visibility_guard = False
            subwindow.show()
            self._apply_single_canvas_layout()
            return

        subwindow.title_bar.show()
        if visible:
            self._sync_canvas_windows()
            if self._restore_subwindow_position(subwindow):
                subwindow.show()
                self._sync_canvas_windows()
            else:
                subwindow.hide()
                self._schedule_restore_retry(channel_index)
        else:
            subwindow.hide()
            self._sync_canvas_windows()

        self._update_canvas_height()

    def _bind_context_menu(self, widget: QWidget) -> None:
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, source_widget=widget: self._show_context_menu_from(source_widget, pos)
        )

    def _show_context_menu_from(self, source_widget: QWidget, pos: QPoint) -> None:
        if self._single_canvas_mode:
            return
        global_pos = source_widget.mapToGlobal(pos)
        canvas_pos = self.canvas.mapFromGlobal(global_pos)
        if self.canvas.rect().contains(canvas_pos):
            child = self.canvas.childAt(canvas_pos)
            if child is not None and child is not self.canvas:
                return
        self._controller.show_context_menu(global_pos)


def load_qss():
    from PyQt5.QtCore import QFile, QTextStream

    path = ":/ui/style/dongyuan_style.qss"
    # path = ":/ui/style/jingcheng_style.qss"
    file = QFile(path)
    if not file.open(QFile.ReadOnly | QFile.Text):
        raise RuntimeError(f"Failed to open QSS: {path}")
    stream = QTextStream(file)
    qss = stream.readAll()
    file.close()
    return qss


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyleSheet(load_qss())
    window = ChannelPlotWorkspace()
    window.set_channels([0, 1, 2, 3])
    window.show()
    sys.exit(app.exec_())
