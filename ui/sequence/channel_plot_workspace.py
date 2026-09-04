from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
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

from consts import ui_style_const
from consts.recording_preview_consts import (
    PLOT_PRESENTATION_COMPLETE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)


@dataclass(frozen=True)
class _ChannelPlotState:
    data: tuple[np.ndarray, np.ndarray] | None
    presentation_mode: str
    x_range: tuple[float, float]
    x_auto_range: bool | float


class ChannelPlotPresentationMixin:
    """Shared live/final curve projection and opaque rollback state."""

    def _initialize_plot_presentation_lifecycle(self) -> None:
        self._deferred_view_guard = None
        self._deferred_view_guard_timer = None

    def _set_curve_data(self, x, y):
        raise NotImplementedError

    def _set_presentation_mode(self, mode: str) -> None:
        if mode not in (
            PLOT_PRESENTATION_COMPLETE,
            PREVIEW_TIME_MODE_CUMULATIVE,
            PREVIEW_TIME_MODE_RELATIVE_LATEST,
        ):
            raise ValueError(f"unsupported plot presentation mode: {mode!r}")
        self.presentation_mode = mode
        self.is_live_preview = mode in (
            PREVIEW_TIME_MODE_RELATIVE_LATEST,
            PREVIEW_TIME_MODE_CUMULATIVE,
        )

    @staticmethod
    def _restore_auto_range_state(view_box, axis, value) -> None:
        view_box.enableAutoRange(axis=axis, enable=value)
        state_index = 0 if axis == pg.ViewBox.XAxis else 1
        # pyqtgraph normalizes True to 1.0 and schedules a paint-time update.
        # Keep the opaque snapshot value exact and let only later item changes
        # request the next auto-range calculation.
        view_box.state["autoRange"][state_index] = value
        view_box._autoRangeNeedsUpdate = False

    @classmethod
    def _restore_axis_view_state(cls, view_box, axis, axis_range, auto_range):
        range_name = "xRange" if axis == pg.ViewBox.XAxis else "yRange"
        view_box.setRange(
            **{
                range_name: axis_range,
                "padding": 0,
                "disableAutoRange": False,
            }
        )
        cls._restore_auto_range_state(view_box, axis, auto_range)

    def _release_deferred_view_guard(self, *_args) -> None:
        guard = getattr(self, "_deferred_view_guard", None)
        timer = getattr(self, "_deferred_view_guard_timer", None)
        self._deferred_view_guard = None
        self._deferred_view_guard_timer = None

        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                # Parent-driven Qt teardown may already have deleted children.
                pass
            try:
                timer.timeout.disconnect()
            except (RuntimeError, TypeError):
                # Qt may already have removed the private timeout connection.
                pass
            try:
                timer.deleteLater()
            except RuntimeError:
                # A parent-driven teardown may already have deleted the timer.
                pass

        if guard is None:
            return
        (
            view_box,
            handler,
            plot_item,
            original_set_range,
            original_enable_auto_range,
            _timer,
        ) = guard
        try:
            view_box.sigRangeChanged.disconnect(handler)
        except (RuntimeError, TypeError):
            # Qt can destroy or auto-disconnect the ViewBox before its owning
            # row's final Python-side cleanup reaches this wrapper.
            pass
        try:
            view_box.setRange = original_set_range
        except RuntimeError:
            # Restoring a Python override is impossible after C++ deletion.
            pass
        try:
            view_box.enableAutoRange = original_enable_auto_range
        except RuntimeError:
            # Restoring a Python override is impossible after C++ deletion.
            pass
        if plot_item is not None:
            try:
                plot_item.sigPlotChanged.disconnect(
                    self._release_deferred_view_guard
                )
            except (RuntimeError, TypeError):
                # PlotDataItem follows the same parent-owned Qt lifecycle.
                pass

    def _install_deferred_view_guard(
        self,
        view_box,
        *,
        x_view_state=None,
        y_view_state=None,
    ) -> None:
        self._release_deferred_view_guard()
        restoring_guarded_view_state = False
        guard = None
        release_timer = QTimer(self)
        release_timer.setSingleShot(True)

        def release_if_current():
            if getattr(self, "_deferred_view_guard", None) is guard:
                self._release_deferred_view_guard()

        def enforce_view_state(*_args):
            nonlocal restoring_guarded_view_state
            if restoring_guarded_view_state:
                return
            restoring_guarded_view_state = True
            try:
                if x_view_state is not None:
                    self._restore_axis_view_state(
                        view_box,
                        pg.ViewBox.XAxis,
                        *x_view_state,
                    )
                if y_view_state is not None:
                    self._restore_axis_view_state(
                        view_box,
                        pg.ViewBox.YAxis,
                        *y_view_state,
                    )
            finally:
                restoring_guarded_view_state = False
            release_timer.start(0)

        plot_item = self.plot_item
        original_set_range = view_box.setRange
        original_enable_auto_range = view_box.enableAutoRange

        def release_before_user_range_change(*args, **kwargs):
            # Presentation updates can queue more than one ViewBox layout pass.
            # Keep the guard for those internal passes, but remove it before the
            # next explicit range request so it cannot override user zoom/pan.
            if kwargs.get("disableAutoRange", True) is not False:
                self._release_deferred_view_guard()
            return original_set_range(*args, **kwargs)

        def release_before_user_auto_range_change(*args, **kwargs):
            # Guard restoration must reinstate its captured auto-range state,
            # while an explicit caller request takes ownership immediately.
            if not restoring_guarded_view_state:
                self._release_deferred_view_guard()
            return original_enable_auto_range(*args, **kwargs)

        guard = (
            view_box,
            enforce_view_state,
            plot_item,
            original_set_range,
            original_enable_auto_range,
            release_timer,
        )
        self._deferred_view_guard = guard
        self._deferred_view_guard_timer = release_timer
        view_box.setRange = release_before_user_range_change
        view_box.enableAutoRange = release_before_user_auto_range_change
        release_timer.timeout.connect(release_if_current)
        view_box.sigRangeChanged.connect(enforce_view_state)
        if plot_item is not None:
            plot_item.sigPlotChanged.connect(
                self._release_deferred_view_guard
            )

        # If no guarded range signal arrives, do not retain an idle connection.
        release_timer.start(100)

    def closeEvent(self, event) -> None:
        self._release_deferred_view_guard()
        super().closeEvent(event)

    def deleteLater(self) -> None:
        self._release_deferred_view_guard()
        super().deleteLater()

    @contextmanager
    def _preserve_y_view_state(self):
        self._release_deferred_view_guard()
        view_box = self.plot_widget.getViewBox()
        y_range = self.plot_widget.viewRange()[1]
        y_range = (float(y_range[0]), float(y_range[1]))
        y_auto_range = view_box.state["autoRange"][1]
        try:
            yield view_box
        finally:
            y_view_state = (y_range, y_auto_range)
            self._restore_axis_view_state(
                view_box,
                pg.ViewBox.YAxis,
                *y_view_state,
            )
            self._install_deferred_view_guard(
                view_box,
                y_view_state=y_view_state,
            )

    def clear_plot(self) -> None:
        with self._preserve_y_view_state() as view_box:
            self.plot_widget.clear()
            self.plot_item = None
            self._set_presentation_mode(PLOT_PRESENTATION_COMPLETE)
            view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            view_box.updateAutoRange()

    def set_relative_preview_data(self, x, y):
        with self._preserve_y_view_state() as view_box:
            updated = self._set_curve_data(x, y)
            if updated is False:
                return False
            view_box.disableAutoRange(axis=pg.ViewBox.XAxis)
            self.plot_widget.setXRange(-10.0, 0.0, padding=0)
            self._set_presentation_mode(PREVIEW_TIME_MODE_RELATIVE_LATEST)
            return updated

    def set_live_data(self, x, y):
        """Compatibility alias for the relative latest preview."""
        return self.set_relative_preview_data(x, y)

    def set_cumulative_preview_data(self, x, y):
        with self._preserve_y_view_state() as view_box:
            updated = self._set_curve_data(x, y)
            if updated is False:
                return False
            view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            view_box.updateAutoRange()
            self._set_presentation_mode(PREVIEW_TIME_MODE_CUMULATIVE)
            return updated

    def set_data(self, x, y):
        with self._preserve_y_view_state() as view_box:
            updated = self._set_curve_data(x, y)
            if updated is False:
                return False
            self._set_presentation_mode(PLOT_PRESENTATION_COMPLETE)
            view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            view_box.updateAutoRange()
            return updated

    def snapshot_plot_state(self):
        """Copy curve and X-view state for an opaque rollback snapshot."""
        data = None
        if self.plot_item is not None:
            x_data, y_data = self.plot_item.getData()
            data = (
                np.asarray([]) if x_data is None else np.asarray(x_data).copy(),
                np.asarray([]) if y_data is None else np.asarray(y_data).copy(),
            )
        view_box = self.plot_widget.getViewBox()
        x_range = self.plot_widget.viewRange()[0]
        return _ChannelPlotState(
            data=data,
            presentation_mode=self.presentation_mode,
            x_range=(float(x_range[0]), float(x_range[1])),
            x_auto_range=view_box.state["autoRange"][0],
        )

    def restore_plot_state(self, state) -> None:
        """Restore an opaque snapshot without changing the channel identity."""
        if state is None:
            self.clear_plot()
            return

        if not isinstance(state, _ChannelPlotState):
            x_data, y_data = state
            self.set_data(x_data, y_data)
            return

        self._release_deferred_view_guard()
        if state.data is None:
            self.plot_widget.clear()
            self.plot_item = None
        else:
            self._set_curve_data(*state.data)

        self._set_presentation_mode(state.presentation_mode)
        view_box = self.plot_widget.getViewBox()
        y_range = self.plot_widget.viewRange()[1]
        y_view_state = (
            (float(y_range[0]), float(y_range[1])),
            view_box.state["autoRange"][1],
        )
        view_box.disableAutoRange(axis=pg.ViewBox.XAxis)
        self.plot_widget.setXRange(*state.x_range, padding=0)
        x_view_state = (state.x_range, state.x_auto_range)
        self._restore_axis_view_state(
            view_box,
            pg.ViewBox.XAxis,
            *x_view_state,
        )
        self._install_deferred_view_guard(
            view_box,
            x_view_state=x_view_state,
            y_view_state=y_view_state,
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
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.setLayout(layout)
        self.setFixedHeight(26)
        self.setCursor(Qt.SizeAllCursor)
        self.setStyleSheet(ui_style_const.waveform_title_bar_style)

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


class ChannelPlotSubWindow(ChannelPlotPresentationMixin, QFrame):
    def __init__(self, canvas: "ChannelPlotCanvas", channel_index: int):
        super().__init__(canvas)
        self._initialize_plot_presentation_lifecycle()
        self._canvas = canvas
        self.channel_index = int(channel_index)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = None
        self._set_presentation_mode(PLOT_PRESENTATION_COMPLETE)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(ui_style_const.waveform_frame_style)

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

    def set_title(self, title: str) -> None:
        self.title_bar.title_label.setText(str(title or ""))

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

    def _set_curve_data(self, x, y) -> None:
        self._release_deferred_view_guard()
        if self.plot_item is None:
            self.plot_item = self.plot_widget.plot(x, y, pen="k")
        else:
            self.plot_item.setData(x, y)


class ChannelPlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._windows: List[ChannelPlotSubWindow] = []
        # Make the workspace area visually distinct from the surrounding UI.
        self.setStyleSheet(ui_style_const.waveform_canvas_style)

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
        self._preserve_positions = True
        self._forced_cols = None
        self.scroll = QScrollArea(self)
        # Resizable is important so the canvas width tracks the viewport; otherwise during early init
        # viewport width may be 0/1 and child windows get fully clipped.
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        try:
            self.scroll.viewport().setStyleSheet(ui_style_const.waveform_viewport_style)
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

    def set_channels(self, channel_indices: Sequence[int]) -> None:
        channel_indices = [int(i) for i in (channel_indices or [])]

        if channel_indices == self._channel_indices and self._subwins:
            return

        for w in self._subwins:
            w._release_deferred_view_guard()
            try:
                w.hide()
                w.deleteLater()
            except Exception:
                pass
        self.canvas.set_windows([])

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

    def set_preserve_positions(self, preserve: bool) -> None:
        self._preserve_positions = bool(preserve)
        if self._subwins:
            QTimer.singleShot(0, lambda: self._tile_subwindows(keep_positions=self._preserve_positions))

    def set_forced_columns(self, cols: int | None) -> None:
        try:
            cols = int(cols) if cols is not None else None
        except Exception:
            cols = None
        self._forced_cols = cols if cols and cols > 0 else None
        if self._subwins:
            QTimer.singleShot(0, lambda: self._tile_subwindows(keep_positions=self._preserve_positions))

    def set_window_titles(self, titles: List[str]) -> None:
        for idx, w in enumerate(self._subwins):
            title = titles[idx] if idx < len(titles or []) else f"In{idx + 1}"
            w.set_title(title)

    def all_subwindows(self) -> List[ChannelPlotSubWindow]:
        return list(self._subwins)

    def subwindows(self) -> List[ChannelPlotSubWindow]:
        """Compatibility alias for older callers."""
        return self.all_subwindows()

    def clear_plots(self) -> None:
        for w in self._subwins:
            w.clear_plot()

    def _release_subwindow_plot_guards(self) -> None:
        for window in self._subwins:
            window._release_deferred_view_guard()

    def closeEvent(self, event) -> None:
        self._release_subwindow_plot_guards()
        super().closeEvent(event)

    def deleteLater(self) -> None:
        self._release_subwindow_plot_guards()
        super().deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep canvas width in sync with the viewport; directional waveform mode
        # disables free positioning so windows always reflow to fill the area.
        self._tile_subwindows(keep_positions=self._preserve_positions)
        if self._preserve_positions and not self._layout_is_valid():
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
        forced_cols = self._forced_cols
        if forced_cols:
            cols = max(1, min(int(forced_cols), int(num_windows) or 1))
            min_win_w = 160
        else:
            cols = 2 if usable_w >= (min_win_w * 2 + gap) else 1

        win_w = int((usable_w - gap * (cols - 1)) / cols)
        win_w = max(min_win_w, min(win_w, usable_w))

        usable_h = max(1, viewport_h - 2 * pad)
        rows = max(1, (int(num_windows) + cols - 1) // cols)

        win_h = int((usable_h - gap * (rows - 1)) / rows)
        win_h = max(220, min(win_h, usable_h))

        if int(num_windows) == 4 and viewport_h > 0:
            if rows == 2:
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
