"""Compact, external chart legend with a menu for entries that do not fit."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QMenu, QToolButton, QWidget

from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY, MAIN_UI_SMALL_FONT_FAMILY_NAME


class StationPlotLegend(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.entries = []
        self.align_right = False
        self._visible_count = 0
        font = QFont(MAIN_UI_SMALL_FONT_FAMILY_NAME)
        font.setPixelSize(12)
        self.setFont(font)
        self.setFixedHeight(24)
        self.more = QToolButton(self)
        self.more.setText('更多')
        self.more.setStyleSheet(f'QToolButton {{ font-family: {MAIN_UI_SMALL_FONT_FAMILY}; font-size: 12px; color: #64748B; }}')
        self.more.setPopupMode(QToolButton.InstantPopup)
        self.menu = QMenu(self.more)
        self.menu.setStyleSheet(f'QMenu {{ font-family: {MAIN_UI_SMALL_FONT_FAMILY}; font-size: 13px; }}')
        self.more.setMenu(self.menu)
        self.more.hide()

    def set_entries(self, entries):
        self.entries = entries
        self.menu.clear()
        for name, pen in entries:
            sample = QPixmap(28, 14)
            sample.fill(Qt.transparent)
            painter = QPainter(sample)
            painter.setPen(pen)
            painter.drawLine(2, 7, 26, 7)
            painter.end()
            self.menu.addAction(QIcon(sample), name)
        self._fit_entries()
        self.update()

    def _entry_width(self, name):
        return 32 + self.fontMetrics().horizontalAdvance(name) + 14

    def _fit_entries(self):
        widths = [self._entry_width(name) for name, _ in self.entries]
        overflow = sum(widths) > self.width() - 20
        available = self.width() - 20 - (50 if overflow else 0)
        used, self._visible_count = 0, 0
        for width in widths:
            if used + width > available:
                break
            used += width
            self._visible_count += 1
        self.more.setGeometry(max(0, self.width() - 52), 0, 48, 24)
        self.more.setVisible(overflow)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_entries()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(self.font())
        width = sum(self._entry_width(name) for name, _ in self.entries[:self._visible_count])
        x = max(10, self.width() - width - 10) if self.align_right and not self.more.isVisible() else 10
        for name, pen in self.entries[:self._visible_count]:
            painter.setPen(pen)
            painter.drawLine(x, 12, x + 23, 12)
            painter.setPen(Qt.darkGray)
            painter.drawText(x + 30, 16, name)
            x += self._entry_width(name)
        painter.end()
