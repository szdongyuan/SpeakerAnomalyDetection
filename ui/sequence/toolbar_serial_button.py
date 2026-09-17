"""Serial status presentation without changing the native button text/state."""

from PyQt5.QtCore import QRect, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter
from PyQt5.QtWidgets import QPushButton, QStyle, QStyleOptionButton, QStylePainter

from consts import ui_style_const


class ToolbarSerialButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.compact = False
        self._status_color = QColor(ui_style_const.COLOR_TEXT_MUTED)

    def set_compact(self, compact):
        if self.compact != compact:
            self.compact = compact
            self.setFixedWidth(48 if compact else 124)
            self.update()

    def set_connection_state(self, connected, has_response):
        if not connected:
            color = ui_style_const.COLOR_TEXT_MUTED
        else:
            color = ui_style_const.COLOR_OK if has_response else ui_style_const.COLOR_NG
        self._status_color = QColor(color)
        self.update()

    def _display_text(self):
        return "" if self.compact else self.text()

    def status_dot_rect(self):
        if not self.compact:
            return QRectF()
        return QRectF(self.width() - 8, (self.height() - 4) / 2, 4, 4)

    def compact_icon_rect(self):
        size = self.iconSize()
        return QRect((self.width() - size.width()) // 2 - 2,
                     (self.height() - size.height()) // 2,
                     size.width(), size.height())

    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = self._display_text()
        if self.compact:
            option.icon = QIcon()
        painter = QStylePainter(self)
        painter.drawControl(QStyle.CE_PushButton, option)
        if self.compact:
            mode = QIcon.Normal if self.isEnabled() else QIcon.Disabled
            self.icon().paint(painter, self.compact_icon_rect(), Qt.AlignCenter, mode)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._status_color)
            painter.drawEllipse(self.status_dot_rect())
