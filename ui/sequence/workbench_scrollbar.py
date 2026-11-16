"""Shared workbench horizontal appearance with native Qt scrolling."""

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QPixmap, QTransform
from PyQt5.QtWidgets import QScrollBar, QStyle, QStyleOptionSlider

from consts.ui_style_const import MOTOR_WORKBENCH_SCROLLBAR_STYLE


class WorkbenchHorizontalScrollBar(QScrollBar):
    """Paint Qt's triangle PNGs over the shared horizontal scrollbar style."""

    def __init__(self, parent, *, object_name):
        super().__init__(Qt.Horizontal, parent)
        self.setObjectName(object_name)
        self.setStyleSheet(MOTOR_WORKBENCH_SCROLLBAR_STYLE)
        arrow = QPixmap(':/qt-project.org/styles/commonstyle/images/titlebar-shade-16.png')
        self._left_arrow = arrow.transformed(QTransform().rotate(-90))
        self._right_arrow = arrow.transformed(QTransform().rotate(90))

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        with QPainter(self) as painter:
            for control, arrow in ((QStyle.SC_ScrollBarSubLine, self._left_arrow),
                                   (QStyle.SC_ScrollBarAddLine, self._right_arrow)):
                button = self.style().subControlRect(QStyle.CC_ScrollBar, option, control, self)
                target = QRect(0, 0, 12, 14)
                target.moveCenter(button.center())
                painter.drawPixmap(target, arrow)
