"""Single-line, plain-text corner labels outside the plot's coordinate scene."""

from html import escape

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolTip, QWidget


class _CornerLabel(QLabel):
    def __init__(self, parent, *, alignment, color, pixel_size=13):
        super().__init__(parent)
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(alignment | Qt.AlignVCenter)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        font = self.font()
        font.setPixelSize(pixel_size)
        self.setFont(font)
        self.setStyleSheet(f'QLabel {{ color: {color}; background: transparent; '
                           f'border: 0; padding: 0; font-size: {pixel_size}px; }}')

    def set_full_text(self, text):
        self.setToolTip(text)
        self.setVisible(bool(text))
        self._elide()

    def _elide(self):
        # Preserve the submission in the tooltip, but keep the visible bar to
        # one line even when the caller supplies line breaks or tabs.
        text = ' '.join(self.toolTip().splitlines()).replace('\t', ' ')
        self.setText(self.fontMetrics().elidedText(
            text, Qt.ElideRight, self.contentsRect().width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def event(self, event):
        if event.type() == QEvent.ToolTip and self.toolTip():
            # Qt auto-detects rich text for tooltips independently of QLabel's
            # PlainText setting. Escape at the popup boundary, keeping raw text
            # in toolTip() for elision and caller inspection.
            QToolTip.showText(event.globalPos(),
                              '<qt><p style="white-space: pre-wrap;">'
                              + escape(self.toolTip()) + '</p></qt>', self)
            return True
        return super().event(event)


class StationPlotHeader(QWidget):
    """Render already-validated text; hidden sides consume no layout space."""

    def __init__(self, parent):
        super().__init__(parent)
        self.left = _CornerLabel(self, alignment=Qt.AlignLeft, color='#183957')
        self.right = _CornerLabel(self, alignment=Qt.AlignRight, color='#5d7691')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(8)
        layout.addWidget(self.left, 2)
        layout.addWidget(self.right, 1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.set_texts('', '')

    def set_texts(self, left_text, right_text):
        self.left.set_full_text(left_text)
        self.right.set_full_text(right_text)
        self.setVisible(bool(left_text or right_text))


class StationPlotSubheader(QWidget):
    """Three fixed slots aligned to the axis margin and actual ViewBox halves."""

    resized = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.left = _CornerLabel(self, alignment=Qt.AlignLeft, color='#5d7691', pixel_size=11)
        self.first_half = _CornerLabel(self, alignment=Qt.AlignHCenter, color='#5d7691', pixel_size=11)
        self.second_half = _CornerLabel(self, alignment=Qt.AlignHCenter, color='#5d7691', pixel_size=11)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setFixedHeight(self.left.fontMetrics().height() + 6)
        self.set_texts('', '', '')

    def set_texts(self, left_text, first_half_text, second_half_text):
        for label, text in zip((self.left, self.first_half, self.second_half),
                               (left_text, first_half_text, second_half_text)):
            label.set_full_text(text)
        self.setVisible(bool(left_text or first_half_text or second_half_text))

    def set_plot_span(self, left, right):
        # Round shared boundaries once. Padding leaves adjacent slots disjoint
        # without changing their centers when another label is empty.
        left, right = round(left), round(right)
        middle = round((left + right) / 2)
        for label, start, end in ((self.left, 8, left - 4),
                                  (self.first_half, left + 4, middle - 4),
                                  (self.second_half, middle + 4, right - 4)):
            label.setGeometry(start, 3, max(0, end - start), self.height() - 6)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
