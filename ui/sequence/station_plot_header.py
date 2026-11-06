"""Single-line, plain-text corner labels outside the plot's coordinate scene."""

from html import escape

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolTip, QWidget

from consts.ui_style_const import (
    MAIN_UI_SMALL_FONT_FAMILY, MAIN_UI_SMALL_FONT_FAMILY_NAME, main_window_tooltip_style,
)


class _CornerLabel(QLabel):
    def __init__(self, parent, *, alignment, color, pixel_size=13):
        super().__init__(parent)
        self._full_text = ''
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(alignment | Qt.AlignVCenter)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        font = self.font()
        font.setFamily(MAIN_UI_SMALL_FONT_FAMILY_NAME)
        font.setPixelSize(pixel_size)
        self.setFont(font)
        self.setStyleSheet(
            f'QLabel {{ color: {color}; background: transparent; border: 0; padding: 0; '
            f'font-family: {MAIN_UI_SMALL_FONT_FAMILY}; font-size: {pixel_size}px; }}'
            + main_window_tooltip_style
        )

    def set_full_text(self, text, *, detail=''):
        self._full_text = text
        self.setToolTip(detail or text)
        self.setVisible(bool(text))
        self._elide()

    def _single_line_text(self):
        # Keep the visible bar to one line even if the text contains line breaks or tabs.
        return ' '.join(self._full_text.splitlines()).replace('\t', ' ')

    def full_text_width(self):
        return self.fontMetrics().horizontalAdvance(self._single_line_text())

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(
            self._single_line_text(), Qt.ElideRight, self.contentsRect().width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def event(self, event):
        if event.type() == QEvent.ToolTip and self.toolTip():
            # Qt auto-detects rich text for tooltips independently of QLabel's
            # PlainText setting. Escape at the popup boundary, keeping raw text
            # in toolTip() for caller inspection.
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

    def set_texts(self, left_text, right_text, *, right_text_detail=''):
        self.left.set_full_text(left_text)
        self.right.set_full_text(right_text, detail=right_text_detail)
        self.setVisible(bool(left_text or right_text))


class StationPlotSubheader(QWidget):
    """Three labels aligned to the axis margin and actual ViewBox halves."""

    resized = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.left = _CornerLabel(self, alignment=Qt.AlignLeft, color='#5d7691', pixel_size=11)
        self.first_half = _CornerLabel(self, alignment=Qt.AlignHCenter, color='#5d7691', pixel_size=11)
        self.second_half = _CornerLabel(self, alignment=Qt.AlignHCenter, color='#5d7691', pixel_size=11)
        self._plot_span = (0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setFixedHeight(self.left.fontMetrics().height() + 6)
        self.set_texts('', '', '')

    def set_texts(self, left_text, first_half_text, second_half_text):
        for label, text in zip((self.left, self.first_half, self.second_half),
                               (left_text, first_half_text, second_half_text)):
            label.set_full_text(text)
        self.setVisible(bool(left_text or first_half_text or second_half_text))
        self._update_label_geometry()

    def set_plot_span(self, left, right):
        self._plot_span = round(left), round(right)
        self._update_label_geometry()

    def _update_label_geometry(self):
        left, right = self._plot_span
        middle = round((left + right) / 2)
        unit_start, unit_end = 8, left - 4
        first_start, first_end = left + 4, middle - 4
        first_width = max(0, first_end - first_start)
        reserve = min(first_width, 2 * self.first_half.fontMetrics().height())
        shift = unit_start + self.left.full_text_width() - unit_end
        if shift > 0 and first_width - 2 * shift >= reserve:
            # Keep the gap and the first label's center by narrowing both sides.
            unit_end += shift
            first_start += shift
            first_end -= shift
        for label, start, end in ((self.left, unit_start, unit_end),
                                  (self.first_half, first_start, first_end),
                                  (self.second_half, middle + 4, right - 4)):
            label.setGeometry(start, 3, max(0, end - start), self.height() - 6)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
