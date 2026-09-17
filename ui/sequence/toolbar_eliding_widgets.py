"""Toolbar-local presentation that never replaces the native widget's value."""

from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtGui import QPainter, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionFrame,
    QStylePainter,
    QToolTip,
)


class _FullTextPresentation:
    """Keep operational hints alongside the complete native text."""

    def __init__(self, *args, **kwargs):
        self._operation_tooltip = ""
        self._operation_description = ""
        self._explicit_accessible_name = False
        super().__init__(*args, **kwargs)
        self._sync_full_text()

    def _full_text(self):
        return self.text()

    def _sync_full_text(self, *_):
        full = self._full_text()
        tooltip = "\n".join(
            dict.fromkeys(filter(None, (self._operation_tooltip, full)))
        )
        description = "\n".join(
            dict.fromkeys(filter(None, (self._operation_description, tooltip)))
        )
        super().setToolTip(tooltip)
        super().setAccessibleDescription(description)
        if not self._explicit_accessible_name:
            super().setAccessibleName(full)

    def setToolTip(self, text):
        self._operation_tooltip = text
        self._sync_full_text()

    def setAccessibleDescription(self, text):
        self._operation_description = text
        self._sync_full_text()

    def setAccessibleName(self, text):
        self._explicit_accessible_name = True
        super().setAccessibleName(text)

    def event(self, event):
        # Business refreshes intentionally block Qt value-change signals. Read
        # the native value again when presenting it, including the spin editor.
        if event.type() in (QEvent.Show, QEvent.Paint, QEvent.ToolTip):
            self._sync_full_text()
        return super().event(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.FontChange, QEvent.StyleChange, QEvent.LayoutDirectionChange
        ):
            self.updateGeometry()
            self.update()


class ElidingLabel(_FullTextPresentation, QLabel):
    def setText(self, text):
        super().setText(text)
        self._sync_full_text()

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def _text_rect(self):
        rect = self.contentsRect()
        margin = self.margin()
        rect.adjust(margin, margin, -margin, -margin)
        indent = self.indent()
        if indent < 0:
            indent = (
                self.fontMetrics().horizontalAdvance("x") // 2
                if self.frameWidth() else 0
            )
        if self.alignment() & Qt.AlignLeft:
            rect.adjust(indent, 0, 0, 0)
        elif self.alignment() & Qt.AlignRight:
            rect.adjust(0, 0, -indent, 0)
        return rect

    def _display_text(self):
        return self.fontMetrics().elidedText(
            self.text(), Qt.ElideRight, max(0, self._text_rect().width())
        )

    def paintEvent(self, event):
        QFrame.paintEvent(self, event)
        painter = QPainter(self)
        rect = self._text_rect()
        painter.setClipRect(rect)
        flags = int(self.alignment()) | Qt.TextSingleLine
        if self.buddy() is not None:
            flags |= Qt.TextShowMnemonic
        self.style().drawItemText(
            painter, rect, flags, self.palette(), self.isEnabled(),
            self._display_text(), self.foregroundRole(),
        )


class ToolbarFieldLabel(ElidingLabel):
    """Paint a complete short caption while retaining the full native meaning."""

    def __init__(self, text, short_text, parent=None):
        self._short_text = short_text
        self._compact = False
        super().__init__(text, parent)

    def set_compact(self, compact):
        if self._compact != compact:
            self._compact = compact
            self.update()

    def caption_width(self, compact):
        text = self._short_text if compact else self.text()
        metrics = self.fontMetrics()
        return max(metrics.horizontalAdvance(text), metrics.boundingRect(text).width())

    def _display_text(self):
        return self._short_text if self._compact else self.text()


class ElidingLineEdit(_FullTextPresentation, QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textChanged.connect(self._sync_full_text)

    def minimumSizeHint(self):
        return QSize(55, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(160, super().sizeHint().height())

    def _text_rect(self):
        option = QStyleOptionFrame()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QStyle.SE_LineEditContents, option, self)
        margins = self.textMargins()
        # QLineEdit reserves two additional pixels at each horizontal edge.
        return rect.adjusted(
            margins.left() + 2, margins.top(),
            -margins.right() - 2, -margins.bottom(),
        )

    def _display_text(self):
        if self.hasFocus():
            return self.displayText()
        return self.fontMetrics().elidedText(
            self.displayText(), Qt.ElideRight, max(0, self._text_rect().width())
        )

    def paintEvent(self, event):
        display = self._display_text()
        if self.hasFocus() or display == self.displayText():
            super().paintEvent(event)
            return
        option = QStyleOptionFrame()
        self.initStyleOption(option)
        painter = QStylePainter(self)
        painter.drawPrimitive(QStyle.PE_PanelLineEdit, option)
        rect = self._text_rect()
        painter.setClipRect(rect)
        alignment = QStyle.visualAlignment(self.layoutDirection(), self.alignment())
        self.style().drawItemText(
            painter, rect, int(alignment | Qt.AlignVCenter | Qt.TextSingleLine),
            option.palette, self.isEnabled(), display, QPalette.Text,
        )


class _FullTextItemDelegate(QStyledItemDelegate):
    def helpEvent(self, event, view, option, index):
        if event.type() == QEvent.ToolTip and index.isValid():
            full = str(index.data(Qt.DisplayRole) or "")
            operation = str(index.data(Qt.ToolTipRole) or "")
            text = "\n".join(dict.fromkeys(filter(None, (operation, full))))
            QToolTip.showText(event.globalPos(), text, view)
            return True
        return super().helpEvent(event, view, option, index)


class ElidingComboBox(_FullTextPresentation, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(_FullTextItemDelegate(self))
        self.currentTextChanged.connect(self._sync_full_text)

    def _full_text(self):
        return self.currentText()

    def minimumSizeHint(self):
        return QSize(55, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(200, super().sizeHint().height())

    def _display_text(self):
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        rect = self.style().subControlRect(
            QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self
        )
        width = rect.width() - 2
        if not option.currentIcon.isNull():
            width -= option.iconSize.width() + 4
        return self.fontMetrics().elidedText(
            self.currentText(), Qt.ElideRight, max(0, width)
        )

    def paintEvent(self, event):
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self._display_text()
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)


class ElidingSpinBox(_FullTextPresentation, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # QAbstractSpinBox installs its own validator on this editor.
        self.setLineEdit(ElidingLineEdit(self))
        self.lineEdit().textChanged.connect(self._sync_full_text)
        self.textChanged.connect(self._sync_full_text)
        self.textChanged.connect(self.lineEdit()._sync_full_text)
        self._sync_full_text()

    def setToolTip(self, text):
        super().setToolTip(text)
        # Hover over the editor is handled by the child, not the spinbox frame.
        # Share the operation hint; each widget still reads its native full text.
        self.lineEdit().setToolTip(text)

    def minimumSizeHint(self):
        return QSize(55, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(60, super().sizeHint().height())


class ElidingCheckBox(_FullTextPresentation, QCheckBox):
    def setText(self, text):
        super().setText(text)
        self._sync_full_text()

    def minimumSizeHint(self):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        width = self.style().pixelMetric(QStyle.PM_IndicatorWidth, option, self)
        width += self.style().pixelMetric(QStyle.PM_CheckBoxLabelSpacing, option, self)
        return QSize(width, super().minimumSizeHint().height())

    def _display_text(self):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QStyle.SE_CheckBoxContents, option, self)
        return self.fontMetrics().elidedText(
            self.text(), Qt.ElideRight, max(0, rect.width()), Qt.TextShowMnemonic
        )

    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = self._display_text()
        painter = QStylePainter(self)
        painter.drawControl(QStyle.CE_CheckBox, option)

    def hitButton(self, position):
        return self.rect().contains(position)
