"""Complete acquisition parameter comparison shown when a project cannot save."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialogButtonBox, QFrame, QLabel, QListWidget, QMessageBox,
    QStyle, QVBoxLayout, QWidget,
)

from ui.dialog_enter_policy import install_dialog_enter_policy


class ProductQueueConflictDialog(QMessageBox):
    def __init__(self, summary, rows, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setIcon(QMessageBox.Warning)
        self.setTextFormat(Qt.PlainText)
        self.setText(summary)
        self.setStandardButtons(QMessageBox.Ok)
        confirm = self.button(QMessageBox.Ok)
        confirm.setText("确定")
        self.setDefaultButton(confirm)
        self.setEscapeButton(confirm)
        self.prompt_label = self.findChild(QLabel, "qt_msgbox_label")
        self.prompt_label.setWordWrap(True)
        self.buttons = self.findChild(QDialogButtonBox)

        self.conflict_list = QListWidget(self)
        self.conflict_list.setFrameShape(QFrame.NoFrame)
        self.conflict_list.setStyleSheet("QListWidget { background-color: transparent; border: none; }")
        self.conflict_list.viewport().setAutoFillBackground(False)
        self.conflict_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.conflict_list.setTextElideMode(Qt.ElideNone)
        self.conflict_list.setWordWrap(False)
        self.conflict_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.conflict_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.conflict_list.addItems(rows)

        self.ensurePolished()
        self.conflict_list.setFont(self.prompt_label.font())
        indent = max(4, self.prompt_label.fontMetrics().horizontalAdvance(" ") * 2)
        middle = QWidget(self)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(indent, 0, 0, 0)
        middle_layout.addWidget(self.conflict_list)

        # Insert only before the standard button box; leave the native icon,
        # text and spacers in their original grid positions.
        layout = self.layout()
        row, column, row_span, column_span = layout.getItemPosition(layout.indexOf(self.buttons))
        _, text_column, _, text_span = layout.getItemPosition(layout.indexOf(self.prompt_label))
        layout.removeWidget(self.buttons)
        layout.addWidget(middle, row, text_column, 1, text_span)
        layout.addWidget(self.buttons, row + 1, column, row_span, column_span)

        margins = layout.contentsMargins()
        available_width = self.screen().availableGeometry().width()
        icon_width = self.iconPixmap().width()
        width_budget = available_width - margins.left() - margins.right() - icon_width - indent
        width_budget -= max(0, layout.horizontalSpacing()) + 32
        scrollbar_width = self.conflict_list.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        content_width = self.conflict_list.sizeHintForColumn(0)
        if self.conflict_list.count() > 8:
            content_width += scrollbar_width
        self.conflict_list.setFixedWidth(max(1, min(content_width + 2, width_budget)))
        self.conflict_list.horizontalScrollBar().rangeChanged.connect(self._size_list)
        self._size_list()
        install_dialog_enter_policy(self, confirm)

    def _size_list(self):
        row_height = max(self.conflict_list.sizeHintForRow(0), self.conflict_list.fontMetrics().height())
        height = row_height * min(self.conflict_list.count(), 8) + 1
        if self.conflict_list.horizontalScrollBar().maximum() > 0:
            height += self.conflict_list.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.conflict_list.setFixedHeight(height)
