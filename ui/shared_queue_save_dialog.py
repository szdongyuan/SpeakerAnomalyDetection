"""Explicit confirmation for shared or incompletely checked queue writes."""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QMessageBox, QPlainTextEdit, QSizePolicy, QStyle


class SharedQueueSaveDialog(QMessageBox):
    def __init__(self, target_path, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("修改共享测试队列")
        self.setTextFormat(Qt.PlainText)
        name = os.path.splitext(os.path.basename(target_path))[0]
        introduction = f"测试队列“{name}”被以下工况共同使用："
        if result.issues:
            introduction += "\n无法完整检查队列引用，保存可能影响未列出的工况"
            self.setToolTip("\n".join(
                f"{issue.product_name} — {issue.path or '未保存草稿'}：{issue.message}"
                for issue in result.issues
            ))
        self.setText(introduction)
        self.setInformativeText("此队列的修改将同时影响以上所有工况。建议另存为后重新选择。")
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.save_button = self.button(QMessageBox.Ok)
        self.save_button.setText("确认")
        self.cancel_button = self.button(QMessageBox.Cancel)
        self.cancel_button.setText("取消")
        self.setDefaultButton(self.cancel_button)
        self.setEscapeButton(self.cancel_button)

        rows = list(dict.fromkeys(
            f"{names.product_name}/{names.group_name}/{names.condition_name}"
            for reference in result.references
            for names in reference.display_names
        ))
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setFrameShape(QPlainTextEdit.NoFrame)
        self.details.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.details.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if len(rows) > 4 else Qt.ScrollBarAlwaysOff
        )
        self.details.document().setDocumentMargin(0)
        self.details.setPlainText("\n".join(rows))
        self.details.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.details.viewport().setAutoFillBackground(False)
        self._visible_rows = min(len(rows), 4)
        if rows:
            # Insert between the native message and informative text; leave Qt's
            # message sizing and standard button layout in charge of the window.
            layout = self.layout()
            footer = self.findChild(QLabel, "qt_msgbox_informativelabel")
            row, column, row_span, column_span = layout.getItemPosition(layout.indexOf(footer))
            for index in reversed(range(layout.count())):
                item = layout.itemAt(index)
                r, c, rs, cs = layout.getItemPosition(index)
                if r >= row:
                    layout.addWidget(item.widget(), r + 1, c, rs, cs)
            layout.addWidget(self.details, row, column, row_span, column_span)
            self.details.horizontalScrollBar().rangeChanged.connect(self._size_details)
            self._size_details()
        else:
            self.details.hide()

    def _size_details(self):
        label = self.findChild(QLabel, "qt_msgbox_label")
        label.ensurePolished()
        self.details.setFont(label.font())
        margin = self.details.document().documentMargin()
        # QPlainTextEdit counts a row in its scroll page only when its bottom is
        # strictly inside the viewport, so retain one pixel beyond the last row.
        height = self._visible_rows * self.details.fontMetrics().lineSpacing() + 2 * margin + 1
        if self.details.horizontalScrollBar().maximum() > 0:
            height += self.details.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.details.setFixedHeight(int(height))
