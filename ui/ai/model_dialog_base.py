"""Shared project-themed widgets for model registry presentation only."""

from pathlib import PureWindowsPath

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const as theme


MODEL_DIALOG_STYLE = f"""
QDialog#aiModelDialog {{ background: {theme.COLOR_PAGE_BG}; }}
QDialog#aiModelDialog QLabel, QDialog#aiModelDialog QPushButton,
QDialog#aiModelDialog QTableWidget, QDialog#aiModelDialog QHeaderView {{
    font-family: {theme.UI_FONT_FAMILY}; font-size: 16px;
    color: {theme.COLOR_TEXT};
}}
QWidget#aiModelHeading {{
    background: {theme.COLOR_TITLE_BAR_BG};
    border-bottom: 1px solid {theme.COLOR_BORDER};
}}
QDialog#aiModelDialog QLabel#aiModelEyebrow {{ font-size: 12px; color: {theme.COLOR_TEXT_MUTED}; }}
QDialog#aiModelDialog QLabel#aiModelTitle {{ font-size: 23px; font-weight: 500; }}
QDialog#aiModelDialog QLabel#aiModelMuted {{ color: {theme.COLOR_TEXT_MUTED}; font-size: 14px; }}
QDialog#aiModelDialog QLabel#aiModelError {{ color: {theme.COLOR_NG}; font-size: 14px; }}
QWidget#aiModelToolbar, QWidget#aiModelFooter {{
    background: {theme.COLOR_TOOLBAR_BG};
}}
QWidget#aiModelToolbar {{ border-bottom: 1px solid {theme.COLOR_BORDER}; }}
QWidget#aiModelFooter {{ border-top: 1px solid {theme.COLOR_BORDER}; }}
QDialog#aiModelDialog QPushButton {{
    min-height: 28px; padding: 4px 12px;
    border: 1px solid {theme.COLOR_BORDER_STRONG}; border-radius: 4px;
    background: {theme.COLOR_TOOLBAR_BUTTON_BG};
}}
QDialog#aiModelDialog QPushButton:hover {{ background: {theme.COLOR_CONTROL_HOVER}; }}
QDialog#aiModelDialog QPushButton:disabled {{
    background: {theme.COLOR_DISABLED_BG}; color: {theme.COLOR_DISABLED_TEXT};
}}
QDialog#aiModelDialog QPushButton#aiModelPrimary {{
    background: {theme.COLOR_PRIMARY}; color: white; border-color: {theme.COLOR_PRIMARY};
}}
QDialog#aiModelDialog QPushButton#aiModelPrimary:hover {{
    background: {theme.COLOR_PRIMARY_HOVER};
}}
QDialog#aiModelDialog QPushButton#aiModelPrimary:disabled {{
    background: {theme.COLOR_DISABLED_BG}; color: {theme.COLOR_DISABLED_TEXT};
    border-color: {theme.COLOR_BORDER_STRONG};
}}
QDialog#aiModelDialog QPushButton#aiModelDetails {{
    border: none; background: transparent; color: {theme.COLOR_PRIMARY}; padding: 2px 8px;
}}
QDialog#aiModelDialog QPushButton#aiModelDetails:hover {{
    background: {theme.COLOR_CONTROL_HOVER};
}}
QDialog#aiModelDialog QTableWidget#aiModelTable {{
    background: {theme.COLOR_PANEL_BG}; border: 1px solid {theme.COLOR_BORDER};
    border-radius: 4px; selection-background-color: {theme.COLOR_CONTROL_PRESSED};
    selection-color: {theme.COLOR_TEXT};
}}
QDialog#aiModelDialog QTableWidget#aiModelTable::item {{ padding: 8px; border-bottom: 1px solid {theme.COLOR_BORDER}; }}
QDialog#aiModelDialog QTableWidget#aiModelTable::item:selected {{
    background: {theme.COLOR_CONTROL_PRESSED}; color: {theme.COLOR_TEXT};
}}
QDialog#aiModelDialog QTableWidget#aiModelTable QHeaderView::section {{
    background: {theme.COLOR_TOOLBAR_BG}; border: none;
    border-bottom: 1px solid {theme.COLOR_BORDER};
    color: {theme.COLOR_TEXT_MUTED}; padding: 10px 8px; font-weight: normal;
}}
QDialog#aiModelDialog QWidget#aiModelDetailsBody,
QDialog#aiModelDialog QScrollArea#aiModelDetailsScroll {{
    background: {theme.COLOR_PANEL_BG}; border: none;
}}
QDialog#aiModelDialog QLabel#aiModelMetric {{ font-size: 25px; }}
"""


MODEL_HEADER_STYLE = f"""
QHeaderView::section {{
    font-family: {theme.UI_FONT_FAMILY}; font-size: 14px; font-weight: normal;
    color: {theme.COLOR_TEXT_MUTED}; background: {theme.COLOR_TOOLBAR_BG};
    border: none; border-bottom: 1px solid {theme.COLOR_BORDER}; padding: 11px 12px;
}}
"""


class ModelNameDelegate(QStyledItemDelegate):
    """Separate the model name and filename while retaining native selection."""

    def paint(self, painter, option, index):
        view_option = QStyleOptionViewItem(option)
        self.initStyleOption(view_option, index)
        view_option.text = ""
        option.widget.style().drawControl(QStyle.CE_ItemViewItem, view_option, painter, option.widget)
        rect = option.rect.adjusted(12, 0, -12, 0)
        name_font = QFont(theme.UI_FONT_FAMILY_NAME)
        name_font.setPixelSize(16)
        file_font = QFont(name_font)
        file_font.setPixelSize(13)
        name_metrics = QFontMetrics(name_font)
        file_metrics = QFontMetrics(file_font)
        filename = index.data(Qt.UserRole)
        show_filename = index.data() != PureWindowsPath(filename).stem
        block_height = name_metrics.height()
        if show_filename:
            block_height += file_metrics.height() + 5
        top = rect.top() + (rect.height() - block_height) // 2
        painter.save()
        painter.setClipRect(option.rect)
        painter.setFont(name_font)
        painter.setPen(QColor(theme.COLOR_TEXT))
        painter.drawText(
            rect.left(), top + name_metrics.ascent(),
            name_metrics.elidedText(index.data(), Qt.ElideRight, rect.width()),
        )
        if show_filename:
            painter.setFont(file_font)
            painter.setPen(QColor(theme.COLOR_TEXT_MUTED))
            painter.drawText(
                rect.left(), top + name_metrics.height() + 5 + file_metrics.ascent(),
                file_metrics.elidedText(filename, Qt.ElideMiddle, rect.width()),
            )
        painter.restore()


def model_button(text, parent, *, primary=False):
    button = QPushButton(text, parent)
    button.setAutoDefault(False)
    if primary:
        button.setObjectName("aiModelPrimary")
    return button


def model_table(headers, parent):
    table = QTableWidget(0, len(headers), parent)
    table.setObjectName("aiModelTable")
    table.setFrameShape(QTableWidget.NoFrame)
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(64)
    header = table.horizontalHeader()
    header.setHighlightSections(False)
    header.setStyleSheet(MODEL_HEADER_STYLE)
    header.setFixedHeight(40)
    header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setMinimumSectionSize(72)
    return table


class AiModelDialogBase(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("aiModelDialog")
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        # Reuse project tokens without inheriting configuration-form sizing.
        self.setStyleSheet(MODEL_DIALOG_STYLE)
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

    def add_heading(self, title, subtitle=""):
        heading = QWidget(self)
        heading.setObjectName("aiModelHeading")
        layout = QHBoxLayout(heading)
        layout.setContentsMargins(24, 18, 24, 18)
        titles = QVBoxLayout()
        titles.setSpacing(5)
        if subtitle:
            eyebrow = QLabel(subtitle, heading)
            eyebrow.setObjectName("aiModelEyebrow")
            titles.addWidget(eyebrow)
        label = QLabel(title, heading)
        label.setObjectName("aiModelTitle")
        titles.addWidget(label)
        layout.addLayout(titles)
        layout.addStretch()
        self.root_layout.addWidget(heading)
        return layout

    def add_footer(self):
        footer = QWidget(self)
        footer.setObjectName("aiModelFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 10, 24, 10)
        self.root_layout.addWidget(footer)
        return layout
