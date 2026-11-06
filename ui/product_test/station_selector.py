"""Fixed-width horizontally scrolling station selector."""
from PyQt5.QtCore import QEvent, QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QScrollArea, QPushButton, QLineEdit, QHBoxLayout, QStyle
class _PortSelectorBar(QWidget):
    BUTTON_WIDTH = 112
    BUTTON_HEIGHT = 32
    BUTTON_TEXT_MARGIN = 18

    currentChanged = pyqtSignal(int)
    scroll_requested = pyqtSignal(int)
    tabTextEdited = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = []
        self._current_index = -1
        self._editing_index = -1
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._name_editor = QLineEdit(self)
        self._name_editor.setObjectName("productProjectPortNameEditor")
        self._name_editor.setFixedSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        self._name_editor.editingFinished.connect(self._finish_inline_rename)
        self._name_editor.installEventFilter(self)
        self._name_editor.hide()

    def count(self):
        return len(self._buttons)

    def addTab(self, text):
        button = QPushButton(self)
        button.setObjectName("productProjectPortButton")
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setFixedSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        self._set_button_text(button, text)
        button.installEventFilter(self)
        button.clicked.connect(
            lambda _checked=False, target=button: self._select_button(target)
        )
        self._buttons.append(button)
        self._layout.addWidget(button)
        if self._current_index < 0:
            self.setCurrentIndex(0)
        return len(self._buttons) - 1

    def removeTab(self, index):
        if index < 0 or index >= len(self._buttons):
            return
        button = self._buttons.pop(index)
        self._layout.removeWidget(button)
        # A modal dialog can defer deletion until its event loop exits.
        button.hide()
        button.deleteLater()
        if not self._buttons:
            self.setCurrentIndex(-1)
        elif self._current_index >= len(self._buttons):
            self.setCurrentIndex(len(self._buttons) - 1)
        else:
            self._sync_checked_state()

    def currentIndex(self):
        return self._current_index

    def setCurrentIndex(self, index):
        target_index = index if 0 <= index < len(self._buttons) else -1
        changed = target_index != self._current_index
        self._current_index = target_index
        self._sync_checked_state()
        if changed:
            self.currentChanged.emit(target_index)

    def tabText(self, index):
        if 0 <= index < len(self._buttons):
            return str(self._buttons[index].property("fullText") or "")
        return ""

    def setTabText(self, index, text):
        if 0 <= index < len(self._buttons):
            self._set_button_text(self._buttons[index], text)

    def tabRect(self, index):
        if 0 <= index < len(self._buttons):
            return self._buttons[index].geometry()
        return self.rect()

    def _select_button(self, button):
        try:
            index = self._buttons.index(button)
        except ValueError:
            return
        self.setCurrentIndex(index)

    def _set_button_text(self, button, text):
        full_text = str(text or "")
        available_width = self.BUTTON_WIDTH - self.BUTTON_TEXT_MARGIN
        button.setProperty("fullText", full_text)
        button.setText(
            button.fontMetrics().elidedText(
                full_text,
                Qt.ElideRight,
                available_width,
            )
        )
        button.setToolTip(full_text)

    def eventFilter(self, watched, event):
        if (
            watched in self._buttons
            and event.type() == QEvent.MouseButtonDblClick
            and event.button() == Qt.LeftButton
        ):
            index = self._buttons.index(watched)
            self.setCurrentIndex(index)
            self._start_inline_rename(index)
            return True
        if (
            watched is self._name_editor
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Escape
        ):
            self._cancel_inline_rename()
            return True
        return super().eventFilter(watched, event)

    def _start_inline_rename(self, index):
        if index < 0 or index >= len(self._buttons):
            return
        button = self._buttons[index]
        self._editing_index = index
        self._name_editor.setGeometry(button.geometry())
        self._name_editor.setText(self.tabText(index))
        self._name_editor.show()
        self._name_editor.raise_()
        self._name_editor.setFocus()
        self._name_editor.selectAll()

    def _finish_inline_rename(self):
        index = self._editing_index
        if index < 0:
            return
        name = self._name_editor.text().strip() or f"新端口{index + 1}"
        self._editing_index = -1
        self._name_editor.hide()
        self.tabTextEdited.emit(index, name)

    def _cancel_inline_rename(self):
        self._editing_index = -1
        self._name_editor.hide()

    def _sync_checked_state(self):
        for index, button in enumerate(self._buttons):
            signals_blocked = button.blockSignals(True)
            button.setChecked(index == self._current_index)
            button.blockSignals(signals_blocked)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            event.ignore()
            return
        self.scroll_requested.emit(-delta)
        event.accept()

    def content_width(self):
        if not self._buttons:
            return 1
        button_widths = self.BUTTON_WIDTH * len(self._buttons)
        return button_widths + self._layout.spacing() * (len(self._buttons) - 1)

class _PortTabScrollArea(QScrollArea):
    TAB_HEIGHT = 40
    SCROLL_BAR_HEIGHT = 8

    def __init__(self, tab_bar, parent=None):
        super().__init__(parent)
        self._tab_bar = tab_bar
        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self.sync_content_width)
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(tab_bar)
        # QScrollArea enables background filling on its content; keep the tab
        # container transparent so it does not look like a second button frame.
        tab_bar.setAutoFillBackground(False)
        scroll_bar = self.horizontalScrollBar()
        scroll_bar.setFixedHeight(self.SCROLL_BAR_HEIGHT)
        scroll_bar.setStyleSheet(
            """
            QScrollBar:horizontal {
                height: 8px;
                margin: 0;
                border: none;
                border-radius: 4px;
                background: #EDF2F7;
            }
            QScrollBar::handle:horizontal {
                min-width: 36px;
                margin: 1px 0;
                border-radius: 3px;
                background: #B7C5D6;
            }
            QScrollBar::handle:horizontal:hover {
                background: #95A9C0;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                height: 0px;
                border: none;
                background: none;
            }
            QScrollBar::left-arrow:horizontal,
            QScrollBar::right-arrow:horizontal {
                width: 0px;
                height: 0px;
                border: none;
                image: none;
                background: none;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                border-radius: 4px;
                background: #EDF2F7;
            }
            """
        )
        self.setFixedHeight(self.TAB_HEIGHT)

    def sync_content_width(self):
        content_width = self._tab_bar.content_width()
        viewport_width = max(1, self.viewport().width())
        self._tab_bar.resize(max(1, content_width), self.TAB_HEIGHT)
        scroll_bar_height = (
            self.SCROLL_BAR_HEIGHT if content_width > viewport_width else 0
        )
        target_height = self.TAB_HEIGHT + scroll_bar_height
        if self.height() != target_height:
            self.setFixedHeight(target_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_timer.start(0)
