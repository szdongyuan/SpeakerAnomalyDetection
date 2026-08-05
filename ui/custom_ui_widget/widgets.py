from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QLabel,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QMenuBar,
    QMenu,
    QListView,
    QRadioButton,
    QTabWidget,
    QTextEdit,
    QTableView,
    QTreeView,
    QTreeWidget,
    QToolButton,
    QAction,
    QPlainTextEdit,
    QMessageBox,
    QTableWidget,
)

from consts.ui_style_const import (
    CUSTOM_WIDGET_BUTTON_VERTICAL_PADDING_PX,
    CUSTOM_WIDGET_CONTROL_VERTICAL_PADDING_PX,
    scale_size_px,
)


def _build_font(pixel_size):
    font = QFont()
    font.setFamily("SimSun")
    font.setPixelSize(scale_size_px(pixel_size))
    return font


def _font_safe_height(widget, vertical_padding_px=CUSTOM_WIDGET_CONTROL_VERTICAL_PADDING_PX):
    try:
        metrics = QFontMetrics(widget.font())
        metric_height = int(metrics.height())
    except Exception:
        metric_height = 0
    fallback_height = scale_size_px(28)
    return max(fallback_height, metric_height + scale_size_px(vertical_padding_px))


def _apply_font(widget, pixel_size):
    widget.font_size = scale_size_px(pixel_size)
    widget._font = _build_font(pixel_size)
    widget.setFont(widget._font)
    if hasattr(widget, "_safe_height"):
        widget.setMinimumHeight(widget._safe_height())


class _FontSafeHeightMixin:
    _vertical_padding_px = CUSTOM_WIDGET_CONTROL_VERTICAL_PADDING_PX

    def _safe_height(self):
        return _font_safe_height(self, self._vertical_padding_px)

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(hint.width(), max(hint.height(), self._safe_height()))

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QSize(hint.width(), max(hint.height(), self._safe_height()))


class Label(QLabel):
    def __init__(self, *args):
        super(Label, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)

    def set_font_size(self, font_size):
        self.font_size = scale_size_px(20)
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class CheckBox(_FontSafeHeightMixin, QCheckBox):
    def __init__(self, *args):
        super(CheckBox, self).__init__(*args)
        _apply_font(self, 20)


class ComboBox(_FontSafeHeightMixin, QComboBox):
    def __init__(self, parent=None):
        super(ComboBox, self).__init__(parent)
        _apply_font(self, 18)


class SpinBox(_FontSafeHeightMixin, QSpinBox):
    def __init__(self, parent=None):
        super(SpinBox, self).__init__(parent)
        _apply_font(self, 20)


class DoubleSpinBox(_FontSafeHeightMixin, QDoubleSpinBox):
    def __init__(self, parent=None):
        super(DoubleSpinBox, self).__init__(parent)
        _apply_font(self, 20)


class GroupBox(QGroupBox):
    def __init__(self, *args):
        super(GroupBox, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class PushButton(_FontSafeHeightMixin, QPushButton):
    _vertical_padding_px = CUSTOM_WIDGET_BUTTON_VERTICAL_PADDING_PX

    def __init__(self, *args):
        super(PushButton, self).__init__(*args)

        _apply_font(self, 20)


class LineEdit(_FontSafeHeightMixin, QLineEdit):
    def __init__(self, *args):
        super(LineEdit, self).__init__(*args)
        _apply_font(self, 20)


class MarkPushButton(QPushButton):
    def __init__(self, *args):
        super(MarkPushButton, self).__init__(*args)
        self.font_size = scale_size_px(70)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class MenuBar(QMenuBar):
    def __init__(self, parent=None):
        super(MenuBar, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class Menu(QMenu):
    def __init__(self, *args):
        super(Menu, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class ListView(QListView):
    def __init__(self, parent=None):
        super(ListView, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class RadioButton(QRadioButton):
    def __init__(self, *args):
        super(RadioButton, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class TabWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class TextEdit(QTextEdit):
    def __init__(self, *args):
        super(TextEdit, self).__init__(*args)
        self.font_size = scale_size_px(30)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)

    def set_font_size(self, font_size):
        self.font_size = scale_size_px(font_size)
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class TableView(QTableView):
    def __init__(self, parent=None):
        super(TableView, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)
        self.horizontalHeader().setFont(self.font)
        self.verticalHeader().setFont(self.font)


class TreeView(QTreeView):
    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class TreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super(TreeWidget, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)

    def set_font_size(self, font_size):
        self.font_size = scale_size_px(font_size)
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class ToolButton(QToolButton):
    def __init__(self, parent=None):
        super(ToolButton, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class Action(QAction):
    def __init__(self, *args):
        super(Action, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class PlainTextEdit(QPlainTextEdit):
    def __init__(self, *args):
        super(PlainTextEdit, self).__init__(*args)
        self.font_size = scale_size_px(12)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class MessageBox(QMessageBox):
    def __init__(self, *args):
        super(MessageBox, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)

    def _sync_buttons_style_and_text(self):
        # If caller does not configure any button, keep a single confirm button by default.
        if not self.buttons():
            self.setStandardButtons(QMessageBox.Ok)

        for button in self.buttons():
            button.setFont(self.font)
            standard_button = self.standardButton(button)
            if standard_button in (QMessageBox.Ok, QMessageBox.Yes):
                button.setText("确认")
                button.setObjectName("okbtn")
            elif standard_button in (QMessageBox.Cancel, QMessageBox.No):
                button.setText("取消")
                button.setObjectName("cancelbtn")

    @classmethod
    def _show_static_message(
        cls, icon, parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton
    ):
        """
        Keep QMessageBox static-call style, but route through MessageBox instance so
        custom font settings still apply.
        """
        msg_box = cls(parent)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStandardButtons(buttons)
        if defaultButton != QMessageBox.NoButton:
            msg_box.setDefaultButton(defaultButton)
        return msg_box.exec_()

    def exec_(self):
        self._sync_buttons_style_and_text()
        return super().exec_()

    def open(self):
        self._sync_buttons_style_and_text()
        return super().open()

    @classmethod
    def warning(cls, parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        return cls._show_static_message(QMessageBox.Warning, parent, title, text, buttons, defaultButton)

    @classmethod
    def information(cls, parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        return cls._show_static_message(QMessageBox.Information, parent, title, text, buttons, defaultButton)

    @classmethod
    def critical(cls, parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        return cls._show_static_message(QMessageBox.Critical, parent, title, text, buttons, defaultButton)

    @classmethod
    def question(
        cls,
        parent,
        title,
        text,
        buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No),
        defaultButton=QMessageBox.NoButton,
    ):
        return cls._show_static_message(QMessageBox.Question, parent, title, text, buttons, defaultButton)


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self._apply_font()

    def _apply_font(self):
        self.setFont(self.font)
        self.horizontalHeader().setFont(self.font)
        self.verticalHeader().setFont(self.font)

    def set_font_size(self, font_size):
        self.font_size = scale_size_px(font_size)
        self.font.setPixelSize(self.font_size)
        self._apply_font()
