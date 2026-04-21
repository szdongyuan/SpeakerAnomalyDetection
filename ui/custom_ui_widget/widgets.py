from PyQt5.QtGui import QFont
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
    QToolButton,
    QAction,
    QPlainTextEdit,
    QMessageBox,
    QTableWidget,
)

from consts.ui_style_const import scale_size_px


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


class CheckBox(QCheckBox):
    def __init__(self, *args):
        super(CheckBox, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class ComboBox(QComboBox):
    def __init__(self, parent=None):
        super(ComboBox, self).__init__(parent)
        self.font_size = scale_size_px(18)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class SpinBox(QSpinBox):
    def __init__(self, parent=None):
        super(SpinBox, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super(DoubleSpinBox, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class GroupBox(QGroupBox):
    def __init__(self, *args):
        super(GroupBox, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class PushButton(QPushButton):

    def __init__(self, *args):
        super(PushButton, self).__init__(*args)

        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


class LineEdit(QLineEdit):
    def __init__(self, *args):
        super(LineEdit, self).__init__(*args)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


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
            if standard_button == QMessageBox.Ok:
                button.setText("确认")
                button.setObjectName("okbtn")
            elif standard_button == QMessageBox.Cancel:
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
        self.setFont(self.font)

    def set_font_size(self, font_size):
        self.font_size = scale_size_px(font_size)
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)
