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


class TableView(QTableView):
    def __init__(self, parent=None):
        super(TableView, self).__init__(parent)
        self.font_size = scale_size_px(20)
        self.font = QFont()
        self.font.setFamily("SimSun")
        self.font.setPixelSize(self.font_size)
        self.setFont(self.font)


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
