import re

from PyQt5.QtWidgets import QApplication


_BASE_SCREEN_HEIGHT = 1080
_MIN_FONT_SCALE = 0.70
_MAX_FONT_SCALE = 1.30


def _get_font_scale():
    app = QApplication.instance()
    if app is None:
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    size = screen.size()
    if size.width() <= 0 or size.height() <= 0:
        return 1.0
    scale = size.height() / float(_BASE_SCREEN_HEIGHT)
    return max(_MIN_FONT_SCALE, min(scale, _MAX_FONT_SCALE))


_FONT_SCALE = _get_font_scale()
_FONT_SIZE_PATTERN = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(\s*)(pt|px)(\s*;)", re.IGNORECASE)


def _format_size(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _scale_font_sizes(style):
    def _replace(match):
        raw_size = float(match.group(2))
        scaled_size = max(1, int(round(raw_size * _FONT_SCALE)))
        return f"{match.group(1)}{_format_size(scaled_size)}{match.group(3)}{match.group(4)}{match.group(5)}"

    return _FONT_SIZE_PATTERN.sub(_replace, style)


def scale_font_px(size_px):
    return max(1, int(round(size_px * _FONT_SCALE)))

qtoolbutton_style = """
            QToolButton {
                border: None;
                background-color: transparent;
                font-family: 'SimSun';
                font-size: 20px;
            }
            QToolButton[hot="true"] {
                background-color: #5099ccff;
            }
        """

qpushbutton_style = """
            QPushButton {
                border-left: 1px solid rgb(128, 128, 128);
                border-top: 1px solid rgb(128, 128, 128);
                border-right: 3px solid rgb(128, 128, 128);
                border-bottom: 3px solid rgb(128, 128, 128);
                background-color: rgb(225, 225, 225);
                font-family: 'SimSun';
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #5099ccff;
                color: black;
                border-color: #803333ff;
            }
            QPushButton:pressed {
                background-color: #8099ccff;
                border-color: #3333ff;
            }
        """
qlineedit_style = """
            QLineEdit {
                border: 1px solid rgb(122, 122, 122);
                font-family: 'SimSun';
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
            QLineEdit:disabled {
                background-color: #d3d3d3;
                color: #808080;
            }
        """
qcombobox_style = """
            QComboBox {
                border: 1px solid rgb(173, 173, 173);
                background-color: rgb(225, 225, 225);
                font-family: 'SimSun';
                font-size: 18px;
                border-radius: 3px;
                padding: 3px;
            }
            QComboBox QAbstractItemView {
                font-family: 'SimSun';
                font-size: 18px;
            }
            QComboBox QAbstractItemView::item {
                font-family: 'SimSun';
                font-size: 18px;
            }
        """
qgroupbox_style = """
            QGroupBox {
                background: transparent;
                font-size: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                padding-left: 0px;
                padding-top: 0px;
                margin-top: 0px;
            }
        """
qspinbox_style = """
            QSpinBox {
                background-color: rgb(225, 225, 225);
                border: 1px solid rgb(173, 173, 173);
                font-family: 'SimSun';
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
            QSpinBox::up-button {
                width: 20px;
            }
            QSpinBox::down-button {
                width: 20px;
            }
        """
qdoublespinbox_style = """
            QDoubleSpinBox {
                background-color: rgb(225, 225, 225);
                border: 1px solid rgb(173, 173, 173);
                font-family: 'SimSun';
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
            QDoubleSpinBox::up-button {
                width: 20px;
            }
            QDoubleSpinBox::down-button {
                width: 20px;
            }
        """

qdialog_style = """
            QDialog {
                border-radius: 20px;
            }
"""

qlabel_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 20px;
                color: black;
            }
"""

sequence_qlabel_style = """
            QLabel {
                background-color: #4472c4;
                color: white;
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

sequence_qpushbutton_style = """
            QPushButton {
                background-color: #d0ceca;
                color: black;
                font-family: 'SimSun';
                font-size: 70px;
                border-radius: 20px;
                border: 3px solid rgb(128, 128, 128);
                padding-left: 0px;
            }
            QPushButton:hover {
                border-color: #0080ff;
            }
            QPushButton:pressed {
                background-color: #aba8a0;
                border-color: #3333ff;
            }

"""

main_window_menubar_style = """
            QMenuBar {
                background-color:transparent;
                font-family: 'SimSun';
                font-size: 20px;
            }
            QMenu {
                font-family: 'SimSun';
                font-size: 20px;
            }
            QMenu::item {
                font-family: 'SimSun';
                font-size: 20px;
                padding-left: 30px;
                padding-right: 10px;
            }
            QMenu::item:selected {
                background-color: #8099ccff;
            }
"""

qframe_style = """
            QFrame {
                color: rgb(173, 173, 173);
            }
"""

hardware_qframe_style = """
            QFrame {
                color: rgb(204, 204, 204);
            }
"""

toolbar_button_style = """
            QPushButton {
                border: None;
                background-color: transparent;
                font-family: 'SimSun';
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #5099ccff;
            }
"""

qcheckbox_style = """
            QCheckBox {
                font-family: 'SimSun';
                font-size: 20px;
            }
            QCheckBox:focus { outline: none; }
"""

qlistview_style = """
            QListView {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

qmainwindow_style = """
            QMainWindow  {
                background-color: #f0f0f0;
                border: 1px solid rgb(0, 0, 0);
            }
"""

qradiobutton_style = """
            QRadioButton {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

qtabwidget_style = """
            QTabWidget {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""
qtextedit_style = """
            QTextEdit{
                background-color: white;
                font-family: 'SimSun';
                font-size: 30px;
            }
             QTextEdit:disabled {
                color: rgb(0, 0, 0);
            }
"""

qtableview_style = """
            QTableView {
                font-family: 'SimSun';
                font-size: 20px;
            }
            QHeaderView::section {
                font-family: 'SimSun';
                font-size: 20px;
            }
            QTableView::item {
                border-top: 1px solid rgb(130, 135, 144);
                color: black;
            }
            QTableView::item:selected {
                background-color: rgb(0, 120, 215);
            }
"""

qtreeview_style = """
            QTreeView {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""


_STYLE_NAMES = (
    "qpushbutton_style",
    "qlineedit_style",
    "qcombobox_style",
    "qgroupbox_style",
    "qspinbox_style",
    "qdoublespinbox_style",
    "qdialog_style",
    "qlabel_style",
    "sequence_qlabel_style",
    "sequence_qpushbutton_style",
    "main_window_menubar_style",
    "qframe_style",
    "hardware_qframe_style",
    "toolbar_button_style",
    "qcheckbox_style",
    "qlistview_style",
    "qmainwindow_style",
    "qradiobutton_style",
    "qtabwidget_style",
    "qtextedit_style",
    "qtableview_style",
    "qtreeview_style",
)

for _style_name in _STYLE_NAMES:
    globals()[_style_name] = _scale_font_sizes(globals()[_style_name])
