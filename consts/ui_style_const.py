# ── Instrument Panel Design System ────────────────────────────────────────────
#
#   Philosophy: strong contrast, clear structure, zero ambiguity.
#   A light-based industrial palette — readable on a bright factory floor,
#   with enough visual weight to feel authoritative.
#
#   Background:  #D6E8F5  (medium-light steel blue — not white, has real body)
#   Surface:     #EEF6FF  (light — dialogs, inputs, tables)
#   Panel:       #C4D8EC  (toolbar / side-panel — slightly darker for structure)
#   Input bg:    #F4F9FF
#   Title bar:   QColor(12, 48, 92)  — deep navy, painted in paintEvent
#
#   Accent:      #0A66B8  (deep, saturated blue — more authority than sky blue)
#   Acc hover:   #0857A0
#   Acc pressed: #064884
#   Acc tint:    #AACCEE  (visible fill for hover backgrounds)
#
#   Text:        #081828  (near-black navy — maximum contrast)
#   Text sec:    #2A4A68  (muted dark blue)
#   Text dis:    #7A9AB4  (clearly disabled)
#
#   Border:      #7AAAC8  (strong — structure is visible at a glance)
#   Brd focus:   #0A66B8
#   Dis bg:      #D0E4F4
#
#   Success:     #088A38  (vivid green – OK)
#   Danger:      #C41826  (deep red – NG, error)
#
#   Font:        'Microsoft YaHei', 'SimSun'
#   Base size:   15px
# ──────────────────────────────────────────────────────────────────────────────

qpushbutton_style = """
    QPushButton {
        border: 1px solid #7AAAC8;
        background-color: #EEF6FF;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        border-radius: 3px;
        padding: 3px 10px;
        min-height: 26px;
    }
    QPushButton:hover {
        background-color: #AACCEE;
        border-color: #0A66B8;
        color: #064884;
    }
    QPushButton:pressed {
        background-color: #88BBDD;
        border-color: #0857A0;
        color: #042E60;
    }
    QPushButton:disabled {
        background-color: #D0E4F4;
        border-color: #A8C4D8;
        color: #7A9AB4;
    }
"""

qlineedit_style = """
    QLineEdit {
        border: 1px solid #7AAAC8;
        background-color: #F4F9FF;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        border-radius: 3px;
        padding: 3px 7px;
        selection-background-color: #88BBDD;
        selection-color: #042E60;
    }
    QLineEdit:focus {
        border-color: #0A66B8;
        background-color: #EEF6FF;
    }
    QLineEdit:disabled {
        background-color: #D8E8F4;
        border-color: #A8C4D8;
        color: #5A7A94;
    }
"""

qcombobox_style = """
    QComboBox {
        border: 1px solid #7AAAC8;
        background-color: #F4F9FF;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        border-radius: 3px;
        padding: 2px 7px;
        min-height: 26px;
    }
    QComboBox:hover {
        border-color: #0A66B8;
        background-color: #EEF6FF;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox QAbstractItemView {
        border: 1px solid #7AAAC8;
        background-color: #EEF6FF;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        outline: none;
        selection-background-color: #AACCEE;
        selection-color: #064884;
    }
    QComboBox QAbstractItemView::item {
        padding: 4px 8px;
        min-height: 24px;
    }
"""

qgroupbox_style = """
    QGroupBox {
        background: transparent;
        border: 1px solid #7AAAC8;
        border-radius: 4px;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 14px;
        color: #0A66B8;
        margin-top: 12px;
        padding-top: 4px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 4px;
        color: #0A66B8;
        background-color: transparent;
    }
"""

qspinbox_style = """
    QSpinBox {
        background-color: #F4F9FF;
        border: 1px solid #7AAAC8;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        border-radius: 3px;
        padding: 2px 5px;
        min-height: 26px;
    }
    QSpinBox:focus { border-color: #0A66B8; }
    QSpinBox::up-button {
        width: 20px;
        border-left: 1px solid #7AAAC8;
        background: #D6E8F5;
        border-top-right-radius: 2px;
    }
    QSpinBox::down-button {
        width: 20px;
        border-left: 1px solid #7AAAC8;
        background: #D6E8F5;
        border-bottom-right-radius: 2px;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: #AACCEE;
    }
"""

qdoublespinbox_style = """
    QDoubleSpinBox {
        background-color: #F4F9FF;
        border: 1px solid #7AAAC8;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        border-radius: 3px;
        padding: 2px 5px;
        min-height: 26px;
    }
    QDoubleSpinBox:focus { border-color: #0A66B8; }
    QDoubleSpinBox::up-button {
        width: 20px;
        border-left: 1px solid #7AAAC8;
        background: #D6E8F5;
        border-top-right-radius: 2px;
    }
    QDoubleSpinBox::down-button {
        width: 20px;
        border-left: 1px solid #7AAAC8;
        background: #D6E8F5;
        border-bottom-right-radius: 2px;
    }
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
        background: #AACCEE;
    }
"""

qdialog_style = """
    QDialog {
        background-color: #D6E8F5;
        border-radius: 4px;
    }
"""

qlabel_style = """
    QLabel {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        background-color: transparent;
    }
"""

sequence_qlabel_style = """
    QLabel {
        background-color: #0A66B8;
        color: #FFFFFF;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        padding: 2px 6px;
        border-radius: 2px;
    }
"""

# OK / NG mark buttons — large, high contrast
sequence_qpushbutton_style = """
    QPushButton {
        background-color: #EEF6FF;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 46px;
        font-weight: bold;
        border-radius: 6px;
        border: 2px solid #7AAAC8;
        padding: 2px;
    }
    QPushButton:hover {
        background-color: #AACCEE;
        border-color: #0A66B8;
        color: #064884;
    }
    QPushButton:pressed {
        background-color: #88BBDD;
        border-color: #0857A0;
        color: #042E60;
    }
    QPushButton:disabled {
        background-color: #D0E4F4;
        border-color: #A8C4D8;
        color: #7A9AB4;
    }
"""

main_window_menubar_style = """
    QMenuBar {
        background-color: transparent;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        spacing: 2px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 3px 10px;
        border-radius: 3px;
    }
    QMenuBar::item:selected {
        background-color: #AACCEE;
        color: #064884;
    }
    QMenuBar::item:pressed {
        background-color: #88BBDD;
    }
    QMenu {
        background-color: #EEF6FF;
        border: 1px solid #7AAAC8;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        padding: 3px 0;
    }
    QMenu::item {
        padding: 5px 28px 5px 14px;
    }
    QMenu::item:selected {
        background-color: #AACCEE;
        color: #064884;
    }
    QMenu::separator {
        height: 1px;
        background: #7AAAC8;
        margin: 3px 6px;
    }
"""

qframe_style = """
    QFrame {
        color: #7AAAC8;
    }
"""

hardware_qframe_style = """
    QFrame {
        color: #7AAAC8;
    }
"""

toolbar_button_style = """
    QPushButton {
        border: none;
        background-color: transparent;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        border-radius: 3px;
        padding: 3px;
    }
    QPushButton:hover {
        background-color: #AACCEE;
    }
    QPushButton:pressed {
        background-color: #88BBDD;
    }
    QPushButton:disabled {
        background-color: transparent;
        color: #7A9AB4;
    }
"""

qcheckbox_style = """
    QCheckBox {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #7AAAC8;
        border-radius: 2px;
        background-color: #F4F9FF;
    }
    QCheckBox::indicator:checked {
        background-color: #0A66B8;
        border-color: #0A66B8;
    }
    QCheckBox::indicator:hover {
        border-color: #0A66B8;
    }
    QCheckBox:focus { outline: none; }
"""

qlistview_style = """
    QListView {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        background-color: #F4F9FF;
        border: 1px solid #7AAAC8;
        border-radius: 3px;
        outline: none;
    }
    QListView::item {
        padding: 2px 5px;
    }
    QListView::item:selected {
        background-color: #AACCEE;
        color: #064884;
    }
    QListView::item:hover {
        background-color: #D6E8F5;
    }
"""

qmainwindow_style = """
    QMainWindow {
        background-color: #D6E8F5;
        border: 1px solid #5A8AAA;
    }
"""

qradiobutton_style = """
    QRadioButton {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        spacing: 5px;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #7AAAC8;
        border-radius: 7px;
        background-color: #F4F9FF;
    }
    QRadioButton::indicator:checked {
        background-color: #0A66B8;
        border-color: #0A66B8;
    }
    QRadioButton::indicator:hover {
        border-color: #0A66B8;
    }
"""

qtabwidget_style = """
    QTabWidget {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
    }
    QTabWidget::pane {
        border: 1px solid #7AAAC8;
        background-color: #EEF6FF;
        border-radius: 0 3px 3px 3px;
    }
    QTabBar::tab {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 14px;
        background-color: #C4D8EC;
        border: 1px solid #7AAAC8;
        border-bottom: none;
        padding: 4px 14px;
        color: #2A4A68;
        border-radius: 3px 3px 0 0;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background-color: #EEF6FF;
        color: #064884;
        border-bottom: 2px solid #0A66B8;
    }
    QTabBar::tab:hover:!selected {
        background-color: #AACCEE;
        color: #064884;
    }
"""

qtextedit_style = """
    QTextEdit {
        background-color: #F4F9FF;
        border: 1px solid #7AAAC8;
        color: #081828;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 16px;
        border-radius: 3px;
        padding: 3px;
    }
    QTextEdit:disabled {
        color: #081828;
        background-color: #EEF6FF;
    }
"""

qtableview_style = """
    QTableView {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 14px;
        color: #081828;
        background-color: #F4F9FF;
        alternate-background-color: #E4F0FA;
        border: 1px solid #7AAAC8;
        gridline-color: #A8C4D8;
        border-radius: 3px;
        outline: none;
    }
    QHeaderView::section {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 14px;
        font-weight: bold;
        background-color: #C4D8EC;
        color: #081828;
        border: none;
        border-bottom: 2px solid #7AAAC8;
        border-right: 1px solid #A8C4D8;
        padding: 4px 7px;
    }
    QTableView::item {
        border-top: 1px solid #C4D8EC;
        color: #081828;
        padding: 2px 4px;
    }
    QTableView::item:selected {
        background-color: #AACCEE;
        color: #064884;
    }
"""

qtreeview_style = """
    QTreeView {
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 15px;
        color: #081828;
        background-color: #F4F9FF;
        border: 1px solid #7AAAC8;
        outline: none;
    }
    QTreeView::item {
        padding: 2px;
    }
    QTreeView::item:selected {
        background-color: #AACCEE;
        color: #064884;
    }
    QTreeView::item:hover {
        background-color: #D6E8F5;
    }
"""

qscrollbar_style = """
    QScrollBar:vertical {
        border: none;
        background: #C4D8EC;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #7AAAC8;
        border-radius: 4px;
        min-height: 28px;
    }
    QScrollBar::handle:vertical:hover {
        background: #0A66B8;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none; background: none; height: 0px;
    }
    QScrollBar:horizontal {
        border: none;
        background: #C4D8EC;
        height: 8px;
        margin: 0px;
    }
    QScrollBar::handle:horizontal {
        background: #7AAAC8;
        border-radius: 4px;
        min-width: 28px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #0A66B8;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        border: none; background: none; width: 0px;
    }
"""

qstatusbar_style = """
    QStatusBar {
        background-color: #C4D8EC;
        color: #2A4A68;
        font-family: 'Microsoft YaHei', 'SimSun';
        font-size: 13px;
        border-top: 1px solid #7AAAC8;
    }
"""
