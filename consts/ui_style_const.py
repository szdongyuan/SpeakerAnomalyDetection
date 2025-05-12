qpushbutton_stytle = """
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
qlineedit_stytle = """
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
qcombobox_stytle = """
            QComboBox {
                border: 1px solid rgb(173, 173, 173);
                background-color: rgb(225, 225, 225);
                font-family: 'SimSun';
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
        """
qgroupbox_stytle = """
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
qspinbox_stytle = """
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
qdoublespinbox_stytle = """
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

qdialog_stytle = """
            QDialog {
                border-radius: 20px;
            }
"""

qlabel_stytle = """
            QLabel {
                font-family: 'SimSun';
                font-size: 20px;
                color: black;
            }
"""

sequence_qlabel_stytle = """
            QLabel {
                background-color: #4472c4;
                color: white;
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

sequence_qpushbutton_stytle = """
            QPushButton {
                background-color: #d0ceca;
                color: white;
                font-family: 'SimSun';
                font-size: 70px;
                border-radius: 20px;
                border: 1px solid rgb(128, 128, 128);
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

main_window_menubar_stytle = """
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

qframe_stytle = """
            QFrame {
                color: rgb(173, 173, 173);
            }
"""
hardware_qframe_stytle = """
            QFrame {
                color: rgb(204, 204, 204);
            }
"""

toolbar_button_stytle = """
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

qcheckbox_stytle = """
            QCheckBox {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

qlistview_stytle = """
            QListView {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

qmainwindow_stytle = """
            QMainWindow  {
                background-color: #f0f0f0;
                border: 1px solid rgb(0, 0, 0);
            }
"""

qradiobutton_stytle = """
            QRadioButton {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""

qtabwidget_stytle = """
            QTabWidget {
                font-family: 'SimSun';
                font-size: 20px;
            }
"""
qtextedit_stytle = """                    
            QTextEdit{
                background-color: white;
                font-family: 'SimSun';
                font-size: 30px;
            }
             QTextEdit:disabled {
                color: rgb(0, 0, 0);
            }
"""

qtableview_stytle = """
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