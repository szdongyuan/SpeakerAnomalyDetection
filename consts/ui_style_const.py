qpushbutton_stytle = """
            QPushButton {
                border-left: 1px solid rgb(128, 128, 128);
                border-top: 1px solid rgb(128, 128, 128);
                border-right: 3px solid rgb(128, 128, 128);
                border-bottom: 3px solid rgb(128, 128, 128);
                background-color: rgb(225, 225, 225);
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
                font-size: 20px;
                border-radius: 3px;
                padding: 3px;
            }
        """
qcombobox_stytle = """
            QComboBox {
                border: 1px solid rgb(173, 173, 173);
                background-color: rgb(225, 225, 225);
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
                font-size: 20px;
                color: black;
            }
"""

sequence_qlabel_stytle = """
            QLabel {
                background-color: #4472c4;
                color: white;
                font-size: 20px;
            }
"""

sequence_qpushbutton_stytle = """
            QPushButton {
                background-color: #4472c4;
                color: white;
                font-size: 40px;
            }
            QPushButton:hover {
               border-color: #803333ff;
            }
"""

main_window_menubar_stytle = """
            QMenuBar {
                background-color:transparent;
                font-size: 20px;
            }
            QMenu {
                font-size: 20px;
            }
            QMenu::item {
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

toolbar_button_stytle = """
            QPushButton {
                border: None;
                background-color: transparent;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #5099ccff;
            }
"""

qcheckbox_stytle = """
            QCheckBox {
                font-size: 20px;
            }
"""

qlistview_stytle = """
            QListView {
                font-size: 20px;
            }
"""