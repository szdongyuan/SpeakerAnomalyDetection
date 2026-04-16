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
                background-color: #F0F0F0;
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

motor_left_panel_style = """
            QWidget {
                background: transparent;
            }
"""

motor_left_panel_scroll_area_style = """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
"""

motor_section_card_style = """
            QFrame {
                background-color: #f3f6fb;
                border: 1px solid #b7c7dd;
                border-radius: 5px;
            }
"""

motor_section_title_style = """
            QLabel {
                background-color: #4472c4;
                color: white;
                font-family: 'SimSun';
                font-size: 18px;
                font-weight: bold;
                padding: 7px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
"""

motor_field_label_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 18px;
                color: #2f4770;
                background: transparent;
            }
"""

motor_value_label_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 17px;
                color: #24344f;
                background-color: #ffffff;
                border: 1px solid #bfd0e6;
                border-radius: 5px;
                padding: 6px 8px;
            }
"""

motor_status_badge_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 17px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 10px;
            }
"""

motor_status_badge_compact_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 30px;
                font-weight: bold;
                border-radius: 5px;
                padding: 11px 14px;
            }
"""

motor_status_badge_large_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 46px;
                font-weight: bold;
                border-radius: 5px;
                padding: 18px 18px;
            }
"""

motor_status_badge_pending_style = """
            QLabel {
                background-color: #e8eef8;
                color: #4a628a;
                border: 1px solid #a9bddb;
            }
"""

motor_status_badge_running_style = """
            QLabel {
                background-color: #fff3d8;
                color: #8a6720;
                border: 1px solid #d8b66b;
            }
"""

motor_status_badge_ok_style = """
            QLabel {
                background-color: #deefdf;
                color: #29613a;
                border: 1px solid #87b48e;
            }
"""

motor_status_badge_ng_style = """
            QLabel {
                background-color: #f6dddd;
                color: #9b3434;
                border: 1px solid #d49a9a;
            }
"""

motor_summary_hint_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 16px;
                color: #5f6368;
                background: transparent;
                padding: 4px 2px;
            }
"""

motor_result_caption_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 16px;
                font-weight: bold;
                color: #3c5f9d;
                background: transparent;
                padding: 2px 0px;
            }
"""

motor_inline_title_label_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 15px;
                font-weight: bold;
                color: #5f79a4;
                background-color: #f9fbff;
                border: 1px solid #d2ddec;
                border-radius: 5px;
                padding: 6px 10px;
            }
"""

motor_inline_title_label_pending_style = """
            QLabel {
                color: #5f79a4;
                background-color: #f9fbff;
                border: 1px solid #d2ddec;
            }
"""

motor_inline_title_label_running_style = """
            QLabel {
                color: #8a6a27;
                background-color: #fffaf0;
                border: 1px solid #e4d0a1;
            }
"""

motor_inline_title_label_ok_style = """
            QLabel {
                color: #5b7f60;
                background-color: #f8fcf8;
                border: 1px solid #c6ddc9;
            }
"""

motor_inline_title_label_ng_style = """
            QLabel {
                color: #9a6262;
                background-color: #fff8f8;
                border: 1px solid #e8caca;
            }
"""

motor_result_group_frame_style = """
            QFrame {
                background-color: #edf3fa;
                border: 1px solid #c2d1e4;
                border-radius: 5px;
            }
"""

motor_score_label_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 13px;
                color: #657b9f;
                background: transparent;
                border: none;
                padding: 0px 2px;
            }
"""

motor_score_title_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 13px;
                color: #5f759a;
                background: transparent;
                border: none;
                padding: 0px;
            }
"""

motor_final_result_title_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 14px;
                color: #7186aa;
                background: transparent;
                border: none;
                padding: 0px 2px;
            }
"""

motor_result_divider_style = """
            QFrame {
                color: #c8d5e6;
                background-color: #c8d5e6;
                border: none;
            }
"""

motor_stage_frame_style = """
            QFrame {
                background-color: #edf3fa;
                border: 1px solid #c2d1e4;
                border-radius: 5px;
            }
"""

motor_stage_title_style = """
            QLabel {
                font-family: 'SimSun';
                font-size: 16px;
                font-weight: bold;
                color: #3c5f9d;
                background: transparent;
                padding: 2px 0px;
            }
"""

motor_mode_switch_button_base_style = """
            QPushButton {
                min-height: 40px;
                min-width: 90px;
                border: 1px solid #bfd0e5;
                border-radius: 5px;
                background-color: #fafcff;
                color: #5f7599;
                font-family: 'SimSun';
                font-size: 17px;
                font-weight: bold;
                padding: 4px 18px;
            }
"""

motor_mode_switch_button_active_style = """
            QPushButton {
                background-color: #4472c4;
                color: white;
                border: 1px solid #4472c4;
            }
"""

motor_mode_switch_button_inactive_style = """
            QPushButton {
                background-color: #f8fbff;
                color: #7a8ead;
                border: 1px solid #d1dceb;
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
            # QTreeView::branch:has-siblings:!adjoins-item {
            #     border-image: url(vline.png) 0;
            # }

            # QTreeView::branch:has-siblings:adjoins-item {
            #     border-image: url(branch-more.png) 0;
            # }

            # QTreeView::branch:!has-children:!has-siblings:adjoins-item {
            #     border-image: url(branch-end.png) 0;
            # }

            # QTreeView::branch:has-children:!has-siblings:closed,
            # QTreeView::branch:closed:has-children:has-siblings {
            #     border-image: none;
            #     image: url(branch-closed.png);
            # }

            # QTreeView::branch:open:has-children:!has-siblings,
            # QTreeView::branch:open:has-children:has-siblings  {
            #     border-image: none;
            #     image: url(branch-open.png);
            # }
"""
