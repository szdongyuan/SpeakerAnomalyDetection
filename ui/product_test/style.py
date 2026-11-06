"""Original product configuration theme, extracted without changing its palette."""
from consts import ui_style_const

CONTROL_STYLE = f"font-family: {ui_style_const.UI_FONT_FAMILY}; font-size: 16px;"

SELECTOR_STYLE = ui_style_const.qcombobox_style + f"""
    QComboBox, QComboBox QAbstractItemView, QComboBox QAbstractItemView::item {{
        {CONTROL_STYLE}
    }}
    QComboBox {{ padding: 0 8px; }}
"""

REMOVE_BUTTON_STYLE = f"""
    QToolButton {{
        border: none;
        border-radius: 3px;
        background: transparent;
        padding: 0;
    }}
    QToolButton:hover {{ background: {ui_style_const.COLOR_CONTROL_HOVER}; }}
    QToolButton:pressed {{ background: {ui_style_const.COLOR_CONTROL_PRESSED}; }}
"""

STATION_MENU_STYLE = f"""
    QMenu {{
        {CONTROL_STYLE}
        background-color: {ui_style_const.COLOR_PANEL_BG};
        color: {ui_style_const.COLOR_TEXT};
        border: 1px solid {ui_style_const.COLOR_BORDER};
        padding: 3px 0;
    }}
    QCheckBox {{
        {CONTROL_STYLE}
        color: {ui_style_const.COLOR_TEXT};
        padding: 5px 8px;
        spacing: 8px;
        background: transparent;
    }}
    QCheckBox:hover {{ background: {ui_style_const.COLOR_CONTROL_HOVER}; }}
"""

TABLE_STYLE = ui_style_const.product_test_program_table_style + f"""
    QTableWidget#productProgramTable,
    QTableWidget#productProgramTable QHeaderView::section {{
        {CONTROL_STYLE}
        font-weight: normal;
    }}
"""

DIALOG_STYLE = (
ui_style_const.product_test_program_dialog_style
            + """
            QDialog#productTestProjectDialog {
                background-color: #F3F6F9;
            }
            QDialog#productTestProjectDialog QLabel,
            QDialog#productTestProjectDialog QPushButton,
            QDialog#productTestProjectDialog QLineEdit,
            QDialog#productTestProjectDialog QSpinBox,
            QDialog#productTestProjectDialog QComboBox,
            QDialog#productTestProjectDialog QTableWidget,
            QDialog#productTestProjectDialog QHeaderView::section {
                font-family: __UI_FONT_FAMILY__;
                font-size: 16px;
            }
            QDialog#productTestProjectDialog QTableWidget#productProgramTable QHeaderView::section {
                font-weight: 500;
            }
            QLabel#productProjectSectionHeader {
                min-height: 30px;
                max-height: 30px;
                padding: 0 12px;
                color: #1F2937;
                background-color: #D4E1F2;
                border: none;
                font-family: __UI_FONT_FAMILY__;
                font-size: 16px;
                font-weight: 400;
            }
            QWidget#productProjectConditionHeader {
                min-height: 38px;
                max-height: 38px;
                background-color: #D4E1F2;
                border: none;
            }
            QLabel#productProjectConditionTitle {
                color: #1F2937;
                background-color: transparent;
                font-family: __UI_FONT_FAMILY__;
                font-size: 16px;
                font-weight: 400;
            }
            QPushButton#productProjectPortButton {
                min-height: 30px;
                padding: 0 8px;
                color: #334155;
                border: 1px solid #C9D6E8;
                border-radius: 3px;
                background: #FFFFFF;
                font-weight: normal;
                outline: none;
            }
            QPushButton#productProjectPortButton:checked {
                color: #245C96;
                border-color: #3B6FB6;
                background: #EAF2FB;
                font-weight: normal;
            }
            QPushButton#productProjectPortButton:hover:!checked {
                background: #F3F7FC;
                border-color: #9EB5D1;
            }
            QLineEdit#productProjectPortNameEditor {
                padding: 0 8px;
                color: #245C96;
                border: 1px solid #3B6FB6;
                border-radius: 3px;
                background: #FFFFFF;
                selection-color: #FFFFFF;
                selection-background-color: #3B6FB6;
            }
            QScrollArea#productProjectPortScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea#productProjectPortScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }
            """.replace("__UI_FONT_FAMILY__", ui_style_const.UI_FONT_FAMILY)
)
