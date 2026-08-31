from string import Template

from PyQt5.QtWidgets import QApplication

from consts.running_consts import DEFAULT_DIR

UI_FONT_FAMILY = "'SimSun'"
MAIN_UI_FONT_FAMILY = "'Microsoft YaHei UI'"
COMBO_DOWN_ARROW_ICON = DEFAULT_DIR + "ui/ui_pic/select_analysis_model/down.svg"

_BASE_SCREEN_HEIGHT = 1080
_MIN_FONT_SCALE = 0.70
_MAX_FONT_SCALE = 1.50
CUSTOM_WIDGET_CONTROL_VERTICAL_PADDING_PX = 8
CUSTOM_WIDGET_BUTTON_VERTICAL_PADDING_PX = 10

COLOR_TITLE_BAR_BG = "#E9EEF5"
COLOR_MENU_BAR_BG = "#F5F7FA"
COLOR_TOOLBAR_BG = "#EEF3F8"
COLOR_TOOLBAR_BUTTON_BG = "#EEF3F8"
COLOR_CONTROL_HOVER = "#E7EEF7"
COLOR_CONTROL_PRESSED = "#DDE7F3"
COLOR_PRIMARY = "#3B6FB6"
COLOR_PRIMARY_HOVER = "#2F5FA8"
COLOR_WAVEFORM = "#5B8FD9"
COLOR_PAGE_BG = "#F3F6FA"
COLOR_CARD_BG = "#FFFFFF"
COLOR_PANEL_BG = "#FFFFFF"
COLOR_FIELD_DISABLED_BG = "#F7FAFE"
COLOR_BORDER = "#C9D6E8"
COLOR_BORDER_STRONG = "#AFC0D6"
COLOR_TEXT = "#1F2937"
COLOR_TEXT_MUTED = "#64748B"
COLOR_DISABLED_BG = "#EEF3F8"
COLOR_DISABLED_TEXT = "#94A3B8"
COLOR_OK = "#16A34A"
COLOR_NG = "#DC2626"


def _get_font_scale() -> float:
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


def scale_size_px(size_px: float) -> int:
    return max(1, int(round(size_px * _FONT_SCALE)))


def _style(template: str) -> str:
    return Template(template).substitute(globals())


qpushbutton_style = _style("""
            QPushButton {
                border: 1px solid $COLOR_BORDER_STRONG;
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                border-radius: 5px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: $COLOR_CONTROL_HOVER;
                border-color: $COLOR_PRIMARY;
            }
            QPushButton:pressed {
                background-color: $COLOR_CONTROL_PRESSED;
                border-color: $COLOR_PRIMARY_HOVER;
            }
            QPushButton:disabled {
                background-color: $COLOR_DISABLED_BG;
                color: $COLOR_DISABLED_TEXT;
                border-color: $COLOR_BORDER_STRONG;
            }
        """)

qlineedit_style = _style("""
            QLineEdit {
                border: 1px solid $COLOR_BORDER_STRONG;
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                border-radius: 4px;
                padding: 4px 8px;
                selection-background-color: $COLOR_PRIMARY;
            }
            QLineEdit:disabled {
                background-color: $COLOR_FIELD_DISABLED_BG;
                color: $COLOR_DISABLED_TEXT;
                border-color: $COLOR_BORDER;
            }
        """)

qcombobox_style = _style("""
            QComboBox {
                border: 1px solid $COLOR_BORDER_STRONG;
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                selection-background-color: $COLOR_PRIMARY;
            }
            QComboBox QAbstractItemView::item {
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
            }
        """)

qgroupbox_style = _style("""
            QGroupBox {
                background: transparent;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                padding-left: 0px;
                padding-top: 0px;
                margin-top: 0px;
            }
        """)

qspinbox_style = _style("""
            QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid $COLOR_BORDER_STRONG;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QSpinBox::up-button {
                width: 20px;
            }
            QSpinBox::down-button {
                width: 20px;
            }
        """)

qdoublespinbox_style = _style("""
            QDoubleSpinBox {
                background-color: #FFFFFF;
                border: 1px solid $COLOR_BORDER_STRONG;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QDoubleSpinBox::up-button {
                width: 20px;
            }
            QDoubleSpinBox::down-button {
                width: 20px;
            }
        """)

qdialog_style = _style("""
            QDialog {
                border-radius: 10px;
                background-color: $COLOR_PAGE_BG;
            }
""")

qlabel_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                color: $COLOR_TEXT;
            }
""")

sequence_qlabel_style = _style("""
            QLabel {
                background-color: $COLOR_PRIMARY;
                color: white;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
""")

sequence_qpushbutton_style = _style("""
            QPushButton {
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 70px;
                border-radius: 20px;
                border: 3px solid $COLOR_BORDER_STRONG;
                padding-left: 0px;
            }
            QPushButton:hover {
                border-color: $COLOR_PRIMARY;
            }
            QPushButton:pressed {
                background-color: #D4E1F2;
                border-color: $COLOR_PRIMARY_HOVER;
            }

""")

main_window_title_label_style = _style("""
            QLabel {
                color: $COLOR_TEXT;
                background-color: transparent;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 20px;
                font-weight: 600;
            }
""")

main_window_title_row_style = _style("""
            QWidget#mainWindowTitleRow {
                background-color: $COLOR_TITLE_BAR_BG;
                border: none;
            }
""")

main_window_title_button_style = """
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #D8E2EF;
            }
            QPushButton:pressed {
                background-color: #C8D6E7;
            }
"""

main_window_close_button_style = """
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #F5D6D6;
            }
            QPushButton:pressed {
                background-color: #EAB8B8;
            }
"""

main_window_menubar_style = _style("""
            QMenuBar#mainWindowMenuBar {
                background-color: transparent;
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: 600;
                border: none;
                padding-left: 0px;
                padding-top: 3px;
                margin: 0px;
            }
            QMenuBar#mainWindowMenuBar::item {
                background-color: transparent;
                padding: 2px 8px;
            }
            QMenuBar#mainWindowMenuBar::item:selected {
                background-color: #E1EAF6;
                color: $COLOR_PRIMARY_HOVER;
            }
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid $COLOR_BORDER;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
            }
            QMenu::item {
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                padding: 5px 18px 5px 30px;
            }
            QMenu::item:selected {
                background-color: #E1EAF6;
                color: $COLOR_PRIMARY_HOVER;
            }
""")

main_window_menu_row_style = _style("""
            QWidget#mainWindowMenuRow {
                background-color: $COLOR_MENU_BAR_BG;
                border-top: 1px solid $COLOR_BORDER;
                border-bottom: 1px solid $COLOR_BORDER;
            }
""")

qframe_style = _style("""
            QFrame {
                color: $COLOR_BORDER;
            }
""")

main_window_statusbar_style = _style("""
            QStatusBar {
                background-color: $COLOR_MENU_BAR_BG;
                border-top: 1px solid $COLOR_BORDER;
                color: $COLOR_TEXT_MUTED;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 15px;
                font-weight: 400;
            }
""")

main_window_status_label_style = _style("""
            QLabel {
                color: $COLOR_TEXT_MUTED;
                background-color: transparent;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 15px;
                font-weight: 400;
            }
""")

motor_left_panel_style = """
            QWidget {
                background: transparent;
            }
"""

motor_port_combo_style = _style("""
            QComboBox {
                background: #FFFFFF;
                color: $COLOR_TEXT;
                border: 1px solid #B7C8DC;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                selection-background-color: $COLOR_PRIMARY;
                selection-color: #FFFFFF;
            }
            QComboBox QAbstractItemView::item {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: $COLOR_PRIMARY;
                color: #FFFFFF;
            }
""")

motor_left_panel_scroll_area_style = """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
"""

motor_section_card_style = _style("""
            QFrame {
                background-color: $COLOR_PANEL_BG;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
            }
""")

motor_section_content_style = _style("""
            QWidget#motorSectionContent {
                background-color: $COLOR_PANEL_BG;
                border: none;
            }
""")

motor_section_scroll_area_style = _style("""
            QScrollArea#motorSectionScrollArea {
                background-color: $COLOR_PANEL_BG;
                border: none;
            }
            QScrollArea#motorSectionScrollArea QWidget#motorSectionContent {
                background-color: $COLOR_PANEL_BG;
                border: none;
            }
""")

motor_section_viewport_style = _style("""
            background-color: $COLOR_PANEL_BG;
            border: none;
""")

motor_section_title_style = _style("""
            QLabel {
                background-color: $COLOR_PRIMARY;
                color: white;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: bold;
                padding: 7px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
""")

motor_field_label_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                color: $COLOR_PRIMARY_HOVER;
                background: transparent;
            }
""")

motor_value_label_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 17px;
                color: $COLOR_TEXT;
                background-color: $COLOR_CARD_BG;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
                padding: 6px 8px;
            }
""")

motor_status_badge_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 17px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 10px;
            }
""")

motor_status_badge_compact_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 30px;
                font-weight: bold;
                border-radius: 5px;
                padding: 11px 14px;
            }
""")

motor_status_badge_large_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 46px;
                font-weight: bold;
                border-radius: 5px;
                padding: 18px 18px;
            }
""")

motor_status_badge_pending_style = _style("""
            QLabel {
                background-color: #EAF1FA;
                color: $COLOR_PRIMARY_HOVER;
                border: 1px solid #B7C8E0;
            }
""")

motor_status_badge_running_style = """
            QLabel {
                background-color: #fff3d8;
                color: #8a6720;
                border: 1px solid #d8b66b;
            }
"""

motor_status_badge_ok_style = _style("""
            QLabel {
                background-color: #E8F7EE;
                color: #166534;
                border: 1px solid $COLOR_OK;
            }
""")

motor_status_badge_ng_style = _style("""
            QLabel {
                background-color: #FCE8E8;
                color: #991B1B;
                border: 1px solid $COLOR_NG;
            }
""")

motor_summary_hint_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 16px;
                color: $COLOR_TEXT_MUTED;
                background: transparent;
                padding: 4px 2px;
            }
""")

motor_result_caption_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 16px;
                font-weight: bold;
                color: $COLOR_PRIMARY_HOVER;
                background: transparent;
                padding: 2px 0px;
            }
""")

motor_inline_title_label_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 15px;
                font-weight: bold;
                color: $COLOR_PRIMARY_HOVER;
                background-color: #F7FAFE;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
                padding: 6px 10px;
            }
""")

motor_inline_title_label_pending_style = _style("""
            QLabel {
                color: $COLOR_PRIMARY_HOVER;
                background-color: #F7FAFE;
                border: 1px solid $COLOR_BORDER;
            }
""")

motor_inline_title_label_running_style = """
            QLabel {
                color: #8a6a27;
                background-color: #fffaf0;
                border: 1px solid #e4d0a1;
            }
"""

motor_inline_title_label_ok_style = """
            QLabel {
                color: #166534;
                background-color: #F6FCF8;
                border: 1px solid #B8E0C8;
            }
"""

motor_inline_title_label_ng_style = """
            QLabel {
                color: #991B1B;
                background-color: #FFF8F8;
                border: 1px solid #F0B8B8;
            }
"""

motor_result_group_frame_style = _style("""
            QFrame {
                background-color: #EEF4FB;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
            }
""")

motor_score_label_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 13px;
                color: $COLOR_TEXT_MUTED;
                background: transparent;
                border: none;
                padding: 0px 2px;
            }
""")

motor_score_title_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 13px;
                color: $COLOR_TEXT_MUTED;
                background: transparent;
                border: none;
                padding: 0px;
            }
""")

motor_final_result_title_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 14px;
                color: $COLOR_TEXT_MUTED;
                background: transparent;
                border: none;
                padding: 0px 2px;
            }
""")

motor_result_divider_style = _style("""
            QFrame {
                color: $COLOR_BORDER;
                background-color: $COLOR_BORDER;
                border: none;
            }
""")

motor_stage_frame_style = _style("""
            QFrame {
                background-color: #EEF4FB;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
            }
""")

motor_stage_title_style = _style("""
            QLabel {
                font-family: $UI_FONT_FAMILY;
                font-size: 16px;
                font-weight: bold;
                color: $COLOR_PRIMARY_HOVER;
                background: transparent;
                padding: 2px 0px;
            }
""")

motor_mode_switch_button_base_style = _style("""
            QPushButton {
                min-height: 40px;
                min-width: 90px;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
                background-color: #F8FBFF;
                color: $COLOR_TEXT_MUTED;
                font-family: $UI_FONT_FAMILY;
                font-size: 17px;
                font-weight: bold;
                padding: 4px 18px;
            }
""")

motor_mode_switch_button_active_style = _style("""
            QPushButton {
                background-color: $COLOR_PRIMARY;
                color: white;
                border: 1px solid $COLOR_PRIMARY;
            }
""")

motor_mode_switch_button_inactive_style = _style("""
            QPushButton {
                background-color: #F8FBFF;
                color: $COLOR_TEXT_MUTED;
                border: 1px solid $COLOR_BORDER;
            }
""")

hardware_qframe_style = """
            QFrame {
                color: rgb(204, 204, 204);
            }
"""

toolbar_container_style = _style("""
            QWidget#sequenceToolsBar {
                background-color: $COLOR_TOOLBAR_BG;
                border-top: 1px solid $COLOR_BORDER;
                border-bottom: 1px solid $COLOR_BORDER_STRONG;
            }
""")

toolbar_separator_style = _style("""
            QFrame {
                color: $COLOR_BORDER;
                background-color: $COLOR_BORDER;
                border: none;
            }
""")

toolbar_button_style = _style("""
            QPushButton {
                border: 1px solid $COLOR_BORDER_STRONG;
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                color: $COLOR_TEXT;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 20px;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:focus {
                outline: none;
                border: 1px solid $COLOR_BORDER_STRONG;
            }
            QPushButton:hover {
                background-color: $COLOR_CONTROL_HOVER;
                border-color: $COLOR_PRIMARY;
            }
            QPushButton:pressed {
                background-color: $COLOR_CONTROL_PRESSED;
                border-color: $COLOR_PRIMARY_HOVER;
            }
            QPushButton:disabled {
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                border-color: $COLOR_BORDER_STRONG;
                color: $COLOR_TEXT_MUTED;
            }
""")

toolbar_field_label_style = _style("""
            QLabel {
                color: $COLOR_TEXT;
                background-color: transparent;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: 600;
            }
""")

toolbar_input_style = _style("""
            QLineEdit {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 4px;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 18px;
                padding: 3px 8px;
                selection-background-color: $COLOR_PRIMARY;
            }
            QLineEdit:disabled {
                background-color: #FFFFFF;
                color: $COLOR_DISABLED_TEXT;
                border-color: $COLOR_BORDER;
            }
""")

toolbar_combobox_style = _style("""
            QComboBox {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 4px;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 18px;
                padding: 3px 30px 3px 8px;
            }
            QComboBox::drop-down {
                border-left: 1px solid $COLOR_BORDER;
                width: 24px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: url($COMBO_DOWN_ARROW_ICON);
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                selection-background-color: #E1EAF6;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 18px;
            }
""")

toolbar_spinbox_style = _style("""
            QSpinBox {
                background-color: #FFFFFF;
                color: $COLOR_TEXT;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 4px;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 18px;
                padding: 3px 8px;
                selection-background-color: $COLOR_PRIMARY;
            }
            QSpinBox:disabled {
                background-color: #FFFFFF;
                color: $COLOR_DISABLED_TEXT;
                border-color: $COLOR_BORDER;
            }
""")


toolbar_checkbox_style = _style("""
            QCheckBox {
                color: $COLOR_TEXT;
                background-color: transparent;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: 600;
                spacing: 4px;
            }
            QCheckBox:focus { outline: none; }
""")

serial_trigger_button_base_style = _style("""
            QPushButton {
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 4px;
                padding: 2px 10px;
                font-family: $MAIN_UI_FONT_FAMILY;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:focus {
                outline: none;
                border: 1px solid $COLOR_BORDER_STRONG;
            }
            QPushButton:hover {
                background-color: $COLOR_CONTROL_HOVER;
                border-color: $COLOR_PRIMARY;
            }
            QPushButton:pressed {
                background-color: $COLOR_CONTROL_PRESSED;
                border-color: $COLOR_PRIMARY_HOVER;
            }
            QPushButton:disabled {
                background-color: $COLOR_DISABLED_BG;
                color: $COLOR_DISABLED_TEXT;
                border: 1px solid $COLOR_BORDER_STRONG;
            }
""")

serial_trigger_button_inactive_style = _style("""
            QPushButton {
                color: $COLOR_TEXT_MUTED;
            }
""")

serial_trigger_button_open_style = _style("""
            QPushButton {
                color: $COLOR_PRIMARY_HOVER;
            }
""")

serial_trigger_button_connected_style = _style("""
            QPushButton {
                color: $COLOR_OK;
            }
""")

count_board_ok_button_style = _style("""
            QPushButton {
                background-color: #E8F7EE;
                color: #166534;
                font-family: $UI_FONT_FAMILY;
                font-size: 24px;
                font-weight: bold;
                border-radius: 8px;
                border: 1px solid $COLOR_OK;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #D6F4E2;
                border-color: #15803D;
            }
            QPushButton:pressed {
                background-color: #BCE8CF;
                border-color: #166534;
                padding: 7px 10px 3px 14px;
            }
""")

count_board_ng_button_style = _style("""
            QPushButton {
                background-color: #FCE8E8;
                color: #991B1B;
                font-family: $UI_FONT_FAMILY;
                font-size: 24px;
                font-weight: bold;
                border-radius: 8px;
                border: 1px solid $COLOR_NG;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #FBD0D0;
                border-color: #B91C1C;
            }
            QPushButton:pressed {
                background-color: #F4B7B7;
                border-color: #991B1B;
                padding: 7px 10px 3px 14px;
            }
""")

count_board_mode_active_style = _style("""
            QPushButton {
                background-color: $COLOR_PRIMARY;
                color: white;
                border: 1px solid $COLOR_PRIMARY;
                border-radius: 5px;
                font-family: $UI_FONT_FAMILY;
                font-size: 17px;
                font-weight: bold;
            }
""")

count_board_mode_inactive_style = _style("""
            QPushButton {
                background-color: $COLOR_TOOLBAR_BUTTON_BG;
                color: $COLOR_TEXT_MUTED;
                border: 1px solid $COLOR_BORDER;
                border-radius: 5px;
                font-family: $UI_FONT_FAMILY;
                font-size: 17px;
                font-weight: bold;
            }
""")

count_board_container_style = _style("""
            QWidget#sequenceCountBoard,
            QWidget#sequenceCountBoardModeSwitch,
            QWidget#sequenceCountBoardPage {
                background-color: transparent;
                border: none;
            }
            QWidget#sequenceCountBoard QLabel {
                background-color: transparent;
            }
""")

recent_session_card_title_style = _style("""
            QLabel {
                background-color: $COLOR_PRIMARY;
                color: white;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: bold;
                padding: 7px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
""")

recent_session_table_style = _style("""
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8FBFF;
                border: 1px solid $COLOR_BORDER;
                gridline-color: #E6EEF8;
                color: $COLOR_TEXT;
                selection-background-color: #DDE9F8;
                selection-color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
            }
            QTableWidget::item {
                padding: 2px 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #DDE9F8;
                color: $COLOR_TEXT;
            }
            QHeaderView::section {
                background-color: #F5F8FC;
                color: $COLOR_TEXT_MUTED;
                border: none;
                border-right: 1px solid #E6EEF8;
                border-bottom: 1px solid #DCE6F2;
                padding: 2px 4px;
                font-family: $UI_FONT_FAMILY;
            }
            QTableCornerButton::section {
                background-color: #F5F8FC;
                border: none;
                border-right: 1px solid #E6EEF8;
                border-bottom: 1px solid #DCE6F2;
            }
""")

product_test_program_dialog_style = _style("""
            QDialog#productTestProgramDialog {
                background-color: $COLOR_PAGE_BG;
            }
            QLabel#productProgramFieldLabel,
            QLabel#productProgramSectionTitle {
                color: $COLOR_TEXT;
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton#productProgramPrimaryButton {
                color: #FFFFFF;
                background-color: $COLOR_PRIMARY;
                border-color: $COLOR_PRIMARY;
                font-weight: bold;
            }
            QPushButton#productProgramPrimaryButton:hover {
                background-color: $COLOR_PRIMARY_HOVER;
                border-color: $COLOR_PRIMARY_HOVER;
            }
            QFrame#productProgramFooterSeparator {
                color: $COLOR_BORDER;
            }
""")

product_test_program_config_selector_style = _style("""
            QComboBox,
            QComboBox QLineEdit {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
""")

product_test_program_table_style = recent_session_table_style + _style("""
            QTableWidget#productProgramTable {
                font-size: 17px;
            }
            QTableWidget#productProgramTable QHeaderView::section {
                font-size: 17px;
                font-weight: bold;
                padding: 5px 6px;
            }
""")

recent_session_action_button_style = """
            QToolButton {
                border: 1px solid rgba(75, 85, 99, 0.24);
                border-radius: 4px;
                background-color: rgba(238, 243, 248, 0.95);
                padding: 2px;
            }
            QToolButton:hover {
                background-color: rgba(225, 234, 246, 0.95);
            }
            QToolButton:pressed {
                background-color: rgba(212, 225, 242, 0.95);
            }
            QToolButton:disabled {
                color: rgba(148, 163, 184, 0.42);
                border-color: rgba(148, 163, 184, 0.14);
                background-color: rgba(248, 250, 252, 0.42);
            }
"""

waveform_frame_style = _style("""
            QFrame {
                background: #FFFFFF;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 2px;
            }
""")

waveform_title_bar_style = _style("""
            QWidget {
                background-color: $COLOR_PRIMARY;
                border: none;
            }
            QWidget#directionWaveformHeader {
                border-top-left-radius: 2px;
                border-top-right-radius: 2px;
            }
            QWidget#directionWaveformBody {
                background-color: #FFFFFF;
                border: none;
            }
            QLabel {
                color: white;
                background-color: transparent;
                font-family: $UI_FONT_FAMILY;
                font-weight: 600;
            }
""")

waveform_title_label_style = _style("""
            QLabel {
                background-color: $COLOR_PRIMARY;
                color: white;
                font-family: $UI_FONT_FAMILY;
                font-weight: 600;
                padding-left: 8px;
            }
""")

waveform_action_button_style = _style("""
            QPushButton {
                min-width: 34px;
                min-height: 20px;
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 3px;
                background-color: #F8FBFF;
                color: $COLOR_PRIMARY_HOVER;
                font-family: $UI_FONT_FAMILY;
                font-size: 12px;
                font-weight: 600;
                padding: 1px 6px;
            }
            QPushButton:hover {
                background-color: #EAF2FC;
                border-color: $COLOR_PRIMARY;
            }
            QPushButton:pressed {
                background-color: #DDE9F8;
            }
            QPushButton#conditionOkButton {
                color: #166534;
                border-color: $COLOR_OK;
                background-color: #E8F7EE;
            }
            QPushButton#conditionNgButton {
                color: #991B1B;
                border-color: $COLOR_NG;
                background-color: #FCE8E8;
            }
""")

waveform_mark_button_style = _style("""
            QPushButton {
                border: 1px solid $COLOR_BORDER_STRONG;
                border-radius: 4px;
                background-color: #F8FBFF;
                color: $COLOR_PRIMARY_HOVER;
                font-family: $UI_FONT_FAMILY;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #DDEBFA;
                border-color: $COLOR_PRIMARY;
                border-width: 2px;
                color: #1E4F95;
            }
            QPushButton:pressed {
                background-color: #C7D9F2;
                border-width: 2px;
                padding-top: 2px;
                padding-left: 1px;
            }
            QPushButton:disabled {
                background-color: $COLOR_DISABLED_BG;
                color: $COLOR_DISABLED_TEXT;
                border-color: $COLOR_BORDER;
            }
            QPushButton#conditionOkButton {
                color: #065F46;
                border-color: $COLOR_OK;
                background-color: #E8F7EE;
            }
            QPushButton#conditionOkButton:hover {
                color: #064E3B;
                background-color: #C8F1D8;
                border-color: #0F8F3A;
                border-width: 2px;
            }
            QPushButton#conditionOkButton:pressed {
                color: #064E3B;
                background-color: #A9E2C1;
                border-color: #047A2D;
                border-width: 2px;
                padding-top: 2px;
                padding-left: 1px;
            }
            QPushButton#conditionNgButton {
                color: #8B0000;
                border-color: $COLOR_NG;
                background-color: #FCE8E8;
            }
            QPushButton#conditionNgButton:hover {
                color: #7F1D1D;
                background-color: #F8BEBE;
                border-color: #B91C1C;
                border-width: 2px;
            }
            QPushButton#conditionNgButton:pressed {
                color: #7F1D1D;
                background-color: #EF9F9F;
                border-color: #991B1B;
                border-width: 2px;
                padding-top: 2px;
                padding-left: 1px;
            }
""")

waveform_canvas_style = _style("""
            QWidget {
                background-color: #F7FAFE;
                border: 1px solid $COLOR_BORDER;
            }
""")

waveform_viewport_style = "background-color: #EEF4FB;"

qcheckbox_style = _style("""
            QCheckBox {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                color: $COLOR_TEXT;
            }
            QCheckBox:focus { outline: none; }
""")

qlistview_style = _style("""
            QListView {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
""")

qmainwindow_style = _style("""
            QMainWindow  {
                background-color: $COLOR_PAGE_BG;
                border: 1px solid rgb(0, 0, 0);
            }
""")

main_window_base_style = (
    qlabel_style.replace(UI_FONT_FAMILY, MAIN_UI_FONT_FAMILY)
    + qpushbutton_style.replace(UI_FONT_FAMILY, MAIN_UI_FONT_FAMILY)
    + qmainwindow_style
)

qradiobutton_style = _style("""
            QRadioButton {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
                color: $COLOR_TEXT;
            }
""")

qtabwidget_style = _style("""
            QTabWidget {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
""")

qtextedit_style = _style("""
            QTextEdit{
                background-color: white;
                font-family: $UI_FONT_FAMILY;
                font-size: 30px;
                color: $COLOR_TEXT;
            }
             QTextEdit:disabled {
                color: $COLOR_TEXT;
            }
""")

qtableview_style = _style("""
            QTableView {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
            QHeaderView::section {
                font-family: $UI_FONT_FAMILY;
                font-size: 20px;
            }
            QTableView::item {
                border-top: 1px solid $COLOR_BORDER;
                color: $COLOR_TEXT;
            }
            QTableView::item:selected {
                background-color: #DDE9F8;
                color: $COLOR_TEXT;
            }
""")

qtreeview_style = _style("""
            QTreeView {
                font-family: $UI_FONT_FAMILY;
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
""")
