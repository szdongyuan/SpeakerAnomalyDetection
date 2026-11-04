"""Light daily-details styling scoped to the dialog's object name."""

from consts.running_consts import DEFAULT_DIR


MOTOR_DAILY_DETAILS_STYLE = """
QDialog#motorDailyDetailsDialog,
QDialog#motorDailyDetailsDialog QWidget {
    background-color: #eef3f8; color: #293b4d; border: none;
    font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: @body-font-size@pt;
}
QDialog#motorDailyDetailsDialog QLabel {
    background: transparent; color: #293b4d; border: none;
}
QDialog#motorDailyDetailsDialog QLabel[statisticValue="true"] {
    font-size: @value-font-size@pt;
}
QDialog#motorDailyDetailsDialog QLabel[secondary="true"] {
    color: #617183;
}
QDialog#motorDailyDetailsDialog QLabel[status="OK"] { color: #00824e; }
QDialog#motorDailyDetailsDialog QLabel[status="NG"] { color: #c32b42; }
QDialog#motorDailyDetailsDialog QWidget#detailsStatistics {
    border-bottom: 1px solid #dce5ee;
}
QDialog#motorDailyDetailsDialog QComboBox,
QDialog#motorDailyDetailsDialog QLineEdit,
QDialog#motorDailyDetailsDialog QPushButton {
    background-color: #ffffff; color: #293b4d;
    border: 1px solid #dce5ee; border-radius: 4px; padding: 3px 8px;
}
QDialog#motorDailyDetailsDialog QComboBox {
    padding-right: 26px;
}
QDialog#motorDailyDetailsDialog QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right;
    width: 20px; border: none; background: transparent;
}
QDialog#motorDailyDetailsDialog QComboBox::down-arrow {
    image: url("@combo-down-arrow@"); width: 10px; height: 6px;
}
QDialog#motorDailyDetailsDialog QComboBox QAbstractItemView {
    background: #ffffff; color: #293b4d;
    selection-background-color: #edf4fc; selection-color: #293b4d;
}
QDialog#motorDailyDetailsDialog QLineEdit:focus,
QDialog#motorDailyDetailsDialog QComboBox:focus,
QDialog#motorDailyDetailsDialog QPushButton:focus { border-color: #3e75ac; }
QDialog#motorDailyDetailsDialog QPushButton:hover { background-color: #edf4fc; }
QDialog#motorDailyDetailsDialog QPushButton:disabled {
    color: #9aa7b5; background-color: #f3f6fa;
}
QDialog#motorDailyDetailsDialog QTableWidget {
    background: #ffffff; alternate-background-color: #f3f6fa;
    color: #293b4d; gridline-color: #dce5ee; border: 1px solid #dce5ee;
}
QDialog#motorDailyDetailsDialog QTableWidget::item {
    border-bottom: 1px solid #dce5ee; padding: 0px 5px;
}
QDialog#motorDailyDetailsDialog QTableWidget::item:hover { background: #edf4fc; }
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar:vertical {
    background: #fafafa; width: 14px; margin: 14px 0px;
    border: none;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::handle:vertical {
    background: #8b8b8b; border-radius: 4px;
    margin: 0px 3px; min-height: 20px;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::sub-line:vertical {
    background: #fafafa; border: none; height: 14px;
    subcontrol-origin: margin; subcontrol-position: top;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::add-line:vertical {
    background: #fafafa; border: none; height: 14px;
    subcontrol-origin: margin; subcontrol-position: bottom;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::up-arrow:vertical {
    image: url(:/qt-project.org/styles/commonstyle/images/titlebar-shade-16.png);
    width: 14px; height: 12px;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::down-arrow:vertical {
    image: url(:/qt-project.org/styles/commonstyle/images/titlebar-unshade-16.png);
    width: 14px; height: 12px;
}
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::sub-page:vertical,
QDialog#motorDailyDetailsDialog QScrollBar#detailsVerticalScrollBar::add-page:vertical {
    background: #fafafa;
}
QDialog#motorDailyDetailsDialog QHeaderView::section {
    background: #f3f6fa; color: #617183; border: none;
    border-bottom: 1px solid #dce5ee; padding: 4px 5px;
}
QDialog#motorDailyDetailsDialog QLabel#detailsEmpty { color: #617183; }
""".replace("@combo-down-arrow@", DEFAULT_DIR + "ui/ui_analysis_config/assets/combobox_down_arrow.svg")
