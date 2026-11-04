"""Station display above a summary that stays visible during station scrolling."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from consts.ui_style_const import MOTOR_DASHBOARD_PANEL_STYLE
from ui.sequence.motor_information_summary_panel import MotorInformationSummaryPanel
from ui.sequence.motor_station_test_panel import MotorStationTestPanel


class MotorDashboardPanel(QWidget):
    """Own independent station and template-summary components for each instance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("motorDashboardPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(MOTOR_DASHBOARD_PANEL_STYLE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.station_panel = MotorStationTestPanel(self)
        self.summary_panel = MotorInformationSummaryPanel(self)
        layout.addWidget(self.station_panel, 1)
        layout.addWidget(self.summary_panel)
