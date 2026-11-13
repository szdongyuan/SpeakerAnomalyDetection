from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_dashboard_panel import MotorDashboardPanel
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel


class MotorDetectionLeftPanel(QWidget):
    """
    Composite left sidebar for motor detection mode.

    It delegates the top station dashboard and the bottom video-monitor area to two
    dedicated sub-widgets so later business wiring can evolve independently.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dashboard_panel = MotorDashboardPanel(self)
        self.station_panel = self.dashboard_panel.station_panel
        self.video_monitor_panel = MotorVideoMonitorPanel(self)
        self._init_ui()

    def _init_ui(self):
        # The two inner sections are normally moved into outer QSplitters
        # via take_split_sections(), so the panel itself only needs a sensible
        # minimum so it does not collapse if it ever gets shown standalone.
        self.setMinimumWidth(340)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setStyleSheet(ui_style_const.motor_left_panel_style)

        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout()
        self.content_layout.addWidget(self.dashboard_panel)
        self.content_layout.addWidget(self.video_monitor_panel)
        self.content_layout.addStretch(1)
        self.content_layout.setContentsMargins(0, 0, 6, 0)
        self.content_layout.setSpacing(12)
        self.content_widget.setLayout(self.content_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidget(self.content_widget)
        self.scroll_area.setStyleSheet(ui_style_const.motor_left_panel_scroll_area_style)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    def take_split_sections(self):
        return (
            self._detach_section_widget(self.dashboard_panel),
            self._detach_section_widget(self.video_monitor_panel),
        )

    def _detach_section_widget(self, widget: QWidget):
        if widget is None:
            return None
        if getattr(self, "content_layout", None) is not None:
            index = self.content_layout.indexOf(widget)
            if index >= 0:
                item = self.content_layout.takeAt(index)
                if item is not None and item.widget() is not None:
                    item.widget().setParent(None)
        if widget.parent() is not None:
            widget.setParent(None)
        return widget

    def set_station_count(self, count):
        return self.station_panel.set_station_count(count)

    def set_daily_records(self, records, record_date=None):
        return self.dashboard_panel.summary_panel.set_daily_records(records, record_date)

    def set_station_row_count(self, station_id, count):
        return self.station_panel.set_station_row_count(station_id, count)

    def set_station_row(self, station_id, row, values):
        return self.station_panel.set_station_row(station_id, row, values)

    def set_station_column(self, station_id, column, values):
        return self.station_panel.set_station_column(station_id, column, values)

    def set_station_cell(self, station_id, row, column, value):
        return self.station_panel.set_station_cell(station_id, row, column, value)

    def set_station_results(self, station_id, forward, reverse):
        return self.station_panel.set_station_results(station_id, forward, reverse)

    def get_station_count(self):
        return self.station_panel.get_station_count()

    def get_station_data(self, station_id):
        return self.station_panel.get_station_data(station_id)
