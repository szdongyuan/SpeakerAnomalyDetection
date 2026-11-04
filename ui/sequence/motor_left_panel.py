from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_dashboard_panel import MotorDashboardPanel
from ui.sequence.product_condition_state import ProductConditionState
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel


class MotorDetectionLeftPanel(QWidget):
    condition_selected = pyqtSignal(str)

    """
    Composite left sidebar for motor detection mode.

    It delegates the top station dashboard and the bottom video-monitor area to two
    dedicated sub-widgets so later business wiring can evolve independently.
    """

    def __init__(
        self, summary_widget: QWidget, parent=None, condition_configs=None, queue_catalog=None
    ):
        super().__init__(parent)
        # Mode switch panel removed from UI (no "测试/标记" toggle buttons).
        self.mode_switch_panel = None
        self.count_board = summary_widget
        if self.count_board is not None:
            self.count_board.hide()
        self.condition_state = ProductConditionState(
            self, condition_configs=condition_configs, queue_catalog=queue_catalog
        )
        self.condition_state.condition_selected.connect(self.condition_selected.emit)
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

    def reset_result_panel(self):
        """Reset existing test-round business state independently of the dashboard."""
        self.condition_state.reset()

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        self.condition_state.set_current_stage(stage_text, tone=tone)

    def clear_current_stage(self):
        self.condition_state.clear_current_stage()

    def set_forward_result(self, result_text: str, tone: str = None):
        self.condition_state.set_forward_result(result_text, tone=tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        self.condition_state.set_reverse_result(result_text, tone=tone)

    def set_final_result(self, result_text: str, tone: str = None):
        self.condition_state.set_final_result(result_text, tone=tone)

    def get_automatic_round_result(self):
        return self.condition_state.get_automatic_round_result()

    def set_condition_result(self, condition, result_text: str, tone: str = None):
        return self.condition_state.set_condition_result(condition, result_text, tone=tone)

    def set_condition_analysis_details(self, condition, detail_values):
        return self.condition_state.set_condition_analysis_details(condition, detail_values)

    def set_condition_channel_results(
        self, condition, channel_results, *, restore_channels=False,
        restored_analysis_channels=None, restored_analysis_column_channels=None,
    ):
        return self.condition_state.set_condition_channel_results(
            condition, channel_results, restore_channels=restore_channels,
            restored_analysis_channels=restored_analysis_channels,
            restored_analysis_column_channels=restored_analysis_column_channels,
        )

    def set_channels(self, channels):
        self.condition_state.set_channels(channels)

    def set_current_round(self, round_number):
        self.condition_state.set_current_round(round_number)

    def set_condition_configs(self, condition_configs, *, queue_catalog=None):
        self.condition_state.set_condition_configs(condition_configs, queue_catalog=queue_catalog)

    def refresh_condition_configs(self, condition_configs, *, queue_catalog=None, preserve_results=False):
        return self.condition_state.refresh_condition_configs(
            condition_configs, queue_catalog=queue_catalog, preserve_results=preserve_results)

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
