from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
from ui.sequence.motor_mode_switch_panel import MotorModeSwitchPanel
from ui.sequence.motor_summary_panel import MotorSummaryPanel


class MotorDetectionLeftPanel(QWidget):
    """
    Composite left sidebar for motor detection mode.

    It delegates the top AI-result area and the bottom summary area to two
    dedicated sub-widgets so later business wiring can evolve independently.
    """

    def __init__(self, summary_widget: QWidget, parent=None):
        super().__init__(parent)
        self.mode_switch_panel = MotorModeSwitchPanel(summary_widget, self)
        self.ai_result_panel = MotorAiResultPanel(self)
        self.summary_panel = MotorSummaryPanel(summary_widget, self.mode_switch_panel, self)
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
        self.content_layout.addWidget(self.ai_result_panel)
        self.content_layout.addWidget(self.summary_panel)
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
            self._detach_section_widget(self.ai_result_panel),
            self._detach_section_widget(self.summary_panel),
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

    def reset_ai_result_panel(self):
        self.ai_result_panel.reset()

    def set_current_barcode(self, barcode: str):
        # Kept for compatibility with the existing workflow, but barcode is no
        # longer displayed in the left industrial panel.
        return None

    def set_current_timestamp(self, timestamp_text: str):
        # Kept for compatibility with the existing workflow, but timestamp is no
        # longer displayed in the left industrial panel.
        return None

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        self.ai_result_panel.set_current_stage(stage_text, tone=tone)

    def set_forward_result(self, result_text: str, tone: str = None):
        self.ai_result_panel.set_forward_result(result_text, tone=tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        self.ai_result_panel.set_reverse_result(result_text, tone=tone)

    def set_final_result(self, result_text: str, tone: str = None):
        self.ai_result_panel.set_final_result(result_text, tone=tone)

    def set_forward_scores(self, ok_score=None, ng_score=None):
        self.ai_result_panel.set_forward_scores(ok_score, ng_score)

    def set_reverse_scores(self, ok_score=None, ng_score=None):
        self.ai_result_panel.set_reverse_scores(ok_score, ng_score)
