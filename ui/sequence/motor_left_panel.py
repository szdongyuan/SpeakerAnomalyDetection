from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

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
        self.summary_panel = MotorSummaryPanel(summary_widget, self)
        self._init_ui()

    def _init_ui(self):
        self.setFixedWidth(380)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setStyleSheet(ui_style_const.motor_left_panel_style)

        layout = QVBoxLayout()
        layout.addWidget(self.mode_switch_panel)
        layout.addWidget(self.ai_result_panel)
        layout.addWidget(self.summary_panel)
        layout.addStretch(1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.setLayout(layout)

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
