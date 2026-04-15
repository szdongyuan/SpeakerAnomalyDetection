from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ui.sequence.motor_panel_common import MotorSectionCard


class MotorSummaryPanel(QWidget):
    def __init__(self, summary_widget: QWidget, parent=None):
        super().__init__(parent)
        self.summary_widget = summary_widget
        self._init_ui()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        card = MotorSectionCard("汇总信息")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.content_layout.setContentsMargins(6, 4, 6, 4)
        card.content_layout.setSpacing(0)

        if self.summary_widget is not None:
            if hasattr(self.summary_widget, "set_mode_switch_visible"):
                self.summary_widget.set_mode_switch_visible(False)
            self.summary_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.content_layout.addWidget(self.summary_widget)
            card.content_layout.addSpacing(42)

        layout.addWidget(card)
        self.setLayout(layout)
