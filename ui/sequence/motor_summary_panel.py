from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class MotorSummaryPanel(QWidget):
    def __init__(self, summary_widget: QWidget, mode_switch_panel: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.summary_widget = summary_widget
        self.mode_switch_panel = mode_switch_panel
        self._init_ui()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        card = MotorSectionCard("操作面板")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.content_layout.setContentsMargins(14, 14, 14, 14)
        card.content_layout.setSpacing(12)

        if self.mode_switch_panel is not None:
            card.content_layout.addWidget(self._create_caption("模式切换"))
            self.mode_switch_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.content_layout.addWidget(self.mode_switch_panel)
            card.content_layout.addWidget(self._create_divider())

        if self.summary_widget is not None:
            card.content_layout.addWidget(self._create_caption("汇总信息"))
            if hasattr(self.summary_widget, "set_mode_switch_visible"):
                self.summary_widget.set_mode_switch_visible(False)
            self.summary_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.content_layout.addWidget(self.summary_widget)

        layout.addWidget(card)
        self.setLayout(layout)

    @staticmethod
    def _create_caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setStyleSheet(ui_style_const.motor_result_caption_style)
        return label

    @staticmethod
    def _create_divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        divider.setFixedHeight(2)
        divider.setStyleSheet(ui_style_const.motor_result_divider_style)
        return divider
