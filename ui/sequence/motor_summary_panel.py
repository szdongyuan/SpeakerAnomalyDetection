from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class MotorSummaryPanel(QWidget):
    def __init__(self, summary_widget: QWidget, mode_switch_panel: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.summary_widget = summary_widget
        self.mode_switch_panel = mode_switch_panel
        self._init_ui()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        card = MotorSectionCard("操作面板")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.content_layout.setContentsMargins(0, 0, 0, 0)
        card.content_layout.setSpacing(0)

        content_widget = QWidget(card)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(12)

        if self.mode_switch_panel is not None:
            content_layout.addWidget(self._create_caption("模式切换"))
            self.mode_switch_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            content_layout.addWidget(self.mode_switch_panel)
            content_layout.addWidget(self._create_divider())

        if self.summary_widget is not None:
            content_layout.addWidget(self._create_caption("汇总信息"))
            if hasattr(self.summary_widget, "set_mode_switch_visible"):
                self.summary_widget.set_mode_switch_visible(False)
            self.summary_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            content_layout.addWidget(self.summary_widget)

        content_layout.addStretch(1)
        content_widget.setLayout(content_layout)

        scroll_area = QScrollArea(card)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(content_widget)

        card.content_layout.addWidget(scroll_area, stretch=1)
        layout.addWidget(card, stretch=1)
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
