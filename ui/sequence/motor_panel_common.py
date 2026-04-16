from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

from consts import ui_style_const


class MotorSectionCard(QFrame):
    def __init__(self, title_text: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(ui_style_const.motor_section_card_style)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if title_text:
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(ui_style_const.motor_section_title_style)
            layout.addWidget(title)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        layout.addLayout(self.content_layout)
        self.setLayout(layout)


def create_field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    label.setStyleSheet(ui_style_const.motor_field_label_style)
    return label


def create_value_label() -> QLabel:
    label = QLabel("--")
    label.setAlignment(Qt.AlignCenter)
    label.setWordWrap(True)
    label.setStyleSheet(ui_style_const.motor_value_label_style)
    return label


def create_badge_label() -> QLabel:
    label = QLabel("--")
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumHeight(38)
    label.setWordWrap(True)
    label.setStyleSheet(
        ui_style_const.motor_status_badge_style + ui_style_const.motor_status_badge_pending_style
    )
    return label
