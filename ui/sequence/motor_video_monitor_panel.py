from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class _CameraOutline(QWidget):
    def sizeHint(self):
        return QSize(34, 28)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#7A8B9C"), 2))
        body = QRectF(4, 8, max(1, self.width() - 8), max(1, self.height() - 12))
        painter.drawRoundedRect(body, 3, 3)
        painter.drawRect(QRectF(self.width() * 0.34, 4, self.width() * 0.32, 5))
        lens_size = min(body.width(), body.height()) * 0.45
        painter.drawEllipse(
            QRectF(
                (self.width() - lens_size) / 2,
                body.top() + (body.height() - lens_size) / 2,
                lens_size,
                lens_size,
            )
        )


class MotorVideoMonitorPanel(QWidget):
    """Video-monitor card with a stable placeholder for later camera integration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.card = MotorSectionCard("")
        self.card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.header = QFrame(self.card)
        self.header.setObjectName("videoMonitorHeader")
        self.header.setFixedHeight(34)
        self.header.setStyleSheet(
            f"QFrame#videoMonitorHeader {{ background:{ui_style_const.COLOR_PRIMARY}; "
            "border:none; border-top-left-radius:4px; border-top-right-radius:4px; }}"
        )
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(8)
        title = QLabel("视频监控")
        preview = QLabel("2K预览")
        for label in (title, preview):
            label.setStyleSheet(self._header_label_style())
            label.setAlignment(Qt.AlignVCenter)
        preview.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(preview)
        self.card.content_layout.addWidget(self.header)

        content = QWidget(self.card)
        content.setObjectName("motorSectionContent")
        content.setStyleSheet(ui_style_const.motor_section_content_style)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)

        self.video_placeholder = QFrame(content)
        self.video_placeholder.setObjectName("videoMonitorPlaceholder")
        self.video_placeholder.setMinimumHeight(150)
        self.video_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_placeholder.setStyleSheet(
            "QFrame#videoMonitorPlaceholder {"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #E7EDF3,stop:1 #DCE5EC);"
            "border:1px solid #D2DCE5; border-radius:0;"
            "}"
        )
        placeholder_layout = QVBoxLayout(self.video_placeholder)
        placeholder_layout.setContentsMargins(12, 12, 12, 12)
        placeholder_layout.setSpacing(7)
        placeholder_layout.addStretch(1)

        camera_icon = _CameraOutline(self.video_placeholder)
        camera_icon.setFixedSize(camera_icon.sizeHint())
        placeholder_layout.addWidget(camera_icon, alignment=Qt.AlignHCenter)

        live_label = QLabel("实时视频画面")
        live_label.setObjectName("videoMonitorLiveLabel")
        live_label.setAlignment(Qt.AlignCenter)
        live_label.setStyleSheet(self._placeholder_text_style("#657789", 15))
        placeholder_layout.addWidget(live_label)

        status_label = QLabel("摄像头待接入")
        status_label.setObjectName("videoMonitorStatusLabel")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet(self._placeholder_text_style("#657789", 14))
        placeholder_layout.addWidget(status_label)
        placeholder_layout.addStretch(1)

        content_layout.addWidget(self.video_placeholder, stretch=1)
        self.card.content_layout.addWidget(content, stretch=1)
        root.addWidget(self.card, stretch=1)

    @staticmethod
    def _header_label_style():
        return (
            "QLabel { background:transparent; border:none; color:#FFFFFF; "
            f"font-family:{ui_style_const.UI_FONT_FAMILY}; font-size:16px; font-weight:bold; }}"
        )

    @staticmethod
    def _placeholder_text_style(color, font_size):
        return (
            "QLabel { background:transparent; border:none; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; "
            f"font-size:{font_size}px; }}"
        )
