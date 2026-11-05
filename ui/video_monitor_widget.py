"""Compact video card shared by the main application and the simulation demo."""

from PyQt5.QtCore import QRect, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QToolButton, QVBoxLayout, QWidget

from consts import ui_style_const


class VideoCanvas(QWidget):
    def __init__(self, parent=None, *, simulation=False):
        super().__init__(parent)
        self.image = QImage()
        self.message = "正在连接摄像头…"
        self.warning = False
        self.simulation = simulation
        self.setMinimumSize(280, 180)

    def set_image(self, image):
        self.image = image
        self.update()

    def set_message(self, message, *, clear_image=False, warning=False):
        self.message = message
        self.warning = warning
        self.setToolTip(message)
        if clear_image:
            self.image = QImage()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont(ui_style_const.MAIN_UI_SMALL_FONT_FAMILY_NAME)
        font.setPixelSize(13)
        painter.setFont(font)
        if self.image.isNull():
            self._paint_placeholder(painter)
        else:
            painter.fillRect(self.rect(), QColor("#E7EDF3"))
            size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
            left = (self.width() - size.width()) // 2
            top = (self.height() - size.height()) // 2
            painter.drawImage(left, top, self.image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if self.message:
                notice = QRect(10, self.height() - 76, self.width() - 20, 56)
                painter.fillRect(notice, QColor("#FFF3DB") if self.warning else QColor("#EDF3F8"))
                painter.setPen(QColor("#80551C") if self.warning else QColor("#536B80"))
                painter.drawText(notice.adjusted(10, 4, -10, -4), Qt.AlignCenter | Qt.TextWordWrap, self.message)
        if self.simulation:
            painter.setPen(QColor("#657789") if self.image.isNull() else QColor("#eef5fc"))
            painter.drawText(
                self.rect().adjusted(8, 8, -8, -12), Qt.AlignBottom | Qt.AlignHCenter,
                "模拟画面 · 不连接摄像头",
            )
        painter.setPen(QColor("#D2DCE5"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _paint_placeholder(self, painter):
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#E7EDF3"))
        gradient.setColorAt(1, QColor("#DCE5EC"))
        painter.fillRect(self.rect(), gradient)
        painter.setRenderHint(QPainter.Antialiasing)
        center_x, center_y = self.width() / 2, self.height() / 2
        painter.setPen(QPen(QColor("#7A8B9C"), 2))
        painter.drawRoundedRect(QRectF(center_x - 15, center_y - 30, 30, 21), 3, 3)
        painter.drawRect(QRectF(center_x - 6, center_y - 35, 12, 5))
        painter.drawEllipse(QRectF(center_x - 5, center_y - 25, 10, 10))
        painter.setPen(QColor("#80551C") if self.warning else QColor("#657789"))
        painter.drawText(
            QRect(20, int(center_y + 4), self.width() - 40, int(center_y - 30)),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            self.message or "等待视频画面",
        )


class VideoMonitorWidget(QWidget):
    settings_requested = pyqtSignal()

    def __init__(self, bridge, parent=None, *, simulation=False,
                 title_font_family=ui_style_const.UI_FONT_FAMILY):
        super().__init__(parent)
        self.bridge = bridge
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.header = QFrame(self)
        self.header.setObjectName("videoHeader")
        self.header.setFixedHeight(34)
        self.header.setStyleSheet(
            f"QFrame#videoHeader {{background:{ui_style_const.COLOR_PRIMARY}; border:none;"
            "border-top-left-radius:4px; border-top-right-radius:4px;}"
            "QLabel {color:white; background:transparent; border:none;}"
            "QToolButton {color:white; background:transparent; border:1px solid #9cbbe0;"
            f"font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; font-size:13px;"
            "font-weight:normal; border-radius:3px; padding:0 7px;}"
            "QToolButton:hover {background:#5389c9;}"
            "QToolButton::menu-indicator {image:none; width:0px;}"
        )
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 8, 0)
        header_layout.setSpacing(8)
        self.title_label = QLabel("视频监控 · 模拟" if simulation else "视频监控")
        self.title_label.setStyleSheet(
            f"font-family:{title_font_family}; font-size:16px; font-weight:bold;"
        )
        self.more_button = QToolButton()
        self.more_button.setText("…")
        self.more_button.setFixedSize(28, 24)
        self.more_button.setToolTip("更多")
        self.more_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.more_button)
        menu.setStyleSheet(
            f"QMenu {{font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; font-size:13px;"
            "color:#30485E; background:white; border:1px solid #C5D4E3; padding:4px 0;}"
            "QMenu::item {min-height:24px; padding:4px 12px; background:transparent;}"
            "QMenu::item:selected {background:#E7F0FB; color:#234E80;}"
            "QMenu::item:disabled {color:#8998A8; background:transparent;}"
        )
        self.settings_action = menu.addAction("摄像头设置")
        self.settings_action.triggered.connect(self.settings_requested.emit)
        self.more_button.setMenu(menu)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.more_button)
        root.addWidget(self.header)
        self.content = QFrame(self)
        self.content.setObjectName("motorSectionContent")
        self.content.setStyleSheet(ui_style_const.motor_section_content_style)
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        self.canvas = VideoCanvas(self.content, simulation=simulation)
        content_layout.addWidget(self.canvas)
        root.addWidget(self.content, 1)
        bridge.status_changed.connect(self.update_status)
        bridge.preview_changed.connect(self.canvas.set_image)
        self.update_status(bridge.service.status)

    def update_status(self, status):
        if not self.bridge.preview_requested and status.connection != "unavailable":
            message = "预览已关闭"
        else:
            message = {
                "ready": "", "connecting": "正在连接摄像头…", "closed": "视频服务已关闭",
                "unavailable": "摄像头不可用", "reconnecting": "未检测到摄像头，请连接设备。",
            }.get(status.connection, "摄像头待配置")
        self.canvas.set_message(message, clear_image=bool(message),
                                warning=status.connection in {"unavailable", "reconnecting"})
        self.canvas.setToolTip(status.connection_detail or message)
