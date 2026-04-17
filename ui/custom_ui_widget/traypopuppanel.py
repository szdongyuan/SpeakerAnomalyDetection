import sys

from PyQt5.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QParallelAnimationGroup, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class TrayPopupPanel(QFrame):
    """
    托盘风格弹出面板（类似 Windows 托盘“隐藏图标”面板的体验）

    - Qt.Popup：点击面板外自动关闭
    - 无边框 + 阴影 + 圆角
    """

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 注意：即使不使用 QGraphicsDropShadowEffect，Windows/Qt 也可能对 Popup 窗口加“系统阴影”
        # Qt.NoDropShadowWindowHint 用于关闭这层系统阴影（不同平台/主题下效果可能不同）
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        # 关键：让弹窗外层背景透明，否则会出现矩形白底“托着”圆角卡片，导致圆角看起来像直角
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setObjectName("trayPopupPanel")

        # 内容容器：这里不使用阴影效果，仅做圆角卡片
        self._card = QFrame(self)
        self._card.setObjectName("trayPopupCard")
        self._card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.in_device: QLabel = None
        self.out_device: QLabel = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._card)
        self.in_device = QLabel()
        self.out_device = QLabel()

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        in_device_layout = QHBoxLayout()
        in_device_layout.setSpacing(8)
        in_device_layout.addWidget(QLabel("输入设备："))
        in_device_layout.addWidget(self.in_device)
        in_device_layout.addStretch()

        out_device_layout = QHBoxLayout()
        out_device_layout.setSpacing(8)
        out_device_layout.addWidget(QLabel("输出设备："))
        out_device_layout.addWidget(self.out_device)
        out_device_layout.addStretch()

        card_layout.addLayout(in_device_layout)
        card_layout.addLayout(out_device_layout)
        card_layout.addStretch(1)
        self.refresh_size_from_content()

    def refresh_size_from_content(self):
        self._card.adjustSize()
        card_size = self._card.sizeHint()
        self._card.setFixedSize(card_size)
        self.setFixedSize(card_size)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QFrame#trayPopupCard {
                background: rgb(245, 248, 250);
                border-radius: 12px;
                border: 1px solid #d9d9d9;
            }
            """
        )

    def hideEvent(self, event):
        super().hideEvent(event)
        self.closed.emit()


class TrayPopupButton(QToolButton):
    """点击后弹出 TrayPopupPanel 的按钮（定位到按钮下方，并做屏幕边界修正）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("设备")
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)

        self._panel = TrayPopupPanel()
        self._panel.closed.connect(self._on_panel_closed)

        self.clicked.connect(self.toggle_popup)

        # 动画相关
        self._anim_group = None
        self._is_closing_anim = False
        self._hover_sync_timer = QTimer(self)
        self._hover_sync_timer.setInterval(50)
        self._hover_sync_timer.timeout.connect(self._refresh_hot_state)
        self.setProperty("hot", False)

    def toggle_popup(self):
        if self._panel.isVisible():
            self.close_popup(animated=True)
        else:
            self.open_popup(animated=True)

    def open_popup(self, animated=True):
        self._panel.refresh_size_from_content()
        # 计算理想位置：按钮下方居中
        panel_w = self._panel.width()
        panel_h = self._panel.height()

        anchor_global = self.mapToGlobal(QPoint(0, self.height()))
        x = anchor_global.x() + (self.width() - panel_w) // 2
        y = anchor_global.y() + 6

        # 屏幕边界修正（使用可用区域，避免被任务栏遮挡）
        screen = QApplication.screenAt(anchor_global)
        if screen is None:
            screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()

        x = max(avail.left() + 8, min(x, avail.right() - panel_w - 8))
        # 如果下方放不下，就放到按钮上方
        if y + panel_h > avail.bottom() - 8:
            above_y = self.mapToGlobal(QPoint(0, 0)).y() - panel_h - 6
            y = max(avail.top() + 8, above_y)

        # 设置初始位置并显示
        end_pos = QPoint(x, y)
        start_pos = QPoint(x, y - (8 if animated else 0))

        self._panel.setWindowOpacity(1.0)
        self._panel.move(start_pos)
        self._panel.show()
        self._panel.raise_()
        self._hover_sync_timer.start()
        self._refresh_hot_state()

        if animated:
            self._start_open_animation(start_pos, end_pos)

    def close_popup(self, animated=True):
        if not self._panel.isVisible():
            return
        if not animated:
            self._panel.hide()
            return
        self._start_close_animation()

    def _start_open_animation(self, start_pos: QPoint, end_pos: QPoint):
        self._stop_anims()
        self._is_closing_anim = False

        geom_anim = QPropertyAnimation(self._panel, b"pos")
        geom_anim.setStartValue(start_pos)
        geom_anim.setEndValue(end_pos)
        geom_anim.setDuration(160)
        geom_anim.setEasingCurve(QEasingCurve.OutCubic)

        op_anim = QPropertyAnimation(self._panel, b"windowOpacity")
        op_anim.setStartValue(0.0)
        op_anim.setEndValue(1.0)
        op_anim.setDuration(160)
        op_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self._panel)
        group.addAnimation(geom_anim)
        group.addAnimation(op_anim)
        group.start()
        self._anim_group = group

    def _start_close_animation(self):
        self._stop_anims()
        self._is_closing_anim = True

        start_pos = self._panel.pos()
        end_pos = QPoint(start_pos.x(), start_pos.y() - 6)

        geom_anim = QPropertyAnimation(self._panel, b"pos")
        geom_anim.setStartValue(start_pos)
        geom_anim.setEndValue(end_pos)
        geom_anim.setDuration(120)
        geom_anim.setEasingCurve(QEasingCurve.InCubic)

        op_anim = QPropertyAnimation(self._panel, b"windowOpacity")
        op_anim.setStartValue(self._panel.windowOpacity())
        op_anim.setEndValue(0.0)
        op_anim.setDuration(120)
        op_anim.setEasingCurve(QEasingCurve.InCubic)

        group = QParallelAnimationGroup(self._panel)
        group.addAnimation(geom_anim)
        group.addAnimation(op_anim)
        group.finished.connect(self._finish_close_animation)
        group.start()
        self._anim_group = group

    def _finish_close_animation(self):
        # 避免 “点击外部关闭” 触发 hideEvent 时被我们重复 hide
        if self._panel.isVisible():
            self._panel.hide()
        self._panel.setWindowOpacity(1.0)
        self._is_closing_anim = False

    def _stop_anims(self):
        if self._anim_group is not None:
            self._anim_group.stop()
            self._anim_group.deleteLater()
            self._anim_group = None

    def _on_panel_closed(self):
        self._hover_sync_timer.stop()
        self._refresh_hot_state()

    def _set_hot(self, hot: bool):
        hot = bool(hot)
        if bool(self.property("hot")) == hot:
            return
        self.setProperty("hot", hot)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _refresh_hot_state(self):
        hovered = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_hot(hovered)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._refresh_hot_state()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._refresh_hot_state()

    def set_in_device(self, device: str):
        self._panel.in_device.setText(device)
        self._panel.refresh_size_from_content()

    def set_out_device(self, device: str):
        self._panel.out_device.setText(device)
        self._panel.refresh_size_from_content()

def main():
    app = QApplication(sys.argv)

    w = QWidget()
    w.setWindowTitle("托盘弹出面板按钮 Demo")
    # w.setStyleSheet("background: #f2f2f2;")

    layout = QVBoxLayout(w)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(12)

    tip = QLabel("点击下面按钮")
    tip.setWordWrap(True)
    tip.setStyleSheet("color: #333;")
    layout.addWidget(tip)

    btn = TrayPopupButton()
    btn.setFixedWidth(120)
    layout.addWidget(btn, alignment=Qt.AlignCenter)

    layout.addStretch(1)

    w.resize(520, 260)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
