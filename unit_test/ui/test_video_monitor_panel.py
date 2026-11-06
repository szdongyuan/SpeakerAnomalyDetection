
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from consts import ui_style_const
from base.video.models import VideoStatus
from ui.video_demo import configure_demo_font
from ui.video_monitor_widget import VideoCanvas, VideoMonitorWidget
from ui.video_controller import VideoServiceBridge
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel


class StubService:
    def __init__(self):
        self.status = VideoStatus(connection="ready")
        self.is_closed = False
        self.calls = []

    def latest_preview(self, sequence=0):
        return None

@pytest.mark.parametrize(
    "size, margin_points",
    [((548, 296), [(4, 148), (543, 148)]),
     ((400, 300), [(200, 4), (200, 295)])],
)
def test_preview_letterbox_uses_light_background(ui_qapp, tmp_path, size, margin_points):
    canvas = VideoCanvas()
    canvas.resize(*size)
    frame = QImage(320, 180, QImage.Format_RGB888)
    frame.fill(QColor("#6997AE"))
    canvas.set_image(frame)
    canvas.set_message("")
    canvas.show()
    ui_qapp.processEvents()
    try:
        pixels = canvas.grab().toImage()
        for x, y in margin_points:
            assert pixels.pixelColor(x, y).name() == "#e7edf3"
        assert pixels.pixelColor(size[0] // 2, size[1] // 2).name() == "#6997ae"
        assert canvas.image == frame
        screenshot = tmp_path / "video-preview-light-margins.png"
        assert pixels.save(str(screenshot))
        print(f"\nVideo preview screenshot: {screenshot}")
    finally:
        canvas.close()


@pytest.mark.parametrize("state", ["ready", "reconnecting"])
def test_live_video_fits_sidebar_after_window_shrinks(ui_qapp, tmp_path, state):
    configure_demo_font(ui_qapp)
    service = StubService()
    service.status = VideoStatus(connection=state)
    bridge = VideoServiceBridge(service)
    bridge._timer.stop()
    bridge.preview_requested = False
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    splitter = QSplitter(Qt.Horizontal)
    sidebar = QSplitter(Qt.Vertical)
    task = QWidget()
    task.setMinimumWidth(340)
    wrapper = MotorVideoMonitorPanel()
    wrapper.setMinimumWidth(340)
    sidebar.addWidget(task)
    sidebar.addWidget(wrapper)
    sidebar.setChildrenCollapsible(False)
    workspace = QWidget()
    workspace.setMinimumWidth(650)
    splitter.addWidget(sidebar)
    splitter.addWidget(workspace)
    splitter.setChildrenCollapsible(False)
    layout.addWidget(splitter)
    panel = wrapper.bind_bridge(bridge)
    panel.update_status(service.status)
    try:
        for width in (1600, 1024, 1600, 1024):
            window.resize(width, 600)
            window.show()
            sidebar_width = 340 if width == 1024 else 480
            splitter.setSizes([sidebar_width, width - sidebar_width - splitter.handleWidth()])
            ui_qapp.processEvents()
            assert panel.width() == wrapper.width()
            controls = [panel.title_label, panel.settings_button]
            visible = [control for control in controls if control.isVisible()]
            for control in visible:
                assert control.geometry().right() < wrapper.width()
                assert control.width() >= control.fontMetrics().horizontalAdvance(control.text())
            for left, right in zip(visible, visible[1:]):
                assert left.geometry().right() < right.geometry().left()
        assert wrapper.width() == 340
        assert wrapper.grab().save(str(tmp_path / f"narrow-video-{state}.png"))
    finally:
        window.close()


def test_video_titles_use_main_font_only_when_embedded(ui_qapp):
    bridge = VideoServiceBridge(StubService())
    bridge._timer.stop()
    wrapper = MotorVideoMonitorPanel()
    standalone = VideoMonitorWidget(bridge)
    try:
        wrapper.show()
        ui_qapp.processEvents()
        for label in wrapper.header.findChildren(QLabel):
            font = label.font()
            assert (font.family(), font.pixelSize(), font.bold()) == ("Microsoft YaHei UI", 16, True)
        panel = wrapper.bind_bridge(bridge)
        standalone.show()
        ui_qapp.processEvents()
        for widget, family in ((panel, "Microsoft YaHei UI"), (standalone, "SimSun")):
            font = widget.title_label.font()
            assert (font.family(), font.pixelSize(), font.bold()) == (family, 16, True)
    finally:
        wrapper.close()
        standalone.close()


def test_video_status_uses_pixel_font_in_paint_path(ui_qapp, monkeypatch):
    from PyQt5.QtGui import QPainter
    from ui import video_monitor_widget

    seen = []

    class PainterSpy(QPainter):
        def drawText(self, rect, flags, text):
            seen.append((self.font().family(), self.font().pixelSize(), text))
            return super().drawText(rect, flags, text)

    monkeypatch.setattr(video_monitor_widget, "QPainter", PainterSpy)
    canvas = VideoCanvas()
    try:
        canvas.set_message("Camera offline")
        canvas.show()
        assert not canvas.grab().isNull()
        assert seen and set(seen) == {("Microsoft YaHei UI", 13, "Camera offline")}
    finally:
        canvas.close()


def test_settings_button_emits_one_intent_for_the_standalone_subscriber(ui_qapp):
    bridge = VideoServiceBridge(StubService())
    widget = VideoMonitorWidget(bridge, simulation=True)
    requests = QSignalSpy(widget.settings_requested)
    try:
        widget.resize(340, 220)
        widget.show()
        ui_qapp.processEvents()
        assert "模拟" in widget.title_label.text()
        assert widget.canvas.simulation
        button = widget.settings_button
        assert button.isVisible()
        assert button.text() == "设置"
        assert len(requests) == 0
        QTest.mouseClick(button, Qt.LeftButton)
        assert len(requests) == 1
    finally:
        widget.close()
        bridge._timer.stop()


def test_settings_button_style_and_geometry(ui_qapp, tmp_path):
    configure_demo_font(ui_qapp)
    bridge = VideoServiceBridge(StubService())
    wrapper = MotorVideoMonitorPanel()
    widget = wrapper.bind_bridge(bridge)
    try:
        wrapper.resize(340, 350)
        wrapper.show()
        ui_qapp.processEvents()
        button = widget.settings_button
        assert button.isVisible()
        assert (button.font().family(), button.font().pixelSize()) == ("Microsoft YaHei UI", 13)
        assert button.text() == "设置"
        assert button.width() >= button.fontMetrics().horizontalAdvance(button.text()) + 16
        assert button.geometry().right() < widget.header.width()
        assert widget.header.grab().save(str(tmp_path / "video-settings-button.png"))
    finally:
        wrapper.close()
        bridge._timer.stop()
