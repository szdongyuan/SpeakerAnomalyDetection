
import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget

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
            controls = [panel.title_label, panel.more_button]
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


def test_more_menu_and_simulation_disclosure(ui_qapp):
    bridge = VideoServiceBridge(StubService())
    widget = VideoMonitorWidget(bridge, simulation=True)
    calls = []
    widget.settings_requested.connect(lambda: calls.append("settings"))
    try:
        assert "模拟" in widget.title_label.text()
        assert widget.canvas.simulation
        widget.settings_action.trigger()
        assert calls == ["settings"]
        assert widget.more_button.menu().actions() == [widget.settings_action]
    finally:
        widget.close()
        bridge._timer.stop()


def test_more_menu_style_geometry_and_actions(ui_qapp, tmp_path):
    configure_demo_font(ui_qapp)
    bridge = VideoServiceBridge(StubService())
    widget = VideoMonitorWidget(bridge)
    menu = widget.more_button.menu()
    calls = []
    try:
        widget.resize(568, 350)
        widget.show()
        menu.popup(widget.more_button.mapToGlobal(QPoint(0, widget.more_button.height())))
        ui_qapp.processEvents()
        assert menu.font().pixelSize() == 13
        assert widget.settings_action.text() == "摄像头设置"
        assert widget.more_button.menu().actions() == [widget.settings_action]
        for action in menu.actions():
            geometry = menu.actionGeometry(action)
            assert geometry.height() >= 32
            assert geometry.width() >= menu.fontMetrics().horizontalAdvance(action.text()) + 24
        menu.setActiveAction(widget.settings_action)
        ui_qapp.processEvents()
        screenshot = tmp_path / "video-more-menu.png"
        assert menu.grab().save(str(screenshot))
        print(f"\nVideo menu screenshot: {screenshot}")
    finally:
        menu.close()
        widget.close()
        bridge._timer.stop()
