from dataclasses import replace

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor, QImage

from consts import ui_style_const
from base.video.models import Event, EventKind, VideoState, VideoStatus
from ui.video_demo import configure_demo_font
from ui.video_monitor_widget import VideoCanvas, VideoMonitorWidget, format_elapsed
from ui.video_controller import VideoServiceBridge


class StubService:
    def __init__(self):
        self.status = VideoStatus(connection="ready")
        self.is_closed = False
        self.calls = []

    def latest_preview(self, sequence=0):
        return None

    def start_recording(self):
        self.calls.append("start")

    def stop_recording(self):
        self.calls.append("stop")


def test_runtime_shedding_keeps_recording_ui_without_warning(ui_qapp, tmp_path):
    import av
    import numpy as np

    from base.video.models import Command, CommandKind
    from base.video.runtime import VideoRuntime
    from unit_test.video.test_recording import config_for
    from unit_test.video.test_runtime import SyntheticCapture

    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime._command(Command(CommandKind.START, 1, "start", "ui-shedding"))
    runtime.accept_frames = True
    state = VideoState(1)
    state.apply(Event(EventKind.READY, 1, 1, 1))
    state.request_start("ui-shedding")
    state.apply(Event(EventKind.STARTED, 1, 2, 2, "ui-shedding"))
    service = StubService()
    service.status = state.status
    bridge = VideoServiceBridge(service)
    bridge._timer.stop()
    widget = VideoMonitorWidget(bridge)
    image = QImage(320, 180, QImage.Format_RGB888)
    image.fill(QColor("#6997AE"))
    widget.canvas.set_image(image)
    widget.update_status(state.status, 3)
    before = (widget.canvas.message, widget.record_button.toolTip(), widget.record_button.text())
    try:
        frame = av.VideoFrame.from_ndarray(np.zeros((90, 160, 3), dtype=np.uint8), format="rgb24")
        for i in range(100):
            runtime.on_frame(frame, 100 + i / 30)
        runtime._report_diagnostics()
        assert runtime.admission.dropped > 0
        assert runtime.events.empty()  # No failure or recovery event reaches the UI.
        widget.update_status(state.status, 6)
        assert (widget.canvas.message, widget.record_button.toolTip(), widget.record_button.text()) == before
        assert widget.record_button.text() == "停止录像"
        assert widget.record_button.isEnabled() and not widget.record_indicator.isHidden()
        assert widget.timer_label.text() == "00:00:06"
    finally:
        widget.close()


def test_camera_failure_and_recovery_keep_recording_error_visible(ui_qapp, tmp_path):
    configure_demo_font(ui_qapp)
    service = StubService()
    bridge = VideoServiceBridge(service)
    widget = VideoMonitorWidget(bridge)
    state = VideoState(1)
    state.apply(Event(EventKind.READY, 1, 1, 1))
    state.request_start("s1")
    state.apply(Event(EventKind.RECORDING_FAILED, 1, 2, 2, "s1", detail="MP4写入拒绝访问"))
    state.apply(Event(EventKind.OFFLINE, 1, 3, 3, detail="摄像头解码失败"))
    try:
        widget.resize(568, 350)
        widget.show()
        widget.update_status(state.status, 0)
        ui_qapp.processEvents()
        assert "录像失败：MP4写入拒绝访问" in widget.canvas.message
        assert "摄像头解码失败" in widget.canvas.message
        assert "录像失败：MP4写入拒绝访问" in widget.record_button.toolTip()
        assert "摄像头解码失败" in widget.record_button.toolTip()
        assert widget.grab().save(str(tmp_path / "separate-video-errors.png"))
        state.apply(Event(EventKind.READY, 1, 4, 4))
        widget.update_status(state.status, 0)
        assert widget.record_button.toolTip() == "录像失败：MP4写入拒绝访问"
    finally:
        widget.close()
        bridge._timer.stop()


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


def test_manual_buttons_and_compact_header(ui_qapp):
    configure_demo_font(ui_qapp)
    service = StubService()
    bridge = VideoServiceBridge(service)
    # This layout test supplies elapsed time explicitly; live polling would overwrite it.
    bridge._timer.stop()
    widget = VideoMonitorWidget(bridge, simulation=True)
    widget.resize(420, 300)
    widget.show()
    ui_qapp.processEvents()
    try:
        widget.record_button.click()
        assert service.calls == ["start"]
        service.status = replace(service.status, recording="recording", record_intent=True, started_at=1)
        widget.update_status(service.status, 360001)
        ui_qapp.processEvents()
        assert widget.timer_label.text() == "100:00:01"
        assert widget.record_indicator.isVisible()
        assert widget.record_indicator.palette().color(widget.record_indicator.foregroundRole()).name() == "#63e6a2"
        assert widget.timer_label.palette().color(widget.timer_label.foregroundRole()).name() == "#ffffff"
        assert widget.record_button.text() == "停止录像"
        assert widget.record_button.parent() is widget.header
        assert widget.more_button.parent() is widget.header
        assert widget.title_label.geometry().right() < widget.timer_label.geometry().left()
        assert widget.timer_label.geometry().right() < widget.record_button.geometry().left()
        assert widget.record_button.geometry().right() < widget.more_button.geometry().left()
        assert widget.more_button.geometry().right() < widget.width()
        assert widget.layout().count() == 2  # Header + picture, no lower information rows.
        widget.record_button.click()
        assert service.calls == ["start", "stop"]
    finally:
        widget.close()
        bridge._timer.stop()


def test_recovering_keeps_stop_and_clears_stale_picture(ui_qapp):
    bridge = VideoServiceBridge(StubService())
    widget = VideoMonitorWidget(bridge)
    try:
        image = QImage(10, 10, QImage.Format_RGB888)
        image.fill(0)
        widget.canvas.set_image(image)
        state = VideoStatus(connection="reconnecting", recording="recovering", record_intent=True)
        widget.update_status(state, 55)
        assert widget.canvas.image.isNull()
        assert widget.canvas.message
        assert widget.record_button.isEnabled()
        assert widget.record_button.text() == "停止录像"
        assert widget.record_indicator.isHidden()
        widget.update_status(replace(state, recording="stopping", record_intent=False), 55)
        assert widget.record_button.text() == "正在保存…"
        assert widget.record_indicator.isHidden()
        assert not widget.record_button.isEnabled()
    finally:
        widget.close()
        bridge._timer.stop()


@pytest.mark.parametrize("outcome", ["recording", "failed"])
def test_preparing_button_keeps_preview_without_timer(ui_qapp, tmp_path, outcome):
    configure_demo_font(ui_qapp)
    bridge = VideoServiceBridge(StubService())
    bridge._timer.stop()
    widget = VideoMonitorWidget(bridge)
    widget.resize(440, 350)
    picture = QImage(320, 180, QImage.Format_RGB888)
    picture.fill(QColor("#6997AE"))
    widget.canvas.set_image(picture)
    status = VideoStatus(connection="ready", recording="starting", record_intent=True)
    widget.show()
    try:
        widget.update_status(status, 99)
        ui_qapp.processEvents()
        assert widget.record_button.text() == "正在准备…"
        assert not widget.record_button.isEnabled()
        assert not widget.timer_label.isVisible()
        assert not widget.record_indicator.isVisible()
        assert widget.canvas.image == picture
        assert not widget.canvas.message
        widget.record_button.click()
        assert not bridge.service.calls
        screenshot = tmp_path / "video-preparing.png"
        assert widget.grab().save(str(screenshot))
        print(f"\nVideo preparing screenshot: {screenshot}")
        widget.update_status(replace(
            status, recording=outcome, record_intent=outcome == "recording",
            started_at=1 if outcome == "recording" else None,
            recording_detail="编码器初始化失败" if outcome == "failed" else "",
        ), 0)
        assert widget.record_button.isEnabled()
        assert widget.record_button.text() == ("停止录像" if outcome == "recording" else "开始录像")
        assert widget.timer_label.isVisible() == (outcome == "recording")
        assert widget.record_indicator.isVisible() == (outcome == "recording")
        if outcome == "failed":
            assert "编码器初始化失败" in widget.canvas.message
    finally:
        widget.close()


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
        assert not widget.directory_action.isEnabled()
    finally:
        widget.close()
        bridge._timer.stop()


def test_more_menu_style_geometry_and_actions(ui_qapp, tmp_path):
    configure_demo_font(ui_qapp)
    bridge = VideoServiceBridge(StubService())
    widget = VideoMonitorWidget(bridge)
    menu = widget.more_button.menu()
    calls = []
    widget.directory_requested.connect(lambda: calls.append("directory"))
    try:
        widget.resize(568, 350)
        widget.show()
        menu.popup(widget.more_button.mapToGlobal(QPoint(0, widget.more_button.height())))
        ui_qapp.processEvents()
        assert menu.font().pixelSize() == 13
        assert widget.settings_action.text() == "摄像头设置"
        assert widget.directory_action.text() == "打开录像文件夹"
        assert not widget.directory_action.isEnabled()
        for action in menu.actions():
            geometry = menu.actionGeometry(action)
            assert geometry.height() >= 32
            assert geometry.width() >= menu.fontMetrics().horizontalAdvance(action.text()) + 24
        menu.setActiveAction(widget.settings_action)
        ui_qapp.processEvents()
        screenshot = tmp_path / "video-more-menu.png"
        assert menu.grab().save(str(screenshot))
        print(f"\nVideo menu screenshot: {screenshot}")
        widget.directory_action.setEnabled(True)
        widget.directory_action.trigger()
        assert calls == ["directory"]
    finally:
        menu.close()
        widget.close()
        bridge._timer.stop()


def test_elapsed_does_not_wrap_after_24_hours_or_31_days():
    assert format_elapsed(3600) == "01:00:00"
    assert format_elapsed(24 * 3600) == "24:00:00"
    assert format_elapsed(31 * 24 * 3600 + 1) == "744:00:01"


@pytest.mark.parametrize("width", [400, 568])
@pytest.mark.parametrize("mode", ["disabled", "recording", "failed"])
def test_video_card_matches_light_theme_and_keeps_controls_readable(ui_qapp, tmp_path, width, mode):
    configure_demo_font(ui_qapp)
    service = StubService()
    service.status = VideoStatus(
        connection="disabled" if mode == "disabled" else "ready",
        recording=mode if mode != "disabled" else "idle",
        record_intent=mode == "recording",
        recording_detail="磁盘空间不足，已有录像已保留" if mode == "failed" else "",
    )
    bridge = VideoServiceBridge(service)
    widget = VideoMonitorWidget(bridge)
    widget.resize(width, 350)
    try:
        if mode != "disabled":
            image = QImage(320, 180, QImage.Format_RGB888)
            image.fill(QColor("#6997AE"))
            widget.canvas.set_image(image)
        widget.update_status(service.status, 31 * 24 * 3600)
        widget.show()
        ui_qapp.processEvents()
        assert widget.header.height() == 34
        assert widget.title_label.font().pixelSize() == 16
        assert widget.title_label.font().bold()
        assert widget.record_button.font().pixelSize() == 13
        assert widget.record_button.height() == widget.more_button.height() == 24
        assert widget.canvas.mapTo(widget, QPoint()).x() == 10
        assert widget.canvas.width() == widget.width() - 20
        assert widget.title_label.geometry().right() < widget.record_button.geometry().left()
        assert widget.record_button.geometry().right() < widget.more_button.geometry().left()
        assert widget.more_button.geometry().right() < width
        if mode == "recording":
            assert widget.timer_label.fontMetrics().horizontalAdvance(widget.timer_label.text()) <= widget.timer_label.width()
        pixels = widget.canvas.grab().toImage()
        if mode == "disabled":
            assert not widget.canvas.warning
            # No dark full-width strip in the neutral empty state.
            assert pixels.pixelColor(15, pixels.height() // 2).lightness() > 200
        else:
            assert not widget.canvas.image.isNull()
            assert widget.canvas.warning == (mode == "failed")
        header_pixels = widget.header.grab().toImage()
        assert header_pixels.pixelColor(2, 20).name().upper() == ui_style_const.COLOR_PRIMARY
        screenshot = tmp_path / f"video-card-{width}-{mode}.png"
        assert widget.grab().save(str(screenshot))
        print(f"\nVideo style screenshot: {screenshot}")
    finally:
        bridge._timer.stop()
        widget.close()
