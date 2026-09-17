import time

from base.video.preview import PreviewFrame
from base.video.service import VideoService
from base.video.models import VideoStatus
from base.video.worker import SimulationOptions, simulated_video_worker
from ui.video_demo import configure_demo_font
from ui.video_monitor_widget import VideoMonitorWidget
from ui.video_controller import VideoServiceBridge


def spin(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt video state timed out")


def test_bridge_copies_rgb_memory_and_emits_closed_once(ui_qapp):
    class Service:
        status = VideoStatus(connection="ready")
        is_closed = False

        def __init__(self):
            self.pixels = bytearray([255, 0, 0] * 4)

        def latest_preview(self, after):
            if after == 0:
                return PreviewFrame(2, 2, 1, self.pixels)
            return None

    service = Service()
    bridge = VideoServiceBridge(service)
    images, closed = [], []
    bridge.preview_changed.connect(images.append)
    bridge.closed.connect(lambda: closed.append(True))
    bridge.poll()
    service.pixels[:] = bytes(12)
    assert images[0].pixelColor(0, 0).red() == 255
    service.is_closed = True
    bridge.poll()
    bridge.poll()
    assert closed == [True]
    assert not bridge._timer.isActive()


def test_real_spawn_to_qt_preview_start_stop_and_async_exit(ui_qapp, tmp_path):
    configure_demo_font(ui_qapp)
    service = VideoService(worker_target=simulated_video_worker, worker_options=SimulationOptions())
    bridge = VideoServiceBridge(service)
    widget = VideoMonitorWidget(bridge, simulation=True)
    widget.resize(560, 350)
    closed = []
    bridge.closed.connect(lambda: closed.append(True))
    widget.show()
    try:
        service.start()
        spin(ui_qapp, lambda: widget.record_button.isEnabled() and not widget.canvas.image.isNull())
        widget.record_button.click()
        spin(ui_qapp, lambda: widget._status.recording == "recording")
        screenshot = tmp_path / "video_demo_recording.png"
        ui_qapp.processEvents()
        assert widget.grab().save(str(screenshot))
        print(f"\nVideo demo screenshot: {screenshot}")
        session = service.status.session_id
        assert session
        widget.record_button.click()
        assert widget.record_button.text() == "正在保存…"
        spin(ui_qapp, lambda: widget._status.recording == "completed")
        assert widget.record_button.text() == "开始录像"
        assert not widget.canvas.image.isNull()
        bridge.shutdown()
        spin(ui_qapp, lambda: bool(closed))
        assert service.is_closed
        assert widget._status.connection == "closed"
    finally:
        service.shutdown()
        assert service.wait_closed()
        bridge._timer.stop()
        widget.close()
