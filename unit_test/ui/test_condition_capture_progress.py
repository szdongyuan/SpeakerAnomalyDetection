"""Actual-frame progress stays separate from channel verdicts and waveform previews."""
from types import SimpleNamespace
from time import monotonic
import os
from pathlib import Path

import pytest
from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QThread
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtTest import QTest

from base.recording_process_protocol import RecordingProgress, RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from ui.recording_service_bridge import RecordingServiceBridge
from ui.sequence.motor_result_panel import MotorResultPanel
from ui.sequence.recording_process_context import RecordingProcessContext
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin


@pytest.fixture
def panel(ui_qapp):
    # Windows offscreen starts without system fonts; dummy glyph widths do not
    # represent the running application's layout.
    if os.name == "nt" and not QFontDatabase().families():
        font_path = Path(os.environ["WINDIR"]) / "Fonts" / "msyh.ttc"
        assert QFontDatabase.addApplicationFont(str(font_path)) >= 0
    widget = MotorResultPanel(condition_configs=[
        {"key": "first", "group_name": "端口1", "condition_name": "10V3A"},
        {"key": "second", "group_name": "端口2", "condition_name": "长档位名称10V5A"},
    ], queue_catalog={})
    widget.set_channels(list(range(5)))
    widget.resize(570, 460)
    widget.show()
    ui_qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def test_small_capture_increment_is_painted(panel, ui_qapp):
    bar = panel.rows["first"]["capture_bar"]
    painted_values = []

    class PaintObserver(QObject):
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Paint:
                painted_values.append(watched.value())
            return False

    observer = PaintObserver(bar)
    bar.installEventFilter(observer)
    panel.set_condition_capture_progress("first", 3, 30)
    QTest.qWait(50)
    assert painted_values[-1] == 100
    painted_values.clear()
    # 0.3 seconds is 1.5 pixels on the 150-pixel bar, below Qt 5's
    # default nine-pixel repaint threshold when progress text is hidden.
    panel.set_condition_capture_progress("first", 3.3, 30)
    QTest.qWait(50)
    assert bar.value() == 110
    assert 110 in painted_values


def test_capture_analysis_failure_retry_and_reset(panel):
    row = panel.rows["first"]
    panel.set_condition_result("first", "采集中")
    panel.set_condition_capture_progress("first", 15, 30)
    assert row["capture_bar"].value() == 500
    assert row["labels"]["capture_time"].text() == "15.0 / 30.0 秒"
    assert row["labels"]["progress"].text() == "通道判定：0/5"
    panel.set_condition_result("first", "数据保存中")
    assert row["capture_bar"].maximum() == 1000
    assert row["capture_bar"].value() == 500
    panel.set_condition_result("first", "待判定")
    assert row["capture_bar"].maximum() == 1000
    panel.set_condition_result("first", "分析中")
    assert row["capture_bar"].maximum() == 1000
    panel.set_condition_result("first", "分析失败", "ng")
    assert row["capture_bar"].maximum() == 1000
    assert row["capture_bar"].value() == 500
    panel.set_condition_result("first", "准备采集")
    assert row["capture_bar"].value() == 0
    panel.set_condition_capture_progress("first", 30, 30)
    for phase in ("数据保存中", "分析排队", "分析中", "分段分析"):
        panel.set_condition_result("first", phase)
        assert row["capture_bar"].maximum() == row["capture_bar"].value() == 1000
        assert row["labels"]["capture_time"].text() == "30.0 / 30.0 秒"
    panel.set_condition_result("first", "NG", "ng")
    assert row["capture_bar"].value() == 1000
    panel.reset()
    assert row["capture_bar"].value() == 0
    assert row["labels"]["capture_time"].text() == "-- / -- 秒"


def test_configured_durations_are_visible_before_start_and_survive_reset(panel):
    conditions = [
        {"key": "first", "condition_name": "10V3A", "test_queue": "short"},
        {"key": "second", "condition_name": "10V5A", "test_queue": "long"},
    ]
    catalog = {"short": {"duration": 30}, "long": {"duration": 45}}
    panel.set_condition_configs(conditions, queue_catalog=catalog)
    first, second = panel.rows["first"], panel.rows["second"]
    assert first["labels"]["capture_time"].text() == "-- / 30.0 秒"
    assert second["labels"]["capture_time"].text() == "-- / 45.0 秒"
    panel.set_condition_result("first", "准备采集")
    assert first["labels"]["capture_time"].text() == "-- / 30.0 秒"
    panel.set_condition_result("first", "采集中")
    panel.start_condition_capture("first", 30)
    assert first["labels"]["capture_time"].text() == "0.0 / 30.0 秒"
    panel.set_condition_capture_progress("first", 12, 30)
    assert panel.refresh_condition_configs(
        conditions, queue_catalog={"short": {"duration": 60}, "long": {"duration": 90}},
        preserve_results=True)
    assert first["labels"]["capture_time"].text() == "12.0 / 30.0 秒"
    assert second["labels"]["capture_time"].text() == "-- / 90.0 秒"
    panel.reset()
    assert first["labels"]["capture_time"].text() == "-- / 60.0 秒"
    assert second["labels"]["capture_time"].text() == "-- / 90.0 秒"


@pytest.mark.parametrize("width", [400, 480, 570])
def test_single_row_fits_and_switching_ports_preserves_progress(panel, ui_qapp, width):
    panel.resize(width, 460)
    panel.set_condition_capture_progress("second", 15, 30)
    panel.select_condition("second")
    ui_qapp.processEvents()
    row = panel.rows["second"]
    button = row["button"]
    widgets = [row["labels"]["name"], row["capture_bar"],
               row["labels"]["capture_time"], row["labels"]["progress"], row["labels"]["result"]]
    rects = [QRect(widget.mapTo(button, QPoint()), widget.size()) for widget in widgets]
    assert button.height() == 40
    assert all(button.rect().contains(rect) for rect in rects)
    assert all(a.right() < b.left() for a, b in zip(rects, rects[1:]))
    label = row["labels"]["capture_time"]
    assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
    panel.select_condition("first")
    panel.select_condition("second")
    assert row["capture_bar"].value() == 500
    assert panel.rows["first"]["capture_bar"].value() == 0


def test_progress_uses_frozen_condition_trim_and_session_identity(panel):
    class Host(SequenceWidgetRecordingProcessOpsMixin):
        pass

    request = SimpleNamespace(request_id="capture", sample_rate=100,
                              target_samples=3100, trim_samples=100)
    session = SimpleNamespace(request=request, generation=1)
    context = RecordingProcessContext(request, "", False, session=session, condition_key="first")
    host = Host()
    host.left_panel = SimpleNamespace(result_panel=panel)
    host._recording_process_contexts = {request.request_id: context}
    host._active_recording_process_id = request.request_id
    # Viewed/active UI selection must not redirect a recording's progress.
    panel.select_condition("second")
    host._on_process_recording_progress(session, RecordingProgress("capture", 1, 50, 1.0))
    assert panel.rows["first"]["capture_bar"].value() == 0
    host._on_process_recording_progress(session, RecordingProgress("capture", 1, 1600, 2.0))
    assert panel.rows["first"]["capture_bar"].value() == 500
    assert panel.rows["second"]["capture_bar"].value() == 0
    host._on_process_recording_progress(session, RecordingProgress("capture", 2, 3100, 3.0))
    assert panel.rows["first"]["capture_bar"].value() == 500
    host._set_capture_terminal_phase(context, "已停止")
    context.cancelled = True
    host._on_process_recording_progress(session, RecordingProgress("capture", 1, 3100, 4.0))
    assert panel.rows["first"]["capture_bar"].value() == 500
    assert "已停止" in panel.rows["first"]["capture_bar"].toolTip()
    assert panel.rows["first"]["labels"]["result"].text() == "已停止"


def test_bridge_coalesces_updates_and_drops_updates_after_terminal(ui_qapp):
    bridge = RecordingServiceBridge(SimpleNamespace())
    session = SimpleNamespace(request=SimpleNamespace(request_id="run"))
    received = []
    bridge._callbacks["run"] = RecordingCallbacks(progress=lambda s, p: received.append(p))
    for frames in range(1, 101):
        bridge._enqueue("progress", session, frames)
    assert len(bridge._progress) == 1
    ui_qapp.processEvents()
    assert received == [100]
    bridge._enqueue("progress", session, 101)
    bridge._enqueue("cancelled", session, None)
    ui_qapp.processEvents()
    assert received == [100]
    assert not bridge._progress
    bridge.deleteLater()


@pytest.mark.parametrize("phase", ["已停止", "采集失败"])
def test_waiting_time_updates_without_audio_and_freezes_on_stop(panel, ui_qapp, phase):
    row = panel.rows["first"]
    panel.set_condition_result("first", "采集中")
    panel.start_condition_capture("first", 30)
    panel.select_condition("second")
    assert row["labels"]["result"].text() == "等待数据"
    QTest.qWait(350)
    assert row["capture_elapsed_seconds"] >= .2
    assert row["labels"]["capture_time"].text() == "0.0 / 30.0 秒"
    assert "尚未收到音频数据" in row["labels"]["capture_time"].toolTip()
    assert row["capture_bar"].value() == 0
    panel.set_condition_capture_phase("first", phase)
    frozen = row["capture_elapsed_seconds"]
    QTest.qWait(150)
    assert row["capture_elapsed_seconds"] == frozen
    assert not panel._capture_timer.isActive()


def test_wait_capture_stall_save_and_retry(panel, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("ui.sequence.motor_result_panel.monotonic", lambda: clock[0])
    row = panel.rows["first"]
    panel.set_condition_result("first", "采集中")
    panel.start_condition_capture("first", 30)
    clock[0] += 2
    panel._refresh_capture_elapsed()
    assert row["labels"]["capture_time"].text() == "0.0 / 30.0 秒"
    assert "已用 2.0 秒" in row["labels"]["capture_time"].toolTip()
    panel.set_condition_capture_progress("first", 15, 30)
    assert row["labels"]["result"].text() == "采集中"
    assert row["capture_bar"].value() == 500
    clock[0] += 3.1
    panel._refresh_capture_elapsed()
    assert row["labels"]["result"].text() == "等待数据"
    assert row["capture_bar"].value() == 500
    assert row["labels"]["capture_time"].text() == "15.0 / 30.0 秒"
    assert "停滞" in row["labels"]["capture_time"].toolTip()
    panel.set_condition_capture_progress("first", 30, 30)
    panel.set_condition_result("first", "数据保存中")
    assert not panel._capture_timer.isActive()
    assert row["labels"]["capture_time"].text() == "30.0 / 30.0 秒"
    panel.set_condition_capture_progress("first", 20, 30)
    assert row["labels"]["capture_time"].text() == "30.0 / 30.0 秒"
    panel.set_condition_result("first", "采集中")
    panel.start_condition_capture("first", 30)
    assert row["capture_bar"].value() == 0
    panel.reset()
    assert not panel._capture_timer.isActive()


def test_late_progress_cannot_change_finalizing_capture(panel):
    class Host(SequenceWidgetRecordingProcessOpsMixin):
        pass

    request = SimpleNamespace(request_id="capture", sample_rate=100,
                              target_samples=3000, trim_samples=0)
    session = SimpleNamespace(request=request, generation=1)
    context = RecordingProcessContext(request, "", False, session=session,
                                      condition_key="first", processing_notified=True)
    host = Host()
    host.left_panel = SimpleNamespace(result_panel=panel)
    host._recording_process_contexts = {request.request_id: context}
    host._active_recording_process_id = request.request_id
    panel.set_condition_capture_progress("first", 30, 30)
    panel.set_condition_result("first", "数据保存中")
    host._on_process_recording_progress(session, RecordingProgress("capture", 1, 1000, 1.0))
    assert panel.rows["first"]["capture_seconds"] == 30


def test_left_and_waveform_statuses_stay_in_sync(panel, tmp_path, monkeypatch):
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel

    class Host(SequenceWidgetRecordingProcessOpsMixin):
        pass

    workspace = AnalysisWaveformPanel(
        condition_configs=[{"key": "first", "condition_name": "10V3A"}],
        channel_layout_path=str(tmp_path / "channels.json"))
    host = Host()
    host.channel_workspace = workspace
    panel.capture_status_changed.connect(host._on_condition_capture_status_changed)
    clock = [10.0]
    monkeypatch.setattr("ui.sequence.motor_result_panel.monotonic", lambda: clock[0])
    row = panel.rows["first"]
    try:
        panel.set_condition_result("first", "采集中")
        panel.start_condition_capture("first", 30)
        assert workspace.status_label.text() == row["labels"]["result"].text() == "等待数据"
        panel.set_condition_capture_progress("first", 1, 30)
        assert workspace.status_label.text() == row["labels"]["result"].text() == "采集中"
        clock[0] += 3.1
        panel._refresh_capture_elapsed()
        assert workspace.status_label.text() == row["labels"]["result"].text() == "等待数据"
        panel.set_condition_capture_phase("first", "采集失败")
        assert workspace.status_label.text() == row["labels"]["result"].text() == "采集失败"
    finally:
        workspace.close()
        workspace.deleteLater()


def test_real_process_bridge_updates_panel_without_waveform_preview(panel, ui_qapp, tmp_path):
    import soundfile as sf
    from unit_test.base.recording_process_fakes import device_info

    def pump(predicate):
        deadline = monotonic() + 10
        while not predicate() and monotonic() < deadline:
            QTest.qWait(10)
        assert predicate()

    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
        backend_options={"trace_dir": str(tmp_path), "manual": True, "pace_writer": True})
    bridge = RecordingServiceBridge(service)
    request = RecordingRequest("progress-ui", "main", 100, 90, (0, 2), device_info(),
                               str(tmp_path / "progress.wav"), False, 2, {}, None,
                               {"enabled": False})

    class Host(SequenceWidgetRecordingProcessOpsMixin):
        pass

    context = RecordingProcessContext(request, "", False, condition_key="first")
    host = Host()
    host.left_panel = SimpleNamespace(result_panel=panel)
    host._recording_process_contexts = {request.request_id: context}
    host._active_recording_process_id = request.request_id
    errors, previews, threads = [], [], []

    def started(session):
        panel.set_condition_result("first", "采集中")
        panel.start_condition_capture("first", .88)

    def progress(session, value):
        threads.append(QThread.currentThread())
        host._on_process_recording_progress(session, value)

    def finalizing(session):
        context.processing_notified = True
        host._update_condition_capture_progress(session, request.target_samples)
        panel.set_condition_result("first", "数据保存中")

    try:
        session = bridge.start(request, RecordingCallbacks(
            started=started, progress=progress, finalizing=finalizing,
            preview=lambda s, p: previews.append(p),
            failed=lambda s, f: errors.append(f),
            result_ready=lambda s, result: s.accept_result()))
        context.session = session
        row = panel.rows["first"]
        pump(lambda: panel._capture_timer.isActive())
        assert row["labels"]["result"].text() == "等待数据"
        for index, frames in enumerate((3, 7, 90)):
            (tmp_path / f"feed-{index}").touch()
            pump(lambda: row["capture_seconds"] == (frames - 2) / 100)
        pump(session.released.is_set)
        assert not errors and not previews
        assert threads and all(thread is ui_qapp.thread() for thread in threads)
        assert row["labels"]["result"].text() == "数据保存中"
        assert not panel._capture_timer.isActive()
        assert sf.info(request.path).frames == 88
    finally:
        bridge.shutdown()
        pump(service.closed.is_set)
        bridge.deleteLater()
