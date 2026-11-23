"""Pause through real Qt delivery, spawned recording and deferred file cleanup."""
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import Mock
import time

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QPushButton

from base.raw_audio_csv_service import RawAudioCsvService
from ui.sequence.motor_result_panel import MotorResultPanel
from ui.sequence.sequence_widget_pause_ops import SequenceWidgetPauseOpsMixin
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from ui.sequence.sequence_widget_pause_ops import TestPauseContext
from unit_test.ui.test_round_reset import host as reset_host, start_round, add_record
from unit_test.ui.test_recording_process_integration import main_host, pump, service


@pytest.fixture
def host(ui_qapp, service, tmp_path):
    original = main_host(service, tmp_path)

    class PauseHost(SequenceWidgetPauseOpsMixin, SequenceWidgetRawCsvOpsMixin,
                    type(original), QWidget):
        def __init__(self):
            QWidget.__init__(self)
            self.__dict__.update(original.__dict__)

    widget = PauseHost()
    widget._init_test_pause()
    widget.raw_audio_csv_service = RawAudioCsvService()
    widget._pending_raw_audio_csv_recording = None
    widget._raw_audio_csv_recording_paths = set()
    widget.product_test_condition_configs = [
        {"key": "first", "group_name": "端口1", "condition_name": "10V3A", "trigger_state": " "},
        {"key": "second", "group_name": "端口2", "condition_name": "10V3A"},
    ]
    panel = MotorResultPanel(condition_configs=widget.product_test_condition_configs, queue_catalog={})
    panel.set_channels([0, 2])
    panel.set_condition_result("first", "OK", "ok")
    widget.left_panel = SimpleNamespace(
        result_panel=panel, set_current_stage=panel.set_current_stage,
        set_condition_result=panel.set_condition_result, set_final_result=panel.set_final_result,
        clear_current_stage=panel.clear_current_stage)
    widget.player_btn = QPushButton()
    widget._refresh_round_reset_button = Mock()
    widget._analysis_has_pending_tasks = Mock(return_value=False)
    widget._manual_product_condition_group_id = "round-1"
    widget._active_product_condition_key = "second"
    widget._get_active_product_condition_key = lambda: "second"
    widget._manual_product_condition_index = 1
    widget._manual_product_condition_completed_keys = {"first"}
    widget._serial_trigger_delay_timer = QTimer(widget)
    widget._serial_trigger_runtime_status = {"connected": True}
    widget._barcode_debounce_timer = QTimer(widget)
    widget.channel_workspace.clear_plots = Mock()
    widget.channel_workspace.clear_direction = Mock()
    widget.channel_workspace.set_condition_context = Mock()
    widget._can_prepare_recording_workflow = lambda: (
        not widget._test_pause_blocks_start()
        and not widget._recording_contexts()
        and widget.recording_bridge.service.can_start_recording)
    widget.recent_test_sessions = []

    def begin_recent():
        widget._current_recent_session_id = "current"
        widget.recent_test_session_by_id = {"current": {"state": "recording"}}
        widget.recent_test_sessions = ["current"]

    widget._begin_recent_session_for_current_run = begin_recent
    yield widget
    widget._test_pause_timer.stop()
    widget.raw_audio_csv_service.begin_shutdown()
    pump(ui_qapp, widget.raw_audio_csv_service.closed.is_set)
    panel.deleteLater()
    widget.player_btn.deleteLater()
    widget.deleteLater()


def start(host):
    recorded, rate = host.reset_work_pram(None)
    host._start_process_recording(recorded, rate)
    host.player_status_flag = True
    return host._recording_process_session


def paused(host):
    return host._test_pause is not None and host._test_pause.phase == "paused"


@pytest.mark.parametrize("blocked_by", [None, "round_analysis", "recording_resources"])
def test_background_analysis_allows_next_manual_capture_only_when_ready(
        host, service, ui_qapp, blocked_by):
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin

    host._analysis_has_pending_tasks.return_value = True
    host.player_status_flag = host._record_workflow_busy = False
    host._analysis_round_completion_pending = blocked_by == "round_analysis"
    host._load_sequence_config_for_product_condition = Mock()
    host._next_manual_product_condition_display_name = lambda: "下一档位"
    host._prepare_next_manual_product_condition_recording = Mock(return_value=True)
    if blocked_by == "recording_resources":
        host._can_prepare_recording_workflow = lambda: False
    host.start_this_play = Mock(side_effect=lambda label: start(host))
    host.update_player_btn_is_paused = MethodType(
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused, host)
    host.player_btn.clicked.connect(
        lambda: SequenceWidgetAnalysisOpsMixin.on_clicked_player_btn(host))

    host.update_player_btn_is_paused()
    assert host.player_btn.toolTip() == "开始录制：下一档位"
    assert host.player_btn.isEnabled() == (blocked_by is None)
    host.player_btn.click()
    assert host._test_pause is None
    if blocked_by is not None:
        host.start_this_play.assert_not_called()
        return
    host.start_this_play.assert_called_once()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._recording_contexts())
    assert session.state == "completed"
    assert host._analysis_has_pending_tasks()
    assert host.left_panel.result_panel.rows["first"]["result"] == "OK"
    assert host._manual_product_condition_completed_keys == {"first"}
    assert host._test_pause is None


def test_partial_cancel_deletes_only_current_file_then_restarts(host, service, tmp_path, ui_qapp):
    service._backend_options["manual"] = True
    completed = tmp_path / "completed.wav"
    completed.write_bytes(b"previous condition must survive")
    session = start(host)
    (tmp_path / "feed-0").touch()
    pump(ui_qapp, lambda: session._progress_frames > 0)
    host._request_test_pause()
    host._request_test_pause()
    assert host._test_pause_is_busy()
    pump(ui_qapp, lambda: paused(host))
    assert session.state == "cancelled"
    assert session.released.is_set()
    assert not Path(session.request.path).exists()
    assert completed.read_bytes() == b"previous condition must survive"
    assert host._manual_product_condition_completed_keys == {"first"}
    assert host._manual_product_condition_index == 1
    host.channel_workspace.clear_direction.assert_called_once_with("second")
    host.channel_workspace.clear_plots.assert_not_called()
    row = host.left_panel.result_panel.rows["second"]
    assert row["labels"]["capture_time"].text().startswith("--")
    assert row["result"] == "已暂停"
    assert not host._test_pause_timer.isActive()
    host.recorded_path = str(tmp_path / "retry.wav")
    host.on_clicked_player_btn = lambda: start(host)
    host._resume_test()
    retry = host._recording_process_session
    assert retry is not session and retry.request.request_id != session.request.request_id
    assert host._test_pause is None
    # A delayed terminal from the old session cannot cancel this attempt.
    host._on_process_recording_cancelled(session, session.descriptor)
    assert not retry.cancel_requested
    (tmp_path / "feed-1").touch()
    (tmp_path / "feed-2").touch()
    pump(ui_qapp, lambda: retry.released.is_set() and not host._recording_contexts())
    assert retry.state == "completed"
    import soundfile as sf
    assert sf.info(retry.request.path).frames == retry.request.target_samples - retry.request.trim_samples


def test_pause_after_service_acceptance_preserves_file(host, service, ui_qapp):
    # Pause in the accepted callback, before the GUI publishes the result.
    accepted = host._on_process_recording_accepted
    def pause_then_publish(current, audio):
        host._request_test_pause()
        accepted(current, audio)
    host._on_process_recording_accepted = pause_then_publish
    session = start(host)
    pump(ui_qapp, lambda: paused(host))
    assert session.state == "completed"
    assert Path(session.request.path).is_file()
    assert host._test_pause.outcome == ""


def test_delete_failure_blocks_resume_and_retry_cleans_exact_path(
        host, service, tmp_path, ui_qapp, monkeypatch):
    service._backend_options["manual"] = True
    session = start(host)
    pump(ui_qapp, lambda: Path(session.request.path).exists())
    unlink = Path.unlink
    def deny_current(path, *args, **kwargs):
        if str(path) == session.request.path:
            raise PermissionError("held by external reader")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", deny_current)
    host._request_test_pause()
    pump(ui_qapp, lambda: host._test_pause.phase == "cleanup_failed")
    assert "external reader" in host._test_pause.error
    host._resume_test()
    assert host._recording_process_session is session
    monkeypatch.setattr(Path, "unlink", unlink)
    host._handle_test_pause_button()
    pump(ui_qapp, lambda: paused(host))
    assert not Path(session.request.path).exists()


def test_waits_for_existing_analysis_without_stopping_its_queue(host, ui_qapp):
    host._analysis_has_pending_tasks.return_value = True
    host.player_status_flag = True
    host._request_test_pause()
    host.player_status_flag = False
    host._poll_test_pause()
    assert host._test_pause.phase == "pausing"
    host._analysis_has_pending_tasks.return_value = False
    pump(ui_qapp, lambda: paused(host))


def test_serial_resume_waits_for_new_frame_and_does_not_start_capture(host):
    host.product_test_condition_configs[1]["trigger_state"] = "AA"
    host._request_test_pause()
    assert paused(host)
    host.on_clicked_player_btn = Mock()
    before = time.monotonic()
    host._resume_test()
    host.on_clicked_player_btn.assert_not_called()
    assert host._test_resume_after >= before
    assert host._test_pause is None


def test_real_button_refresh_exposes_pause_while_capturing(host, service, ui_qapp):
    service._backend_options["manual"] = True
    session = start(host)
    SequenceWidgetUiOpsMixin.update_player_btn_is_playing(host)
    assert host.player_btn.isEnabled()
    assert host.player_btn.toolTip() == "暂停测试"
    host._request_test_pause()
    assert not host.player_btn.isEnabled()
    pump(ui_qapp, lambda: paused(host))
    assert host.player_btn.isEnabled()
    assert "继续测试" in host.player_btn.toolTip()
    assert host.left_panel.result_panel.stage_label.text() == "已暂停"


def test_round_finishing_during_pause_does_not_leave_continue_button(host, ui_qapp):
    host._analysis_has_pending_tasks.return_value = True
    host.player_status_flag = True
    host._request_test_pause()
    host.player_status_flag = False
    host._manual_product_condition_group_id = ""
    host._analysis_has_pending_tasks.return_value = False
    pump(ui_qapp, lambda: host._test_pause is None)
    assert not host._test_pause_timer.isActive()


@pytest.mark.parametrize("decision", [None, False])
def test_existing_reset_cancel_keeps_pause_and_success_clears_it(reset_host, tmp_path, decision):
    start_round(reset_host)
    wav = tmp_path / "saved.wav"
    add_record(reset_host, wav)
    pause = TestPauseContext(reset_host._manual_product_condition_group_id, "", None, None,
                             phase="paused")
    reset_host._test_pause = pause
    reset_host._test_pause_timer = QTimer()
    reset_host._test_trigger_epoch = 0
    reset_host._clear_test_pause = MethodType(SequenceWidgetPauseOpsMixin._clear_test_pause, reset_host)
    reset_host._test_pause_is_busy = MethodType(SequenceWidgetPauseOpsMixin._test_pause_is_busy, reset_host)
    reset_host._confirm_round_reset = Mock(return_value=decision)
    reset_host._on_reset_current_round()
    assert wav.read_bytes() == b"test"
    assert (reset_host._test_pause is pause) is (decision is None)


@pytest.mark.parametrize("index", [0, 1, 3, 5])
def test_resume_uses_original_condition_preparation_without_rewinding_completed(index):
    from unit_test.test_manual_product_condition_cycle import _DummyManualCycleWidget
    widget = _DummyManualCycleWidget()
    widget.product_test_condition_configs = [
        dict(key=f"port{port}/gear{gear}", group_name=f"端口{port}",
             condition_name=f"档位{gear}", test_queue=f"queue{port}{gear}")
        for port in (1, 2) for gear in (1, 2, 3)]
    widget._manual_product_condition_group_id = "original-round"
    widget._manual_product_condition_index = index
    completed = {item["key"] for item in widget.product_test_condition_configs[:index]}
    widget._manual_product_condition_completed_keys = completed.copy()
    widget._manual_product_condition_results = {key: "OK" for key in completed}
    widget._test_pause = TestPauseContext("original-round", "", None, None, phase="paused")
    widget._test_trigger_epoch = 0
    widget._can_prepare_recording_workflow = lambda: True
    widget._pause_uses_serial_trigger = lambda: False
    original_start = widget.start_this_play
    def admitted_start(label):
        original_start(label)
        widget._recording_process_session = object()
    widget.start_this_play = admitted_start
    SequenceWidgetPauseOpsMixin._resume_test(widget)
    assert widget.loaded_queues == [widget.product_test_condition_configs[index]["test_queue"]]
    assert widget.started[0][1] == widget.product_test_condition_configs[index]["key"]
    assert widget._manual_product_condition_completed_keys == completed
    assert widget._manual_product_condition_group_id == "original-round"


@pytest.mark.parametrize("received", [None, -1.0, 10.0, float("nan"), "bad"])
def test_resume_ignores_stale_or_missing_serial_timestamp(host, received):
    host._test_resume_after = 10.0
    assert not host._test_pause_accepts_serial_frame({"received_monotonic": received})
    assert host._test_pause_accepts_serial_frame({"received_monotonic": 10.01})


def test_serial_handler_cannot_advance_on_idle_close_or_start_during_pause():
    from unit_test.test_serial_product_condition_runtime import (
        _SerialProductHost, FRAME_6000, FRAME_IDLE, FRAME_CLOSE)
    widget = _SerialProductHost()
    widget._test_pause_accepts_serial_frame = lambda payload: False
    for frame in (FRAME_IDLE, FRAME_CLOSE, FRAME_6000):
        widget.on_serial_full_frame_received({"raw_hex": frame, "received_monotonic": time.monotonic()})
    assert widget.started == []
    assert widget._manual_product_condition_index == 0


def test_serial_abort_during_pause_keeps_completed_records_and_round():
    from unit_test.test_serial_product_condition_runtime import _SerialProductHost
    widget = _SerialProductHost()
    widget._test_pause_blocks_start = lambda: True
    widget._manual_product_condition_group_id = "existing-round"
    widget._manual_product_condition_completed_keys = {"completed"}
    widget._delete_serial_product_round_records = Mock()
    widget._abort_serial_product_round("connection lost")
    widget._delete_serial_product_round_records.assert_not_called()
    assert widget._manual_product_condition_group_id == "existing-round"
    assert widget._manual_product_condition_completed_keys == {"completed"}


def test_serial_resume_keeps_pause_until_connection_returns(host, monkeypatch):
    from ui.sequence.sequence_widget_pause_ops import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", Mock())
    host.product_test_condition_configs[1]["trigger_state"] = "AA"
    host._request_test_pause()
    host._serial_trigger_runtime_status = {"connected": False}
    host._resume_test()
    assert paused(host)
    host._serial_trigger_runtime_status = {"connected": True}
    host._resume_test()
    assert host._test_pause is None


def test_already_scheduled_direction_trigger_is_invalidated_by_pause(host, ui_qapp):
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
    host._queued_directional_trigger = "forward"
    host.on_directional_triggered = Mock()
    SequenceWidgetBarcodeOpsMixin._drain_queued_directional_trigger(host)
    host._test_trigger_epoch += 1
    deadline = time.monotonic() + .15
    pump(ui_qapp, lambda: time.monotonic() >= deadline)
    host.on_directional_triggered.assert_not_called()


def test_ui_pause_and_resume_without_audio_waiting(host, service, ui_qapp, monkeypatch):
    from ui.sequence.sequence_widget_pause_ops import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", Mock())
    service._backend_options["manual"] = True
    start(host)
    host._request_test_pause()
    assert not host._test_pause_allows_close()
    pump(ui_qapp, lambda: paused(host))
    assert host._test_pause_allows_close()
    assert host._manual_product_condition_completed_keys == {"first"}


def test_pause_reentered_during_startup_never_opens_capture(host, monkeypatch):
    from ui.sequence import sequence_widget_analysis_ops as analysis
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    host._can_prepare_recording_workflow = lambda: not host._test_pause_blocks_start()
    host._can_start_recording_workflow = host._can_prepare_recording_workflow
    host.update_player_btn_is_playing = lambda: SequenceWidgetUiOpsMixin.update_player_btn_is_playing(host)
    host._start_process_recording = Mock()
    monkeypatch.setattr(analysis.QApplication, "processEvents", host._request_test_pause)
    host.judge_play_and_record()
    host._start_process_recording.assert_not_called()
    assert paused(host)
    assert not host.player_status_flag and not host._record_workflow_busy
    assert host._pending_raw_audio_csv_recording is None


def test_save_error_during_pause_preserves_round_and_can_settle(host):
    from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
    host._analysis_has_pending_tasks.return_value = True
    host.player_status_flag = True
    host._request_test_pause()
    host._analysis_has_pending_tasks.return_value = False
    assert SequenceWidgetSerialTriggerOpsMixin._on_serial_product_runtime_error(host, "database failed")
    assert host._test_pause.phase == "cleanup_failed"
    assert host._manual_product_condition_completed_keys == {"first"}
    assert host._manual_product_condition_group_id == "round-1"
    host._handle_test_pause_button()
    assert paused(host)


def test_paused_controls_render_without_changing_completed_row(host, ui_qapp, tmp_path):
    import os
    from PyQt5.QtGui import QFontDatabase
    from PyQt5.QtWidgets import QVBoxLayout
    if os.name == "nt" and not QFontDatabase().families():
        QFontDatabase.addApplicationFont(str(Path(os.environ["WINDIR"]) / "Fonts" / "msyh.ttc"))
    panel = host.left_panel.result_panel
    panel.start_condition_capture("second", 30)
    panel.set_condition_result("second", "采集中", "running")
    panel.set_condition_capture_progress("second", 15, 30)
    panel.pause_condition_capture("second")
    panel.select_condition("second")
    host._test_pause = TestPauseContext("round-1", "second", None, None, phase="paused")
    host._apply_test_pause_presentation()
    layout = QVBoxLayout(host)
    host.player_btn.setFixedSize(120, 40)
    layout.addWidget(host.player_btn)
    layout.addWidget(panel)
    host.resize(590, 400)
    host.show()
    ui_qapp.processEvents()
    assert panel.rows["first"]["result"] == "OK"
    assert panel.rows["second"]["capture_bar"].value() == 0
    assert panel.rows["second"]["labels"]["capture_time"].text() == "-- / 30.0 秒"
    assert host.player_btn.isEnabled()
    assert panel.stage_label.text() == "已暂停"
    assert host.grab().save(str(tmp_path / "paused_controls.png"))
