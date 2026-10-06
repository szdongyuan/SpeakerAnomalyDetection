"""Current-condition recovery through production process, serial and UI handlers."""
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.recording_process_protocol import RecordingFailure
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin as Analysis
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin as Serial
from unit_test.ui.test_recording_process_integration import failure_recovery_host, deliver_failure_recovery_terminal

FRAMES = ["01 04 02 00 01 78 F0", "FE 02 01 02 91 9C", "01 04 02 00 03 F9 31"]


def product_host(tmp_path, *, serial=True, index=1):
    host, service, context, session = failure_recovery_host(tmp_path)
    for name, method in vars(Serial).items():
        if callable(method) and not name.startswith("__"):
            setattr(host, name, method.__get__(host))
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin as Ui
    host._next_manual_product_condition_display_name = Ui._next_manual_product_condition_display_name.__get__(host)
    host.SERIAL_PRODUCT_ERROR_MESSAGE = Serial.SERIAL_PRODUCT_ERROR_MESSAGE
    host._serial_trigger_config = {"enabled": serial}
    host.product_test_condition_configs = [
        {"key": key, "condition_name": key, "test_queue": "queue", "trigger_state": frame,
         "group_name": "USB-A" if i == 0 else "USB-C"}
        for i, (key, frame) in enumerate(zip("abc", FRAMES))]
    host._manual_product_condition_group_id = "round-7"
    host._displayed_manual_product_condition_group_id = "round-7"
    host._current_cycle_recorded_count = "round-7"
    host._manual_product_condition_index = index
    host._manual_product_condition_completed_keys = set("abc"[:index])
    host._manual_product_condition_results = {key: "OK" for key in "abc"[:index]}
    host._manual_product_condition_counted_group_labels = {}
    host._active_product_condition_key = "abc"[index]
    host._active_product_condition_config = host.product_test_condition_configs[index].copy()
    host._serial_product_port_index = 0 if index == 0 else 1
    host._serial_product_condition_executing = serial
    host._serial_product_session_started = serial
    host._serial_product_latched_frame = FRAMES[index]
    host._analysis_round_config_locked = True
    host._product_round_locked_sn = "barcode-7"
    host._active_test_round_metadata = {"sample_number": "sample-7", "test_round": 7}
    host._analysis_task_queue = [object()]
    host._analysis_has_pending_tasks = lambda: bool(host._analysis_task_queue)
    host._load_sequence_config_for_product_condition = lambda condition: (True, "")
    host._is_manual_product_condition_cycle_active = Analysis._is_manual_product_condition_cycle_active.__get__(host)
    host._advance_manual_product_condition_cycle_after_recording = Analysis._advance_manual_product_condition_cycle_after_recording.__get__(host)
    host._discard_current_recent_session = Analysis._discard_current_recent_session.__get__(host)
    host._reset_manual_product_condition_cycle = Analysis._reset_manual_product_condition_cycle.__get__(host)
    host._delete_serial_product_round_records = Mock()
    host.recent_session_panel = None
    host.recent_test_sessions = [*"abc"[:index], "failed"]
    host.recent_test_session_by_id = {key: {"result": "OK"} for key in "abc"[:index]}
    host.recent_test_session_by_id["failed"] = {"result": "waiting"}
    host._current_recent_session_id = context.recent_session_id = "failed"
    host.started = []
    def start(label):
        host.started.append(host._active_product_condition_key)
        host._record_workflow_busy = host.player_status_flag = True
    host.start_this_play = start
    return host, service, context, session


def fail(host, session):
    host._on_process_recording_failed(session, RecordingFailure(
        session.request.request_id, "capture", session.request.path, "injected failure"))


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_failure_preserves_round_and_only_discards_current_placeholder(ui_qapp, tmp_path, monkeypatch, serial, index):
    host, service, context, session = product_host(tmp_path, serial=serial, index=index)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    wav, csv = tmp_path / "previous.wav", tmp_path / "previous.csv"
    wav.write_bytes(b"completed audio")
    csv.write_bytes(b"completed analysis")
    with sqlite3.connect(tmp_path / "records.db") as db:
        db.execute("create table recordings (condition text)")
        db.execute("insert into recordings values ('a')")
    retained = {name: getattr(host, name) for name in (
        "_manual_product_condition_results", "_active_product_condition_config",
        "_analysis_task_queue", "_active_test_round_metadata")}
    fail(host, session)
    assert host._manual_product_condition_group_id == "round-7"
    assert host._manual_product_condition_index == index
    assert host._active_product_condition_key == "abc"[index]
    assert host._manual_product_condition_completed_keys == set("abc"[:index])
    assert host._serial_product_port_index == (0 if index == 0 else 1)
    assert host._analysis_round_config_locked
    assert host._product_round_locked_sn == "barcode-7"
    assert all(getattr(host, name) is value for name, value in retained.items())
    assert host.recent_test_sessions == list("abc"[:index])
    assert "failed" not in host.recent_test_session_by_id
    assert wav.read_bytes() == b"completed audio" and csv.read_bytes() == b"completed analysis"
    with sqlite3.connect(tmp_path / "records.db") as db:
        assert db.execute("select * from recordings").fetchall() == [("a",)]
    host._delete_serial_product_round_records.assert_not_called()
    host._on_streaming_complete.assert_not_called()
    host.clear_all_direction_waveforms.assert_not_called()
    assert not host.player_status_flag and not host._record_workflow_busy
    assert not host._serial_product_condition_executing
    assert "测试异常" in host.left_panel.set_current_stage.call_args.args[0]


def test_serial_retry_needs_fresh_current_frame_after_dialog_and_release(ui_qapp, tmp_path, monkeypatch):
    host, service, context, session = product_host(tmp_path)
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    def frame(index=1, received=11.0):
        host.on_serial_full_frame_received({"raw_hex": FRAMES[index], "received_monotonic": received})
    def dialog(*args):
        frame(received=10.0)
        assert host.started == []
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", dialog)
    fail(host, session)
    assert host.started == []
    frame(received=11.0)  # resource ownership still held; cannot queue retry
    clock[0] = 12.0
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    assert host.started == []
    for value in [None, "invalid", float("nan"), float("inf"), 10.0, 11.0]:
        frame(received=value)
    frame(index=2, received=13.0)
    assert host.started == []
    frame(received=13.0)
    frame(received=14.0)
    assert host.started == ["b"]
    assert host._manual_product_condition_group_id == "round-7"

@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("terminal", ["released", "release_failed"])
@pytest.mark.parametrize("timing", ["before", "inside", "after"])
def test_product_release_order_recovers_buttons_without_starting(ui_qapp, tmp_path, monkeypatch, terminal, timing, serial):
    host, service, context, session = product_host(tmp_path, serial=serial)
    host.update_player_btn_is_paused = Mock(wraps=host.update_player_btn_is_paused)
    def release():
        service.can_start_recording = terminal == "released"
        deliver_failure_recovery_terminal(host, session, terminal)
    if timing == "before":
        # The service may have already released resources before queued failure delivery.
        service.can_start_recording = terminal == "released"
        if terminal == "released":
            session.released.set()
    def dialog(*args):
        if timing == "inside":
            release()
        assert not host.player_btn.isEnabled()
        assert not host._can_prepare_recording_workflow()
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", dialog)
    fail(host, session)
    if timing == "after":
        assert not host.player_btn.isEnabled()
        release()
    assert host.update_player_btn_is_paused.call_count >= 2
    assert host._can_prepare_recording_workflow() == (terminal == "released")
    assert host.player_btn.isEnabled() == (terminal == "released" and not serial)
    assert host.started == []
    before = host.player_btn.isEnabled()
    deliver_failure_recovery_terminal(host, session, terminal)
    fail(host, session)
    assert host.player_btn.isEnabled() == before
    assert host.started == []


@pytest.mark.parametrize("action", ["reset", "cancel", "close", "new_workflow", "new_alias"])
def test_modal_return_cannot_rearm_obsolete_recovery(ui_qapp, tmp_path, monkeypatch, action):
    host, service, context, session = product_host(tmp_path)
    def dialog(*args):
        if action == "reset":
            host._reset_manual_product_condition_cycle()
        elif action == "cancel":
            host._cancel_process_recording()
        elif action == "close":
            host._closing = True
        elif action == "new_workflow":
            host._recording_workflow_token = object()
        else:
            host._recording_process_session = SimpleNamespace(request=session.request)
        host.player_btn.setEnabled(False)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", dialog)
    fail(host, session)
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    assert host._current_product_recording_retry() is None
    assert not host.player_btn.isEnabled()
    assert host.started == []


def test_earlier_analysis_completion_keeps_failed_condition_stage(ui_qapp, tmp_path, monkeypatch):
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin as ProcessAnalysis
    host, service, context, session = product_host(tmp_path)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    host._analysis_task_queue.clear()
    host.left_panel.set_current_stage.reset_mock()
    assert ProcessAnalysis._restore_waiting_stage_after_automatic_analysis(host)
    assert "测试异常" in host.left_panel.set_current_stage.call_args.args[0]
    assert host._analysis_round_config_locked
    assert host._manual_product_condition_completed_keys == {"a"}


def test_release_without_notification_requires_another_fresh_frame(ui_qapp, tmp_path, monkeypatch):
    host, service, context, session = product_host(tmp_path)
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    service.can_start_recording = True
    clock[0] = 12.0
    # An old busy-period frame is delivered after readiness, before any release callback.
    host.on_serial_full_frame_received({"raw_hex": FRAMES[1], "received_monotonic": 11.0})
    assert host.started == []
    host.on_serial_full_frame_received({"raw_hex": FRAMES[1], "received_monotonic": 13.0})
    assert host.started == ["b"]


def install_attempt(host, service, previous_session, request_id):
    from dataclasses import replace
    from base.recording_service import RecordingSession, RecordingCallbacks
    from ui.sequence.recording_process_context import RecordingProcessContext
    request = replace(previous_session.request, request_id=request_id)
    session = RecordingSession(service, request, RecordingCallbacks())
    host._recording_workflow_token = object()
    context = RecordingProcessContext(request, host._active_product_condition_key, False,
        session=session, workflow_token=host._recording_workflow_token,
        recent_session_id=request_id, recorded_signal_info=host.recorded_signal_info)
    host._recording_contexts()[request_id] = context
    host._recording_process_session = session
    host._active_recording_process_id = host._recording_process_id = request_id
    host._recording_process_request = request
    host._recording_process_cancelled = False
    host.recent_test_sessions.append(request_id)
    host.recent_test_session_by_id[request_id] = {"result": "waiting"}
    host._current_recent_session_id = request_id
    host.player_status_flag = host._record_workflow_busy = True
    return context, session


@pytest.mark.parametrize("serial", [False, True])
def test_repeated_failure_then_success_advances_once(ui_qapp, tmp_path, monkeypatch, serial):
    import numpy as np
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin as Streaming
    host, service, context, session = product_host(tmp_path, serial=serial)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    assert host._prepare_next_manual_product_condition_recording() is True
    host._serial_product_condition_executing = serial
    second_context, second = install_attempt(host, service, session, "retry-1")
    fail(host, second)
    assert host._manual_product_condition_index == 1
    assert host.recent_test_sessions == ["a"]
    deliver_failure_recovery_terminal(host, second, "released")
    assert host._prepare_next_manual_product_condition_recording() is True
    host._serial_product_condition_executing = serial
    third_context, third = install_attempt(host, service, second, "retry-2")
    # Old duplicates cannot drop the newly installed context or its placeholder.
    fail(host, session)
    deliver_failure_recovery_terminal(host, second, "released")
    assert host._recording_context_for_session(third) is third_context
    assert host._current_recent_session_id == "retry-2"
    saved = Mock(return_value=(0, "saved"))
    monkeypatch.setattr("ui.sequence.sequence_widget_streaming_ops.RecordingManager.save_signal_info_to_db", saved)
    host._on_streaming_complete = Streaming._on_streaming_complete.__get__(host)
    third.state = "completed"
    third.released.set()
    audio = SimpleNamespace(mono=np.ones(9, dtype=np.float32), multi=np.ones((9, 2), dtype=np.float32),
                            descriptor=SimpleNamespace(sample_rate=100, warnings=()))
    third_context.validated_audio = third_context.accepted_audio = audio
    third_context.final_windows = host.channel_workspace.all_subwindows()
    host._publish_recording_context(third_context)
    host._publish_recording_context(third_context)
    assert saved.call_count == 1
    assert host._manual_product_condition_completed_keys == {"a", "b"}
    assert host._manual_product_condition_index == 2
    assert host._manual_product_condition_group_id == "round-7"
    assert host._active_test_round_metadata["test_round"] == 7
    assert host._analysis_task_queue


@pytest.mark.parametrize("lease", ["none", "recording", "csv"])
def test_legacy_rejected_audio_respects_file_owners_and_recovers_manual_button(ui_qapp, tmp_path, monkeypatch, lease):
    from pathlib import Path
    from base.raw_audio_csv_tasks import CsvTaskLedger
    host, service, context, session = product_host(tmp_path, serial=False)
    host._recording_contexts().clear()
    host._active_recording_process_id = None
    host._recording_process_session = None
    service.can_start_recording = True
    service.can_start_recording = lease != "recording"
    service.is_path_leased = lambda path: lease == "recording"
    host.raw_audio_csv_service = CsvTaskLedger()
    path = Path(host.recorded_path)
    path.write_bytes(b"rejected audio")
    permit = host.raw_audio_csv_service.try_acquire_mutation((str(path),)) if lease == "csv" else None
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    host._handle_invalid_recording("invalid audio")
    assert path.exists() == (lease != "none")
    assert host.player_btn.isEnabled() == (lease != "recording")
    assert host._manual_product_condition_index == 1
    assert host._manual_product_condition_completed_keys == {"a"}
    if permit is not None:
        host.raw_audio_csv_service.release_mutation(permit)


def test_csv_readiness_event_allows_first_fresh_frame_without_auto_start(ui_qapp, tmp_path, monkeypatch):
    from base.raw_audio_csv_tasks import CsvTaskLedger
    from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
    from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin as RawCsv
    host, service, context, session = product_host(tmp_path)
    for name, method in vars(RawCsv).items():
        if callable(method) and not name.startswith("__"):
            setattr(host, name, method.__get__(host))
    host.raw_audio_csv_service = CsvTaskLedger()
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: True}
    host._owned_raw_audio_csv_tasks = set()
    tokens = [host.raw_audio_csv_service.reserve(str(i)).reservation for i in range(16)]
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    assert not host._can_prepare_recording_workflow()
    clock[0] = 12.0
    host.raw_audio_csv_service.release_reservation(tokens[0])
    host._on_raw_audio_csv_service_event(SimpleNamespace(kind="released",
        task=SimpleNamespace(request=SimpleNamespace(task_id="earlier-analysis-csv"))))
    assert host.started == []
    clock[0] = 14.0
    host.on_serial_full_frame_received({"raw_hex": FRAMES[1], "received_monotonic": 11.0})
    assert host.started == []
    host.on_serial_full_frame_received({"raw_hex": FRAMES[1], "received_monotonic": 13.0})
    assert host.started == ["b"]


@pytest.mark.parametrize("stage", ["capture_release_timeout", "release"])
def test_ve_release_failure_keeps_current_round_and_file_lease(ui_qapp, tmp_path, monkeypatch, stage):
    from pathlib import Path
    from unit_test.base.ve3668n_fakes import capture_request
    host, service, context, session = product_host(tmp_path)
    request = capture_request(Path(session.request.path), request_id=session.request.request_id)
    context.request = session.request = request
    Path(request.path).write_bytes(b"unpublished leased audio")
    warning = Mock()
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", warning)
    if stage == "release":
        host._on_process_recording_release_failed(session, "still leased")
    else:
        host._on_process_recording_failed(session, RecordingFailure(
            request.request_id, stage, request.path, "capture release timed out", handles_released=False))
    assert host._manual_product_condition_group_id == "round-7"
    assert host._manual_product_condition_index == 1
    assert Path(request.path).read_bytes() == b"unpublished leased audio"
    assert not host._can_prepare_recording_workflow()
    assert len(warning.call_args_list) == 1
    host._on_streaming_complete.assert_not_called()


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("index", [0, 2])
def test_first_and_final_conditions_retry_the_same_index(ui_qapp, tmp_path, monkeypatch, serial, index):
    host, service, context, session = product_host(tmp_path, serial=serial, index=index)
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    if serial:
        host.on_serial_full_frame_received({"raw_hex": "01 04 02 00 00 B9 30", "received_monotonic": 11.0})
        assert host.started == []
        host.on_serial_full_frame_received({"raw_hex": FRAMES[index], "received_monotonic": 11.0})
        assert host.started == ["abc"[index]]
    else:
        assert host.started == []
        assert host._prepare_next_manual_product_condition_recording() is True
    assert host._manual_product_condition_index == index
    assert host._active_product_condition_key == "abc"[index]
    assert host._manual_product_condition_completed_keys == set("abc"[:index])


def test_earlier_analysis_stage_does_not_replace_retry_prompt(ui_qapp, tmp_path, monkeypatch):
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin as ProcessAnalysis
    host, service, context, session = product_host(tmp_path)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)
    fail(host, session)
    host.left_panel.set_current_stage.reset_mock()
    ProcessAnalysis._set_condition_analysis_stage(host, "a", "分析中", "running")
    host.left_panel.set_condition_result.assert_called_with("a", "分析中", tone="running")
    host.left_panel.set_current_stage.assert_not_called()


@pytest.mark.parametrize("event", ["disconnect", "explicit_abort"])
def test_abort_during_retry_dialog_invalidates_recovery_once(ui_qapp, tmp_path, monkeypatch, event):
    host, service, context, session = product_host(tmp_path)
    host.serial_trigger_btn = Mock()
    host.update_player_btn_is_paused = Mock(wraps=host.update_player_btn_is_paused)
    # Deletion can dispatch nested callbacks too. Claim abort ownership first.
    host._delete_serial_product_round_records.side_effect = lambda group: host._abort_serial_product_round("nested abort")
    refreshes_after_abort = []
    def dialog(*args):
        if event == "disconnect":
            status = {"enabled": True, "connected": False, "running": False,
                      "error": "port disconnected", "message": "port disconnected"}
            host.on_serial_trigger_status_changed(status)
            host.on_serial_trigger_status_changed(status)
        else:
            host._abort_serial_product_round("explicit abort")
            host._abort_serial_product_round("duplicate abort")
        assert host._manual_product_condition_group_id == ""
        assert host._current_product_recording_retry() is None
        assert host._serial_product_error_dialog_open
        refreshes_after_abort.append(host.update_player_btn_is_paused.call_count)
    warning = Mock(side_effect=dialog)
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", warning)
    fail(host, session)
    service.can_start_recording = True
    deliver_failure_recovery_terminal(host, session, "released")
    assert warning.call_count == 1
    host._delete_serial_product_round_records.assert_called_once_with("round-7")
    host.clear_all_direction_waveforms.assert_called_once()
    assert host._manual_product_condition_completed_keys == set()
    assert host._manual_product_condition_index == host._serial_product_port_index == 0
    assert host.update_player_btn_is_paused.call_count == refreshes_after_abort[0]
    assert not host._serial_product_error_dialog_open
    assert host._recording_failure_recovery is None
    assert host.started == []
    host.on_serial_full_frame_received({"raw_hex": FRAMES[1], "received_monotonic": 999999999.0})
    assert host.started == []
