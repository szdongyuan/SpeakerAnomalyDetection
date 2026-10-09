"""GUI startup boundaries exercised through the existing workflow mixins."""
from collections import deque
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.raw_audio_csv_tasks import CsvTaskLedger
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_analysis_ops as analysis
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from unit_test.ui.test_recording_process_integration import main_host, CapturingBridge


@pytest.fixture
def clock(monkeypatch):
    from base import recording_startup_trace
    value = SimpleNamespace(ns=0)
    monkeypatch.setattr(recording_startup_trace, "perf_counter_ns", lambda: value.ns)
    return value


def records(caplog):
    return [dict(part.split("=", 1) for part in row.message.split()[1:])
            for row in caplog.records if row.message.startswith("recording_startup ")]


def startup_host(tmp_path, enabled=False):
    host = main_host(SimpleNamespace(), tmp_path, streaming=False)
    host.recording_bridge = CapturingBridge()
    host.clicked_player_flag = True
    host._reserve_recorded_count_for_run = lambda: "run-1"
    # Attach the real admission implementation without replacing workflow guards.
    for name in ("_reserve_raw_audio_csv_recording", "_release_raw_audio_csv_recording",
                 "_release_raw_audio_csv_context"):
        setattr(host, name, getattr(SequenceWidgetRawCsvOpsMixin, name).__get__(host))
    host.raw_audio_csv_service = CsvTaskLedger()
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: enabled}
    host._csv_admission_notice = Mock()
    host.recent_test_sessions = []
    host.recent_session_panel = None
    return host


@pytest.mark.parametrize("entry", ["start_this_play", "judge_play_and_record"])
@pytest.mark.parametrize("enabled", [False, True])
def test_attempt_precedes_admission_and_maps_request_without_success_summary(
        ui_qapp, tmp_path, monkeypatch, caplog, clock, entry, enabled):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, enabled)
    reserve = host._reserve_raw_audio_csv_recording
    def observed_reserve(**kwargs):
        assert records(caplog)[0]["event"] == "gui_entry"
        return reserve(**kwargs)
    host._reserve_raw_audio_csv_recording = observed_reserve
    permit = host.raw_audio_csv_service.try_acquire_mutation
    host.raw_audio_csv_service.try_acquire_mutation = Mock(wraps=permit)
    assert getattr(host, entry)() is None
    rows = records(caplog)
    context = host._recording_contexts()[host.recording_bridge.request.request_id]
    assert context.startup_trace.trace_id == rows[0]["trace_id"]
    assert len({r["trace_id"] for r in rows}) == 1
    assert len([r for r in rows if r["stage"] == "csv_admission" and r["event"] == "begin"]) == 1
    assert len([r for r in rows if r["event"] == "request_link"]) == 1
    assert not [r for r in rows if r["event"] == "summary"]
    end = next(r for r in rows if r["stage"] == "csv_path_permit" and r["event"] == "end")
    assert end["outcome"] == "ok"
    assert {key: end[key] for key in ("sample_rate", "channel_count", "target_samples",
            "target_duration_seconds", "startup_trim_samples", "export_mode")} == {
        "sample_rate": "100", "channel_count": "2", "target_samples": "9",
        "target_duration_seconds": "0.09", "startup_trim_samples": "2",
        "export_mode": "wav_csv" if enabled else "wav"}
    host.raw_audio_csv_service.try_acquire_mutation.assert_called_once()
    assert context.startup_trace.dropped_events == 0
    assert len(rows) <= 43  # leave capacity for service / Qt and reserved events


def test_process_events_reentrant_attempt_rejected_with_separate_identity(
        ui_qapp, tmp_path, monkeypatch, caplog, clock):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, True)
    monkeypatch.setattr(analysis.QApplication, "processEvents", host.start_this_play)
    host.start_this_play()
    rows = records(caplog)
    entries = [r for r in rows if r["event"] == "gui_entry"]
    assert len(entries) == 2
    assert entries[0]["trace_id"] != entries[1]["trace_id"]
    summary, = [r for r in rows if r["event"] == "summary"]
    assert (summary["trace_id"], summary["outcome"]) == (entries[1]["trace_id"], "rejected")
    rejection, = [r for r in rows if r["event"] == "rejected"]
    assert rejection["trace_id"] == entries[1]["trace_id"]
    assert rejection["rejection_reason"] == summary["rejection_reason"] == "csv_workflow_busy"
    assert all(r["rejection_reason"] == "unknown" for r in rows
               if r["trace_id"] == entries[0]["trace_id"])
    assert host._recording_process_request.request_id == host._recording_contexts()[
        host._recording_process_request.request_id].startup_trace.request_id
    assert host.raw_audio_csv_service.snapshot().reserved == 1


@pytest.mark.parametrize("entry,reason", [
    ("start_this_play", "csv_capacity_full"),
    ("judge_play_and_record", "csv_service_unavailable"),
    ("start_this_play", "window_closing"),
    ("start_this_play", "workflow_prepare_denied"),
    ("start_this_play", "workflow_start_denied"),
    ("judge_play_and_record", "workflow_start_denied"),
    ("judge_play_and_record", "ve_config_unavailable"),
    ("judge_play_and_record", "service_busy"),
    ("start_this_play", "preflight_failed"),
    ("judge_play_and_record", "preflight_failed"),
    ("judge_play_and_record", "replay_unavailable"),
    ("judge_play_and_record", "metadata_preflight_denied"),
    ("start_this_play", "workflow_busy"),
    ("judge_play_and_record", "workflow_busy"),
])
def test_rejection_reason_uses_original_decision_once(
        ui_qapp, tmp_path, monkeypatch, caplog, clock, entry, reason):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, True)
    monkeypatch.setattr(analysis.QMessageBox, "warning", Mock())
    check = None
    if reason == "csv_capacity_full":
        for i in range(16):
            host.raw_audio_csv_service.reserve(str(i))
    elif reason == "csv_service_unavailable":
        host.raw_audio_csv_service.begin_shutdown()
    elif reason == "window_closing":
        host._close_in_progress = True
    elif reason == "workflow_prepare_denied":
        check = host._can_prepare_recording_workflow = Mock(return_value=False)
    elif reason in ("workflow_start_denied", "ve_config_unavailable"):
        host._can_prepare_recording_workflow = Mock(return_value=True)
        check = host._can_start_recording_workflow = Mock(return_value=False)
        if reason == "ve_config_unavailable":
            host._ve_recording_config_error = "unavailable profile"
    elif reason == "service_busy":
        check = host._can_start_recording_workflow = Mock(return_value=True)
        host.recording_bridge.service.can_start_recording = False
    elif reason == "preflight_failed":
        check = host.checked_work_status_message = Mock(return_value=True)
    elif reason == "metadata_preflight_denied":
        check = host._begin_test_round_metadata = Mock(return_value=False)
    elif reason == "workflow_busy":
        host._can_prepare_recording_workflow = None
        host._can_start_recording_workflow = None
        host._reserve_raw_audio_csv_recording = None
        host._record_workflow_busy = True
    reserve = host.raw_audio_csv_service.reserve
    host.raw_audio_csv_service.reserve = Mock(wraps=reserve)
    # Recording this reason must never add a diagnostic snapshot query.
    host.raw_audio_csv_service.snapshot = Mock(side_effect=AssertionError("extra snapshot"))
    kwargs = {"is_replay": True} if reason == "replay_unavailable" else {}
    assert getattr(host, entry)(**kwargs) is None
    rows = records(caplog)
    rejection, = [row for row in rows if row["event"] == "rejected"]
    summary, = [row for row in rows if row["event"] == "summary"]
    assert rejection["rejection_reason"] == summary["rejection_reason"] == reason
    assert rejection["trace_id"] == summary["trace_id"] == rows[0]["trace_id"]
    assert summary["outcome"] == "rejected"
    assert rows.index(rejection) < rows.index(summary)
    assert host.raw_audio_csv_service.reserve.call_count <= 1
    host.raw_audio_csv_service.snapshot.assert_not_called()
    if check is not None:
        check.assert_called_once()
    assert not hasattr(host.recording_bridge, "request")


@pytest.mark.parametrize("failure", ["preflight", "reset", "permit", "close"])
def test_synchronous_terminal_summary_once_preserves_cleanup(
        ui_qapp, tmp_path, monkeypatch, caplog, clock, failure):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, True)
    if failure == "preflight":
        host.checked_work_status_message = lambda: True
    elif failure == "reset":
        host.reset_work_pram = Mock(side_effect=ValueError("bad parameters"))
    elif failure == "permit":
        host.raw_audio_csv_service.try_acquire_mutation = Mock(return_value=None)
    else:
        monkeypatch.setattr(analysis.QApplication, "processEvents",
                            lambda: setattr(host, "_close_in_progress", True))
    host.start_this_play()
    rows = records(caplog)
    summary, = [r for r in rows if r["event"] == "summary"]
    assert summary["outcome"] == {"preflight": "rejected", "close": "cancelled"}.get(failure, "failed")
    assert not hasattr(host.recording_bridge, "request")
    assert host.raw_audio_csv_service.snapshot().reserved == 0
    if failure == "permit":
        ends = [r for r in rows if r["stage"] == "csv_path_permit" and r["event"] == "end"]
        assert len(ends) == 1 and ends[0]["outcome"] == "failed"
        host.raw_audio_csv_service.try_acquire_mutation.assert_called_once()


def test_recent_session_delay_is_attributed_separately(ui_qapp, tmp_path, caplog, clock):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    original = host._begin_recent_session_for_current_run
    def delayed():
        clock.ns += 2_200_000_000
        original()
    host._begin_recent_session_for_current_run = delayed
    host.judge_play_and_record()
    ends = {r["stage"]: r for r in records(caplog) if r["event"] == "end"}
    assert ends["recent_session"]["elapsed_ms"] == "2200.0"
    assert ends["request_build"]["elapsed_ms"] == "0.0"


def test_first_analysis_window_cleanup_has_its_own_boundary(ui_qapp, tmp_path, caplog, clock):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    calls = []
    def close():
        if not calls:
            clock.ns += 2_200_000_000
        calls.append(True)
    host._close_analysis_windows = close
    host.start_this_play()
    ends = {r["stage"]: r for r in records(caplog) if r["event"] == "end"}
    assert ends["entry_ui_cleanup"]["elapsed_ms"] == "2200.0"
    assert ends["ui_cleanup"]["elapsed_ms"] == "0.0"
    assert len(calls) == 2


def test_entry_background_uses_only_cached_state(ui_qapp, tmp_path, caplog, clock):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host._analysis_active_request = SimpleNamespace(task_id="analysis-A", instances=(
        SimpleNamespace(analysis_type="SPL"), SimpleNamespace(analysis_type="FFT")))
    host._analysis_task_queue = deque([object(), object()])
    host._raw_audio_csv_cached_snapshot = SimpleNamespace(active=1, queued=3)
    host.raw_audio_csv_service.snapshot = Mock(side_effect=AssertionError("extra query"))
    host.judge_play_and_record()
    entry = records(caplog)[0]
    assert entry["analysis_active"] == "True"
    assert entry["analysis_queued"] == "2"
    assert entry["analysis_items"] == "SPL,FFT"
    assert entry["csv_active"] == "1" and entry["csv_queued"] == "3"
    host.raw_audio_csv_service.snapshot.assert_not_called()


@pytest.mark.parametrize("slow_stage", ["parameters", "mac_address", "directory"])
def test_real_reset_attributes_preparation_and_preserves_result(
        ui_qapp, tmp_path, monkeypatch, caplog, clock, slow_stage):
    from base import play_and_record
    from base.recording_startup_trace import RecordingStartupTrace
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host.data_struct = SimpleNamespace(sample_rate=100, clear_data=lambda: None)
    host.lineedit_type = SimpleNamespace(text=lambda: "model")
    host.lineedit_s_or_n = SimpleNamespace(text=lambda: "barcode")
    host._resolve_recording_name_suffix = lambda: ""
    host._get_active_product_condition_key = lambda: ""
    host._snapshot_recording_input_channels = lambda rec: (0, 2)
    host.sequence_config[0]["seq1"]["acq"]["detail"].update(total_time=1,
        recording_root_directory=str(tmp_path))
    def delay(value, stage):
        if slow_stage == stage:
            clock.ns += 2_200_000_000
        return value
    mac = Mock(side_effect=lambda: delay("AA:BB", "mac_address"))
    monkeypatch.setattr(play_and_record, "get_mac_address", mac)
    monkeypatch.setattr(play_and_record.FileOps, "get_recording_store_dir",
                        lambda *a: delay(str(tmp_path / "audio"), "directory"))
    monkeypatch.setattr(analysis.LoadUiConfig, "get_rec_and_play_dict_base_sequence_dict",
                        lambda *a: delay(({}, {"num_frames": 100}), "parameters"))
    trace = RecordingStartupTrace(host.default_logger, process="parent")
    recorded, rate = analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram(
        host, "not_labeled", startup_trace=trace)
    assert rate == 100 and recorded["num_frames"] == 102
    assert recorded["startup_trim_samples"] == 2
    assert host._active_input_channels == [0, 2]
    assert (tmp_path / "audio").is_dir()
    mac.assert_called_once()
    ends = {r["stage"]: r for r in records(caplog) if r["event"] == "end"}
    assert ends[slow_stage]["elapsed_ms"] == "2200.0"
    assert ends["parameters"]["sample_rate"] == "100"
    assert ends["parameters"]["target_samples"] == "102"
    for stage in {"parameters", "mac_address", "directory"} - {slow_stage}:
        assert ends[stage]["elapsed_ms"] == "0.0"

    # Exercise the complete entry chain with the real reset implementation too.
    caplog.clear()
    host.reset_work_pram = analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram.__get__(host)
    host.start_this_play()
    rows = records(caplog)
    assert len(rows) == 43
    assert not [row for row in rows if row["event"] == "summary"]
    context = host._recording_contexts()[host._recording_process_request.request_id]
    assert context.startup_trace.dropped_events == 0
    assert len({row["stage"] for row in rows if row["event"] == "begin"}) == 20


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_synchronous_bridge_terminal_finishes_owned_trace_once(
        ui_qapp, tmp_path, caplog, clock, terminal):
    from base.recording_process_protocol import RecordingFailure
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, True)
    bridge = host.recording_bridge
    bridge.service.is_path_leased = lambda path: False
    original_start = bridge.start
    def terminal_start(request, callbacks, *, startup_trace=None):
        session = original_start(request, callbacks)
        if terminal == "failed":
            callbacks.failed(session, RecordingFailure(request.request_id, "open", request.path,
                                                      "device failure", handles_released=True))
        else:
            callbacks.cancelled(session, None)
        return session
    bridge.start = terminal_start
    host.start_this_play()
    summary, = [row for row in records(caplog) if row["event"] == "summary"]
    assert summary["outcome"] == terminal
    assert summary["request_id"] == bridge.request.request_id
    assert not host._recording_contexts()
    assert host.raw_audio_csv_service.snapshot().reserved == 0


def test_submit_background_refresh_and_trace_transfer(ui_qapp, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host._raw_audio_csv_cached_snapshot = SimpleNamespace(active=0, queued=0)
    host._analysis_active_request = SimpleNamespace(task_id="analysis-A", instances=(
        SimpleNamespace(analysis_type="SPL"),))
    host._analysis_task_queue = deque()
    original = host._begin_recent_session_for_current_run
    def change_background():
        original()
        host._raw_audio_csv_cached_snapshot = SimpleNamespace(active=1, queued=4)
        host._analysis_active_request = SimpleNamespace(task_id="analysis-B", instances=(
            SimpleNamespace(analysis_type="FFT"),))
        host._analysis_task_queue.append(object())
    host._begin_recent_session_for_current_run = change_background
    captured = []
    start = host.recording_bridge.start
    def traced_start(request, callbacks, *, startup_trace=None):
        captured.append(startup_trace)
        return start(request, callbacks)
    host.recording_bridge.start = traced_start
    host.start_this_play()
    rows = records(caplog)
    assert rows[0]["csv_active"] == "0"
    submit, = [r for r in rows if r["event"] == "service_submit"]
    assert submit["csv_active"] == "1" and submit["csv_queued"] == "4"
    assert submit["domain"] == "GUI"
    assert rows[0]["analysis_task_id"] == "analysis-A"
    assert rows[0]["analysis_items"] == "SPL"
    assert submit["analysis_task_id"] == "analysis-B" and submit["analysis_items"] == "FFT"
    assert submit["analysis_queued"] == "1"
    assert captured[0] is host._recording_contexts()[host.recording_bridge.request.request_id].startup_trace


@pytest.mark.parametrize("terminal", [None, "failed", "cancelled"])
def test_real_qt_queue_delay_callback_and_late_event(ui_qapp, tmp_path, caplog, terminal):
    import time
    from base.recording_service import RecordingCallbacks, RecordingSession
    from base.recording_startup_trace import RecordingStartupTrace
    from ui.recording_service_bridge import RecordingServiceBridge
    from unit_test.base.test_recording_service import request
    caplog.set_level(logging.INFO)
    trace = RecordingStartupTrace(logging.getLogger("test.qt.startup"), process="parent")
    trace.link_request("one")
    session = RecordingSession(SimpleNamespace(), request(tmp_path), RecordingCallbacks())
    session.startup_trace = trace
    from dataclasses import replace
    session._startup_timing = replace(session._startup_timing, accepted_seconds=time.perf_counter())
    session._observe_capture_started()
    bridge = RecordingServiceBridge(SimpleNamespace())
    called = []
    def started(s):
        called.append(s)
        if terminal == "failed":
            raise RuntimeError("UI callback failure")
    bridge._callbacks["one"] = RecordingCallbacks(started=started)
    bridge._enqueue("started", session, None)
    if terminal == "cancelled":
        trace.finish("cancelled", domain="service_supervisor")
    else:
        time.sleep(2.2)
    ui_qapp.processEvents()
    bridge._enqueue("started", session, None)
    ui_qapp.processEvents()
    summary, = [r for r in records(caplog) if r["event"] == "summary"]
    assert summary["outcome"] == (terminal or "started")
    assert summary["completion_boundary"] == "qt_callback_return"
    assert len(called) == 1
    if terminal != "cancelled":
        delivery, = [r for r in records(caplog) if r["event"] == "qt_delivery"]
        assert float(delivery["capture_to_qt_ms"]) >= 2000
        assert session.startup_timing.qt_delivery_seconds >= 2
        assert not [r for r in records(caplog) if r["stage"] in ("wav_open", "device_bind")]


def test_gui_service_and_qt_share_one_trace_and_summary(ui_qapp, tmp_path, monkeypatch, caplog):
    from base.recording_service import RecordingService, _Worker
    from base.recording_process_protocol import RecordingEvent
    from ui.recording_service_bridge import RecordingServiceBridge
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(RecordingService, "_start_thread", lambda *a, **k: None)
    service = RecordingService()
    worker = _Worker(7, SimpleNamespace(pid=987), None, None, None)
    worker.ready = True
    worker.control = SimpleNamespace(send=lambda event: worker.stop.set())
    service._worker = worker
    host = startup_host(tmp_path)
    host.recording_bridge = RecordingServiceBridge(service)
    host.start_this_play()
    session = service._capture_session
    context = host._recording_contexts()[session.request.request_id]
    assert session.startup_trace is context.startup_trace
    service._dispatch(service._inbox.get_nowait())
    service._send(worker)
    service._event(worker, RecordingEvent(7, session.request.request_id, "started"))
    assert not [r for r in records(caplog) if r["event"] == "summary"]
    ui_qapp.processEvents()
    rows = records(caplog)
    summary, = [r for r in rows if r["event"] == "summary"]
    assert summary["outcome"] == "started"
    assert summary["completion_boundary"] == "qt_callback_return"
    assert summary["generation"] == "7" and summary["worker_pid"] == "987"
    assert len({r["trace_id"] for r in rows}) == 1
    assert rows[-2]["event"] == "callback_return"
    assert len(rows) <= 64
    assert summary["dropped_events"] == summary["dropped_fields"] == "0"


def test_gui_cancel_finishes_before_service_notification(ui_qapp, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host.recording_bridge.service.cancel = Mock()
    host.start_this_play()
    host._cancel_process_recording()
    summary, = [r for r in records(caplog) if r["event"] == "summary"]
    assert summary["outcome"] == "cancelled"
