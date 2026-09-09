"""PR1C review regressions across real request completion and cleanup boundaries."""
from types import SimpleNamespace
from unittest import mock

from ui.sequence import sequence_widget_streaming_ops as streaming
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from unit_test.ui.test_recording_process_integration import (
    _late_context, main_host, pump, service,
)
from unit_test.ui.test_recording_result_overlap import _result_session


def test_label_persistence_retry_precedes_count_and_tcp_completion(
        ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(service, tmp_path)
    context = _late_context(host, "retry-label", tmp_path / "A.wav", object(),
        direction="", session_id="session-A", tcp=("127.0.0.1", 1001))
    context.analysis_required = True
    context.count_mode = "test"
    context.recorded_signal_info["labels"] = "not_labeled"
    context.recent_session_config_snapshot = {"analysis_config": {
        "display_sequence": ["spl"], "spl": {
            "type": "SPL", "analysis_channel": 0, "limit_checked": True,
            "limit_metric": "overall_spl", "scalar_upper_enabled": True,
            "scalar_upper_value": 200.0, "scalar_lower_enabled": True,
            "scalar_lower_value": -200.0}}}
    session, audio = _result_session("retry-label", str(tmp_path / "A.wav"))
    context.session, context.accepted_audio = session, audio
    host._recording_process_contexts = {context.request.request_id: context}
    host._active_recording_process_id = "B"
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = (
        streaming.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    scheduled = []
    host._schedule_request_scoped_recording_retry = lambda _delay, callback: scheduled.append(callback)
    shared_count = mock.Mock(return_value=True)
    monkeypatch.setattr(counts, "increment_shared_result", shared_count)

    class DatabaseBoundary:
        save_calls = 0
        update_calls = 0
        stored_label = None

        def save_signal_info_to_db(self, signal_info, stimulus):
            self.save_calls += 1
            self.stored_label = signal_info["labels"]
            return 0, "saved"

        def update_audio_label(self, signal_info, path):
            assert path == context.request.path
            self.update_calls += 1
            if self.update_calls == 1:
                return 1, "temporary label-write failure"
            self.stored_label = signal_info["labels"]
            return 0, "updated"

    database = DatabaseBoundary()
    monkeypatch.setattr(streaming, "RecordingManager", lambda: database)
    host._publish_recording_context(context)
    pump(ui_qapp, lambda: bool(scheduled))
    assert database.update_calls == 1
    assert database.stored_label == "not_labeled"
    assert not context.business_completed and not context.publication_delivered
    assert "database:analysis-label" not in context.business_effect_ledger
    shared_count.assert_not_called()
    host._send_recording_tcp_finish.assert_not_called()

    scheduled.pop(0)()
    pump(ui_qapp, lambda: context.publication_delivered)
    assert database.update_calls == 2
    assert database.stored_label == "OK"
    assert database.save_calls == 1
    assert context.business_effect_attempts["analysis:compute"] == 1
    assert context.business_completed
    shared_count.assert_called_once_with("OK")
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1001))
    host._publish_recording_context(context)
    assert database.update_calls == 2
    shared_count.assert_called_once()
    host._send_recording_tcp_finish.assert_called_once()


def test_old_cancellation_drops_only_old_context_with_real_cleanup_available(
        ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    context_a = _late_context(host, "A", tmp_path / "A.wav", object(),
        direction="forward", session_id="session-A", tcp=None)
    token_b = object()
    context_b = _late_context(host, "B", tmp_path / "B.wav", token_b,
        direction="reverse", session_id="session-B", tcp=None)
    session_a, _ = _result_session("A", str(tmp_path / "A.wav"))
    session_b, _ = _result_session("B", str(tmp_path / "B.wav"))
    context_a.session, context_b.session = session_a, session_b
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_process_session = session_b
    host._recording_workflow_token = token_b
    host.streaming_processor = processor_b = object()
    host._recording_ve_device = device_b = {"backend": "vkinging", "owner": "B"}
    host._recording_wav_calibration_metadata = metadata_b = {"owner": "B"}
    host._recording_input_channels = (7, 1)
    host.player_status_flag = host._record_workflow_busy = True
    host.sn_locked = host._serial_product_condition_executing = True
    host._abort_recording_channel_selection = mock.Mock(
        side_effect=lambda: setattr(host, "_recording_input_channels", None))
    host._unlock_sn_after_recording_if_needed = mock.Mock(
        side_effect=lambda: setattr(host, "sn_locked", False))
    host._on_serial_product_runtime_error = mock.Mock(
        side_effect=lambda _reason: setattr(host, "_serial_product_condition_executing", False))
    host._cleanup_failed_recording_initialization = mock.Mock(wraps=(
        SequenceWidgetAnalysisOpsMixin._cleanup_failed_recording_initialization.__get__(host)))
    host._discard_current_recent_session = mock.Mock()

    host._on_process_recording_cancelled(session_a, SimpleNamespace())

    assert context_a.cancelled and context_a.final
    assert host._recording_process_contexts == {"B": context_b}
    assert host._active_recording_process_id == "B"
    assert host._recording_process_session is session_b
    assert host.streaming_processor is processor_b
    assert host._recording_ve_device is device_b
    assert host._recording_wav_calibration_metadata is metadata_b
    assert host._recording_input_channels == (7, 1)
    assert host.player_status_flag and host._record_workflow_busy
    assert host.sn_locked and host._serial_product_condition_executing
    host._cleanup_failed_recording_initialization.assert_not_called()
    host._discard_current_recent_session.assert_not_called()
    host.update_player_btn_is_paused.assert_not_called()
