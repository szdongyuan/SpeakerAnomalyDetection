from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.raw_audio_csv_tasks import CsvTaskLedger
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class Host(SequenceWidgetRawCsvOpsMixin,
           SequenceWidgetAnalysisOpsMixin):
    def __init__(self, enabled=True):
        self.raw_audio_csv_service = CsvTaskLedger()
        self.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: enabled}
        self.default_logger = Mock()
        self._reserve_recorded_count_for_run = Mock()
        self.checked_work_status_message = Mock(return_value=False)
        self.clicked_player_flag = False
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.judge_play_and_record = Mock()
        self._csv_admission_notice = Mock()


def fill(host, count=16):
    return [host.raw_audio_csv_service.reserve(str(i)).reservation for i in range(count)]


def test_full_queries_do_not_allocate_and_start_has_no_side_effects():
    host = Host()
    fill(host)
    for _ in range(20):
        assert host._can_prepare_recording_workflow() is False
        assert host._can_start_recording_workflow() is False
    host.start_this_play()
    host._reserve_recorded_count_for_run.assert_not_called()
    host.checked_work_status_message.assert_not_called()
    host.judge_play_and_record.assert_not_called()
    assert host.raw_audio_csv_service.snapshot().outstanding == 16


def test_full_csv_does_not_block_wav_only_or_calibration():
    host = Host(enabled=False)
    fill(host)
    assert host._can_start_recording_workflow()
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = True
    assert host._can_start_calibration_workflow()


def test_sixteenth_reservation_is_reused_and_snapshot_survives_toggle():
    host = Host()
    fill(host, 15)
    admission = host._reserve_raw_audio_csv_recording()
    assert admission.csv_enabled_snapshot
    assert host._can_start_recording_workflow()
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = False
    assert host._reserve_raw_audio_csv_recording() is admission
    assert host.raw_audio_csv_service.snapshot().outstanding == 16
    host._release_raw_audio_csv_recording(admission)
    assert host.raw_audio_csv_service.snapshot().outstanding == 15


def test_failed_preflight_releases_only_its_reservation():
    host = Host()
    host.checked_work_status_message.return_value = True
    host.start_this_play()
    assert host.raw_audio_csv_service.snapshot().outstanding == 0


def test_manual_preparation_full_rejects_before_metadata_and_condition_load():
    host = Host()
    fill(host)
    host._validate_test_round_metadata = Mock(return_value=True)
    host._load_sequence_config_for_product_condition = Mock()
    assert host._prepare_next_manual_product_condition_recording() is None
    host._validate_test_round_metadata.assert_not_called()
    host._load_sequence_config_for_product_condition.assert_not_called()



def test_serial_atomic_reserve_failure_keeps_original_index_and_round():
    from unit_test.test_serial_product_condition_runtime import _SerialProductHost, _payload, FRAME_6000
    class SerialHost(SequenceWidgetRawCsvOpsMixin, _SerialProductHost):
        pass
    host = SerialHost()
    host.raw_audio_csv_service = CsvTaskLedger()
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: True}
    host._csv_admission_notice = Mock()
    host._manual_product_condition_index = 2
    # Pure preflight saw free capacity, another window filled it before reserve.
    host._can_prepare_recording_workflow = lambda: True
    fill(host)
    host.on_serial_full_frame_received(_payload(FRAME_6000))
    assert host._manual_product_condition_index == 2
    assert host.loaded_queues == []
    assert host._manual_product_condition_group_id == ''
    assert host.started == []


def directional_host():
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
    class DirectionHost(Host, SequenceWidgetBarcodeOpsMixin):
        pass
    host = DirectionHost()
    host._queued_directional_trigger = ''
    host._current_cycle_recorded_count = 'previous-round'
    host._reset_direction_cycle_panel_state = Mock()
    host._should_lock_sn_for_cycle = Mock(return_value=False)
    host._set_active_recording_direction = Mock()
    host._is_test_mode = lambda: False
    return host


def test_csv_full_directional_trigger_is_dropped_without_replay_queue():
    host = directional_host()
    tokens = fill(host)
    host.on_directional_triggered('forward')
    assert host._queued_directional_trigger == ''
    host.raw_audio_csv_service.release_reservation(tokens[0])
    host._reset_direction_cycle_panel_state.assert_not_called()
    host.judge_play_and_record.assert_not_called()


def test_delayed_atomic_reserve_failure_precedes_direction_mutations():
    host = directional_host()
    host._pending_serial_trigger_direction = 'forward'
    host._can_start_recording_workflow = lambda: True
    fill(host)
    host._on_serial_trigger_delay_timeout()
    assert host._current_cycle_recorded_count == 'previous-round'
    host._reset_direction_cycle_panel_state.assert_not_called()
    host._set_active_recording_direction.assert_not_called()
    host.judge_play_and_record.assert_not_called()
    assert host._pending_serial_trigger_direction == ''
    assert host._queued_directional_trigger == ''


def test_old_preflight_cleanup_preserves_reentrant_new_reservation():
    host = Host()
    old = host._reserve_raw_audio_csv_recording()
    newer = []
    def reject_after_reentry():
        host._release_raw_audio_csv_recording(old)
        newer.append(host._reserve_raw_audio_csv_recording())
        return True
    host.checked_work_status_message = reject_after_reentry
    host.start_this_play()
    assert host._pending_raw_audio_csv_recording is newer[0]
    assert host.raw_audio_csv_service.snapshot().reserved == 1
    host._release_raw_audio_csv_recording(newer[0])


def test_capacity_notifications_refresh_button_without_reserving_or_replaying(ui_qapp):
    from base.raw_audio_csv_protocol import CsvServiceEvent
    host = Host()
    host.player_btn = Mock()
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    host.sequence_config = [{}]
    host.update_player_btn_is_paused = Mock(
        side_effect=lambda: SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host))
    host._drain_queued_directional_trigger = Mock()
    tokens = fill(host)
    host._on_raw_audio_csv_service_event(CsvServiceEvent('state', host.raw_audio_csv_service.snapshot()))
    host.player_btn.setDisabled.assert_called_with(True)
    assert '16' in host.player_btn.setToolTip.call_args.args[0]
    assert host.raw_audio_csv_service.snapshot().reserved == 16
    host.raw_audio_csv_service.release_reservation(tokens[0])
    host._on_raw_audio_csv_service_event(CsvServiceEvent('state', host.raw_audio_csv_service.snapshot()))
    assert host.update_player_btn_is_paused.call_count == 2
    assert host._can_prepare_recording_workflow()
    assert host.raw_audio_csv_service.snapshot().reserved == 15
    host._drain_queued_directional_trigger.assert_not_called()


def test_delayed_csv_full_rejection_logs_csv_reason_and_does_not_replay():
    host = directional_host()
    host._pending_serial_trigger_direction = 'forward'
    fill(host)
    host._on_serial_trigger_delay_timeout()
    assert any('CSV' in str(call) and '16' in str(call)
               for call in host.default_logger.info.call_args_list)
    assert host._pending_serial_trigger_direction == ''
    assert host._queued_directional_trigger == ''
    host._reset_direction_cycle_panel_state.assert_not_called()


def test_regular_player_button_refresh_preserves_csv_reason_and_toggle_recovery(ui_qapp):
    from PyQt5.QtWidgets import QPushButton
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    host = Host()
    host.player_btn = QPushButton()
    host.sequence_config = [{}]
    fill(host)
    reason = host._raw_audio_csv_admission_reason()
    host.player_btn.setToolTip(reason)
    SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
    assert not host.player_btn.isEnabled()
    assert host.player_btn.toolTip() == reason
    host.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = False
    SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
    assert host.player_btn.isEnabled()
    assert host.player_btn.toolTip() == '开始录制'


def test_busy_reentrant_start_cannot_consume_existing_sixteenth_reservation():
    host = Host()
    fill(host, 15)
    outer = host._reserve_raw_audio_csv_recording()
    host._record_workflow_busy = True
    host.start_this_play()
    assert host.raw_audio_csv_service.snapshot().outstanding == 16
    assert host._pending_raw_audio_csv_recording is outer
    assert outer.csv_reservation is not None
    host._release_raw_audio_csv_recording(outer)


def test_nested_early_return_borrows_active_scope_then_prepare_handoff_owns_cleanup():
    from ui.sequence.sequence_widget_raw_csv_ops import CsvRecordingAdmissionScope
    host = Host()
    with CsvRecordingAdmissionScope(host) as preparation:
        admission = preparation.admission
        host.checked_work_status_message.return_value = True
        host.start_this_play()
        assert host._pending_raw_audio_csv_recording is admission
        assert host.raw_audio_csv_service.snapshot().reserved == 1
        preparation.keep = True
    # Preparation has returned successfully. The later independent startup now
    # owns failure rollback even though it reuses the existing admission.
    host.start_this_play()
    assert host._pending_raw_audio_csv_recording is None
    assert host.raw_audio_csv_service.snapshot().reserved == 0


@pytest.mark.parametrize('csv_full', [True, False])
def test_ordinary_serial_busy_rejection_reports_actual_csv_reason(csv_full):
    from unit_test.test_serial_product_condition_runtime import _SerialProductHost, _payload, FRAME_6000
    class SerialHost(SequenceWidgetRawCsvOpsMixin, SequenceWidgetRecordingProcessOpsMixin,
                     _SerialProductHost):
        pass
    host = SerialHost()
    host.raw_audio_csv_service = CsvTaskLedger()
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: True}
    if csv_full:
        fill(host)
    else:
        host._record_workflow_busy = True
    host.on_serial_full_frame_received(_payload(FRAME_6000))
    logs = [message for level, message in host.default_logger.messages]
    if csv_full:
        assert any('CSV' in message and '16' in message for message in logs)
    else:
        assert any('serial_product_frame_ignored_manual_busy' in message for message in logs)
        assert not any('csv_busy' in message for message in logs)
    assert host.started == []
    assert host.loaded_queues == []
