from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
from ui.sequence import sequence_widget_test_metadata_ops as test_metadata_ops
from ui.sequence.sequence_widget_test_metadata_ops import SequenceWidgetTestMetadataOpsMixin


class _MetadataHost(
    SequenceWidgetTestMetadataOpsMixin,
    SequenceWidgetBarcodeOpsMixin,
    SequenceWidgetSerialTriggerOpsMixin,
    SequenceWidgetAnalysisOpsMixin,
    QWidget,
):
    def __init__(self):
        QWidget.__init__(self)
        self.toolsbar = SequenceToolsBar()
        self.left_panel = Mock()
        self.lineedit_s_or_n = self.toolsbar.lineedit_s_or_n
        self.lineedit_type = self.toolsbar.lineedit_type
        self.barcode_scanner_box = self.toolsbar.barcode_scanner_box
        self.replayer_btn = self.toolsbar.replayer_btn
        self.data_btn = self.toolsbar.data_btn
        self.barcode_scanner_box.setChecked(False)
        self.product_test_condition_configs = [
            {"key": "a", "condition_name": "0.1", "test_queue": "q1"},
            {"key": "b", "condition_name": "0.3", "test_queue": "q2"},
        ]
        self.product_test_close_trigger_state = ""
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._current_cycle_recorded_count = None
        self._current_trigger_direction = ""
        self._current_run_recording_token = "run-1"
        self._serial_product_condition_executing = False
        self._sn_locked_for_product_round = False
        self._sn_locked_for_cycle = False
        self._record_workflow_busy = False
        self.player_status_flag = False
        self.count_board = SimpleNamespace(mode="mark")
        self.default_logger = Mock()
        self.sequence_config = [
            {"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {"total_time": 4}}}}
        ]
        self.analysis_config = {}
        self.recorded_signal_info = {}
        self.data_struct = SimpleNamespace(sample_rate=44100, clear_data=Mock())
        self.mic = {"name": "test input"}
        self.mic_channels = [0, 1, 2, 3, 4]
        self._snapshot_recording_input_channels = Mock(return_value=tuple(self.mic_channels))
        self.channel_workspace = None
        self._load_sequence_config_for_product_condition = Mock(return_value=(True, ""))
        self._generate_recording_token = Mock(side_effect=["round-1", "round-2"])
        self.clear_all_direction_waveforms = Mock()
        self._clear_plot_area = Mock()
        self.update_player_btn_is_paused = Mock()
        self._init_test_round_metadata()


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(
        test_metadata_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: None),
    )
    monkeypatch.setattr(
        test_metadata_ops,
        "save_recorded_data_to_json",
        Mock(),
    )
    app = QApplication.instance() or QApplication([])
    widget = _MetadataHost()
    widget.toolsbar.sample_number_lineedit.setText("  SAMPLE-001  ")
    widget.toolsbar.current_round_spinbox.setValue(7)
    yield widget
    widget.toolsbar.close()
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_empty_sample_blocks_before_preparing_a_condition(host, text):
    host.toolsbar.sample_number_lineedit.setText(text)
    with patch("ui.sequence.sequence_widget_test_metadata_ops.QMessageBox.warning") as warning:
        assert host._prepare_next_manual_product_condition_recording() is None
    warning.assert_called_once()
    host._load_sequence_config_for_product_condition.assert_not_called()
    assert host._manual_product_condition_group_id == ""
    assert host._test_round_metadata is None


@pytest.mark.parametrize("text", ["", "0", "-1", "1.5", "10000", "abc"])
def test_invalid_round_is_not_silently_replaced_before_validation(host, text):
    host.toolsbar.current_round_spinbox.lineEdit().setText(text)
    with patch("ui.sequence.sequence_widget_test_metadata_ops.QMessageBox.warning"):
        assert not host._validate_test_round_metadata()


def test_serial_reentry_does_not_open_another_validation_dialog(host):
    host.toolsbar.sample_number_lineedit.clear()
    with patch(
        "ui.sequence.sequence_widget_test_metadata_ops.QMessageBox.warning",
        side_effect=lambda *_: host._validate_test_round_metadata(),
    ) as warning:
        assert not host._validate_test_round_metadata()
    assert warning.call_count == 1


def test_repeated_frames_with_same_missing_input_do_not_repeat_warning(host):
    host.toolsbar.sample_number_lineedit.clear()
    with patch("ui.sequence.sequence_widget_test_metadata_ops.QMessageBox.warning") as warning:
        for _ in range(3):
            assert not host._validate_test_round_metadata()
    assert warning.call_count == 1


def test_serial_missing_microphone_does_not_leave_executing_or_locked(host):
    host.mic = None
    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        assert not host._start_serial_product_condition("01 04")
    assert not host._serial_product_condition_executing
    assert host._manual_product_condition_group_id == ""
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()


def test_round_uses_one_snapshot_and_unlocks_only_after_all_conditions(host):
    assert host._prepare_next_manual_product_condition_recording() is True
    host._reserve_recorded_count_for_run()
    assert host.toolsbar.sample_number_lineedit.text() == "SAMPLE-001"
    assert host.toolsbar.sample_number_lineedit.isReadOnly()
    assert host.toolsbar.current_round_spinbox.isReadOnly()
    first = {}
    host._attach_test_round_metadata(first)
    host._advance_manual_product_condition_cycle_after_recording()
    assert host.toolsbar.sample_number_lineedit.isReadOnly()

    # Programmatic changes cannot change already captured or future round records.
    host.toolsbar.sample_number_lineedit.setText("OTHER")
    host.toolsbar.current_round_spinbox.setValue(99)
    assert host._prepare_next_manual_product_condition_recording() is True
    second = {}
    host._attach_test_round_metadata(second)
    assert first == second == {"sample_number": "SAMPLE-001", "test_round": 7}
    host.left_panel.set_current_round.assert_called_with(7)
    host._advance_manual_product_condition_cycle_after_recording()
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()
    assert not host.toolsbar.current_round_spinbox.isReadOnly()
    assert host._test_round_metadata is None
    assert host._begin_test_round_metadata()
    assert host._test_round_metadata == {"sample_number": "OTHER", "test_round": 99}
    assert first == {"sample_number": "SAMPLE-001", "test_round": 7}


def test_lock_blocks_keyboard_editing_without_changing_field_geometry(host):
    before = host.toolsbar.current_round_spinbox.size()
    host._begin_test_round_metadata()
    QTest.keyClick(host.toolsbar.current_round_spinbox, Qt.Key_Up)
    QTest.keyClicks(host.toolsbar.sample_number_lineedit, "changed")
    assert host.toolsbar.current_round_spinbox.value() == 7
    assert host.toolsbar.sample_number_lineedit.text() == "SAMPLE-001"
    assert host.toolsbar.current_round_spinbox.size() == before


def test_serial_close_keeps_fields_locked_until_round_is_closed(host):
    host._prepare_next_manual_product_condition_recording()
    host._reserve_recorded_count_for_run()
    group_id = host._manual_product_condition_group_id
    host.product_test_close_trigger_state = "01 04"
    host._serial_product_condition_executing = True
    host._manual_product_condition_completed_keys = {"a", "b"}
    host._advance_manual_product_condition_cycle_after_recording()
    assert host._serial_product_waiting_for_close
    assert host.toolsbar.sample_number_lineedit.isReadOnly()
    host._finish_serial_product_round(group_id, "01 04")
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()
    assert host.toolsbar.current_round_spinbox.value() == 7


def test_missing_microphone_releases_first_prepared_round(host):
    host._prepare_next_manual_product_condition_recording()
    host.mic = None
    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host.start_this_play()
    assert host._manual_product_condition_group_id == ""
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()


@pytest.mark.parametrize("partly_complete", [False, True])
def test_import_cancel_releases_only_an_empty_round(host, partly_complete):
    host._prepare_next_manual_product_condition_recording()
    if partly_complete:
        host._manual_product_condition_completed_keys = {"a"}
        host._manual_product_condition_index = 1
        host._prepare_next_manual_product_condition_recording()
    host._abort_imported_product_condition_step()
    assert host.toolsbar.sample_number_lineedit.isReadOnly() is partly_complete
    assert bool(host._manual_product_condition_group_id) is partly_complete


@pytest.mark.parametrize("partly_complete", [False, True])
def test_recording_service_busy_releases_only_an_empty_round(host, partly_complete):
    host._prepare_next_manual_product_condition_recording()
    if partly_complete:
        host._manual_product_condition_completed_keys = {"a"}
        host._manual_product_condition_index = 1
        host._prepare_next_manual_product_condition_recording()
    host.recording_bridge = SimpleNamespace(service=SimpleNamespace(busy=True))
    host._start_process_recording = Mock()

    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host.judge_play_and_record()

    host._start_process_recording.assert_not_called()
    assert host.toolsbar.sample_number_lineedit.isReadOnly() is partly_complete
    assert host.toolsbar.current_round_spinbox.isReadOnly() is partly_complete
    if partly_complete:
        assert host._test_round_metadata == {"sample_number": "SAMPLE-001", "test_round": 7}
    else:
        assert host._test_round_metadata is None
        assert host._manual_product_condition_group_id == ""


def test_manual_failure_releases_fields_and_cannot_report_success(host):
    host._prepare_next_manual_product_condition_recording()
    host._record_workflow_busy = True
    host.player_status_flag = True
    with patch("ui.sequence.sequence_widget_test_metadata_ops.QMessageBox.warning"):
        assert host._on_serial_product_runtime_error("保存失败") is True
    assert host._test_round_metadata is None
    assert not host._record_workflow_busy
    assert not host.player_status_flag
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()


def test_record_parameters_receive_snapshot_not_current_controls(host):
    host._prepare_next_manual_product_condition_recording()
    host.toolsbar.sample_number_lineedit.setText("CHANGED")
    with patch(
        "ui.sequence.sequence_widget_analysis_ops.get_recorded_info",
        return_value=("test.wav", {"file_path": "test.wav"}),
    ), patch(
        "ui.sequence.sequence_widget_analysis_ops.LoadUiConfig.get_rec_and_play_dict_base_sequence_dict",
        return_value=(None, {}),
    ):
        host.reset_work_pram("not_labeled")
    assert host.recorded_signal_info["sample_number"] == "SAMPLE-001"
    assert host.recorded_signal_info["test_round"] == 7


def test_result_uses_saved_metadata_after_inputs_change(host):
    host._prepare_next_manual_product_condition_recording()
    info = {"file_path": "test.wav", "labels": "OK"}
    host._attach_test_round_metadata(info)
    host.recorded_signal_info = info
    host.recorded_path = "test.wav"
    host._recent_session_seq = 0
    host._build_recent_session_config_snapshot = Mock(return_value={})
    record = host._build_recent_session_record("OK")
    assert record["sample_number"] == "SAMPLE-001"
    assert record["test_round"] == 7
