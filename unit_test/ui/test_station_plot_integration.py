"""Recording and analysis business remains independent of plot availability."""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from base.recording_final_result import prepare_main_recording_result
from base.recording_process_protocol import RecordingFinalEnvelope, RecordingQualityEvidence
from ui.sequence.recording_process_context import RecordingProcessContext
from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin
from unit_test.ui.test_product_test_program_runtime_refresh import refresh_host
from unit_test.ui.test_round_reset import host as round_reset_host, start_round
from unit_test.base.recording_process_fakes import known_audio
from unit_test.base.test_recording_finalization import prepared_ve_result
from unit_test.ui.test_recording_process_integration import (
    assert_saved_main_envelope, main_host, pump, service,
)


@pytest.mark.parametrize("plot_columns", [None, 1, 2, 3])
def test_valid_recording_saves_and_queues_without_matching_plots(
    ui_qapp, service, tmp_path, monkeypatch, caplog, plot_columns,
):
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path, streaming=False)
    if plot_columns is not None:
        from ui.sequence.station_plot_grid import StationPlotGrid
        host.plot_grid = StationPlotGrid()
        host.plot_grid.set_grid_size(1, plot_columns)
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    host._schedule_raw_audio_csv_export = mock.Mock()
    host.analysis_config = {
        "auto_analysis": True, "display_sequence": ["SPL"],
        "SPL": {"type": "SPL", "analysis_channels": [0, 2], "limit_checked": False},
    }
    host.recorded_signal_info["active_input_channels"] = [0, 2]
    host._live_mic_channel_v2pa_factors = {0: 1.0, 2: 2.0}
    def capture_calibration_snapshot():
        host._recording_wav_calibration_metadata = {"recorded_channels": [
            {"wav_channel_index": column, "physical_input_channel": channel,
             "v2pa_factor": 1.0, "standard_spl": 94.0, "calibrated": True}
            for column, channel in enumerate((0, 2))
        ]}
    host._capture_recording_wav_calibration_metadata = capture_calibration_snapshot
    host._analysis_task_queue = deque()
    host._analysis_task_records = {}
    host._should_run_silent_analysis_after_recording = lambda: True
    host._get_active_product_condition_key = lambda: "first"
    host._resolve_condition_record = lambda key: {
        "recorded_path": host.recorded_path, "recorded_signal_info": host.recorded_signal_info,
    }
    host._analysis_record_wav_path = lambda record: record["recorded_path"]
    host._build_process_analysis_request = (
        SequenceWidgetAnalysisProcessOpsMixin._build_process_analysis_request.__get__(host)
    )
    host._set_condition_analysis_stage = mock.Mock()
    host._start_next_queued_analysis = mock.Mock()
    host._refresh_analysis_action_state = mock.Mock()
    host._enqueue_automatic_analysis_current_recording = (
        SequenceWidgetAnalysisProcessOpsMixin._enqueue_automatic_analysis_current_recording.__get__(host)
    )

    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)

    assert session.state == "completed"
    assert_saved_main_envelope(host, session, known_audio()[2:9, (0, 2)])
    save.assert_called_once()
    host._schedule_raw_audio_csv_export.assert_called_once_with((0, 2))
    assert len(host._analysis_task_queue) == 1
    queued = host._analysis_task_queue[0]
    assert queued.wav_path == session.request.path
    assert {item.raw_channel for item in queued.instances} == {0, 2}
    assert {item.analysis_type for item in queued.instances} == {"SPL"}
    assert host._recording_process_contexts == {}
    host._handle_invalid_recording.assert_not_called()
    assert not any("Error in streaming completion" in record.getMessage() for record in caplog.records)


def test_recording_start_does_not_consult_plot_contract(ui_qapp, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import CapturingBridge
    from ui.sequence import sequence_widget_recording_process_ops as recording

    layout = {"CH1": "入口", "CH3": "出口"}
    monkeypatch.setattr(recording, "load_channel_layout", lambda: dict(layout))
    host = main_host(SimpleNamespace(), tmp_path, streaming=False)
    host.recording_bridge = CapturingBridge()
    host.recent_test_sessions = ["capture-placeholder"]
    host.judge_play_and_record()

    assert host.recording_bridge.request.channels == (0, 2)
    context = host._recording_process_contexts[host.recording_bridge.request.request_id]
    assert context.recorded_signal_info["analysis_task_config"]["channel_labels"] == layout
    layout["CH1"] = "changed later"
    assert context.recorded_signal_info["analysis_task_config"]["channel_labels"]["CH1"] == "入口"


@pytest.mark.parametrize("fault", ["request_identity", "ve_frames", "ve_metadata"])
def test_no_plots_does_not_admit_invalid_prepared_result(ui_qapp, tmp_path, fault):
    req, descriptor, multi, _ = prepared_ve_result(tmp_path, {"enabled": False})
    columns = tuple((np.arange(len(multi), dtype=np.int64), multi[:, column]) for column in range(2))
    descriptor = replace(descriptor, digest_algorithm=None, sample_digest=None,
        final_envelope=RecordingFinalEnvelope(1, len(multi), req.channels, columns, None,
            RecordingQualityEvidence(len(multi), "skipped", None, None, None, True)))
    audio = prepare_main_recording_result(descriptor, req)
    if fault == "request_identity":
        session_request = replace(req)
    else:
        session_request = req
        if fault == "ve_frames":
            # Simulate corrupted delivery after preparation; admission must recheck VE evidence.
            object.__setattr__(descriptor, "raw_frames", descriptor.raw_frames + 1)
        else:
            metadata = req.calibration_metadata.to_dict()
            metadata["acquisition"]["machine_id"] = "wrong-machine"
            object.__setattr__(req, "calibration_metadata", metadata)
    host = main_host(SimpleNamespace(), tmp_path, streaming=False)
    session = SimpleNamespace(request=session_request, accept_result=mock.Mock(), reject_result=mock.Mock())
    context = RecordingProcessContext(req, "first", session=session)
    host._recording_process_contexts = {req.request_id: context}
    host._active_recording_process_id = req.request_id

    host._on_process_recording_result(session, audio)

    session.accept_result.assert_not_called()
    session.reject_result.assert_called_once()
    assert context.failed
    assert host.data_struct.recording_source is None


@pytest.mark.parametrize("saved_location", [None, "analysis_task_config", "config_snapshot"])
def test_analysis_request_aliases_use_model_with_saved_snapshot_priority(monkeypatch, saved_location):
    from ui.sequence import sequence_widget_analysis_process_ops as analysis

    captured = {}
    monkeypatch.setattr(analysis, "load_channel_layout", lambda: {"CH1": "current"})
    def build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(instances=())
    monkeypatch.setattr(analysis, "build_analysis_task_request", build)
    host = SequenceWidgetAnalysisProcessOpsMixin()
    record = {"recorded_signal_info": {"analysis_storage": {"project_name": "P"}}}
    if saved_location == "analysis_task_config":
        record["recorded_signal_info"][saved_location] = {"channel_labels": {"CH1": "saved"}}
    elif saved_location == "config_snapshot":
        record[saved_location] = {"channel_labels": {"CH1": "saved"}}

    host._build_process_analysis_request("first", "recording.wav", "自动分析", record)

    assert captured["storage_snapshot"]["channel_labels"] == {"CH1": "current" if saved_location is None else "saved"}
    assert captured["storage_snapshot"]["project_name"] == "P"


@pytest.mark.parametrize("segmented", [False, True])
def test_manual_result_aliases_use_model_without_plots_and_keep_segment_snapshot(monkeypatch, segmented):
    from ui.sequence import sequence_widget_analysis_process_ops as analysis

    window = mock.Mock()
    factory = mock.Mock(return_value=window)
    monkeypatch.setattr(analysis, "AnalysisMultichannelResultWindow", factory)
    monkeypatch.setattr(analysis, "load_channel_layout", lambda: {"CH1": "current"})
    host = SequenceWidgetAnalysisProcessOpsMixin()
    host._analysis_active_request = None
    result = SimpleNamespace(instance_results=(SimpleNamespace(config_key="SPL"),),
        segments=(object(),) if segmented else (),
        channel_labels=SimpleNamespace(to_dict=lambda: {"CH1": "segment snapshot"}))

    assert host._show_manual_analysis_result_windows(result) == 1

    assert factory.call_args.kwargs["channel_labels"] == {"CH1": "segment snapshot" if segmented else "current"}


@pytest.fixture
def grid_host(ui_qapp):
    import logging
    from PyQt5.QtWidgets import QWidget
    from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin

    class Host(QWidget, SequenceWidgetStreamingOpsMixin, SequenceWidgetUiOpsMixin):
        def closeEvent(self, event):
            event.accept()

    host = Host()
    host.default_logger = logging.getLogger(__name__)
    host.count_board = None
    host.left_panel = MotorDetectionLeftPanel(None)
    host._active_input_channels = [0]
    host._configured_input_channels = (0,)
    host._recording_input_channels = None
    host._pending_configured_input_channels = None
    host.setLayout(host.create_waveform_layout())
    yield host
    host.close()
    host.deleteLater()


def test_main_layout_exposes_empty_independent_grid(grid_host):
    from PyQt5.QtWidgets import QSplitter
    from ui.sequence.station_plot_grid import StationPlotGrid
    from ui.sequence.station_plot_cell import StationPlotCell
    host = grid_host
    assert isinstance(host.plot_grid, StationPlotGrid)
    assert not hasattr(host, 'channel_workspace')
    cells = host.plot_grid.findChildren(StationPlotCell)
    assert len(cells) == 6
    assert all(not cell.plot_widget.listDataItems() for cell in cells)
    splitters = host.findChildren(QSplitter)
    assert any(s.count() == 2 and s.widget(1) is host.plot_grid for s in splitters)
    assert any(s.count() == 2 and s.widget(1) is host.left_panel.video_monitor_panel for s in splitters)


def test_channel_changes_freeze_and_sync_left_without_touching_grid(grid_host):
    host = grid_host
    host.plot_grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    cell = host.plot_grid._cells[0, 0]
    curve = cell.plot_widget.listDataItems()[0]
    host.left_panel.set_channels = mock.Mock()
    host.mic = {'max_input_channels': 4}
    host.mic_channels = [2, 0]
    host.refresh_channel_windows()
    assert host._active_input_channels == [0, 2]
    host.left_panel.set_channels.assert_called_once_with((0, 2))
    recorded = {}
    assert host._snapshot_recording_input_channels(recorded) == (0, 2)
    host.mic_channels = [1]
    host.refresh_channel_windows()
    assert host._recording_input_channels == (0, 2)
    assert host._active_input_channels == [0, 2]
    host._finalize_recording_channel_selection()
    assert host._active_input_channels == [1]
    host.left_panel.set_channels.assert_called_with((1,))
    assert host.plot_grid._cells[0, 0] is cell
    assert cell.plot_widget.listDataItems() == [curve]
    np.testing.assert_array_equal(curve.yData, [2, 3])


def test_mode_round_reset_keeps_caller_grid_data_and_buttons(grid_host):
    host = grid_host
    host.plot_grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    host.plot_grid.set_mark_buttons_visible(False)
    cell = host.plot_grid._cells[0, 0]
    host._persist_sequence_page_state = mock.Mock()
    host._clear_recent_session_history = mock.Mock()
    host._last_recent_session_mode = 'test'
    host._on_recent_session_mode_changed({'mode': 'mark'})
    host._clear_recent_session_history.assert_called_once()
    assert host.plot_grid._cells[0, 0] is cell
    np.testing.assert_array_equal(cell.plot_widget.listDataItems()[0].yData, [2, 3])
    assert host.plot_grid._marks_visible is False


def test_condition_lookup_uses_business_config_without_plot_workspace():
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    host = SequenceWidgetStreamingOpsMixin()
    host.product_test_condition_configs = [{'key': 'first'}, {'trigger_state': 'second'}]
    host._current_trigger_direction = 'second'
    assert host._waveform_condition_keys() == ['first', 'second']
    assert host._resolve_waveform_direction('') == 'second'


def test_condition_keys_keep_composite_identity_over_serial_trigger():
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    host = SequenceWidgetStreamingOpsMixin()
    host.product_test_condition_configs = [{'key': 'group_1:condition_1', 'trigger_state': '01 04'}]
    assert host._waveform_condition_keys() == ['group_1:condition_1']


def test_product_configuration_refresh_preserves_caller_grid(refresh_host):
    host, project, _, warnings = refresh_host
    grid = host.plot_grid
    cell = grid._cells[0, 0]
    curve = cell.plot_widget.listDataItems()[0]
    grid.set_column_options(0, [('caller option', {'id': 7})])
    grid.set_mark_buttons_visible(False)
    project['test_groups'][0]['test_conditions'][0]['input_voltage'] = '12'
    assert host.product_program_manager.save_project('A.json', project)[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == 'ready'
    assert host.plot_grid is grid and grid._cells[0, 0] is cell
    assert cell.plot_widget.listDataItems() == [curve]
    np.testing.assert_array_equal(curve.yData, [2, 3])
    assert grid._headers[0].combo.currentText() == 'caller option'
    assert not grid._marks_visible
    assert warnings == []


def test_directional_barcode_cycle_reset_preserves_grid(grid_host):
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
    host = grid_host
    host.plot_grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    cell = host.plot_grid._cells[0, 0]
    SequenceWidgetBarcodeOpsMixin._reset_direction_cycle_panel_state(host)
    assert host.plot_grid._cells[0, 0] is cell
    np.testing.assert_array_equal(cell.plot_widget.listDataItems()[0].yData, [2, 3])


def test_actual_round_reset_preserves_independent_grid(round_reset_host):
    from ui.sequence.station_plot_grid import StationPlotGrid
    host = round_reset_host
    host.plot_grid = StationPlotGrid(host)
    host.plot_grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    cell = host.plot_grid._cells[0, 0]
    start_round(host)
    host._confirm_round_reset = mock.Mock(return_value=True)
    host._on_reset_current_round()
    assert not host._round_reset_group_id
    assert host.plot_grid._cells[0, 0] is cell
    np.testing.assert_array_equal(cell.plot_widget.listDataItems()[0].yData, [2, 3])
