"""GUI direct logs preserve admission, preparation and Qt delivery."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from base.raw_audio_csv_tasks import CsvTaskLedger
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_analysis_ops as analysis
from ui.sequence.sequence_widget_raw_csv_ops import CsvRecordingAdmissionScope, SequenceWidgetRawCsvOpsMixin
from unit_test.ui.test_recording_process_integration import main_host, CapturingBridge
from unit_test.base.test_recording_capture_startup_timing import records


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


@pytest.mark.parametrize('enabled', [False, True])
def test_gui_preparation_and_request_logs(ui_qapp, tmp_path, monkeypatch, caplog, enabled):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, enabled)
    host.mic['machine_id'] = 'device-7'
    clock = [10.0]
    monkeypatch.setattr('ui.sequence.sequence_widget_recording_process_ops.perf_counter', lambda: clock[0], raising=False)
    original = host._begin_recent_session_for_current_run
    def delayed():
        clock[0] += 2.2
        original()
    host._begin_recent_session_for_current_run = delayed
    host.start_this_play()
    rows = records(caplog)
    identity = host.recording_bridge.request.request_id
    admission_rows = [r for r in rows if r['stage'] == 'csv_admission']
    assert all(r['machine_id'] == 'device-7' for r in admission_rows)
    assert any(r['event'] == 'begin' and r['request'] == 'unassigned' for r in admission_rows)
    assert any(r['event'] == 'end' and r['request'] == identity for r in admission_rows)
    assert {'gui_reset', 'calibration', 'csv_admission', 'request_build', 'recent_session'} <= {r['stage'] for r in rows}
    recent, = [r for r in rows if r['stage'] == 'recent_session']
    assert recent['request'] == identity and float(recent['seconds']) == pytest.approx(2.2)
    built, = [r for r in rows if r['stage'] == 'request_build']
    assert built['request'] == identity and float(built['seconds']) == 0


def test_csv_rejection_keeps_original_admission_outcome(ui_qapp, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path, True)
    host.mic['machine_id'] = 'device-7'
    host._current_run_recording_token = 'previous-run'
    for i in range(16):
        host.raw_audio_csv_service.reserve(str(i))
    host.start_this_play()
    assert not hasattr(host.recording_bridge, 'request')
    assert host.raw_audio_csv_service.snapshot().reserved == 16
    admission_rows = [r for r in records(caplog) if r['stage'] == 'csv_admission']
    assert [r['event'] for r in admission_rows] == ['begin', 'end']
    assert all(r['request'] == 'unassigned' and r['machine_id'] == 'device-7'
               and 'token' not in r for r in admission_rows)
    assert admission_rows[-1]['allowed'] == 'False'


@pytest.mark.parametrize('attributes', [{}, {'mic': None}, {'mic': {}}])
def test_csv_admission_tolerates_partial_host_identity(caplog, attributes):
    caplog.set_level(logging.INFO)
    host = SimpleNamespace(default_logger=logging.getLogger(__name__),
                           _reserve_raw_audio_csv_recording=lambda: None, **attributes)
    with CsvRecordingAdmissionScope(host) as scope:
        assert not scope.allowed
    rows = records(caplog)
    assert [r['event'] for r in rows] == ['begin', 'end']
    assert all(r['machine_id'] == '' and r['request'] == 'unassigned' for r in rows)


def test_qt_started_delivery_logs_once_at_existing_slot(ui_qapp, tmp_path, monkeypatch, caplog):
    from dataclasses import replace
    from base.recording_service import RecordingCallbacks, RecordingSession
    from ui.recording_service_bridge import RecordingServiceBridge
    from unit_test.base.test_recording_service import request
    caplog.set_level(logging.INFO)
    clock = [10.0]
    monkeypatch.setattr('base.recording_service.perf_counter', lambda: clock[0])
    session = RecordingSession(SimpleNamespace(), request(tmp_path), RecordingCallbacks())
    session._startup_timing = replace(session._startup_timing, accepted_seconds=9.0)
    session._observe_capture_started()
    bridge = RecordingServiceBridge(SimpleNamespace())
    called = []
    bridge._callbacks['one'] = RecordingCallbacks(started=called.append)
    bridge._enqueue('started', session, None)
    assert not [r for r in records(caplog) if r['stage'] == 'qt_started']
    clock[0] += 2.2
    ui_qapp.processEvents()
    bridge._enqueue('started', session, None)
    ui_qapp.processEvents()
    assert called == [session]
    row, = [r for r in records(caplog, 'one') if r['stage'] == 'qt_started']
    assert float(row['seconds']) == pytest.approx(2.2)


@pytest.mark.parametrize('slow_stage', ['gui_reset', 'calibration'])
@pytest.mark.parametrize('is_replay', [False, True])
def test_gui_reset_and_calibration_measure_the_actual_call(ui_qapp, tmp_path, monkeypatch, caplog, slow_stage, is_replay):
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host.mic['machine_id'] = 'device-7'
    host._current_run_recording_token = 'run-1'
    host.last_play_count = 'replay-1'
    clock = [10.0]
    monkeypatch.setattr(analysis, 'perf_counter', lambda: clock[0])
    name = 'reset_work_pram' if slow_stage == 'gui_reset' else '_capture_recording_wav_calibration_metadata'
    original = getattr(host, name)
    def delayed(*args, **kwargs):
        clock[0] += 2.2
        return original(*args, **kwargs)
    setattr(host, name, delayed)
    if is_replay:
        host.judge_play_and_record(is_replay=True)
    else:
        host.start_this_play()
    measured, = [r for r in records(caplog) if r['stage'] == slow_stage]
    assert float(measured['seconds']) == pytest.approx(2.2)
    assert measured['request'] == 'unassigned'
    assert measured['token'] == ('replay-1' if is_replay else 'run-1')
    assert measured['machine_id'] == 'device-7'


def test_real_reset_measures_path_preparation_and_preserves_channels(ui_qapp, tmp_path, monkeypatch, caplog):
    from base import play_and_record
    caplog.set_level(logging.INFO)
    host = startup_host(tmp_path)
    host._current_run_recording_token = 'run-path-1'
    host.data_struct = SimpleNamespace(sample_rate=100, clear_data=lambda: None)
    host.lineedit_type = SimpleNamespace(text=lambda: 'model')
    host.lineedit_s_or_n = SimpleNamespace(text=lambda: 'barcode')
    host._resolve_recording_name_suffix = lambda: ''
    host._get_active_product_condition_key = lambda: ''
    host._snapshot_recording_input_channels = lambda rec: (0, 2)
    host.sequence_config[0]['seq1']['acq']['detail'].update(total_time=1, recording_root_directory=str(tmp_path))
    clock = [10.0]
    monkeypatch.setattr(analysis, 'perf_counter', lambda: clock[0])
    def mac():
        clock[0] += 2.2
        return 'AA:BB'
    monkeypatch.setattr(play_and_record, 'get_mac_address', mac)
    monkeypatch.setattr(play_and_record.FileOps, 'get_recording_store_dir', lambda *a: str(tmp_path / 'audio'))
    monkeypatch.setattr(analysis.LoadUiConfig, 'get_rec_and_play_dict_base_sequence_dict', lambda *a: ({}, {'num_frames': 100}))
    recorded, rate = analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram(host, 'not_labeled')
    assert rate == 100 and recorded['num_frames'] == 102
    assert recorded['startup_trim_samples'] == 2
    assert host._active_input_channels == [0, 2] and (tmp_path / 'audio').is_dir()
    measured, = [r for r in records(caplog) if r['stage'] == 'recorded_info']
    assert float(measured['seconds']) == pytest.approx(2.2)
    assert measured['request'] == 'unassigned'
    assert measured['token'] == host._current_run_recording_token
