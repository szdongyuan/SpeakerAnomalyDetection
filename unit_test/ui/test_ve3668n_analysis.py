"""Task11: frozen voltage provenance, real analysis inputs and source lifetime."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtWidgets import QApplication, QWidget

from base.data_struct.data_deal_struct import DataDealStruct
from base.recording_process_protocol import FrozenConfig
from base.wav_calibration_metadata import WavCalibrationMetadataReadStatus as ReadStatus
from base.wav_calibration_metadata import append_wav_calibration_metadata, inspect_wav_calibration_metadata
from ui.sequence import sequence_widget_analysis_ops as analysis
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui import signal_analysis_window as signals
from ui.sequence.analysis_report_snapshot import build_analysis_report_items
from unit_test.ui.test_ve3668n_recording import (
    host_factory as recording_host_factory, started_audio, finish_ve_capture, save_calibration,
)


def metadata(factors=(None, None), *, rate=44100, calibration_rate=51200):
    return {
        "schema_version": 1, "backend": "vkinging",
        "acquisition": {"model": "VE3668N", "machine_id": "test-device",
                        "input_mode": "IEPE", "unit": "V", "range_min": -10.0,
                        "range_max": 10.0, "sample_rate": rate},
        "recorded_channels": [
            {"wav_channel_index": index, "physical_input_channel": physical,
             "factor_source": "none" if factor is None else "measured",
             "calibrated": factor is not None, "v2pa_factor": factor,
             "calibration": None if factor is None else {
                 "standard_spl": 94.0, "calibrated_at": "2026-08-28T10:00:00+08:00",
                 "sample_rate": calibration_rate, "duration_seconds": 10.0}}
            for index, (physical, factor) in enumerate(zip((7, 1), factors))
        ],
    }


class AnalysisHost(analysis.SequenceWidgetAnalysisOpsMixin, SequenceWidgetStreamingOpsMixin):
    def __init__(self, data, *, imported=False, mode="view"):
        self.data_struct = data
        self.imported = imported
        self.mic = {"index": 99, "name": "current soundcard", "hostapi": 0}
        self.v2pa_factor = 999.0
        self._active_input_channels = [7, 1]
        self.channel_workspace = None
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.analysis_config = {"display_sequence": ["item"], "item": {
            "type": "SPL", "analysis_channel": 0 if imported else 7,
            "limit_checked": True}}
        self.sequence_config = [{"seq1": {"acq": {"mode": "import" if imported else "record",
                                                   "detail": {"sample_rate": 44100}}}}]
        self.count_board = SimpleNamespace(mode=mode, set_test_result_file=mock.Mock(),
                                           set_test_text=mock.Mock())
        self.recorded_path = "unused.wav"
        self.recorded_signal_info = {"labels": "not_labeled"}
        self.product_test_pdf_report_config = {"enabled": True}
        self.default_logger = SimpleNamespace(error=mock.Mock(), warning=mock.Mock())
        self._capture_excel_export_cache = mock.Mock()
        self._maybe_export_excel_results = mock.Mock()
        self._sync_left_panel_analysis_details = mock.Mock()
        self._update_current_recent_session_result = mock.Mock()
        self._update_recent_session = mock.Mock()
        self._finalize_test_run = mock.Mock()
        self._maybe_show_analysis_result_summary = mock.Mock()

    def _is_import_audio_mode(self):
        return self.imported

    def _close_analysis_windows(self):
        for window in self.analysis_window:
            if hasattr(window, "close"):
                window.close()
        self.analysis_window = []
        self._analysis_result_summary_window = None

    def screen(self):
        return SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900))


@pytest.fixture
def host_factory(monkeypatch):
    app = QApplication.instance() or QApplication([])
    previous = DataDealStruct._instance
    DataDealStruct._instance = None
    data = DataDealStruct()
    hosts = []
    monkeypatch.setattr(analysis.QMessageBox, "warning", mock.Mock())
    monkeypatch.setattr(analysis.QMessageBox, "critical", mock.Mock())

    def make(factors=(None, None), *, imported=False, mode="view", rate=44100):
        data.clear_data()
        data.analysis_result_dict.clear()
        data.pd_peak_grid_points_map.clear()
        data.store_wave_data_multi = np.tile([[0.2, 0.4], [-0.2, -0.4]], (4096, 1)).astype("float32")
        data.store_wave_data = data.store_wave_data_multi.mean(axis=1)
        data.sample_rate = rate
        data.audio_lenth = len(data.store_wave_data)
        data.wav_calibration_metadata = FrozenConfig.snapshot(metadata(factors, rate=rate))
        data.wav_calibration_metadata_authoritative = True
        data.wav_calibration_declared_backend = "vkinging"
        data.wav_calibration_read_status = ReadStatus.VALID
        host = AnalysisHost(data, imported=imported, mode=mode)
        hosts.append(host)
        return host

    yield make
    for host in hosts:
        host._close_analysis_windows()
    DataDealStruct._instance = previous
    app.processEvents()


@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize("mode", ["view", "test"])
def test_none_blocks_manual_and_automatic_pressure_and_clears_old_pass(host_factory, monkeypatch, imported, mode):
    host = host_factory(imported=imported, mode=mode)
    host.data_struct.analysis_result_dict["old-PASS"] = (True, 0.0)
    host.analysis_window = [SimpleNamespace(old=True)]
    host._analysis_result_summary_window = object()
    constructor = mock.Mock()
    registry = mock.Mock(side_effect=AssertionError("current registry must not be read"))
    monkeypatch.setattr(analysis, "get_class_mapping", lambda: {"SPL": constructor})
    monkeypatch.setattr(analysis, "load_mic_channel_v2pa_factors", registry)
    raw = host.data_struct.store_wave_data_multi.copy()

    assert host.run(show_windows=False, report_session_id="session") is False

    constructor.assert_not_called()
    registry.assert_not_called()
    assert host.analysis_window == []
    assert host._analysis_result_summary_window is None
    assert host.data_struct.analysis_result_dict == {}
    assert host._can_output_ok_ng()[0] is False
    host._finalize_test_run.assert_not_called()
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    skip = host._analysis_preflight_skips["item"]
    assert "校准" in skip.reason
    if not imported:
        report = host._update_recent_session.call_args.kwargs
        assert report["analysis_report_state"] == "failed"
        assert report["analysis_report_items"][0]["state"] == "skipped"


@pytest.mark.parametrize("item_type", ["SPLF", "FR", "HD", "RB", "PRB", "AI", "Spec", "LP", "PD", "PM", "ED", "FBA", "FFT", "LOUD"])
@pytest.mark.parametrize("imported", [False, True])
def test_none_gates_entire_existing_non_rsc_calibration_batch(host_factory, monkeypatch, item_type, imported):
    host = host_factory(imported=imported)
    host.analysis_config["item"]["type"] = item_type
    constructor = mock.Mock()
    monkeypatch.setattr(analysis, "get_class_mapping", lambda: {item_type: constructor})
    assert host.run(show_windows=False) is False
    constructor.assert_not_called()


@pytest.mark.parametrize("imported", [False, True])
def test_actual_spl_measured_once_only_selected_column_and_report(host_factory, monkeypatch, imported):
    host = host_factory((10.0, None), imported=imported)
    host.analysis_config["item"].update(limit_metric="overall_spl", scalar_upper_value=101.0)
    registry = mock.Mock(side_effect=AssertionError("must not consult changed current registry"))
    monkeypatch.setattr(analysis, "load_mic_channel_v2pa_factors", registry)
    raw = host.data_struct.store_wave_data_multi.copy()
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    assert instance.result["overall_spl"] == pytest.approx(100.0, abs=0.001)
    assert instance.v2pa_factor == 10.0
    assert host.data_struct.analysis_result_dict["item--通道1" if imported else "item--通道8"][0]
    assert host._can_output_ok_ng() == (True, "")
    assert build_analysis_report_items(host.analysis_window, host.analysis_config,
                                       host.data_struct.analysis_result_dict)[0]["status"] == "OK"
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    registry.assert_not_called()
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))


def test_expanded_brother_runs_but_missing_selected_brother_prevents_whole_pass(host_factory):
    host = host_factory((10.0, None), mode="test")
    host.analysis_config["item"].update(analysis_channels=[7, 1], limit_metric="overall_spl", scalar_upper_value=101.0)
    assert host.run(show_windows=False, report_session_id="session") is True
    assert [window._sequence_runtime_key for window in host.analysis_window] == ["item--通道8"]
    assert host.data_struct.analysis_result_dict["item--通道8"][0]
    assert set(host._analysis_preflight_skips) == {"item--通道2"}
    assert not host._can_output_ok_ng()[0]
    host._finalize_test_run.assert_not_called()
    host.count_board.set_test_result_file.assert_not_called()
    report = host._update_recent_session.call_args.kwargs
    assert report["analysis_report_state"] == "failed"
    assert {(item["name"], item["state"]) for item in report["analysis_report_items"]} == {
        ("item--通道8", "completed"), ("item--通道2", "skipped")}


def test_actual_spectrogram_uses_pressure_not_raw_voltage(host_factory):
    host = host_factory((None, 5.0), imported=True)
    host.analysis_config["item"].update(type="Spec", analysis_channel=1, n_fft=256, hop_length=64)
    raw = host.data_struct.store_wave_data_multi.copy()
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    expected = signals.librosa.amplitude_to_db(np.abs(signals.librosa.stft(
        y=raw[:, 1] * 5, n_fft=256, hop_length=64, window="hann")), ref=20e-6)
    np.testing.assert_allclose(instance.img_item.image, expected.T, atol=1e-4)
    assert instance.data_struct.analysis_result_dict is host.data_struct.analysis_result_dict
    assert instance._sequence_analysis_key == "item"
    report = build_analysis_report_items(host.analysis_window, host.analysis_config,
                                         host.data_struct.analysis_result_dict)
    assert report[0]["name"] == "item" and report[0]["state"] == "completed"
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))


@pytest.mark.parametrize("item_type", ["PD", "ED"])
def test_real_joint_consumer_requires_all_columns(host_factory, item_type):
    host = host_factory((10.0, None))
    host.analysis_config["item"].update(type=item_type)
    assert host.run(show_windows=False) is False
    assert host.analysis_window == []


@pytest.mark.parametrize("item_type", ["PD", "ED"])
def test_actual_joint_pressure_is_mean_of_each_scaled_column(host_factory, item_type):
    host = host_factory((10.0, 2.0))
    host.analysis_config["item"].update(type=item_type,
        head={"type": "PD", "config": {}}, tail={"type": "PM", "config": {}})
    raw = host.data_struct.store_wave_data_multi.copy()
    expected = (raw.astype("float64") * [10.0, 2.0]).mean(axis=1)
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    np.testing.assert_allclose(instance.data_struct.store_wave_data, expected)
    assert instance.v2pa_factor == 1.0
    expected_result = signals.peak_detection(expected, 44100, {}, v2pa_factor=1.0)
    actual_series = instance.result["spl_db_series"] if item_type == "PD" else instance._last_spl_series
    np.testing.assert_allclose(actual_series, expected_result["spl_db_series"])
    key = "item--通道8" + ("-PD" if item_type == "ED" else "")
    assert key in host.data_struct.pd_peak_grid_points_map
    assert instance.data_struct.pd_peak_grid_points_map is host.data_struct.pd_peak_grid_points_map
    assert instance.data_struct.analysis_result_dict is host.data_struct.analysis_result_dict
    report = build_analysis_report_items(host.analysis_window, host.analysis_config,
                                         host.data_struct.analysis_result_dict)
    assert report[0]["name"] == "item" and report[0]["state"] == "completed"
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))


def test_source_fields_are_instance_defaults_and_clear_together(host_factory):
    host = host_factory()
    fresh = object.__new__(DataDealStruct)
    fresh.__init__()
    assert fresh.wav_calibration_declared_backend is None
    assert fresh.wav_calibration_read_status is None
    host._ve_calibration_skips = {"old": object()}
    host._ve_analysis_decisions = {"old": object()}
    host._clear_imported_wav_calibration_state()
    assert host.data_struct.wav_calibration_declared_backend is None
    assert host.data_struct.wav_calibration_read_status is None
    assert host._ve_calibration_skips == {} and host._ve_analysis_decisions == {}
    host.data_struct.wav_calibration_declared_backend = "vkinging"
    host.data_struct.wav_calibration_read_status = ReadStatus.INVALID
    host.data_struct.clear_data()
    assert host.data_struct.wav_calibration_declared_backend is None
    assert host.data_struct.wav_calibration_read_status is None


def prepare_file_host(host):
    host._get_active_product_condition_key = lambda: ""
    host._is_manual_product_condition_cycle_active = lambda: False
    host._clear_plot_area = mock.Mock()
    host.plot_waveform_to_workspace = mock.Mock()
    host.data_btn = SimpleNamespace(setEnabled=mock.Mock())
    host.update_player_btn_is_paused = mock.Mock()
    return host


def write_wav(path, factors=(None, None), *, rate=44100):
    raw = np.tile([[0.2, 0.4], [-0.2, -0.4]], (4096, 1)).astype("float32")
    sf.write(path, raw, rate, subtype="FLOAT")
    if factors is not False:
        assert append_wav_calibration_metadata(path, metadata(factors, rate=rate))
    return raw


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("rate", [44100, 96000])
def test_accepted_file_source_and_historical_rate_do_not_follow_current_mic(host_factory, tmp_path, monkeypatch, reload, rate):
    host = prepare_file_host(host_factory((10.0, 2.0), imported=True))
    host.mic = {"backend": "vkinging", "input_config": {"sample_rate": 51200}}
    path = tmp_path / "historical-none.wav"
    raw = write_wav(path, rate=rate)
    host.sequence_config[0]["seq1"]["acq"]["detail"]["sample_rate"] = 32000
    if reload:
        host._load_audio_file_to_data_struct(str(path))
    else:
        monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
        assert host.import_audio_and_analyze() is True
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert host.data_struct.sample_rate == rate
    assert host.data_struct.wav_calibration_metadata["acquisition"]["sample_rate"] == rate
    assert host._active_input_channels == [7, 1]
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    assert host.run(show_windows=False) is False


def test_ve_to_unmarked_legacy_file_clears_source_and_keeps_factor_one(host_factory, tmp_path):
    host = prepare_file_host(host_factory(imported=True))
    host.run(show_windows=False)
    path = tmp_path / "legacy.wav"
    write_wav(path, factors=False, rate=96000)
    host._load_audio_file_to_data_struct(str(path), sample_rate=96000)
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_declared_backend is None
    assert host.data_struct.wav_calibration_read_status is ReadStatus.ABSENT
    assert host._ve_calibration_skips == {}
    host.mic = {"backend": "vkinging"}
    host.analysis_config["item"].update(limit_metric="overall_spl", scalar_upper_value=101)
    assert host.run(show_windows=False) is True
    assert host.analysis_window[0].v2pa_factor == 1.0
    assert host.data_struct.sample_rate == 96000


def test_presentation_snapshot_restores_explicit_source_and_gate_state(host_factory):
    host = prepare_file_host(host_factory())
    host.run(show_windows=False)
    snapshot = host._snapshot_import_presentation_state()
    assert json.loads(json.dumps(snapshot["wav_calibration_read_status"])) == "valid"
    host._clear_imported_wav_calibration_state()
    host._restore_import_presentation_state(snapshot)
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host._ve_calibration_skips and not host._can_output_ok_ng()[0]


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("measured", [False, True])
def test_recording_success_publishes_frozen_source_before_real_gate(recording_host_factory, live, measured):
    recorder = recording_host_factory(44100, live)
    if measured:
        save_calibration(recorder)
    recorder.analysis_config = {"auto_analysis": True, "display_sequence": ["item"], "item": {
        "type": "SPL", "analysis_channel": 7, "limit_checked": True,
        "limit_metric": "overall_spl", "scalar_upper_value": 200}}
    session, capture, audio = started_audio(recorder)
    original = session.request.calibration_metadata.to_dict()
    save_calibration(recorder, factor=99.0)
    recorder.ve_profile_store.set_sample_rate(recorder.mic, 48000, recorder.ve_calibration_store)
    recorder.mic = {"backend": "soundcard", "index": 8}
    recorder.data_struct.analysis_result_dict = {"old": (True, 0.0)}
    recorder.data_struct.pd_peak_grid_points_map = {}
    recorder.screen = lambda: SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900))
    recorder._close_analysis_windows = SequenceWidgetStreamingOpsMixin._close_analysis_windows.__get__(recorder)
    recorder._capture_excel_export_cache = mock.Mock()
    recorder._maybe_export_excel_results = mock.Mock()
    recorder._sync_left_panel_analysis_details = mock.Mock()
    outcomes = []
    def run(**kwargs):
        result = analysis.SequenceWidgetAnalysisOpsMixin.run(recorder, **kwargs)
        outcomes.append(result)
        return result
    recorder.run = run
    try:
        finish_ve_capture(recorder, session, audio)
        assert recorder.data_struct.wav_calibration_declared_backend == "vkinging"
        assert recorder.data_struct.wav_calibration_read_status is ReadStatus.VALID
        assert recorder.data_struct.wav_calibration_metadata.to_dict() == original
        assert recorder.data_struct.sample_rate == 44100
        assert outcomes == []  # Request-frozen SPL executes on the scoped analyzer.
        recorder.saved.assert_called_once()
        recorder._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))
        np.testing.assert_array_equal(recorder.data_struct.store_wave_data_multi, audio.multi)
        if measured:
            assert not recorder.analysis_window
            assert original["recorded_channels"][0]["calibration"]["sample_rate"] == 51200
        else:
            assert not recorder.analysis_window and not recorder.data_struct.analysis_result_dict
    finally:
        recorder._close_analysis_windows()
        assert capture.wait(1) is not None


def test_changed_live_channel_selection_does_not_reinterpret_recorded_columns(host_factory):
    host = host_factory((10.0, 2.0))
    host._active_input_channels = [1, 7]  # current hardware selection is not audio provenance
    host.analysis_config["item"].update(limit_metric="overall_spl", scalar_upper_value=101)
    assert host.run(show_windows=False)
    assert host.analysis_window[0].analysis_config["analysis_channel"] == 0
    assert host.analysis_window[0].result["overall_spl"] == pytest.approx(100.0, abs=.001)


def test_calibration_skip_cannot_make_a_summary_pass(host_factory):
    host = host_factory((10.0, None))
    host.analysis_config["item"].update(analysis_channels=[7, 1], limit_metric="overall_spl", scalar_upper_value=101)
    assert host.run(show_windows=False)
    assert host._summarize_ok_ng()[0] is False
    analysis.SequenceWidgetAnalysisOpsMixin._maybe_show_analysis_result_summary(host, 100, 100)
    assert host._analysis_result_summary_window is None


def test_recent_view_restores_previous_source_and_calibration_skips(host_factory, tmp_path):
    host = prepare_file_host(host_factory(imported=True))
    host.run(show_windows=False)
    previous = host.data_struct.wav_calibration_metadata
    path = tmp_path / "measured.wav"
    write_wav(path, factors=(10.0, 2.0))
    host._resolve_recent_session = lambda key: {"sample_rate": 96000}
    host._resolve_recent_session_path = lambda record: str(path)
    host._apply_recent_session_config_for_view = lambda record: (True, "")
    host.analysis_config["item"].update(limit_metric="overall_spl", scalar_upper_value=101)
    host._excel_export_cache = {}
    host._excel_exported_record_id = None
    outcomes = []
    def run_history(**kwargs):
        assert host.data_struct.sample_rate == 44100
        outcomes.append(analysis.SequenceWidgetAnalysisOpsMixin.run(host, show_windows=False))
    host.run = run_history
    host._show_recent_session_analysis_by_id("history")
    assert outcomes == [True]
    assert host.data_struct.wav_calibration_metadata == previous
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert host._ve_calibration_skips and not host._can_output_ok_ng()[0]


@pytest.mark.parametrize("action", ["failed", "cancelled", "cancel"])
def test_abort_without_published_audio_clears_source_but_not_product_rate(recording_host_factory, action):
    host = recording_host_factory()
    host.judge_play_and_record()
    session = host._recording_process_session
    host.data_struct.wav_calibration_metadata = metadata()
    host.data_struct.wav_calibration_declared_backend = "vkinging"
    host.data_struct.wav_calibration_read_status = ReadStatus.VALID
    host._recording_product_sample_rate = 32000
    if action == "failed":
        host._on_process_recording_failed(session, SimpleNamespace(stage="read", message="fake failure"))
    elif action == "cancelled":
        host._on_process_recording_cancelled(session, None)
    else:
        host._cancel_process_recording()
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_declared_backend is None
    assert host.data_struct.wav_calibration_read_status is None
    assert host._recording_product_sample_rate == 32000


@pytest.mark.parametrize("path", ["mark", "product", "legacy_config", "barcode", "reset"])
def test_existing_lifecycle_fallbacks_clear_complete_source(host_factory, monkeypatch, path):
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    from ui.sequence import sequence_widget_config_ops as config_ops
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
    data = host_factory().data_struct
    no_op = mock.Mock()
    button = SimpleNamespace(setDisabled=no_op, setEnabled=no_op)
    host = SimpleNamespace(data_struct=data, player_status_flag=False,
        _clear_plot_area=no_op, _close_analysis_windows=no_op,
        update_player_btn_is_paused=no_op, replayer_btn=button, data_btn=button,
        lineedit_s_or_n=SimpleNamespace(isEnabled=lambda: False), setFocus=no_op,
        using_file_combobox=SimpleNamespace(currentData=lambda: "product" if path == "product" else None,
                                           clearFocus=no_op),
        registry={}, get_sequence_config_from_json=no_op, init_data_struct_stimulus_config=no_op,
        reset_test_reord=no_op, default_logger=SimpleNamespace(error=no_op),
        _is_recording_in_progress_for_labeling=lambda: False, sequence_config=[],
        recorded_signal_info={}, update_audio_label_info=no_op, _maybe_export_excel_results=no_op,
        update_recorded_signal_info_to_db=lambda: (0, ""), mark_result=no_op,
        _reset_barcode_commit_dedup=no_op,
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
        signal_info=SimpleNamespace(clear=no_op))
    manager = SimpleNamespace(load_project=lambda _: (0, {}), validate_project=lambda *args: {"is_usable": True},
                              load_registry=lambda: {}, save_registry=lambda _: True)
    host._get_product_program_manager = lambda: manager
    monkeypatch.setattr(config_ops.LoadUiConfig, "update_using_config_path", no_op)
    if path == "mark":
        SequenceWidgetUiOpsMixin.on_mark_btn_clicked(host)
    elif path in ("product", "legacy_config"):
        config_ops.SequenceWidgetConfigOpsMixin.on_using_file_combobox_changed(host, "test")
    elif path == "barcode":
        SequenceWidgetBarcodeOpsMixin.clicked_ok_or_ng(host)
    else:
        SequenceWidgetStreamingOpsMixin.on_reset_statistics_clicked(host)
    assert data.wav_calibration_metadata is None
    assert data.wav_calibration_declared_backend is None
    assert data.wav_calibration_read_status is None


@pytest.mark.parametrize("state", [ReadStatus.INVALID, ReadStatus.ABSENT, None])
@pytest.mark.parametrize("imported", [False, True])
def test_identified_invalid_source_never_uses_registry_or_factor_one(host_factory, monkeypatch, state, imported):
    host = host_factory(imported=imported)
    host.data_struct.wav_calibration_metadata = None
    host.data_struct.wav_calibration_read_status = state
    forbidden = mock.Mock(side_effect=AssertionError("legacy fallback"))
    monkeypatch.setattr(analysis, "load_mic_channel_v2pa_factors", forbidden)
    assert host.run(show_windows=False) is False
    forbidden.assert_not_called()
    assert host._ve_calibration_skips
    assert "无效" in host._analysis_preflight_skips["item"].reason


def test_live_batch_preparation_cannot_read_registry_for_identified_invalid_ve(host_factory, monkeypatch):
    host = host_factory()
    host.data_struct.wav_calibration_metadata = None
    host.data_struct.wav_calibration_read_status = ReadStatus.INVALID
    registry = mock.Mock(return_value={7: 33.0})
    monkeypatch.setattr(analysis, "load_mic_channel_v2pa_factors", registry)
    host._prepare_live_mic_calibration_batch()
    registry.assert_not_called()
    with pytest.raises(ValueError, match="VE"):
        host._resolve_live_mic_channel_v2pa_factor(7)


@pytest.mark.parametrize("invalid_first", [False, True])
def test_invalid_ve_with_legacy_comment_is_rejected_without_replacing_previous_audio(host_factory, tmp_path, monkeypatch, invalid_first):
    from unit_test.base.test_wav_calibration_metadata import _append_raw_chunk, _chunk
    host = prepare_file_host(host_factory((10.0, 2.0), imported=True))
    old_source = host.data_struct.wav_calibration_metadata
    old_audio = host.data_struct.store_wave_data_multi.copy()
    path = tmp_path / "invalid.wav"
    write_wav(path, factors=False)
    legacy = {"recorded_channels": [{"wav_channel_index": col, "calibrated": True,
               "v2pa_factor": 7.0, "standard_spl": 94.0} for col in (0, 1)]}
    invalid = metadata()
    invalid["schema_version"] = 99
    comments = [invalid, legacy] if invalid_first else [legacy, invalid]
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + b"".join(
        _chunk(b"ICMT", b"mic_calibration=" + json.dumps(item).encode() + b"\0") for item in comments)))
    diagnostic = inspect_wav_calibration_metadata(path)
    assert diagnostic.status is ReadStatus.INVALID and diagnostic.declared_backend == "vkinging"
    monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    assert host.import_audio_and_analyze() is False
    assert host.data_struct.wav_calibration_metadata is old_source
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, old_audio)
    assert "导入音频失败" in analysis.QMessageBox.warning.call_args.args[-1]


@pytest.mark.parametrize("columns,allowed,factors", [((0,), True, (10.0,)), ((1,), False, ()), ((0, 1), False, ())])
def test_pure_policy_uses_actual_participants_only(columns, allowed, factors):
    from base.wav_calibration_metadata import WavCalibrationMetadataReadResult
    from ui.sequence.ve3668n_analysis_policy import resolve_ve_analysis
    source = WavCalibrationMetadataReadResult(ReadStatus.VALID, metadata((10.0, None)), "vkinging")
    decision = resolve_ve_analysis(source, columns, column_count=2)
    assert decision.allowed is allowed
    assert decision.columns == columns and decision.factors == factors
    assert bool(decision.diagnostic) is (not allowed)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(schema_version=2),
    lambda value: value["recorded_channels"][0].update(calibrated=False),
    lambda value: value["recorded_channels"][1].update(v2pa_factor=1.0),
])
def test_pure_policy_does_not_partially_trust_invalid_snapshot(mutate):
    from base.wav_calibration_metadata import WavCalibrationMetadataReadResult
    from ui.sequence.ve3668n_analysis_policy import resolve_ve_analysis, verified_recorded_channels
    value = metadata((10.0, None))
    mutate(value)
    source = WavCalibrationMetadataReadResult(ReadStatus.VALID, value, "vkinging")
    assert not resolve_ve_analysis(source, (0,), column_count=2).allowed
    assert verified_recorded_channels(source) is None


def test_file_replacement_and_snapshot_restore_own_voltage_hints(host_factory, tmp_path):
    host = prepare_file_host(host_factory(imported=True))
    windows = [QWidget(), QWidget()]
    for window in windows:
        window.setToolTip("original-help")
    host.channel_workspace = SimpleNamespace(all_subwindows=lambda: windows, clear_plots=lambda: None)
    # Use the actual file projection boundary with a minimal plot sink.
    host._plot_file_audio_to_workspace = lambda *args: None
    ve_path = tmp_path / "voltage.wav"
    write_wav(ve_path, factors=(10.0, None))
    host._load_audio_file_to_data_struct(str(ve_path))
    assert "实测校准有效" in windows[0].toolTip()
    assert "未校准" in windows[1].toolTip()
    snapshot = host._snapshot_import_presentation_state()
    legacy_path = tmp_path / "legacy.wav"
    write_wav(legacy_path, factors=False)
    host._load_audio_file_to_data_struct(str(legacy_path))
    assert [window.toolTip() for window in windows] == ["original-help"] * 2
    host._restore_import_presentation_state(snapshot)
    assert "实测校准有效" in windows[0].toolTip()
    assert "未校准" in windows[1].toolTip()
    for window in windows:
        window.close()


@pytest.mark.parametrize("item_type", ["PD", "ED"])
def test_actual_joint_fallback_plots_do_not_reapply_calibration(host_factory, monkeypatch, item_type):
    host = host_factory((10.0, 2.0))
    host.analysis_config["item"].update(type=item_type,
        head={"type": "PD", "config": {}}, tail={"type": "PM", "config": {}})
    expected = (host.data_struct.store_wave_data_multi.astype("float64") * [10.0, 2.0]).mean(axis=1)
    core = signals.peak_detection
    calls = []
    def no_cached_spl(signal, rate, config, *, v2pa_factor):
        # Exercise the widget's documented missing-series module-boundary
        # recovery, while still executing the real PD calculation.
        calls.append((signal.copy(), v2pa_factor))
        result = core(signal, rate, config, v2pa_factor=v2pa_factor)
        return {**result, "spl_db_series": []}
    monkeypatch.setattr(signals, "peak_detection", no_cached_spl)
    assert host.run(show_windows=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], expected)
    assert calls[0][1] == 1.0
    instance = host.analysis_window[0]
    expected_spl = signals.AudioThdFrequencyResponseAnalysis.spl_calculation(expected, v2pa_factor=1.0)
    if item_type == "PD":
        actual_spl = instance.analysis_plot.listDataItems()[0].getData()[1]
    else:
        actual_spl = instance._last_spl_series
    np.testing.assert_allclose(actual_spl, expected_spl)


def test_actual_ed_nested_pattern_matching_receives_pressure_segments(host_factory, tmp_path):
    host = host_factory((10.0, 2.0))
    pulse = .001 + .2 * np.exp(-((np.arange(8192) - 4096) / 200) ** 2)
    raw = np.column_stack([pulse, pulse * 2]).astype("float32")
    host.data_struct.store_wave_data_multi = raw
    host.data_struct.store_wave_data = raw.mean(axis=1)
    expected = (raw.astype("float64") * [10.0, 2.0]).mean(axis=1)
    head = {"spl_window_unit": "points", "spl_window_points": 31,
            "nms_enabled": True, "nms_unit": "points", "nms_points": 8192, "peak_count": 1}
    peaks = signals.peak_detection(expected, 44100, head, v2pa_factor=1.0)["peaks_index"]
    assert len(peaks) == 1
    peak = int(peaks[0])
    path = tmp_path / "pressure-pattern.wav"
    sf.write(path, expected[peak - 16:peak + 16], 44100, subtype="FLOAT")
    host.analysis_config["item"].update(type="ED", head={"type": "PD", "config": head},
        tail={"type": "PM", "config": {"feature_type": "waveform", "pattern_save_path": str(path)}},
        left_grid=16, right_grid=16)
    assert host.run(show_windows=False)
    instance = host.analysis_window[0]
    assert instance.table_widget.rowCount() == 1
    assert float(instance.table_widget.item(0, 3).text().strip("%")) == pytest.approx(100, abs=.01)
    assert instance.data_struct.pd_peak_grid_points_map["item--通道8-PD"] == [peak]
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)


@pytest.mark.parametrize("item_type", ["SPLF", "FR", "HD", "RB", "PRB", "AI", "LP", "FBA", "FFT", "LOUD"])
def test_imported_other_consumers_get_file_column_k_without_prescaling_model_input(host_factory, monkeypatch, item_type):
    host = host_factory((None, 5.0), imported=True)
    host.analysis_config["item"].update(type=item_type, analysis_channel=1)
    # The admission/instance boundary is inspected for all existing consumers;
    # the actual Pa algorithms are separately exercised above, not inferred.
    class Consumer:
        def __init__(self, title):
            self.title = title
    monkeypatch.setattr(analysis, "get_class_mapping", lambda: {item_type: Consumer})
    assert host.run(show_windows=False)
    instance = host.analysis_window[0]
    assert instance.v2pa_factor == 5.0
    assert instance.analysis_config["analysis_channel"] == 1
    assert instance.data_struct is host.data_struct


def test_imported_product_report_never_marks_calibration_skip_completed(host_factory):
    host = host_factory((10.0, None), imported=True)
    host.analysis_config = {"display_sequence": ["good", "missing"],
        "good": {"type": "SPL", "analysis_channel": 0, "limit_checked": True,
                 "limit_metric": "overall_spl", "scalar_upper_value": 101},
        "missing": {"type": "SPL", "analysis_channel": 1}}
    host._get_active_product_condition_key = lambda: "01"
    host._manual_product_condition_group_id = "group"
    host._manual_product_condition_results = {"01": "OK"}  # previous attempt
    assert host.run(show_windows=False)
    host._capture_imported_product_condition_record()
    record = host._condition_record_cache["01"]
    assert record["analysis_report_state"] == "failed"
    assert record["result_label"] != "OK"
    assert {(item["name"], item["state"]) for item in record["analysis_report_items"]} == {
        ("good", "completed"), ("missing", "skipped")}


def test_none_import_without_auto_analysis_drops_previous_pressure_results(host_factory, tmp_path, monkeypatch):
    host = prepare_file_host(host_factory((10.0, 2.0), imported=True))
    host.analysis_config["item"].update(limit_metric="overall_spl", scalar_upper_value=101)
    assert host.run(show_windows=False)
    assert host.data_struct.analysis_result_dict and host.analysis_window
    host._excel_export_cache = {"old": "PASS"}
    path = tmp_path / "none.wav"
    write_wav(path)
    monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    assert host.import_audio_and_analyze()
    assert not host.data_struct.analysis_result_dict
    assert not host.analysis_window
    assert not host._excel_export_cache


def test_all_blocked_run_drops_previous_export_cache(host_factory):
    host = host_factory()
    host._excel_export_cache = {"old": "PASS"}
    assert host.run(show_windows=False) is False
    assert not host._excel_export_cache


def test_new_recording_discards_previous_calibration_gate_state(recording_host_factory):
    host = recording_host_factory()
    host._ve_calibration_skips = {"old": object()}
    host._ve_analysis_decisions = {"old": object()}
    host._analysis_preflight_skips = {"old": object()}
    host.judge_play_and_record()
    assert host._ve_calibration_skips == {}
    assert host._ve_analysis_decisions == {}
    assert host._analysis_preflight_skips == {}


def test_policy_keeps_positive_historical_rates_and_original_measurement():
    from base.wav_calibration_metadata import WavCalibrationMetadataReadResult
    from ui.sequence.ve3668n_analysis_policy import resolve_ve_analysis
    value = metadata((10.0, 2.0), rate=96000, calibration_rate=51200)
    before = deepcopy(value)
    source = WavCalibrationMetadataReadResult(ReadStatus.VALID, value, "vkinging")
    decision = resolve_ve_analysis(source, (0, 1), column_count=2)
    assert decision.allowed and decision.factors == (10.0, 2.0)
    assert value == before


@pytest.mark.parametrize("channels", [[7, 0], [0]])
@pytest.mark.parametrize("mode", ["view", "test"])
def test_absent_required_ve_channel_prevents_completion(host_factory, channels, mode):
    host = host_factory((10.0, 2.0), mode=mode)
    host.analysis_config["item"].update(
        analysis_channels=channels, limit_metric="overall_spl", scalar_upper_value=101.0)
    host.data_struct.analysis_result_dict["old-PASS"] = (True, 0.0)
    host._excel_export_cache = {"old": "PASS"}
    host._excel_exported_record_id = "old.wav"
    raw = host.data_struct.store_wave_data_multi.copy()

    assert host.run(show_windows=False, report_session_id="session") is (7 in channels)

    skip_key = "item--通道1" if len(channels) > 1 else "item"
    assert set(host._analysis_preflight_skips) == {skip_key}
    assert host._ve_calibration_skips == {}  # Absent channel is not missing calibration.
    can_output, reason = host._can_output_ok_ng()
    assert can_output is False
    assert "通道" in reason and "不存在" in reason and "校准" not in reason
    assert host._summarize_ok_ng() == (False, "NG")
    host._finalize_test_run.assert_not_called()
    host.count_board.set_test_result_file.assert_not_called()
    analysis.SequenceWidgetAnalysisOpsMixin._maybe_show_analysis_result_summary(host, 100, 100)
    assert host._analysis_result_summary_window is None
    assert not host._excel_export_cache and host._excel_exported_record_id is None
    assert "old-PASS" not in host.data_struct.analysis_result_dict
    report = host._update_recent_session.call_args.kwargs
    assert report["analysis_report_state"] == "failed"
    skipped = [item for item in report["analysis_report_items"] if item["state"] == "skipped"]
    assert [item["name"] for item in skipped] == [skip_key]
    assert "不存在" in skipped[0]["reason"] and "校准" not in skipped[0]["reason"]
    if 7 in channels:
        assert [window._sequence_runtime_key for window in host.analysis_window] == ["item--通道8"]
        assert host.analysis_window[0].result["overall_spl"] == pytest.approx(100.0, abs=.001)
        assert host.data_struct.analysis_result_dict["item--通道8"][0] is True
        assert ("item--通道8", "completed") in {
            (item["name"], item["state"]) for item in report["analysis_report_items"]}
    else:
        assert not host.analysis_window and not host.data_struct.analysis_result_dict
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)


def prepare_condition_import_host(host, *, partial, missing_channel=False, previous=False):
    host = prepare_file_host(host)
    host.analysis_config = {"display_sequence": ["good", "missing"] if partial else ["missing"],
        "good": {"type": "SPL", "analysis_channel": 0, "limit_checked": True,
                 "limit_metric": "overall_spl", "scalar_upper_value": 101.0},
        "missing": {"type": "SPL", "analysis_channel": 2 if missing_channel else 1,
                    "limit_checked": True, "limit_metric": "overall_spl", "scalar_upper_value": 101.0}}
    host._active_product_condition_key = "01"
    host._get_active_product_condition_key = lambda: host._active_product_condition_key
    host._is_manual_product_condition_cycle_active = lambda: True
    host._manual_product_condition_group_id = "group-import"
    host._manual_product_condition_results = {"00": "OK", **({"01": "OK"} if previous else {})}
    host._manual_product_condition_completed_keys = {"00", "01"} if previous else {"00"}
    host._manual_product_condition_keys = lambda: ["00", "01"]
    host._condition_record_cache = {"00": {"recorded_path": "other.wav"},
        "01": {"recorded_path": "previous.wav", "result_label": "OK",
               "analysis_report_state": "completed", "analysis_result_dict": {"old-PASS": (True, 0)}}}
    host._refresh_current_manual_product_final_from_group = mock.Mock()
    host._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    host.left_panel = SimpleNamespace(set_condition_result=mock.Mock(), set_current_stage=mock.Mock(),
                                     set_final_result=mock.Mock())
    host.replayer_btn = SimpleNamespace(setDisabled=mock.Mock())
    host._excel_export_cache = {"old": "PASS"}
    host._excel_exported_record_id = "previous.wav"
    real_run = host.run

    def hidden_real_run(*args, **kwargs):
        kwargs["show_windows"] = False  # Host is not QObject; retain real source/calculation/cycle logic.
        return real_run(*args, **kwargs)

    host.run = hidden_real_run
    return host


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("missing_channel", [False, True])
@pytest.mark.parametrize("previous", [False, True])
def test_incomplete_ve_import_preserves_audio_without_completing_condition(
        host_factory, tmp_path, monkeypatch, partial, missing_channel, previous):
    host = prepare_condition_import_host(host_factory(imported=True, mode="test"),
        partial=partial, missing_channel=missing_channel, previous=previous)
    path = tmp_path / "incomplete.wav"
    raw = write_wav(path, (10.0, 2.0 if missing_channel else None))
    monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))

    imported = host.import_audio_and_analyze()

    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert host.recorded_path == str(path)
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))
    host.plot_waveform_to_workspace.assert_called_once()
    host._clear_plot_area.assert_called_once()  # Only before replacement, never after accepted preview.
    assert bool(host.analysis_window) is partial
    if partial:
        assert host.analysis_window[0].result["overall_spl"] == pytest.approx(100.0, abs=.001)
        assert host.data_struct.analysis_result_dict["good--通道1"][0] is True
    assert host._manual_product_condition_completed_keys == {"00"}
    assert host._manual_product_condition_results == {"00": "OK"}
    host._advance_manual_product_condition_cycle_after_recording.assert_not_called()
    host._refresh_current_manual_product_final_from_group.assert_not_called()
    assert host._active_product_condition_key == "01"
    assert host._condition_record_cache["00"] == {"recorded_path": "other.wav"}
    record = host._condition_record_cache["01"]
    assert record["recorded_path"] == str(path)
    assert record["analysis_report_state"] == "failed"
    assert record["result_label"] == record["recorded_signal_info"]["labels"] == "not_labeled"
    assert "old-PASS" not in record["analysis_result_dict"]
    skipped = [item for item in record["analysis_report_items"] if item["state"] == "skipped"]
    assert [item["name"] for item in skipped] == ["missing"]
    assert ("不存在" if missing_channel else "校准") in skipped[0]["reason"]
    host.left_panel.set_condition_result.assert_called()
    assert host.left_panel.set_condition_result.call_args.kwargs["tone"] == "pending"
    host.left_panel.set_final_result.assert_called_with("待判定", tone="pending")
    assert "未执行" in host.left_panel.set_current_stage.call_args.args[0]
    assert not host._excel_export_cache and host._excel_exported_record_id is None
    assert not host._can_output_ok_ng()[0]
    assert imported is False  # File was accepted, but the requested analysis/condition is incomplete.


@pytest.mark.parametrize("channels", [[7, 0], [0]])
@pytest.mark.parametrize("live", [False, True])
def test_absent_pressure_channel_does_not_complete_recording_condition(recording_host_factory, channels, live):
    recorder = recording_host_factory(44100, live)
    save_calibration(recorder)
    save_calibration(recorder, physical=1, factor=2.0)
    recorder.analysis_config = {"auto_analysis": True, "display_sequence": ["item"], "item": {
        "type": "SPL", "analysis_channels": channels, "limit_checked": True,
        "limit_metric": "overall_spl", "scalar_upper_value": 200}}
    recorder._should_run_silent_analysis_after_recording = (
        SequenceWidgetStreamingOpsMixin._should_run_silent_analysis_after_recording.__get__(recorder))
    recorder._is_manual_product_condition_cycle_active = lambda: True
    recorder._get_active_product_condition_key = lambda: "item-condition"
    recorder._manual_product_condition_keys = lambda: ["item-condition"]
    session, capture, audio = started_audio(recorder)
    recorder.data_struct.analysis_result_dict = {}
    recorder.data_struct.pd_peak_grid_points_map = {}
    recorder.screen = lambda: SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900))
    recorder._close_analysis_windows = SequenceWidgetStreamingOpsMixin._close_analysis_windows.__get__(recorder)
    recorder._capture_excel_export_cache = mock.Mock()
    recorder._maybe_export_excel_results = mock.Mock()
    recorder._sync_left_panel_analysis_details = mock.Mock()
    recorder._mark_manual_product_condition_recording_completed = mock.Mock()
    recorder._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    recorder._finalize_serial_product_condition_analysis_failure = mock.Mock(return_value=False)
    outcomes = []

    def real_run(**kwargs):
        result = analysis.SequenceWidgetAnalysisOpsMixin.run(recorder, **kwargs)
        outcomes.append(result)
        return result

    recorder.run = real_run
    try:
        finish_ve_capture(recorder, session, audio)
        assert outcomes == []  # Request-frozen SPL executes on the scoped analyzer.
        assert recorder._ve_calibration_skips == {}
        recorder._mark_manual_product_condition_recording_completed.assert_not_called()
        recorder._advance_manual_product_condition_cycle_after_recording.assert_not_called()
        reason = recorder._finalize_serial_product_condition_analysis_failure.call_args.args[0]
        assert "不存在" in reason and "校准" not in reason
        recorder.saved.assert_called_once()
        recorder._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))
        np.testing.assert_array_equal(recorder.data_struct.store_wave_data_multi, audio.multi)
    finally:
        recorder._close_analysis_windows()
        assert capture.wait(1) is not None


def test_complete_ve_import_retry_can_finish_condition(host_factory, tmp_path, monkeypatch):
    host = prepare_condition_import_host(host_factory(imported=True), partial=True, previous=True)
    path = tmp_path / "retry.wav"
    write_wav(path, (10.0, None))
    monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    assert host.import_audio_and_analyze() is False
    write_wav(path, (10.0, 2.0))
    assert host.import_audio_and_analyze() is True
    assert host._manual_product_condition_completed_keys == {"00", "01"}
    host._advance_manual_product_condition_cycle_after_recording.assert_called_once()
    host.left_panel.set_condition_result.assert_called_with("01", "完成", tone="ok")
    assert host._condition_record_cache["01"]["analysis_report_state"] == "completed"
    assert not host._ve_calibration_skips and not host._ve_channel_skips


@pytest.mark.parametrize("restore", ["snapshot", "recent"])
def test_channel_completion_guard_restores_with_audio_context(host_factory, tmp_path, restore):
    host = prepare_file_host(host_factory((10.0, 2.0), imported=True))
    host.analysis_config["item"].update(analysis_channel=2, limit_metric="overall_spl", scalar_upper_value=101)
    assert host.run(show_windows=False) is False
    path = tmp_path / "legacy.wav"
    write_wav(path, factors=False)
    if restore == "snapshot":
        snapshot = host._snapshot_import_presentation_state()
        host._load_audio_file_to_data_struct(str(path))
        assert not host._ve_channel_skips
        host._restore_import_presentation_state(snapshot)
    else:
        host._resolve_recent_session = lambda key: {"sample_rate": 44100}
        host._resolve_recent_session_path = lambda record: str(path)
        host._apply_recent_session_config_for_view = lambda record: (True, "")
        host._excel_export_cache = {}
        host._excel_exported_record_id = None
        outcomes = []

        def run_history(**kwargs):
            host.analysis_config["item"]["analysis_channel"] = 0
            outcomes.append(analysis.SequenceWidgetAnalysisOpsMixin.run(host, show_windows=False))
            assert not host._ve_channel_skips

        host.run = run_history
        host._show_recent_session_analysis_by_id("history")
        assert outcomes == [True]
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host._ve_channel_skips
    assert host._can_output_ok_ng()[0] is False
    assert "不存在" in host._can_output_ok_ng()[1]


@pytest.mark.parametrize("partial", [False, True])
def test_unmarked_legacy_preflight_completion_semantics_are_unchanged(host_factory, partial):
    host = host_factory(imported=True)
    host.data_struct.clear_wav_calibration_context()
    host.analysis_config = {"display_sequence": ["good", "missing"] if partial else ["missing"],
        "good": {"type": "SPL", "analysis_channel": 0, "limit_checked": True,
                 "limit_metric": "overall_spl", "scalar_upper_value": 101},
        "missing": {"type": "SPL", "analysis_channel": 2, "limit_checked": True}}
    assert host.run(show_windows=False, report_session_id="legacy") is partial
    assert host._ve_channel_skips == {}
    assert host._can_output_ok_ng()[0] is partial
    host._capture_current_analysis_report_snapshot("legacy")
    assert host._update_recent_session.call_args.kwargs["analysis_report_state"] == "completed"


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("live", [False, True])
def test_empty_analysis_selection_keeps_ve_raw_recording_success(recording_host_factory, measured, live):
    recorder = recording_host_factory(44100, live)
    if measured:
        save_calibration(recorder)
        save_calibration(recorder, physical=1, factor=2.0)
    session, capture, audio = started_audio(recorder)
    recorder.data_struct.analysis_result_dict = {"old-PASS": (True, 0.0)}
    recorder.data_struct.pd_peak_grid_points_map = {}
    recorder.analysis_config = {"auto_analysis": True, "display_sequence": []}
    recorder.screen = lambda: SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900))
    recorder._close_analysis_windows = SequenceWidgetStreamingOpsMixin._close_analysis_windows.__get__(recorder)
    recorder._should_run_silent_analysis_after_recording = (
        SequenceWidgetStreamingOpsMixin._should_run_silent_analysis_after_recording.__get__(recorder))
    recorder._capture_excel_export_cache = mock.Mock()
    recorder._maybe_export_excel_results = mock.Mock()
    recorder._sync_left_panel_analysis_details = mock.Mock()
    recorder._finalize_test_run = mock.Mock()
    recorder._finalize_serial_product_condition_analysis_failure = mock.Mock(return_value=False)
    outcomes = []

    def real_run(**kwargs):
        assert recorder._should_run_silent_analysis_after_recording() is True
        result = analysis.SequenceWidgetAnalysisOpsMixin.run(recorder, **kwargs)
        outcomes.append(result)
        return result

    recorder.run = real_run
    try:
        finish_ve_capture(recorder, session, audio)
        recorder.saved.assert_called_once()
        np.testing.assert_array_equal(recorder.data_struct.store_wave_data_multi, audio.multi)
        recorder._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))
        # Detached publication uses the pure request-owned analyzer and must
        # not call the mutable active-window ``run`` method.  The surrounding
        # saved/TCP assertions prove the successful no-op completion.
        assert outcomes == []
        assert not recorder._ve_calibration_skips and not recorder._ve_channel_skips
        assert not recorder.analysis_window and not recorder.data_struct.analysis_result_dict
        assert not recorder._can_output_ok_ng()[0]
        assert recorder._summarize_ok_ng() == (False, "NG")
        recorder._finalize_test_run.assert_not_called()
        recorder._finalize_serial_product_condition_analysis_failure.assert_not_called()
    finally:
        recorder._close_analysis_windows()
        assert capture.wait(1) is not None


@pytest.mark.parametrize("ve_source", [False, True])
def test_empty_analysis_remains_noop_without_pass_or_failed_report(host_factory, ve_source):
    host = host_factory(mode="test")
    if not ve_source:
        host.data_struct.clear_wav_calibration_context()
    host.analysis_config = {"auto_analysis": True, "display_sequence": []}
    host.data_struct.analysis_result_dict = {"old-PASS": (True, 0.0)}
    assert host.run(show_windows=False, report_session_id="empty") is True
    assert not host.analysis_window and not host.data_struct.analysis_result_dict
    assert not host._can_output_ok_ng()[0]
    assert host._summarize_ok_ng() == (False, "NG")
    host._finalize_test_run.assert_not_called()
    report = host._update_recent_session.call_args.kwargs
    assert report["analysis_report_state"] == "not_required"
    assert report["analysis_report_items"] == []


def prepare_eight_channel_pressure_host(host):
    frames, rate = 1024, 51200
    physical_order = (7, 1, 6, 0, 5, 2, 4, 3)
    factors = (10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    times = np.arange(frames, dtype=np.float64) / rate
    raw = np.column_stack([
        (.1 + column / 100) * np.sin(2 * np.pi * (1000 + 125 * column) * times)
        for column in range(8)
    ]).astype("float32")
    value = metadata((10.0, 2.0), rate=rate)
    prototype = value["recorded_channels"][0]
    value["recorded_channels"] = [
        {**deepcopy(prototype), "wav_channel_index": column,
         "physical_input_channel": physical, "v2pa_factor": factors[column]}
        for column, physical in enumerate(physical_order)
    ]
    host.data_struct.store_wave_data_multi = raw
    host.data_struct.store_wave_data = raw.mean(axis=1)
    host.data_struct.audio_lenth = frames
    host.data_struct.wav_calibration_metadata = FrozenConfig.snapshot(value)
    host._active_input_channels = list(physical_order)
    return raw.copy(), physical_order, factors


def test_real_expanded_spec_retains_only_selected_pressure_vectors(host_factory):
    host = host_factory((10.0, 2.0), rate=51200)
    raw, physical_order, factors = prepare_eight_channel_pressure_host(host)
    source_metadata = host.data_struct.wav_calibration_metadata
    host.analysis_config = {"display_sequence": ["spec"], "spec": {
        "type": "Spec", "analysis_channels": list(physical_order), "n_fft": 256, "hop_length": 64}}
    config_before = deepcopy(host.analysis_config)
    assert host.run(show_windows=False) is True
    assert len(host.analysis_window) == 8
    for instance in host.analysis_window:
        physical = instance._analysis_raw_channel
        column = physical_order.index(physical)
        key = f"spec--通道{physical + 1}"
        assert instance._sequence_analysis_key == "spec"
        assert instance._sequence_runtime_key == instance._sequence_window_key == key
        assert instance._sequence_multi_channel_expansion is True
        assert host._analysis_channel_local_columns[key] == column
        assert host._ve_analysis_decisions[key].columns == (column,)
        assert host._ve_analysis_decisions[key].factors == (factors[column],)
        expected = raw[:, column].astype("float64") * factors[column]
        np.testing.assert_array_equal(instance.data_struct.store_wave_data, expected)
        expected_image = signals.librosa.amplitude_to_db(np.abs(signals.librosa.stft(
            y=expected.astype("float32"), n_fft=256, hop_length=64, window="hann")), ref=20e-6)
        np.testing.assert_allclose(instance.img_item.image, expected_image.T, atol=1e-4)
        assert instance.v2pa_factor == 1.0
        assert instance.data_struct.sample_rate == 51200
        assert instance.data_struct.audio_lenth == len(raw)
        assert instance.data_struct.analysis_result_dict is host.data_struct.analysis_result_dict
        assert instance.data_struct.pd_peak_grid_points_map is host.data_struct.pd_peak_grid_points_map
    report = build_analysis_report_items(host.analysis_window, host.analysis_config,
                                        host.data_struct.analysis_result_dict)
    assert {(item["name"], item["state"]) for item in report} == {
        (f"spec--通道{physical + 1}", "completed") for physical in physical_order}
    assert host.analysis_config == config_before
    assert host.data_struct.wav_calibration_metadata is source_metadata
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))
    matrix_bytes = sum(getattr(window.data_struct.store_wave_data_multi, "nbytes", 0)
                       for window in host.analysis_window)
    mono_bytes = sum(window.data_struct.store_wave_data.nbytes for window in host.analysis_window)
    assert (matrix_bytes, mono_bytes) == (0, 8 * len(raw) * np.dtype("float64").itemsize)
    for instance in host.analysis_window:
        assert instance.analysis_config["analysis_channel"] == 0
        assert instance.data_struct.store_wave_data.flags.owndata


@pytest.mark.parametrize("imported", [False, True])
def test_real_nonzero_spec_retains_one_vector_and_source_identity(host_factory, imported):
    host = host_factory((None, 5.0), imported=imported)
    host.analysis_config["item"].update(type="Spec", analysis_channel=1, n_fft=256, hop_length=64)
    config_before = deepcopy(host.analysis_config)
    raw = host.data_struct.store_wave_data_multi.copy()
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    expected = raw[:, 1].astype("float64") * 5.0
    np.testing.assert_array_equal(instance.data_struct.store_wave_data, expected)
    expected_image = signals.librosa.amplitude_to_db(np.abs(signals.librosa.stft(
        y=expected.astype("float32"), n_fft=256, hop_length=64, window="hann")), ref=20e-6)
    np.testing.assert_allclose(instance.img_item.image, expected_image.T, atol=1e-4)
    assert host._ve_analysis_decisions["item"].columns == (1,)
    assert host._analysis_channel_local_columns["item"] == 1
    assert instance._analysis_raw_channel == 1
    assert instance._sequence_analysis_key == instance._sequence_window_key == "item"
    assert instance._sequence_runtime_key == "item--通道2"
    assert instance._sequence_multi_channel_expansion is False
    assert host.analysis_config == config_before
    assert not host._ve_calibration_skips
    report = build_analysis_report_items(host.analysis_window, host.analysis_config,
                                        host.data_struct.analysis_result_dict)
    assert [(item["name"], item["state"]) for item in report] == [("item", "completed")]
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))
    assert instance.data_struct.store_wave_data_multi is None
    assert instance.data_struct.store_wave_data.nbytes == len(raw) * np.dtype("float64").itemsize
    assert instance.data_struct.store_wave_data.flags.owndata
    assert instance.analysis_config["analysis_channel"] == 0


@pytest.mark.parametrize("item_type", ["PD", "ED"])
def test_real_joint_consumer_retains_only_mean_pressure_vector(host_factory, item_type):
    host = host_factory((10.0, 2.0), rate=51200)
    raw, physical_order, factors = prepare_eight_channel_pressure_host(host)
    pd_config = {"spl_window_unit": "points", "spl_window_points": 31}
    host.analysis_config["item"].update(type=item_type, analysis_channel=1, **pd_config,
        head={"type": "PD", "config": pd_config}, tail={"type": "PM", "config": {}})
    config_before = deepcopy(host.analysis_config)
    source_metadata = host.data_struct.wav_calibration_metadata
    expected = (raw.astype("float64") * factors).mean(axis=1)
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    np.testing.assert_allclose(instance.data_struct.store_wave_data, expected, rtol=1e-14, atol=1e-14)
    expected_result = signals.peak_detection(expected, 51200, pd_config, v2pa_factor=1.0)
    actual_series = instance.result["spl_db_series"] if item_type == "PD" else instance._last_spl_series
    np.testing.assert_allclose(actual_series, expected_result["spl_db_series"])
    assert instance.v2pa_factor == 1.0
    assert instance._analysis_raw_channel == 1
    assert instance._sequence_analysis_key == instance._sequence_window_key == "item"
    assert instance._sequence_runtime_key == "item--通道2"
    assert instance._sequence_multi_channel_expansion is False
    assert host._analysis_channel_local_columns["item"] == physical_order.index(1)
    assert host._ve_analysis_decisions["item"].columns == tuple(range(8))
    assert host._ve_analysis_decisions["item"].factors == factors
    assert instance.data_struct.sample_rate == 51200 and instance.data_struct.audio_lenth == len(raw)
    assert instance.data_struct.analysis_result_dict is host.data_struct.analysis_result_dict
    assert instance.data_struct.pd_peak_grid_points_map is host.data_struct.pd_peak_grid_points_map
    assert "item--通道2" + ("-PD" if item_type == "ED" else "") in host.data_struct.pd_peak_grid_points_map
    report = build_analysis_report_items(host.analysis_window, host.analysis_config,
                                        host.data_struct.analysis_result_dict)
    assert [(item["name"], item["state"]) for item in report] == [("item", "completed")]
    assert host.analysis_config == config_before
    assert host.data_struct.wav_calibration_metadata is source_metadata
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, raw.mean(axis=1))
    matrix_bytes = getattr(instance.data_struct.store_wave_data_multi, "nbytes", 0)
    mono_bytes = instance.data_struct.store_wave_data.nbytes
    assert (matrix_bytes, mono_bytes) == (0, len(raw) * np.dtype("float64").itemsize)
    assert instance.data_struct.store_wave_data.flags.owndata
    assert instance.analysis_config["analysis_channel"] == 0


def test_legacy_nonzero_spec_keeps_shared_raw_input_and_mapped_index(host_factory):
    host = host_factory(imported=True)
    host.data_struct.clear_wav_calibration_context()
    host.analysis_config["item"].update(type="Spec", analysis_channel=1, n_fft=256, hop_length=64)
    raw = host.data_struct.store_wave_data_multi.copy()
    assert host.run(show_windows=False) is True
    instance = host.analysis_window[0]
    assert instance.data_struct is host.data_struct
    assert instance.analysis_config["analysis_channel"] == 1
    expected = signals.librosa.amplitude_to_db(np.abs(signals.librosa.stft(
        y=raw[:, 1], n_fft=256, hop_length=64, window="hann")), ref=20e-6)
    np.testing.assert_allclose(instance.img_item.image, expected.T, atol=1e-4)
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, raw)
