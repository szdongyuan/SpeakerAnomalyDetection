"""Main capture publishes bounded envelopes through real spawn and Qt delivery."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from unit_test.ui.test_recording_process_integration import main_host, pump, service
from unit_test.base.recording_process_fakes import known_audio


@pytest.mark.parametrize("preview", [False, True])
def test_main_never_constructs_reader_and_publishes_saved_source(
        ui_qapp, service, tmp_path, monkeypatch, preview):
    from ui.sequence import sequence_widget_streaming_ops as stream
    forbidden = mock.Mock(side_effect=AssertionError("main must not construct a reader"))
    service._reader_factory = forbidden
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    host = main_host(service, tmp_path, streaming=preview)
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    assert session.state == "completed"
    forbidden.assert_not_called()
    assert session.reader is None and session._reader_released
    assert not hasattr(session.audio, "multi") and not hasattr(session.audio, "mono")
    assert host.data_struct.store_wave_data is None
    assert host.data_struct.store_wave_data_multi is None
    assert host.data_struct.audio_lenth == 7
    assert host.data_struct.recording_source.request is session.request
    assert host.data_struct.recording_source.path == session.request.path
    expected = known_audio()[2:9, (0, 2)]
    for column, (times, values) in enumerate(session.audio.waveforms):
        np.testing.assert_array_equal(times, np.arange(7) / 100)
        np.testing.assert_array_equal(values, expected[:, column])
    save.assert_called_once()
    host._handle_invalid_recording.assert_not_called()


@pytest.mark.parametrize("failure", ["second_channel", "second_declines", "envelope"])
def test_display_failure_hides_curves_but_keeps_new_wav_business(
        ui_qapp, service, tmp_path, monkeypatch, failure):
    from dataclasses import replace
    from ui.sequence import sequence_widget_streaming_ops as stream
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    warning = mock.Mock()
    monkeypatch.setattr(stream.QMessageBox, "warning", warning)
    host = main_host(service, tmp_path, streaming=False)
    other_source = object()
    host.data_struct.recording_sources = {"other": other_source}
    host.data_struct.recording_display_sources = {"other": other_source}
    workspace = AnalysisWaveformPanel(channel_layout_path=str(tmp_path / "layout.json"))
    workspace.set_channels([0, 2])
    host.channel_workspace = workspace
    host._schedule_raw_audio_csv_export = mock.Mock()
    host._enqueue_automatic_analysis_current_recording = mock.Mock()
    windows = workspace.all_subwindows()
    for window in windows:
        window.set_data(np.arange(3), np.ones(3))
    if failure == "second_channel":
        monkeypatch.setattr(windows[1], "set_data", mock.Mock(side_effect=RuntimeError("plot failed")))
    elif failure == "second_declines":
        monkeypatch.setattr(windows[1], "set_data", mock.Mock(return_value=False))
    else:
        # Acceptance must return the exact admitted object, so inject at the service factory.
        import base.recording_final_result as final
        factory = final.prepare_main_recording_result
        def failed_display(descriptor, request):
            descriptor = replace(descriptor, final_envelope=replace(
                descriptor.final_envelope, columns=(), display_error="reducer failed"))
            return factory(descriptor, request)
        monkeypatch.setattr(final, "prepare_main_recording_result", failed_display)
    try:
        host.judge_play_and_record()
        session = host._recording_process_session
        pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
        assert session.state == "completed"
        assert host.data_struct.recording_source.request is session.request
        assert host.data_struct.audio_lenth == 7
        assert all(window.plot_item is None for window in windows)
        assert all(not window.plot_widget.isEnabled() for window in windows)
        assert host.data_struct.recording_display_sources["other"] is other_source
        assert host.data_struct.recording_sources["other"] is other_source
        assert all(source is None for key, source in host.data_struct.recording_display_sources.items()
                   if key != "other")
        save.assert_called_once()
        host._schedule_raw_audio_csv_export.assert_called_once_with((0, 2))
        host._enqueue_automatic_analysis_current_recording.assert_called_once()
        warning.assert_called_once()
    finally:
        workspace.close()


def test_cancel_cleanup_preserves_published_file_provenance():
    from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
    source = object()
    clear = mock.Mock()
    host = SimpleNamespace(data_struct=SimpleNamespace(
        store_wave_data=None, store_wave_data_multi=None, recording_source=source),
        _clear_audio_source_analysis_state=clear)
    SequenceWidgetRecordingProcessOpsMixin._clear_unpublished_recording_source(host)
    clear.assert_not_called()


def test_data_reset_clears_saved_and_display_provenance():
    from base.data_struct.data_deal_struct import DataDealStruct
    data = SimpleNamespace(recording_source=object(), recording_sources={"old": object()},
                           recording_display_sources={"old": object()},
                           clear_wav_calibration_context=mock.Mock())
    DataDealStruct.clear_data(data)
    assert data.recording_source is None
    assert data.recording_sources == data.recording_display_sources == {}


@pytest.mark.parametrize("replacement", ["clear", "relative", "cumulative", "final", "condition", "context_reset"])
def test_failed_plot_recovers_on_reset_or_successful_replacement(ui_qapp, tmp_path, replacement):
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    conditions = [{"key": "a", "name": "A"}, {"key": "b", "name": "B"}]
    panel = AnalysisWaveformPanel(condition_configs=conditions,
                                  channel_layout_path=str(tmp_path / "layout.json"))
    panel.set_channels([0])
    window = panel.all_subwindows()[0]
    try:
        window.show_recording_display_failure()
        ui_qapp.processEvents()
        assert not window.plot_widget.isEnabled()
        assert window.plot_widget.getPlotItem().titleLabel.text == "波形刷新失败"
        if replacement == "clear":
            window.clear_plot()
        elif replacement == "condition":
            assert panel.set_active_condition("b")
            assert panel.set_active_condition("a")
        elif replacement == "context_reset":
            panel.set_conditions(conditions, reset_context=True)
        else:
            setter = {"relative": window.set_relative_preview_data,
                      "cumulative": window.set_cumulative_preview_data,
                      "final": window.set_data}[replacement]
            assert setter(np.arange(3), np.ones(3))
        assert window.plot_widget.isEnabled()
        assert window.plot_widget.getPlotItem().titleLabel.text == ""
        # A reused/reset row must also accept either preview presentation.
        for setter in (window.set_relative_preview_data, window.set_cumulative_preview_data):
            assert setter(np.arange(3), np.ones(3))
            assert window.plot_widget.isEnabled()
    finally:
        panel.close()


@pytest.mark.parametrize("with_analysis_helper", [False, True])
def test_explicit_mode_reset_clears_all_recording_identities(with_analysis_helper):
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
    data = SimpleNamespace(recording_source=object(), recording_sources={"old": object()},
                           recording_display_sources={"old": object()})
    host = SimpleNamespace(data_struct=data, _clear_plot_area=mock.Mock(),
                           _close_analysis_windows=mock.Mock(), update_player_btn_is_paused=mock.Mock(),
                           replayer_btn=mock.Mock(), data_btn=mock.Mock())
    if with_analysis_helper:
        host._clear_audio_source_analysis_state = (
            SequenceWidgetAnalysisOpsMixin._clear_audio_source_analysis_state.__get__(host))
    SequenceWidgetUiOpsMixin.on_mark_btn_clicked(host)
    assert data.recording_source is None
    assert data.recording_sources == data.recording_display_sources == {}


def test_new_run_reset_then_cancel_preserves_published_sources(ui_qapp, tmp_path, monkeypatch):
    from base.data_struct.data_deal_struct import DataDealStruct
    from base import play_and_record
    from ui.sequence import sequence_widget_analysis_ops as analysis
    from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
    from unit_test.ui.test_recording_startup_timing import startup_host
    host = startup_host(tmp_path)
    data = object.__new__(DataDealStruct)
    data.__init__()
    host.data_struct = data
    data.sample_rate = 100
    source, other = object(), object()
    data.recording_source = source
    data.recording_sources = {"a": source, "b": other}
    data.recording_display_sources = {"a": source, "b": other}
    metadata = {"old": True}
    data.wav_calibration_metadata = metadata
    data.wav_calibration_metadata_authoritative = True
    host._analysis_preflight_warning_shown = True
    host._analysis_preflight_skips = {"old": object()}
    host._analysis_channel_local_columns = {"old": 1}
    host._clear_audio_source_analysis_state = analysis.SequenceWidgetAnalysisOpsMixin._clear_audio_source_analysis_state.__get__(host)
    host.lineedit_type = SimpleNamespace(text=lambda: "model")
    host.lineedit_s_or_n = SimpleNamespace(text=lambda: "barcode")
    host._resolve_recording_name_suffix = lambda: ""
    host._get_active_product_condition_key = lambda: "a"
    host._snapshot_recording_input_channels = lambda rec: (0, 2)
    host.sequence_config[0]["seq1"]["acq"]["detail"].update(total_time=1, recording_root_directory=str(tmp_path))
    monkeypatch.setattr(play_and_record, "get_mac_address", lambda: "AA:BB")
    monkeypatch.setattr(play_and_record.FileOps, "get_recording_store_dir", lambda *a: str(tmp_path / "audio"))
    monkeypatch.setattr(analysis.LoadUiConfig, "get_rec_and_play_dict_base_sequence_dict", lambda *a: ({}, {"num_frames": 100}))
    analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram(host, "not_labeled")
    SequenceWidgetRecordingProcessOpsMixin._clear_unpublished_recording_source(host)
    assert data.recording_source is source
    assert data.recording_sources == data.recording_display_sources == {"a": source, "b": other}
    assert data.wav_calibration_metadata is metadata
    assert data.wav_calibration_metadata_authoritative
    assert not host._analysis_preflight_warning_shown
    assert host._analysis_preflight_skips == host._analysis_channel_local_columns == {}


@pytest.mark.parametrize("clear_all", [False, True])
def test_plot_clear_invalidates_only_corresponding_display_identity(clear_all):
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    source, other = object(), object()
    data = SimpleNamespace(recording_source=source, recording_sources={"a": source, "b": other},
                           recording_display_sources={"a": source, "b": other})
    host = SimpleNamespace(data_struct=data, channel_workspace=SimpleNamespace(
        _active_condition_key="a", clear_plots=mock.Mock()))
    method = (SequenceWidgetStreamingOpsMixin.clear_all_direction_waveforms if clear_all
              else SequenceWidgetStreamingOpsMixin._clear_plot_area)
    method(host)
    assert data.recording_source is source
    assert data.recording_sources == {"a": source, "b": other}
    assert data.recording_display_sources == ({} if clear_all else {"b": other})


def test_five_channel_long_payload_qt_publication_timing(ui_qapp, tmp_path, monkeypatch):
    """Measure only Qt publication of a bounded 600 s payload, not full UI I/O."""
    import json
    from pathlib import Path
    from time import perf_counter, process_time
    from base.recording_final_result import prepare_main_recording_result
    from base.recording_process_protocol import (
        RecordingFinalEnvelope, RecordingQualityEvidence, RecordingResult)
    from consts.recording_result_consts import RECORDING_FINAL_ENVELOPE_VERSION
    from unit_test.performance.benchmark_recording_final_envelope import request_for
    from ui.sequence import sequence_widget_streaming_ops as stream
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = main_host(SimpleNamespace(busy=False), tmp_path, streaming=False)
    request = request_for(tmp_path / 'display-only.wav', 44100 * 600, 5, 44100)
    # A valid fixed-size payload isolates GUI cost; this test does not synthesize a WAV.
    indices = np.linspace(0, request.target_samples - 1, 48000, dtype=np.int64)
    columns = tuple((indices.copy(), np.sin(np.arange(48000) / 31 + i).astype(np.float32))
                    for i in range(5))
    quality = RecordingQualityEvidence(request.target_samples, 'passed', .9, .3, .09, None)
    envelope = RecordingFinalEnvelope(RECORDING_FINAL_ENVELOPE_VERSION,
        request.target_samples, request.channels, columns, None, quality)
    descriptor = RecordingResult(request.request_id, 'main', request.path, 44100,
        request.channels, request.target_samples, request.target_samples, False,
        final_envelope=envelope)
    audio = prepare_main_recording_result(descriptor, request)
    panel = AnalysisWaveformPanel(channel_layout_path=str(tmp_path / 'timed-layout.json'))
    panel.set_channels(list(request.channels))
    host.channel_workspace = panel
    host._recording_process_direction = 'timed'
    host._schedule_raw_audio_csv_export = mock.Mock()
    host._enqueue_automatic_analysis_current_recording = mock.Mock()
    context = RecordingProcessContext(request=request, direction='timed', preview_enabled=False,
                                      final_windows=panel.all_subwindows())
    monkeypatch.setattr(stream, 'RecordingManager', lambda: SimpleNamespace(
        save_signal_info_to_db=lambda *args: (0, 'saved')))
    try:
        began, cpu = perf_counter(), process_time()
        assert host._on_recording_envelope_complete(audio, context)
        elapsed = dict(wall_seconds=perf_counter() - began, cpu_seconds=process_time() - cpu,
                       channels=5, points_per_channel=48000, represented_seconds=600,
                       scope='Qt set_data + file-business mocks; excludes painting, actual database, CSV and analysis')
        assert host.data_struct.audio_lenth == request.target_samples
        assert host.data_struct.store_wave_data is host.data_struct.store_wave_data_multi is None
        for window, (times, values) in zip(context.final_windows, audio.waveforms):
            x, y = window.plot_item.getData()
            np.testing.assert_array_equal(x, times)
            np.testing.assert_array_equal(y, values)
        Path('.pytest_cache/recording-final-envelope-gui-profile.json').write_text(
            json.dumps(elapsed, indent=2), encoding='utf-8')
    finally:
        panel.close()


def test_relabel_updates_saved_source_and_current_file_consumers(ui_qapp, service, tmp_path, monkeypatch):
    from pathlib import Path
    from ui.sequence import sequence_widget_streaming_ops as stream
    host = main_host(service, tmp_path, streaming=False)
    manager = SimpleNamespace(save_signal_info_to_db=mock.Mock(return_value=(0, 'saved')),
                              update_audio_label=mock.Mock(return_value=(0, 'updated')))
    monkeypatch.setattr(stream, 'RecordingManager', lambda: manager)
    host._normalize_db_audio_path = lambda path: str(path)
    host._resolve_active_recording_waveform_direction = lambda fallback='': 'condition'
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    source = host.data_struct.recording_source
    original = Path(source.path).read_bytes()
    other = object()
    host.data_struct.recording_sources['other'] = other
    host.count_board.mode = 'mark'
    host.recorded_signal_info['labels'] = 'not_labeled'
    host._condition_record_cache = {'condition': dict(recorded_path=source.path,
        recorded_signal_info=dict(host.recorded_signal_info))}
    host._resolve_condition_record = lambda key: host._condition_record_cache[key]
    host._resolve_labelable_condition_record_path = lambda record: record['recorded_path']
    host._condition_key_matches_record = lambda *args: True
    host.on_waveform_condition_mark_clicked('condition', 'OK')
    renamed = host.data_struct.recording_source
    assert renamed.request is source.request
    assert renamed.path != source.path
    assert Path(renamed.path).read_bytes() == original
    assert not Path(source.path).exists()
    assert host.data_struct.recording_sources == {'condition': renamed, 'other': other}
    assert host.recorded_path == host._condition_record_cache['condition']['recorded_path'] == renamed.path
    assert host.data_struct.store_wave_data is host.data_struct.store_wave_data_multi is None
    # A later manual analysis resolves the updated path; in-flight CSV blocks relabeling.
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin
    assert SequenceWidgetAnalysisProcessOpsMixin._analysis_record_wav_path(
        host, host._condition_record_cache['condition']) == renamed.path
