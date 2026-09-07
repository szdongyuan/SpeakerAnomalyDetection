"""Task12: production Qt/capture/reader/analysis paths, injected SDK and stores.

Reuse the Task9--11 admission fakes; do not emulate calibration, metadata,
completion decisions or pressure conversion here. All files live in pytest TEMP.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit

from base import hardware_selection
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingPreview, RecordingResult
from base.recording_result_reader import RecordingAudio, ResultReader
from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
from base.ve3668n_capture import Ve3668nInputStream
from base.ve3668n_stores import VECalibrationStore, VEInputProfileStore
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus as ReadStatus,
    inspect_wav_calibration_metadata,
)
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from ui import calibration_window as calibration
from ui import hardware_window as hardware_ui
from ui import signal_analysis_window as signals
from ui.sequence import sequence_widget_analysis_ops as analysis
from ui.sequence.analysis_report_snapshot import build_analysis_report_items
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from unit_test.base.recording_process_fakes import device_info as soundcard
from unit_test.base.test_ve3668n_calibration import sine_capture_dependencies, sine_volts
from unit_test.base.ve3668n_fakes import CaptureSDK, device_info
from unit_test.ui.test_ve3668n_analysis import (
    host_factory as analysis_host_factory,
    prepare_condition_import_host,
    prepare_file_host,
)
from unit_test.ui.test_ve3668n_calibration import (
    setup as calibration_setup,
    accept,
    offered,
    release,
    seed_old,
)
from unit_test.ui.test_ve3668n_hardware import ControlledDiscovery, controls
from unit_test.ui.test_ve3668n_recording import (
    host_factory as recording_host_factory,
    finish_ve_capture,
    save_calibration,
)


def select_hardware(controls, app, rate, *, profiles=None, calibrations=None):
    """Accept an actual dialog, restoring the persisted selection when present."""
    profiles = profiles if profiles is not None else controls.profiles
    calibrations = calibrations if calibrations is not None else controls.calibrations
    if controls.path.exists():
        mic, _, channels, _ = hardware_selection.restore_or_default(
            path=controls.path, apply_defaults=False)
        # Explicitly exercise input-only acceptance again; legacy output
        # restoration is independent and may offer an ordinary soundcard.
        state = hardware_ui.HardwareSelectionState(
            mic_device=mic, mic_channels=channels, speaker_device=None)
    else:
        state = hardware_ui.HardwareSelectionState(mic_device=device_info(), mic_channels=[7, 1])
    controller = hardware_ui.HardwareSelectionController(
        hardware_ui.HardwareSelectionModel(state), hardware_ui.HardwareSelectionView(),
        profile_store=profiles, calibration_store=calibrations,
        discovery_factory=ControlledDiscovery, selection_path=controls.path)
    try:
        panel = controller.view.ve_controls
        panel.discovery.service.deliver([device_info()])
        app.processEvents()
        combo = panel.sample_rate_combo
        assert not combo.isEditable()
        assert [combo.itemData(i) for i in range(combo.count())] == [44100, 48000, 51200]
        if not profiles.path.exists():
            assert combo.currentData() == 51200
        for child in panel.findChildren(QLabel) + panel.findChildren(QLineEdit):
            assert "灵敏度" not in child.text() and "sensitivity" not in child.objectName().lower()
        combo.setCurrentIndex(combo.findData(rate))
        controller._on_ok_clicked()
        assert controller.view.result() == QDialog.Accepted
        selected = deepcopy(controller.model.state)
        assert selected.speaker_device is None and selected.mic_channels == [7, 1]
        assert selected.mic_device["input_config"] == {
            "sample_rate": rate, "input_mode": "IEPE", "unit": "V",
            "range_min": -10.0, "range_max": 10.0}
        assert not controls.defaults.pair.writes and not controls.defaults.calls
        return selected
    finally:
        controller.view.reject()
        app.processEvents()


def enable_real_analysis(host, config):
    """Only surrounding report/UI sinks are fakes; run and consumers are real."""
    host.data_struct.analysis_result_dict = {"old-PASS": (True, 0.0)}
    host.data_struct.pd_peak_grid_points_map = {}
    host.analysis_config = config
    host.screen = lambda: SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900))
    host._close_analysis_windows = SequenceWidgetStreamingOpsMixin._close_analysis_windows.__get__(host)
    host._should_run_silent_analysis_after_recording = (
        SequenceWidgetStreamingOpsMixin._should_run_silent_analysis_after_recording.__get__(host))
    for name in ("_capture_excel_export_cache", "_maybe_export_excel_results",
                 "_sync_left_panel_analysis_details", "_finalize_test_run", "_update_recent_session"):
        setattr(host, name, mock.Mock())
    host._finalize_serial_product_condition_analysis_failure = mock.Mock(return_value=False)
    host.product_test_pdf_report_config = {"enabled": True}
    host.run = analysis.SequenceWidgetAnalysisOpsMixin.run.__get__(host)


def read_capture(request, **dependencies):
    """Use the real writer AND reader, including the actual FLOAT subtype check."""
    sdk_factory = dependencies.pop("ve_sdk_factory", None)
    if sdk_factory is not None:
        dependencies["ve_stream_factory"] = lambda **kwargs: Ve3668nInputStream(
            **kwargs, sdk_factory=sdk_factory)
    if not dependencies:
        dependencies = {"ve_stream_factory": lambda **kwargs: Ve3668nInputStream(
            **kwargs, sdk_factory=lambda: CaptureSDK(counts=(31, 7, 93)))}
    capture = RecordingCapture(request, **dependencies)
    capture.start()
    descriptor = capture.wait(10)
    assert isinstance(descriptor, RecordingResult), descriptor
    outcomes = []
    reader = ResultReader(descriptor, outcomes.append)
    reader.start()
    reader.thread.join(5)
    assert not reader.thread.is_alive()
    assert len(outcomes) == 1 and outcomes[0].handles_released and outcomes[0].error is None
    assert sf.info(request.path).subtype == "FLOAT"
    return capture, outcomes[0].audio


def start_capture(host):
    host.judge_play_and_record(tcp_completion_address=("127.0.0.1", 1234))
    session = host._recording_process_session
    assert session is not None
    capture, audio = read_capture(session.request)
    return session, capture, audio


@pytest.fixture
def workflow(controls, calibration_setup, recording_host_factory, monkeypatch, tmp_path):
    paths = []

    def next_path(*args, **kwargs):
        path = tmp_path / f"workflow-{len(paths)}.wav"
        paths.append(path)
        return str(path), {"name": "workflow", "labels": "not_labeled"}

    monkeypatch.setattr(analysis, "get_recorded_info", next_path)
    widgets, hosts = [], []

    def recorder(selected, live=False):
        host = recording_host_factory(selected.mic_device["input_config"]["sample_rate"], live)
        host.mic, host.mic_channels = deepcopy(selected.mic_device), list(selected.mic_channels)
        host.ve_profile_store, host.ve_calibration_store = controls.profiles, controls.calibrations
        host.refresh_channel_windows()
        hosts.append(host)
        return host

    def calibrator(selected):
        widget = calibration.InputCalibration(
            selected.mic_device, selected.mic_channels, recording_bridge=calibration_setup.bridge,
            ve_profile_store=controls.profiles, ve_calibration_store=controls.calibrations)
        widgets.append(widget)
        return widget

    yield SimpleNamespace(recorder=recorder, calibrator=calibrator, controls=controls)
    for widget in widgets:
        widget.close_recording()
        widget.close()
    for host in hosts:
        host._close_analysis_windows()


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("live", [False, True])
def test_hardware_to_none_voltage_capture_and_pressure_gate(workflow, ui_qapp, rate, live):
    selected = select_hardware(workflow.controls, ui_qapp, rate)
    host = workflow.recorder(selected, live)
    enable_real_analysis(host, {"auto_analysis": True, "display_sequence": []})
    session, capture, audio = start_capture(host)
    request = session.request
    assert request.channels == (7, 1) and request.sample_rate == rate
    assert request.target_samples == int(.04 * rate) + round(.01 * rate)
    assert request.trim_samples == round(.01 * rate) and request.effective_streaming is live
    assert audio.descriptor.raw_frames == request.target_samples
    assert audio.descriptor.final_frames == sf.info(request.path).frames == int(.04 * rate)
    np.testing.assert_array_equal(audio.multi, np.tile(np.float32([8.25, 2.5]), (int(.04 * rate), 1)))
    preview = capture.snapshot(generation=session.generation, sequence=1)
    if live:
        host._on_process_recording_preview(session, preview)
    else:
        assert preview is None
    finish_ve_capture(host, session, audio)
    assert host.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert host.data_struct.wav_calibration_declared_backend == "vkinging"
    assert host.data_struct.sample_rate == sf.info(request.path).samplerate == rate
    metadata = inspect_wav_calibration_metadata(request.path).metadata
    assert metadata == request.calibration_metadata.to_dict()
    assert [entry["physical_input_channel"] for entry in metadata["recorded_channels"]] == [7, 1]
    for entry in metadata["recorded_channels"]:
        assert (entry["factor_source"], entry["calibrated"], entry["v2pa_factor"]) == ("none", False, None)
    for column, window in enumerate(host.channel_workspace.all_subwindows()):
        x, y = window.plot_item.getData()
        np.testing.assert_array_equal(y, audio.multi[:, column])
        assert x[-1] == pytest.approx((len(audio.multi) - 1) / rate)
        assert "Amplitude(V)" in window.plot_widget.getAxis("left").labelText
        assert "未校准，仅电压数据" in window.toolTip()
    assert not host.analysis_window and not host.data_struct.analysis_result_dict
    assert not host._can_output_ok_ng()[0]
    host._finalize_test_run.assert_not_called()
    host.saved.assert_called_once()
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("standard", [94, 114])
def test_calibrate_record_switch_restart_reset_and_frozen_history(
        workflow, analysis_host_factory, ui_qapp, rate, standard):
    controls = workflow.controls
    selected = select_hardware(controls, ui_qapp, rate)
    recorder = workflow.recorder(selected)
    none_session, _, none_audio = start_capture(recorder)
    finish_ve_capture(recorder, none_session, none_audio)
    widget = workflow.calibrator(selected)
    widget.standard_spl_flag = standard == 94
    events = []
    widget.calibration_finished.connect(events.append)
    assert widget.clicked_calibration()
    session = widget.streaming_processor.session
    request = session.request
    assert replace(widget._ve_capture_context.request, path=request.path) == request
    assert request.sample_rate == rate and request.target_samples == 10 * rate
    assert request.channels == (7,) and request.trim_samples == 0
    assert request.calibration_metadata is None and request.purpose == "calibration"
    _, audio = read_capture(request, **sine_capture_dependencies(standard=standard))
    raw_wav = sf.read(request.path, dtype="float32")[0]
    np.testing.assert_array_equal(raw_wav, sine_volts(rate, standard))
    session.state = "delivering"
    widget._on_calibration_result_ready(session, audio)
    accept(widget, session, audio)
    assert events == [] and controls.calibrations.get_record(selected.mic_device, 7) is None
    release(widget, session)
    assert events == [True]
    record = controls.calibrations.get_record(selected.mic_device, 7)
    assert record["v2pa_factor"] == pytest.approx(10.0, rel=.002)
    assert abs(widget._calculate_spl_from_data(raw_wav * record["v2pa_factor"]) - standard) < .02
    assert record["standard_spl"] == standard and record["calibrated_at"]
    assert record["calibration_sample_rate"] == rate and record["calibration_duration_seconds"] == 10.0
    np.testing.assert_array_equal(sf.read(request.path, dtype="float32")[0], raw_wav)
    sibling = controls.calibrations.save(selected.mic_device, 1, v2pa_factor=3.0,
        standard_spl=94, calibration_sample_rate=48000, calibration_duration_seconds=10.0,
        calibrated_at="2026-08-28T10:00:00+08:00")
    raw_calibration = controls.calibrations.path.read_bytes()
    # Every starting rate visits every rate and returns, reopening stores AND Qt.
    for next_rate in (44100, rate, 48000, rate, 51200, rate):
        selected = select_hardware(controls, ui_qapp, next_rate)
        fresh_profiles = VEInputProfileStore(controls.profiles.path)
        fresh_calibrations = VECalibrationStore(controls.calibrations.path)
        selected = select_hardware(controls, ui_qapp, next_rate,
            profiles=fresh_profiles, calibrations=fresh_calibrations)
        assert fresh_profiles.load(selected.mic_device, fresh_calibrations)["sample_rate"] == next_rate
        assert fresh_calibrations.get_record(selected.mic_device, 7) == record
        assert fresh_calibrations.get_record(selected.mic_device, 1) == sibling
        assert controls.calibrations.path.read_bytes() == raw_calibration
        restarted = calibration.InputCalibration(selected.mic_device, [7, 1],
            recording_bridge=widget.recording_bridge,
            ve_profile_store=fresh_profiles, ve_calibration_store=fresh_calibrations)
        try:
            assert restarted.saved_v2pa_factors == {7: record["v2pa_factor"], 1: 3.0}
            assert "实测校准有效" in restarted.channel_status_label.text()
        finally:
            restarted.close_recording()
            restarted.close()
    # Capture at a DIFFERENT rate; measurement provenance must not follow it.
    recording_rate = 44100 if rate != 44100 else 48000
    selected = select_hardware(controls, ui_qapp, recording_rate)
    recorder = workflow.recorder(selected, live=True)
    enable_real_analysis(recorder, {"auto_analysis": True, "display_sequence": ["spec"],
        "spec": {"type": "Spec", "analysis_channel": 7, "n_fft": 256, "hop_length": 64}})
    measured_session, _, measured_audio = start_capture(recorder)
    finish_ve_capture(recorder, measured_session, measured_audio)
    measured_meta = inspect_wav_calibration_metadata(measured_session.request.path).metadata
    entry = measured_meta["recorded_channels"][0]
    assert measured_meta["acquisition"]["sample_rate"] == recording_rate
    assert entry["calibration"] == {"sample_rate": rate, "duration_seconds": 10.0,
        "standard_spl": standard, "calibrated_at": record["calibrated_at"]}
    assert entry["factor_source"] == "measured" and entry["v2pa_factor"] == record["v2pa_factor"]
    assert not recorder.analysis_window  # SPEC executes from the frozen request off-UI.
    assert "spec" in recorder.data_struct.analysis_result_dict
    assert controls.calibrations.path.read_bytes() == raw_calibration
    # Current store replacement cannot reinterpret either historical WAV.
    controls.calibrations.save(selected.mic_device, 7, v2pa_factor=99,
        standard_spl=114, calibration_sample_rate=recording_rate, calibration_duration_seconds=10,
        calibrated_at="2026-08-29T12:00:00+08:00")
    imported = prepare_file_host(analysis_host_factory(imported=True))
    imported.mic = selected.mic_device
    imported.ve_calibration_store = controls.calibrations
    imported._load_audio_file_to_data_struct(none_session.request.path)
    assert imported.run(show_windows=False) is False
    assert imported.data_struct.wav_calibration_metadata["recorded_channels"][0]["v2pa_factor"] is None
    imported._load_audio_file_to_data_struct(measured_session.request.path)
    assert imported.run(show_windows=False) is True
    assert imported.analysis_window[0].v2pa_factor == record["v2pa_factor"] != 99
    np.testing.assert_array_equal(imported.data_struct.store_wave_data_multi, measured_audio.multi)
    profile_bytes = controls.profiles.path.read_bytes()
    assert widget.refresh_ve_calibration_state()
    widget._select_channel(7)  # Successful calibration advances to the next uncalibrated channel.
    widget.reset_btn_clicked()
    assert controls.calibrations.get_record(selected.mic_device, 7) is None
    assert controls.calibrations.get_record(selected.mic_device, 1) == sibling
    assert controls.profiles.path.read_bytes() == profile_bytes
    assert "未校准，仅电压数据" in widget.channel_status_label.text()


@pytest.mark.parametrize("fault", ["cancel", "stale", "release"])
def test_calibration_unsuccessful_paths_preserve_old_bytes(calibration_setup, fault):
    widget = calibration_setup.make()
    seed_old(calibration_setup, widget)
    before = calibration_setup.calibrations.path.read_bytes()
    events = []
    widget.calibration_finished.connect(events.append)
    session, audio = offered(widget)
    if fault == "stale":
        audio = RecordingAudio(replace(audio.descriptor, request_id="stale"), audio.multi, audio.mono)
        widget._on_calibration_result_ready(session, audio)
    else:
        accept(widget, session, audio)
        if fault == "cancel":
            widget.cancel_calibration()
        else:
            session.release_error = "injected release denied"
            widget._on_calibration_release_failed(session, session.release_error)
    release(widget, session)
    assert True not in events
    assert calibration_setup.calibrations.path.read_bytes() == before


@pytest.mark.parametrize("fault", ["stale", "release"])
def test_untrusted_or_unreleased_main_capture_cannot_publish(recording_host_factory, fault):
    host = recording_host_factory(live=True)
    session, _, audio = start_capture(host)
    if fault == "stale":
        audio = replace(audio, descriptor=replace(audio.descriptor, request_id="stale"))
    host._on_process_recording_result(session, audio)
    if fault == "stale":
        session.reject_result.assert_called_once()
        session.accept_result.assert_not_called()
    else:
        session.accept_result.assert_called_once()
        session.state = "completed"
        host._on_process_recording_accepted(session, audio)
        session.release_error = "injected release denied"
        host._on_process_recording_release_failed(session, session.release_error)
    # Late success-looking events cannot resurrect rejected/unreleased data.
    session.state = "completed"
    session.released.set()
    host._on_process_recording_accepted(session, audio)
    host._on_process_recording_released(session)
    assert host.data_struct.store_wave_data_multi is None
    host.run.assert_not_called()
    host.saved.assert_not_called()
    host._send_recording_tcp_finish.assert_not_called()


@pytest.mark.parametrize("fault", ["unavailable", "bad_rate", "monitor"])
def test_invalid_ve_blocks_admission_without_soundcard_fallback(recording_host_factory, fault):
    host = recording_host_factory()
    if fault == "unavailable":
        host.mic["available"] = False
    elif fault == "bad_rate":
        payload = json.loads(host.ve_profile_store.path.read_text(encoding="utf-8"))
        payload["devices"][host.mic["machine_id"]]["sample_rate"] = 96000
        host.ve_profile_store.path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        host.sequence_config[0]["seq1"]["acq"]["detail"]["monitor_playback"] = True
    host.judge_play_and_record()
    assert host.recording_bridge.requests == []
    assert host.mic["backend"] == "vkinging" and not host._record_workflow_busy
    warning = analysis.QMessageBox.warning.call_args.args[-1]
    assert warning
    if fault == "monitor":
        assert "关闭" in warning and "监听" in warning


def test_missing_sdk_keeps_explicit_soundcard_selection_usable(controls, ui_qapp, recording_host_factory):
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker))
    panel = controller.view.ve_controls
    panel.backend_combo.setCurrentIndex(1)  # Discovery is deliberately lazy.
    panel.discovery.service.deliver(diagnostics=("SDK missing; install VkDaq",))
    ui_qapp.processEvents()
    assert "SDK missing" in panel.diagnostic_label.text()
    panel.backend_combo.setCurrentIndex(0)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_device == controls.old_mic
    host = recording_host_factory()
    host.mic, host.mic_channels = controller.model.state.mic_device, [1]
    host.speaker = controls.speaker
    host.refresh_channel_windows()
    recorded, rate = host.reset_work_pram("not_labeled")
    host._recording_wav_calibration_metadata = None
    host._start_process_recording(recorded, rate)
    request = host._recording_process_session.request
    assert request.device.get("backend", "sounddevice") == "sounddevice" and request.sample_rate == 32000
    assert request.device["index"] == controls.old_mic["index"]


@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_new_soundcard_request_keeps_product_rate_and_replaces_only_owned_hint(
    recording_host_factory, outcome, preview_mode
):
    host = recording_host_factory(51200, live=True)
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = preview_mode
    host.mic_channels = [1]
    host.refresh_channel_windows()
    window = host.channel_workspace.all_subwindows()[0]
    window.setToolTip("operator channel note")
    session, capture, audio = start_capture(host)
    host._on_process_recording_preview(session, capture.snapshot(generation=1, sequence=1))
    finish_ve_capture(host, session, audio, outcome)
    previous_hint = window.toolTip()
    previous_plot = window.plot_item
    previous_data = None if previous_plot is None else tuple(value.copy() for value in previous_plot.getData())
    host.mic = soundcard()
    host.refresh_channel_windows()
    assert host.data_struct.sample_rate == 51200
    assert host.channel_workspace.all_subwindows()[0] is window
    assert window.toolTip() == previous_hint and "电压" in previous_hint
    if previous_data is None:
        assert window.plot_item is None  # Failure may already have cleared its preview.
    else:
        np.testing.assert_array_equal(window.plot_item.getData()[0], previous_data[0])
        np.testing.assert_array_equal(window.plot_item.getData()[1], previous_data[1])
    recorded, rate = host.reset_work_pram("not_labeled")
    host._recording_wav_calibration_metadata = None
    host._start_process_recording(recorded, rate)
    new_session = host._recording_process_session
    request = new_session.request
    assert request.sample_rate == 32000 and request.target_samples == 1600 and request.trim_samples == 320
    assert window.toolTip() == previous_hint
    # A queued old result cannot publish into the new soundcard session.
    host._on_process_recording_accepted(session, audio)
    assert host._recording_process_session is new_session
    waveform = StreamingWaveformSnapshot(
        (
            (np.arange(1280, dtype=np.float64) - 1279) / rate
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else np.arange(1280, dtype=np.float64) / rate
        ),
        np.full(1280, .05, dtype=np.float32),
        1280,
    )
    host._on_process_recording_preview(new_session, RecordingPreview(
        request.request_id, new_session.generation, 1, 1280, (1,), (waveform,),
        preview_mode))
    assert window.toolTip() == "operator channel note"
    if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST:
        assert window.plot_item.getData()[0][-1] == 0.0
    else:
        assert window.plot_item.getData()[0][0] == 0.0
        assert window.plot_item.getData()[0][-1] > 0.0
    assert window.is_live_preview is True


@pytest.mark.parametrize("missing_channel", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_captured_partial_or_absent_import_cannot_reuse_old_condition_ok(
        workflow, analysis_host_factory, ui_qapp, monkeypatch, missing_channel, partial):
    selected = select_hardware(workflow.controls, ui_qapp, 44100)
    recorder = workflow.recorder(selected)
    save_calibration(recorder)
    if missing_channel:
        save_calibration(recorder, physical=1, factor=2)
    session, _, audio = start_capture(recorder)
    finish_ve_capture(recorder, session, audio)
    imported = prepare_condition_import_host(analysis_host_factory(imported=True, mode="test"),
        partial=partial, missing_channel=missing_channel, previous=True)
    imported.lineedit_type = SimpleNamespace(text=lambda: "workflow")
    imported.lineedit_s_or_n = SimpleNamespace(text=lambda: "serial")
    imported._resolve_recording_name_suffix = lambda: ""
    imported._create_analysis_storage_context = lambda: None
    monkeypatch.setattr(analysis.QFileDialog, "getOpenFileName", lambda *args: (session.request.path, ""))
    assert imported.import_audio_and_analyze() is False
    np.testing.assert_array_equal(imported.data_struct.store_wave_data_multi, audio.multi)
    assert imported.data_struct.wav_calibration_read_status is ReadStatus.VALID
    assert imported.data_struct.wav_calibration_declared_backend == "vkinging"
    assert imported._manual_product_condition_completed_keys == {"00"}
    assert imported._manual_product_condition_results == {"00": "OK"}
    imported._advance_manual_product_condition_cycle_after_recording.assert_not_called()
    imported._refresh_current_manual_product_final_from_group.assert_not_called()
    assert not imported._can_output_ok_ng()[0]
    assert bool(imported.analysis_window) is partial
    if partial:
        sibling = imported.analysis_window[0]
        assert sibling.v2pa_factor == 10
        assert sibling.result["overall_spl"] == pytest.approx(20 * np.log10(8.25 * 10 / 20e-6), abs=.001)
    record = imported._condition_record_cache["01"]
    assert record["analysis_report_state"] == "failed" and record["result_label"] == "not_labeled"
    assert "old-PASS" not in record["analysis_result_dict"]
    assert any(item["state"] == "skipped" for item in record["analysis_report_items"])
    assert bool(imported._ve_channel_skips) is missing_channel
    assert bool(imported._ve_calibration_skips) is not missing_channel


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("mode", ["empty", "absent", "exception"])
def test_raw_capture_completion_distinguishes_noop_preflight_and_analysis_exception(
        recording_host_factory, live, mode, monkeypatch, tmp_path):
    host = recording_host_factory(44100, live)
    save_calibration(host)
    config = {"auto_analysis": True, "display_sequence": [] if mode == "empty" else ["spl"],
        "spl": {"type": "SPL", "analysis_channel": 0 if mode == "absent" else 7,
                "limit_checked": True, "limit_metric": "overall_spl",
                "scalar_upper_value": 200}}
    enable_real_analysis(host, config)
    host._is_manual_product_condition_cycle_active = lambda: True
    host._get_active_product_condition_key = lambda: "item-condition"
    host._manual_product_condition_keys = lambda: ["item-condition"]
    host.product_test_project_context = {
        "project_name": "workflow",
        "result_root_directory": str(tmp_path),
    }
    # This test owns completion semantics after admission. Channel preflight is
    # covered separately and must not stop the fake capture before a session exists.
    host._validate_automatic_analysis_channels_before_recording = lambda: True
    host._mark_manual_product_condition_recording_completed = mock.Mock()
    host._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    if mode == "exception":
        def broken_consumer(*args, **kwargs):
            raise RuntimeError("injected external analysis constructor failure")
        host._recording_request_analysis_executor = broken_consumer
    session, _, audio = start_capture(host)
    if mode == "exception":
        host._on_process_recording_result(session, audio)
        session.state = "completed"
        host._on_process_recording_accepted(session, audio)
        session.released.set()
        host._on_process_recording_released(session)
        assert host.data_struct.store_wave_data_multi is None
        assert not host.analysis_window
        assert host.data_struct.analysis_result_dict == {"old-PASS": (True, 0.0)}
        host.saved.assert_called_once()
        host._send_recording_tcp_finish.assert_not_called()
        return
    finish_ve_capture(host, session, audio)
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
    assert "old-PASS" not in host.data_struct.analysis_result_dict
    host._finalize_test_run.assert_not_called()
    host.saved.assert_called_once()
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))
    if mode == "empty":
        assert not host.analysis_window and not host._ve_calibration_skips and not host._ve_channel_skips
        host._mark_manual_product_condition_recording_completed.assert_called_once()
        host._advance_manual_product_condition_cycle_after_recording.assert_called_once()
        host._capture_current_analysis_report_snapshot("empty")
        report = host._update_recent_session.call_args.kwargs
        assert report["analysis_report_state"] == "not_required" and report["analysis_report_items"] == []
    else:
        host._mark_manual_product_condition_recording_completed.assert_not_called()
        host._advance_manual_product_condition_cycle_after_recording.assert_not_called()
        host._finalize_serial_product_condition_analysis_failure.assert_called_once()


@pytest.mark.parametrize("kind", ["Spec", "PD", "ED"])
def test_captured_pressure_consumers_keep_local_mono_and_physical_report_identity(workflow, ui_qapp, kind):
    selected = select_hardware(workflow.controls, ui_qapp, 51200)
    host = workflow.recorder(selected)
    save_calibration(host, physical=7, factor=10)
    save_calibration(host, physical=1, factor=3)
    pd_config = {"spl_window_unit": "points", "spl_window_points": 31}
    config = {"auto_analysis": True, "display_sequence": ["item"], "item": {
        "type": kind, "analysis_channel": 1, "n_fft": 256, "hop_length": 64,
        **pd_config, "head": {"type": "PD", "config": pd_config}, "tail": {"type": "PM", "config": {}}}}
    original = deepcopy(config)
    enable_real_analysis(host, config)
    session, _, audio = start_capture(host)
    mutable_after_entry = original
    if kind == "ED":
        mutable_after_entry = {"auto_analysis": True, "display_sequence": ["item"],
            "item": {"type": "PD", "analysis_channel": 0, **pd_config}}
        host.analysis_config = deepcopy(mutable_after_entry)
    finish_ve_capture(host, session, audio)
    if kind == "Spec":
        assert not host.analysis_window  # SPEC executes from the frozen request off-UI.
        assert host.data_struct.analysis_result_dict["item"][0] is None
        assert host.analysis_config == mutable_after_entry
        np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
        assert host.data_struct.wav_calibration_metadata == session.request.calibration_metadata
        return
    instance = host.analysis_window[0]
    expected = (audio.multi[:, 1].astype("float64") * 3 if kind == "Spec"
        else (audio.multi.astype("float64") * (10, 3)).mean(axis=1))
    view = instance.data_struct
    assert view.store_wave_data_multi is None and view.store_wave_data.dtype == np.float64
    assert view.store_wave_data.ndim == 1 and view.store_wave_data.nbytes == len(audio.multi) * 8
    np.testing.assert_array_equal(view.store_wave_data, expected)
    assert instance.v2pa_factor == 1.0 and instance.analysis_config["analysis_channel"] == 0
    assert instance._analysis_raw_channel == 1
    assert instance._sequence_analysis_key == instance._sequence_window_key == "item"
    assert instance._sequence_runtime_key == "item--通道2"
    assert host._analysis_channel_local_columns["item"] == 1
    assert view.analysis_result_dict is host.data_struct.analysis_result_dict
    assert view.pd_peak_grid_points_map is host.data_struct.pd_peak_grid_points_map
    assert host.analysis_config == mutable_after_entry
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
    assert host.data_struct.wav_calibration_metadata == session.request.calibration_metadata
    if kind == "Spec":
        spectrum = signals.librosa.amplitude_to_db(np.abs(signals.librosa.stft(
            y=expected.astype("float32"), n_fft=256, hop_length=64, window="hann")), ref=20e-6)
        np.testing.assert_allclose(instance.img_item.image, spectrum.T, atol=1e-4)
    else:
        result = signals.peak_detection(expected, 51200, pd_config, v2pa_factor=1.0)
        actual = instance.result["spl_db_series"] if kind == "PD" else instance._last_spl_series
        np.testing.assert_allclose(actual, result["spl_db_series"])
    report = build_analysis_report_items(
        host.analysis_window, original, host.data_struct.analysis_result_dict)
    assert [(item["name"], item["state"]) for item in report] == [("item", "completed")]


@pytest.mark.parametrize(
    "frozen_golden,expected_golden",
    [("frozen-golden.json", "frozen-golden.json"), (None, None)],
)
def test_serialized_real_completion_freezes_analysis_and_excel_configuration(
    workflow, ui_qapp, monkeypatch, frozen_golden, expected_golden,
):
    selected = select_hardware(workflow.controls, ui_qapp, 51200)
    host = workflow.recorder(selected)
    save_calibration(host, physical=7, factor=10)
    save_calibration(host, physical=1, factor=3)
    item = {
        "type": "ED", "analysis_channel": 1,
        "spl_window_unit": "points", "spl_window_points": 31,
        "frozen_item_schema": "item-v1",
    }
    frozen = {
        "auto_analysis": True,
        "display_sequence": ["item", "excel"],
        "golden_sample_result_path": frozen_golden,
        "item": item,
        "excel": {
            "type": "Excel", "fast_mode": False,
            "file_path": "frozen-results.xlsx",
            "selected_analysis_items": ["item"],
            "frozen_excel_schema": "columns-v1",
        },
    }
    enable_real_analysis(host, deepcopy(frozen))
    host._capture_excel_export_cache = (
        analysis.SequenceWidgetAnalysisOpsMixin._capture_excel_export_cache
        .__get__(host))
    host._maybe_export_excel_results = (
        analysis.SequenceWidgetAnalysisOpsMixin._maybe_export_excel_results
        .__get__(host))
    exports = []

    def export(excel_config, **kwargs):
        exports.append((deepcopy(excel_config), deepcopy(kwargs)))
        return SimpleNamespace(ok=True, message="saved")

    monkeypatch.setattr(analysis, "export_analysis_to_excel", export)
    session, _, audio = start_capture(host)
    host.analysis_config = {
        "auto_analysis": True,
        "display_sequence": ["item", "excel"],
        "golden_sample_result_path": "mutated-golden.json",
        "item": {"type": "PD", "analysis_channel": 0,
                 "mutable_item_schema": "item-v2"},
        "excel": {"type": "Excel", "fast_mode": False,
                  "file_path": "mutated-results.xlsx",
                  "selected_analysis_items": [],
                  "mutable_excel_schema": "columns-v2"},
    }

    finish_ve_capture(host, session, audio)

    assert len(host.analysis_window) == 1
    runtime_config = host.analysis_window[0].analysis_config
    assert runtime_config["type"] == "ED"
    assert runtime_config["frozen_item_schema"] == "item-v1"
    if expected_golden is None:
        assert "golden_sample_result_path" not in runtime_config
    else:
        assert runtime_config["golden_sample_result_path"] == expected_golden
    assert len(exports) == 1
    excel_config, export_kwargs = exports[0]
    assert excel_config["file_path"] == "frozen-results.xlsx"
    assert excel_config["frozen_excel_schema"] == "columns-v1"
    assert export_kwargs["analysis_config"] == frozen
    assert host._excel_export_cache["analysis_items_data"]["item--通道2"]["type"] == "ED"
